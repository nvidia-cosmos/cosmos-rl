# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""torchrun --nproc-per-node=2 tests/trainer_batching_canary.py [--cpu]."""

import os
import sys
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist

from cosmos_rl.policy.trainer.batching import (
    ExpandedSampleBatching,
    ExpandedTrainingBatch,
    RecoverablePreparationError,
    run_training_step,
    agree_batching_schedule,
)


class CanaryTrainer:
    batching_contract = ExpandedSampleBatching(partial_tail="include")

    def __init__(self, device):
        self.config = SimpleNamespace(
            train=SimpleNamespace(
                train_policy=SimpleNamespace(mini_batch=2, mu_iterations=2)
            )
        )
        self.weight = torch.nn.Parameter(
            torch.tensor([0.5], dtype=torch.float64, device=device)
        )
        self.optimizer = torch.optim.SGD([self.weight], lr=0.1, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.9
        )
        self.updates = 0
        self.saved = False

    def prepare_training_batch(self, episodes):
        if episodes is None:
            raise RecoverablePreparationError("unavailable episode")
        samples = [sample for episode in episodes for sample in episode]
        return ExpandedTrainingBatch(
            tuple(samples[i : i + 2] for i in range(0, len(samples), 2))
        )

    def step_expanded_training(self, batch, **kwargs):
        self.plan = batch
        for _ in range(batch.mu_iterations):
            for index, samples in enumerate(batch.minibatches):
                self.optimizer.zero_grad()
                # Real models need matching forward/backward collectives too,
                # e.g. a masked dummy sample. This toy has one manual reduction.
                loss = (
                    ((self.weight * torch.stack(samples) - 1) ** 2).sum()
                    if samples
                    else self.weight.sum() * 0
                )
                scale = (
                    batch.mean_gradient_scale(index, dist.get_world_size())
                    if batch.global_sample_counts is not None
                    else 1 / self.config.train.train_policy.mini_batch
                )
                (loss * scale).backward()
                dist.all_reduce(self.weight.grad)
                self.weight.grad.div_(dist.get_world_size())
                self.optimizer.step()
                self.scheduler.step()
                self.updates += 1
        self.saved = kwargs.get("do_save_checkpoint", False)
        return {"updates": self.updates}


def values(case, rank):
    if case == "all_empty" or (
        case in ("one_empty", "preparation_error") and rank == 1
    ):
        return []
    if case == "nonfinite" and rank == 1:
        return [float("nan"), 3.0, 4.0]
    if case == "all_nonfinite":
        return [float("inf")]
    if case == "unequal_steps" and rank == 1:
        return [2.0]
    if case == "empty_slot":
        return [float("nan"), float("nan"), 3.0 + rank]
    return [1.0 + rank, 2.0 + rank, 3.0 + rank]


def main():
    cpu = "--cpu" in sys.argv
    if not cpu:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cpu" if cpu else "cuda")
    dist.init_process_group("gloo" if cpu else "nccl", timeout=timedelta(seconds=60))
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        for case in (
            "healthy",
            "one_empty",
            "all_empty",
            "nonfinite",
            "all_nonfinite",
            "unequal_steps",
            "preparation_error",
            "empty_slot",
        ):
            trainer = CanaryTrainer(device)
            data = torch.tensor(values(case, rank), dtype=torch.float64, device=device)
            episodes = None if case == "preparation_error" and rank == 1 else [data]
            setup_calls = []
            run_training_step(
                trainer,
                before_step=lambda: setup_calls.append(True),
                rollouts=episodes,
                do_save_checkpoint=True,
            )
            reference = torch.nn.Parameter(
                torch.tensor([0.5], dtype=torch.float64, device=device)
            )
            optimizer = torch.optim.SGD([reference], lr=0.1, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=1, gamma=0.9
            )
            per_rank = [values(case, r) for r in range(2)]
            updates = 0
            for _ in range(2):
                for offset in range(0, max(map(len, per_rank)), 2):
                    explicit = torch.tensor(
                        [
                            x
                            for samples in per_rank
                            for x in samples[offset : offset + 2]
                        ],
                        dtype=torch.float64,
                        device=device,
                    )
                    explicit = explicit[torch.isfinite(explicit)]
                    if not len(explicit):
                        continue
                    optimizer.zero_grad()
                    ((reference * explicit - 1) ** 2).mean().backward()
                    optimizer.step()
                    scheduler.step()
                    updates += 1
            assert trainer.updates == updates
            assert len(setup_calls) == bool(updates)
            assert trainer.saved
            torch.testing.assert_close(
                trainer.weight, reference, rtol=1e-12, atol=1e-12
            )
            assert trainer.scheduler.state_dict() == scheduler.state_dict()
            if updates:
                torch.testing.assert_close(
                    trainer.optimizer.state[trainer.weight]["momentum_buffer"],
                    optimizer.state[reference]["momentum_buffer"],
                    rtol=1e-12,
                    atol=1e-12,
                )
            else:
                assert not trainer.optimizer.state
            dist.barrier()
            print(
                f"rank={rank} case={case} updates={updates} numerical_parity=PASS",
                flush=True,
            )
        fixed_schedule_canary(device, rank)
    finally:
        dist.destroy_process_group()


def fixed_schedule_canary(device, rank):
    from unittest.mock import patch

    trainer = CanaryTrainer(device)
    trainer.batching_contract = ExpandedSampleBatching(
        partial_tail="include", fixed_minibatches=2
    )
    reference = torch.nn.Parameter(
        torch.tensor([0.5], dtype=torch.float64, device=device)
    )
    optimizer = torch.optim.SGD([reference], lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    with patch.object(
        dist, "all_gather_object", wraps=dist.all_gather_object
    ) as gather:
        agree_batching_schedule(trainer)
        for case in (
            "healthy",
            "one_empty",
            "all_empty",
            "nonfinite",
            "preparation_error",
            "unequal_steps",
        ):
            data = torch.tensor(values(case, rank), dtype=torch.float64, device=device)
            episodes = None if case == "preparation_error" and rank == 1 else [data]
            run_training_step(trainer, rollouts=episodes)
            for _ in range(2):
                for offset in (0, 2):
                    samples = torch.tensor(
                        [
                            x
                            for r in range(2)
                            for x in values(case, r)[offset : offset + 2]
                        ],
                        dtype=torch.float64,
                        device=device,
                    )
                    samples = samples[torch.isfinite(samples)]
                    optimizer.zero_grad()
                    # Fixed nominal denominator: empty samples have zero weight.
                    # This is deliberately NOT a mean over valid samples.
                    loss = ((reference * samples - 1) ** 2).sum() / 4
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
            torch.testing.assert_close(
                trainer.weight, reference, rtol=1e-12, atol=1e-12
            )
            torch.testing.assert_close(
                trainer.optimizer.state[trainer.weight]["momentum_buffer"],
                optimizer.state[reference]["momentum_buffer"],
                rtol=1e-12,
                atol=1e-12,
            )
            assert trainer.scheduler.state_dict() == scheduler.state_dict()
            assert gather.call_count == 1
            print(
                f"rank={rank} fixed_case={case} single_agreement=PASS numerical_parity=PASS",
                flush=True,
            )


if __name__ == "__main__":
    main()
