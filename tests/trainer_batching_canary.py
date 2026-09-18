# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run with torchrun --nproc-per-node=2 on CUDA for expanded-batch validation."""

import json
import os
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist

from cosmos_rl.policy.trainer.batching import (
    ExpandedSampleBatching,
    ExpandedTrainingBatch,
    run_training_step,
)


class CanaryTrainer:
    batching_contract = ExpandedSampleBatching(partial_tail="include")

    def __init__(self, device):
        self.config = SimpleNamespace(
            train=SimpleNamespace(train_policy=SimpleNamespace(mini_batch=2))
        )
        self.weight = torch.nn.Parameter(
            torch.tensor([0.5], dtype=torch.float64, device=device)
        )
        self.optimizer = torch.optim.SGD([self.weight], lr=0.1, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.9
        )
        self.updates = 0

    def prepare_training_batch(self, episodes):
        samples = [sample for episode in episodes for sample in episode]
        return ExpandedTrainingBatch(
            tuple(samples[offset : offset + 2] for offset in range(0, len(samples), 2))
        )

    def step_expanded_training(self, batch, **kwargs):
        for samples in batch.minibatches:
            self.optimizer.zero_grad()
            loss = ((self.weight * torch.stack(samples) - 1) ** 2).mean()
            loss.backward()
            dist.all_reduce(self.weight.grad)
            self.weight.grad.div_(dist.get_world_size())
            self.optimizer.step()
            self.scheduler.step()
            self.updates += 1
        return {"updates": self.updates}


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("nccl", timeout=timedelta(seconds=60))
    try:
        rank = dist.get_rank()
        world = dist.get_world_size()
        assert world == 2
        trainer = CanaryTrainer(device)
        samples = torch.arange(1 + rank, 4 + rank, dtype=torch.float64, device=device)
        episodes = (
            [samples[:1], samples[1:]] if rank == 0 else [samples[:2], samples[2:]]
        )
        assert run_training_step(trainer, rollouts=episodes) == {"updates": 2}

        reference = torch.nn.Parameter(
            torch.tensor([0.5], dtype=torch.float64, device=device)
        )
        optimizer = torch.optim.SGD([reference], lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        for offset in (0, 2):
            explicit = torch.cat(
                [
                    torch.arange(1 + r, 4 + r, dtype=torch.float64, device=device)[
                        offset : offset + 2
                    ]
                    for r in range(world)
                ]
            )
            optimizer.zero_grad()
            ((reference * explicit - 1) ** 2).mean().backward()
            optimizer.step()
            scheduler.step()
        torch.testing.assert_close(trainer.weight, reference, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            trainer.optimizer.state[trainer.weight]["momentum_buffer"],
            optimizer.state[reference]["momentum_buffer"],
            rtol=1e-12,
            atol=1e-12,
        )
        assert trainer.scheduler.state_dict() == scheduler.state_dict()
        print(
            json.dumps(
                {
                    "rank": rank,
                    "numerical_parity": True,
                    "optimizer_updates": trainer.updates,
                }
            ),
            flush=True,
        )

        for case in ("one_empty", "all_empty", "nonfinite", "unequal_steps"):
            candidate = episodes
            if case == "all_empty" or (case == "one_empty" and rank == 1):
                candidate = []
            elif case == "nonfinite" and rank == 1:
                candidate = [
                    torch.tensor(
                        [float("nan"), 2, 3], dtype=torch.float64, device=device
                    )
                ]
            elif case == "unequal_steps" and rank == 1:
                candidate = [samples[:2]]
            try:
                run_training_step(trainer, rollouts=candidate)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{case} unexpectedly trained")
            assert trainer.updates == 2
            assert trainer.scheduler.state_dict() == scheduler.state_dict()
            torch.testing.assert_close(
                trainer.weight, reference, rtol=1e-12, atol=1e-12
            )
            dist.barrier()
            print(
                json.dumps(
                    {"rank": rank, "case": case, "rejected_before_training": True}
                ),
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
