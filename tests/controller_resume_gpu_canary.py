# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-process GPU checkpoint restart test of the public controller adapter.

Run in two fresh process groups, using the same empty checkpoint directory::

    torchrun --standalone --nproc-per-node=2 tests/controller_resume_gpu_canary.py save CHECKPOINT_DIR
    torchrun --standalone --nproc-per-node=2 tests/controller_resume_gpu_canary.py resume CHECKPOINT_DIR

Requires two CUDA devices. Exercises application-owned shard cursors and SGD
state through the public controller data fetcher; not large-model checkpointing.
"""

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from cosmos_rl.dispatcher.data.data_fetcher import ControllerDataFetcher
from cosmos_rl.dispatcher.data.resume import ControllerResumeMetadata


class Prompts(Dataset):
    def __len__(self):
        return 20

    def __getitem__(self, index):
        return str(index)

    def setup(self, config):
        pass

    def get_reference_answer(self, index):
        return str(index)


class ShardSampler:
    def __init__(self, rank, size):
        self.order = list(range(rank, 20, size))
        self.position = 0
        self.iterations = 0

    def set_epoch(self, epoch):
        self.position = 0

    def __iter__(self):
        self.iterations += 1
        while self.position < len(self.order):
            index = self.order[self.position]
            self.position += 1
            yield index


class Adapter:
    def load_metadata(self, config):
        saved = torch.load(config.train.resume, map_location="cpu", weights_only=True)
        return ControllerResumeMetadata.model_validate(saved["metadata"])

    def restore_sampler(self, sampler, metadata):
        assert sampler.iterations == 0, "sampler was consumed before restoration"
        sampler.position = metadata.sampler_state["position"]


def config(resume):
    return SimpleNamespace(
        train=SimpleNamespace(
            resume=resume,
            epoch=2,
            local_dataset=False,
            train_batch_per_replica=2,
            train_policy=SimpleNamespace(
                type="grpo",
                dataloader_batch_size=2,
                dataloader_shuffle=False,
                dataloader_seed=42,
                dataloader_num_workers=0,
                dataloader_prefetch_factor=None,
                data_dispatch_as_rank_in_mesh=False,
            ),
        ),
        rollout=SimpleNamespace(
            batch_size=2,
            n_generation=1,
            multi_turn_config=SimpleNamespace(enable=False),
        ),
        validation=SimpleNamespace(enable=False, dataset=SimpleNamespace(name="")),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["save", "resume"])
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    try:
        args.root.mkdir(parents=True, exist_ok=True)
        checkpoint = args.root / f"rank-{rank}.pt"
        expected_path = args.root / f"expected-{rank}.pt"
        sampler = ShardSampler(dist.get_rank(), dist.get_world_size())
        fetcher = ControllerDataFetcher(
            config(str(checkpoint) if args.phase == "resume" else False),
            dataset=Prompts(),
            sampler=sampler,
            resume_adapter=Adapter(),
        )
        weight = torch.nn.Parameter(torch.tensor([0.25], device=f"cuda:{rank}"))
        optimizer = torch.optim.SGD([weight], lr=0.01, momentum=0.9)

        def update():
            indices, _ = next(fetcher.train_dataloader_iter)
            inputs = torch.tensor([int(i) + 1 for i in indices], device=weight.device)
            optimizer.zero_grad()
            ((weight * inputs - 1) ** 2).mean().backward()
            dist.all_reduce(weight.grad)
            weight.grad.div_(dist.get_world_size())
            optimizer.step()
            return (
                indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)
            )

        if args.phase == "save":
            for _ in range(2):
                update()
            metadata = ControllerResumeMetadata(
                checkpoint_path=str(checkpoint),
                checkpoint_id="gpu-canary-step-2",
                completed_training_steps=2,
                completed_optimizer_updates=2,
                remaining_completions=12,
                epoch=1,
                sampling_owner="sampler",
                sampler_state={"position": sampler.position},
            )
            temporary = checkpoint.with_suffix(".tmp")
            torch.save(
                {
                    "metadata": metadata.model_dump(),
                    "weight": weight.detach(),
                    "optimizer": optimizer.state_dict(),
                },
                temporary,
            )
            temporary.replace(checkpoint)
            records = []
            for step in (3, 4):
                indices = update()
                records.append(
                    {
                        "step": step,
                        "indices": indices,
                        "weight": weight.detach().cpu().clone(),
                        "momentum": optimizer.state[weight]["momentum_buffer"]
                        .cpu()
                        .clone(),
                    }
                )
            torch.save(records, expected_path)
            print(
                f"[RESUME-CANARY] rank={rank} saved_step=2 baseline_through=4",
                flush=True,
            )
        else:
            saved = torch.load(
                checkpoint, map_location=weight.device, weights_only=True
            )
            with torch.no_grad():
                weight.copy_(saved["weight"])
            optimizer.load_state_dict(saved["optimizer"])
            fetcher.validate_after_resume(
                ControllerResumeMetadata.model_validate(
                    saved["metadata"]
                ).to_checkpoint_extra_info()
            )
            assert fetcher.ckpt_extra_info["step"] == 2
            records = torch.load(expected_path, weights_only=True)
            for expected in records:
                assert update() == expected["indices"]
                torch.testing.assert_close(
                    weight.cpu(), expected["weight"], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    optimizer.state[weight]["momentum_buffer"].cpu(),
                    expected["momentum"],
                    rtol=0,
                    atol=0,
                )
                print(
                    f"[RESUME-CANARY] rank={rank} resumed_update={expected['step']} exact_parity=True",
                    flush=True,
                )
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
