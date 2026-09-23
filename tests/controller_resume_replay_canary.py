# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Application-owned checkpoint recovery with outstanding/out-of-order work.

Run save and resume in fresh groups against one new directory, for example::

    torchrun --standalone --nproc-per-node=2 tests/controller_resume_replay_canary.py save DIR --device cuda
    torchrun --standalone --nproc-per-node=2 tests/controller_resume_replay_canary.py resume DIR --device cuda

Use --device cpu for Gloo validation. Trainer serialization and shard selection
are application-owned. This exercises the public adapter, replay ledger and
manifest, not the full launcher or a large-model checkpoint format.
"""

import argparse
import os
from pathlib import Path
import random
from uuid import uuid4

import torch
import torch.distributed as dist

from controller_resume_gpu_canary import Prompts, ShardSampler, config
from cosmos_rl.dispatcher.data.checkpoint_manifest import CheckpointManifest
from cosmos_rl.dispatcher.data.data_fetcher import ControllerDataFetcher
from cosmos_rl.dispatcher.data.replay import SamplingBoundary, SamplingReplayLedger
from cosmos_rl.dispatcher.data.resume import ControllerResumeMetadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["save", "resume"])
    parser.add_argument("root", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if args.device == "cuda" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        rank, size = dist.get_rank(), dist.get_world_size()
        root = args.root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        required = {f"trainer-{r}.pt" for r in range(size)}
        compatibility = {
            "dataset": "prompts-0-through-19-v1",
            "world_size": size,
            "n_generation": 1,
        }
        sampler = ShardSampler(rank, size)

        class Adapter:
            def load_metadata(self, cfg):
                return CheckpointManifest.load(
                    Path(cfg.train.resume),
                    required_artifacts=required,
                    compatibility=compatibility,
                ).metadata

            def restore_sampler(self, stream, metadata):
                if stream.iterations:
                    raise RuntimeError("sampler advanced before restoration")
                stream.position = metadata.sampler_state["ranks"][str(rank)]["position"]

        cfg = config(str(root) if args.phase == "resume" else False)
        cfg.train.epoch = 1
        fetcher = ControllerDataFetcher(
            cfg,
            dataset=Prompts(),
            sampler=sampler,
            resume_adapter=Adapter(),
        )
        parameter = torch.nn.Parameter(torch.tensor([0.25], device=device))
        optimizer = torch.optim.SGD([parameter], lr=0.01, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        def update(indices):
            inputs = torch.tensor([int(i) + 1 for i in indices], device=device)
            # Exercise trainer RNG without imposing post-restart replay parity.
            noise = torch.rand_like(inputs, dtype=torch.float32) + random.random()
            optimizer.zero_grad()
            ((parameter * inputs + noise - 1) ** 2).mean().backward()
            dist.all_reduce(parameter.grad)
            parameter.grad.div_(size)
            optimizer.step()
            scheduler.step()

        def boundary():
            return SamplingBoundary(
                epoch=1,
                sampler_state={"position": sampler.position},
                remaining_completions=len(sampler.order) - sampler.position,
            )

        if args.phase == "save":
            ledger = SamplingReplayLedger(boundary())
            batches, tokens = [], []
            for _ in range(3):
                indices, _ = next(fetcher.train_dataloader_iter)
                batches.append(list(indices))
                tokens.append(ledger.issue(boundary(), completions=len(indices)))
            # Batch 1 is still outstanding, while batch 2 has already trained.
            # Saving the fetched cursor (6) would silently skip batch 1.
            for index in (0, 2):
                update(batches[index])
                for completion in range(len(batches[index])):
                    ledger.settle(tokens[index], completion)
            safe = ledger.snapshot()
            if safe.sampler_state["position"] != 2 or sampler.position != 6:
                raise RuntimeError("expected conservative replay behind fetched cursor")
            checkpoint_ids = [uuid4().hex if rank == 0 else None]
            dist.broadcast_object_list(checkpoint_ids)
            checkpoint_id = checkpoint_ids[0]
            torch.save(
                {
                    "checkpoint_id": checkpoint_id,
                    "completed_training_steps": 2,
                    "completed_optimizer_updates": 2,
                    "parameter": parameter.detach().cpu(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device).cpu()
                    if device.type == "cuda"
                    else None,
                    "python_rng": random.getstate(),
                    "safe": safe.model_dump(),
                    "outstanding": batches[1],
                    "repeat_allowed": batches[2],
                    "old_token": tokens[1],
                },
                root / f"trainer-{rank}.pt",
            )
            states = [None] * size
            dist.all_gather_object(states, safe.model_dump())
            dist.barrier()  # every required shard is closed before publication
            if rank == 0:
                metadata = ControllerResumeMetadata(
                    checkpoint_path=str(root),
                    checkpoint_id=checkpoint_id,
                    completed_training_steps=2,
                    completed_optimizer_updates=2,
                    remaining_completions=sum(
                        state["remaining_completions"] for state in states
                    ),
                    epoch=1,
                    sampling_owner="sampler",
                    sampler_state={
                        "ranks": {
                            str(r): state["sampler_state"]
                            for r, state in enumerate(states)
                        }
                    },
                )
                CheckpointManifest.publish(
                    root,
                    metadata,
                    required_artifacts=required,
                    compatibility=compatibility,
                )
            dist.barrier()
            print(
                f"REPLAY_CANARY rank={rank} committed_step=2 safe_cursor=2 fetched_cursor=6 outstanding=True",
                flush=True,
            )
        else:
            manifest = CheckpointManifest.load(
                root, required_artifacts=required, compatibility=compatibility
            )
            saved = torch.load(
                root / f"trainer-{rank}.pt", map_location="cpu", weights_only=True
            )
            if saved["checkpoint_id"] != manifest.metadata.checkpoint_id:
                raise RuntimeError("trainer shard belongs to a different checkpoint")
            for name in ("completed_training_steps", "completed_optimizer_updates"):
                if saved[name] != getattr(manifest.metadata, name):
                    raise RuntimeError("trainer/controller checkpoint counters differ")
            with torch.no_grad():
                parameter.copy_(saved["parameter"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            torch.set_rng_state(saved["torch_rng"])
            random.setstate(saved["python_rng"])
            if device.type == "cuda":
                torch.cuda.set_rng_state(saved["cuda_rng"], device)
            torch.testing.assert_close(
                parameter.cpu(), saved["parameter"], rtol=0, atol=0
            )
            torch.testing.assert_close(
                optimizer.state[parameter]["momentum_buffer"].cpu(),
                saved["optimizer"]["state"][0]["momentum_buffer"].cpu(),
                rtol=0,
                atol=0,
            )
            assert scheduler.state_dict() == saved["scheduler"]
            assert torch.equal(torch.get_rng_state(), saved["torch_rng"])
            assert random.getstate() == saved["python_rng"]
            if device.type == "cuda":
                assert torch.equal(torch.cuda.get_rng_state(device), saved["cuda_rng"])
            fetcher.validate_after_resume(manifest.metadata.to_checkpoint_extra_info())
            ledger = SamplingReplayLedger(
                SamplingBoundary.model_validate(saved["safe"])
            )
            assert not ledger.settle(saved["old_token"])
            initial = parameter.detach().clone()
            for expected, step in (
                (saved["outstanding"], 3),
                (saved["repeat_allowed"], 4),
            ):
                indices, _ = next(fetcher.train_dataloader_iter)
                assert list(indices) == expected
                update(indices)
                print(
                    f"REPLAY_CANARY rank={rank} resumed_step={step} indices={list(indices)} recovery_ok=True",
                    flush=True,
                )
            assert not torch.equal(parameter, initial)
            assert scheduler.last_epoch == 4
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
