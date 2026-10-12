# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Two saving ranks: paused/failed writer, retention and strict stage restore.

Uses actual checkpoint artifacts, CUDA staging and native collectives. The tiny
stage modules exercise PP checkpoint layout, not a pipeline schedule or model.
"""

import argparse
import copy
import os
import threading
import time
from pathlib import Path
from unittest.mock import patch

import cosmos_rl
import torch
import torch.distributed as dist

from cosmos_rl.policy.config import Config
from cosmos_rl.utils.checkpoint import CheckpointMananger
from cosmos_rl.utils.parallelism import ParallelDims


def immutable_shard_promotion(root, rank, device):
    """Promote one rank while its peer waits: comparisons must stay local."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    mesh = init_device_mesh(device.type, (2,))
    dims = ParallelDims(
        dp_replicate=1,
        dp_shard=2,
        cp=1,
        tp=1,
        pp=1,
        world_size=2,
        pp_dynamic_shape=False,
    )
    for mode in ("sync", "async"):
        cfg = Config.from_dict(
            {
                "train": {
                    "output_dir": str(root / f"immutable-{mode}"),
                    "timestamp": "run",
                    "ckpt": {
                        "enable_checkpoint": True,
                        "save_mode": mode,
                        "export_safetensors": False,
                    },
                }
            }
        )
        manager = CheckpointMananger(cfg, dims, global_rank=rank)
        model = torch.nn.Linear(2, 1).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        shard = distribute_tensor(torch.arange(8, device=device), mesh, [Shard(0)])
        state = {"shard": shard}
        try:
            path = Path(
                manager.save_checkpoint(
                    state, optimizer, scheduler, 1, 10, cursor=3, is_final=False
                )
            )
            manager._wait_for_pending_async_saves()
            dist.barrier()
            before = {p.name: p.read_bytes() for p in path.glob(f"*rank_{rank}*")}
            for owner in range(2):
                if rank == owner:
                    manager.invalidate_completion_marker(1)
                    manager.save_checkpoint(
                        state, optimizer, scheduler, 1, 10, cursor=3, is_final=True
                    )
                # The other rank reaches this while the owner verifies DTensors.
                dist.barrier()
            shard.to_local().add_(1)
            try:
                manager.save_checkpoint(state, optimizer, scheduler, 1, 10, cursor=3)
            except ValueError as error:
                assert "immutable" in str(error)
            else:
                raise AssertionError("conflicting shard was accepted")
            assert {
                p.name: p.read_bytes() for p in path.glob(f"*rank_{rank}*")
            } == before
            deadline = time.monotonic() + 90
            while not manager.ckpt_path_check(str(path)):
                assert time.monotonic() < deadline, "peer marker not visible"
                time.sleep(0.1)
            print(f"CHECKPOINT_IMMUTABLE_PASS rank={rank} mode={mode}", flush=True)
        finally:
            manager.finalize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--expected-package-root", type=Path, required=True)
    args = parser.parse_args()
    assert (
        Path(cosmos_rl.__file__).resolve().parent
        == args.expected_package_root.resolve()
    )
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    assert world == 2
    device = torch.device("cpu" if args.cpu else f"cuda:{os.environ['LOCAL_RANK']}")
    if not args.cpu:
        assert torch.cuda.is_available()
        torch.cuda.set_device(device)
    dist.init_process_group("gloo" if args.cpu else "nccl")
    dims = ParallelDims(
        dp_replicate=1,
        dp_shard=1,
        cp=1,
        tp=1,
        pp=2,
        world_size=2,
        pp_dynamic_shape=False,
    )
    try:
        immutable_shard_promotion(args.root, rank, device)
        for case in ("healthy", "writer-failure", "missing-key"):
            cfg = Config.from_dict(
                {
                    "train": {
                        "output_dir": str(args.root / case / "run"),
                        "timestamp": "run",
                        "ckpt": {
                            "enable_checkpoint": True,
                            "save_mode": "async",
                            "max_keep": 1,
                            "export_safetensors": False,
                        },
                    }
                }
            )
            manager = CheckpointMananger(cfg, dims, global_rank=rank)
            parts = [
                torch.nn.ModuleDict({f"layer{i}": torch.nn.Linear(2, 1)}).to(device)
                for i in range(2)
            ]
            model = torch.nn.ModuleList(parts)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

            def state():
                return {
                    key: value
                    for part in parts
                    for key, value in part.state_dict().items()
                }

            old = Path(manager.save_checkpoint(state(), optimizer, scheduler, 1, 10))
            manager._wait_for_pending_async_saves()
            dist.barrier()
            manager.save_check(1)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(2)
            expected = copy.deepcopy(state())
            started, release = threading.Event(), threading.Event()
            original_save = torch.save

            def save(value, path):
                if rank == 1 and "step_2" in str(path) and "model_rank" in str(path):
                    started.set()
                    assert release.wait(30), "harness did not release held writer"
                    if case == "writer-failure":
                        raise OSError("injected remote checkpoint writer failure")
                original_save(value, path)

            try:
                with patch("cosmos_rl.utils.checkpoint.torch.save", side_effect=save):
                    selected = Path(
                        manager.save_checkpoint(state(), optimizer, scheduler, 2, 10)
                    )
                    if rank == 1:
                        assert started.wait(10)
                    else:
                        manager._wait_for_pending_async_saves()
                    dist.barrier()
                    manager.save_check(2, val_score=1.0)
                    assert old.exists(), "pruned before the peer committed"
                    assert not manager.ckpt_path_check(str(selected))
                    if manager._is_master_rank():
                        assert manager.best_score == float("inf")
                    dist.barrier()
                    release.set()
                    try:
                        manager._wait_for_pending_async_saves()
                    except OSError as error:
                        assert (
                            rank == 1
                            and case == "writer-failure"
                            and "injected" in str(error)
                        )
                    else:
                        assert rank == 0 or case != "writer-failure"
                dist.barrier()
                manager.save_check(2, val_score=1.0)
                if case == "writer-failure":
                    assert old.exists() and not manager.ckpt_path_check(str(selected))
                else:
                    manager._wait_for_pending_async_saves()
                    dist.barrier()
                    # A collective orders processes, not shared-filesystem
                    # directory/negative-entry caches. Observe the completed
                    # mutation with a finite deadline; never retry a save or
                    # relax the earlier no-prune-before-commit assertion.
                    deadline = time.monotonic() + 90
                    while old.exists() or not manager.ckpt_path_check(str(selected)):
                        if time.monotonic() >= deadline:
                            raise AssertionError(
                                "Committed checkpoint/deletion did not become visible"
                            )
                        time.sleep(0.1)
                    if case == "missing-key":
                        file = selected / f"model_rank_{rank}.pth"
                        saved = torch.load(file, weights_only=False)
                        saved.pop(next(iter(saved)))
                        torch.save(saved, file)
                    cfg.train.resume = str(selected)
                    with torch.no_grad():
                        for parameter in model.parameters():
                            parameter.zero_()
                    try:
                        manager.load_checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            "unused",
                            pp_model_parts=parts,
                            pp_model_module_paths=["", ""],
                        )
                    except ValueError as error:
                        assert (
                            case == "missing-key"
                            and "Incomplete pipeline checkpoint" in str(error)
                        )
                        assert all(
                            torch.count_nonzero(p).item() == 0
                            for p in model.parameters()
                        )
                    else:
                        assert case == "healthy"
                        torch.testing.assert_close(state(), expected, rtol=0, atol=0)
                dist.barrier()
                print(f"CHECKPOINT_COMMIT_PASS rank={rank} case={case}", flush=True)
            finally:
                release.set()
                try:
                    manager.finalize()
                except OSError:
                    assert rank == 1 and case == "writer-failure"
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
