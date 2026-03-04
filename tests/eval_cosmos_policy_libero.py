#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Single entry point for CosmosPolicy LIBERO evaluation via OpenVLARollout.evaluate().

CLI-only (single GPU):
    MUJOCO_GL=egl python tests/eval_cosmos_policy_libero.py \
        --ckpt-path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
        --task-suite libero_10 \
        --trials 3

TOML config (optional overrides):
    MUJOCO_GL=egl python tests/eval_cosmos_policy_libero.py \
        --config configs/cosmos-policy/cosmos-policy-libero10-eval.toml

Multi-GPU (tasks split across ranks):
    MUJOCO_GL=egl torchrun --nproc-per-node 8 tests/eval_cosmos_policy_libero.py \
        --config configs/cosmos-policy/cosmos-policy-libero10-eval.toml --trials 50
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402


def _init_distributed() -> tuple[int, int]:
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        return dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(0)
    return 0, 1


def _load_toml(path: str) -> dict:
    try:
        import tomli
    except ImportError:
        import tomllib as tomli  # type: ignore[no-redef]
    with open(path, "rb") as f:
        return tomli.load(f)


def _build_config_from_args(args: argparse.Namespace):
    from cosmos_rl.policy.config import Config

    ckpt = args.ckpt_path
    return Config.model_validate({
        "mode": "colocated",
        "custom": {
            "ckpt_path": ckpt,
            "dataset_stats_path": f"{ckpt}/libero_dataset_statistics.json",
            "t5_text_embeddings_path": f"{ckpt}/libero_t5_embeddings.pkl",
            "task_suite_name": args.task_suite,
            "num_denoising_steps": 5,
            "chunk_size": 16,
            "seed": 1,
        },
        "train": {"output_dir": args.output_dir},
        "policy": {"model_name_or_path": ckpt},
        "rollout": {"backend": "vla", "n_generation": 1},
        "validation": {
            "enable": True,
            "dataset": {"name": "libero", "subset": args.task_suite},
        },
        "vla": {
            "vla_type": "cosmos-policy",
            "use_proprio": True,
            "num_envs": args.num_envs,
            "proprio_dim": 9,
            "num_images_in_input": 2,
            "training_chunk_size": 16,
            "save_video": args.save_video,
            "max_steps": args.max_steps,
        },
    })


def _parse_args():
    p = argparse.ArgumentParser(description="CosmosPolicy LIBERO evaluation")
    p.add_argument("--config", type=str, default=None, help="TOML config (optional)")
    p.add_argument("--ckpt-path", default="nvidia/Cosmos-Policy-LIBERO-Predict2-2B")
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=520)
    p.add_argument("--trials", type=int, default=1, help="Trials per task")
    p.add_argument("--task-ids", type=int, nargs="*", default=None)
    p.add_argument("--save-video", action="store_true", default=True)
    p.add_argument("--no-save-video", dest="save_video", action="store_false")
    p.add_argument("--no-video", action="store_true", help="Alias for --no-save-video when using --config")
    p.add_argument("--output-dir", default="./outputs/cosmos-policy-eval")
    return p.parse_args()


def main():
    args = _parse_args()
    rank, world_size = _init_distributed()

    from cosmos_rl.policy.config import Config
    from cosmos_rl.utils.logging import logger

    if args.config:
        raw = _load_toml(args.config)
        if args.task_suite != "libero_10":
            raw.setdefault("validation", {}).setdefault("dataset", {})["subset"] = args.task_suite
            raw.setdefault("custom", {})["task_suite_name"] = args.task_suite
        if args.num_envs != 1:
            raw.setdefault("vla", {})["num_envs"] = args.num_envs
        if args.no_video or not args.save_video:
            raw.setdefault("vla", {})["save_video"] = False
        config = Config.from_dict(raw)
        task_suite = config.validation.dataset.subset
        trials_per_task = raw.get("custom", {}).get("trials_per_task", 50)
        if args.trials is not None:
            trials_per_task = args.trials
    else:
        config = _build_config_from_args(args)
        task_suite = args.task_suite
        trials_per_task = args.trials

    from cosmos_rl.simulators.libero.utils import get_benchmark_overridden

    suite = get_benchmark_overridden(task_suite)()
    all_task_ids = args.task_ids if args.task_ids is not None else list(range(suite.n_tasks))
    my_task_ids = all_task_ids[rank::world_size]

    if rank == 0:
        logger.info(
            f"Evaluation: suite={task_suite}, tasks={len(all_task_ids)}, "
            f"trials/task={trials_per_task}, ranks={world_size}"
        )
    logger.info(f"[Rank {rank}] {len(my_task_ids)} tasks: {my_task_ids}")

    local_successes, local_total = 0, 0
    if my_task_ids:
        from cosmos_rl.rollout.vla_rollout import OpenVLARollout

        rollout = OpenVLARollout(config=config, parallel_dims=None, device=torch.device("cuda"))
        result = rollout.evaluate(task_ids=my_task_ids, trials_per_task=trials_per_task)

        local_successes = result["n_success"]
        local_total = result["n_total"]

        per_task: dict[int, list[bool]] = {}
        for r in result["per_task"]:
            per_task.setdefault(r["task_id"], []).append(r["complete"])
        for tid in sorted(per_task):
            results = per_task[tid]
            logger.info(f"[Rank {rank}]   task {tid}: {sum(results)}/{len(results)}")

        rollout.env_manager.stop_simulator()

    if world_size > 1:
        dev = torch.device("cuda")
        s_t = torch.tensor([local_successes], dtype=torch.long, device=dev)
        t_t = torch.tensor([local_total], dtype=torch.long, device=dev)
        dist.all_reduce(s_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_t, op=dist.ReduceOp.SUM)
        global_successes, global_total = s_t.item(), t_t.item()
    else:
        global_successes, global_total = local_successes, local_total

    if rank == 0:
        sr = global_successes / global_total * 100 if global_total else 0
        logger.info(
            f"\n{'=' * 60}\n"
            f"FINAL: {global_successes}/{global_total} = {sr:.1f}%\n"
            f"Suite: {task_suite}, Tasks: {len(all_task_ids)}, "
            f"Trials/task: {trials_per_task}, Ranks: {world_size}\n"
            f"{'=' * 60}"
        )

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
