#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Single entry point for VLA LIBERO evaluation via OpenVLARollout.evaluate().

Supports cosmos-policy and openvla-oft (and other vla_type from config).

CosmosPolicy (CLI):
    MUJOCO_GL=egl python tests/eval_cosmos_policy_libero.py \\
        --vla-type cosmos-policy --ckpt-path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \\
        --task-suite libero_10 --trials 3

OpenVLA-OFT (CLI):
    MUJOCO_GL=egl python tests/eval_cosmos_policy_libero.py \\
        --vla-type openvla-oft --ckpt-path Haozhan72/Openvla-oft-SFT-libero10-trajall \\
        --task-suite libero_10 --trials 3

TOML config (vla_type and model from config):
    MUJOCO_GL=egl python tests/eval_cosmos_policy_libero.py \\
        --config configs/cosmos-policy/cosmos-policy-libero10-eval.toml

Multi-GPU (tasks×trials split across ranks):
    MUJOCO_GL=egl torchrun --nproc-per-node 8 tests/eval_cosmos_policy_libero.py \\
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
    vla_type = getattr(args, "vla_type", "cosmos-policy")

    base = {
        "train": {"output_dir": args.output_dir},
        "policy": {"model_name_or_path": ckpt},
        "rollout": {"backend": "vla", "n_generation": 1},
        "validation": {
            "enable": True,
            "dataset": {"name": "libero", "subset": args.task_suite},
        },
        "vla": {
            "vla_type": vla_type,
            "num_envs": args.num_envs,
            "training_chunk_size": 16,
            "save_video": args.save_video,
            "max_steps": args.max_steps,
        },
    }

    if vla_type == "cosmos-policy":
        base["custom"] = {
            "ckpt_path": ckpt,
            "dataset_stats_path": f"{ckpt}/libero_dataset_statistics.json",
            "t5_text_embeddings_path": f"{ckpt}/libero_t5_embeddings.pkl",
            "task_suite_name": args.task_suite,
            "num_denoising_steps": 5,
            "chunk_size": 16,
            "seed": 1,
        }
        base["vla"]["use_proprio"] = True
        base["vla"]["proprio_dim"] = 9
        base["vla"]["num_images_in_input"] = 2
    else:
        # openvla-oft / openvla
        base["vla"]["use_proprio"] = False
        base["vla"]["proprio_dim"] = 7
        base["vla"]["num_images_in_input"] = 1

    return Config.model_validate(base)


def _parse_args():
    p = argparse.ArgumentParser(
        description="VLA LIBERO evaluation (cosmos-policy, openvla-oft, etc.)"
    )
    p.add_argument("--config", type=str, default=None, help="TOML config (optional)")
    p.add_argument("--ckpt-path", default="nvidia/Cosmos-Policy-LIBERO-Predict2-2B")
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=520)
    p.add_argument("--trials", type=int, default=1, help="Trials per task")
    p.add_argument("--task-ids", type=int, nargs="*", default=None)
    p.add_argument("--save-video", action="store_true", default=True)
    p.add_argument("--no-save-video", dest="save_video", action="store_false")
    p.add_argument(
        "--no-video",
        action="store_true",
        help="Alias for --no-save-video when using --config",
    )
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
            raw.setdefault("validation", {}).setdefault("dataset", {})["subset"] = (
                args.task_suite
            )
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
    all_task_ids = (
        args.task_ids if args.task_ids is not None else list(range(suite.n_tasks))
    )
    all_pairs = [
        (tid, trial) for tid in all_task_ids for trial in range(trials_per_task)
    ]
    # Group by (task_id, trial_id) so each rank's first batches are same-task (same description length).
    all_pairs = sorted(all_pairs, key=lambda p: (p[0], p[1]))
    # Assign contiguous chunks to ranks so rollout enqueue gets same-task batches.
    chunk_size = (len(all_pairs) + world_size - 1) // world_size
    start = rank * chunk_size
    my_payload_pairs = all_pairs[start : start + chunk_size]

    if rank == 0:
        logger.info(
            f"Evaluation: suite={task_suite}, tasks={len(all_task_ids)}, "
            f"trials/task={trials_per_task}, total jobs={len(all_pairs)}, ranks={world_size}"
        )
    logger.info(f"[Rank {rank}] {len(my_payload_pairs)} jobs (task×trial pairs)")

    local_successes, local_total = 0, 0
    rollout_obj = None
    if my_payload_pairs:
        from cosmos_rl.rollout.vla_rollout import OpenVLARollout

        # Use explicit cuda:local_rank so model weights load on the correct GPU (avoids all ranks allocating on GPU 0).
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        rollout = OpenVLARollout(
            config=config, parallel_dims=None, device=device
        )
        rollout_obj = rollout
        result = rollout.evaluate(payload_pairs=my_payload_pairs)

        local_successes = result["n_success"]
        local_total = result["n_total"]

        per_task: dict[int, list[bool]] = {}
        for r in result["per_task"]:
            per_task.setdefault(r["task_id"], []).append(r["complete"])

    if world_size > 1:
        dev = torch.device("cuda")
        s_t = torch.tensor([local_successes], dtype=torch.long, device=dev)
        t_t = torch.tensor([local_total], dtype=torch.long, device=dev)
        dist.all_reduce(s_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_t, op=dist.ReduceOp.SUM)
        global_successes, global_total = s_t.item(), t_t.item()
    else:
        global_successes, global_total = local_successes, local_total

    if rollout_obj is not None:
        rollout_obj.env_manager.stop_simulator()

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
