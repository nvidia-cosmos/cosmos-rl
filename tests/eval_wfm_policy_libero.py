#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Standalone evaluation of CosmosPolicyModel on LIBERO using the
WFMPolicyRollout + CosmosPolicyVLA wrapper.

Requires no Redis or distributed framework – just instantiates the rollout
and calls ``evaluate()`` directly.

Usage:
    MUJOCO_GL=egl python tests/eval_wfm_policy_libero.py \\
        --ckpt-path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \\
        --task-suite libero_10 \\
        --num-envs 4 \\
        --trials 3
"""

import argparse
import os
import sys

if "cosmos_rl" not in sys.modules:
    _stub = type(sys)("cosmos_rl")
    _stub.__path__ = [os.path.join(os.path.dirname(__file__), "..", "cosmos_rl")]
    sys.modules["cosmos_rl"] = _stub

os.environ.setdefault("MUJOCO_GL", "egl")


def main():
    p = argparse.ArgumentParser(description="CosmosPolicyModel LIBERO eval")
    p.add_argument("--ckpt-path", default="nvidia/Cosmos-Policy-LIBERO-Predict2-2B")
    p.add_argument("--dataset-stats", default="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json")
    p.add_argument("--t5-embeddings", default="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl")
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=520)
    p.add_argument("--num-denoising-steps", type=int, default=5)
    p.add_argument("--chunk-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--trials", type=int, default=1, help="Trials per task")
    p.add_argument("--task-ids", type=int, nargs="*", default=None,
                    help="Task IDs to evaluate (default: all)")
    p.add_argument("--save-video", action="store_true", default=True)
    p.add_argument("--no-save-video", dest="save_video", action="store_false")
    p.add_argument("--output-dir", default="./outputs/wfm_policy_eval")
    p.add_argument("--vae-fp32", action="store_true", default=False)
    args = p.parse_args()

    from cosmos_rl.policy.config import Config

    config = Config.model_validate({
        "custom": {
            "ckpt_path": args.ckpt_path,
            "dataset_stats_path": args.dataset_stats,
            "t5_text_embeddings_path": args.t5_embeddings,
            "task_suite_name": args.task_suite,
            "num_envs": args.num_envs,
            "max_steps": args.max_steps,
            "num_denoising_steps": args.num_denoising_steps,
            "chunk_size": args.chunk_size,
            "seed": args.seed,
            "save_video": args.save_video,
            "vae_fp32": args.vae_fp32,
        },
        "train": {"output_dir": args.output_dir, "train_policy": {"type": "grpo"}},
        "rollout": {"backend": "wfm-policy"},
        "validation": {
            "enable": True,
            "dataset": {"name": "libero", "subset": args.task_suite},
        },
    })

    from cosmos_rl.rollout.wfm_policy_rollout import WFMPolicyRollout

    print(f"Creating WFMPolicyRollout (suite={args.task_suite}, envs={args.num_envs}) ...")
    rollout = WFMPolicyRollout(config, parallel_dims=None, device="cuda")

    print("Initializing engine (loading model + assets) ...")
    rollout.init_engine()

    print(f"Evaluating: tasks={args.task_ids or 'all'}, trials={args.trials}")
    result = rollout.evaluate(task_ids=args.task_ids, trials_per_task=args.trials)

    print("\n" + "=" * 60)
    print(f"SUCCESS RATE: {result['success_rate']:.1f}%  "
          f"({result['n_success']}/{result['n_total']})")
    print("=" * 60)

    rollout.env_manager.stop_simulator()


if __name__ == "__main__":
    main()
