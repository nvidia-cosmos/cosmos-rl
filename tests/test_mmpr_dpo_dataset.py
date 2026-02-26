#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Test script for DPODataset (MMPR-v1.2) loading.
Usage:
    cd nemotron_vl
    python test_mmpr_dpo_dataset.py

Optional: override dataset path with --dataset_path
    python test_mmpr_dpo_dataset.py --dataset_path /path/to/MMPR-v1.2
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.util import call_setup
import sys 
print(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nemotron_vl"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nemotron_vl"))
from launcher_dpo import DPODataset


def build_test_config(dataset_path: str) -> CosmosConfig:
    """Build minimal config for DPODataset."""
    return CosmosConfig.from_dict({
        "redis": "12800",
        "train": {
            "resume": False,
            "epoch": 1,
            "output_dir": "/tmp/test_mmpr",
            "train_policy": {
                "type": "sft",
                "trainer_type": "dpo",
                "dataset": {"name": dataset_path, "subset": "", "split": "train"},
                "conversation_column_name": "",
                "mini_batch": 2,
            },
        },
        "policy": {
            "model_name_or_path": "dummy",
            "model_max_length": 2048,
            "parallelism": {
                "n_init_replicas": 1,
                "tp_size": 1,
                "cp_size": 1,
                "dp_shard_size": 1,
                "pp_size": 1,
                "dp_replicate_size": 1,
            },
        },
        "validation": {"enable": False},
        "logging": {"logger": ["console"]},
    })


def main():
    parser = argparse.ArgumentParser(description="Test MMPRDPODataset loading")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/workspace/ruipul/projects/vlm_data_curation/data_clean/debug_samples",
        help="Path to MMPR-v1.2 dataset root",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2,
        help="Number of samples to fetch and print",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full chosen/rejected content",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_path):
        print(f"ERROR: Dataset path does not exist: {args.dataset_path}")
        sys.exit(1)

    print(f"Loading DPODataset (MMPR) from {args.dataset_path}")
    config = build_test_config(args.dataset_path)
    dataset = DPODataset()
    call_setup(dataset, config)

    n = len(dataset)
    print(f"Dataset length: {n}")

    if n == 0:
        print("WARNING: Dataset is empty. Check meta.json and annotation files.")
        sys.exit(0)

    for i in range(min(args.num_samples, n)):
        print(f"\n--- Sample {i} ---")
        try:
            sample = dataset[i]
            assert "chosen" in sample and "rejected" in sample, "Missing chosen/rejected"
            chosen = sample["chosen"]
            rejected = sample["rejected"]

            print(f"  chosen:  {len(chosen)} messages")
            print(f"  rejected: {len(rejected)} messages")

            for j, msg in enumerate(chosen):
                role = msg.get("role", "?")
                content = msg.get("content", [])
                if isinstance(content, str):
                    preview = content[:80] + "..." if len(content) > 80 else content
                elif isinstance(content, list):
                    parts = []
                    for c in content:
                        if isinstance(c, dict):
                            if c.get("type") == "image":
                                img = c.get("image", "")
                                parts.append(f"[image: {img}]")
                            elif c.get("type") == "text":
                                t = c.get("text", "")[:60]
                                parts.append(f"text: {t}..." if len(t) >= 60 else f"text: {t}")
                        else:
                            parts.append(str(c)[:40])
                    preview = "; ".join(parts)
                else:
                    preview = str(content)[:80]
                print(f"    [{j}] {role}: {preview}")

            if args.verbose:
                print("  --- chosen assistant (first 200 chars) ---")
                for m in chosen:
                    if m.get("role") == "assistant":
                        c = m.get("content", "")
                        if isinstance(c, list):
                            for x in c:
                                if isinstance(x, dict) and x.get("type") == "text":
                                    c = x.get("text", "")
                                    break
                        print(f"    {str(c)[:200]}...")
                        break
                print("  --- rejected assistant (first 200 chars) ---")
                for m in rejected:
                    if m.get("role") == "assistant":
                        c = m.get("content", "")
                        if isinstance(c, list):
                            for x in c:
                                if isinstance(x, dict) and x.get("type") == "text":
                                    c = x.get("text", "")
                                    break
                        print(f"    {str(c)[:200]}...")
                        break

        except FileNotFoundError as e:
            print(f"  SKIP (image not found): {e}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n✓ DPODataset load test passed")


if __name__ == "__main__":
    main()
