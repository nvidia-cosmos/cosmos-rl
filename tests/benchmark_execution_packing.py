# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Installed public-launcher benchmark of variable-length recorded text.

Uses the same tiny random Qwen2 setup as the execution canary. Reports warm
optimizer throughput and peak allocation, separate from launcher startup.
"""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


from cosmos_rl.execution.corpus import CorpusWriter
from cosmos_rl.execution.records import ModelIdentity, SampleRecord
from execution_workflow_canary import base, checkpoint, launch


def update_events(path):
    result = []
    for line in path.read_text().splitlines():
        start = line.find('{"event": "recorded_text_update"')
        if start >= 0:
            result.append(json.loads(line[start:]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--profile", choices=("small", "long", "both"), default="both")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True)
    if args.profile == "both":
        combined = {}
        for profile, steps in (("small", args.steps), ("long", min(args.steps, 64))):
            destination = root / profile
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--output",
                    str(destination),
                    "--profile",
                    profile,
                    "--steps",
                    str(steps),
                ],
                check=True,
            )
            combined[profile] = json.loads((destination / "results.json").read_text())
            (root / "results.json").write_text(json.dumps(combined, indent=2))
        print("RECORDED_TEXT_PACKING_BENCHMARK_PASSED", flush=True)
        return
    model = root / "model"
    long = args.profile == "long"
    vocab = checkpoint(
        model,
        **(
            {
                "hidden_size": 512,
                "intermediate_size": 1536,
                "num_hidden_layers": 8,
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "max_position_embeddings": 2048,
            }
            if long
            else {}
        ),
    )
    writer = CorpusWriter(root / "corpus", run_id="packing-benchmark")
    for i in range(args.steps * 4):
        length = ((64, 128, 512, 2048) if long else (16, 32, 64, 256))[i % 4]
        ids = [100 + (i + j) % 1000 for j in range(length)]
        writer.commit(
            SampleRecord(
                sample_id=str(i),
                occurrence_id=str(i),
                group_id=str(i // 4),
                group_size=4,
                group_slot=i % 4,
                producer=ModelIdentity(
                    role="actor", name="uniform", revision="v1", behavior_version=0
                ),
                inputs={},
                targets={"token_ids": ids, "label_ids": ids},
                masks={"policy_gradient": [True] * (length - 1)},
                behavior_logprobs=[-math.log(vocab)] * (length - 1),
                annotations={"advantages": [0.2 if i % 2 else -0.2] * (length - 1)},
            )
        )
    common = base(root, model)
    common["policy"]["model_max_length"] = 2048 if long else 256
    common["execution"].update(
        source={"kind": "corpus", "path": str(root / "corpus")},
        objectives={"supervised": 0.8, "policy_gradient": 0.2},
    )
    common["train"].update(
        max_num_steps=args.steps, train_batch_per_replica=4, param_dtype="bfloat16"
    )
    common["train"]["train_policy"]["mini_batch"] = 4
    common["train"]["ckpt"].update(save_freq=args.steps)
    os.environ["EXECUTION_CANARY_PROFILE"] = "1"
    results = []
    for packed in (False, True):
        config = copy.deepcopy(common)
        config["train"]["sequence_packing"] = packed
        name = "packed" if packed else "unpacked"
        started = time.perf_counter()
        launch(root, name, config, corpus=True)
        elapsed = time.perf_counter() - started
        rows = update_events(root / f"{name}.log")
        assert len(rows) == args.steps, len(rows)
        assert all(row["train/packing_active"] == int(packed) for row in rows)
        warm = rows[8:]
        seconds = sum(row["seconds"] for row in warm)
        results.append(
            {
                "mode": name,
                "profile": args.profile,
                "dtype": "bfloat16",
                "steps": len(rows),
                "wall_seconds": elapsed,
                "optimizer_seconds": sum(row["seconds"] for row in rows),
                "startup_checkpoint_shutdown_seconds": elapsed
                - sum(row["seconds"] for row in rows),
                "warm_steps": len(warm),
                "warm_seconds": seconds,
                "warm_useful_tokens_per_second": sum(
                    row["train/useful_input_tokens"] for row in warm
                )
                / seconds,
                "padding_fraction": 1
                - sum(row["train/useful_input_tokens"] for row in warm)
                / sum(row["train/model_input_tokens"] for row in warm),
                "cuda_peak_bytes": max(row["cuda_peak_bytes"] for row in rows),
                "warm_cuda_peak_bytes": max(row["cuda_peak_bytes"] for row in warm),
                "warm_resident_bytes": max(row["resident_bytes"] for row in warm),
                "mean_loss": statistics.mean(row["train/loss_avg"] for row in warm),
            }
        )
        (root / "results.json").write_text(json.dumps(results, indent=2))
    print("RECORDED_TEXT_PACKING_BENCHMARK_PASSED", flush=True)


if __name__ == "__main__":
    main()
