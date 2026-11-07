# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Application entry used by the real execution-workflow GPU canary."""

import os
import json
import time

import torch

from torch.utils.data import Dataset

from cosmos_rl.launcher.worker_entry import main


class Inputs(Dataset):
    def __init__(self, config):
        self.texts = tuple(
            f"The example number {index} describes supervised prediction over recorded data."
            for index in range(8)
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index]


def reward(completion, reference, **kwargs):
    return (sum(map(ord, completion)) % 101) / 100


if __name__ == "__main__":
    if os.environ.get("EXECUTION_CANARY_PROFILE") == "1":
        from cosmos_rl.policy.trainer.llm_trainer.sft_trainer import SFTTrainer

        original_step = SFTTrainer.step_training

        def profiled_step(self, *args, **kwargs):
            started = time.perf_counter()
            result = original_step(self, *args, **kwargs)
            torch.cuda.synchronize()
            metrics = {
                key: value.item() if isinstance(value, torch.Tensor) else value
                for key, value in result.items()
            }
            print(
                json.dumps(
                    {
                        "event": "recorded_text_update",
                        "seconds": time.perf_counter() - started,
                        "cuda_peak_bytes": torch.cuda.max_memory_allocated(),
                        **metrics,
                    }
                ),
                flush=True,
            )
            return result

        SFTTrainer.step_training = profiled_step
    main(
        dataset=None if os.environ.get("EXECUTION_CANARY_CORPUS") == "1" else Inputs,
        reward_fns=[reward],
    )
