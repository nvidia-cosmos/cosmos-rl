# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Application entry used by the real execution-workflow GPU canary."""

import os

from torch.utils.data import Dataset

from cosmos_rl.launcher.worker_entry import main


class Inputs(Dataset):
    def __init__(self, config):
        self.supervised = config.train.train_policy.type == "sft"

    def __len__(self):
        return 8

    def __getitem__(self, index):
        text = f"The example number {index} describes supervised prediction over recorded data."
        return text if self.supervised else {"prompt": text, "reference": "prediction"}


def reward(completion, reference, **kwargs):
    return (sum(map(ord, completion)) % 101) / 100


if __name__ == "__main__":
    main(
        dataset=None if os.environ.get("EXECUTION_CANARY_CORPUS") == "1" else Inputs,
        reward_fns=[reward],
    )
