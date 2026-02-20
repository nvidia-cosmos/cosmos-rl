# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import sys
# Enable EP mesh to be represented by TP mesh, and also treat EP as a sub-group of Data Parallelism.

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from torch.utils.data import Dataset
from cosmos_rl.policy.config import Config as CosmosConfig

try:
    import wandb
except ImportError:
    wandb = None


def modify_messages(messages, max_pixels=None):
    for message in messages:
        if isinstance(message["content"], str):
            message["content"] = [{"type": "text", "text": message["content"]}]
        for content in message["content"]:
            if content["type"] == "image":
                if max_pixels is not None:
                    content["max_pixels"] = max_pixels
            elif content["type"] == "video":
                content["fps"] = 1
                if max_pixels is not None:
                    content["total_pixels"] = max_pixels
    return messages


class CustomDataset(Dataset):
    """
    Custom dataset for Nemotron-3-Nano Vision-Language Model alignment.
    Assume the dataset are in JSONL format stored in `config.train.train_policy.dataset.name`, and each line is a JSON object with 'messages' key.
    """

    def setup(self, config: CosmosConfig, *args, **kwargs):
        self.data_list = []
        data_path = config.train.train_policy.dataset.name
        jsonl_files = sorted([f for f in os.listdir(data_path) if f.endswith(".jsonl")])
        for file_name in jsonl_files:
            if not config.custom.get("include_video", False) and "webvid" in file_name:
                continue
            with open(os.path.join(data_path, file_name)) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.data_list.append(json.loads(line)["messages"])
        self.max_pixels = config.policy.model_max_length * 0.9 * ((16 * 2) ** 2)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int) -> list[dict]:
        sample = self.data_list[idx]
        sample = modify_messages(sample, self.max_pixels)
        return sample


def get_dataset(config: CosmosConfig):
    return CustomDataset()


if __name__ == "__main__":
    import cosmos_rl

    # Launch the worker
    cosmos_rl.launcher.worker_entry.main(
        # Uncomment this if you want to use a custom dataset
        dataset=get_dataset,
    )
