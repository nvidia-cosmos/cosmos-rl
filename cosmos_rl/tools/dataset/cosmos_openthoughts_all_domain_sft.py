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

import argparse
import toml

from torch.utils.data import Dataset
from datasets import concatenate_datasets
import cosmos_rl.utils.util as util
from cosmos_rl.launcher.worker_entry import main as launch_dispatcher
from cosmos_rl.policy.config import Config


def convert_from_value_to_role_content(sample):
    if isinstance(sample, dict):
        for key in ["conversation", "messages", "data", "content"]:
            if key in sample and isinstance(sample[key], list):
                return convert_from_value_to_role_content(sample[key])
        if "from" in sample or "value" in sample:
            return convert_from_value_to_role_content([sample])

    if isinstance(sample, list):
        converted = []
        for msg in sample:
            if isinstance(msg, dict):
                role_mapping = {
                    "human": "user",
                    "gpt": "assistant",
                    "assistant": "assistant",
                    "user": "user",
                    "system": "system",
                }

                from_value = msg.get("from", "").lower()
                role = role_mapping.get(from_value, "user")

                content = msg.get("value", msg.get("content", ""))

                converted.append({"role": role, "content": content})
            else:
                converted.append(msg)

        return converted

    if isinstance(sample, str):
        return sample

    return sample


class OpenThoughtsSFTDataset(Dataset):
    def __init__(self, dataset, convert_fn=convert_from_value_to_role_content):
        self.dataset = dataset
        self.convert_fn = convert_fn

    def setup(
        self,
        config: Config,
    ):
        self.config = config.train.train_policy

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]["conversations"]
        converted_item = self.convert_fn(item)
        return converted_item


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_known_args()[0]
    with open(args.config, "r") as f:
        config = toml.load(f)
    config = Config.from_dict(config)
    # Download HF dataset only on launcher worker
    dataset = util.load_data_from_disk_or_hf(
        config.train.train_policy.dataset.name,
        config.train.train_policy.dataset.subset,
        config.train.train_policy.dataset.revision or None,
    )
    dataset_list = []
    for split_name in config.train.train_policy.dataset.split:
        print(
            f"Appending split {split_name}, dataset size = {len(dataset[split_name])}"
        )
        dataset_list.append(dataset[split_name])
    train_dataset = concatenate_datasets(dataset_list)
    launch_dispatcher(
        dataset=OpenThoughtsSFTDataset(dataset=train_dataset),
    )
