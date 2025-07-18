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

import os
import argparse
import toml

from torch.utils.data import Dataset
from datasets import concatenate_datasets
import cosmos_rl.utils.util as util
import cosmos_rl.utils.cache as cache
from transformers import AutoTokenizer
from cosmos_rl.dispatcher.run_web_panel import main as launch_dispatcher
from cosmos_rl.policy.config import (
    Config,
    config_hash,
)


# This dataset is used for SFT with raw text input, which is used for models like Mistral
# It converts the conversation list to string format for models requiring raw text input
# This handles cases like Mistral where conversation dicts need string conversion
# to avoid role alternation errors
class SFTRawTextDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def setup(
        self,
        config: Config,
        tokenizer: AutoTokenizer,
    ):
        self.config = config.train.train_policy
        self.tokenizer = tokenizer
        self.column_name = self.config.conversation_column_name
        self.cache = None
        if self.config.enable_dataset_cache:
            cache_folder = os.path.join(
                os.environ.get(
                    "COSMOS_CACHE",
                    os.path.join(os.path.expanduser("~"), ".cache/cosmos/"),
                ),
                "datasets_cache",
                f"{self.config.dataset.name}-{config_hash(config)}",
            )
            print(f"SFTRawTextDataset Cache folder: {cache_folder}")
            self.cache = cache.DiskCache(cache_folder)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Check cache first if enabled
        if self.cache is not None:
            cached_item = self.cache.get(idx)
            if cached_item is not None:
                return cached_item

        # Retrieve raw item from dataset
        raw_item = (
            self.dataset[idx][self.column_name]
            if self.column_name
            else self.dataset[idx]
        )

        # Convert conversation list to string format
        if isinstance(raw_item, list):
            raw_item = "\n".join(
                [f"{turn['role']}: {turn['content']}" for turn in raw_item]
            )

        # Cache the processed item if caching is enabled
        if self.cache is not None:
            self.cache.set(idx, raw_item)

        return raw_item


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
        dataset=SFTRawTextDataset(dataset=train_dataset),
    )
