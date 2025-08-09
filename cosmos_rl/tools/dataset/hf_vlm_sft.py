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

from torch.utils.data import Dataset, ConcatDataset
from datasets import load_dataset
from cosmos_rl.launcher.worker_entry import main as launch_worker
from cosmos_rl.policy.config import Config
from cosmos_rl.policy.config import Config as CosmosConfig
from transformers import AutoTokenizer
import argparse
import toml


class HFVLMSFTDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def setup(self, config: Config, tokenizer: AutoTokenizer, *args, **kwargs):
        """
        Called by launcher after being mounted
        """
        self.config = config
        self.tokenizer = tokenizer

        if config.train.train_policy.dataset.split:
            if isinstance(config.train.train_policy.dataset.split, list):
                dataset_list = []
                for split_name in config.train.train_policy.dataset.split:
                    dataset_list.append(self.dataset[split_name])
                self.dataset = ConcatDataset(dataset_list)
            else:
                assert isinstance(config.train.train_policy.dataset.split, str)
                self.dataset = self.dataset[config.train.train_policy.dataset.split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        # The dataset is a list of {messages, images} pair
        conversations = self.dataset[idx]
        return conversations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_known_args()[0]
    with open(args.config, "r") as f:
        config = toml.load(f)
    config = Config.from_dict(config)

    def get_dataset(config: CosmosConfig) -> Dataset:
        dataset = load_dataset(
            config.train.train_policy.dataset.name,
            config.train.train_policy.dataset.subset,
        )
        return HFVLMSFTDataset(dataset)

    # It is best practice to pass the dataset as a factory function
    # so that the dataset can be loaded on demand. (Not all workers need it)
    launch_worker(
        dataset=get_dataset,
    )
