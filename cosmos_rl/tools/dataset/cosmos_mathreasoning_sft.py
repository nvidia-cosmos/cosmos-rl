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


class MathReasoningSFTDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def setup(
        self,
        config: Config,
    ):
        self.config = config.train.train_policy

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]["messages"]


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

    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("problem")
            solution = example.pop("generated_solution")

            example.clear()

            data = {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": solution},
                ],
            }
            return data

        return process_fn

    dataset_list = []
    for split_name in config.train.train_policy.dataset.split:
        print(
            f"Appending split {split_name}, dataset size = {len(dataset[split_name])}"
        )
        filtered_dataset = dataset[split_name].filter(
            lambda example: example["problem_type"] == "has_answer_extracted"
        )
        dataset_list.append(
            filtered_dataset.map(function=make_map_fn("cot"), with_indices=True)
        )

    train_dataset = concatenate_datasets(dataset_list)
    launch_dispatcher(
        dataset=MathReasoningSFTDataset(dataset=train_dataset),
    )
