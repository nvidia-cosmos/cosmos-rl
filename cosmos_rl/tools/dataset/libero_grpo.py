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

from typing import Any, List, Dict
from torch.utils.data import Dataset, ConcatDataset
from datasets import load_dataset
from cosmos_rl.launcher.worker_entry import main as launch_worker
from cosmos_rl.policy.config import Config
from transformers import AutoTokenizer
from cosmos_rl.dispatcher.data.packer import DataPacker, HFVLMDataPacker
from cosmos_rl.utils.logging import logger
from cosmos_rl.policy.config import Config as CosmosConfig
import argparse
import toml
import base64
import io

import torch


class LIBERODataset(Dataset):
    def __init__(self,
                 num_trials_per_task=50,
                 train_val ="train"):
        self.train_val = train_val
        self.num_trials_per_task = num_trials_per_task

    def setup(self, config: CosmosConfig, *args, **kwargs):
        from libero.libero import benchmark as libero_bench

        self.config = config
        if self.train_val == 'train':
            self.task_suite_name = config.train.train_policy.dataset.subset
        elif self.train_val == 'val':
            self.task_suite_name = config.validation.dataset.subset

        benchmark_dict = libero_bench.get_benchmark_dict()
        task_suite = benchmark_dict[self.task_suite_name]()
        num_tasks_in_suite = task_suite.n_tasks
        dataframes = []
        
        if self.task_suite_name in ["libero_10", "libero_90", "libero_goal",  "libero_object",  "libero_spatial"]:
            for task_id in range(num_tasks_in_suite):
                if self.train_val == "train":
                    trials_range = list(range(0, int(self.num_trials_per_task)))
                elif self.train_val == "val":
                    trials_range = list(range(0, int(self.num_trials_per_task)))  
                else:
                    raise ValueError
                for i in trials_range:
                    data = {
                        "task_suite_name": self.task_suite_name,
                        "task_id": torch.tensor(task_id, dtype=torch.int64).unsqueeze(0),
                        "trial_id": torch.tensor(i, dtype=torch.int64).unsqueeze(0),
                        "trial_seed": torch.tensor(-1, dtype=torch.int64).unsqueeze(0)
                    }
                    dataframes.append(data)
            self.dataframe = dataframes
            print(f'dataset len: {len(self.dataframe)}')
        else:
            raise ValueError
    
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        return self.dataframe[item]


if __name__ == "__main__":

    def get_dataset(config: CosmosConfig) -> Dataset:
        return LIBERODataset(50, "train")
    
    def get_val_dataset(config: CosmosConfig) -> Dataset:
        return LIBERODataset(50, "val")

    launch_worker(
        dataset=get_dataset,
        val_dataset=get_val_dataset,
    )