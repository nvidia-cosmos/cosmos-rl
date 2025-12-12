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

from torch.utils.data import Dataset
from typing import Optional, Any, Dict
from cosmos_rl.launcher.worker_entry import main as launch_worker
from cosmos_rl.utils.logging import logger
from cosmos_rl.policy.config import Config as CosmosConfig

import torch
from libero.libero import benchmark as libero_bench


class LIBERODataset(Dataset):
    def __init__(self, num_trials_per_task=50, train_val="train"):
        self.train_val = train_val
        self.num_trials_per_task = num_trials_per_task

    def setup(self, config: CosmosConfig, *args, **kwargs):

        self.config = config
        if self.train_val == "train":
            self.task_suite_name = config.train.train_policy.dataset.subset
        elif self.train_val == "val":
            self.task_suite_name = config.validation.dataset.subset

        benchmark_dict = libero_bench.get_benchmark_dict()
        task_suite = benchmark_dict[self.task_suite_name]()
        num_tasks_in_suite = task_suite.n_tasks
        dataframes = []

        if self.task_suite_name in [
            "libero_10",
            "libero_90",
            "libero_goal",
            "libero_object",
            "libero_spatial",
        ]:
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
                        "task_id": task_id,
                        "trial_id": i,
                        "trial_seed": -1,
                    }
                    dataframes.append(data)
            self.dataframe = dataframes
            print(f"dataset <{self.train_val}> len: {len(self.dataframe)}")
        else:
            raise ValueError

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        return self.dataframe[item]


def vla_reward_fn(
    to_be_evaluated: Dict[str, Any], reference: Optional[Any] = None, prompt=None, **kwargs
) -> float:
    """
    Custom reward function for VLA tasks.

    For VLA, the reward is computed from environment success/failure stored in metadata.
    The 'to_be_evaluated' is the completion dictionary, but we extract the actual reward
    from the completion dictionary.

    Args:
        to_be_evaluated: The completion dictionary (action sequence text)
        reference: Not used for VLA (environment-based rewards)
        prompt: The task prompt
        **kwargs: Additional arguments, including 'metadata' with VLA episode info

    Returns:
        Reward: 1.0 for success, 0.0 for failure
    """
    return float(to_be_evaluated.get("complete", False))


if __name__ == "__main__":

    def get_dataset(config: CosmosConfig) -> Dataset:
        return LIBERODataset(num_trials_per_task=50, train_val="train")

    def get_val_dataset(config: CosmosConfig) -> Dataset:
        return LIBERODataset(num_trials_per_task=50, train_val="val")

    launch_worker(
        dataset=get_dataset,
        val_dataset=get_val_dataset,
        reward_fns=[vla_reward_fn],
        val_reward_fns=[vla_reward_fn],
    )
