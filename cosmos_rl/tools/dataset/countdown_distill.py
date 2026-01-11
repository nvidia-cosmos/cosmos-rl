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
from typing import Any, Optional
import toml
from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.tools.dataset.math_grpo import (
    MathDataPacker,
)
from cosmos_rl.utils.logging import logger
from torch.utils.data import Dataset
from datasets import concatenate_datasets
import cosmos_rl.utils.util as util
from cosmos_rl.launcher.worker_entry import main as launch_dispatcher
from cosmos_rl.policy.config import Config
import re


class CountdownDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.content = "messages"

    def setup(
        self,
        config: Config,
    ):
        self.config = config
        self.tokenizer = util.setup_tokenizer(config.policy.model_name_or_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        prompt = item[self.content][:-1]
        prompt = self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        return RLPayload(prompt=prompt)


class CountdownValDataset(CountdownDataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.content = "prompt"

    def __getitem__(self, idx):
        item = self.dataset[idx]
        prompt = item[self.content]
        prompt = self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        return RLPayload(prompt=prompt)

    def __len__(self):
        if isinstance(self.config.validation.dataset.test_size, float):
            return int(len(self.dataset) * self.config.validation.dataset.test_size)
        elif isinstance(self.config.validation.dataset.test_size, int):
            return min(self.config.validation.dataset.test_size, len(self.dataset))
        return len(self.dataset)

    def get_reference_answer(self, idx: int) -> Any:
        """
        This is mandatory for GRPO to get a reference answer for reward computation.
        """
        return {
            "nums": self.dataset[idx]["nums"],
            "target": self.dataset[idx]["target"],
        }


def custom_reward_fn(
    to_be_evaluated: str, reference: Optional[Any] = None, *args, **kwargs
) -> float:
    # logger.info(f"Using custom_reward_fn for Countdown dataset {to_be_evaluated}")
    try:
        # Extract answer from content if it has think/answer tags
        content_match = re.search(r"<answer>(.*?)</answer>", to_be_evaluated, re.DOTALL)
        student_answer = (
            content_match.group(1).strip() if content_match else to_be_evaluated.strip()
        )
        from sympy import simplify
        from sympy.parsing.sympy_parser import parse_expr

        # Split into left and right
        # logger.info(f"Student answer extracted: {student_answer}")
        left_str, right_str = student_answer.split("=")
        left = parse_expr(left_str.strip())
        right = parse_expr(right_str.strip())
        # Check if left - right simplifies to 0
        is_true = simplify(left - right) == 0

        numbers = [int(x) for x in re.findall(r"\d+", left_str)]
        result = [int(re.findall(r"\d+", right_str)[0])]

        if sorted(numbers) != sorted(reference["nums"]):
            return 0.0
        if result[0] != reference["target"]:
            return 0.0

        if is_true:
            return 1.0
        else:
            return 0.0
    except Exception:
        # logger.warning(f"Error in custom_reward_fn: {e}")
        return 0.0


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
    logger.info(f"Final training dataset size = {len(train_dataset)}")

    dataset = util.load_data_from_disk_or_hf(
        config.validation.dataset.name,
        config.validation.dataset.subset,
        config.validation.dataset.revision or None,
    )
    dataset_list = []
    for split_name in config.validation.dataset.split:
        print(
            f"Appending split {split_name}, dataset size = {len(dataset[split_name])}"
        )
        dataset_list.append(dataset[split_name])
    test_dataset = concatenate_datasets(dataset_list)
    logger.info(f"Final validation dataset size = {len(test_dataset)}")

    launch_dispatcher(
        dataset=CountdownDataset(dataset=train_dataset),
        val_dataset=CountdownValDataset(dataset=test_dataset),
        val_reward_fns=[custom_reward_fn],
        val_data_packer=MathDataPacker(),
    )
