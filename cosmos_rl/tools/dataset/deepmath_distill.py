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
    custom_reward_fn as math_custom_reward_fn,
)
from cosmos_rl.utils.logging import logger
from torch.utils.data import Dataset
from datasets import concatenate_datasets
import cosmos_rl.utils.util as util
from cosmos_rl.launcher.worker_entry import main as launch_dispatcher
from cosmos_rl.policy.config import Config


class DeepMathDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def setup(
        self,
        config: Config,
    ):
        self.config = config

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        converted_item = item["question"]
        return converted_item


def custom_reward_fn(
    to_be_evaluated: str, reference: Optional[Any] = None, *args, **kwargs
) -> float:
    # Always return 0.0 reward for Distillation tasks
    return 0.0


class AIMEDataSet(Dataset):
    def setup(
        self,
        config: Config,
    ):
        self.config = config
        logger.info(
            f"Loading AIME validation dataset... {config.validation.dataset.name}"
        )
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
        self.dataset = concatenate_datasets(dataset_list)
        self.tokenizer = util.setup_tokenizer(config.policy.model_name_or_path)
        logger.info(f"Final AIME validation dataset size = {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> RLPayload:
        question = self.dataset[idx]["problem"]
        assert isinstance(
            question, str
        ), f"Prompt should be a string, but got {type(question)}, {question}"
        # Convert to templated prompt
        conversation = [
            {
                "role": "user",
                "content": question,
            }
        ]
        prompt = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        return RLPayload(prompt=prompt)

    def get_reference_answer(self, idx: int) -> Any:
        """
        This is mandatory for GRPO to get a reference answer for reward computation.
        """
        response = self.dataset[idx]["answer"]
        if "boxed" not in response:
            response = "$\\boxed{" + response + "}$"
        return response


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

    launch_dispatcher(
        dataset=DeepMathDataset(dataset=train_dataset),
        val_dataset=AIMEDataSet(),
        reward_fns=[custom_reward_fn],
        val_reward_fns=[math_custom_reward_fn],
        val_data_packer=MathDataPacker(),
    )
