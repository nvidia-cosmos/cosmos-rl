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


from typing import Any
from torch.utils.data import Dataset, ConcatDataset
from datasets import load_dataset
from cosmos_rl.launcher.worker_entry import main as launch_worker
from cosmos_rl.policy.config import Config as CosmosConfig
import cosmos_rl.utils.util as util
from cosmos_rl.utils.logging import logger


class DeepSeekV3GRPODataset(Dataset):
    def setup(self, config: CosmosConfig, *args, **kwargs):
        """
        This method is optional and get called by launcher after being mounted
        `config`: config;
        """
        self.config = config
        self.tokenizer = util.setup_tokenizer(config.policy.model_name_or_path)
        self.dataset = load_dataset(
            config.train.train_policy.dataset.name,
            config.train.train_policy.dataset.subset,
        )
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
        logger.info(f"Length of the dataset: {len(self.dataset)}")
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        """
        For DecoderOnlyLLMDataPacker, it should either return:
        - raw text prompt to be converted into input_ids by both rollout and policy models;
        - conversation format:
        ```
        [
            {
                "role": "user",
                "content": f"{question} Let\'s think step by step and output the final answer after \"####\".",
            }
        ]
        ```
        """
        assert hasattr(
            self, "tokenizer"
        ), "`self.tokenizer` should be set by the launcher"
        question = self.dataset[idx]["question"]
        assert isinstance(question, str), "Question should be a string"
        # Convert to templated prompt
        conversation = [
            {
                "role": "user",
                "content": f'{question} Let\'s think step by step and output the final answer after "####".',
            }
        ]
        prompt = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt

    def get_reference_answer(self, idx: int) -> Any:
        """
        This is mandatory for GRPO to get a reference answer for reward computation.
        """
        answer = self.dataset[idx]["answer"]
        return answer


class DeepSeekV3GRPOValDataset(DeepSeekV3GRPODataset):
    """
    This is a validation dataset for DeepSeekV3, which is used to evaluate the performance of the model.
    It should be used in the launcher to evaluate the model during training.
    """

    def setup(self, config: CosmosConfig, *args, **kwargs):
        if not config.validation.enable:
            logger.warning(
                "Validation is not enabled in the config. Skipping setup for GSM8kValDataset."
            )
            return

        self.config = config
        self.tokenizer = util.setup_tokenizer(config.policy.model_name_or_path)

        self.dataset = load_dataset(
            config.validation.dataset.name, config.validation.dataset.subset
        )
        if config.validation.dataset.split:
            if isinstance(config.validation.dataset.split, list):
                dataset_list = []
                for split_name in config.validation.dataset.split:
                    dataset_list.append(self.dataset[split_name])
                self.dataset = ConcatDataset(dataset_list)
            else:
                assert isinstance(config.validation.dataset.split, str)
                self.dataset = self.dataset[config.validation.dataset.split]


if __name__ == "__main__":

    def get_dataset(config: CosmosConfig) -> Dataset:
        return DeepSeekV3GRPODataset()

    def get_val_dataset(config: CosmosConfig) -> Dataset:
        return DeepSeekV3GRPOValDataset()

    launch_worker(
        dataset=get_dataset,
        val_dataset=get_val_dataset,
    )
