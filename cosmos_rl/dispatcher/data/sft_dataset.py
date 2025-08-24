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
from typing import Any, Dict, Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import concatenate_datasets

from cosmos_rl.policy.config import SFTDataConfig
from cosmos_rl.utils import cache
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.util import config_hash, load_data_from_disk_or_hf
from cosmos_rl.dispatcher.data.packer.base import DataPacker


class SFTDataset(Dataset):
    def __init__(
        self,
        config: SFTDataConfig,
        tokenizer: AutoTokenizer,
        dataset: Dataset,
        data_packer: DataPacker,
        is_user_dataset: bool = False,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.column_name = config.conversation_column_name
        self.dataset = dataset
        self.data_packer = data_packer
        self.is_user_dataset = is_user_dataset
        self.cache = None
        if self.config.enable_dataset_cache:
            # TODO(zjx): can we reuse the cache between different training jobs?
            # It's not stable yet, we only checked if the config is the same
            # If there are any problems, it is recommended that the user clears the cache folder
            cache_folder = os.path.join(
                os.environ.get(
                    "COSMOS_CACHE",
                    os.path.join(os.path.expanduser("~"), ".cache/cosmos/"),
                ),
                "datasets_cache",
                f"{self.config.dataset.name}-{config_hash(config)}",
            )
            logger.info(f"[SFTTrainer] SFTDataset Cache folder: {cache_folder}")
            self.cache = cache.DiskCache(cache_folder)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # we only cache on_the_fly result
        if self.cache is not None:
            cache_obj = self.cache.get(idx)
            if cache_obj is not None:
                return cache_obj

        raw_item = (
            self.dataset[idx][self.column_name]
            if not self.is_user_dataset and self.column_name
            else self.dataset[idx]
        )

        item: Dict[str, Any] = self.data_packer.sft_process_sample(raw_item)

        if self.cache is not None:
            # try cache obj
            self.cache.set(idx, item)
        return item


def construct_sft_dataset(
    config: SFTDataConfig,
    tokenizer: AutoTokenizer,
    data_packer: DataPacker,
    user_provided_dataset: Optional[Dataset] = None,
):
    if user_provided_dataset is not None:
        dataset = None
        train_dataset = user_provided_dataset
        logger.info(
            "[SFTTrainer] Using user-provided dataset, which will skip split processing."
        )
    else:
        dataset = load_data_from_disk_or_hf(
            config.dataset.name,
            config.dataset.subset,
            config.dataset.revision or None,
        )
        dataset_list = []
        for split_name in config.dataset.split:
            logger.info(
                f"Appending split {split_name}, dataset size = {len(dataset[split_name])}"
            )
            dataset_list.append(dataset[split_name])
        train_dataset = concatenate_datasets(dataset_list)
    logger.info(f"[SFTTrainer] Final dataset size = {len(train_dataset)}")

    # try:
    #     if dataset is not None:
    #         dataset_list = []
    #         for split_name in config.dataset.split:
    #             dataset_list.append(dataset[split_name])
    #         test_dataset = concatenate_datasets(dataset_list)
    #         if len(test_dataset) == 0:
    #             raise ValueError("Test dataset is empty")
    #     else:
    #         raise ValueError("Test dataset is empty")
    # except Exception:
    if isinstance(train_dataset, torch.utils.data.Dataset):
        # Define the split ratio (e.g., 80% train, 20% test)
        if config.dataset.test_size is None:
            logger.warning(
                "No test size specified, using 10% of the training dataset for testing."
            )
            config.dataset.test_size = 0.1
        if isinstance(config.dataset.test_size, float):
            n_test_samples = int(len(train_dataset) * config.dataset.test_size)
        else:
            n_test_samples = config.dataset.test_size
        n_test_samples = max(min(n_test_samples, len(train_dataset) - 1), 1)

        # Generate deterministic indices
        indices = list(range(len(train_dataset)))
        test_indices = indices[:n_test_samples]
        train_indices = indices[n_test_samples:]

        test_dataset = torch.utils.data.Subset(train_dataset, test_indices)
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    else:
        assert hasattr(
            train_dataset, "train_test_split"
        ), "train_dataset must have train_test_split method"
        split = train_dataset.train_test_split(
            test_size=config.dataset.test_size, shuffle=False
        )
        train_dataset = split["train"]
        test_dataset = split["test"]

    train_sft_dataset = SFTDataset(
        config,
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_packer=data_packer,
        is_user_dataset=user_provided_dataset is not None,
    )
    test_sft_dataset = SFTDataset(
        config,
        tokenizer=tokenizer,
        dataset=test_dataset,
        data_packer=data_packer,
        is_user_dataset=user_provided_dataset is not None,
    )

    return train_sft_dataset, test_sft_dataset
