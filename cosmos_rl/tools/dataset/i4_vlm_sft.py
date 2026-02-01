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


# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

"""
Usage:
PYTHONPATH=path_to/imaginaire4:$PYTHONPATH cosmos-rl --config configs/qwen3/qwen3-vl-custom.toml tools/dataset/i4_vlm_sft.py
"""

import logging

# Set for cosmos logging to not propagate to root logger to make sure the log level is correct
logging.getLogger("cosmos").propagate = False

import importlib
import numpy as np
import torch
from imaginaire.lazy_config import instantiate
from imaginaire.utils.config_helper import get_config_module, override
from typing import Any, Dict, List, Optional
from cosmos_rl.dispatcher.data.packer.base import DataPacker
from cosmos_rl.launcher.worker_entry import main as launch_worker
from cosmos_rl.policy.config import Config
from cosmos_rl.policy.config import Config as CosmosConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer


MAX_PIXELS = 81920
IGNORE_LABEL_ID = -100


# # Instantiate i4 model
# config_file = "projects/cosmos3/vlm/configs/base/config.py"
# config_module = get_config_module(config_file)
# config = importlib.import_module(config_module).make_config()
# experiment = "pre_exp020_000_qwen3_vl_30b_a3b_thinking"
# config = override(
#     config,
#     [
#         "--",
#         f"experiment={experiment}",
#         # "data_train=09_eagle_sft_full_mul_repeat_debug_s3",
#         # "data_train=debug_image_data_qwen",
#     ],
# )
# self.dataloader = instantiate(config.dataloader_train)


class I4WebDatasetDataPacker(DataPacker):
    """
    Data protocol & processing logic for i4 web dataset.
    """

    def setup(self, config: Config, tokenizer: AutoTokenizer, *args, **kwargs):
        super().setup(config, tokenizer, *args, **kwargs)

    def get_rollout_input(self, sample: Dict[str, Any]) -> Any:
        raise NotImplementedError(
            "Get rollout input is not implemented for SFT data packer"
        )

    def get_policy_input(
        self,
        sample: Dict[str, Any],
        rollout_output: Optional[str] = None,
        n_ignore_prefix_tokens: int = 0,
        add_generation_prompt: bool = True,
    ) -> Any:
        raise NotImplementedError(
            "Get policy input is not implemented for SFT data packer"
        )

    def policy_compute_max_len(self, processed_samples: List[Dict[str, Any]]) -> int:
        raise NotImplementedError(
            "Policy compute max len is not implemented for SFT data packer"
        )

    def policy_collate_fn(
        self, processed_samples: List[Dict[str, Any]], computed_max_len: int
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "Policy collate fn is not implemented for SFT data packer"
        )

    def sft_process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accepts directly the raw sample from webdataset which is already dict type.
        """
        return sample

    def sft_compute_max_len(self, processed_samples: Dict[str, Any]) -> int:
        """
        Compute the maximum sequence length in the batch.
        """
        return processed_samples["input_ids"].shape[1]

    def sft_collate_fn(
        self,
        processed_samples: Dict[str, Any],
        computed_max_len: int,
        ignore_label_id: int,
    ) -> Dict[str, Any]:
        """
        Collate the processed samples into model inputs. The raw samples have been already processed.
        """
        model_inputs: Dict[str, Any] = processed_samples
        model_inputs["label_ids"] = processed_samples["labels"]
        del model_inputs["labels"]
        return model_inputs

    def batch_size(self, batch: Dict[str, Any]) -> int:
        """
        Compute the batch size.
        """
        return batch["input_ids"].size(0)

    def slice_batch(
        self,
        batch: Dict[str, Any],
        start: int,
        end: int,
    ) -> Dict[str, Any]:
        """
        Slice the batch from start to end.
        """
        sliced_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) or isinstance(value, np.ndarray):
                # logger.info(f"====== Slicing tensor/ndarray for key: {key}, shape: {value.shape}")
                sliced_batch[key] = value[start:end]
            elif isinstance(value, list):
                # logger.info(f"====== Slicing list for key: {key}, length: {len(value)}")
                sliced_batch[key] = value[start:end]
            else:
                # logger.info(f"====== Keeping original value for key: {key}, type: {type(value)}")
                sliced_batch[key] = value
        return sliced_batch


class CustomSFTDataPacker(I4WebDatasetDataPacker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.IGNORE_LABEL_ID = IGNORE_LABEL_ID


class CosmosSFTDataset(Dataset):
    def setup(self, config: Config, tokenizer: AutoTokenizer = None, *args, **kwargs):
        """
        Called by launcher after being mounted
        """
        self.config = config
        self.tokenizer = tokenizer

        # Instantiate i4 model
        config_file = "projects/cosmos3/vlm/configs/base/config.py"
        config_module = get_config_module(config_file)
        config = importlib.import_module(config_module).make_config()
        experiment = "pre_exp020_000_qwen3_vl_30b_a3b_thinking"
        config = override(
            config,
            [
                "--",
                f"experiment={experiment}",
                # "data_train=09_eagle_sft_full_mul_repeat_debug_s3",
                # "data_train=debug_image_data_qwen",
            ],
        )
        self.dataloader = instantiate(config.dataloader_train)
        self.iterator = iter(self.dataloader)
        self.data_loader = self.dataloader

    def __len__(self):
        return len(self.dataloader)


if __name__ == "__main__":
    # mp.set_start_method('spawn', force=True)
    def get_dataset(config: CosmosConfig) -> Dataset:
        return CosmosSFTDataset()

    # It is best practice to pass the dataset as a factory function
    # so that the dataset can be loaded on demand. (Not all workers need it)
    launch_worker(
        dataset=get_dataset,
        data_packer=CustomSFTDataPacker(),
    )
