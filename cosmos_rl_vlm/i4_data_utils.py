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
import logging
import sys

# Set for cosmos logging to not propagate to root logger to make sure the log level is correct
logging.getLogger("cosmos").propagate = False

import importlib
import numpy as np
import torch
from projects.cosmos3.vlm.datasets.utils import DataSource
from typing import Any, Dict, List, Optional, Union
from cosmos_rl.dispatcher.data.packer.base import DataPacker
from cosmos_rl.policy.config import Config
from torch.utils.data import Dataset
from transformers import AutoTokenizer


IGNORE_LABEL_ID = -100


S3_BUCKET_NAME = "nv-cosmos-zu-videos"
S3_CREDENTIALS_PATH = "credentials/s3_training.secret"


"""
Example data config for i4 web dataset sources.
Allow to combine multiple data sources with different weights.
The following code defines a dataset configuration for using i4 web dataset format:
DATAINFO contains DataSource instances for two datasets: 'robospatial_llava_qa' and 'pixmo_pointing_fix'.
Each with their respective total key counts, paths to their wdinfo files, text and media keys, and a flag indicating they are not text-only datasets.
The data_weight_default dictionary assigns weights to these datasets for training purposes.
The url_to_category function maps dataset URLs to their corresponding categories based on specific substrings in the URL.
These three components are required for instantiating the dataset and must be attributes of an importable module file such as this .py file.
User can specify which datasets to include using DATAINFO and assign weights using data_weight_default to compose the training dataset.
"""
DATAINFO = {
    # robospatial_llava_qa: using v1_1 from webdataset_interleaved
    "robospatial_llava_qa": DataSource(
        total_key_count=3075590,
        wdinfo_path={
            "train": "cosmos_reason2/grounding_2d/v1_1/wdinfo/robospatial_llava_qa/wdinfo.json",
        },
        text_keys=["texts"],
        media_keys=["media"],
        text_only=False,
    ),
    # pixmo_pointing_fix: using v1_1 from webdataset_interleaved
    "pixmo_pointing_fix": DataSource(
        total_key_count=1834800,
        wdinfo_path={
            "train": "cosmos_reason2/grounding_2d/v1_1/wdinfo/pixmo_pointing_fix/wdinfo.json",
        },
        text_keys=["texts"],
        media_keys=["media"],
        text_only=False,
    ),
}

data_weight_default = {
    "robospatial_llava_qa": 3075590,
    "pixmo_pointing_fix": 1834800,
}


def url_to_category(url: str) -> str | None:
    # Map the tar url to the category in the data_weight_dict
    if "eagle_sft/" in url:
        # cosmos_reason2/eagle_sft/GroundUI/v0/
        return url.split("eagle_sft/", 1)[1].split("/")[0]
    elif "grounding_2d/" in url:
        # cosmos_reason2/grounding_2d/v1_1/gqa_s_ext/
        return url.split("grounding_2d/", 1)[1].split("/")[1]
    else:
        return None


def patch_imaginaire_s3_object_store(
    s3_bucket: str = S3_BUCKET_NAME, s3_credentials_path: str = S3_CREDENTIALS_PATH
):
    def object_store_to_bucket_and_credentials(object_store: str):
        if object_store == "s3":
            return s3_bucket, s3_credentials_path
        else:
            raise ValueError(f"Object store {object_store} not supported")

    import projects.cosmos3.vlm.datasets.dataset_provider_sft as dataset_provider_sft

    dataset_provider_sft.object_store_to_bucket_and_credentials = (
        object_store_to_bucket_and_credentials
    )

def patch_joint_dataset():
    from projects.cosmos3.vlm.datasets.joint_dataset_dynamic_batch_webloader import _JointIterableDataset
    def _fill_pool(self):
        if len(self._pool) == 0:
            # Fill the pool with the first sample to determine the seed modality
            self._pool.append(self._get_next_sample())
        seed = self._pool[0]
        seed_modality = self._get_modality(seed)
        same_seed_cnt = 0
        while len(self._pool) < self.pool_size or same_seed_cnt < self.max_batch_size:
            item = self._get_next_sample()
            if self._get_modality(item) == seed_modality:
                same_seed_cnt += 1
            self._pool.append(item)

    _JointIterableDataset._fill_pool = _fill_pool


class I4WebDatasetDataPacker(DataPacker):
    """
    Data protocol & processing logic for i4 web dataset.
    """
    def __init__(self, dataloader, *args, **kwargs):
        self.data_iter = iter(dataloader)

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

    def batch_size(self, batch: Union[List[Dict[str, Any]], Dict[str, Any]]) -> int:
        """
        Compute the batch size. Fake to match the number of minibatch calculation in SFT trainer of cosmos-rl.
        """
        return self.config.train.train_batch_per_replica

    def slice_batch(
        self,
        batch: Union[List[Dict[str, Any]], Dict[str, Any]],
        start: int,
        end: int,
    ) ->  Dict[str, Any]:
        """
        Slice the batch from start to end. Directly index the batch list based on the number of minibatch
        """
        assert self.config.train.train_batch_per_replica % self.config.train.train_policy.mini_batch == 0, f"train_batch_per_replica {self.config.train.train_batch_per_replica} should be divisible by mini_batch {self.config.train.train_policy.mini_batch}"
        if isinstance(batch, list):
            assert len(batch) == self.config.train.train_batch_per_replica // self.config.train.train_policy.mini_batch, f"Batch size {len(batch)} does not match expected batch size {self.config.train.train_batch_per_replica // self.config.train.train_policy.mini_batch}"
            assert start % self.config.train.train_policy.mini_batch == 0, f"Start {start} should be divisible by mini_batch {self.config.train.train_policy.mini_batch}"
            offset = start // self.config.train.train_policy.mini_batch
            assert offset < len(batch), f"Start {start} is out of range for batch size {len(batch) * self.config.train.train_policy.mini_batch}"
        data = next(self.data_iter)  # The real slicing logic is handled by the JointDatasetDynamicBatchingWebLoader in the patch_joint_dataset function, here we just return the next batch from the data loader
        if "raw_image" in data:
            del data["raw_image"]  # Remove raw_image to save memory, the real image data is in pixel_values and image_grid_thw
        if "raw_video" in data:
            del data["raw_video"]  # Remove raw_video to save memory, the real video data is in pixel_values_videos and video_grid_thw
        if "token_mask" in data:
            del data["token_mask"]  # Remove token_mask to save memory, the real token mask is in attention_mask
        if "attention_mask" in data:
            del data["attention_mask"]  # Remove attention_mask to save memory, the real attention mask is in labels
        # from cosmos_rl.utils.logging import logger
        # logger.info(f"keys in the batch: {list(data.keys())}")
        return data

    def _slice_batch_from_collated(
        self,
        batch: Dict[str, Any],
        start: int,
        end: int,
    ) ->  Dict[str, Any]:
        # from cosmos_rl.utils.logging import logger
        # for batch_key, batch_value in batch.items():
        #     if isinstance(batch_value, torch.Tensor):
        #         logger.info(f"====== Slicing batch from {start} to {end}, batch size: {batch_key} {batch_value.shape}")
        #     elif isinstance(batch_value, list):
        #         logger.info(f"====== Slicing batch from {start} to {end}, batch size: {batch_key} {len(batch_value)}")
        #     else:
        #         logger.info(f"====== Slicing batch from {start} to {end}, batch key: {batch_key} value type: {batch_value}")
        sliced_batch = {}
        # Handle image data with pixel_values (flattened format)
        if "pixel_values" in batch and "image_grid_thw" in batch:
            image_grid_thw = batch["image_grid_thw"]
            num_tokens = image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]
            assert num_tokens.sum().item() == batch["pixel_values"].size(0)
            f"Token mismatch: {num_tokens.sum()} vs {batch['pixel_values'].size(0)}"
            pixel_splits = torch.split(batch["pixel_values"], num_tokens.tolist(), dim=0)
            # image_grid_thw might be per-image, need to slice based on lengths
            start_idx = start
            end_idx = end
            if "pixel_values_lengths_per_sample" in batch:
                lengths = batch["pixel_values_lengths_per_sample"][start:end]
                # Calculate cumulative indices for slicing image_grid_thw
                if start == 0:
                    start_idx = 0
                else:
                    prev_lengths = batch["pixel_values_lengths_per_sample"][:start]
                    start_idx = prev_lengths.sum().item()
                end_idx = start_idx + lengths.sum().item()
                sliced_batch["pixel_values_lengths_per_sample"] = lengths
                # Fallback: assume one image per sample
            sliced_batch["image_grid_thw"] = batch["image_grid_thw"][start_idx:end_idx]
            sliced_batch["pixel_values"] = torch.cat(pixel_splits[start_idx:end_idx], dim=0)

        # Handle video data
        if "pixel_values_videos" in batch and "video_grid_thw" in batch:
            video_grid_thw = batch["video_grid_thw"]
            num_tokens = video_grid_thw[:, 0] * video_grid_thw[:, 1] * video_grid_thw[:, 2]
            assert num_tokens.sum().item() == batch["pixel_values_videos"].size(0), f"Token mismatch: {num_tokens.sum()} vs {batch['pixel_values_videos'].size(0)}"
            pixel_splits = torch.split(batch["pixel_values_videos"], num_tokens.tolist(), dim=0)
            start_idx = start
            end_idx = end
            if "pixel_values_videos_lengths_per_sample" in batch:
                lengths = batch["pixel_values_videos_lengths_per_sample"][start:end]
                if start == 0:
                    start_idx = 0
                else:
                    prev_lengths = batch["pixel_values_videos_lengths_per_sample"][:start]
                    start_idx = prev_lengths.sum().item()
                end_idx = start_idx + lengths.sum().item()
                sliced_batch["pixel_values_videos_lengths_per_sample"] = lengths
            sliced_batch["video_grid_thw"] = batch["video_grid_thw"][start_idx:end_idx]
            if "second_per_grid_ts" in batch:
                sliced_batch["second_per_grid_ts"] = batch["second_per_grid_ts"][start_idx:end_idx]
            sliced_batch["pixel_values_videos"] = torch.cat(pixel_splits[start_idx:end_idx], dim=0)

        for key, value in batch.items():
            if key in sliced_batch:
                # Already sliced above for video data, just need to copy to sliced_batch
                continue
            elif isinstance(value, torch.Tensor) or isinstance(value, np.ndarray):
                # logger.info(f"====== Slicing pixel_values for key: {key}, shape: {value.shape}")
                if key == "raw_image":
                    raw_image_splits = torch.split(value, 1, dim=2)
                    sliced_batch[key] = torch.cat(raw_image_splits[start:end], dim=2)
                else:
                    sliced_batch[key] = value[start:end]
            elif isinstance(value, list):
                sliced_batch[key] = value[start:end]
            else:
                sliced_batch[key] = value
        return sliced_batch


def register_data_set(data_modules: Dict[str, str], config: Config):
    from hydra.core.config_store import ConfigStore
    from imaginaire.lazy_config import LazyCall as L
    from projects.cosmos3.vlm.datasets.collate_fn import custom_collate
    from projects.cosmos3.vlm.datasets.joint_dataset_dynamic_batch_webloader import (
        JointDatasetDynamicBatchingWebLoader,
    )
    from projects.cosmos3.vlm.configs.base.defaults.dataloader_weighted_url import (
        create_dataloader_config,
    )

    cs = ConfigStore.instance()
    object_store = "s3"
    for dataset_name, data_module in data_modules.items():
        cs.store(
            group="data_train",
            package="dataloader_train",
            name=dataset_name,
            node=L(JointDatasetDynamicBatchingWebLoader)(
                datasets_cfg={
                    "default": {
                        "dataset": create_dataloader_config(
                            data_module, "train", object_store
                        ),
                        "ratio": 1,
                    }
                },
                # Arguments for the joint dataset
                pool_size=max(config.train.train_policy.mini_batch, 16),
                max_batch_size=config.train.train_policy.mini_batch,
                max_tokens=sys.maxsize,  # Ensure all max_batch_size samples can fit in total max_tokens, later will incorporate load balancing feature in cosmos-rl to enable real max_tokens control.
                model_name_or_path=config.policy.model_name_or_path,  # for flop_based_batching control but `trust_remote_code` is False here which may cause issue
                long_threshold=sys.maxsize,  # Ensure all samples are within long_threshold to reach max_batch_size in the returned batch, later will incorporate load balancing feature in cosmos-rl to enable real long_threshold control.
                length_key="input_ids",
                batching_strategy="prefer_closest",
                # Arguments for the webloader
                batch_size=config.train.train_policy.dataloader_batch_size,  # This is not the real batch size, it wont be used
                num_workers=config.train.train_policy.dataloader_num_workers,
                sampler=None,
                prefetch_factor=config.train.train_policy.dataloader_prefetch_factor,
                persistent_workers=False,
                pin_memory=True,
                collate_fn=custom_collate,
            ),
        )

class DummyDataLoader():
    def __init__(self, dataset_size, *args, **kwargs):
        self.dataset_size = dataset_size

    def __iter__(self):
        # Get the standard iterator from the parent class
        while True:
            yield {"to_be_filled": True}

    def __len__(self):
        return self.dataset_size

def GetI4DataLoader(cosmos_config: Config):
    from imaginaire.lazy_config import instantiate
    from imaginaire.utils.config_helper import get_config_module, override
    # Instantiate i4 model
    config_file = "projects/cosmos3/vlm/configs/base/config.py"
    config_module = get_config_module(config_file)
    config = importlib.import_module(config_module).make_config()

    i4_data_source = cosmos_config.custom.get(
        "i4_data_source", "i4_data_utils.data_weight_default"
    )
    # Register custom dataset
    data_sets = {
        # name : <module_path>.<`data_weight_default` attribute of the module> (example as module of this .py file)
        "cosmos_rl_training_dataset": i4_data_source,
    }
    register_data_set(data_sets, cosmos_config)
    config = override(
        config,
        [
            "--",
            "data_train=cosmos_rl_training_dataset",
            # "checkpoint=s3",
            f"policy.model_name_or_path={cosmos_config.policy.model_name_or_path}",
        ],
    )
    # Setting up S3 object store patching with bucket and credentials
    patch_imaginaire_s3_object_store(
        s3_bucket=cosmos_config.custom.get("s3_bucket", S3_BUCKET_NAME),
        s3_credentials_path=cosmos_config.custom.get(
            "s3_credentials_path", S3_CREDENTIALS_PATH
        ),
    )
    return instantiate(config.dataloader_train)


class CosmosSFTI4DataPacker(I4WebDatasetDataPacker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.IGNORE_LABEL_ID = IGNORE_LABEL_ID


class CosmosSFTI4Dataset(Dataset):
    def __init__(self, dataset_size, *args, **kwargs):
        self.dataset_size = dataset_size

    def setup(self, config: Config, tokenizer: AutoTokenizer = None, *args, **kwargs):
        """
        Called by launcher after being mounted
        """
        self.config = config
        self.tokenizer = tokenizer
        self.data_loader = DummyDataLoader(self.dataset_size)  # Placeholder, the real data loader is handled by the JointDatasetDynamicBatchingWebLoader in the patch_joint_dataset function

    def __len__(self):
        return self.dataset_size


def patch_processor_in_i4(is_nemotron_vl_model: bool = False):
    from projects.cosmos3.vlm.processors.qwen3vl_processor import Qwen3VLProcessor

    if is_nemotron_vl_model:
        original_apply_chat_template = Qwen3VLProcessor.apply_chat_template

        def apply_chat_template(
            self,
            messages,
            add_generation_prompt=False,
            return_tensors="pt",
            tokenize=True,
            **kwargs,
        ):
            messages = [msg for msg in messages if msg.get("role") != "system"]
            return original_apply_chat_template(
                self,
                messages,
                add_generation_prompt=add_generation_prompt,
                return_tensors=return_tensors,
                tokenize=tokenize,
                **kwargs,
            )

        Qwen3VLProcessor.apply_chat_template = apply_chat_template

    def build_processor(
        tokenizer_type: str,
        cache_dir: Optional[str] = None,
        credentials: str = "./credentials/s3_training.secret",
        bucket: str = "checkpoints-us-east-1",
    ):
        processor = Qwen3VLProcessor(
            tokenizer_type, credentials=credentials, bucket=bucket, cache_dir=cache_dir
        )
        if processor.pad_id is None:
            processor.pad_id = processor.eos_id
        return processor

    import projects.cosmos3.vlm.processors

    projects.cosmos3.vlm.processors.build_processor = build_processor
