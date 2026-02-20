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
PYTHONPATH=path_to/imaginaire4:$PYTHONPATH cosmos-rl --config nemotron_based_vlm.toml i4data_launcher.py
"""

import logging

# Set for cosmos logging to not propagate to root logger to make sure the log level is correct
logging.getLogger("cosmos").propagate = False

import importlib
import numpy as np
import torch
from projects.cosmos3.vlm.datasets.utils import DataSource
from typing import Any, Dict, List, Optional
from cosmos_rl.dispatcher.data.packer.base import DataPacker
from cosmos_rl.launcher.worker_entry import main as launch_worker
from cosmos_rl.policy.config import Config
from cosmos_rl.policy.config import Config as CosmosConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os

os.environ["TP_EP_INTERCHANGABLE_WITH_DP_FUSED"] = "1"

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


def patch_imaginaire_s3_object_store():
    def object_store_to_bucket_and_credentials(object_store: str):
        if object_store == "s3":
            return S3_BUCKET_NAME, S3_CREDENTIALS_PATH
        else:
            raise ValueError(f"Object store {object_store} not supported")

    import projects.cosmos3.vlm.datasets.dataset_provider_sft as dataset_provider_sft

    dataset_provider_sft.object_store_to_bucket_and_credentials = (
        object_store_to_bucket_and_credentials
    )


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
        image_grid_thw = batch["image_grid_thw"]
        num_tokens = image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]
        assert num_tokens.sum().item() == batch["pixel_values"].size(0)
        f"Token mismatch: {num_tokens.sum()} vs {batch['pixel_values'].size(0)}"
        pixel_splits = torch.split(batch["pixel_values"], num_tokens.tolist(), dim=0)
        raw_image_splits = torch.split(batch["raw_image"], 1, dim=2)
        sliced_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) or isinstance(value, np.ndarray):
                # logger.info(f"====== Slicing pixel_values for key: {key}, shape: {value.shape}")
                if key == "pixel_values":

                    sliced_batch[key] = torch.cat(
                        pixel_splits[start:end], dim=0
                    )
                elif key == "raw_image":
                    sliced_batch[key] = torch.cat(
                        raw_image_splits[start:end], dim=2
                    )
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
                pool_size=16,
                max_batch_size=config.train.train_batch_per_replica,
                max_tokens=config.policy.model_max_length * config.train.train_batch_per_replica * 8, # Ensure all train_batch_per_replica samples can fit in total max_tokens
                model_name_or_path=config.policy.model_name_or_path, #for flop_based_batching control but `trust_remote_code` is False here which may cause issue
                long_threshold=config.policy.model_max_length * 8, # Ensure all samples are within model_max_length to reach train_batch_per_replica in a returned batch
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

class CustomSFTDataPacker(I4WebDatasetDataPacker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.IGNORE_LABEL_ID = IGNORE_LABEL_ID


class CosmosSFTDataset(Dataset):
    def setup(self, config: Config, tokenizer: AutoTokenizer = None, *args, **kwargs):
        """
        Called by launcher after being mounted
        """
        from imaginaire.lazy_config import instantiate
        from imaginaire.utils.config_helper import get_config_module, override

        self.config = config
        self.tokenizer = tokenizer

        # Instantiate i4 model
        config_file = "projects/cosmos3/vlm/configs/base/config.py"
        config_module = get_config_module(config_file)
        config = importlib.import_module(config_module).make_config()

        # Register custom dataset
        data_sets = {
            # name : <module_path>.<`data_weight_default` attribute of the module> (example as module of this .py file)
            "sample_dataset": "__main__.data_weight_default",
        }
        register_data_set(data_sets, self.config)
        config = override(
            config,
            [
                "--",
                "data_train=sample_dataset",
                # Specify qwen3_vl model in order to use its processor to process from raw data to model input
                "model=qwen3_vl_8b_instruct_cosmos_rl",
                "checkpoint=s3",
            ],
        )
        # Setting up S3 object store patching with bucket and credentials
        patch_imaginaire_s3_object_store()
        self.data_loader = instantiate(config.dataloader_train)

    def __len__(self):
        return len(self.data_loader)

if __name__ == "__main__":
    # Do some monkey patching to support Nemotron-3-Nano Vision-Language Model parallelization.
    import cosmos_rl.policy.model.hf_models
    from launcher import patched_parallelize_fn, convert_weight_from_hf, step_hook, policy_map_local_key_for_export_tensor
    # Override the parallelize_fn to support EP parallelization.
    cosmos_rl.policy.model.hf_models.HFModel.parallelize_fn = property(patched_parallelize_fn)
    # Override the convert_weight_from_hf to support EP weight sharding during initialization
    cosmos_rl.policy.model.hf_models.convert_weight_from_hf = convert_weight_from_hf
    # Override the step_hook to enable aux-free load balancing update bias after each step update.
    cosmos_rl.policy.model.hf_models.HFModel.step_hook = step_hook
    # Map the weight name from custom DeepEP convention back to HF convention for safetensor saving.
    cosmos_rl.policy.model.hf_models.weight_mapper.HFModelWeightMapper.policy_map_local_key_for_export_tensor = policy_map_local_key_for_export_tensor
    
    

    def get_dataset(config: CosmosConfig) -> Dataset:
        return CosmosSFTDataset()

    # It is best practice to pass the dataset as a factory function
    # so that the dataset can be loaded on demand. (Not all workers need it)
    launch_worker(
        dataset=get_dataset,
        data_packer=CustomSFTDataPacker(),
    )
