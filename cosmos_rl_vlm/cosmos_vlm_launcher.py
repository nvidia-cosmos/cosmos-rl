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
PYTHONPATH=path_to/imaginaire4:$PYTHONPATH cosmos-rl --config qwen3-vl-custom.toml cosmos_vlm_launcher.py
"""

import argparse
import logging

# Set for cosmos logging to not propagate to root logger to make sure the log level is correct
logging.getLogger("cosmos").propagate = False

from cosmos_rl.launcher.worker_entry import main as launch_worker
from cosmos_rl.policy.config import Config
from cosmos_rl.policy.config import Config as CosmosConfig
from torch.utils.data import Dataset
import toml
from i4_data_utils import (
    CosmosSFTI4Dataset,
    CosmosSFTI4DataPacker,
    patch_processor_in_i4,
    GetI4DataLoader,
)


def patch_cosmos_rl_policy_model(is_nemotron_vl_model: bool = False):
    if is_nemotron_vl_model:
        # Do some monkey patching to support Nemotron-3-Nano Vision-Language Model parallelization.
        import cosmos_rl.policy.model.hf_models
        from nemotron_based_training.launcher import (
            patched_parallelize_fn,
            convert_weight_from_hf,
            step_hook,
            policy_map_local_key_for_export_tensor,
        )

        # Override the parallelize_fn to support EP parallelization.
        cosmos_rl.policy.model.hf_models.HFModel.parallelize_fn = property(
            patched_parallelize_fn
        )
        # Override the convert_weight_from_hf to support EP weight sharding during initialization
        cosmos_rl.policy.model.hf_models.convert_weight_from_hf = convert_weight_from_hf
        # Override the step_hook to enable aux-free load balancing update bias after each step update.
        cosmos_rl.policy.model.hf_models.HFModel.step_hook = step_hook
        # Map the weight name from custom DeepEP convention back to HF convention for safetensor saving.
        cosmos_rl.policy.model.hf_models.weight_mapper.HFModelWeightMapper.policy_map_local_key_for_export_tensor = policy_map_local_key_for_export_tensor

        import os

        os.environ["TP_EP_INTERCHANGABLE_WITH_DP_FUSED"] = "1"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_known_args()[0]
    with open(args.config, "r") as f:
        config = toml.load(f)
    config = Config.from_dict(config)
    # Decide if the model is a Nemotron VL model based on the model name, and patch the processor and policy model accordingly.
    # Otherwise is treated as a Qwen3VL-like model.
    is_nemotron_vl_model = "nemotron" in config.policy.model_name_or_path.lower()
    is_i4_data = "use_i4" in config.custom and config.custom["use_i4"]
    if is_i4_data:
        patch_processor_in_i4(is_nemotron_vl_model)
    patch_cosmos_rl_policy_model(is_nemotron_vl_model)
    dataloader = GetI4DataLoader(config)
    dataset_size = len(dataloader)  # Get the length of the dataloader to pass to the dataset factory function

    def get_dataset(config: CosmosConfig) -> Dataset:
        return CosmosSFTI4Dataset(dataset_size=dataset_size)  # Return an instance of the dataset with the length of the dataloader

    # It is best practice to pass the dataset as a factory function
    # so that the dataset can be loaded on demand. (Not all workers need it)
    if is_i4_data:
        launch_worker(
            dataset=get_dataset,
            data_packer=CosmosSFTI4DataPacker(dataloader=dataloader),
        )
    else:
        from cosmos_rl.cosmos_rl_vlm.qwen3_based_training.launcher import (
            get_dataset as get_non_i4_dataset,
        )
        # Launch the worker
        launch_worker(
            # Uncomment this if you want to use a custom dataset
            dataset=get_non_i4_dataset,
        )
