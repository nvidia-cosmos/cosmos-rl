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

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch._utils import _get_available_device_type, _get_device_module

from cosmos_rl.policy.model.deepseek_v3.configs.model_config import TrainingConfig
from cosmos_rl.utils.parallelism import ParallelDims


def get_device_info():
    device_type = _get_available_device_type()
    if device_type is None:
        device_type = "cuda"  # default device_type: cuda
    device_module = _get_device_module(device_type)  # default device_module: torch.cuda
    return device_type, device_module


device_type, device_module = get_device_info()


def init_meshes(
    parallelism_config: TrainingConfig,
) -> tuple[dict[str, DeviceMesh], ParallelDims, torch.device]:
    """
    Initialize the meshes for the model. There are generally two meshes, "default"
    for non-MoE modules and "moe" for MoE modules. Each mesh contains several
    dimensions for parallelization.

    Args:
        parallelism_config (TrainingConfig): The parallelism configuration for the model.

    Returns:
        meshes (dict[str, DeviceMesh]): The meshes for the model.
        parallel_dims (ParallelDims): The parallel dimensions for the model.
        device (torch.device): The device for the model.
    """
    parallel_dims = ParallelDims(
        enable_loss_parallel=not parallelism_config.disable_loss_parallel,
        cp=parallelism_config.context_parallel_degree,
        ep=parallelism_config.expert_parallel_degree,
        tp=parallelism_config.tensor_parallel_degree,
        pp=parallelism_config.pipeline_parallel_degree,
        world_size=dist.get_world_size(),
        dp_replicate=parallelism_config.data_parallel_replicate_degree,
        dp_shard=parallelism_config.data_parallel_shard_degree,
        pp_dynamic_shape=False
    )

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device(f"{device_type}:{local_rank}")
    device_module.set_device(device)
    meshes = parallel_dims.build_meshes_with_ep(device_type=device_type)
    return meshes, parallel_dims, device
