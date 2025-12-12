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

from cosmos_rl.utils.parallelism import ParallelDims
import torch
from typing import Tuple


def convert_weight_from_hf(
    tensor: torch.Tensor,
    name: str,
    parallel_dims: ParallelDims,
) -> Tuple[str, torch.Tensor]:
    """
    Convert a weight tensor from HuggingFace checkpoint format to cosmos-rl format.
    
    For VLA models, we apply FSDP sharding along dimension 0 to all parameters.
    This function shards the full checkpoint tensor to match the local shard size
    on this GPU rank.
    
    Args:
        tensor: Full weight tensor from checkpoint
        name: Parameter name
        parallel_dims: Parallelism configuration
        
    Returns:
        Tuple of (parameter_name, sharded_tensor)
    """
    # Get FSDP sharding info
    if parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled:
        dp_shard_rank = parallel_dims.mesh[tuple(("dp_shard_cp",))].get_local_rank()
        dp_shard_size = parallel_dims.mesh[tuple(("dp_shard_cp",))].size()
    else:
        dp_shard_rank = 0
        dp_shard_size = 1

    dest_name = name
    shard = tensor.contiguous()
    
    # Apply FSDP sharding along dimension 0
    if dp_shard_size > 1:
        row_size = shard.shape[0]
        
        if row_size % dp_shard_size == 0:
            # Even split: each rank gets equal share
            shard = shard.tensor_split(dp_shard_size, dim=0)[dp_shard_rank]
        else:
            # Uneven split: calculate boundaries manually
            average_row_size = (row_size + dp_shard_size - 1) // dp_shard_size
            start_idx = dp_shard_rank * average_row_size
            end_idx = min(start_idx + average_row_size, row_size)
            shard = shard[start_idx:end_idx]
    
    return dest_name, shard.contiguous()
