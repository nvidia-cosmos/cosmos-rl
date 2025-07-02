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
from typing import Tuple, Dict, Any
from cosmos_rl.utils.parallelism_registry import register_parallelism_strategy


def convert_weight_from_hf(
    tensor: torch.Tensor,
    name: str,
    src_model_type: str,
    parallel_dims: ParallelDims,
    ignore_unknown_weights: bool = False,
) -> Tuple[str, torch.Tensor]:
    if parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled:
        dp_shard_rank = parallel_dims.mesh[tuple(("dp_shard_cp",))].get_local_rank()
        dp_shard_size = parallel_dims.mesh[tuple(("dp_shard_cp",))].size()
    else:
        dp_shard_rank = 0
        dp_shard_size = 1

    dest_name = name
    shard = tensor
    # Do FSDP sharding
    shard = shard.contiguous()
    shard = shard.tensor_split(dp_shard_size, dim=0)[dp_shard_rank]
    return dest_name, shard.contiguous()


@register_parallelism_strategy("hfllm")
def map_weight_parallel_dims(
    n_dim: int, dest_name: str, parallel_dims: ParallelDims, model_config: Any
) -> Tuple[Dict[str, int], Dict[int, list], int]:
    pp_rank = 0
    dp_shard_size = parallel_dims.dp_shard * parallel_dims.cp

    assert dest_name.startswith("model.") or dest_name.startswith("lm_head.")
    dims_map = {}
    # Do FSDP sharding
    dim = "dp_shard_cp"
    if dp_shard_size > 1:
        dims_map[dim] = 0
    else:
        pass

    tensor_dim_to_parallel_map = {}
    for k, v in dims_map.items():
        if v not in tensor_dim_to_parallel_map:
            tensor_dim_to_parallel_map[v] = []
        tensor_dim_to_parallel_map[v].append(k)

    return dims_map, tensor_dim_to_parallel_map, pp_rank
