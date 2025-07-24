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

from cosmos_rl.policy.model.gpt.weight_converter import (
    convert_weight_from_hf as gpt_weight_from_hf,
)
from cosmos_rl.utils.parallelism import ParallelDims
import torch
from typing import Tuple


def map_key_from_hf(name: str, src_model_type: str) -> str:
    if src_model_type in ["llama", "qwen2", "qwen3", "qwen2_5_vl"]:
        return name.replace("model.", "")
    else:
        raise ValueError(f"Unsupported model type: {src_model_type}")


def convert_weight_from_hf(
    tensor: torch.Tensor,
    name: str,
    src_model_type: str,
    parallel_dims: ParallelDims,
    ignore_unknown_weights: bool = False,
) -> Tuple[str, torch.Tensor]:
    lm_part_name, lm_part_shard = gpt_weight_from_hf(
        tensor, name, src_model_type, parallel_dims, ignore_unknown_weights=True
    )
    if lm_part_name is not None:
        return lm_part_name, lm_part_shard
    assert name.startswith("visual."), f"Unsupported weight: {name}"

    if (
        parallel_dims.dp_shard_enabled
        or parallel_dims.cp_enabled
        or parallel_dims.tp_enabled
    ):
        dp_shard_rank = parallel_dims.mesh["dp_cp_tp"].get_local_rank()
        dp_shard_size = parallel_dims.mesh["dp_cp_tp"].size()
    else:
        dp_shard_rank = 0
        dp_shard_size = 1

    dest_name = name.replace("visual.", "")
    shard = tensor
    # TODO(cjx): Only FSDP sharding is supported for visual part
    shard = shard.contiguous()
    if shard.shape[0] % dp_shard_size == 0:
        shard = shard.tensor_split(dp_shard_size, dim=0)
        shard = shard[dp_shard_rank]
    else:
        chunk_size = (shard.shape[0] + dp_shard_size - 1) // dp_shard_size
        shard = shard[dp_shard_rank * chunk_size : (dp_shard_rank + 1) * chunk_size]

    return dest_name, shard.contiguous()
