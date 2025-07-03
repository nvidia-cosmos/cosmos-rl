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
import re
from typing import Tuple, Dict, Any, List
from cosmos_rl.utils.parallelism_registry import (
    register_parallelism_strategy,
    ParallelismStrategyRole,
)


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
) -> List[Tuple[str, torch.Tensor]]:
    lm_part = gpt_weight_from_hf(
        tensor, name, src_model_type, parallel_dims, ignore_unknown_weights=True
    )
    if lm_part and lm_part[0][0] is not None:
        return lm_part
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
    if "qkv" in dest_name:
        # Split qkv weight for visual
        # weight has shape [3 * head_dim, hidden_dim]
        # kv head ratio is 1, so we can split it into q, k, v
        assert shard.shape[0] % 3 == 0, "Weight shape is not compatible for splitting."
        unit_dim = shard.shape[0] // 3
        q_weight = shard[:unit_dim].contiguous()
        q_shard = q_weight.tensor_split(dp_shard_size, dim=0)[dp_shard_rank]
        k_weight = shard[unit_dim : unit_dim * 2].contiguous()
        k_shard = k_weight.tensor_split(dp_shard_size, dim=0)[dp_shard_rank]
        v_weight = shard[unit_dim * 2 :].contiguous()
        v_shard = v_weight.tensor_split(dp_shard_size, dim=0)[dp_shard_rank]
        return [
            (dest_name.replace("qkv", "q"), q_shard.contiguous()),
            (dest_name.replace("qkv", "k"), k_shard.contiguous()),
            (dest_name.replace("qkv", "v"), v_shard.contiguous()),
        ]
    shard = shard.tensor_split(dp_shard_size, dim=0)[dp_shard_rank]
    return [(dest_name, shard.contiguous())]


@register_parallelism_strategy("qwen2_5_vl", role=ParallelismStrategyRole.ROLLOUT)
def map_weight_parallel_dims(
    n_dim: int, dest_name: str, parallel_dims: ParallelDims, model_config: Any
) -> Tuple[Dict[str, int], Dict[int, list], int]:
    tp_size = parallel_dims.tp
    dp_shard_size = parallel_dims.dp_shard * parallel_dims.cp

    dims_map = {}
    dim = "tp"
    assert dest_name.startswith("visual.")
    if tp_size > 1:
        if (
            match := re.search(r"patch_embed\.proj\.(weight|bias)", dest_name)  # noqa: F841
        ) is not None:
            pass
        elif (
            match := re.search(r"merger\.ln_q\.(weight|bias)", dest_name)  # noqa: F841
        ) is not None:
            pass
        elif (
            match := re.search(r"merger\.mlp\.0\.(weight|bias)", dest_name)  # noqa: F841
        ) is not None:
            dims_map[dim] = 0
        elif (match := re.search(r"merger\.mlp\.2\.weight", dest_name)) is not None:  # noqa: F841
            dims_map[dim] = n_dim - 1
        elif (match := re.search(r"merger\.mlp\.2\.bias", dest_name)) is not None:  # noqa: F841
            pass
        elif (
            match := re.search(  # noqa: F841
                r"blocks\.(\d+)\.attn\.(q|k|v)\.(weight|bias)",
                dest_name,
            )
        ) is not None:
            dims_map[dim] = 0
        elif (
            match := re.search(r"blocks\.(\d+)\.attn\.proj\.weight", dest_name)  # noqa: F841
        ) is not None:
            dims_map[dim] = n_dim - 1
        elif (
            match := re.search(r"blocks\.(\d+)\.attn\.proj\.bias", dest_name)  # noqa: F841
        ) is not None:
            pass
        elif (
            match := re.search(  # noqa: F841
                r"blocks\.(\d+)\.mlp\.(up_proj|gate_proj)\.(weight|bias)", dest_name
            )
        ) is not None:
            dims_map[dim] = 0
        elif (
            match := re.search(r"blocks\.(\d+)\.mlp\.down_proj\.weight", dest_name)  # noqa: F841
        ) is not None:
            dims_map[dim] = n_dim - 1
        elif (
            match := re.search(r"blocks\.(\d+)\.mlp\.down_proj\.bias", dest_name)  # noqa: F841
        ) is not None:
            pass
        elif (
            match := re.search(  # noqa: F841
                r"blocks\.(\d+)\.(norm1|norm2)\.(weight|bias)", dest_name
            )
        ) is not None:
            pass
        else:
            raise ValueError(f"Unsupported weight: {dest_name}")
    else:
        pass

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
    pp_rank = 0
    return dims_map, tensor_dim_to_parallel_map, pp_rank


@register_parallelism_strategy("qwen2_5_vl", role=ParallelismStrategyRole.POLICY)
def map_weight_parallel_dims_no_visual_tp(
    n_dim: int, dest_name: str, parallel_dims: ParallelDims, model_config: Any
) -> Tuple[Dict[str, int], Dict[int, list], int]:
    tp_size = parallel_dims.tp
    dp_shard_size = parallel_dims.dp_shard * parallel_dims.cp

    dims_map = {}
    assert tp_size == 1, f"tp_size should be 1, but got {tp_size}"
    if dest_name.startswith("visual."):
        assert (
            dest_name.startswith("visual.patch_embed.")
            or dest_name.startswith("visual.merger.")
            or dest_name.startswith("visual.blocks.")
        )

        dim = "dp_cp_tp"
        if dp_shard_size > 1:
            dims_map[dim] = 0
        else:
            pass

        tensor_dim_to_parallel_map = {}
        for k, v in dims_map.items():
            if v not in tensor_dim_to_parallel_map:
                tensor_dim_to_parallel_map[v] = []
            tensor_dim_to_parallel_map[v].append(k)
        pp_rank = 0
        return dims_map, tensor_dim_to_parallel_map, pp_rank
    else:
        raise ValueError(f"Unsupported weight for policy parallelism: {dest_name}")
