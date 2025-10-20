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

import re
import torch
import triton
import triton.language as tl
from typing import Any, Dict, Optional, Tuple

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.utils.parallelism_registry import (
    ParallelismStrategyRole,
    register_parallelism_strategy,
)

# Pre-compile regex patterns for better performance
_LAYER_ATTN_NORM_PATTERN = re.compile(
    r"model\.model\.layers\.(\d+)\.self_attn\.(q_norm|k_norm|v_norm)\.(weight|bias)"
)
_LAYER_INPUT_LAYERNORM_PATTERN = re.compile(
    r"model\.model\.layers\.(\d+)\.input_layernorm\.(weight|bias)"
)
_LAYER_ATTN_PROJ_PATTERN = re.compile(
    r"model\.model\.layers\.(\d+)\.self_attn\.(q_proj|kv_b_proj|q_b_proj)\.(weight|bias)"
)
_LAYER_ATTN_KV_A_PATTERN = re.compile(
    r"model\.model\.layers\.(\d+)\.self_attn\.(kv_a_proj_with_mqa)\.(weight|bias)"
)
_LAYER_ATTN_O_PATTERN = re.compile(
    r"model\.model\.layers\.(\d+)\.self_attn\.(o_proj)\.(weight|bias)"
)
_LAYER_MLP_EXPERTS_PATTERN = re.compile(
    r"model\.model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(up_proj|gate_proj|down_proj)\.(weight|bias)"
)
_LAYER_MLP_SHARED_EXPERTS_PATTERN = re.compile(
    r"model\.model\.layers\.(\d+)\.mlp\.shared_experts\.(up_proj|gate_proj)\.(weight|bias)"
)
_LAYER_MLP_SHARED_EXPERTS_DOWN_PATTERN = re.compile(
    r"model\.model\.layers\.(\d+)\.mlp\.shared_experts\.down_proj\.(weight|bias)"
)
_LAYER_MLP_PROJ_PATTERN = re.compile(
    r"model\.model\.layers\.(\d+)\.mlp\.(up_proj|gate_proj)\.(weight|bias)"
)
_LAYER_MLP_DOWN_PATTERN = re.compile(
    r"model\.model\.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)"
)
_LAYER_POST_ATTENTION_LAYERNORM_PATTERN = re.compile(
    r"model\.model\.layers\.(\d+)\.post_attention_layernorm\.(weight|bias)"
)
_LAYER_KV_A_LAYERNORM_PATTERN = re.compile(
    r"model\.model\.layers\.(\d+)\.self_attn\.kv_a_layernorm\.(weight|bias)"
)
_LAYER_Q_A_PATTERN = re.compile(
    r"model\.model\.layers\.(\d+)\.self_attn\.(q_a_layernorm|q_a_proj)\.(weight|bias)"
)
_LAYER_MLP_GATE_WEIGHT_PATTERN = re.compile(
    r"model\.model\.layers\.(\d+)\.mlp\.gate\.weight"
)
_LAYER_ROTARY_EMB_PATTERN = re.compile(
    r"model\.model\.layers\.(\d+)\.self_attn\.rotary_emb\.inv_freq"
)
_LAYER_MLP_GATE_BIAS_PATTERN = re.compile(
    r"model\.model\.layers\.(\d+)\.mlp\.gate\.e_score_correction_bias"
)

# Pre-define common suffixes for faster string matching
_LM_HEAD_WEIGHT = "lm_head.weight"
_LM_HEAD_BIAS = "lm_head.bias"
_EMBED_TOKENS_WEIGHT = "embed_tokens.weight"
_NORM_WEIGHT = "norm.weight"
_NORM_BIAS = "norm.bias"
_BIAS_SUFFIX = ".bias"


def map_key_from_hf(name: str) -> str:
    # The weights in the policy model have a ".model" prefix.
    return "model." + name


def convert_weight_from_hf(
    tensor: torch.Tensor,
    name: str,
    src_model_type: str,
    parallel_dims: ParallelDims,
    n_experts: int,
    ignore_unknown_weights: bool = False,
) -> Tuple[Optional[str], Optional[torch.Tensor], Optional[int]]:
    del src_model_type

    tp_ep_rank, tp_ep_size = parallel_dims.tp_coord
    assert n_experts % tp_ep_size == 0, "n_experts must be divisible by tp_ep_size"

    if parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled:
        dp_shard_rank = parallel_dims.mesh[tuple(("dp_shard_cp",))].get_local_rank()
        dp_shard_size = parallel_dims.mesh[tuple(("dp_shard_cp",))].size()
    else:
        dp_shard_rank = 0
        dp_shard_size = 1

    dest_name = map_key_from_hf(name)

    # Fast path for common cases using string suffix matching
    if dest_name.endswith(_LM_HEAD_WEIGHT):
        shard = tensor.tensor_split(tp_ep_size, dim=0)[tp_ep_rank]
    elif dest_name.endswith(_LM_HEAD_BIAS):
        shard = tensor
    elif dest_name.endswith(_EMBED_TOKENS_WEIGHT):
        shard = tensor.tensor_split(tp_ep_size, dim=0)[tp_ep_rank]
    elif dest_name.endswith(_NORM_WEIGHT) or dest_name.endswith(_NORM_BIAS):
        shard = tensor
    # Regex matching for complex patterns
    elif (match := _LAYER_ATTN_NORM_PATTERN.search(dest_name)) is not None:
        shard = tensor
    elif (match := _LAYER_INPUT_LAYERNORM_PATTERN.search(dest_name)) is not None:
        shard = tensor
    elif (match := _LAYER_ATTN_PROJ_PATTERN.search(dest_name)) is not None:
        shard = tensor.tensor_split(tp_ep_size, dim=0)[tp_ep_rank]
    elif (match := _LAYER_ATTN_KV_A_PATTERN.search(dest_name)) is not None:
        shard = tensor
    elif (match := _LAYER_ATTN_O_PATTERN.search(dest_name)) is not None:
        if dest_name.endswith(_BIAS_SUFFIX):
            shard = tensor
        else:
            shard = tensor.tensor_split(tp_ep_size, dim=-1)[tp_ep_rank]

    elif (match := _LAYER_MLP_EXPERTS_PATTERN.search(dest_name)) is not None:
        # Check whether this expert belongs to the current process
        # Groups example (with 32 experts, and 4 EP groups):
        #  EP=0: 0, 1, 2, 3, 4, 5, 6, 7
        #  EP=1: 8, 9, 10, 11, 12, 13, 14, 15
        #  EP=2: 16, 17, 18, 19, 20, 21, 22, 23
        #  EP=3: 24, 25, 26, 27, 28, 29, 30, 31
        n_expert_per_ep = n_experts // tp_ep_size
        n_expert_per_dp = n_expert_per_ep // dp_shard_size

        expert_id = int(match.group(2))
        belongs_to_current_ep = (expert_id // n_expert_per_ep) == tp_ep_rank

        expert_idx_within_ep = expert_id % n_expert_per_ep
        belongs_to_current_dp_shard = (
            expert_idx_within_ep // n_expert_per_dp
        ) == dp_shard_rank

        if belongs_to_current_ep and belongs_to_current_dp_shard:
            # remove `experts.$ID.` from dest_name
            dest_name = dest_name.replace(f"experts.{expert_id}.", "experts.")
            # change `proj.weight` to `projs`.
            dest_name = dest_name.replace("proj.weight", "projs")

            shard = tensor
            return dest_name, shard.contiguous(), expert_id
        else:
            # If the expert does not belong to the current rank, return None
            # to skip this weight.
            return None, None, None

    elif (match := _LAYER_MLP_SHARED_EXPERTS_PATTERN.search(dest_name)) is not None:
        shard = tensor.tensor_split(tp_ep_size, dim=0)[tp_ep_rank]

    elif (
        match := _LAYER_MLP_SHARED_EXPERTS_DOWN_PATTERN.search(dest_name)
    ) is not None:
        if dest_name.endswith(_BIAS_SUFFIX):
            shard = tensor
        else:
            shard = tensor.tensor_split(tp_ep_size, dim=-1)[tp_ep_rank]
    elif (match := _LAYER_MLP_PROJ_PATTERN.search(dest_name)) is not None:
        shard = tensor.tensor_split(tp_ep_size, dim=0)[tp_ep_rank]
    elif (match := _LAYER_MLP_DOWN_PATTERN.search(dest_name)) is not None:
        if dest_name.endswith(_BIAS_SUFFIX):
            shard = tensor
        else:
            shard = tensor.tensor_split(tp_ep_size, dim=-1)[tp_ep_rank]
    elif (
        match := _LAYER_POST_ATTENTION_LAYERNORM_PATTERN.search(dest_name)
    ) is not None:
        shard = tensor
    elif (match := _LAYER_KV_A_LAYERNORM_PATTERN.search(dest_name)) is not None:
        shard = tensor
    elif (match := _LAYER_Q_A_PATTERN.search(dest_name)) is not None:
        shard = tensor
    elif (match := _LAYER_MLP_GATE_WEIGHT_PATTERN.search(dest_name)) is not None:
        # TODO(cjx): Small enough, forbid FSDP sharding is better
        shard = tensor
    elif (match := _LAYER_ROTARY_EMB_PATTERN.search(dest_name)) is not None:
        return None, None, None
    elif (match := _LAYER_MLP_GATE_BIAS_PATTERN.search(dest_name)) is not None:
        shard = tensor
    elif not ignore_unknown_weights:
        raise ValueError(f"Unsupported weight: {dest_name}")
    else:
        return None, None, None

    # Expert weight are aggregated into (n_experts, in_features, out_features)
    # Weight are loaded in (out_features, in_features) shape
    # So we do not do FSDP sharding on expert weights, instead we filter by expert id

    # Do FSDP sharding
    shard = shard.contiguous()
    if match := _LAYER_ATTN_KV_A_PATTERN.search(dest_name) is not None and (
        576 % dp_shard_size != 0
    ):
        tensor_shard_size = 576 // dp_shard_size + 1
        if dp_shard_rank < (576 // tensor_shard_size):
            shard = shard[
                dp_shard_rank * tensor_shard_size : (dp_shard_rank + 1)
                * tensor_shard_size
            ]
        else:
            shard = shard[dp_shard_rank * tensor_shard_size :]
    else:
        shard = shard.tensor_split(dp_shard_size, dim=0)[dp_shard_rank]

    return dest_name, shard.contiguous(), None


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    assert x.is_contiguous() and s.is_contiguous()
    assert x.dim() == 2 and s.dim() == 2
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_SIZE"]), triton.cdiv(N, meta["BLOCK_SIZE"]))

    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


@register_parallelism_strategy("deepseek_v3", role=ParallelismStrategyRole.ROLLOUT)
def map_weight_parallel_dims(
    n_dim: int,
    dest_name: str,
    parallel_dims: ParallelDims,
    model_config: Any,
) -> Tuple[Dict[str, int], Dict[int, list], int]:
    """
    For the given HuggingFace parameter name (dest_name), return the map from
    parallel dimension name (e.g. "tp", ...) to the corresponding dimension of the tensor.
    """
    del parallel_dims, model_config

    # Why are TP and EP always used together? They should be independent.
    # TODO(aazzolini): 1.2 implement rollout_parallelism_strategy (tp,ep)
    dims_map = {}

    tp_dim_map = {
        # deepseekv3 attention weights
        ".self_attn.q_a_proj.": None,
        ".self_attn.q_a_layernorm.": None,
        ".self_attn.q_b_proj.": 0,
        ".self_attn.kv_a_proj_with_mqa.": None,
        ".self_attn.kv_a_layernorm.": None,
        ".self_attn.kv_b_proj.": 0,
        ".self_attn.o_proj.": 1,
        # deepseekv3 dense+moe mlp weights
        # note on the training side gate_up_proj is fused.
        ".mlp.gate_proj.": -2,
        ".mlp.up_proj.": -2,
        ".mlp.down_proj.": -1,
        # deepseekv3 moe-related weights
        ".mlp.gate.": None,
        ".mlp.shared_experts.gate_proj.": 0,
        ".mlp.shared_experts.up_proj.": 0,
        ".mlp.shared_experts.down_proj.": 1,
        # deepseekv3 layernorms
        ".input_layernorm.weight": None,
        ".post_attention_layernorm.weight": None,
        # deepseekv3 toplevel
        ".norm.weight": None,
        "lm_head.weight": 0,
        ".embed_tokens.": 0,
        # vit attention
        ".attention_norm.": None,
        ".attention.wq_proj.": 0,
        ".attention.wk_proj.": 0,
        ".attention.wv_proj.": 0,
        ".mlp.fc1.bias": 0,
        ".mlp.fc1.weight": 0,
        ".mlp.fc2.weight": 1,
    }

    tp_dim = None
    for pattern, dim in tp_dim_map.items():
        if pattern in dest_name:
            tp_dim = dim
            if tp_dim is not None and tp_dim < 0:
                tp_dim += n_dim
            break

    logger.debug(
        f"Rollout parallelism mapping: Dest_name: {dest_name}, tp_dim: {tp_dim}"
    )

    if tp_dim is not None:
        dims_map["tp"] = tp_dim

    # if dp_shard_size > 1:
    #    dims_map["dp_shard_cp"] = 0

    # What is this mapping for? It looks like this maps each
    # tensor dimension to a list of mesh dimensions.
    tensor_dim_to_parallel_map = {}
    for k, v in dims_map.items():
        if v not in tensor_dim_to_parallel_map:
            tensor_dim_to_parallel_map[v] = []
        tensor_dim_to_parallel_map[v].append(k)
    pp_rank = 0
    return dims_map, tensor_dim_to_parallel_map, pp_rank
