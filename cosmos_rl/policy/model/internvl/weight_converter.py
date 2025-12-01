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
from cosmos_rl.utils.parallelism import ParallelDims
import torch
from typing import Tuple


def map_key_from_hf(name: str, src_model_type: str) -> str:
    if src_model_type in ["internvl", "internvl_chat"]:
        prefix = None
        if name.startswith("language_model.model."):
            prefix = "language_model.model."
        elif name.startswith("language_model."):
            prefix = "language_model."
        elif name.startswith("mlp1."):
            prefix = "mlp1."
        elif name.startswith("vision_model."):
            prefix = "vision_model."
        else:
            raise ValueError(f"Unsupported weight: {name}")
        return name.replace(prefix, "")
    else:
        raise ValueError(f"Unsupported model type: {src_model_type}")


def multi_modal_projector_weight_from_hf(
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

    dest_name = map_key_from_hf(name, src_model_type)
    if dest_name.startswith(("0.", "1.", "3.")):
        # multi_modal_projector.layer_norm/linear_1/linear_2
        shard = tensor
    elif not ignore_unknown_weights:
        raise ValueError(f"Unsupported weight: {dest_name}")
    else:
        return None, None
    # Do FSDP sharding
    shard = shard.tensor_split(dp_shard_size, dim=0)[dp_shard_rank]
    return dest_name, shard.contiguous()


def qwen3_moe_lm_weight_from_hf(
    tensor: torch.Tensor,
    name: str,
    src_model_type: str,
    n_experts: int,
    parallel_dims: ParallelDims,
    ignore_unknown_weights: bool = False,
) -> Tuple[str, torch.Tensor]:
    tp_ep_rank, tp_ep_size = parallel_dims.tp_coord
    assert n_experts % tp_ep_size == 0, "n_experts must be divisible by tp_ep_size"

    load_weight_test = not hasattr(parallel_dims, "mesh")
    if not load_weight_test:
        dp_shard_rank = parallel_dims.mesh[tuple(("dp_shard_cp",))].get_local_rank()
        dp_shard_size = parallel_dims.mesh[tuple(("dp_shard_cp",))].size()
    else:
        dp_shard_rank = 0
        dp_shard_size = 1

    # Expert weight are aggregated into (n_experts, in_features, out_features)
    # Weight are loaded in (out_features, in_features) shape
    # So we do not do FSDP sharding on expert weights, instead we filter by expert id
    should_do_fsdp_sharding = True

    dest_name = map_key_from_hf(name, src_model_type)

    if "lm_head.weight" == dest_name:
        shard = tensor.tensor_split(tp_ep_size, dim=0)[tp_ep_rank]
    elif "lm_head.bias" == dest_name:
        shard = tensor
    elif "embed_tokens.weight" == dest_name:
        shard = tensor.tensor_split(tp_ep_size, dim=0)[tp_ep_rank]
    elif dest_name in ["norm.weight", "norm.bias"]:
        shard = tensor
    elif (
        match := re.search(
            r"layers\.(\d+)\.self_attn\.(q_norm|k_norm|v_norm)\.(weight|bias)",
            dest_name,
        )
    ) is not None:
        shard = tensor
    elif (
        match := re.search(r"layers\.(\d+)\.input_layernorm\.(weight|bias)", dest_name)
    ) is not None:
        shard = tensor
    elif (
        match := re.search(
            r"layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj)\.(weight|bias)",
            dest_name,
        )
    ) is not None:
        shard = tensor.tensor_split(tp_ep_size, dim=0)[tp_ep_rank]
    elif (
        match := re.search(
            r"layers\.(\d+)\.self_attn\.(o_proj)\.(weight|bias)", dest_name
        )
    ) is not None:
        if dest_name.endswith(".bias"):
            shard = tensor
        else:
            shard = tensor.tensor_split(tp_ep_size, dim=-1)[tp_ep_rank]
    elif (
        match := re.search(  # noqa: F841
            r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(up_proj|gate_proj|down_proj)\.(weight|bias)",
            dest_name,
        )
    ) is not None:
        # shard = tensor.tensor_split(tp_ep_size, dim=0)[tp_ep_rank]
        # Check whether this expert belongs to the current process
        # Groups example (with 32 experts, and 4 EP groups):
        #  EP=0: 0, 1, 2, 3, 4, 5, 6, 7
        #  EP=1: 8, 9, 10, 11, 12, 13, 14, 15
        #  EP=2: 16, 17, 18, 19, 20, 21, 22, 23
        #  EP=3: 24, 25, 26, 27, 28, 29, 30, 31
        n_expert_per_ep = n_experts // tp_ep_size
        belongs_to_current_ep = (
            tp_ep_rank * n_expert_per_ep
            <= int(match.group(2))  # Expert index
            < (tp_ep_rank + 1) * n_expert_per_ep
        )
        belongs_to_current_dp_shard = (
            int(match.group(2)) - tp_ep_rank * n_expert_per_ep
        ) // (n_expert_per_ep // dp_shard_size) == dp_shard_rank
        if belongs_to_current_ep and belongs_to_current_dp_shard:
            should_do_fsdp_sharding = False
            shard = tensor
        else:
            # If the expert does not belong to the current process, return None to skip this weight
            return None, None
    elif (
        match := re.search(  # noqa: F841
            r"layers\.(\d+)\.post_attention_layernorm\.(weight|bias)", dest_name
        )
    ) is not None:
        shard = tensor
    elif (
        match := re.search(  # noqa: F841
            r"layers\.(\d+)\.mlp\.gate\.weight", dest_name
        )
    ) is not None:
        # TODO(cjx): Small enough, forbid FSDP sharding is better
        shard = tensor
    elif not ignore_unknown_weights:
        raise ValueError(f"Unsupported weight: {dest_name}")
    else:
        return None, None

    # Do FSDP sharding
    shard = shard.contiguous()
    if should_do_fsdp_sharding:
        shard = shard.tensor_split(dp_shard_size, dim=0)[dp_shard_rank]
    return dest_name, shard.contiguous()


def convert_weight_from_hf(
    tensor: torch.Tensor,
    name: str,
    src_model_type: str,
    lm_type: str,
    n_experts: int,
    parallel_dims: ParallelDims,
    ignore_unknown_weights: bool = False,
) -> Tuple[str, torch.Tensor]:
    # For LM
    lm_part_name, lm_part_shard = None, None
    if lm_type == "qwen3_moe":
        lm_part_name, lm_part_shard = qwen3_moe_lm_weight_from_hf(
            tensor,
            name,
            src_model_type,
            n_experts,
            parallel_dims,
            ignore_unknown_weights=True,
        )
        if lm_part_name is not None:
            return lm_part_name, lm_part_shard
        elif "experts" in name:
            return None, None
    else:
        raise ValueError(f"Unsupported LM type: {lm_type}")

    # For Multi-Modal Projector
    multi_modal_projector_name, multi_modal_projector_shard = (
        multi_modal_projector_weight_from_hf(
            tensor,
            name,
            src_model_type,
            parallel_dims,
            ignore_unknown_weights=True,
        )
    )
    if multi_modal_projector_name is not None:
        return multi_modal_projector_name, multi_modal_projector_shard

    # For Visual
    visual_prefix = "vision_model."
    assert name.startswith(visual_prefix), f"Unsupported weight: {name}"

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

    dest_name = name.replace(visual_prefix, "")
    shard = tensor
    # Do FSDP sharding only for visual part
    shard = shard.contiguous()
    if shard.shape[0] % dp_shard_size == 0:
        shard = shard.tensor_split(dp_shard_size, dim=0)
        shard = shard[dp_shard_rank]
    else:
        chunk_size = (shard.shape[0] + dp_shard_size - 1) // dp_shard_size
        shard = shard[dp_shard_rank * chunk_size : (dp_shard_rank + 1) * chunk_size]

    return dest_name, shard.contiguous()
