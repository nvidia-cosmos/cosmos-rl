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

from typing import cast

from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    ParallelStyle,
)
from torch.distributed.tensor.placement_types import Replicate, Shard

from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
)
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM

from cosmos_rl.utils.logging import logger


def get_tp_plans(model, enable_float8_tensorwise_tp: bool = False):
    if enable_float8_tensorwise_tp:
        from torchao.float8.float8_tensor_parallel import (
            Float8ColwiseParallel,
            Float8RowwiseParallel,
        )

        rowwise_parallel, colwise_parallel = (
            Float8RowwiseParallel,
            Float8ColwiseParallel,
        )
    else:
        rowwise_parallel, colwise_parallel = (
            RowwiseParallel,
            ColwiseParallel,
        )

    tp_plan = None
    model_prefix = "model"
    model_class = model.model_class
    # tensor_name, (slice_dim, slice_bias)
    # only colwise_parallel has slice_bias
    tp_slice_dim_map = None
    if model_class in [
        LlamaForCausalLM,
        Gemma3ForCausalLM,
        Gemma3ForConditionalGeneration,
    ]:
        if model_class is Gemma3ForConditionalGeneration:
            model_prefix = "model.language_model"

        tp_plan: dict[str, ParallelStyle] = {
            f"{model_prefix}.embed_tokens": rowwise_parallel(input_layouts=Replicate()),
            f"{model_prefix}.layers.*.self_attn.q_proj": colwise_parallel(),
            f"{model_prefix}.layers.*.self_attn.k_proj": colwise_parallel(),
            f"{model_prefix}.layers.*.self_attn.v_proj": colwise_parallel(),
            f"{model_prefix}.layers.*.self_attn.o_proj": rowwise_parallel(),
            f"{model_prefix}.layers.*.mlp.up_proj": colwise_parallel(),
            f"{model_prefix}.layers.*.mlp.gate_proj": colwise_parallel(),
            f"{model_prefix}.layers.*.mlp.down_proj": rowwise_parallel(),
            "lm_head": colwise_parallel(
                output_layouts=Replicate(), use_local_output=True
            ),
        }

        tp_slice_dim_map: dict[str, (int, bool)] = {
            f"{model_prefix}.embed_tokens": (0, False),
            f"{model_prefix}.layers.*.self_attn.q_proj": (0, True),
            f"{model_prefix}.layers.*.self_attn.k_proj": (0, True),
            f"{model_prefix}.layers.*.self_attn.v_proj": (0, True),
            f"{model_prefix}.layers.*.self_attn.o_proj": (-1, False),
            f"{model_prefix}.layers.*.mlp.up_proj": (0, True),
            f"{model_prefix}.layers.*.mlp.gate_proj": (0, True),
            f"{model_prefix}.layers.*.mlp.down_proj": (-1, False),
            "lm_head": (0, False),
        }
    elif model_class is Phi3ForCausalLM:
        tp_plan: dict[str, ParallelStyle] = {
            f"{model_prefix}.embed_tokens": rowwise_parallel(
                input_layouts=Replicate(),
                output_layouts=Replicate(),
            ),
            # Fused Attention can not be sharded
            f"{model_prefix}.layers.*.self_attn.qkv_proj": rowwise_parallel(
                input_layouts=Replicate(),
                output_layouts=Replicate(),
            ),
            f"{model_prefix}.layers.*.self_attn.o_proj": colwise_parallel(
                input_layouts=Replicate(),
                output_layouts=Replicate(),
            ),
            # Shard MLP layers
            f"{model_prefix}.layers.*.mlp.gate_up_proj": colwise_parallel(
                input_layouts=Replicate(),
                output_layouts=Shard(-1),
                use_local_output=False,
            ),
            f"{model_prefix}.layers.*.mlp.down_proj": rowwise_parallel(
                input_layouts=Shard(-1),
                output_layouts=Replicate(),
            ),
            f"{model_prefix}.lm_head": colwise_parallel(
                output_layouts=Replicate(),
                use_local_output=True,
            ),
        }

        tp_slice_dim_map: dict[str, (int, bool)] = {
            f"{model_prefix}.embed_tokens": (0, False),
            f"{model_prefix}.layers.*.self_attn.qkv_proj": (-1, False),
            f"{model_prefix}.layers.*.self_attn.o_proj": (0, True),
            f"{model_prefix}.layers.*.mlp.gate_up_proj": (0, True),
            f"{model_prefix}.layers.*.mlp.down_proj": (-1, False),
            "lm_head": (0, False),
        }
    else:
        raise ValueError(
            f"Unsupported model class({model_class}) for tensor parallelism"
        )
    # set tp_slice_dim_map
    n_lm_layers = model.n_lm_layers
    slice_dim_map = {}
    for plan_key, (slice_dim, slice_bias) in tp_slice_dim_map.items():
        if "*" in plan_key:
            for i in range(n_lm_layers):
                expanded_key = plan_key.replace("*", str(i))
                slice_dim_map[expanded_key + ".weight"] = slice_dim
                slice_dim_map[expanded_key + ".bias"] = 0 if slice_bias else None
        else:
            slice_dim_map[plan_key + ".weight"] = slice_dim
            slice_dim_map[plan_key + ".bias"] = 0 if slice_bias else None
    # check if all parameters are in slice_dim_map
    for name, _ in model.named_parameters():
        if name not in slice_dim_map:
            logger.debug(f"{name} is not in tp_slice_dim_map")

    model.tp_slice_dim_map = slice_dim_map

    return cast(dict[str, ParallelStyle], tp_plan)
