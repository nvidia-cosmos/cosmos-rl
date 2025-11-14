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

from cosmos_rl.utils.logging import logger
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

# from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.mistral.modeling_mistral import MistralForCausalLM
from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration,
    )
except ImportError:
    logger.warning(
        "Qwen3VLForConditionalGeneration is not available. Please install transformers >= 4.57.0 to import Qwen3VLForConditionalGeneration, "
    )
    Qwen3VLForConditionalGeneration = None


# For VLMs, we only support TP for the language model, and ignore the vision encoder.
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
    # slice_dim_map is a dictionary that maps the tensor name to the slice dimension and slice bias.
    # key: tensor_name, value: (slice_dim, slice_bias)
    # Note:only colwise_parallel has slice_bias
    if model_class in [
        LlamaForCausalLM,
        MistralForCausalLM,
        Gemma3ForCausalLM,
        Gemma3ForConditionalGeneration,
        Qwen2ForCausalLM,
        Qwen2_5_VLForConditionalGeneration,
        Qwen3ForCausalLM,
        Qwen3VLForConditionalGeneration,
    ]:
        if model_class in [
            Gemma3ForConditionalGeneration,
            Qwen2_5_VLForConditionalGeneration,
            Qwen3VLForConditionalGeneration,
        ]:
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

        slice_dim_map: dict[str, (int, bool)] = {
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
            "lm_head": colwise_parallel(
                output_layouts=Replicate(),
                use_local_output=True,
            ),
        }

        slice_dim_map: dict[str, (int, bool)] = {
            f"{model_prefix}.embed_tokens": (0, False),
            f"{model_prefix}.layers.*.self_attn.qkv_proj": (-1, False),
            f"{model_prefix}.layers.*.self_attn.o_proj": (0, True),
            f"{model_prefix}.layers.*.mlp.gate_up_proj": (0, True),
            f"{model_prefix}.layers.*.mlp.down_proj": (-1, False),
            "lm_head": (0, False),
        }
    # TODO(huik): support TP for attention sinks
    # elif model_class is GptOssForCausalLM:
    #     tp_plan: dict[str, ParallelStyle] = {
    #         f"{model_prefix}.embed_tokens": rowwise_parallel(input_layouts=Replicate()),
    #         f"{model_prefix}.layers.*.self_attn.sinks": colwise_parallel(
    #             input_layouts=Replicate(),
    #             output_layouts=Shard(-1),
    #             use_local_output=False,
    #         ),
    #         f"{model_prefix}.layers.*.self_attn.q_proj": colwise_parallel(),
    #         f"{model_prefix}.layers.*.self_attn.k_proj": colwise_parallel(),
    #         f"{model_prefix}.layers.*.self_attn.v_proj": colwise_parallel(),
    #         f"{model_prefix}.layers.*.self_attn.o_proj": rowwise_parallel(),
    #         # Shard MLP layers
    #         f"{model_prefix}.layers.*.mlp.experts.gate_up_proj": colwise_parallel(
    #             input_layouts=Replicate(),
    #             output_layouts=Shard(-1),
    #             use_local_output=False,
    #         ),
    #         f"{model_prefix}.layers.*.mlp.experts.down_proj": rowwise_parallel(
    #             input_layouts=Shard(-1),
    #             output_layouts=Replicate(),
    #         ),
    #         "lm_head": colwise_parallel(
    #             output_layouts=Replicate(),
    #             use_local_output=True,
    #         ),
    #     }
    #     slice_dim_map: dict[str, (int, bool)] = {
    #         f"{model_prefix}.embed_tokens": (0, False),
    #         f"{model_prefix}.layers.*.self_attn.sinks": (-1, False),
    #         f"{model_prefix}.layers.*.self_attn.q_proj": (0, True),
    #         f"{model_prefix}.layers.*.self_attn.k_proj": (0, True),
    #         f"{model_prefix}.layers.*.self_attn.v_proj": (0, True),
    #         f"{model_prefix}.layers.*.self_attn.o_proj": (-1, False),
    #         f"{model_prefix}.layers.*.mlp.gate_up_proj": (0, True),
    #         f"{model_prefix}.layers.*.mlp.down_proj": (-1, False),
    #         "lm_head": (0, False),
    #     }
    else:
        raise ValueError(
            f"Unsupported model class({model_class}) for TP. Please set tp_size to 1."
        )

    # Generate tp_slice_dim_map for all parameters in slice_dim_map
    n_lm_layers = model.n_lm_layers
    tp_slice_dim_map = {}
    for plan_key, (slice_dim, slice_bias) in slice_dim_map.items():
        if "*" in plan_key:
            for i in range(n_lm_layers):
                expanded_key = plan_key.replace("*", str(i))
                tp_slice_dim_map[expanded_key + ".weight"] = slice_dim
                tp_slice_dim_map[expanded_key + ".bias"] = 0 if slice_bias else None
        else:
            tp_slice_dim_map[plan_key + ".weight"] = slice_dim
            tp_slice_dim_map[plan_key + ".bias"] = 0 if slice_bias else None
    # check if all parameters of model are in tp_slice_dim_map
    for name, _ in model.named_parameters():
        if name not in tp_slice_dim_map:
            logger.debug(f"{name} is not in tp_slice_dim_map")
    # set tp_slice_dim_map
    model.tp_slice_dim_map = tp_slice_dim_map

    return cast(dict[str, ParallelStyle], tp_plan)
