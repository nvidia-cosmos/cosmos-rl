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

import torch

from typing import Dict, Any
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
from vllm.model_executor.layers.fused_moe import FusedMoE
from cosmos_rl.utils.dim_slice_info import DimSliceInfo
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.utils.arithmetic import cdiv


def genereate_dim_rank_info(
    module: FusedMoE,
    param_name: str,
    param: torch.Tensor,
    hf_config: Any,
    parallel_dims: ParallelDims,
) -> Dict[int, DimSliceInfo]:
    """
    Generate the dim_rank_info for the parameter. Main logic is from vLLM sharding strategy for GPT-OSS MXFP4.
    """
    dim_rank_info = {}
    mxfp4_block = 32
    intermediate_size = hf_config.intermediate_size
    tp_rank, tp_size = parallel_dims.tp_coord
    quant_method = getattr(module, "quant_method", None)
    assert quant_method is not None, "Must have quant_method for FusedMoE"
    assert isinstance(
        quant_method, Mxfp4MoEMethod
    ), "Only mxfp4 quant method is supported for now."
    # get the rounded shape of weight
    # 768
    intermediate_size_per_partition_after_pad = quant_method.intermediate_size
    # hidden_size_of_quant_method = quant_method.hidden_size  # This may also padded.

    weight_intermediate_size = intermediate_size_per_partition_after_pad
    scale_factor = 1
    tp_dim = -1
    if "w13_weight" in param_name:
        # shape: [num_experts, intermediate_size_per_partition_after_pad * 2, hidden_size // 2]
        tp_dim = 1
        scale_factor = 2
        weight_intermediate_size *= 2
    elif "w2_weight" in param_name:
        # shape: [num_experts, hidden_size, intermediate_size_per_partition_after_pad // 2]
        tp_dim = 2
    elif "w13_bias" in param_name:
        # shape: [num_experts, intermediate_size_per_partition_after_pad * 2]
        weight_intermediate_size *= 2
        scale_factor = 2
        tp_dim = 1
    elif "w2_bias" in param_name:
        # for w2_bias, each rank will take full shards.
        # But only rank 0 will load the full weight, other ranks will set w2_bias to zeros.
        # shape: [num_experts, hidden_size]
        pass
    else:
        pass

    # determin the lenght of dimsliceinfo
    intermediate_size_block = intermediate_size // mxfp4_block
    per_rank_intermediate_size_block = cdiv(intermediate_size_block, tp_size)
    per_rank_intermediate_size = per_rank_intermediate_size_block * mxfp4_block

    tp_rank_start = tp_rank * per_rank_intermediate_size
    tp_rank_end = min((tp_rank + 1) * per_rank_intermediate_size, intermediate_size)

    length = tp_rank_end - tp_rank_start
    weight_shape = param.shape
    for idx, g_size in enumerate(weight_shape):
        if idx == tp_dim:
            dim_rank_info[idx] = DimSliceInfo(
                offset=scale_factor * tp_rank_start,
                total_size=scale_factor * intermediate_size,
                length=scale_factor * length,
            ).__dict__
        else:
            dim_rank_info[idx] = DimSliceInfo(
                offset=0,
                total_size=g_size,
                length=g_size,
            ).__dict__

    return dim_rank_info, tp_dim
