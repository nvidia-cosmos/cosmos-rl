from typing import Dict, Tuple

import torch
from torch.nn import Parameter

from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
from vllm.model_executor.layers.fused_moe import FusedMoE

from cosmos_rl.utils.arithmetic import cdiv
from cosmos_rl.policy.model import WeightMapper
from cosmos_rl.utils.parallelism import ParallelDims

"""
This file is used to patch the vllm model to use mxfp4 for GPT-OSS now.
"""


def quantize_mx4(w):
    from triton_kernels.numerics_details.mxfp import downcast_to_mxfp

    w, w_scale = downcast_to_mxfp(w.to(torch.bfloat16), torch.uint8, axis=1)
    # w = convert_layout(wrap_torch_tensor(w, dtype=FP4), HopperMXValueLayout, mx_axis=1)
    # w_scale = convert_layout(wrap_torch_tensor(w_scale), StridedLayout)
    return w, w_scale


def replace_weight_of_quantized_module(
    vllm_model: torch.nn.Module,
    cached_weight_map: Dict[str, torch.Tensor],
    weight_mapper: WeightMapper,
):
    """
    Temporarily replace the quantized fp8 layer's weight with the cached weight.
    """
    for name, module in vllm_model.named_modules():
        # Here we use the compatible name as the key, aligned with what we do in
        # `cache_weight_of_quantized_module` and `rollout_prepare_recv`.
        # create a new key for gate_up_proj and down_proj
        w13_weight_name = "w13_weight"
        w2_weight_name = "w2_weight"
        w13_bias_name = "w13_bias"
        for weight_name in [w13_weight_name, w2_weight_name, w13_bias_name]:
            compatible_name = weight_mapper._rollout_vllm_name_to_hf(
                name + f".{weight_name}"
            )
            if compatible_name in cached_weight_map:
                setattr(module, weight_name, cached_weight_map[compatible_name])


def cache_weight_of_quantized_module(
    vllm_model: torch.nn.Module,
    promotion_dtype: torch.dtype,
    weight_mapper: WeightMapper,
    parallel_dims: ParallelDims,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Get the weight from the quantized module."""
    original_weight_map = {}
    hp_weight_map = {}

    for name, module in vllm_model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is None:
            continue
        elif isinstance(quant_method, Mxfp4MoEMethod):
            assert isinstance(
                module, FusedMoE
            ), "Mxfp4MoEMethod should be used with FusedMoE"
            # w13 and w2
            w13_weight_name = "w13_weight"
            w13_compatible_name = weight_mapper._rollout_vllm_name_to_hf(
                name + f".{w13_weight_name}"
            )
            w2_weight_name = "w2_weight"
            w2_compatible_name = weight_mapper._rollout_vllm_name_to_hf(
                name + f".{w2_weight_name}"
            )
            w13_bias_name = "w13_bias"
            w13_bias_compatible_name = weight_mapper._rollout_vllm_name_to_hf(
                name + f".{w13_bias_name}"
            )
            # Keep same as vLLM did: delete the original weight.
            original_weight_map[w13_compatible_name] = None
            original_weight_map[w2_compatible_name] = None
            original_weight_map[w13_bias_compatible_name] = module.w13_bias

            # We assume that MoE high precision weight should have shape:
            # w13:[num_experts, 2 * intermediate_size, hidden_size]
            # w2: [num_experts, hidden_size, intermediate_size]
            # FIXME: (lms) So we just don't support EP for temporarily. When vLLM support EP for gpt-oss, fix this.
            # intermediate_size_per_partition = (
            #     module.intermediate_size_per_partition
            # )  # equals to: intermediate_size // self.tp_size
            hidden_size = module.hidden_size
            w13_bias = getattr(module, "w13_bias", None)
            assert w13_bias is not None, "w13_bias should exist"
            # Note: we assume that Policy HFModel is in BF16.

            intermediate_size = weight_mapper.config.intermediate_size
            mxfp4_block = 32
            intermediate_size_block = intermediate_size // mxfp4_block  # 90
            tp_rank, tp_size = parallel_dims.tp_coord
            per_rank_intermediate_size_block = cdiv(intermediate_size_block, tp_size)
            per_rank_intermediate_size = per_rank_intermediate_size_block * mxfp4_block

            tp_rank_start = tp_rank * per_rank_intermediate_size
            tp_rank_end = min(
                (tp_rank + 1) * per_rank_intermediate_size, intermediate_size
            )

            length = tp_rank_end - tp_rank_start

            w13_hp_weight = torch.empty(
                [
                    module.local_num_experts,
                    2 * length,
                    hidden_size,
                ],
                dtype=promotion_dtype,
                device=w13_bias.device,
            )
            w2_hp_weight = torch.empty(
                [
                    module.local_num_experts,
                    hidden_size,
                    length,
                ],
                dtype=promotion_dtype,
                device=w13_bias.device,
            )

            hp_weight_map[w13_compatible_name] = Parameter(
                w13_hp_weight, requires_grad=False
            )
            hp_weight_map[w2_compatible_name] = Parameter(
                w2_hp_weight, requires_grad=False
            )
            # For w13_bias
            w13_bias_hp_weight = torch.empty(
                [
                    module.local_num_experts,
                    2 * length,
                ],
                dtype=w13_bias.dtype,
                device=w13_bias.device,
            )
            hp_weight_map[w13_bias_compatible_name] = Parameter(
                w13_bias_hp_weight, requires_grad=False
            )
        else:
            # We will not handle other quant methods.
            pass
    return hp_weight_map, original_weight_map


def post_process_view_map_for_mxfp4(
    vllm_weight_inplace_view_map: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Process the view map returned by `rollout_prepare_recv`.
            - remove the weight_scale from the view map.
    Args:
        vllm_weight_inplace_view_map (Dict[str, torch.Tensor]): view map returned by `rollout_prepare_recv`
    Returns:
        Dict[str, torch.Tensor]: view map doesn't contain weight_scale.
    """
    processed_view_map = {}
    for key, value in vllm_weight_inplace_view_map.items():
        if "down_proj_scales" in key or "gate_up_proj_scales" in key:
            continue
        processed_view_map[key] = value
    return processed_view_map
