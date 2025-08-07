from typing import Dict, Tuple

import torch
from torch.nn import Parameter

from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
from vllm.model_executor.layers.fused_moe import FusedMoE

from cosmos_rl.policy.model import WeightMapper

"""
This file is used to patch the vllm model to use mxfp4 for GPT-OSS now.
"""


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
        for weight_name in [w13_weight_name, w2_weight_name]:
            compatible_name = weight_mapper._rollout_vllm_name_to_hf(
                name + f".{weight_name}"
            )
            if compatible_name in cached_weight_map:
                setattr(module, weight_name, cached_weight_map[compatible_name])


def cache_weight_of_quantized_module(
    vllm_model: torch.nn.Module,
    promotion_dtype: torch.dtype,
    weight_mapper: WeightMapper,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Get the weight from the quantized module."""
    original_weight_map = {}
    hp_weight_map = {}

    for name, module in vllm_model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is None:
            continue
        elif isinstance(quant_method, Mxfp4MoEMethod):
            # FIXME: (lms) Bias also should be cached.
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

            # Keep same as vLLM did: delete the original weight.
            original_weight_map[w13_compatible_name] = None
            original_weight_map[w2_compatible_name] = None

            # We assume that MoE high precision weight should have shape:
            # w13:[num_experts, 2 * intermediate_size, hidden_size]
            # w2: [num_experts, hidden_size, intermediate_size]
            # FIXME: (lms) So we just don't support EP for temporarily.
            intermediate_size_per_partition = (
                module.intermediate_size_per_partition
            )  # equals to: intermediate_size // self.tp_size
            hidden_size = module.hidden_size
            w13_bias = getattr(module, "w13_bias", None)
            assert w13_bias is not None, "w13_bias should be set"
            w13_hp_weight = torch.empty(
                [
                    module.local_num_experts,
                    2 * intermediate_size_per_partition,
                    hidden_size,
                ],
                dtype=promotion_dtype,
                device=w13_bias.device,
            )
            w2_hp_weight = torch.empty(
                [
                    module.local_num_experts,
                    hidden_size,
                    intermediate_size_per_partition,
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
