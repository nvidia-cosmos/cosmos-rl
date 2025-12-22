from typing import Dict, Tuple

import torch
from torch.nn import Parameter

from vllm.model_executor.layers.quantization.utils.w8a8_utils import Fp8LinearOp
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils import w8a8_utils
from vllm.model_executor.layers.quantization.fp8 import (
    Fp8MoEMethod,
)


from cosmos_rl.policy.model import WeightMapper
from cosmos_rl.utils.parallelism import ParallelDims

"""
This file is used to patch the vllm model to use rowwise fp8 linear.
"""

USE_PER_TOKEN_IF_DYNAMIC = True


def apply_patch_to_dispatch():
    # ensure that fp8 linear kernel is dispatched to torch._scaled_mm per-token/rowwise
    def dispatch_fp8_linear_kernel_to_torch_scaled_mm(*args, **kwargs):
        return w8a8_utils.torch_per_token_w8a8_scaled_mm

    w8a8_utils.dispatch_w8a8_scaled_mm = dispatch_fp8_linear_kernel_to_torch_scaled_mm


def simplify_process_weights_after_loading_for_linear():
    """
    This function is used to simplify the process_weights_after_loading of Fp8LinearMethod in vLLM, to quantize the
    weight of linear only in `rowwise` mode.
    Refer to the method `process_weights_after_loading`:
    https://github.com/vllm-project/vllm/blob/1a4f35e2eaa3ebdecb8ef9ff8302b01e289305c9/vllm/model_executor/layers/quantization/fp8.py#L319
    """

    def simplified_process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Warning: this is only for rowwise fp8 linear.
        qweight, weight_scale = ops.scaled_fp8_quant(
            layer.weight, scale=None, use_per_token_if_dynamic=USE_PER_TOKEN_IF_DYNAMIC
        )
        # Update the layer with the new values
        layer.weight = Parameter(qweight.t(), requires_grad=False)
        layer.weight_scale = Parameter(weight_scale, requires_grad=False)
        layer.input_scale = None

    # modify the process_weights_after_loading method for rowwise fp8 linear.
    Fp8LinearMethod.process_weights_after_loading = (
        simplified_process_weights_after_loading
    )


def simplify_process_weights_after_loading_for_moe():
    """
    This function is used to simplify the process_weights_after_loading of Fp8MoEMethod in vLLM, to quantize the
    weight of MoE only in `per-tensor` mode.
    Refer to the method `process_weights_after_loading` in `Fp8MoEMethod`:

    """

    def simplified_process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Lazy import to avoid importing triton too early.
        from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
            is_rocm_aiter_moe_enabled,
        )

        self.rocm_aiter_moe_enabled = is_rocm_aiter_moe_enabled()

        fp8_dtype = torch.float8_e4m3fn
        w13_weight = torch.empty_like(layer.w13_weight.data, dtype=fp8_dtype)
        w2_weight = torch.empty_like(layer.w2_weight.data, dtype=fp8_dtype)

        # Re-initialize w13_scale because we directly quantize
        # merged w13 weights and generate a single scaling factor.

        # FIXME: (lms) Now is per-tensor quant for each expert.
        layer.w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                layer.local_num_experts, dtype=torch.float32, device=w13_weight.device
            ),
            requires_grad=False,
        )
        for expert in range(layer.local_num_experts):
            w13_weight[expert, :, :], layer.w13_weight_scale[expert] = (
                ops.scaled_fp8_quant(layer.w13_weight.data[expert, :, :])
            )
            w2_weight[expert, :, :], layer.w2_weight_scale[expert] = (
                ops.scaled_fp8_quant(layer.w2_weight.data[expert, :, :])
            )
        layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)

    Fp8MoEMethod.process_weights_after_loading = (
        simplified_process_weights_after_loading
    )


def simplify_process_weights_after_loading_for_fp8():
    simplify_process_weights_after_loading_for_linear()
    # simplify_process_weights_after_loading_for_moe()


# patch the Linear layer.
def apply_fp8_linear_patch(model: torch.nn.Module):
    apply_patch_to_dispatch()
    for name, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is None:
            continue
        elif isinstance(quant_method, Fp8LinearMethod):
            # replace the fp8_linear op with our own config
            # that use rowwise fp8
            # WARNING: in `Fp8LinearOp` `__init__`, vllm will read the `vllm_config`
            # But at this time, `vllm_config` is empty. So there will have a warning that complains
            # it is not set. This only affects the padding, seems not a big problem.
            quant_method.fp8_linear = Fp8LinearOp(
                # Activation use dynamic quantization.
                act_quant_static=False,
                act_quant_group_shape=GroupShape.PER_TOKEN,  # Using per-token quantization for activation.
            )
        else:
            # We will not handle other quant methods.
            pass


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
        compatible_name = weight_mapper.rollout_map_local_key_to_hf_key(
            name + ".weight"
        )
        if compatible_name in cached_weight_map:
            module.weight = cached_weight_map[compatible_name]


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
        elif isinstance(quant_method, Fp8LinearMethod):
            weight_name = name + ".weight"
            compatible_name = weight_mapper.rollout_map_local_key_to_hf_key(weight_name)
            original_weight_map[compatible_name] = (
                module.weight
            )  # qweight has shape [in_dim, out_dim]
            hp_weight = (
                module.weight.t().to(promotion_dtype).contiguous()
            )  # hp weight has shape [out_dim, in_dim]
            hp_weight_map[compatible_name] = Parameter(hp_weight, requires_grad=False)
        else:
            # We will not handle other quant methods.
            pass
    return hp_weight_map, original_weight_map


def post_process_view_map_for_fp8(
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
        if "weight_scale" in key:
            continue
        processed_view_map[key] = value
    return processed_view_map


def monkey_patch_for_fp8(vllm_config, model):
    from vllm.config import set_current_vllm_config

    with set_current_vllm_config(vllm_config):
        apply_fp8_linear_patch(model)
