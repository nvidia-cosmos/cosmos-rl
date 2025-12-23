from typing import Dict, Tuple

import torch
from torch.nn import Parameter

from vllm.model_executor.layers.quantization.utils.w8a8_utils import Fp8LinearOp
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils import w8a8_utils
from vllm.model_executor.layers.quantization.fp8 import Fp8OnlineMoEMethod


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
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

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
    # With vLLM 0.13.0, for online-dynamic-moe quantization, `Fp8OnlineMoEMethod` is used.
    This function is used to simplify the process_weights_after_loading of Fp8OnlineMoEMethod in vLLM, to quantize the
    weight of MoE only in `per-tensor` mode.
    Refer to the method `process_weights_after_loading` in `Fp8OnlineMoEMethod`:
    """

    def simplified_process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        # This function is simplified only for cuda device.
        # If checkpoint is fp16, quantize in place.
        from vllm.model_executor.utils import replace_parameter

        fp8_dtype = torch.float8_e4m3fn
        w13_weight = torch.empty_like(layer.w13_weight.data, dtype=fp8_dtype)
        w2_weight = torch.empty_like(layer.w2_weight.data, dtype=fp8_dtype)

        for expert in range(layer.local_num_experts):
            w13_weight[expert, :, :], layer.w13_weight_scale[expert] = (
                ops.scaled_fp8_quant(layer.w13_weight.data[expert, :, :])
            )
            w2_weight[expert, :, :], layer.w2_weight_scale[expert] = (
                ops.scaled_fp8_quant(layer.w2_weight.data[expert, :, :])
            )
        # Each expert will have a single scale value
        # Shape of layer.w13_weight_scale and layer.w2_weight_scale is [num_experts]

        replace_parameter(layer, "w13_weight", w13_weight)
        replace_parameter(layer, "w2_weight", w2_weight)

    Fp8OnlineMoEMethod.process_weights_after_loading = (
        simplified_process_weights_after_loading
    )


def simplify_process_weights_after_loading_for_fp8():
    simplify_process_weights_after_loading_for_linear()
    simplify_process_weights_after_loading_for_moe()


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
        elif isinstance(quant_method, Fp8OnlineMoEMethod):
            pass
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
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if parameter_name.endswith("weight"):
                # including: weight | w13_weight | w2_weight
                compatible_name = weight_mapper.rollout_map_local_key_to_hf_key(
                    name + "." + parameter_name
                )
                if compatible_name in cached_weight_map:
                    setattr(module, parameter_name, cached_weight_map[compatible_name])


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
        elif isinstance(quant_method, Fp8OnlineMoEMethod):
            w13_weight_name = name + ".w13_weight"
            w2_weight_name = name + ".w2_weight"
            w13_compatible_name = weight_mapper.rollout_map_local_key_to_hf_key(
                w13_weight_name
            )
            w2_compatible_name = weight_mapper.rollout_map_local_key_to_hf_key(
                w2_weight_name
            )
            original_weight_map[w13_compatible_name] = module.w13_weight
            original_weight_map[w2_compatible_name] = module.w2_weight
            hp_weight = module.w13_weight.to(
                promotion_dtype
            ).contiguous()  # hp weight has shape [out_dim, in_dim]
            hp_weight_map[w13_compatible_name] = Parameter(
                hp_weight, requires_grad=False
            )
            hp_weight = module.w2_weight.to(
                promotion_dtype
            ).contiguous()  # hp weight has shape [out_dim, in_dim]
            hp_weight_map[w2_compatible_name] = Parameter(
                hp_weight, requires_grad=False
            )
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
