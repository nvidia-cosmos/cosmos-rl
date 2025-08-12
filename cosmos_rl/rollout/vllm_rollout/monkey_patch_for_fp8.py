from typing import Dict, Tuple

import torch
from torch.nn import Parameter

from vllm.model_executor.layers.quantization.utils.w8a8_utils import Fp8LinearOp
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils import w8a8_utils

from cosmos_rl.policy.model import WeightMapper


"""
This file is used to patch the vllm model to use rowwise fp8 linear.
"""


def apply_patch_to_dispatch():
    # ensure that fp8 linear kernel is dispatched to torch._scaled_mm per-token/rowwise
    def dispatch_fp8_linear_kernel_to_torch_scaled_mm(*args, **kwargs):
        return w8a8_utils.torch_per_token_w8a8_scaled_mm

    w8a8_utils.dispatch_w8a8_scaled_mm = dispatch_fp8_linear_kernel_to_torch_scaled_mm


apply_patch_to_dispatch()


def simplify_process_weights_after_loading():
    """
    This function is used to simplify the process_weights_after_loading of Fp8LinearMethod in vLLM, to quantize the
    weight of linear only in `rowwise` mode.
    Refer to the method `process_weights_after_loading`:
    https://github.com/vllm-project/vllm/blob/1a4f35e2eaa3ebdecb8ef9ff8302b01e289305c9/vllm/model_executor/layers/quantization/fp8.py#L319
    """

    def simplified_process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Warning: this is only for rowwise fp8 linear.
        qweight, weight_scale = ops.scaled_fp8_quant(
            layer.weight, scale=None, use_per_token_if_dynamic=True
        )

        # Update the layer with the new values
        layer.weight = Parameter(qweight.t(), requires_grad=False)
        layer.weight_scale = Parameter(weight_scale, requires_grad=False)
        layer.input_scale = None

    # modify the process_weights_after_loading method for rowwise fp8 linear.
    Fp8LinearMethod.process_weights_after_loading = (
        simplified_process_weights_after_loading
    )


simplify_process_weights_after_loading()


# patch the Linear layer.
def apply_fp8_linear_patch(model: torch.nn.Module):
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
                # disable cutlass fp8, beacause we want that torch._scaled_mm is used for fp8 linear.
                cutlass_fp8_supported=False,
                # enable per token, because we are using rowwise now.
                use_per_token_if_dynamic=True,
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
        compatible_name = weight_mapper._rollout_vllm_name_to_hf(name + ".weight")
        if compatible_name in cached_weight_map:
            module.weight = cached_weight_map[compatible_name]


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
        elif isinstance(quant_method, Fp8LinearMethod):
            weight_name = name + ".weight"
            compatible_name = weight_mapper._rollout_vllm_name_to_hf(weight_name)
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
