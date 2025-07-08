import torch
import types

from torch.nn import Parameter

from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8KVCacheMethod
from vllm.model_executor.layers.quantization.utils.w8a8_utils import Fp8LinearOp
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils import w8a8_utils


"""
This file is used to patch the vllm model to use rowwise fp8 linear.
"""


def is_cuda_tensor(tensor: torch.Tensor):
    return tensor.device.type == "cuda"


def add_method(obj, method, method_name):
    setattr(obj, method_name, types.MethodType(method, obj))


def apply_patch_to_dispatch():
    # dispatch fp8 linear func to torch._scaled_mm per token/rowwise
    def dispatch_fp8_linear_kernel_to_torch_scaled_mm(*args, **kwargs):
        return w8a8_utils.torch_per_token_w8a8_scaled_mm

    w8a8_utils.dispatch_w8a8_scaled_mm = dispatch_fp8_linear_kernel_to_torch_scaled_mm


apply_patch_to_dispatch()


def set_process_weights_after_loading_empty():
    def empty_process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    Fp8LinearMethod.process_weights_after_loading = empty_process_weights_after_loading


def simplify_process_weights_after_loading():
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
            # it is not set. This only affects the padding, seems no big problem.
            quant_method.fp8_linear = Fp8LinearOp(
                # disable cutlass fp8
                cutlass_fp8_supported=False,
                # enable per token, because we are using rowwise now.
                use_per_token_if_dynamic=True,
            )
        elif isinstance(quant_method, UnquantizedEmbeddingMethod):
            # do nothing for this special method.
            pass
        elif isinstance(quant_method, Fp8KVCacheMethod):
            # do nothing for attention.
            pass
        else:
            raise NotImplementedError(
                f"[Rollout] There are some non-linear quantized method: {type(quant_method)} that are not supported yet. Please contact the maintainer."
            )
