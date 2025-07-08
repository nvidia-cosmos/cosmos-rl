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
from vllm.model_executor.parameter import (
    ModelWeightParameter,
    PackedColumnParameter,
    PackedvLLMParameter,
)
from vllm.distributed import get_tensor_model_parallel_rank

from cosmos_rl.utils.logging import logger


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


def quant_load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs):
    shard_offset = kwargs.get("shard_offset")
    shard_size = kwargs.get("shard_size")
    if (
        isinstance(self, (PackedColumnParameter, PackedvLLMParameter))
        and self.packed_dim == self.output_dim
    ):
        shard_size, shard_offset = self.adjust_shard_indexes_for_packing(
            shard_offset=shard_offset, shard_size=shard_size
        )

    param_data = self.data.t()

    tp_rank = get_tensor_model_parallel_rank()
    param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)
    loaded_weight = loaded_weight.narrow(
        self.output_dim, tp_rank * shard_size, shard_size
    )
    loaded_weight = loaded_weight.to(self.data.device)

    # loaded_weight: [out, in], especially for rowwise.
    qweight, weight_scale = ops.scaled_fp8_quant(
        loaded_weight, scale=None, use_per_token_if_dynamic=True
    )
    qweight = qweight.t()

    param_data = param_data.t()
    logger.info(
        f"[Rollout] MC: tp_rank: {tp_rank} qweight.shape: , {qweight.shape}, self.data.shape: {self.data.shape}, loaded_weight.shape: {loaded_weight.shape}, param_data.shape: {param_data.shape}"
    )
    assert param_data.shape == qweight.shape
    assert param_data.dtype == qweight.dtype
    param_data.t().copy_(qweight)

    # add the weight_scale to the weight parameter as attr, dynamically.
    self.tmp_weight_scale = weight_scale


def quant_load_column_parallel_weight(self, loaded_weight: torch.Tensor):
    tp_rank = get_tensor_model_parallel_rank()
    data = self.data.t()
    shard_size = data.shape[self.output_dim]
    loaded_weight = loaded_weight.narrow(
        self.output_dim, tp_rank * shard_size, shard_size
    )

    loaded_weight = loaded_weight.to(self.data.device)

    # loaded_weight: [out, in], especially for rowwise.
    qweight, weight_scale = ops.scaled_fp8_quant(
        loaded_weight, scale=None, use_per_token_if_dynamic=True
    )
    qweight = qweight.t()

    assert self.data.shape == qweight.shape
    assert self.data.dtype == qweight.dtype
    # update the weight and weight_scale.
    self.data.copy_(qweight)

    # add the weight_scale to the weight parameter as attr, dynamically.
    self.tmp_weight_scale = weight_scale


def quant_load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs):
    shard_offset = kwargs.get("shard_offset")
    shard_size = kwargs.get("shard_size")
    shard_id = kwargs.get("shard_id")
    num_heads = kwargs.get("num_heads")

    if (
        isinstance(self, (PackedColumnParameter, PackedvLLMParameter))
        and self.output_dim == self.packed_dim
    ):
        shard_size, shard_offset = self.adjust_shard_indexes_for_packing(
            shard_offset=shard_offset, shard_size=shard_size
        )

    param_data = self.data.t()
    tp_rank = get_tensor_model_parallel_rank()
    shard_id = tp_rank if shard_id == "q" else tp_rank // num_heads
    param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)
    loaded_weight = loaded_weight.narrow(
        self.output_dim, shard_id * shard_size, shard_size
    )

    loaded_weight = loaded_weight.to(self.data.device)

    # loaded_weight: [out, in], especially for rowwise.
    qweight, weight_scale = ops.scaled_fp8_quant(
        loaded_weight, scale=None, use_per_token_if_dynamic=True
    )
    qweight = qweight.t()
    logger.info(
        f"[Rollout] QKV: tp_rank: {tp_rank} qweight.shape: {qweight.shape}, self.data.shape: {self.data.shape}, loaded_weight.shape: {loaded_weight.shape}"
    )
    assert param_data.shape == qweight.shape
    assert param_data.dtype == qweight.dtype
    param_data.copy_(qweight)

    # add the weight_scale to the weight parameter as attr, dynamically.
    self.tmp_weight_scale = weight_scale


def quant_load_row_parallel_weight(self, loaded_weight: torch.Tensor):
    tp_rank = get_tensor_model_parallel_rank()

    # transpose the data to get right shape.
    data = self.data.t()
    shard_size = data.shape[self.input_dim]
    loaded_weight = loaded_weight.narrow(
        self.input_dim, tp_rank * shard_size, shard_size
    )

    if len(loaded_weight.shape) == 0:
        loaded_weight = loaded_weight.reshape(1)

    # loaded_weight: [out, in], especially for rowwise.
    loaded_weight = loaded_weight.to(self.data.device)

    qweight, weight_scale = ops.scaled_fp8_quant(
        loaded_weight, scale=None, use_per_token_if_dynamic=True
    )
    qweight = qweight.t()
    logger.info(
        f"[Rollout] tp_rank: {tp_rank} qweight.shape: , {qweight.shape}, self.data.shape: {self.data.shape}, loaded_weight.shape: {loaded_weight.shape}"
    )
    assert self.data.shape == qweight.shape
    assert self.data.dtype == qweight.dtype
    self.data.copy_(qweight)

    # add the weight_scale to the weight parameter as attr, dynamically.
    self.tmp_weight_scale = weight_scale


def apply_linear_load_and_quant_patch(model: torch.nn.Module):
    for _, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is None:
            continue
        elif isinstance(quant_method, Fp8LinearMethod):
            # this layer is fp8 linear.
            weight = module.weight
            # change Parameter to ModelWeightParameter
            new_weight = ModelWeightParameter(
                data=weight.data,
                input_dim=1,
                output_dim=0,
                weight_loader=module.weight_loader_v2,
            )
            module.weight = new_weight
            weight = module.weight
            assert isinstance(
                new_weight, ModelWeightParameter
            ), f"weight must be a ModelWeightParameter, but got: {type(weight)}, {type(module)}"

            add_method(
                weight, quant_load_row_parallel_weight, "load_row_parallel_weight"
            )
            add_method(
                weight, quant_load_column_parallel_weight, "load_column_parallel_weight"
            )
            add_method(
                weight, quant_load_merged_column_weight, "load_merged_column_weight"
            )
            add_method(weight, quant_load_qkv_weight, "load_qkv_weight")

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
