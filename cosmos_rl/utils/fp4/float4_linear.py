"""
A simple module swap UX for a float4 version of `torch.nn.Linear`.
"""

from typing import Optional

import torch
from cosmos_rl.utils.fp4.config import Float4LinearConfig, ScalingType

# TODO: (lms) remove the context manager after this PR is released: https://github.com/NVIDIA/TransformerEngine/pull/1913
from contextlib import contextmanager
from importlib.metadata import version as get_pkg_version, PackageNotFoundError


@contextmanager
def importlib_metadata_version_context():
    original_version = get_pkg_version

    def mocked_version(name):
        if name == "flash-attn-3":
            raise PackageNotFoundError
        return original_version(name)

    import importlib.metadata

    importlib.metadata.version = mocked_version
    try:
        yield
    finally:
        importlib.metadata.version = original_version


with importlib_metadata_version_context():
    try:
        import transformer_engine_torch as tex
        from transformer_engine.pytorch.constants import TE_DType
        from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

        te_dtype = tex.DType.kFloat4E2M1
        HAVE_TE = True
    except ImportError:
        HAVE_TE = False


class matmul_with_hp_or_float4_args(torch.autograd.Function):
    """
    Like torch.matmul, but with the arguments in either high precision or float8.
    * if the arguments are in high precision, they are cast to float4 according
      to the specified config
    * if the arguments are in float4, we assume the cast honored the config
    """

    @staticmethod
    def forward(
        ctx,
        input_hp: torch.Tensor,
        weight_hp_t: torch.Tensor,
        enable_input_quantization: bool,
        input_target_dtype: int,
        enable_weight_quantization: bool,
        weight_target_dtype: int,
        enable_grad_output_dim0_quantization: bool,
        grad_output_dim0_target_dtype: int,
        enable_weight_t_dim0_quantization: bool,
        weight_t_dim0_target_dtype: int,
        enable_grad_output_dim1_quantization: bool,
        grad_output_dim1_target_dtype: int,
        enable_input_t_dim1_quantization: bool,
        input_t_dim1_target_dtype: int,
    ):
        ctx.save_for_backward(input_hp, weight_hp_t)

        ctx.enable_grad_output_dim0_quantization = enable_grad_output_dim0_quantization
        ctx.enable_weight_t_dim0_quantization = enable_weight_t_dim0_quantization
        ctx.enable_grad_output_dim1_quantization = enable_grad_output_dim1_quantization
        ctx.enable_input_t_dim1_quantization = enable_input_t_dim1_quantization
        ctx.grad_output_dim0_target_dtype = grad_output_dim0_target_dtype
        ctx.weight_t_dim0_target_dtype = weight_t_dim0_target_dtype
        ctx.grad_output_dim1_target_dtype = grad_output_dim1_target_dtype
        ctx.input_t_dim1_target_dtype = input_t_dim1_target_dtype

        original_input_hp_shape = input_hp.shape
        input_hp_reshaped = input_hp.squeeze(0)
        if enable_input_quantization:
            input_maybe_fp4 = input_hp_reshaped
        else:
            input_quantizer = NVFP4Quantizer(
                fp4_dtype=input_target_dtype,
                rowwise=True,
                columnwise=False,
                with_amax_reduction=False,
                amax_reduction_group=None,
                with_rht=False,
                with_post_rht_amax=False,
            )
            input_maybe_fp4 = input_quantizer.make_empty(
                input_hp_reshaped.shape, dtype=torch.bfloat16, requires_grad=False
            )
            input_maybe_fp4 = input_quantizer.update_quantized(
                input_hp_reshaped, input_maybe_fp4
            )

        if enable_weight_quantization:
            weight_maybe_fp4_t = weight_hp_t
        else:
            weight_quantizer = NVFP4Quantizer(
                fp4_dtype=weight_target_dtype,
                rowwise=False,
                columnwise=True,
                with_amax_reduction=False,
                amax_reduction_group=None,
                with_rht=False,
                with_post_rht_amax=False,
            )
            weight_maybe_fp4_t = weight_quantizer.make_empty(
                weight_hp_t.shape, dtype=torch.bfloat16, requires_grad=False
            )
            weight_maybe_fp4_t = weight_quantizer.update_quantized(
                weight_hp_t, weight_maybe_fp4_t
            )
        workspace = torch.empty(
            64 * 4096 * 4096, dtype=torch.uint8, device=input_maybe_fp4.device
        )
        res_bits = tex.generic_gemm(
            input_maybe_fp4,
            True,
            weight_maybe_fp4_t,
            True,
            None,
            None,
            TE_DType[torch.bfloat16],
            None,
            TE_DType[torch.bfloat16],
            False,
            None,
            False,
            workspace,
            workspace.shape[0],
            False,
            False,
        )[0]
        res_bits = res_bits.t().reshape(
            *original_input_hp_shape[:-1], res_bits.t().shape[-1]
        )
        return res_bits

    @staticmethod
    def backward(ctx, grad_output):
        input_hp, weight_hp_t = ctx.saved_tensors
        grad_output_orig_shape = grad_output.shape
        grad_output_reshaped = grad_output.squeeze(0)
        #
        # calculate grad_input
        #
        if ctx.enable_grad_output_dim0_quantization:
            grad_output_maybe_fp4_dim0 = grad_output_reshaped
        else:
            grad_output_dim0_quantizer = NVFP4Quantizer(
                fp4_dtype=ctx.grad_output_dim0_target_dtype,
                rowwise=True,
                columnwise=False,
                with_amax_reduction=False,
                amax_reduction_group=None,
                with_rht=False,
                with_post_rht_amax=False,
            )
            grad_output_maybe_fp4_dim0 = grad_output_dim0_quantizer.make_empty(
                grad_output_reshaped.shape, dtype=torch.bfloat16
            )
            grad_output_maybe_fp4_dim0 = grad_output_dim0_quantizer.update_quantized(
                grad_output_reshaped, grad_output_maybe_fp4_dim0
            )

        if ctx.enable_weight_t_dim0_quantization:
            weight_t_maybe_fp4_dim0 = weight_hp_t
        else:
            weight_hp_t_quantizer = NVFP4Quantizer(
                fp4_dtype=ctx.weight_t_dim0_target_dtype,
                rowwise=True,
                columnwise=False,
                with_amax_reduction=False,
                amax_reduction_group=None,
                with_rht=False,
                with_post_rht_amax=False,
                with_2d_quantization=True,
            )
            weight_t_maybe_fp4_dim0 = weight_hp_t_quantizer.make_empty(
                weight_hp_t.shape, dtype=torch.bfloat16
            )
            weight_t_maybe_fp4_dim0 = weight_hp_t_quantizer.update_quantized(
                weight_hp_t, weight_t_maybe_fp4_dim0
            )

        workspace = torch.empty(
            64 * 4096 * 4096,
            dtype=torch.uint8,
            device=grad_output_maybe_fp4_dim0.device,
        )
        grad_input = tex.generic_gemm(
            grad_output_maybe_fp4_dim0,
            True,
            weight_t_maybe_fp4_dim0,
            False,
            None,
            None,
            TE_DType[torch.bfloat16],
            None,
            TE_DType[torch.bfloat16],
            False,
            None,
            False,
            workspace,
            workspace.shape[0],
            False,
            False,
        )[0]
        grad_input = grad_input.t().reshape(
            *grad_output_orig_shape[:-1], grad_input.t().shape[-1]
        )
        #
        # calculate grad_weight
        #
        # dY and X must be [M, N] and [M, K] exactly, contiguous on the last dim

        grad_output_reshaped = grad_output.reshape(
            -1, grad_output.shape[-1]
        ).contiguous()  # (M, N)
        input_hp_reshaped = input_hp.reshape(
            -1, input_hp.shape[-1]
        ).contiguous()  # (M, K)

        if ctx.enable_grad_output_dim1_quantization:
            grad_output_maybe_fp4_dim1 = grad_output_reshaped
        else:
            grad_output_dim1_quantizer = NVFP4Quantizer(
                fp4_dtype=ctx.grad_output_dim1_target_dtype,
                rowwise=False,
                columnwise=True,
                with_amax_reduction=False,
                amax_reduction_group=None,
                with_rht=False,
                with_post_rht_amax=False,
            )
            grad_output_maybe_fp4_dim1 = grad_output_dim1_quantizer.make_empty(
                grad_output_reshaped.shape, dtype=torch.bfloat16
            )
            grad_output_maybe_fp4_dim1 = grad_output_dim1_quantizer.update_quantized(
                grad_output_reshaped, grad_output_maybe_fp4_dim1
            )

        if ctx.enable_input_t_dim1_quantization:
            input_maybe_fp4_dim1 = input_hp_reshaped
        else:
            input_hp_quantizer = NVFP4Quantizer(
                fp4_dtype=ctx.input_t_dim1_target_dtype,
                rowwise=False,
                columnwise=True,  # NOTE: rowwise here
                with_amax_reduction=False,
                amax_reduction_group=None,
                with_rht=False,
                with_post_rht_amax=False,
            )
            input_maybe_fp4_dim1 = input_hp_quantizer.make_empty(
                input_hp_reshaped.shape, dtype=torch.bfloat16
            )
            input_maybe_fp4_dim1 = input_hp_quantizer.update_quantized(
                input_hp_reshaped, input_maybe_fp4_dim1
            )

        grad_weight = tex.generic_gemm(
            input_maybe_fp4_dim1,
            False,
            grad_output_maybe_fp4_dim1,
            True,
            None,
            None,
            TE_DType[torch.bfloat16],
            None,
            TE_DType[torch.bfloat16],
            False,
            None,
            True,
            workspace,
            workspace.shape[0],
            False,
            False,
        )[0]

        return (
            grad_input,
            grad_weight.t(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class Float4Linear(torch.nn.Linear):
    """
    Note: this is **not** a public API and is only intended to be used
    inside of this repository. Please file an issue if you would benefit
    from this being a public API.

    A wrapper around a `torch.nn.Linear` module which does fp4 compute.
    """

    def __init__(self, *args, **kwargs):
        """
        Additional arguments on top of `torch.nn.Linear`'s arguments:
        * `config`: Float4LinearConfig
        """
        if not HAVE_TE:
            raise ImportError(
                "Transformer Engine is required for Float4Linear. Please install it with `pip install transformer-engine`."
            )

        config = kwargs.pop("config")
        super().__init__(*args, **kwargs)

        # Defines the scaling behavior of input, weight, grad_output
        self._enable_input_quantization = (
            config.cast_config_input.scaling_type is not ScalingType.DISABLED
        )
        self._input_target_dtype = int(config.cast_config_input.target_dtype)
        self._enable_weight_quantization = (
            config.cast_config_weight.scaling_type is not ScalingType.DISABLED
        )
        self._weight_target_dtype = int(config.cast_config_weight.target_dtype)
        self._enable_grad_output_dim0_quantization = (
            config.cast_config_grad_output.scaling_type is not ScalingType.DISABLED
        )
        self._grad_output_dim0_target_dtype = int(
            config.cast_config_grad_output.target_dtype
        )
        self._enable_weight_t_dim0_quantization = (
            config.cast_config_weight_for_grad_input.scaling_type
            is not ScalingType.DISABLED
        )
        self._weight_t_dim0_target_dtype = int(
            config.cast_config_weight_for_grad_input.target_dtype
        )
        self._enable_grad_output_dim1_quantization = (
            config.cast_config_grad_output_for_grad_weight.scaling_type
            is not ScalingType.DISABLED
        )
        self._grad_output_dim1_target_dtype = int(
            config.cast_config_grad_output_for_grad_weight.target_dtype
        )
        self._enable_input_t_dim1_quantization = (
            config.cast_config_input_for_grad_weight.scaling_type
            is not ScalingType.DISABLED
        )
        self._input_t_dim1_target_dtype = int(
            config.cast_config_input_for_grad_weight.target_dtype
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Duplicate the autocast logic for F.linear, so that the output
        # of our module has the right original precision
        if torch.is_autocast_enabled():
            # For now, hardcode to GPU's autocast dtype
            # if we need CPU support in the future, we can add it
            autocast_dtype = torch.get_autocast_gpu_dtype()
            input = input.to(autocast_dtype)

        output = matmul_with_hp_or_float4_args.apply(
            input,
            self.weight.t(),
            self._enable_input_quantization,
            self._input_target_dtype,
            self._enable_weight_quantization,
            self._weight_target_dtype,
            self._enable_grad_output_dim0_quantization,
            self._grad_output_dim0_target_dtype,
            self._enable_weight_t_dim0_quantization,
            self._weight_t_dim0_target_dtype,
            self._enable_grad_output_dim1_quantization,
            self._grad_output_dim1_target_dtype,
            self._enable_input_t_dim1_quantization,
            self._input_t_dim1_target_dtype,
        )

        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output

    def extra_repr(self):
        c = self.config
        ci = f"i:{c.cast_config_input.short_str()}"
        cw = f"w:{c.cast_config_weight.short_str()}"
        cgo = f"go:{c.cast_config_grad_output.short_str()}"
        parts = [ci, cw, cgo]
        if c.cast_config_input_for_grad_weight != c.cast_config_input:
            parts.append(f"i_gw:{c.cast_config_input_for_grad_weight.short_str()}")
        if c.cast_config_weight_for_grad_input != c.cast_config_weight:
            parts.append(f"w_gi:{c.cast_config_weight_for_grad_input.short_str()}")
        if c.cast_config_grad_output_for_grad_weight != c.cast_config_grad_output:
            parts.append(
                f"go_gw:{c.cast_config_grad_output_for_grad_weight.short_str()}"
            )
        cast_config_str = ",".join(parts)
        s = f"{super().extra_repr()}, cast_configs={cast_config_str}"
        return s

    @classmethod
    def from_float(
        cls,
        mod,
        config: Optional[Float4LinearConfig] = None,
    ):
        """
        Create an nn.Linear with fp4 compute from a regular nn.Linear

        Args:
            mod (torch.nn.Linear): nn.Linear to convert
            config (Optional[Float4LinearConfig]): configuration for conversion to float4
        """
        if config is None:
            config = Float4LinearConfig()
        with torch.device("meta"):
            new_mod = cls(
                mod.in_features,
                mod.out_features,
                bias=False,
                config=config,
            )
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias

        return new_mod
