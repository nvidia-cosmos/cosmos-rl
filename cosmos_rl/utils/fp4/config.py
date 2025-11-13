import enum
from typing import Optional, Union
from dataclasses import dataclass
import transformer_engine_torch as tex

te_dtype = tex.DType.kFloat4E2M1


class ScalingType(enum.Enum):
    DYNAMIC = "dynamic"
    # ScalingType.DISABLED means "skip scaling for this tensor, leave it in
    # its original precision.
    DISABLED = "disabled"

    def short_str(self):
        if self is ScalingType.DYNAMIC:
            return "dyn"
        else:
            assert self is ScalingType.DISABLED
            return "dis"


class ScalingGranularity(enum.Enum):
    """
    Defines the granularity of scaling strategies for casting to float4
    """

    # A single scaling factor for the entire tensor
    TENSORWISE = "tensorwise"
    # Scaling factors computed along one axis of the tensor, reducing it to
    # size 1.
    AXISWISE = "axiswise"

    def short_str(self):
        if self is ScalingGranularity.TENSORWISE:
            return "ten"
        else:
            assert self is ScalingGranularity.AXISWISE
            return "axs"


@dataclass(frozen=True)
class CastConfig:
    """
    Configuration for casting a tensor to float4
    """

    scaling_type: ScalingType = ScalingType.DYNAMIC
    scaling_granularity: ScalingGranularity = ScalingGranularity.TENSORWISE
    target_dtype: Optional[tex.DType] = None

    def short_str(self):
        dtype_name = (
            self.target_dtype.name if self.target_dtype is not None else te_dtype.name
        )
        return f"{self.scaling_type.short_str()}_{self.scaling_granularity.short_str()}_{dtype_name}"

    def __post_init__(self):
        if self.scaling_granularity is ScalingGranularity.AXISWISE:
            assert (
                self.scaling_type is ScalingType.DYNAMIC
            ), "only dynamic scaling is supported for axiswise scaling granularity"


class Float4LinearRecipeName(enum.Enum):
    """
    Recipe name for float4 linear
    """

    # Default, dynamic per-tensor scaling with the cuBLAS tensorwise kernel
    TENSORWISE = "tensorwise"
    # Dynamic rowwise scaling with the CUTLASS rowwise kernel
    ROWWISE = "rowwise"
    # Rowwise scaling with GW-HP
    ROWWISE_WITH_GW_HP = "rowwise_with_gw_hp"


@dataclass(frozen=True)
class Float4LinearConfig:
    """
    Configuration for float4 linear
    """

    # `input`
    cast_config_input: CastConfig = CastConfig()
    cast_config_input_for_grad_weight: Optional[CastConfig] = None

    # `weight`
    cast_config_weight: CastConfig = CastConfig()
    cast_config_weight_for_grad_input: Optional[CastConfig] = None

    # `output`
    cast_config_grad_output: CastConfig = CastConfig()
    cast_config_grad_output_for_grad_weight: Optional[CastConfig] = None

    def __post_init__(self):
        if self.cast_config_input_for_grad_weight is None:
            object.__setattr__(
                self, "cast_config_input_for_grad_weight", self.cast_config_input
            )
        if self.cast_config_weight_for_grad_input is None:
            object.__setattr__(
                self, "cast_config_weight_for_grad_input", self.cast_config_weight
            )
        if self.cast_config_grad_output_for_grad_weight is None:
            object.__setattr__(
                self,
                "cast_config_grad_output_for_grad_weight",
                self.cast_config_grad_output,
            )

        cc_i = self.cast_config_input
        cc_w = self.cast_config_weight
        cc_go = self.cast_config_grad_output
        cc_i_gw = self.cast_config_input_for_grad_weight
        cc_w_gi = self.cast_config_weight_for_grad_input
        cc_go_gw = self.cast_config_grad_output_for_grad_weight

        for cc1, cc2, gemm_name in (
            (cc_i, cc_w, "output"),
            (cc_go, cc_w_gi, "grad_input"),
            (cc_i_gw, cc_go_gw, "grad_weight"),
        ):
            is_disabled_1 = cc1.scaling_type is ScalingType.DISABLED
            is_disabled_2 = cc2.scaling_type is ScalingType.DISABLED
            assert (
                is_disabled_1 == is_disabled_2
            ), f"scaling type of {gemm_name} must be the same, got {cc1.scaling_type} and {cc2.scaling_type}"

        for cc1, cc2, operand_name, default_dtype in [
            (cc_i, cc_i_gw, "input", te_dtype),
            (cc_w, cc_w_gi, "weight", te_dtype),
            (cc_go, cc_go_gw, "grad_output", te_dtype),
        ]:
            if cc1.target_dtype is None:
                object.__setattr__(cc1, "target_dtype", default_dtype)
            if cc2.target_dtype is None:
                object.__setattr__(cc2, "target_dtype", default_dtype)
            assert (
                cc1.target_dtype == cc2.target_dtype
            ), f"{operand_name} must be cast to the same dtype, got {cc1.target_dtype} and {cc2.target_dtype}"

    @staticmethod
    def from_recipe_name(
        recipe_name: Union[Float4LinearRecipeName, str],
    ) -> "Float4LinearConfig":
        """
        Input: `Float4LinearRecipeName` value, or a string representing the recipe name
        Output: `Float4LinearConfig`
        """
        if isinstance(recipe_name, str):
            valid_names = [n.value for n in Float4LinearRecipeName]
            assert (
                recipe_name in valid_names
            ), f"recipe name {recipe_name} is not valid, must be one of {valid_names}"
            recipe_name = Float4LinearRecipeName(recipe_name)

        if recipe_name is Float4LinearRecipeName.TENSORWISE:
            return Float4LinearConfig()

        elif recipe_name is Float4LinearRecipeName.ROWWISE:
            cc_i = CastConfig(
                scaling_type=ScalingType.DYNAMIC,
                scaling_granularity=ScalingGranularity.AXISWISE,
                target_dtype=te_dtype,
            )
            cc_w = CastConfig(
                scaling_type=ScalingType.DYNAMIC,
                scaling_granularity=ScalingGranularity.AXISWISE,
                target_dtype=te_dtype,
            )
            cc_go = CastConfig(
                scaling_type=ScalingType.DYNAMIC,
                scaling_granularity=ScalingGranularity.AXISWISE,
                target_dtype=te_dtype,
            )

            return Float4LinearConfig(
                cast_config_input=cc_i,
                cast_config_weight=cc_w,
                cast_config_grad_output=cc_go,
            )

        elif recipe_name is Float4LinearRecipeName.ROWWISE_WITH_GW_HP:
            cc_i = CastConfig(scaling_granularity=ScalingGranularity.AXISWISE)
            cc_w = CastConfig(scaling_granularity=ScalingGranularity.AXISWISE)

            cc_go = CastConfig(
                scaling_granularity=ScalingGranularity.AXISWISE, target_dtype=te_dtype
            )
            cc_w_gi = CastConfig(scaling_granularity=ScalingGranularity.TENSORWISE)

            cc_i_gw = CastConfig(scaling_type=ScalingType.DISABLED)
            cc_go_gw = CastConfig(
                scaling_type=ScalingType.DISABLED, target_dtype=te_dtype
            )

            return Float4LinearConfig(
                cast_config_input=cc_i,
                cast_config_weight=cc_w,
                cast_config_grad_output=cc_go,
                cast_config_input_for_grad_weight=cc_i_gw,
                cast_config_weight_for_grad_input=cc_w_gi,
                cast_config_grad_output_for_grad_weight=cc_go_gw,
            )

        else:
            raise ValueError(f"recipe name {recipe_name} is not valid")
