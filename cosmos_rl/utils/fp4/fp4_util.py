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
import torch.nn as nn

from strenum import StrEnum
from typing import Union, List
from functools import partial
from cosmos_rl.utils.fp4.float4_linear_utils import convert_to_float4_training
from cosmos_rl.utils.fp4.config import Float4LinearConfig

from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.utils.util import is_cuda_compatible, torch_version_at_least
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.model_converter import ModelConverter

MIN_TORCH_VERSION_FOR_FP4 = "2.7.0"
IS_TORCH_COMPATIBLE_WITH_FP4 = torch_version_at_least(MIN_TORCH_VERSION_FOR_FP4)

if not IS_TORCH_COMPATIBLE_WITH_FP4:
    logger.warning(
        f"[FP4] FP4 is not supported for this version of PyTorch, minimum version required: {MIN_TORCH_VERSION_FOR_FP4}, but got: {torch.__version__}. FP4 setting will take no effect."
    )


class FP4Recipe(StrEnum):
    DYNAMIC_SCALING = "dynamic_scaling"
    DELAYED_SCALING = "delayed_scaling"


def is_valid_fp4_recipe(value: str) -> bool:
    return value in FP4Recipe.__members__.values()


class FP4QuantRecipe(StrEnum):
    ROWWISE = "rowwise"
    TENSORWISE = "tensorwise"


def is_valid_fp4_quant_recipe(value: str) -> bool:
    return value in FP4QuantRecipe.__members__.values()


# Refer to: https://github.com/pytorch/torchtitan/blob/e7c0cae934df78d6e9c2835f42ff1f757dc3fddc/torchtitan/components/quantization/utils.py#L10
def module_filter_fn(mod: nn.Module, fqn: str, filter_fqns: list[str]) -> bool:
    """
    Filter function to determine which modules should be converted.
    For both Float4 and MXFP4, we only convert Linear modules
    with dimensions divisible by 16 and not matching any filtered FQNs.

    Args:
        mod: The module to be converted.
        fqn: The fully qualified name of the module.
        filter_fqns: The list of FQNs to filter.

    Returns:
        True if the module should be converted, False otherwise.
    """
    if not isinstance(mod, nn.Linear):
        return False

    # All dims must be divisible by 16 due to float8 tensorcore hardware requirements.
    dims_multiples_of_16 = (
        mod.weight.shape[0] % 16 == 0 and mod.weight.shape[1] % 16 == 0
    )

    # If the fqn matches any filtered fqn, then we should not convert this module.
    is_filtered_fqn = any(filter_fqn in fqn for filter_fqn in filter_fqns)

    return dims_multiples_of_16 and not is_filtered_fqn


class FP4ModelConverter(ModelConverter):
    def __init__(self, config: CosmosConfig, parallel_dims: ParallelDims):
        super().__init__(config, parallel_dims)
        if not IS_TORCH_COMPATIBLE_WITH_FP4:
            return

        if not is_cuda_compatible(8, 9):
            raise RuntimeError(
                "NVFP4 is only supported for device that has compute capability 10.0 or higher"
            )
        self.fp4_config = config.train.fp4

        assert is_valid_fp4_quant_recipe(self.fp4_config.quant_recipe)
        assert is_valid_fp4_recipe(self.fp4_config.fp4_recipe)

        if self.fp4_config.fp4_recipe == FP4Recipe.DELAYED_SCALING:
            raise NotImplementedError("[FP4] Delayed scaling is not supported yet.")

        self.precompute_scale = False

        if self.fp4_config.quant_recipe == "rowwise":
            # From torchtitan, it reports an issue that RMSNorm will cause NaN when rowwise quantization and torch.compile is enabled,
            # From that issue, it is recommended to set torch._inductor.config.emulate_precision_casts to True to avoid this.
            # Issue: https://github.com/pytorch/pytorch/issues/150859
            torch._inductor.config.emulate_precision_casts = True
            self.nvfp4_config = Float4LinearConfig.from_recipe_name(
                self.fp4_config.quant_recipe
            )
            logger.debug(
                "[FP4] Set torch._inductor.config.emulate_precision_casts to True"
            )
        elif self.fp4_config.quant_recipe == "tensorwise":
            self.precompute_scale = False

    def convert_model(self, model: torch.nn.Module) -> torch.nn.Module:
        if not IS_TORCH_COMPATIBLE_WITH_FP4:
            return

        if not self.fp4_config.enable_fp4:
            return

        convert_to_float4_training(
            model,
            config=self.nvfp4_config,
            module_filter_fn=partial(
                module_filter_fn, filter_fqns=model.fqn_filter_for_quantization()
            ),
        )

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]):
        """
        Not implemented yet.
        """
        return
