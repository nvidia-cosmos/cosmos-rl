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

from abc import ABC, abstractmethod
from typing import Union, List, Dict, Type
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.parallelism import ParallelDims


class ModelConverter(ABC):
    def __init__(self, config: CosmosConfig, parallel_dims: ParallelDims):
        self.config = config
        self.parallel_dims = parallel_dims

    @abstractmethod
    def convert_model(self, model: torch.nn.Module) -> torch.nn.Module: ...

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]):
        """
        Post-optimizer hook (e.g. compute weights statistics).
        """
        ...


# Using torchao, like what torchtitan does.
class QuantizationConverter(ModelConverter):
    """
    Base class for quantization converters, which implements generic validation reusable across all quantization converters.
    """

    enabled: bool = False

    def __init__(self, cosmos_config: CosmosConfig, parallel_dims: ParallelDims):
        self._validate(cosmos_config)

    @staticmethod
    def _validate(cosmos_config: CosmosConfig):
        if cosmos_config.train.quantization.quantization_type != "none":
            # check if either linear or MoE quantization is enabled.
            if (
                not cosmos_config.train.quantization.linear_quantization_config.enable
                and not cosmos_config.train.quantization.moe_quantization_config.enable
            ):
                raise ValueError(
                    "Either linear or MoE quantization must be enabled when quantization type is not none."
                )


# registry for quantization converter classes.
_QUANTIZATION_CONVERTER_MODULE_TYPE_REGISTRY: Dict[
    str, Dict[str, Type[QuantizationConverter]]
] = {
    "linear": {},
    "moe": {},
}


def register_quantization_converter_class(
    module_type: str,  # Linear or MoE
    reg_key: str,  # quantization_type in cosmos_config.train.quantization.quantization_type.
    *,
    allow_override: bool = False,
):
    def decorator(cls: Type[QuantizationConverter]) -> Type[QuantizationConverter]:
        if module_type not in _QUANTIZATION_CONVERTER_MODULE_TYPE_REGISTRY:
            raise ValueError(f"Module type '{module_type}' is not supported.")
        container = _QUANTIZATION_CONVERTER_MODULE_TYPE_REGISTRY[module_type]
        if not allow_override and reg_key in container:
            raise ValueError(
                f"Quantization converter class '{reg_key}' is already registered."
            )
        container[reg_key] = cls
        return cls

    return decorator
