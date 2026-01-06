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

import torch.nn as nn

from typing import List, Union

from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.utils.model_converter import QuantizationConverter

# import all quantization converters
import cosmos_rl.utils.fp8  # noqa: F401
import cosmos_rl.utils.fp4  # noqa: F401


class ModelConvertersContainer(QuantizationConverter):
    """Model converters sequential container.

    The class build the sequence of model converters defined in `model.converters`
    job config, and apply them to the model sequentially.
    """

    def __init__(self, cosmos_config: CosmosConfig, parallel_dims: ParallelDims):
        from cosmos_rl.utils.model_converter import (
            _QUANTIZATION_CONVERTER_MODULE_TYPE_REGISTRY,
        )

        quantization_type = cosmos_config.train.quantization.quantization_type
        converter_classes = []
        for (
            _,
            module_type_registry,
        ) in _QUANTIZATION_CONVERTER_MODULE_TYPE_REGISTRY.items():
            if quantization_type in module_type_registry:
                converter_classes.append(module_type_registry[quantization_type])
        self.converters = [
            converter_cls(cosmos_config, parallel_dims)
            for converter_cls in converter_classes
        ]

    def convert(self, model: nn.Module):
        for converter in self.converters:
            model = converter.convert_model(model)
        return model

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]):
        for converter in self.converters:
            converter.post_optimizer_hook(model)
