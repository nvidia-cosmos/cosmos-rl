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
from typing import Union, List
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
