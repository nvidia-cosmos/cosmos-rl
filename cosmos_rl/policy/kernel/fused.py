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
from cosmos_rl.utils.logging import logger
from transformers.activations import ACT2CLS


class MLPFusedActMul(nn.Module):
    def __init__(self, fused_fn: torch.autograd.Function):
        super().__init__()
        self.fused_fn = fused_fn

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.fused_fn.apply(a, b)


class MLPActMulFunc(nn.Module):
    def __init__(self, act_fn: nn.Module):
        super().__init__()
        self.act_fn = act_fn

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.act_fn(a) * b

    def liger_equivalent(self):
        if isinstance(self.act_fn, ACT2CLS["silu"]):
            from liger_kernel.ops.swiglu import LigerSiLUMulFunction

            return MLPFusedActMul(LigerSiLUMulFunction)
        elif isinstance(self.act_fn, ACT2CLS["gelu"]):
            from liger_kernel.ops.geglu import LigerGELUMulFunction

            return MLPFusedActMul(LigerGELUMulFunction)
        else:
            logger.warning(
                f"Unsupported activation function: {self.act_fn}. Will fallback to the default implementation."
            )
            return self
