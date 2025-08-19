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
from typing import Optional


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, casting_mode: Optional[str] = None):
        """
        RMSNorm is equivalent to T5LayerNorm
        casting_mode:
            - "bf16": cast to bf16
            - "fp16": cast to fp16
            - "fp32": cast to fp32
            - None: no casting
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.casting_mode = casting_mode

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

    def liger_equivalent(self):
        from liger_kernel.transformers.rms_norm import (
            LigerRMSNorm,
            LigerRMSNormForGemma,
            LigerRMSNormForGemma2,
            LigerRMSNormForGemma3,
        )

        casting_mode_to_liger_norm = {
            "llama": LigerRMSNorm,
            "deepseek_v3": LigerRMSNorm,
            "qwen": LigerRMSNorm,
            "qwen2": LigerRMSNorm,
            "qwen2_5_vl": LigerRMSNorm,
            "qwen3": LigerRMSNorm,
            "qwen3_moe": LigerRMSNorm,
            "gemma": LigerRMSNormForGemma,
            "gemma_2": LigerRMSNormForGemma2,
            "gemma_3": LigerRMSNormForGemma3,
        }
        assert self.casting_mode in casting_mode_to_liger_norm.keys()
        return casting_mode_to_liger_norm[self.casting_mode](
            hidden_size=self.weight.shape[0],
            eps=self.variance_epsilon,
        )
