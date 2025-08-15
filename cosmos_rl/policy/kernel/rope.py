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

try:
    from liger_kernel.transformers.rope import liger_rotary_pos_emb
except ImportError:
    liger_rotary_pos_emb = None


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=2):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, use_liger: bool = False):
        super().__init__()
        if use_liger and liger_rotary_pos_emb is None:
            logger.warning(
                "`liger_kernel` is not installed. Will fallback to the default implementation of `apply_rotary_pos_emb`."
            )
            use_liger = False
        self.use_liger = use_liger
        self.func = liger_rotary_pos_emb if use_liger else apply_rotary_pos_emb

    def forward(self, q, k, cos, sin, position_ids=None, unsqueeze_dim=2):
        if self.use_liger:
            return liger_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim)
        else:
            return apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim)

    def liger_equivalent(self):
        self.func = (
            liger_rotary_pos_emb
            if liger_rotary_pos_emb is not None
            else apply_rotary_pos_emb
        )
        # Note: this is a module without parameters, so we can return it directly
        return self
