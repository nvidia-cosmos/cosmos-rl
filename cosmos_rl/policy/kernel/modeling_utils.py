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
from functools import partial

from cosmos_rl.utils.logging import logger


def decide_fa_version():
    major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
    if major >= 9:
        return 3
    return 2


class FlashAttnMeta:
    _flash_attn_meta = None

    def __init__(self):
        self.fa_version = decide_fa_version()
        if self.fa_version == 3:
            try:
                # Just as a check to see if flash_attn_3 is installed
                import flash_attn_3  # noqa: F401

                # According to: https://github.com/Dao-AILab/flash-attention/blob/add175637c5d54b74bc25372e49ce282d6f236fc/README.md?plain=1#L62
                from flash_attn_interface import flash_attn_func, flash_attn_varlen_func
            except ImportError:
                logger.warning(
                    "FlashAttention3 is not installed. Using FlashAttention2 instead."
                )
                self.fa_version = 2
                from flash_attn import flash_attn_func, flash_attn_varlen_func
        else:
            from flash_attn import flash_attn_func, flash_attn_varlen_func
        logger.info(f"[Cosmos-RL] Using FlashAttention-{self.fa_version}.")
        self.flash_attn_func = flash_attn_func
        self.flash_attn_varlen_func = flash_attn_varlen_func
        from flash_attn.layers.rotary import apply_rotary_emb as ori_apply_rotary_emb

        self.apply_rotary_emb = ori_apply_rotary_emb

    def __new__(cls):
        if cls._flash_attn_meta is None:
            cls._flash_attn_meta = super().__new__(cls)
        return cls._flash_attn_meta

    def set_deterministic(self, deterministic: bool):
        self.flash_attn_func = partial(
            self.flash_attn_func, deterministic=deterministic
        )
        self.flash_attn_varlen_func = partial(
            self.flash_attn_varlen_func, deterministic=deterministic
        )


def init_flash_attn_meta(deterministic: bool = False):
    FlashAttnMeta().set_deterministic(deterministic)
    logger.info(f"[Cosmos-RL] FlashAttnMeta initialized to {FlashAttnMeta()}.")
