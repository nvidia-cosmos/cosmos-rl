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
from typing import Optional

from cosmos_rl.utils.logging import logger


def decide_fa_version():
    major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
    if major >= 9:
        return 3
    return 2


def lastdim_contig(x: torch.Tensor) -> torch.Tensor:
    return x if x is None or x.stride(-1) == 1 else x.contiguous()


class FlashAttnMetaSingleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(FlashAttnMetaSingleton, cls).__call__(
                *args, **kwargs
            )
        return cls._instances[cls]


class FlashAttnMeta(metaclass=FlashAttnMetaSingleton):
    def __init__(
        self,
        torch_compile: bool = True,
        user_specified_fa_version: Optional[int] = None,
        enable_fp4: bool = False,
    ):
        # FA3 is not compatible with torch.compile
        # The support is in WIP: https://github.com/Dao-AILab/flash-attention/pull/1769
        self.fa_version = decide_fa_version()
        if user_specified_fa_version is not None:
            # Respect the user's specified version
            self.fa_version = user_specified_fa_version

        if self.fa_version == 3 and torch_compile:
            logger.warning(
                "FlashAttention3 is not compatible with torch.compile. Using FlashAttention2 instead."
            )
            self.fa_version = 2

        if self.fa_version == 3:
            try:
                # Just as a check to see if flash_attn_3 is installed
                import flash_attn_3  # noqa: F401

                # According to: https://github.com/Dao-AILab/flash-attention/blob/add175637c5d54b74bc25372e49ce282d6f236fc/README.md?plain=1#L62
                from flash_attn_interface import flash_attn_func, flash_attn_varlen_func

                self.fa_version = 3
            except ImportError:
                logger.warning(
                    "FlashAttention3 is not installed. Using FlashAttention2 instead."
                )
                self.fa_version = 2
                from flash_attn import flash_attn_func, flash_attn_varlen_func
        else:
            from flash_attn import flash_attn_func, flash_attn_varlen_func
        logger.info(f"[Cosmos-RL] Using FlashAttention-{self.fa_version}.")

        def _flash_attn_func(q, k, v, *args, **kwargs):
            if enable_fp4:
                q = lastdim_contig(q)
                k = lastdim_contig(k)
                v = lastdim_contig(v)
            return flash_attn_func(q, k, v, *args, **kwargs)

        def _flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            *args,
            **kwargs,
        ):
            if enable_fp4:
                q = lastdim_contig(q)
                k = lastdim_contig(k)
                v = lastdim_contig(v)
                cu_seqlens_q = cu_seqlens_q.to(torch.int32).contiguous()
                cu_seqlens_k = cu_seqlens_k.to(torch.int32).contiguous()
            return flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                *args,
                **kwargs,
            )

        self.flash_attn_func = _flash_attn_func
        self.flash_attn_varlen_func = _flash_attn_varlen_func
        from flash_attn.layers.rotary import apply_rotary_emb as ori_apply_rotary_emb

        self.apply_rotary_emb = ori_apply_rotary_emb

    def set_deterministic(self, deterministic: bool):
        self.flash_attn_func = partial(
            self.flash_attn_func, deterministic=deterministic
        )
        self.flash_attn_varlen_func = partial(
            self.flash_attn_varlen_func, deterministic=deterministic
        )


def init_flash_attn_meta(
    deterministic: bool = False,
    compile: bool = True,
    fa_version: Optional[int] = None,
    enable_fp4: bool = False,
):
    FlashAttnMeta(
        torch_compile=compile,
        user_specified_fa_version=fa_version,
        enable_fp4=enable_fp4,
    ).set_deterministic(deterministic)
