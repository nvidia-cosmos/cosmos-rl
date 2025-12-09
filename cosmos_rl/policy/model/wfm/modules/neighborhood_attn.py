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
from collections.abc import Mapping, Sized
from typing import Optional

import torch
from torch import nn

from cosmos_rl.utils.logging import logger

try:
    import natten
    from natten.functional import neighborhood_attention_generic

    natten.use_kv_parallelism_in_fused_na(True)
    natten.set_memory_usage_preference("unrestricted")

    HAS_NATTEN = True

except ImportError:
    HAS_NATTEN = False

    def neighborhood_attention_generic(*args, **kwargs):
        raise RuntimeError(
            "You attempted to run neighborhood attention, but you don't have NATTEN installed."
        )


def get_device_cc(device: Optional[torch.device] = None) -> int:
    if (
        torch.cuda.is_available()
        and torch.version.cuda
        and (device is None or device.type == "cuda")
    ):
        major, minor = torch.cuda.get_device_capability(device)
        return major * 10 + minor

    return 0


# TODO (ahassani): should we move VideoSize definitions to a common place?
from collections import namedtuple

VideoSize = namedtuple("VideoSize", ["T", "H", "W"])


class NeighborhoodAttention(nn.Module):
    def __init__(self, gna_parameters, base_attn_op):
        super(NeighborhoodAttention, self).__init__()

        self.base_attn_op = base_attn_op

        self.gna_parameters = gna_parameters
        if (
            not isinstance(gna_parameters, Mapping)
            or "window_size" not in gna_parameters
            or "layer_id" not in gna_parameters
        ):
            raise ValueError(
                f"Expected `gna_parameters` to be a dict with at least keys `window_size` and `layer_id`, got {gna_parameters=}."
            )

        self.layer_id = gna_parameters["layer_id"]
        self.window_size = gna_parameters["window_size"]
        self.stride = 1 if "stride" not in gna_parameters else gna_parameters["stride"]
        self.dilation = (
            1 if "dilation" not in gna_parameters else gna_parameters["dilation"]
        )
        self.is_causal = (
            False if "is_causal" not in gna_parameters else gna_parameters["is_causal"]
        )

        if not isinstance(self.window_size, Sized) or len(self.window_size) != 3:
            raise ValueError(
                "Invalid window_size value. Expected an iterable of length 3, got "
                f"{self.window_size}."
            )

        if (
            not isinstance(self.stride, Sized) or len(self.stride) != 3
        ) and not isinstance(self.stride, int):
            raise ValueError(
                "Invalid stride value. Expected an iterable of length 3, or integer, got "
                f"{self.stride}."
            )

        if (
            not isinstance(self.dilation, Sized) or len(self.dilation) != 3
        ) and not isinstance(self.dilation, int):
            raise ValueError(
                "Invalid dilation value. Expected an iterable of length 3, or integer, got "
                f"{self.dilation}."
            )

        if (
            not isinstance(self.is_causal, Sized) or len(self.is_causal) != 3
        ) and not isinstance(self.is_causal, bool):
            raise ValueError(
                "Invalid is_causal value. Expected an iterable of length 3, or boolean, got "
                f"{self.is_causal}."
            )

        logger.info(
            f"NeighborhoodAttention op registered for layer {self.layer_id}, with "
            f"window_size={self.window_size}, stride={self.stride}, dilation={self.dilation}."
        )

        self.base_size = (
            None if "base_size" not in gna_parameters else gna_parameters["base_size"]
        )
        if self.base_size is not None and (
            not isinstance(self.base_size, Sized) or len(self.base_size) != 3
        ):
            raise ValueError(
                "Invalid base feature map size. Expected an iterable of length 3, or None, got "
                f"{self.base_size}."
            )

        # Configurations
        # NOTE: these have been tuned for 720p, window size (T, 12, 24),
        # stride (1, 4, 8).
        # They also assume head dim = 128.
        self.performance_configs = {
            80: {
                "backend": "cutlass-fna",
                "q_tile_shape": (4, 4, 4),
                "kv_tile_shape": (4, 4, 8),
                "backward_q_tile_shape": (4, 4, 8),
                "backward_kv_tile_shape": (4, 4, 8),
                "backward_use_pt_reduction": False,
            },
            90: {
                "backend": "hopper-fna",
                "q_tile_shape": (4, 4, 8),
                "kv_tile_shape": (4, 4, 8),
                "backward_q_tile_shape": (4, 4, 4),
                "backward_kv_tile_shape": (4, 4, 8),
            },
            100: {
                "backend": "blackwell-fna",
                "q_tile_shape": (8, 4, 8),
                "kv_tile_shape": (4, 4, 8),
                "backward_q_tile_shape": (4, 4, 8),
                "backward_kv_tile_shape": (4, 4, 8),
                "run_persistent_kernel": True,
            },
        }

    def get_na_parameters(
        self, window_size, stride, dilation, is_causal, input_shape, base_size=None
    ):
        window_size = tuple(w if w > 1 else x for x, w in zip(input_shape, window_size))
        stride = (
            tuple(stride for _ in range(3))
            if isinstance(stride, int)
            else tuple(x for x in stride)
        )
        dilation = (
            tuple(dilation for _ in range(3))
            if isinstance(dilation, int)
            else tuple(x for x in dilation)
        )
        is_causal = (
            tuple(is_causal for _ in range(3))
            if isinstance(is_causal, bool)
            else tuple(x for x in is_causal)
        )

        # Scale window size and stride according to some base input size
        # For example, if window size is (8, 8, 8), stride is (1, 2, 2), for a base
        # input/feature map size of (16, 16, 16); then if the input feat map in this iteration
        # has shape (8, 8, 8), we should use window size (4, 4, 4), and stride (1, 1, 1).
        if base_size is not None:
            base_shape = tuple(
                b if b > 0 else x for x, b in zip(input_shape, base_size)
            )

            scale = tuple(x / b for x, b in zip(input_shape, base_shape))

            scaled_window_size = tuple(
                min(max(2, round(w * s)), x)
                for w, s, x in zip(window_size, scale, input_shape)
            )
            scaled_stride = tuple(
                min(max(1, round(st * s)), w)
                for w, s, st in zip(scaled_window_size, scale, stride)
            )

            max_dilation = tuple(
                x // w for x, w in zip(input_shape, scaled_window_size)
            )
            scaled_dilation = tuple(
                min(max(1, round(d * s)), max_d)
                for d, s, max_d in zip(dilation, scale, max_dilation)
            )

            window_size = scaled_window_size
            stride = scaled_stride
            dilation = scaled_dilation

        assert all(x >= w * d for x, w, d in zip(input_shape, window_size, dilation))
        assert all(w >= s for w, s in zip(window_size, stride))
        assert all(isinstance(c, bool) for c in is_causal)

        return window_size, stride, dilation, is_causal

    def forward(
        self,
        q_B_L_H_D: torch.Tensor,
        k_B_L_H_D: torch.Tensor,
        v_B_L_H_D: torch.Tensor,
        video_size: Optional[VideoSize] = None,
    ):
        if not (q_B_L_H_D.shape == k_B_L_H_D.shape == v_B_L_H_D.shape):
            raise ValueError(
                "NATTEN requires QKV shapes to match, got "
                f"{q_B_L_H_D.shape=}, {k_B_L_H_D.shape=}, {v_B_L_H_D.shape=}."
            )

        device = q_B_L_H_D.device
        device_cc = get_device_cc(device)
        requires_grad = (
            q_B_L_H_D.requires_grad
            or k_B_L_H_D.requires_grad
            or v_B_L_H_D.requires_grad
        )
        is_cuda = (
            torch.cuda.is_available() and torch.version.cuda and device.type == "cuda"
        )

        natten_configuration = None
        assert 80 in self.performance_configs.keys()
        if is_cuda and device_cc in self.performance_configs.keys():
            natten_configuration = self.performance_configs[device_cc]
        elif is_cuda and device_cc >= 80:
            natten_configuration = self.performance_configs[80]
        else:
            raise ValueError(
                f"No NATTEN config found for this use case: {requires_grad=}, {is_cuda=}, {device_cc=}."
            )

        batch, seqlen, heads, head_dim = q_B_L_H_D.shape
        T, H, W = video_size

        if seqlen != T * H * W:
            raise ValueError(
                "Mismatch between seqlen and video_size dimensions; got "
                f"{video_size=}, {seqlen=}."
            )

        if T > 1:
            input_shape = (T, H, W)

            window_size, stride, dilation, is_causal = self.get_na_parameters(
                window_size=self.window_size,
                stride=self.stride,
                dilation=self.dilation,
                is_causal=self.is_causal,
                input_shape=input_shape,
                base_size=self.base_size,
            )

        elif T == 1:
            # Do self attention for image model; skip natten
            return self.base_attn_op(q_B_L_H_D, k_B_L_H_D, v_B_L_H_D)

        else:
            raise ValueError(f"Invalid dimension {T=}.")

        q = q_B_L_H_D.view(batch, *input_shape, heads, head_dim)
        k = k_B_L_H_D.view(batch, *input_shape, heads, head_dim)
        v = v_B_L_H_D.view(batch, *input_shape, heads, head_dim)

        out = neighborhood_attention_generic(
            query=q,
            key=k,
            value=v,
            kernel_size=window_size,
            stride=stride,
            dilation=dilation,
            is_causal=is_causal,
            **natten_configuration,
        )

        return out.view(batch, seqlen, heads, head_dim)
