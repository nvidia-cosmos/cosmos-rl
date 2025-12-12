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

import numpy as np
import torch
import re
from typing import List, Tuple

IMAGE_RES_SIZE_INFO: dict[str, tuple[int, int]] = {
    "1080": {
        "1,1": (1024, 1024),
        "4,3": (1440, 1056),
        "3,4": (1056, 1440),
        "16,9": (1920, 1056),
        "9,16": (1056, 1920),
    },
    # "1024": {"1,1": (1024, 1024), "4,3": (1280, 1024), "3,4": (1024, 1280), "16,9": (1280, 768), "9,16": (768, 1280)},
    "1024": {
        "1,1": (1024, 1024),
        "4,3": (1168, 880),
        "3,4": (880, 1168),
        "16,9": (1360, 768),
        "9,16": (768, 1360),
    },
    "720": {
        "1,1": (960, 960),
        "4,3": (960, 704),
        "3,4": (704, 960),
        "16,9": (1280, 704),
        "9,16": (704, 1280),
    },
    "512": {
        "1,1": (512, 512),
        "4,3": (640, 512),
        "3,4": (512, 640),
        "16,9": (640, 384),
        "9,16": (384, 640),
    },
    "480": {
        "1,1": (480, 480),
        "4,3": (640, 480),
        "3,4": (480, 640),
        "16,9": (768, 432),
        "9,16": (432, 768),
    },
    "480p": {
        "1,1": (640, 640),
        "4,3": (640, 480),
        "3,4": (480, 640),
        "16,9": (832, 480),
        "9,16": (480, 832),
    },
    "720robocasa": {
        "1,1": (720, 720),
        "4,3": (960, 720),
        "3,4": (720, 960),
        "16,9": (1280, 720),
        "9,16": (720, 1280),
    },
    "256": {
        "1,1": (256, 256),
        "4,3": (320, 256),
        "3,4": (256, 320),
        "16,9": (320, 192),
        "9,16": (192, 320),
    },
}


VIDEO_RES_SIZE_INFO: dict[str, tuple[int, int]] = {
    "1080": {
        "1,1": (1024, 1024),
        "4,3": (1440, 1072),
        "3,4": (1072, 1440),
        "16,9": (1920, 1072),
        "9,16": (1072, 1920),
    },
    "1024": {
        "1,1": (1024, 1024),
        "4,3": (1280, 1024),
        "3,4": (1024, 1280),
        "16,9": (1280, 768),
        "9,16": (768, 1280),
    },
    "720": {
        "1,1": (960, 960),
        "4,3": (960, 704),
        "3,4": (704, 960),
        "16,9": (1280, 704),
        "9,16": (704, 1280),
    },
    "512": {
        "1,1": (512, 512),
        "4,3": (640, 512),
        "3,4": (512, 640),
        "16,9": (640, 384),
        "9,16": (384, 640),
    },
    "480": {
        "1,1": (480, 480),
        "4,3": (640, 480),
        "3,4": (480, 640),
        "16,9": (768, 432),
        "9,16": (432, 768),
    },
    # 720, 1280 is Wan2.1 specs
    "480p": {
        "1,1": (640, 640),
        "4,3": (640, 480),
        "3,4": (480, 640),
        "16,9": (832, 480),
        "9,16": (480, 832),
    },
    "720p": {
        "1,1": (960, 960),
        "4,3": (960, 720),
        "3,4": (720, 960),
        "16,9": (1280, 720),
        "9,16": (720, 1280),
    },
    "720robocasa": {
        "1,1": (720, 720),
        "4,3": (960, 720),
        "3,4": (720, 960),
        "16,9": (1280, 720),
        "9,16": (720, 1280),
    },
    "256": {
        "1,1": (256, 256),
        "4,3": (320, 256),
        "3,4": (256, 320),
        "16,9": (320, 192),
        "9,16": (192, 320),
    },
}


def get_aspect_ratios_from_wdinfos(wdinfos: list[str]) -> list[str]:
    aspect_ratios = []
    for wdinfo in wdinfos:
        aspect_ratio_match = re.search(r"aspect_ratio_(\d+_\d+)", wdinfo)
        aspect_ratios.append(aspect_ratio_match.group(1))

    return aspect_ratios


def get_wdinfos_w_aspect_ratio(wdinfos: list[str]) -> List[Tuple[str, str]]:
    aspect_ratios = get_aspect_ratios_from_wdinfos(wdinfos)

    # return a list of (wdinfo_path, aspect_ratio) pairs
    return [
        (wdinfo, aspect_ratio.replace("_", ","))
        for wdinfo, aspect_ratio in zip(wdinfos, aspect_ratios)
    ]


def pad_and_resize(
    arr_np: np.ndarray, ntokens: int, is_mask_all_ones: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Function for padding and resizing a numpy array.
    Args:
        arr (np.ndarray): Input array
        ntokens (int): Number of output tokens after padding
        is_mask_all_ones (bool): if true, set mask to ones
    Returns:
        arr_padded (torch.Tensor): Padded output tensor
        mask (torch.Tensor): Padding mask
    """

    if isinstance(arr_np, np.ndarray):
        arr = torch.from_numpy(arr_np)
    elif isinstance(arr_np, torch.Tensor):
        arr = arr_np.clone().detach()
    else:
        raise TypeError("`arr_np` should be a numpy array or torch tensor.")
    embed_dim = arr.shape[1]

    arr_padded = torch.zeros(ntokens, embed_dim, device=arr.device, dtype=torch.float32)

    # If the input text is larger than num_text_tokens, clip it.
    if arr.shape[0] > ntokens:
        arr = arr[0:ntokens]

    mask = torch.LongTensor(ntokens).zero_()
    if len(arr.shape) > 1:
        mask[0 : arr.shape[0]] = 1

    if len(arr.shape) > 1:
        arr_padded[0 : arr.shape[0]] = arr

    if is_mask_all_ones:
        mask.fill_(1)

    return arr_padded, mask
