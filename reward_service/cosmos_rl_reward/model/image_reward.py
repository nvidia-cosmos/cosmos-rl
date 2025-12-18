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
# Portions of this file are adapted from NVLabs DiffusionNFT (https://github.com/NVlabs/DiffusionNFT)
# ImageReward model from THUDM (https://github.com/THUDM/ImageReward)


import os
from cosmos_rl_reward.handler.reward_base import BaseRewardHandler
from cosmos_rl_reward.handler.registry import RewardRegistry
from cosmos_rl_reward.utils.logging import logger




@RewardRegistry.register()
class ImageReward(BaseRewardHandler):
    NEEDS_LATENT_DECODER = False
    reward_name = "image_reward"


    def __init__(self, dtype="float32", device="cuda", model_path="/workspace", **kwargs):
        super().__init__()
        self.model = None
        self.dtype = dtype
        self.device = device
        self.model_path = model_path


    def set_up(self):
        import torch
        import ImageReward as RM
        cache_root = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache")), "ImageReward")
        if isinstance(self.dtype, str):
            self.dtype = getattr(torch, self.dtype)
        self.model = RM.load("ImageReward-v1.0", device=self.device, download_root=cache_root).eval().to(dtype=self.dtype)
        self.model.requires_grad_(False)


    def calculate_reward(self, images, metadata):
        import torch
        from PIL import Image
        start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if start is not None:
            start.record()
        if not isinstance(images, torch.Tensor) or images.dim() != 4:
            raise ValueError(f"image_reward expects 4D torch.Tensor, got {type(images)} {getattr(images,'shape',None)}")
        x = images
        if x.dtype != torch.uint8:
            x = x.to(torch.uint8)
        if x.shape[-1] == 3:
            x_nhwc = x
        elif x.shape[1] == 3:
            x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        else:
            raise ValueError(f"channel dim must be 3, got shape {x.shape}")
        pil_images = [Image.fromarray(x_nhwc[i].cpu().numpy()) for i in range(x_nhwc.shape[0])]
        prompts = metadata.get("prompts", [""] * len(pil_images))
        if isinstance(prompts, str):
            prompts = [prompts] * len(pil_images)
        if len(prompts) != len(pil_images):
            prompts = [prompts[0]] * len(pil_images)
        _, m = self.model.inference_rank(prompts, pil_images)
        diag = torch.tensor(m, device=self.device, dtype=self.dtype).reshape(len(prompts), len(prompts)).diagonal(0).float().cpu().tolist()
        if end is not None:
            end.record()
            torch.cuda.synchronize()
            duration = f"{start.elapsed_time(end)/1000.0:.2f}"
        else:
            duration = "0.00"
        return {
            "scores": {"image_reward": diag},
            "input_info": metadata.get("input_info", {}),
            "duration": duration,
            "decoded_duration": metadata.get("decode_duration", "N/A"),
            "type": self.reward_name,
        }