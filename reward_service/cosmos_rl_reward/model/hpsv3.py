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
#
# HPSv3 reward integration using the upstream `hpsv3` package and model weights from:
#   MizzenAI/HPSv3 (https://github.com/MizzenAI/HPSv3)

from __future__ import annotations

import os
import time
from typing import List

import torch
from PIL import Image

from cosmos_rl_reward.handler.reward_base import BaseRewardHandler
from cosmos_rl_reward.handler.registry import RewardRegistry
from cosmos_rl_reward.utils.logging import logger


class HPSv3Scorer:

    def __init__(
        self,
        device: str = "cuda",
        config_path: str | None = None,
        checkpoint_path: str | None = None,
        download_path: str | None = None,
        differentiable: bool = False,
    ):
        from hpsv3 import HPSv3RewardInferencer

        if checkpoint_path is None and download_path:
            cand = os.path.join(download_path, "hpsv3", "HPSv3.safetensors")
            if os.path.exists(cand):
                checkpoint_path = cand

        kwargs = {"device": device, "differentiable": differentiable}
        if config_path is not None:
            kwargs["config_path"] = config_path
        if checkpoint_path is not None:
            kwargs["checkpoint_path"] = checkpoint_path

        self.inferencer = HPSv3RewardInferencer(**kwargs)

    @torch.inference_mode()
    def score(self, prompts: List[str], images_uint8_nhwc: torch.Tensor) -> List[float]:
        pil_images: List[Image.Image] = [
            Image.fromarray(images_uint8_nhwc[i].contiguous().cpu().numpy()).convert("RGB")
            for i in range(images_uint8_nhwc.shape[0])
        ]

        rewards = self.inferencer.reward(prompts=prompts, image_paths=pil_images)
        # HPSv3 returns (miu, sigma) per sample by default. We use miu (index 0).
        if isinstance(rewards, torch.Tensor) and rewards.ndim == 2 and rewards.shape[-1] >= 1:
            mu = rewards[:, 0]
        else:
            mu = rewards
        return mu.float().cpu().tolist()

@RewardRegistry.register()
class HPSv3Reward(BaseRewardHandler):
    NEEDS_LATENT_DECODER = False
    reward_name = "hpsv3"

    def __init__(
        self,
        device: str = "cuda",
        dtype: str | torch.dtype = "float16",
        download_path: str = "",
        config_path: str | None = None,
        checkpoint_path: str | None = None,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype 
        self.download_path = download_path
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.model: HPSv3Scorer | None = None

    def set_up(self):
        try:
            self.model = HPSv3Scorer(
                device=self.device,
                config_path=self.config_path,
                checkpoint_path=self.checkpoint_path,
                download_path=self.download_path,
                differentiable=False,
            )
            logger.info("[hpsv3] Initialization complete.")
        except Exception as e:
            logger.error(f"[hpsv3] Failed to initialize: {e}")
            self.model = None

    def calculate_reward(self, images, metadata):
        time_start = time.time()

        def _error(msg: str):
            logger.error(msg)
            return {
                "error": msg,
                "scores": None,
                "input_info": metadata.get("input_info", {}),
                "duration": f"{time.time() - time_start:.2f}",
                "decoded_duration": metadata.get("decode_duration", "N/A"),
                "type": self.reward_name,
            }

        if self.model is None:
            return _error("[hpsv3] model is not initialized. Check setup logs / dependencies.")
        if images is None or not isinstance(images, torch.Tensor):
            return _error(f"[hpsv3] expects torch.Tensor in BHWC/NHWC layout; got type={type(images)}")
        if images.dim() != 4:
            return _error(f"[hpsv3] expects 4D tensor (B,H,W,C) or (B,C,H,W); got shape={getattr(images,'shape',None)}")
        if images.shape[0] == 0:
            return _error("[hpsv3] batch size is zero.")

        prompts = metadata.get("prompts")
        if prompts is None:
            return _error("[hpsv3] prompts are required and cannot be None.")
        if isinstance(prompts, str):
            prompts = [prompts]
        if len(prompts) != images.shape[0]:
            return _error(f"[hpsv3] prompts length ({len(prompts)}) must match batch size ({images.shape[0]}).")

        x = images
        if x.dtype != torch.uint8:
            x = x.to(torch.uint8)

        if x.shape[-1] == 3:
            x_nhwc = x
        elif x.shape[1] == 3:
            x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        else:
            return _error(f"[hpsv3] channel dim must be 3, got shape={x.shape}")

        x_nhwc = x_nhwc.contiguous()
        try:
            scores = self.model.score(prompts, x_nhwc)
        except Exception as e:
            return _error(f"[hpsv3] inference failed: {e}")

        return {
            "scores": {"hpsv3": scores},
            "input_info": metadata.get("input_info", {}),
            "duration": f"{time.time() - time_start:.2f}",
            "decoded_duration": metadata.get("decode_duration", "N/A"),
            "type": self.reward_name,
        }


