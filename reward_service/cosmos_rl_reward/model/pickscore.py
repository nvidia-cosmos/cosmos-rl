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
# PickScore model from https://github.com/yuvalkirstain/PickScore

import os
from typing import Any, List

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from cosmos_rl_reward.handler.reward_base import BaseRewardHandler
from cosmos_rl_reward.handler.registry import RewardRegistry
from cosmos_rl_reward.utils.logging import logger


class PickScoreScorer(torch.nn.Module):

    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype | str = "float32",
        processor_path: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        model_path: str = "yuvalkirstain/PickScore_v1",
    ):
        super().__init__()
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.device = device
        self.dtype = dtype
        cache_root = os.environ.get("HF_HOME", os.path.expanduser("~/.cache"))
        self.processor = AutoProcessor.from_pretrained(processor_path, cache_dir=cache_root)
        self.model = AutoModel.from_pretrained(model_path, cache_dir=cache_root).eval().to(device)
        self.model = self.model.to(dtype=dtype)

    @torch.no_grad()
    def forward(self, prompts: List[str], images: List[Image.Image]) -> torch.Tensor:
        def _extract_features(features: Any) -> torch.Tensor:
            # HF models may return tensors or ModelOutput objects (e.g., BaseModelOutputWithPooling).
            if isinstance(features, torch.Tensor):
                return features
            for attr in ("image_embeds", "text_embeds", "embeds", "pooler_output"):
                if hasattr(features, attr):
                    value = getattr(features, attr)
                    if isinstance(value, torch.Tensor):
                        return value
            if isinstance(features, (tuple, list)) and len(features) > 0 and isinstance(features[0], torch.Tensor):
                return features[0]
            raise TypeError(
                f"[pickscore] expected tensor-like features; got {type(features)} with attrs={dir(features)}"
            )

        if len(prompts) != len(images):
            raise ValueError(f"[pickscore] prompts length ({len(prompts)}) must match images length ({len(images)}).")
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = {k: v.to(device=self.device) for k, v in image_inputs.items()}
        text_inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(device=self.device) for k, v in text_inputs.items()}

        image_features = self.model.get_image_features(**image_inputs)
        image_embs = _extract_features(image_features)
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)

        text_features = self.model.get_text_features(**text_inputs)
        text_embs = _extract_features(text_features)
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)

        logit_scale = self.model.logit_scale.exp()
        scores = logit_scale * (text_embs @ image_embs.T)
        scores = scores.diag() / 26
        return scores.detach()


@RewardRegistry.register()
class PickScoreReward(BaseRewardHandler):
    NEEDS_LATENT_DECODER = False
    reward_name = "pickscore"

    def __init__(self, dtype: str | torch.dtype = "float32", device: str = "cuda", **kwargs):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.model: PickScoreScorer | None = None

    def set_up(self):
        self.model = PickScoreScorer(device=self.device, dtype=self.dtype)

    def calculate_reward(self, images, metadata):
        import time

        def _error(msg: str):
            logger.error(msg)
            return {
                "error": msg,
                "scores": None,
                "input_info": metadata.get("input_info", {}),
                "duration": "0.00",
                "decoded_duration": metadata.get("decode_duration", "N/A"),
                "type": self.reward_name,
            }

        if images is None or not isinstance(images, torch.Tensor):
            return _error(f"[pickscore] expects torch.Tensor in BHWC/NHWC layout; got type={type(images)}")
        if images.dim() != 4:
            return _error(f"[pickscore] expects 4D tensor (B,H,W,C); got shape={getattr(images,'shape',None)}")
        if images.shape[0] == 0:
            return _error("[pickscore] batch size is zero.")

        prompts = metadata.get("prompts")
        if prompts is None:
            return _error("[pickscore] prompts are required and cannot be None.")
        if isinstance(prompts, str):
            prompts = [prompts]

        x = images
        if x.dtype != torch.uint8:
            x = x.to(torch.uint8)
        if x.shape[-1] == 3:
            x_nhwc = x
        elif x.shape[1] == 3:
            x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        else:
            return _error(f"[pickscore] channel dim must be 3, got shape {x.shape}")

        pil_images = [Image.fromarray(x_nhwc[i].cpu().numpy()) for i in range(x_nhwc.shape[0])]
        if len(prompts) != len(pil_images):
            return _error(f"[pickscore] prompts length ({len(prompts)}) must match batch size ({len(pil_images)}).")

        start_time = time.time()
        scores = self.model(prompts, pil_images).float().cpu().tolist()
        duration = f"{time.time() - start_time:.2f}"
        return {
            "scores": {"pickscore": scores},
            "input_info": metadata.get("input_info", {}),
            "duration": duration,
            "decoded_duration": metadata.get("decode_duration", "N/A"),
            "type": self.reward_name,
        }

