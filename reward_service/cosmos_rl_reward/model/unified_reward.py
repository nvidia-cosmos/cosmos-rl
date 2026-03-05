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
# Portions of this file are adapted from flowgrpo (https://github.com/yifan123/flow_grpo)

import base64
import re
import time
from io import BytesIO
from cosmos_rl_reward.handler.reward_base import BaseRewardHandler
from cosmos_rl_reward.handler.registry import RewardRegistry
from cosmos_rl_reward.utils.logging import logger


@RewardRegistry.register()
class UnifiedReward(BaseRewardHandler):
    NEEDS_LATENT_DECODER = False
    reward_name = "unified_reward"

    def __init__(self, endpoint_url="", endpoint_api_key="", **kwargs):
        super().__init__()
        self.endpoint_url = endpoint_url
        self.endpoint_api_key = endpoint_api_key

    def set_up(self):
        from openai import OpenAI

        self.client = OpenAI(api_key=self.endpoint_api_key, base_url=self.endpoint_url)

    def pil_image_to_base64(self, image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        return base64_qwen

    def _extract_scores(self, text_outputs):
        scores = []
        pattern = r"Final Score:\s*([1-5](?:\.\d+)?)"
        for text in text_outputs:
            match = re.search(pattern, text)
            if match:
                try:
                    scores.append(float(match.group(1)))
                except ValueError:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return scores

    def evaluate_image(self, prompt, image):
        question = f"<image>\nYou are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nBased on the above criteria, assign a score from 1 to 5 after 'Final Score:'.\nYour task is provided as follows:\nText Caption: [{prompt}]"
        images_base64 = self.pil_image_to_base64(image)
        response = self.client.chat.completions.create(
            model="UnifiedReward-7b-v1.5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": images_base64},
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                },
            ],
            temperature=0,
            stream=False,
        )
        return response.choices[0].message.content

    def evaluate_batch_image(self, images, prompts):
        results = [
            self.evaluate_image(prompt, img) for prompt, img in zip(prompts, images)
        ]
        return results

    def calculate_reward(self, images, metadata):
        import torch
        from PIL import Image

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

        t0 = time.perf_counter()
        if images is None:
            return _error("[unified_reward] images tensor is None.")
        if not isinstance(images, torch.Tensor) or images.dim() != 4:
            return _error(
                f"[unified_reward] expects 4D torch.Tensor in BHWC/NHWC layout (B,H,W,C); got type={type(images)} shape={getattr(images, 'shape', None)} dtype={getattr(images, 'dtype', None)}"
            )
        if images.shape[0] == 0:
            return _error("[unified_reward] images batch size is zero.")
        x = images
        if x.dtype != torch.uint8:
            x = x.to(torch.uint8)
        if x.shape[-1] == 3:
            x_nhwc = x
        elif x.shape[1] == 3:
            x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        else:
            raise ValueError(f"channel dim must be 3, got shape {x.shape}")
        pil_images = [
            Image.fromarray(x_nhwc[i].cpu().numpy()) for i in range(x_nhwc.shape[0])
        ]
        prompts = metadata.get("prompts")
        if prompts is None:
            return _error("[unified_reward] prompts are required and cannot be None.")
        if isinstance(prompts, str):
            prompts = [prompts]
        if len(prompts) != len(pil_images):
            return _error(
                f"[unified_reward] prompts length ({len(prompts)}) must match batch size ({len(pil_images)})."
            )

        text_outputs = self.evaluate_batch_image(pil_images, prompts)
        scores = self._extract_scores(text_outputs)
        scores = [
            sc / 5.0 for sc in scores
        ]  # Normalize to [0,1] assuming the original score is in [0,5]
        duration = f"{(time.perf_counter() - t0):.2f}"
        return {
            "scores": {"unified_reward": scores},
            "input_info": metadata.get("input_info", {}),
            "duration": duration,
            "decoded_duration": metadata.get("decode_duration", "N/A"),
            "type": self.reward_name,
        }
