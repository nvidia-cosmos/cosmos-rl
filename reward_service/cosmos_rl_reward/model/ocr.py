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
# OCR functionality powered by PaddleOCR (https://github.com/PaddlePaddle/PaddleOCR)

import os
import time
from cosmos_rl_reward.utils.logging import logger
from cosmos_rl_reward.handler.reward_base import BaseRewardHandler
from cosmos_rl_reward.handler.registry import RewardRegistry
import numpy as np
from typing import List, Union



class OcrScorer:
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(lang="en", use_gpu=self.use_gpu, show_log=False)

    def __call__(
        self, images: Union[List["PILImage.Image"], List["np.ndarray"]], prompts: List[str]
    ):
        from PIL import Image as PILImage
        from Levenshtein import distance
        prompts = [prompt.split('"')[1] for prompt in prompts]
        assert len(images) == len(
            prompts
        ), "Images and prompts must have the same length"
        rewards: List[float] = []
        for img, prompt in zip(images, prompts):
            if isinstance(img, PILImage.Image):
                img = np.array(img)
            try:

                result = self.ocr.ocr(img)
                recognized_text = (
                    "".join([res[1][0] if res[1][1] > 0 else "" for res in result[0]])
                    if result and result[0]
                    else ""
                )
                recognized_text = recognized_text.replace(" ", "").lower()
                prompt_n = prompt.replace(" ", "").lower()

                hw = getattr(img, 'shape', None)
                logger.info(f"[OCR DEBUG] img_shape={hw} prompt='{prompt}' rec='{recognized_text}'")

                if prompt_n in recognized_text:
                    dist = 0
                else:
                    dist = distance(recognized_text, prompt_n)
                if dist > len(prompt_n):
                    dist = len(prompt_n)
            except Exception as e:
                logger.error(f"[ocr] OCR processing failed: {e}")
                dist = len(prompt)

            logger.info(f"[OCR DEBUG] dist={dist} len_prompt={len(prompt)} score={1 - dist / max(1, len(prompt))}")

            reward = 1 - dist / max(1, len(prompt))
            rewards.append(reward)
        return rewards


@RewardRegistry.register()
class OcrReward(BaseRewardHandler):
    NEEDS_LATENT_DECODER = False
    reward_name = "ocr"

    def __init__(self, device="cpu", **kwargs):
        super().__init__()
        self.device = device
        self.use_gpu = device == "cuda"

    def set_up(self):
        self.scorers = {}
        self.scorers[self.use_gpu] = OcrScorer(use_gpu=self.use_gpu)

    def calculate_reward(self, images, metadata):
        import torch 
        time_start = time.time()
        # Expect 4D image batch; client must provide BHWC/NHWC uint8 (B,H,W,C)
        if images.dim() != 4:
            raise ValueError(
                f"OCR expects 4D uint8 tensor in BHWC/NHWC layout (B,H,W,C); got shape={images.shape}, dtype={images.dtype}"
            )
        if images.dtype != torch.uint8:
            images = images.to(torch.uint8)
        images_np = images.contiguous().cpu().numpy()
        images_list = [frame for frame in images_np]
        prompts = metadata.get("prompts", [""] * len(images_list))
        # If batch==1 and prompts is a string, wrap to list for scorer API
        if len(images_list) == 1 and isinstance(prompts, str):
            prompts = [prompts]
        use_gpu = bool(metadata.get("ocr_use_gpu", self.use_gpu))
        scorer = self.scorers.get(use_gpu)
        if scorer is None:
            scorer = OcrScorer(use_gpu=use_gpu)
            self.scorers[use_gpu] = scorer
        rewards = scorer(images_list, prompts)

        return {
            "scores": {"ocr_reward": rewards},
            "input_info": metadata.get("input_info", {}),
            "duration": f"{time.time() - time_start:.2f}",
            "decoded_duration": metadata.get("decode_duration", "N/A"),
            "type": self.reward_name,
        }