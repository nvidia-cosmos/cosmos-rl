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
# HPSv2 model from tgxs002 (https://github.com/tgxs002/HPSv2)

import os
import torch
from torchvision.transforms import Normalize, Compose, InterpolationMode
import torchvision.transforms.functional as F
from cosmos_rl_reward.handler.reward_base import BaseRewardHandler
from cosmos_rl_reward.handler.registry import RewardRegistry
from cosmos_rl_reward.utils.logging import logger


OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


class ResizeMaxSize(torch.nn.Module):
    def __init__(self, max_size, interpolation=InterpolationMode.BICUBIC, fn="max", fill=0):
        super().__init__()
        self.max_size = max_size
        self.interpolation = interpolation
        self.fn = min if fn == "min" else min
        self.fill = fill

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            h, w = img.shape[-2:]
        else:
            w, h = img.size
        scale = self.max_size / float(max(h, w))
        if scale != 1.0:
            new_size = (round(h * scale), round(w * scale))
            img = F.resize(img, new_size, self.interpolation)
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            img = F.pad(img, padding=[pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2], fill=self.fill)
        return img


class MaskAwareNormalize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.normalize = Normalize(mean=mean, std=std)

    def forward(self, tensor):
        if tensor.shape[1] == 4:
            parts = []
            for i in range(tensor.shape[0]):
                img = tensor[i]
                parts.append(torch.cat([self.normalize(img[:3]), img[3:]], dim=0))
            return torch.stack(parts, dim=0)
        return self.normalize(tensor)


def image_transform_tensor(image_size: int, mean=None, std=None, fill_color: int = 0):
    mean = mean or OPENAI_DATASET_MEAN
    std = std or OPENAI_DATASET_STD
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3
    return Compose([ResizeMaxSize(image_size, fill=fill_color), MaskAwareNormalize(mean=mean, std=std)])


@RewardRegistry.register()
class HPSv2Reward(BaseRewardHandler):
    NEEDS_LATENT_DECODER = False
    reward_name = "hpsv2"

    def __init__(self, model_path: str = "", download_path: str = "", device: str = "cuda", dtype: str = "float32", **kwargs):
        super().__init__()
        self.device = device
        self.dtype = torch.float32
        self.model_path = model_path
        self.download_path = download_path
        self.model = None
        self.preprocess = None
        self.tokenizer = None

    def set_up(self):
        try:
            from hpsv2.src.open_clip import create_model, get_tokenizer  # type: ignore

            # Resolve ckpt from download_path only
            base_dir = self.download_path
            hps_ckpt = os.path.join(base_dir, "hpsv2", "ckpts", "HPS_v2.1_compressed.pt")
            if not os.path.exists(hps_ckpt):
                raise FileNotFoundError(
                    f"[hpsv2] checkpoint not found at {hps_ckpt}. Run setup script or update rewards.toml."
                )

            model = create_model(
                "ViT-H-14",
                precision="amp",
                device=self.device,
                jit=False,
                force_quick_gelu=False,
                force_custom_text=False,
                force_patch_dropout=False,
                force_image_size=None,
                pretrained_image=False,
                output_dict=True,
            )

            image_mean = getattr(model.visual, "image_mean", None)
            image_std = getattr(model.visual, "image_std", None)
            image_size = model.visual.image_size
            if isinstance(image_size, tuple):
                image_size = image_size[0]
            self.model = model
            self.preprocess = image_transform_tensor(image_size, mean=image_mean, std=image_std)
            checkpoint = torch.load(hps_ckpt, map_location="cpu")
            self.model.load_state_dict(checkpoint["state_dict"])
            self.tokenizer = get_tokenizer("ViT-H-14")
            self.model.eval()
            logger.info("[hpsv2] Initialization complete.")
        except Exception as e:
            logger.error(f"[hpsv2] Failed to initialize: {e}")
            self.model = None

    def calculate_reward(self, images, metadata):
        if not isinstance(images, torch.Tensor) or images.dim() != 4:
            raise ValueError(f"Hpsv2 expects 4D torch.Tensor in BHWC/NHWC layout (B,H,W,C); got type={type(images)} shape={getattr(images,'shape',None)} dtype={getattr(images,'dtype',None)}")
        start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if start is not None:
            start.record()
        x = images
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2).contiguous()
        x = x / 255.0
        x = self.preprocess(x.to(device=self.device, non_blocking=True))
        prompts = metadata.get("prompts", [""] * x.shape[0])
        if isinstance(prompts, str):
            prompts = [prompts] * x.shape[0]
        text = self.tokenizer(prompts).to(device=self.device, non_blocking=True)
        with torch.no_grad():
            outputs = self.model(x, text)
            img_f, txt_f = outputs["image_features"], outputs["text_features"]
            logits = img_f @ txt_f.T
            scores = torch.diagonal(logits, 0).float().cpu().tolist()
        if end is not None:
            end.record()
            torch.cuda.synchronize()
            duration = f"{start.elapsed_time(end)/1000.0:.2f}"
        else:
            duration = "0.00"
        return {
            "scores": {"hpsv2": scores},
            "input_info": metadata.get("input_info", {}),
            "duration": duration,
            "decoded_duration": metadata.get("decode_duration", "N/A"),
            "type": self.reward_name,
        }