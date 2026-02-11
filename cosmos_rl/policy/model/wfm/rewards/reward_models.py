# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------


"""Reward models for GRPO training."""

import io
import json
import numpy as np
import os
import requests
from abc import ABC, abstractmethod
from functools import partial
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from typing import Any, Dict, List

import torch
import torch.nn as nn

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.network_util import make_request_with_retry


def convert_to_pil_image(generated_samples: torch.Tensor) -> List[Image.Image]:
    # Sameple has shape [B, 3, T, H, W]
    b, c, t, h, w = generated_samples.shape
    generated_samples = generated_samples.permute(0, 2, 1, 3, 4)
    generated_samples = generated_samples.reshape(b * t, c, h, w)

    # Convert to PIL images
    images = []
    for i in range(generated_samples.shape[0]):
        # Convert from tensor to numpy
        sample = generated_samples[i].detach().cpu()

        # Ensure correct range [0, 1]
        if sample.min() < 0:
            sample = (sample + 1) / 2  # Convert from [-1, 1] to [0, 1]

        # Convert to uint8
        sample = (sample * 255).clamp(0, 255).to(torch.uint8)

        # Convert to PIL Image
        if sample.shape[0] == 3:  # RGB
            sample = sample.permute(1, 2, 0)  # CHW -> HWC
            image = Image.fromarray(sample.numpy(), mode="RGB")
        else:
            # Handle other formats if needed
            image = Image.fromarray(sample.numpy().squeeze(), mode="L")

        images.append(image)
    return images


class BaseRewardModel(nn.Module, ABC):
    """Base class for all reward models."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.enabled = config.enabled

        logger.info(f"Initializing {self.__class__.__name__}. Enabled: {self.enabled}")

    @abstractmethod
    def compute_reward(
        self, data_batch: Dict[str, Any], generated_samples: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Compute reward for generated samples."""
        pass


class RemoteReward(BaseRewardModel):
    """Remote reward model that calls external API for reward computation."""

    def __init__(self, config):
        super().__init__(config)
        if self.enabled:
            self.token = config.token
            self.score_key = config.score_key
            self.enqueue_url = config.enqueue_url
            self.fetch_url = config.fetch_url
            self.reward_fn = config.reward_fn
            self.scale = config.scale

            logger.info(f"RemoteReward initialized with score_key: {self.score_key}")

    @torch.no_grad()
    def compute_reward(
        self,
        data_batch: Dict[str, Any],
        generated_samples: torch.Tensor,
        latents: torch.Tensor = None,
        is_async: bool = False,
    ) -> torch.Tensor:
        """
        Compute reward using remote API.

        Args:
            data_batch: Dictionary containing prompts and other data
            generated_samples: Generated video samples [B, C, T, H, W] in pixel space [-1, 1] (not used)
            latents: Latent representations [B, C, T, H, W] to send to remote service

        Returns:
            rewards: Tensor of shape [B] with overall reward scores
        """
        tensor = latents.cpu().numpy()

        # Prepare data for API (for entire batch)   A list of prompts
        prompts = data_batch["ai_caption"]

        # Create video info for entire batch (assuming 16 FPS as default)
        video_infos = []
        for _ in range(latents.shape[0]):
            video_infos.append({"video_fps": 16.0})

        data = {
            "prompts": prompts,
            "reward_fn": {
                f"{self.reward_fn}": 1.0,
            },
            "video_infos": video_infos,
        }

        # Enqueue request (single call for entire batch)
        uuid, replica_id = self.enqueue_request(tensor, data)
        if is_async:
            logger.info(
                f"Reward compute is async, returning uuid: {uuid}, replica_id: {replica_id}"
            )
            return (
                torch.zeros((latents.shape[0],), device=latents.device),
                uuid,
                replica_id,
            )

        # Poll for reward
        reward = self.fetch_reward(uuid, replica_id)

        return reward.to(latents.device), None, None

    def enqueue_request(self, tensor, data):
        """Enqueue the request and return UUID."""

        buffer = io.BytesIO()
        np.save(buffer, tensor)
        buffer.seek(0)

        # Combine JSON + binary data
        payload = json.dumps(data).encode("utf-8") + b"\n" + buffer.getvalue()

        response = make_request_with_retry(
            partial(
                requests.post,
                data=payload,
                headers={
                    "Content-Type": "application/octet-stream",
                    "Authorization": f"Bearer {self.token}",
                },
                timeout=10.0,
            ),
            [self.enqueue_url],
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Enqueue failed with status {response.status_code}: {response.text}"
            )

        uuid = response.json()["uuid"]
        replica_id = response.json().get("replica_id", None)
        # logger.info(f"[RemoteReward] Enqueued request with UUID: {uuid}, replica_Id: {replica_id}, latents.max(): {tensor.max()}")
        return uuid, replica_id

    def fetch_reward(self, uuid, replica_id, return_all: bool = False):
        """Poll for reward until ready."""

        # Specify replica_id header if available for lepton endpoint
        headers = {
            "Authorization": f"Bearer {self.token}",
        }
        if (
            replica_id is not None
            and not os.environ.get("COSMOS_DISABLE_REMOTE_REWARD_USE_REPLICA", "0")
            == "1"
        ):
            headers["X-Lepton-Replica-Target"] = replica_id

        response = make_request_with_retry(
            partial(
                requests.post,
                data={"uuid": uuid, "type": self.reward_fn},
                headers=headers,
                timeout=5.0,
            ),
            [self.fetch_url],
        )

        response_json = response.json()

        # Extract overall reward
        if return_all:
            return response_json["scores"]
        if self.score_key not in response_json["scores"]:
            rewards = [
                torch.tensor(response_json["scores"][w])
                for w in self.score_key.split("+")
            ]
            reward = sum(rewards)
        else:
            reward = torch.tensor(response_json["scores"][self.score_key])
        return (
            torch.clamp(
                reward, min=self.config.reward_clip_min, max=self.config.reward_clip_max
            )
            * self.scale
        )


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)


class AestheticReward(BaseRewardModel):
    def __init__(self, config, device="cuda"):
        super().__init__(config)
        if self.enabled:
            hf_home = (
                os.environ.get(
                    "HF_HOME",
                    os.path.expanduser("~/.cache/huggingface/transformers/"),
                ),
            )
            self.clip = CLIPModel.from_pretrained(
                "openai/clip-vit-large-patch14", cache_dir=hf_home
            )
            self.processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-large-patch14", cache_dir=hf_home
            )
            self.mlp = MLP()

            from pathlib import Path

            state_name = "sac+logos+ava1-l14-linearMSE.pth"
            path = os.path.join(hf_home, state_name)
            if not Path(path).exists():
                url = f"https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/{state_name}?raw=true"
                import requests

                r = requests.get(url)
                with open(path, "wb") as f:
                    f.write(r.content)

            # Load the pretrained aesthetic model weights
            state_dict = torch.load(path, map_location="cpu")
            self.mlp.load_state_dict(state_dict)

            self.eval()
            self.to(device)
            self.device = device

            self.requires_grad_(False)

    @torch.no_grad()
    def compute_reward(
        self,
        data_batch: Dict[str, Any],
        generated_samples: torch.Tensor,
        is_async: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        b, c, t, h, w = generated_samples.shape

        images = convert_to_pil_image(generated_samples)

        # Get aesthetic scores

        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        reward = self.mlp(embed).squeeze(1)
        return reward.reshape(b, t).mean(dim=1), None


class ImageReward(BaseRewardModel):
    """ImageReward model for computing text-image alignment scores."""

    def __init__(self, config, device="cuda"):
        super().__init__(config)
        if self.enabled:
            # Lazy import ImageReward to avoid unnecessary dependencies
            import ImageReward as RM

            self.device = device

            # Initialize the ImageReward model
            self.model = RM.load(
                "ImageReward-v1.0",
                device=device,
                download_root=os.environ.get(
                    "HF_HOME",
                    os.path.expanduser("~/.cache/huggingface/transformers/"),
                ),
            ).eval()
            self.model.to(device)
            self.model.requires_grad_(False)

            logger.info("ImageReward model loaded")

    @torch.no_grad()
    def compute_reward(
        self,
        data_batch: Dict[str, Any],
        generated_samples: torch.Tensor,
        is_async: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute ImageReward scores for generated samples.

        Args:
            data_batch: Dictionary containing prompts and other data
            generated_samples: Generated video/image samples [B, C, T, H, W]

        Returns:
            rewards: Tensor of shape [B] with reward scores
        """

        b, c, t, h, w = generated_samples.shape

        assert b == 1, "We only handle logic for batch size 1 in ImageReward"
        # Extract prompts from data_batch
        # Try different possible keys for prompts

        images = convert_to_pil_image(generated_samples)

        _, rewards = self.model.inference_rank(data_batch["ai_caption"], images)

        # Convert to tensor
        reward_tensor = torch.tensor(
            rewards, device=generated_samples.device, dtype=torch.float32
        )

        return reward_tensor.reshape(b, t).mean(dim=1), None


class FakeReward(BaseRewardModel):
    def __init__(self, config):
        super().__init__(config)
        if self.enabled:
            self.reward_fn = config.reward_fn
            self.scale = config.scale

    @torch.no_grad()
    def compute_reward(
        self,
        data_batch: Dict[str, Any],
        generated_samples: torch.Tensor,
        is_async: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        return torch.zeros(
            (generated_samples.shape[0],), device=generated_samples.device
        ), None


if __name__ == "__main__":
    from cosmos_rl.policy.config.wfm import RemoteRewardConfig

    # reward_models = [AestheticReward(AestheticRewardConfig()), ImageReward(ImageRewardConfig())]
    reward_models = [RemoteReward(RemoteRewardConfig())]
    samples = torch.randn(8, 3, 4, 256, 256).to("cuda")
    latents = torch.randn(8, 16, 4, 64, 64).to("cuda")
    # images = [Image.open("cat.jpeg"), Image.open("cat2.jpg")]
    scores = [
        reward_model.compute_reward(
            data_batch={"ai_caption": ["a white cat"]},
            generated_samples=samples,
            latents=latents,
        )
        for reward_model in reward_models
    ]
    print(scores)
