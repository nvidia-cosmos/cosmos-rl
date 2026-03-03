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

"""
VLA-compatible wrapper around CosmosPolicyModel.

Mirrors the OpenVLA / PI05 pattern: an ``nn.Module`` that holds the
underlying generative model and exposes ``process_input`` and
``generate_action`` for the VLA rollout engine.
"""

from __future__ import annotations

import io
import json
import os
import pickle
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image

from cosmos_rl.policy.model.wfm.models.cosmos_policy_model import (
    ACTION_DIM,
    COSMOS_IMAGE_SIZE,
    COSMOS_TEMPORAL_COMPRESSION_FACTOR,
    CosmosPolicyModel,
    CosmosPolicyModelConfig,
    extract_action_chunk,
    extract_value,
    load_cosmos_policy_model,
)
from cosmos_rl.utils.logging import logger

# ---------------------------------------------------------------------------
# Config for the VLA wrapper (eval-time knobs)
# ---------------------------------------------------------------------------

@dataclass
class CosmosPolicyVLAConfig:
    ckpt_path: str = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
    dataset_stats_path: str = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json"
    t5_text_embeddings_path: str = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl"
    num_denoising_steps: int = 5
    chunk_size: int = 16
    seed: int = 1
    use_jpeg_compression: bool = True
    trained_with_image_aug: bool = True
    use_variance_scale: bool = False
    vae_fp32: bool = False


# ---------------------------------------------------------------------------
# Image preprocessing helpers (match cosmos-policy train-time transforms)
# ---------------------------------------------------------------------------

def _resolve_hf_path(path: str) -> str:
    if path is None or path == "":
        return path
    if "/" in path and not path.startswith("/") and not path.startswith("./"):
        parts = path.split("/")
        if len(parts) == 2:
            from huggingface_hub import snapshot_download
            local_dir = snapshot_download(repo_id=path, resume_download=True)
            for candidate in ("model", ):
                d = os.path.join(local_dir, candidate)
                if os.path.isdir(d):
                    return d
            pt_files = [f for f in os.listdir(local_dir) if f.endswith(".pt")]
            if pt_files:
                return os.path.join(local_dir, pt_files[0])
            return local_dir
        elif len(parts) >= 3:
            from huggingface_hub import hf_hub_download
            repo_id = f"{parts[0]}/{parts[1]}"
            filename = "/".join(parts[2:])
            return hf_hub_download(repo_id=repo_id, filename=filename, resume_download=True)
    return path


def _jpeg_compress(images: np.ndarray, quality: int = 95) -> np.ndarray:
    out = []
    for img in images:
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        out.append(np.array(Image.open(buf)))
    return np.stack(out)


def _resize_images(images: np.ndarray, size: int) -> np.ndarray:
    if images.shape[-3:] == (size, size, 3):
        return images.copy()
    return np.stack([np.array(Image.fromarray(im).resize((size, size))) for im in images])


def _apply_image_transforms(images: np.ndarray) -> np.ndarray:
    """90%-area center crop + resize back."""
    _, H, W, _ = images.shape
    crop = int(H * 0.9 ** 0.5)
    t = torch.from_numpy(images).permute(0, 3, 1, 2)
    out = torch.stack([TF.resize(TF.center_crop(x, crop), [H, W], antialias=True) for x in t])
    return out.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)


def _preprocess_cameras(imgs: np.ndarray, jpeg: bool, aug: bool) -> np.ndarray:
    if jpeg:
        imgs = _jpeg_compress(imgs)
    imgs = _resize_images(imgs, COSMOS_IMAGE_SIZE)
    if aug:
        imgs = _apply_image_transforms(imgs)
    return imgs


def _rescale_proprio(proprio: np.ndarray, stats: dict) -> np.ndarray:
    lo, hi = stats["proprio_min"], stats["proprio_max"]
    return 2 * ((proprio - lo) / (hi - lo)) - 1


def _unnormalize_actions(actions: np.ndarray, stats: dict) -> np.ndarray:
    lo, hi = stats["actions_min"], stats["actions_max"]
    shape = actions.shape
    actions = actions.reshape(-1, lo.shape[0])
    actions = 0.5 * (actions + 1) * (hi - lo) + lo
    return actions.reshape(shape)


# ---------------------------------------------------------------------------
# CosmosPolicyVLA – the nn.Module wrapper
# ---------------------------------------------------------------------------

class CosmosPolicyVLA(nn.Module):
    """VLA-compatible wrapper around :class:`CosmosPolicyModel`.

    Follows the same interface contract as ``OpenVLA`` and ``PI05``:

    * ``process_input(inputs, unnorm_key)``  – env obs → model batch
    * ``generate_action(inputs, …)``         – model batch → ``{"action": …}``
    * ``model_input_keys / model_output_keys / model_train_keys``
    * ``processor``, ``tokenizer`` (stubs)
    """

    # Keys expected by VLA rollout bookkeeping
    model_input_keys: List[str] = []
    model_output_keys: List[str] = ["action"]
    model_train_keys: List[str] = ["action"]

    # Stubs – the VLA rollout reads these but they're unused for WFM
    processor = None
    tokenizer = SimpleNamespace(pad_token_id=0)

    def __init__(
        self,
        model: CosmosPolicyModel,
        dataset_stats: dict,
        t5_cache: dict,
        vla_cfg: CosmosPolicyVLAConfig,
    ):
        super().__init__()
        self.model = model
        self.dataset_stats = dataset_stats
        self.t5_cache = t5_cache
        self.vla_cfg = vla_cfg

    # -- compatibility stubs ------------------------------------------------

    @property
    def parallelize_fn(self):
        return (lambda *_a, **_kw: None, None)

    def load_hf_weights(self, *args, **kwargs):
        pass

    def _set_fsdp_reshard_after_forward(self, *args, **kwargs):
        pass

    # -- factory ------------------------------------------------------------

    @classmethod
    def from_config(cls, vla_cfg: CosmosPolicyVLAConfig) -> "CosmosPolicyVLA":
        """Build the full model from a :class:`CosmosPolicyVLAConfig`."""
        ckpt_path = _resolve_hf_path(vla_cfg.ckpt_path)
        model_cfg = CosmosPolicyModelConfig(vae_fp32=vla_cfg.vae_fp32)
        model, _ = load_cosmos_policy_model(ckpt_path, config=model_cfg)

        stats_path = _resolve_hf_path(vla_cfg.dataset_stats_path)
        with open(stats_path, "r") as f:
            dataset_stats = {k: np.array(v) for k, v in json.load(f).items()}

        t5_path = _resolve_hf_path(vla_cfg.t5_text_embeddings_path)
        with open(t5_path, "rb") as f:
            raw = pickle.load(f)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t5_cache = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in raw.items()}

        return cls(model, dataset_stats, t5_cache, vla_cfg)

    # -- core interface: process_input --------------------------------------

    def process_input(
        self,
        inputs: Dict[str, Any],
        unnorm_key: str = "",
    ) -> Dict[str, Any]:
        """Convert environment observations into a batched data-dict.

        Args:
            inputs: dict with ``full_images`` (N,H,W,3), ``wrist_images``
                (N,H,W,3), ``states`` (N,D), ``task_descriptions`` [str].
            unnorm_key: ignored (kept for interface compat).

        Returns:
            Data-dict ready for :meth:`generate_action`.
        """
        full_images = inputs["full_images"]
        wrist_images = inputs["wrist_images"]
        proprio_9d = inputs.get("proprio_9d", inputs.get("states"))
        task_descriptions = inputs["task_descriptions"]
        B = full_images.shape[0]
        cfg = self.vla_cfg

        videos, proprios, t5_list = [], [], []

        for i in range(B):
            cam = np.stack([wrist_images[i][:, ::-1], full_images[i][:, ::-1]])
            cam = _preprocess_cameras(cam, cfg.use_jpeg_compression, cfg.trained_with_image_aug)
            wrist, primary = cam[0], cam[1]
            blank = np.zeros_like(primary)
            dup = lambda arr: np.stack([arr] * COSMOS_TEMPORAL_COMPRESSION_FACTOR)  # noqa: E731
            blank4 = dup(blank)

            seq, idx = [], 0
            seq.append(blank[None]); idx += 1                       # 0: header
            seq.append(blank4.copy()); proprio_idx = idx; idx += 1   # 1: proprio
            seq.append(dup(wrist)); wrist_idx = idx; idx += 1        # 2: wrist
            seq.append(dup(primary)); image_idx = idx; idx += 1      # 3: primary
            seq.append(blank4.copy()); action_idx = idx; idx += 1    # 4: action
            seq.append(blank4.copy()); fut_proprio_idx = idx; idx += 1  # 5: future proprio
            seq.append(dup(wrist)); fut_wrist_idx = idx; idx += 1    # 6: future wrist
            seq.append(dup(primary)); fut_image_idx = idx; idx += 1  # 7: future primary
            seq.append(blank4.copy()); value_idx = idx; idx += 1     # 8: value

            video = np.transpose(np.concatenate(seq), (3, 0, 1, 2))  # (3, T, H, W)
            videos.append(video)
            proprios.append(_rescale_proprio(proprio_9d[i], self.dataset_stats))

            emb = self.t5_cache.get(task_descriptions[i])
            if emb is None:
                emb = next(iter(self.t5_cache.values()))
                logger.warning(f"T5 cache miss for '{task_descriptions[i]}', using fallback.")
            if emb.dim() == 3 and emb.shape[0] == 1:
                emb = emb.squeeze(0)
            t5_list.append(emb)

        idx_map = {
            "current_proprio_latent_idx": proprio_idx,
            "current_wrist_image_latent_idx": wrist_idx,
            "current_image_latent_idx": image_idx,
            "action_latent_idx": action_idx,
            "future_proprio_latent_idx": fut_proprio_idx,
            "future_wrist_image_latent_idx": fut_wrist_idx,
            "future_image_latent_idx": fut_image_idx,
            "value_latent_idx": value_idx,
        }
        unused_keys = [
            "current_wrist_image2_latent_idx", "current_image2_latent_idx",
            "future_wrist_image2_latent_idx", "future_image2_latent_idx",
        ]

        data_batch: Dict[str, Any] = {
            "dataset_name": "video_data",
            "video": torch.from_numpy(np.stack(videos)).to(dtype=torch.uint8, device="cuda"),
            "t5_text_embeddings": torch.stack(t5_list).to(dtype=torch.bfloat16, device="cuda"),
            "fps": torch.full((B,), 16, dtype=torch.bfloat16, device="cuda"),
            "padding_mask": torch.zeros(B, 1, COSMOS_IMAGE_SIZE, COSMOS_IMAGE_SIZE, dtype=torch.bfloat16, device="cuda"),
            "num_conditional_frames": self.model.config.min_num_conditional_frames,
            "proprio": torch.from_numpy(np.stack(proprios)).reshape(B, -1).to(dtype=torch.bfloat16, device="cuda"),
        }
        for key, val in idx_map.items():
            data_batch[key] = torch.full((B,), val, dtype=torch.int64, device="cuda")
        for key in unused_keys:
            data_batch[key] = torch.full((B,), -1, dtype=torch.int64, device="cuda")

        return data_batch

    # -- core interface: generate_action ------------------------------------

    @torch.no_grad()
    def generate_action(
        self,
        inputs: Dict[str, Any],
        is_valid: bool = False,
        temperature: float = 0.0,
        unnorm_key: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        """Run diffusion sampling and extract the action chunk.

        Args:
            inputs: data-dict produced by :meth:`process_input`.

        Returns:
            ``{"action": tensor}`` with shape ``(B, chunk_size, action_dim)``.
        """
        cfg = self.vla_cfg
        B = inputs["video"].shape[0]

        generated = self.model.generate_samples_from_batch(
            inputs,
            n_sample=B,
            num_steps=cfg.num_denoising_steps,
            seed=cfg.seed,
            is_negative_prompt=False,
            use_variance_scale=cfg.use_variance_scale,
        )

        actions_np = (
            extract_action_chunk(
                generated, (cfg.chunk_size, ACTION_DIM), inputs["action_latent_idx"],
            )
            .to(torch.float32)
            .cpu()
            .numpy()
        )
        actions_np = _unnormalize_actions(actions_np, self.dataset_stats)
        return {"action": torch.from_numpy(actions_np)}
