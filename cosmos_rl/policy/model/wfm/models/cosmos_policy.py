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
Cosmos Policy model for robot manipulation.

Provides a unified ``from_pretrained`` interface (like OpenVLA / PI05)::

    model = CosmosPolicy.from_pretrained(None, "nvidia/Cosmos-Policy-LIBERO-Predict2-2B")
    batch = model.process_input(observations)
    out   = model.generate_action(batch)
    actions = out["action"]

Internally composes:
  * MinimalV1LVGDiT   – DiT backbone
  * Wan2pt1VAEInterface – VAE tokenizer
  * Video2WorldConditioner – T5 text + video conditioning
  * EDMSDE / RectifiedFlowScaling – noise schedule
  * CosmosPolicySampler – denoising sampler
"""

from __future__ import annotations

import io
import json
import math
import os
import pickle
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from einops import rearrange
from PIL import Image
from transformers import PretrainedConfig, AutoConfig

from cosmos_rl.policy.config.wfm import ConditionerConfig, NetConfig
from cosmos_rl.policy.model.base import BaseModel, ModelRegistry, WeightMapper
from cosmos_rl.policy.model.wfm.conditioner.video2world import (
    Video2WorldCondition,
    Video2WorldConditioner,
)
from cosmos_rl.policy.model.wfm.networks.minimal_v1_lvg_dit import MinimalV1LVGDiT
from cosmos_rl.policy.model.wfm.sampler import EDMSDE, RectifiedFlowScaling
from cosmos_rl.policy.model.wfm.sampler.cosmos_policy_sampler import CosmosPolicySampler
from cosmos_rl.policy.model.wfm.tokenizer.wan2pt1 import Wan2pt1VAEInterface
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.wfm.utils import DataType, DenoisePrediction


# ---------------------------------------------------------------------------
# HuggingFace config registration (so AutoConfig.from_pretrained recognises
# model repos whose config.json has "model_type": "cosmos-policy")
# ---------------------------------------------------------------------------


class CosmosPolicyHFConfig(PretrainedConfig):
    model_type = "cosmos-policy"


AutoConfig.register("cosmos-policy", CosmosPolicyHFConfig)


class CosmosPolicyWeightMapper(WeightMapper):
    """No-op weight mapper – cosmos-policy loads its own DCP checkpoint."""

    def policy_map_local_key_to_hf_key(self, name: str) -> str:
        return name

    def rollout_map_local_key_to_hf_key(self, name: str) -> str:
        return name

    def rollout_split_local_key_n_param_to_hf_key_n_param(self, name, param):
        return [(self.rollout_map_local_key_to_hf_key(name), param)]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ACTION_DIM = 7
COSMOS_IMAGE_SIZE = 224
COSMOS_TEMPORAL_COMPRESSION_FACTOR = 4

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class CosmosPolicyConfig:
    """All knobs needed to build and run a :class:`CosmosPolicy`.

    Parameters are split into two groups:

    **Model architecture** (rarely changed):
        ``sigma_*``, ``state_*``, ``scaling``, ``precision``, ``net``, …

    **Evaluation / asset paths** (set per run):
        ``ckpt_path``, ``dataset_stats_path``, ``t5_text_embeddings_path``,
        ``num_denoising_steps``, ``chunk_size``, ``seed``, …
    """

    # -- model architecture -------------------------------------------------
    sigma_data: float = 1.0
    sigma_max: float = 80.0
    sigma_min: float = 4.0
    sigma_conditional: float = 0.0
    state_ch: int = 16
    state_t: int = 9
    min_num_conditional_frames: int = 4
    max_num_conditional_frames: int = 4
    conditioning_strategy: str = "frame_replace"
    denoise_replace_gt_frames: bool = True
    use_rf_ckpts: bool = False
    precision: str = "bfloat16"
    scaling: str = "rectified_flow"
    rectified_flow_t_scaling_factor: float = 1.0
    rectified_flow_loss_weight_uniform: bool = True
    tokenizer_chunk_duration: int = 33
    adjust_video_noise: bool = True
    atten_backend: str = "minimal_a2a"
    tokenizer_weights: str = (
        "hf://nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth"
    )
    vae_fp32: bool = False
    conditioner: List[ConditionerConfig] = field(
        default_factory=lambda: [
            ConditionerConfig(
                name="fps",
                type="remap_key",
                input_key="fps",
                output_key="fps",
                dropout_rate=0.0,
            ),
            ConditionerConfig(
                name="padding_mask",
                type="remap_key",
                input_key="padding_mask",
                output_key="padding_mask",
                dropout_rate=0.0,
            ),
            ConditionerConfig(
                name="text",
                type="text_attr",
                input_key=["t5_text_embeddings"],
                output_key="crossattn_emb",
                dropout_rate=0.0,
            ),
            ConditionerConfig(
                name="use_video_condition",
                type="boolean_flag",
                input_key="fps",
                output_key="use_video_condition",
                dropout_rate=0.0,
            ),
        ]
    )
    net: NetConfig = field(
        default_factory=lambda: NetConfig(atten_backend="minimal_a2a")
    )

    # -- evaluation / assets ------------------------------------------------
    ckpt_path: str = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
    dataset_stats_path: str = (
        "nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json"
    )
    t5_text_embeddings_path: str = (
        "nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl"
    )
    num_denoising_steps: int = 5
    chunk_size: int = 16
    seed: int = 1
    use_jpeg_compression: bool = True
    trained_with_image_aug: bool = True
    use_variance_scale: bool = False

    # -- task suite (optional, used by rollout) -----------------------------
    task_suite_name: str = "libero_10"
    max_steps: int = 520
    num_envs: int = 1
    save_video: bool = True


# ---------------------------------------------------------------------------
# Latent injection / extraction helpers
# ---------------------------------------------------------------------------


def arch_invariant_rand(shape, dtype, device, seed=None):
    """GPU-architecture-invariant random tensor (matches cosmos-policy)."""
    rng = np.random.RandomState(seed)
    return torch.from_numpy(rng.standard_normal(shape).astype(np.float32)).to(
        dtype=dtype, device=device
    )


def _replace_latent_with_flat(
    x0: torch.Tensor, flat_data: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    batch_indices = torch.arange(x0.shape[0], device=x0.device)
    target = x0[batch_indices, :, indices, :, :]
    B, C, H, W = target.shape
    total = C * H * W
    flat = flat_data.to(dtype=x0.dtype).reshape(B, -1)
    n = flat.shape[1]
    assert n <= total, f"Data elements ({n}) > latent elements ({total})"
    reps = (total + n - 1) // n
    repeated = flat.repeat(1, reps)[:, :total]
    x0[batch_indices, :, indices, :, :] = repeated.reshape(B, C, H, W)
    return x0


def _extract_action_chunk(
    generated: torch.Tensor,
    action_shape: Tuple[int, int],
    action_indices: torch.Tensor,
) -> torch.Tensor:
    batch_indices = torch.arange(generated.shape[0], device=generated.device)
    latent = generated[batch_indices, :, action_indices, :, :]
    flat = latent.reshape(generated.shape[0], -1)
    n_elements = action_shape[0] * action_shape[1]
    return flat[:, :n_elements].reshape(generated.shape[0], *action_shape)


def _extract_value(
    generated: torch.Tensor, value_indices: torch.Tensor
) -> torch.Tensor:
    batch_indices = torch.arange(generated.shape[0], device=generated.device)
    latent = generated[batch_indices, :, value_indices, :, :]
    return latent.reshape(generated.shape[0], -1).mean(dim=1)


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
            for candidate in ("model",):
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
            return hf_hub_download(
                repo_id=repo_id, filename=filename, resume_download=True
            )
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
    return np.stack(
        [np.array(Image.fromarray(im).resize((size, size))) for im in images]
    )


def _apply_image_transforms(images: np.ndarray) -> np.ndarray:
    """90%-area center crop + resize back."""
    _, H, W, _ = images.shape
    crop = int(H * 0.9**0.5)
    t = torch.from_numpy(images).permute(0, 3, 1, 2)
    out = torch.stack(
        [TF.resize(TF.center_crop(x, crop), [H, W], antialias=True) for x in t]
    )
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
# Internal generative model (DiT + VAE + conditioner + sampler)
# ---------------------------------------------------------------------------


class _GenerativeModel(nn.Module):
    """The diffusion backbone – not exposed directly; use :class:`CosmosPolicy`."""

    def __init__(self, config: CosmosPolicyConfig):
        super().__init__()
        self.config = config
        self.sigma_data = config.sigma_data

        precision_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.precision = precision_map[config.precision]
        self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}

        self.sde = EDMSDE(
            p_mean=math.log(4.0),
            p_std=1.2,
            sigma_max=config.sigma_max,
            sigma_min=config.sigma_min,
        )
        self.scaling = (
            RectifiedFlowScaling(
                config.sigma_data,
                config.rectified_flow_t_scaling_factor,
                config.rectified_flow_loss_weight_uniform,
            )
            if config.scaling == "rectified_flow"
            else None
        )
        self.sampler = CosmosPolicySampler()
        self.conditioner = Video2WorldConditioner(config.conditioner)

        net_cfg = config.net.model_copy()
        with torch.device("meta"):
            self.net = MinimalV1LVGDiT(**net_cfg.model_dump())
        self.net.to_empty(device="cuda")
        self.net.init_weights()
        self.net.to(**self.tensor_kwargs)

        self._tokenizer = Wan2pt1VAEInterface(
            chunk_duration=config.tokenizer_chunk_duration,
        )
        if config.vae_fp32:
            self._promote_vae_to_fp32()

    # -- VAE fp32 promotion -------------------------------------------------

    def _promote_vae_to_fp32(self):
        from cosmos_rl.utils.util import resolve_model_path
        from cosmos_rl.utils.wfm.io.easy_io import easy_io

        vae = self._tokenizer.model
        vae_pth = "Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
        vae_local_folder = resolve_model_path("/".join(vae_pth.split("/")[:-1]))
        vae_local_path = os.path.join(vae_local_folder, vae_pth.split("/")[-1])
        ckpt = easy_io.load(vae_local_path, map_location="cuda")
        vae.model.load_state_dict(ckpt, assign=True)
        del ckpt
        vae.dtype = torch.float32
        vae.mean = vae.mean.float()
        vae.std = vae.std.float()
        vae.scale = [vae.mean, 1.0 / vae.std]

    # -- properties ---------------------------------------------------------

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def spatial_compression_factor(self) -> int:
        return self._tokenizer.spatial_compression_factor

    @property
    def temporal_compression_factor(self) -> int:
        return self._tokenizer.temporal_compression_factor

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        return self._tokenizer.encode(video)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self._tokenizer.decode(latent)

    # -- normalise + tokenise -----------------------------------------------

    def _normalize_video(self, data_batch: dict) -> dict:
        key = "video"
        if key in data_batch and data_batch[key] is not None:
            v = data_batch[key]
            if v.dtype == torch.uint8:
                dtype = torch.float32 if self.config.vae_fp32 else self.precision
                data_batch[key] = v.to(dtype) / 127.5 - 1.0
        return data_batch

    def _tokenize(self, video: torch.Tensor) -> torch.Tensor:
        return self._tokenizer.encode(video).contiguous().float()

    def get_data_and_condition(self, data_batch: dict):
        self._normalize_video(data_batch)
        video = data_batch["video"]
        latent = self._tokenize(video)
        condition = self.conditioner.get_condition_uncondition(data_batch)[0]
        return video, latent, condition

    # -- denoising ----------------------------------------------------------

    def denoise(
        self,
        xt_B_C_T_H_W: torch.Tensor,
        sigma: torch.Tensor,
        condition: Video2WorldCondition,
    ) -> DenoisePrediction:
        if sigma.ndim == 1:
            sigma_B_T = rearrange(sigma, "b -> b 1")
        else:
            sigma_B_T = sigma

        sigma_B_1_T_1_1 = rearrange(sigma_B_T, "b t -> b 1 t 1 1")
        c_skip, c_out, c_in, c_noise = self.scaling(sigma=sigma_B_1_T_1_1)

        net_input = xt_B_C_T_H_W * c_in

        if condition.is_video:
            cond_input = condition.gt_frames.type_as(net_input) / self.sigma_data
            if not condition.use_video_condition:
                cond_input = cond_input * 0

            _, C, _, _, _ = xt_B_C_T_H_W.shape
            cond_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(
                1, C, 1, 1, 1
            ).type_as(net_input)

            net_input = cond_input * cond_mask + net_input * (1 - cond_mask)
            if not self.config.use_rf_ckpts:
                sigma_cond = (
                    torch.ones_like(sigma_B_1_T_1_1) * self.config.sigma_conditional
                )
                _, _, _, c_noise_cond = self.scaling(sigma=sigma_cond)
                mask_avg = cond_mask.mean(dim=[1, 3, 4], keepdim=True)
                c_noise = c_noise_cond * mask_avg + c_noise * (1 - mask_avg)

        net_out = self.net(
            x_B_C_T_H_W=net_input.to(**self.tensor_kwargs),
            timesteps_B_T=c_noise.squeeze(dim=[1, 3, 4]).to(**self.tensor_kwargs),
            **condition.to_dict(),
        ).float()

        x0_pred = c_skip * xt_B_C_T_H_W + c_out * net_out
        if condition.is_video and self.config.denoise_replace_gt_frames:
            x0_pred = condition.gt_frames.type_as(x0_pred) * cond_mask + x0_pred * (
                1 - cond_mask
            )

        eps_pred = (xt_B_C_T_H_W - x0_pred) / sigma_B_1_T_1_1
        return DenoisePrediction(x0_pred, eps_pred, None)

    # -- x0_fn builder (policy-specific) ------------------------------------

    def get_x0_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
        skip_vae_encoding: bool = False,
        previous_generated_latent: Optional[torch.Tensor] = None,
        return_orig_clean_latent_frames: bool = False,
    ):
        num_conditional_frames = data_batch.get("num_conditional_frames", 1)
        ncf_min = self.config.min_num_conditional_frames
        ncf_max = self.config.max_num_conditional_frames

        if is_negative_prompt:
            condition, uncondition = (
                self.conditioner.get_condition_with_negative_prompt(data_batch)
            )
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(
                data_batch
            )

        condition.data_type = DataType.VIDEO
        if uncondition is not None:
            uncondition.data_type = DataType.VIDEO

        if skip_vae_encoding:
            assert previous_generated_latent is not None
            x0 = previous_generated_latent.clone()
        else:
            _, x0, _ = self.get_data_and_condition(data_batch)

        condition = condition.set_video_condition(
            x0, ncf_min, ncf_max, num_conditional_frames
        )
        if uncondition is not None:
            uncondition = uncondition.set_video_condition(
                x0, ncf_min, ncf_max, num_conditional_frames
            )

        condition = condition.edit_for_inference(
            is_cfg_conditional=True, num_conditional_frames=num_conditional_frames
        )
        if uncondition is not None:
            uncondition = uncondition.edit_for_inference(
                is_cfg_conditional=False,
                num_conditional_frames=num_conditional_frames,
            )

        orig_gt_frames = condition.gt_frames.clone()
        B = condition.condition_video_input_mask_B_C_T_H_W.shape[0]

        if "proprio" in data_batch and data_batch["proprio"] is not None:
            prop_idx = data_batch.get("current_proprio_latent_idx")
            if prop_idx is not None and torch.all(prop_idx != -1):
                batch_idx = torch.arange(B, device=prop_idx.device)
                condition.condition_video_input_mask_B_C_T_H_W[
                    batch_idx, :, prop_idx, :, :
                ] = 1
                condition.gt_frames = _replace_latent_with_flat(
                    condition.gt_frames, data_batch["proprio"], prop_idx
                )
                if uncondition is not None:
                    uncondition.condition_video_input_mask_B_C_T_H_W[
                        batch_idx, :, prop_idx, :, :
                    ] = 1
                    uncondition.gt_frames = _replace_latent_with_flat(
                        uncondition.gt_frames, data_batch["proprio"], prop_idx
                    )

        if data_batch.get("mask_current_state_action_for_value_prediction", False):
            batch_idx = torch.arange(
                B, device=condition.condition_video_input_mask_B_C_T_H_W.device
            )
            for k in [
                "current_proprio_latent_idx",
                "current_wrist_image_latent_idx",
                "current_wrist_image2_latent_idx",
                "current_image_latent_idx",
                "current_image2_latent_idx",
                "action_latent_idx",
            ]:
                if k in data_batch and torch.all(data_batch[k] != -1):
                    condition.condition_video_input_mask_B_C_T_H_W[
                        batch_idx, :, data_batch[k], :, :
                    ] = 0

        def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            cond_x0 = self.denoise(noise_x, sigma, condition).x0
            if uncondition is not None:
                uncond_x0 = self.denoise(noise_x, sigma, uncondition).x0
                return cond_x0 + guidance * (cond_x0 - uncond_x0)
            return cond_x0

        if return_orig_clean_latent_frames:
            return x0_fn, orig_gt_frames
        return x0_fn

    # -- sample generation --------------------------------------------------

    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        n_sample: int = 1,
        num_steps: int = 5,
        solver_option: str = "2ab",
        is_negative_prompt: bool = False,
        use_variance_scale: bool = False,
        skip_vae_encoding: bool = False,
        previous_generated_latent: Optional[torch.Tensor] = None,
        return_orig_clean_latent_frames: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        self._normalize_video(data_batch)
        video = data_batch.get("video")
        if video is not None:
            _T, _H, _W = video.shape[-3:]
            state_shape = [
                self.config.state_ch,
                self._tokenizer.get_latent_num_frames(_T),
                _H // self.spatial_compression_factor,
                _W // self.spatial_compression_factor,
            ]
        else:
            raise ValueError("data_batch must contain 'video'")

        if return_orig_clean_latent_frames:
            x0_fn, orig_clean = self.get_x0_fn_from_batch(
                data_batch,
                guidance,
                is_negative_prompt,
                skip_vae_encoding=skip_vae_encoding,
                previous_generated_latent=previous_generated_latent,
                return_orig_clean_latent_frames=True,
            )
        else:
            x0_fn = self.get_x0_fn_from_batch(
                data_batch,
                guidance,
                is_negative_prompt,
                skip_vae_encoding=skip_vae_encoding,
                previous_generated_latent=previous_generated_latent,
            )

        sigma_max_scale = 1.0
        sigma_min_scale = 1.0
        if use_variance_scale:
            torch.manual_seed(seed)
            sigma_max_scale = torch.rand(1).item() * 2.0 + 1.0
            sigma_min_scale = torch.rand(1).item() * 0.9 + 0.1

        x_sigma_max = (
            arch_invariant_rand(
                (n_sample,) + tuple(state_shape), torch.float32, "cuda", seed
            )
            * self.sde.sigma_max
            * sigma_max_scale
        )

        samples = self.sampler(
            x0_fn,
            x_sigma_max,
            num_steps=num_steps,
            sigma_max=self.sde.sigma_max * sigma_max_scale,
            sigma_min=self.sde.sigma_min * sigma_min_scale,
            solver_option=solver_option,
        )

        if return_orig_clean_latent_frames:
            return samples, orig_clean
        return samples


# ---------------------------------------------------------------------------
# CosmosPolicy – the public API
# ---------------------------------------------------------------------------


@ModelRegistry.register(CosmosPolicyWeightMapper)
class CosmosPolicy(BaseModel):
    """Cosmos Policy model for robot manipulation.

    Follows the same interface contract as ``OpenVLA`` and ``PI05``:

    * ``CosmosPolicy.from_pretrained(hf_config, model_name_or_path)``
    * ``process_input(inputs, unnorm_key)``  – env obs → model batch
    * ``generate_action(inputs, …)``         – model batch → ``{"action": …}``
    """

    model_input_keys: List[str] = []
    model_output_keys: List[str] = ["action"]
    model_train_keys: List[str] = ["action"]
    processor = None
    tokenizer = SimpleNamespace(pad_token_id=0)

    def __init__(
        self,
        generative_model: _GenerativeModel,
        dataset_stats: dict,
        t5_cache: dict,
        config: CosmosPolicyConfig,
    ):
        super().__init__(hf_config=None)
        self.generative_model = generative_model
        self.dataset_stats = dataset_stats
        self.t5_cache = t5_cache
        self.cfg = config

    # -- BaseModel abstract method implementations --------------------------

    @staticmethod
    def supported_model_types():
        return "cosmos-policy"

    @property
    def parallelize_fn(self):
        return (lambda *_a, **_kw: None, None)

    def apply_pipeline_split(self, pp_rank, pp_size):
        pass

    def get_position_ids(self, **kwargs):
        return torch.empty(0), torch.empty(0), 0

    def load_hf_weights(
        self, model_name_or_path, parallel_dims=None, device=None, revision=None
    ):
        pass

    def separate_model_parts(self):
        return [self]

    @classmethod
    def get_nparams_and_flops(cls, seq_len: int):
        return 0, 0

    def _set_fsdp_reshard_after_forward(self, *args, **kwargs):
        pass

    # -- factory: from_pretrained -------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        hf_config,
        model_name_or_path: str,
        max_position_embeddings: Optional[int] = None,
    ) -> "CosmosPolicy":
        """Build a :class:`CosmosPolicy` from a pretrained checkpoint.

        Called by ``ModelRegistry.build_hf_model``::

            model = CosmosPolicy.from_pretrained(
                hf_config, model_name_or_path, max_position_embeddings=4096
            )
        """
        dataset_stats_path = f"{model_name_or_path}/libero_dataset_statistics.json"
        t5_text_embeddings_path = f"{model_name_or_path}/libero_t5_embeddings.pkl"

        config = CosmosPolicyConfig(
            ckpt_path=model_name_or_path,
            dataset_stats_path=dataset_stats_path,
            t5_text_embeddings_path=t5_text_embeddings_path,
        )
        return cls.from_config(config)

    @classmethod
    def from_config(cls, config: CosmosPolicyConfig) -> "CosmosPolicy":
        """Build from an explicit :class:`CosmosPolicyConfig`."""
        ckpt_path = _resolve_hf_path(config.ckpt_path)
        generative_model = _GenerativeModel(config)

        from cosmos_rl.utils.wfm.io.easy_io import easy_io

        state_dict = easy_io.load(ckpt_path, weights_only=False)
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        elif isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        missing, unexpected = generative_model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.info(f"[CosmosPolicy] Missing keys (first 10): {missing[:10]}")
        if unexpected:
            logger.info(f"[CosmosPolicy] Unexpected keys (first 10): {unexpected[:10]}")
        generative_model.eval()
        generative_model.to("cuda")

        stats_path = _resolve_hf_path(config.dataset_stats_path)
        with open(stats_path, "r") as f:
            dataset_stats = {k: np.array(v) for k, v in json.load(f).items()}

        t5_path = _resolve_hf_path(config.t5_text_embeddings_path)
        with open(t5_path, "rb") as f:
            raw = pickle.load(f)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t5_cache = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in raw.items()
        }

        return cls(generative_model, dataset_stats, t5_cache, config)

    # -- core interface: process_input --------------------------------------

    def process_input(
        self,
        inputs: Dict[str, Any],
        unnorm_key: str = "",
    ) -> Dict[str, Any]:
        """Convert environment observations into a batched data-dict.

        Args:
            inputs: dict with ``full_images``, ``wrist_images``,
                ``states``, ``task_descriptions``.
        """
        full_images = inputs["full_images"]
        wrist_images = inputs["wrist_images"]
        states = inputs["states"]
        task_descriptions = inputs["task_descriptions"]
        B = full_images.shape[0]
        cfg = self.cfg

        videos, proprios, t5_list = [], [], []

        for i in range(B):
            cam = np.stack([wrist_images[i][:, ::-1], full_images[i][:, ::-1]])
            cam = _preprocess_cameras(
                cam, cfg.use_jpeg_compression, cfg.trained_with_image_aug
            )
            wrist, primary = cam[0], cam[1]
            blank = np.zeros_like(primary)
            dup = lambda arr: np.stack([arr] * COSMOS_TEMPORAL_COMPRESSION_FACTOR)  # noqa: E731
            blank4 = dup(blank)

            seq, idx = [], 0
            seq.append(blank[None])
            idx += 1
            seq.append(blank4.copy())
            proprio_idx = idx
            idx += 1
            seq.append(dup(wrist))
            wrist_idx = idx
            idx += 1
            seq.append(dup(primary))
            image_idx = idx
            idx += 1
            seq.append(blank4.copy())
            action_idx = idx
            idx += 1
            seq.append(blank4.copy())
            fut_proprio_idx = idx
            idx += 1
            seq.append(dup(wrist))
            fut_wrist_idx = idx
            idx += 1
            seq.append(dup(primary))
            fut_image_idx = idx
            idx += 1
            seq.append(blank4.copy())
            value_idx = idx
            idx += 1

            video = np.transpose(np.concatenate(seq), (3, 0, 1, 2))
            videos.append(video)
            # Reorder raw state [eef_pos(3), eef_quat(4), gripper(2)]
            # to cosmos-policy format [gripper(2), eef_pos(3), eef_quat(4)]
            s = states[i]
            proprio_9d = np.concatenate([s[7:], s[:3], s[3:7]])
            proprios.append(_rescale_proprio(proprio_9d, self.dataset_stats))

            emb = self.t5_cache.get(task_descriptions[i])
            if emb is None:
                emb = next(iter(self.t5_cache.values()))
                logger.warning(
                    f"T5 cache miss for '{task_descriptions[i]}', using fallback."
                )
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
            "current_wrist_image2_latent_idx",
            "current_image2_latent_idx",
            "future_wrist_image2_latent_idx",
            "future_image2_latent_idx",
        ]

        data_batch: Dict[str, Any] = {
            "dataset_name": "video_data",
            "video": torch.from_numpy(np.stack(videos)).to(
                dtype=torch.uint8, device="cuda"
            ),
            "t5_text_embeddings": torch.stack(t5_list).to(
                dtype=torch.bfloat16, device="cuda"
            ),
            "fps": torch.full((B,), 16, dtype=torch.bfloat16, device="cuda"),
            "padding_mask": torch.zeros(
                B,
                1,
                COSMOS_IMAGE_SIZE,
                COSMOS_IMAGE_SIZE,
                dtype=torch.bfloat16,
                device="cuda",
            ),
            "num_conditional_frames": self.generative_model.config.min_num_conditional_frames,
            "proprio": torch.from_numpy(np.stack(proprios))
            .reshape(B, -1)
            .to(dtype=torch.bfloat16, device="cuda"),
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

        Returns:
            ``{"action": tensor}`` with shape ``(B, chunk_size, action_dim)``.
        """
        cfg = self.cfg
        B = inputs["video"].shape[0]

        generated = self.generative_model.generate_samples_from_batch(
            inputs,
            n_sample=B,
            num_steps=cfg.num_denoising_steps,
            seed=cfg.seed,
            is_negative_prompt=False,
            use_variance_scale=cfg.use_variance_scale,
        )

        actions_np = (
            _extract_action_chunk(
                generated,
                (cfg.chunk_size, ACTION_DIM),
                inputs["action_latent_idx"],
            )
            .to(torch.float32)
            .cpu()
            .numpy()
        )
        actions_np = _unnormalize_actions(actions_np, self.dataset_stats)
        actions_np[..., -1] = np.sign(actions_np[..., -1])
        return {"action": torch.from_numpy(actions_np)}
