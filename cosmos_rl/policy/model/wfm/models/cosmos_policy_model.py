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
Inference-only model for Cosmos Policy robot tasks (e.g. LIBERO).
Reuses cosmos-rl WFM components: MinimalV1LVGDiT, Wan2pt1VAEInterface,
Video2WorldConditioner, EDMSDE, RectifiedFlowScaling, and sampler primitives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from cosmos_rl.policy.config.wfm import ConditionerConfig, NetConfig
from cosmos_rl.policy.model.wfm.conditioner.video2world import (
    Video2WorldCondition,
    Video2WorldConditioner,
)
from cosmos_rl.policy.model.wfm.networks.minimal_v1_lvg_dit import MinimalV1LVGDiT
from cosmos_rl.policy.model.wfm.sampler import EDMSDE, RectifiedFlowScaling
from cosmos_rl.policy.model.wfm.sampler.cosmos_policy_sampler import CosmosPolicySampler
from cosmos_rl.policy.model.wfm.tokenizer.wan2pt1 import Wan2pt1VAEInterface
from cosmos_rl.utils.wfm.utils import DataType, DenoisePrediction

ACTION_DIM = 7
COSMOS_IMAGE_SIZE = 224
COSMOS_TEMPORAL_COMPRESSION_FACTOR = 4


def arch_invariant_rand(shape, dtype, device, seed=None):
    """GPU-architecture-invariant random tensor (matches cosmos-policy)."""
    rng = np.random.RandomState(seed)
    return torch.from_numpy(rng.standard_normal(shape).astype(np.float32)).to(dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Latent injection / extraction helpers
# ---------------------------------------------------------------------------

def replace_latent_with_flat(x0: torch.Tensor, flat_data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Replace the latent volume at ``indices`` with ``flat_data`` (action chunk or proprio)."""
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


def extract_action_chunk(generated: torch.Tensor, action_shape: Tuple[int, int], action_indices: torch.Tensor) -> torch.Tensor:
    batch_indices = torch.arange(generated.shape[0], device=generated.device)
    latent = generated[batch_indices, :, action_indices, :, :]
    flat = latent.reshape(generated.shape[0], -1)
    n_elements = action_shape[0] * action_shape[1]
    return flat[:, :n_elements].reshape(generated.shape[0], *action_shape)


def extract_value(generated: torch.Tensor, value_indices: torch.Tensor) -> torch.Tensor:
    batch_indices = torch.arange(generated.shape[0], device=generated.device)
    latent = generated[batch_indices, :, value_indices, :, :]
    return latent.reshape(generated.shape[0], -1).mean(dim=1)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CosmosPolicyModelConfig:
    """Lightweight config for inference-only CosmosPolicyModel."""

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
    tokenizer_weights: str = "hf://nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth"
    vae_fp32: bool = False
    conditioner: List[ConditionerConfig] = field(default_factory=lambda: [
        ConditionerConfig(name="fps", type="remap_key", input_key="fps", output_key="fps", dropout_rate=0.0),
        ConditionerConfig(name="padding_mask", type="remap_key", input_key="padding_mask", output_key="padding_mask", dropout_rate=0.0),
        ConditionerConfig(name="text", type="text_attr", input_key=["t5_text_embeddings"], output_key="crossattn_emb", dropout_rate=0.0),
        ConditionerConfig(name="use_video_condition", type="boolean_flag", input_key="fps", output_key="use_video_condition", dropout_rate=0.0),
    ])
    net: NetConfig = field(default_factory=lambda: NetConfig(atten_backend="minimal_a2a"))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CosmosPolicyModel(nn.Module):
    """Inference-only model for Cosmos Policy robot tasks.

    Reuses cosmos-rl WFM components directly:
    - MinimalV1LVGDiT (DiT backbone)
    - Wan2pt1VAEInterface (VAE tokenizer)
    - Video2WorldConditioner (T5 text + video conditioning)
    - EDMSDE / RectifiedFlowScaling (noise schedule)
    - CosmosPolicySampler (sampler with clean-step logic)

    The denoise method is ported from Vid2VidModel; get_x0_fn_from_batch
    adds policy-specific logic (proprio injection, mask manipulation,
    uncondition=None handling).
    """

    def __init__(self, config: CosmosPolicyModelConfig):
        super().__init__()
        self.config = config
        self.sigma_data = config.sigma_data

        precision_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        self.precision = precision_map[config.precision]
        self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}

        self.sde = EDMSDE(
            p_mean=math.log(4.0),
            p_std=1.2,
            sigma_max=config.sigma_max,
            sigma_min=config.sigma_min,
        )
        self.scaling = RectifiedFlowScaling(
            config.sigma_data,
            config.rectified_flow_t_scaling_factor,
            config.rectified_flow_loss_weight_uniform,
        ) if config.scaling == "rectified_flow" else None
        self.sampler = CosmosPolicySampler()

        self.conditioner = Video2WorldConditioner(config.conditioner)
        net_cfg = config.net.model_copy()
        with torch.device("meta"):
            self.net = MinimalV1LVGDiT(**net_cfg.model_dump())
        self.net.to_empty(device="cuda")
        self.net.init_weights()
        self.net.to(**self.tensor_kwargs)
        # Tokenizer is not an nn.Module – manages its own weights
        self._tokenizer = Wan2pt1VAEInterface(
            chunk_duration=config.tokenizer_chunk_duration,
        )
        if config.vae_fp32:
            self._promote_vae_to_fp32()

    def _promote_vae_to_fp32(self):
        """Reload VAE weights from the original fp32 checkpoint.

        Wan2pt1VAEInterface hard-codes dtype=bfloat16, which truncates the
        original float32 checkpoint weights.  We reload them here so the
        encoder/decoder run in true float32 precision.
        """
        import os
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

    # ------------------------------------------------------------------
    # Core methods (ported from Vid2VidModel)
    # ------------------------------------------------------------------

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
        """Encode video into latent space and build conditioner output."""
        self._normalize_video(data_batch)
        video = data_batch["video"]
        latent = self._tokenize(video)
        condition = self.conditioner.get_condition_uncondition(data_batch)[0]
        return video, latent, condition

    def denoise(
        self,
        xt_B_C_T_H_W: torch.Tensor,
        sigma: torch.Tensor,
        condition: Video2WorldCondition,
    ) -> DenoisePrediction:
        """Single denoising step (ported from Vid2VidModel.denoise)."""
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
            cond_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(net_input)

            # FRAME_REPLACE strategy
            net_input = cond_input * cond_mask + net_input * (1 - cond_mask)
            if not self.config.use_rf_ckpts:
                sigma_cond = torch.ones_like(sigma_B_1_T_1_1) * self.config.sigma_conditional
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
            x0_pred = condition.gt_frames.type_as(x0_pred) * cond_mask + x0_pred * (1 - cond_mask)

        eps_pred = (xt_B_C_T_H_W - x0_pred) / sigma_B_1_T_1_1
        return DenoisePrediction(x0_pred, eps_pred, None)

    def get_x0_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
        skip_vae_encoding: bool = False,
        previous_generated_latent: Optional[torch.Tensor] = None,
        return_orig_clean_latent_frames: bool = False,
    ):
        """Build x0_fn closure for the sampler with policy-specific logic."""
        num_conditional_frames = data_batch.get("num_conditional_frames", 1)
        ncf_min = self.config.min_num_conditional_frames
        ncf_max = self.config.max_num_conditional_frames

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        condition.data_type = DataType.VIDEO
        if uncondition is not None:
            uncondition.data_type = DataType.VIDEO

        if skip_vae_encoding:
            assert previous_generated_latent is not None
            x0 = previous_generated_latent.clone()
        else:
            _, x0, _ = self.get_data_and_condition(data_batch)

        condition = condition.set_video_condition(x0, ncf_min, ncf_max, num_conditional_frames)
        if uncondition is not None:
            uncondition = uncondition.set_video_condition(x0, ncf_min, ncf_max, num_conditional_frames)

        condition = condition.edit_for_inference(is_cfg_conditional=True, num_conditional_frames=num_conditional_frames)
        if uncondition is not None:
            uncondition = uncondition.edit_for_inference(is_cfg_conditional=False, num_conditional_frames=num_conditional_frames)

        orig_gt_frames = condition.gt_frames.clone()

        B = condition.condition_video_input_mask_B_C_T_H_W.shape[0]

        # Inject proprio into gt_frames and mark as conditioning
        if "proprio" in data_batch and data_batch["proprio"] is not None:
            prop_idx = data_batch.get("current_proprio_latent_idx")
            if prop_idx is not None and torch.all(prop_idx != -1):
                batch_idx = torch.arange(B, device=prop_idx.device)
                condition.condition_video_input_mask_B_C_T_H_W[batch_idx, :, prop_idx, :, :] = 1
                condition.gt_frames = replace_latent_with_flat(
                    condition.gt_frames, data_batch["proprio"], prop_idx
                )
                if uncondition is not None:
                    uncondition.condition_video_input_mask_B_C_T_H_W[batch_idx, :, prop_idx, :, :] = 1
                    uncondition.gt_frames = replace_latent_with_flat(
                        uncondition.gt_frames, data_batch["proprio"], prop_idx
                    )

        # Mask manipulation for value prediction modes
        if data_batch.get("mask_current_state_action_for_value_prediction", False):
            batch_idx = torch.arange(B, device=condition.condition_video_input_mask_B_C_T_H_W.device)
            for k in ["current_proprio_latent_idx", "current_wrist_image_latent_idx",
                       "current_wrist_image2_latent_idx", "current_image_latent_idx",
                       "current_image2_latent_idx", "action_latent_idx"]:
                if k in data_batch and torch.all(data_batch[k] != -1):
                    condition.condition_video_input_mask_B_C_T_H_W[batch_idx, :, data_batch[k], :, :] = 0

        def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            cond_x0 = self.denoise(noise_x, sigma, condition).x0
            if uncondition is not None:
                uncond_x0 = self.denoise(noise_x, sigma, uncondition).x0
                return cond_x0 + guidance * (cond_x0 - uncond_x0)
            return cond_x0

        if return_orig_clean_latent_frames:
            return x0_fn, orig_gt_frames
        return x0_fn

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
        """Generate samples – policy-specific version with CosmosPolicySampler logic."""
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
                data_batch, guidance, is_negative_prompt,
                skip_vae_encoding=skip_vae_encoding,
                previous_generated_latent=previous_generated_latent,
                return_orig_clean_latent_frames=True,
            )
        else:
            x0_fn = self.get_x0_fn_from_batch(
                data_batch, guidance, is_negative_prompt,
                skip_vae_encoding=skip_vae_encoding,
                previous_generated_latent=previous_generated_latent,
            )

        sigma_max_scale = 1.0
        sigma_min_scale = 1.0
        if use_variance_scale:
            torch.manual_seed(seed)
            sigma_max_scale = torch.rand(1).item() * 2.0 + 1.0
            sigma_min_scale = torch.rand(1).item() * 0.9 + 0.1

        x_sigma_max = arch_invariant_rand(
            (n_sample,) + tuple(state_shape), torch.float32, "cuda", seed,
        ) * self.sde.sigma_max * sigma_max_scale

        samples = self.sampler(
            x0_fn, x_sigma_max,
            num_steps=num_steps,
            sigma_max=self.sde.sigma_max * sigma_max_scale,
            sigma_min=self.sde.sigma_min * sigma_min_scale,
            solver_option=solver_option,
        )

        if return_orig_clean_latent_frames:
            return samples, orig_clean
        return samples


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def get_default_libero_config() -> CosmosPolicyModelConfig:
    """Return config matching the cosmos-policy LIBERO Predict2-2B checkpoint."""
    return CosmosPolicyModelConfig(
        sigma_data=1.0,
        sigma_max=80.0,
        sigma_min=4.0,
        sigma_conditional=0.0,
        state_ch=16,
        state_t=9,
        min_num_conditional_frames=4,
        max_num_conditional_frames=4,
        conditioning_strategy="frame_replace",
        denoise_replace_gt_frames=True,
        use_rf_ckpts=False,
        precision="bfloat16",
        scaling="rectified_flow",
        tokenizer_chunk_duration=33,
        adjust_video_noise=True,
        atten_backend="minimal_a2a",
        net=NetConfig(atten_backend="minimal_a2a"),
    )


def load_cosmos_policy_model(
    ckpt_path: str,
    config: Optional[CosmosPolicyModelConfig] = None,
    device: str = "cuda",
) -> Tuple[CosmosPolicyModel, CosmosPolicyModelConfig]:
    """Load checkpoint into a CosmosPolicyModel.

    Args:
        ckpt_path: Path to .pt checkpoint or directory with model/ subfolder.
        config: Model config. Defaults to LIBERO config.
        device: Device to load model to.

    Returns:
        (model, config) tuple.
    """
    if config is None:
        config = get_default_libero_config()

    model = CosmosPolicyModel(config)

    from cosmos_rl.utils.wfm.io.easy_io import easy_io

    state_dict = easy_io.load(ckpt_path, weights_only=False)
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    elif isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load_cosmos_policy_model] Missing keys (first 10): {missing[:10]}")
    if unexpected:
        print(f"[load_cosmos_policy_model] Unexpected keys (first 10): {unexpected[:10]}")

    model.eval()
    model.to(device)
    return model, config
