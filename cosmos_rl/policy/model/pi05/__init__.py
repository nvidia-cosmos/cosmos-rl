from cosmos_rl.policy.model.base import BaseModel, ModelRegistry
from cosmos_rl.policy.model.pi05.weight_mapper import Pi05WeightMapper

# weight mapper, weight converter, data packer, parallelize, to do in this file.
from cosmos_rl.policy.model.pi05.model_utils import get_config
from cosmos_rl.policy.model.pi05.model_utils import PaliGemmaWithExpertModel
from cosmos_rl.policy.model.pi05.explore_noise_net import ExploreNoiseNet

import logging
import math
import os
import random

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Callable, List, Optional, Tuple, Literal

from collections.abc import Sequence

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.util import resolve_model_path
from transformers import AutoConfig
from safetensors import safe_open



IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

IMAGE_RESOLUTION = (224, 224)


def preprocess_observation_pytorch(
    observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
):
    """Torch.compile-compatible version of preprocess_observation_pytorch with simplified type annotations.

    This function avoids complex type annotations that can cause torch.compile issues.
    """
    if not set(image_keys).issubset(observation.images):
        raise ValueError(
            f"images dict missing keys: expected {image_keys}, got {list(observation.images)}"
        )

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]

        # Handle both [B, C, H, W] and [B, H, W, C] formats
        is_channels_first = image.shape[1] == 3  # Check if channels are in dimension 1

        if is_channels_first:
            # Convert [B, C, H, W] to [B, H, W, C] for processing
            image = image.permute(0, 2, 3, 1)

        # Align with OpenPI Observation.from_dict(): if image is uint8 in [0,255],
        # convert to float32 in [-1, 1] before any resizing/augmentations.
        if image.dtype == torch.uint8:
            image = image.to(torch.float32) / 255.0 * 2.0 - 1.0

        if image.shape[1:3] != image_resolution:
            logging.info(
                f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}"
            )
            image = resize_with_pad_torch(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for PyTorch augmentations
            image = image / 2.0 + 0.5

            # Apply PyTorch-based augmentations
            if "wrist" not in key:
                # Geometric augmentations for non-wrist cameras
                height, width = image.shape[1:3]

                # Random crop and resize
                crop_height = int(height * 0.95)
                crop_width = int(width * 0.95)

                # Random crop
                max_h = height - crop_height
                max_w = width - crop_width
                if max_h > 0 and max_w > 0:
                    # Use tensor operations instead of .item() for torch.compile compatibility
                    start_h = torch.randint(0, max_h + 1, (1,), device=image.device)
                    start_w = torch.randint(0, max_w + 1, (1,), device=image.device)
                    image = image[
                        :,
                        start_h : start_h + crop_height,
                        start_w : start_w + crop_width,
                        :,
                    ]

                # Resize back to original size
                image = torch.nn.functional.interpolate(
                    image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w]
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

                # Random rotation (small angles)
                # Use tensor operations instead of .item() for torch.compile compatibility
                angle = (
                    torch.rand(1, device=image.device) * 10 - 5
                )  # Random angle between -5 and 5 degrees
                if torch.abs(angle) > 0.1:  # Only rotate if angle is significant
                    # Convert to radians
                    angle_rad = angle * torch.pi / 180.0

                    # Create rotation matrix
                    cos_a = torch.cos(angle_rad)
                    sin_a = torch.sin(angle_rad)

                    # Apply rotation using grid_sample
                    grid_x = torch.linspace(-1, 1, width, device=image.device)
                    grid_y = torch.linspace(-1, 1, height, device=image.device)

                    # Create meshgrid
                    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")

                    # Expand to batch dimension
                    grid_x = grid_x.unsqueeze(0).expand(image.shape[0], -1, -1)
                    grid_y = grid_y.unsqueeze(0).expand(image.shape[0], -1, -1)

                    # Apply rotation transformation
                    grid_x_rot = grid_x * cos_a - grid_y * sin_a
                    grid_y_rot = grid_x * sin_a + grid_y * cos_a

                    # Stack and reshape for grid_sample
                    grid = torch.stack([grid_x_rot, grid_y_rot], dim=-1)

                    image = torch.nn.functional.grid_sample(
                        image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w]
                        grid,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

            # Color augmentations for all cameras
            # Random brightness
            # Use tensor operations instead of .item() for torch.compile compatibility
            brightness_factor = (
                0.7 + torch.rand(1, device=image.device) * 0.6
            )  # Random factor between 0.7 and 1.3
            image = image * brightness_factor

            # Random contrast
            # Use tensor operations instead of .item() for torch.compile compatibility
            contrast_factor = (
                0.6 + torch.rand(1, device=image.device) * 0.8
            )  # Random factor between 0.6 and 1.4
            mean = image.mean(dim=[1, 2, 3], keepdim=True)
            image = (image - mean) * contrast_factor + mean

            # Random saturation (convert to HSV, modify S, convert back)
            # For simplicity, we'll just apply a random scaling to the color channels
            # Use tensor operations instead of .item() for torch.compile compatibility
            saturation_factor = (
                0.5 + torch.rand(1, device=image.device) * 1.0
            )  # Random factor between 0.5 and 1.5
            gray = image.mean(dim=-1, keepdim=True)
            image = gray + (image - gray) * saturation_factor

            # Clamp values to [0, 1]
            image = torch.clamp(image, 0, 1)

            # Back to [-1, 1]
            image = image * 2.0 - 1.0

        # Convert back to [B, C, H, W] format if it was originally channels-first
        if is_channels_first:
            image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = torch.ones(
                batch_shape, dtype=torch.bool, device=observation.state.device
            )
        else:
            out_masks[key] = observation.image_masks[key]

    # Create a simple object with the required attributes instead of using the complex Observation class
    class SimpleProcessedObservation:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    return SimpleProcessedObservation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
    )


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device=torch.device("cpu"),
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


from cosmos_rl.dispatcher.data.packer.pi_data_packer import PIDataPacker


@ModelRegistry.register(Pi05WeightMapper, default_data_packer_cls=PIDataPacker)
class PI05(BaseModel):
    def __init__(self, model_name_or_path: str, hf_config):
        """
        Initialize PI05 model.

        Args:
            model_name_or_path: Path to pretrained model or model name
            hf_config: HuggingFace-style config (can be AutoConfig or Pi05ModelConfig)
        """
        # Initialize base class with hf_config for weight_mapper
        super().__init__(hf_config)

        self.hf_config = hf_config
        self.model_name_or_path = model_name_or_path

        # Extract pi05-specific config from hf_config
        self.pi05 = hf_config.pi05
        self.action_dim = hf_config.action_dim
        self.action_horizon = hf_config.action_horizon

        self.num_steps = hf_config.num_steps
        self.action_chunk = hf_config.action_chunk
        self.action_env_dim = hf_config.action_env_dim
        # Optional knobs: if not specified in toml, fall back to code defaults.
        self.noise_method = getattr(hf_config, "noise_method", "flow_sde")
        self.noise_level = getattr(hf_config, "noise_level", 0.5)
        self.noise_anneal = getattr(hf_config, "noise_anneal", False)
        self.noise_params = getattr(hf_config, "noise_params", [0.7, 0.3, 400])
        self.noise_logvar_range = getattr(hf_config, "noise_logvar_range", [0.08, 0.16])
        self.joint_logprob = getattr(hf_config, "joint_logprob", False)
        self.safe_get_logprob = getattr(hf_config, "safe_get_logprob", False)
        self.ignore_last = getattr(hf_config, "ignore_last", False)
        self.train_expert_only = hf_config.train_expert_only
        self.discrete_state_input = hf_config.discrete_state_input
        self.max_token_len = hf_config.max_token_len
        self.global_step = 0  # Used for noise annealing
        paligemma_variant = hf_config.paligemma_variant
        action_expert_variant = hf_config.action_expert_variant
        dtype = hf_config.dtype

        paligemma_config = get_config(paligemma_variant)
        action_expert_config = get_config(action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=dtype,
        )

        self.action_in_proj = nn.Linear(self.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, self.action_dim)

        if self.pi05:
            self.time_mlp_in = nn.Linear(
                action_expert_config.width, action_expert_config.width
            )
            self.time_mlp_out = nn.Linear(
                action_expert_config.width, action_expert_config.width
            )
        else:
            self.state_proj = nn.Linear(self.action_dim, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(
                2 * action_expert_config.width, action_expert_config.width
            )
            self.action_time_mlp_out = nn.Linear(
                action_expert_config.width, action_expert_config.width
            )

        torch.set_float32_matmul_precision("high")
        if getattr(hf_config, "cosmos_compile", False):
            self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        if self.noise_method == "flow_noise":
            self.noise_head = ExploreNoiseNet(
                in_dim=1024,
                out_dim=self.action_dim,
                hidden_dims=[128, 64],
                activation_type="tanh",
                noise_logvar_range=self.noise_logvar_range,
                noise_scheduler_type="learn",
            )


    def get_trained_model_state_dict(self):
        return {n: p for n, p in self.named_parameters() if p.requires_grad}

    @staticmethod
    def preprocess_hf_config(config):
        """
        Load the minimal PI05 `config.json` from `model_name_or_path` and inject runtime-only
        GRPO/OpenPI knobs from Cosmos config (RLinf-style).

        This keeps checkpoints clean (only core model fields in config.json), while still allowing
        training-time overrides via Cosmos config.
        """
        hf_config = AutoConfig.from_pretrained(
            config.policy.model_name_or_path, trust_remote_code=True
        )
        hf_config.cosmos_compile = bool(getattr(config.train, "compile", False))

        # Unified assignment via the toml's custom field
        if hasattr(config, "custom") and isinstance(config.custom, dict):
            for k, v in config.custom.items():
                setattr(hf_config, k, v)

        return hf_config

    def set_global_step(self, global_step: int):
        """Set global training step (used for noise annealing)."""
        self.global_step = global_step

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = (
            True
        )
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = (
            False
        )
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def freeze_vlm(self):
        """Freeze VLM (PaliGemma) and only train action expert (RLinf-style)."""
        if self.train_expert_only:
            self.paligemma_with_expert.paligemma.eval()
            for params in self.paligemma_with_expert.paligemma.parameters():
                params.requires_grad = False
            logging.info("Frozen PaliGemma VLM, only training action expert")

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = preprocess_observation_pytorch(observation, train=train)
        imgs = list(observation.images.values())
        if imgs and imgs[0].ndim == 4 and imgs[0].shape[-1] == 3:
            imgs = [x.permute(0, 3, 1, 2).contiguous() for x in imgs]
        return (
            imgs,
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):
            # NOTE: SigLIP vision tower expects NCHW (B, 3, H, W).
            # Online rollout path usually goes through `_preprocess_observation()` which already converts,
            # but GRPO replay/data-packer path can pass NHWC tensors here.
            if torch.is_tensor(img):
                if img.dtype == torch.uint8:
                    img = img.to(torch.float32) / 255.0 * 2.0 - 1.0
                if img.ndim == 4 and img.shape[-1] == 3 and img.shape[1] != 3:
                    img = img.permute(0, 3, 1, 2).contiguous()
                elif img.ndim == 3 and img.shape[-1] == 3 and img.shape[0] != 3:
                    img = img.permute(2, 0, 1).unsqueeze(0).contiguous()

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(
            bsize, action_time_dim, dtype=torch.bool, device=timestep.device
        )
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """SFT Forward Pass for training."""
        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=True)
        )

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(state, x_t, time)
        )
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[
                0
            ].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(
            prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        ):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func,
            prefix_embs,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
        )

        suffix_out = suffix_out[:, -self.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, mode="train") -> dict:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)
        
        Matches RLinf's OpenPi0ForRLActionPrediction.sample_actions() interface.
        
        Args:
            device: torch device
            observation: Observation object
            noise: Optional initial noise
            num_steps: Number of denoise steps
            mode: "train" or "eval" - controls whether to collect chains for GRPO
            
        Returns:
            dict with 'actions', 'chains', 'denoise_inds'
        """
        fp = os.path.expanduser("~/pi05_first_input.pt")
        dbg = str(os.getenv("DEBUG", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}
        fixed = torch.load(fp, map_location="cpu", weights_only=False) if (dbg and os.path.exists(fp)) else None
        if isinstance(fixed, dict) and all(k in fixed for k in ("images", "img_masks", "lang_tokens", "lang_masks", "state")):
            images = [x.to(device=device) for x in fixed["images"]]
            img_masks = [x.to(device=device) for x in fixed["img_masks"]]
            lang_tokens = fixed["lang_tokens"].to(device=device)
            lang_masks = fixed["lang_masks"].to(device=device)
            state = fixed["state"].to(device=device)
            bsize = int(state.shape[0])
            if noise is None:
                noise = fixed.get("noise_init", None)
                if torch.is_tensor(noise):
                    noise = noise.to(device=device)
            if noise is None:
                actions_shape = (bsize, self.action_horizon, self.action_dim)
                noise = self.sample_noise(actions_shape, device)
        else:
            bsize = observation.state.shape[0]
            if noise is None:
                actions_shape = (bsize, self.action_horizon, self.action_dim)
                noise = self.sample_noise(actions_shape, device)
            images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
            if (
                str(os.getenv("SAVE_FIRST_INPUT", os.getenv("save_first_input", "0"))).strip().lower()
                in {"1", "true", "yes", "y", "on"}
                and int(os.getenv("RANK", os.getenv("LOCAL_RANK", "0")) or 0) == 0
                and not getattr(self, "_pi05_first_input_saved", False)
            ):
                if not os.path.exists(fp):
                    torch.save(
                        {
                            "images": [x.detach().cpu() for x in images],
                            "img_masks": [x.detach().cpu() for x in img_masks],
                            "lang_tokens": lang_tokens.detach().cpu(),
                            "lang_masks": lang_masks.detach().cpu(),
                            "state": state.detach().cpu(),
                            "noise_init": noise.detach().cpu() if torch.is_tensor(noise) else None,
                        },
                        fp,
                    )
                setattr(self, "_pi05_first_input_saved", True)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        x_t = noise
        chains = [x_t]
        log_probs = []

        if self.joint_logprob:
            initial_log_prob = self.get_logprob_norm(
                x_t, torch.zeros_like(noise), torch.ones_like(noise)
            )
            log_probs.append(initial_log_prob)

        # RLinf-style denoise index selection
        if mode == "train":
            if self.joint_logprob:
                denoise_inds = torch.arange(self.num_steps, device=device)
            else:
                if self.ignore_last:
                    denoise_inds = torch.tensor(
                        [random.randint(0, self.num_steps - 2)] * self.num_steps, device=device
                    )
                else:
                    denoise_inds = torch.tensor(
                        [random.randint(0, self.num_steps - 1)] * self.num_steps, device=device
                    )
        else:
            denoise_inds = torch.tensor([-1] * self.num_steps, device=device)
        denoise_inds = denoise_inds[None].repeat(bsize, 1)

        for idx in range(self.num_steps):
            if idx == denoise_inds[0][idx]:
                sample_mode = "train"
            else:
                sample_mode = "eval"

            x_t_mean, x_t_std = self.sample_mean_var_val(
                x_t,
                idx,
                state,
                prefix_pad_masks,
                past_key_values,
                sample_mode,
                self.num_steps,
            )
            
            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
            log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)
            chains.append(x_t)
            log_probs.append(log_prob)
        x_0 = x_t
        chains = torch.stack(chains, dim=1)
        log_probs = torch.stack(log_probs, dim=1)[:, :, : self.action_chunk, : self.action_env_dim]
        if self.joint_logprob:
            log_probs = log_probs.mean(dim=1)
        else:
            log_probs = log_probs[torch.arange(log_probs.shape[0]), denoise_inds[:, 0]]

        x_0 = x_0[..., : self.action_env_dim] if self.action_env_dim <= self.action_dim else x_0

        return {
            "actions": x_0,
            "chains": chains,
            "old_log_probs": log_probs,
            "denoise_inds": denoise_inds,
        }

    def sample_mean_var_val(
        self,
        x_t,
        idx,
        state,
        prefix_pad_masks,
        past_key_values,
        mode,
        denoise_steps,
    ):
        """
        Sample the mean, variance and value of the action at a given timestep.
        Rollout sample (idx is int) and actor get_log_prob_value (idx is tensor) will load this function.
        """
        # expand the shape
        bsize = state.shape[0]
        device = state.device
        if isinstance(idx, int):
            idx = torch.tensor(idx).expand(bsize)
        # build parameters
        if self.noise_anneal:
            # noise annealing
            noise_start, noise_end, anneal_steps = self.noise_params
            noise_level = (
                noise_start
                + (noise_end - noise_start)
                * min(self.global_step, anneal_steps)
                / anneal_steps
            )
            noise_level = torch.tensor(noise_level).to(device)
        else:
            # fixed noise level
            noise_level = torch.tensor(self.noise_level).to(device)
        timesteps = torch.linspace(1, 1 / denoise_steps, denoise_steps, device=device)
        timesteps = torch.cat([timesteps, torch.tensor([0.0], device=device)])
        # input parameters
        t_input = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1]
        # velocity prediction
        suffix_out = self.get_suffix_out(
            state,
            prefix_pad_masks,
            past_key_values,
            x_t,
            t_input,
        )
        v_t = self.action_out_proj(suffix_out)  # [bs,n_action_steps,max_action_dim]

        # ode sde mix sampling
        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)
        if mode == "eval":
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta
            x_t_std = torch.zeros_like(t_input)
        elif mode == "train":
            if self.noise_method == "flow_sde":
                sigmas = (
                    noise_level
                    * torch.sqrt(
                        timesteps
                        / (1 - torch.where(timesteps == 1, timesteps[1], timesteps))
                    )[:-1]
                )
                sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = t_input - delta - sigma_i**2 * delta / (2 * t_input)
                x_t_std = torch.sqrt(delta) * sigma_i
            elif self.noise_method == "flow_cps":
                pi = torch.pi
                cos_term = torch.cos(pi * noise_level / 2).to(device)
                sin_term = torch.sin(pi * noise_level / 2).to(device)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = (t_input - delta) * cos_term
                x_t_std = (t_input - delta) * sin_term
            elif self.noise_method == "flow_noise":
                x0_weight = 1 - (t_input - delta)
                x1_weight = t_input - delta
                x_t_std = self.noise_head(suffix_out)
            else:
                raise ValueError(f"Invalid noise method: {self.noise_method}")
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std


    def get_suffix_out(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(state, x_t, timestep)
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = (
            "eager"  # noqa: SLF001
        )

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return suffix_out

    def _set_fsdp_reshard_after_forward(self, policy_str):
        logging.info("FSDP reshard_after_forward setting is not applicable for PI05 model.")
        pass

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(state, x_t, timestep)
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = (
            "eager"  # noqa: SLF001
        )

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

    def get_logprob_norm(self, sample, mu, sigma):
        """Compute Gaussian log probability: log p(x|mu,sigma)."""
        if self.safe_get_logprob:
            log_prob = -torch.pow((sample - mu), 2)
        else:
            mask = sigma == 0
            sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
            constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
                2 * torch.pi * torch.ones_like(sample)
            )
            exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
            log_prob = constant_term + exponent_term
            log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
        return log_prob

    def gaussian_entropy(self, sigma):
        """Compute Gaussian entropy: H = 0.5 * log(2 * pi * e * sigma^2)."""
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        entropy = 0.5 * torch.log(2 * math.pi * math.e * (sigma_safe**2))
        return entropy
    
    def get_log_prob_value(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        chains,
        denoise_inds,
        compute_values=False,
    ):
        bsize = state.shape[0]
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        # Compute image and language key value cache
        [prefix_output, _], past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        chains_log_probs = []
        chains_values = []
        chains_entropy = []

        # get log prob
        if self.joint_logprob:
            num_steps = self.num_steps
            initial_log_prob = self.get_logprob_norm(
                chains[:, 0],
                torch.zeros_like(chains[:, 0]),
                torch.ones_like(chains[:, 0]),
            )
            initial_entropy = self.gaussian_entropy(torch.ones_like(chains[:, 0]))
            chains_log_probs.append(initial_log_prob)
            chains_entropy.append(initial_entropy)
        else:
            num_steps = 1
        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            chains_pre = chains[torch.arange(bsize), denoise_ind]
            chains_next = chains[torch.arange(bsize), denoise_ind + 1]
            x_t_mean, x_t_std = self.sample_mean_var_val(
                chains_pre,
                denoise_ind,
                state,
                prefix_pad_masks,
                past_key_values,
                "train",
                self.num_steps,
            )
            log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            entropy = self.gaussian_entropy(x_t_std)
            chains_log_probs.append(log_probs)
            chains_entropy.append(entropy)
        chains_log_probs = torch.stack(chains_log_probs, dim=1)

        # entropy is only available for flow-noise method
        if self.noise_method == "flow_noise":
            chains_entropy = torch.stack(chains_entropy, dim=1)
        else:
            chains_entropy = torch.zeros_like(chains_log_probs)
        return chains_log_probs, chains_entropy


    @staticmethod
    def supported_model_types():
        return ["pi05", "pi0"]

    @property
    def parallelize_fn(self) -> Tuple[Callable, nn.Module]:
        from cosmos_rl.policy.model.pi05.parallelize import parallelize

        return parallelize, self

    def post_to_empty_hook(self, cosmos_config):
        # Apply training-time freezing policy here (requested): if `train_expert_only=True`,
        # freeze the VLM (PaliGemma) so optimizer won't include it.
        self.freeze_vlm()
        return

    def apply_pipeline_split(self, pp_rank, pp_size):
        raise NotImplementedError("PI05 does not support PP")

    def get_position_ids(self, **kwargs):
        raise NotImplementedError(
            "PI05 does not use token-based inputs; CP is not supported"
        )

    def separate_model_parts(self) -> List[nn.Module]:
        return [self]

    @classmethod
    def from_pretrained(
        cls,
        hf_config,
        model_name_or_path: str,
        max_position_embeddings: Optional[int] = None,
    ) -> "PI05":
        # Set max position embeddings if provided
        if max_position_embeddings is not None:
            hf_config.max_position_embeddings = max_position_embeddings
        model = cls(model_name_or_path, hf_config)
        return model

    def get_nparams_and_flops(self, seq_len: int) -> tuple[int, int]:
        return 0, 0

    def load_hf_weights(
        self,
        model_name_or_path: str,
        parallel_dims=None,
        device: torch.device = None,
        revision: Optional[str] = None,
    ):
        """
        Load PI05 weights from a HuggingFace-style checkpoint directory/repo.
        Simple DDP-compatible version that loads weights directly.
        """
        logger.info(f"Loading PI05 weights from {model_name_or_path}")

        # Resolve local path (downloads if HF repo_id provided)
        model_path = resolve_model_path(model_name_or_path, revision=revision)

        device = device or torch.device("cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device.index or torch.cuda.current_device())
        if any(p.is_meta for p in self.parameters()) or any(
            b.is_meta for b in self.buffers()
        ):
            self.to_empty(device=device)
        else:
            self.to(device)
        weight_path = os.path.join(model_path, "model.safetensors")

        state_dict = {}
        with safe_open(weight_path, framework="pt", device=("cuda" if device.type == "cuda" else "cpu")) as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        # logger.info(f'{state_dict.keys()}')
        # If embed_tokens is missing in the checkpoint, mirror lm_head (official tying).
        embed_key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
        lm_head_key = "paligemma_with_expert.paligemma.lm_head.weight"
        if embed_key not in state_dict and lm_head_key in state_dict:
            state_dict[embed_key] = state_dict[lm_head_key]

        missing, unexpected = self.load_state_dict(state_dict, strict=True)
        if missing:
            logger.warning(f"PI05 relaxed load: {len(missing)} missing keys. First 10: {missing[:10]}")
        if unexpected:
            logger.info(f"PI05 relaxed load: {len(unexpected)} unexpected keys (ignored)")


    def check_cp_compatible(self, cp_size: int, tp_size: int):
        raise ValueError("PI05 does not support context parallelism")

    def check_tp_compatible(self, tp_size: int):
        raise ValueError("PI05 does not support tensor parallelism")


def resize_with_pad_torch(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        # Convert to channels-first for torch operations
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else -1.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(
            0, 2, 3, 1
        )  # [b, c, h, w] -> [b, h, w, c]
        if batch_size == 1 and images.shape[0] == 1:
            padded_images = padded_images.squeeze(
                0
            )  # Remove batch dimension if it was added

    return padded_images
