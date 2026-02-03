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

import torch
import json
import os
import inspect
from functools import cached_property
from typing import Tuple, List, Optional, Dict, Any
from transformers import AutoConfig
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

from .processing_utils import (
    normalize_gripper_action,
    invert_gripper_action,
    obs_to_vla_input,
    center_crop_image,
)


from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.model.base import BaseModel, ModelRegistry
from cosmos_rl.policy.model.vla.weight_mapper import VLAWeightMapper
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils import util
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.utils.util import resolve_model_path


@ModelRegistry.register(VLAWeightMapper)
class OpenVLA(BaseModel):
    """
    VLA (Vision-Language-Action) Model for Embodied AI

    Supports loading SimpleVLA-RL models:
    - OpenVLA: Original OpenVLA models
    - OpenVLA-OFT: Models with Online Fine-Tuning support

    Following cosmos-rl Qwen VL pattern - direct nn.Module instantiation,
    no AutoModel complexity.
    """

    @staticmethod
    def supported_model_types():
        return ["openvla", "openvla-oft"]

    def __init__(self, model_name_or_path: str, hf_config: AutoConfig):
        """
        Initialize VLA model following cosmos-rl pattern

        Args:
            model_name_or_path: Model name or path to the pretrained model
            hf_config: HuggingFace configuration
        """
        super().__init__(hf_config)
        self.hf_config = hf_config

        # Create the actual VLA model instance directly (no AutoModel)
        if hf_config.vla_type == "openvla-oft":
            from cosmos_rl.policy.model.vla.openvla_oft.modeling_prismatic import (
                OpenVLAForActionPrediction,
            )
            from cosmos_rl.policy.model.vla.openvla_oft.processing_prismatic import (
                PrismaticProcessor,
            )
        else:  # openvla (default)
            from cosmos_rl.policy.model.vla.openvla.modeling_prismatic import (
                OpenVLAForActionPrediction,
            )
            from cosmos_rl.policy.model.vla.openvla.processing_prismatic import (
                PrismaticProcessor,
            )

        with torch.device("cpu"):
            self.processor = PrismaticProcessor.from_pretrained(
                model_name_or_path, trust_remote_code=True
            )
            self.tokenizer = self.processor.tokenizer

        # Create model with specified device (use "meta" for fast initialization)
        with torch.device("cuda"):
            self.model = OpenVLAForActionPrediction(self.hf_config).to(
                hf_config.torch_dtype
            )

        self.is_vlm = True
        self.norm_stats = self.hf_config.norm_stats

        self.model_input_keys = ["input_ids", "pixel_values", "attention_mask"]
        self.model_output_keys = ["responses", "old_log_probs"]
        self.model_train_keys = self.model_input_keys + self.model_output_keys

    @cached_property
    def model_forward_valid_kwargs(self):
        """Get valid keyword arguments for model forward pass"""
        sig = inspect.signature(self.model.forward)
        return sig.parameters.keys()

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        """Forward pass through VLA model"""
        # Filter to only valid kwargs
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if k in self.model_forward_valid_kwargs
        }

        # Prepare model inputs
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            **filtered_kwargs,
        }

        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values

        # Remove None values
        model_inputs = {k: v for k, v in model_inputs.items() if v is not None}

        try:
            outputs = self.model(**model_inputs)

            # OpenVLA-OFT returns raw logits tensor, not output object
            # Wrap it in a simple namespace to provide .logits attribute
            if isinstance(outputs, torch.Tensor):
                from types import SimpleNamespace

                outputs = SimpleNamespace(logits=outputs, logprobs=None, entropy=None)

            return outputs
        except Exception as e:
            logger.error(f"VLA model forward pass failed: {e}")
            logger.error(
                f"Input shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in model_inputs.items()]}"
            )
            raise

    def forward_with_trajectory_structure(
        self,
        input_ids: torch.Tensor,  # (batch, num_steps, seq_len)
        pixel_values: torch.Tensor,  # (batch, num_steps, C, H, W)
        attention_mask: torch.Tensor,  # (batch, num_steps, seq_len)
        labels: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        **kwargs,
    ):
        """
        Forward pass for VLA training with per-step trajectory structure.

        Matches SimpleVLA-RL approach:
        1. Forward pass to get logits
        2. Slice vocab to action tokens [vocab_size-256-64 : vocab_size-64]
        3. Apply temperature scaling

        Args:
            input_ids: (batch, num_steps, seq_len) - per-step input tokens
            pixel_values: (batch, num_steps, C, H, W) - per-step images
            attention_mask: (batch, num_steps, seq_len) - per-step masks
            temperature: Temperature for scaling logits (default: 1.0)
            return_action_logits_only: If True, slice vocab to action tokens (default: True)

        Returns:
            Output with logits: (batch*steps, output_len, 256) if return_action_logits_only
                          else: (batch*steps, output_len, vocab_size)
        """
        # if torch.distributed.get_rank() == 0:
        #     logger.info(f"input_ids {input_ids.shape}, {input_ids.dtype}, {input_ids}")
        #     logger.info(f"attention_mask {attention_mask.shape}, {attention_mask[0].dtype}, {attention_mask}")
        #     logger.info(f"pixel_values {pixel_values.shape}, {pixel_values[0].dtype}, {pixel_values}")
        #     logger.info(f"labels {labels.shape}, {labels[0].dtype}, {labels}")
        #     logger.info(f"temperature {temperature}")

        # Use autocast to compute in bfloat16 (params are stored in float32 via master_dtype)
        outputs = self.forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            **kwargs,
        )

        # Get raw logits: (batch*steps, output_len, vocab)
        logits = outputs.logits

        # Slice vocab to action tokens: [vocab_size-256-64 : vocab_size-64]
        # This extracts the 256 action tokens from the full vocabulary
        vocab_size = logits.shape[-1]
        start_index = vocab_size - 256 - 64
        logits = logits[..., start_index : start_index + 256]
        labels_remapped = labels - start_index

        # Apply temperature scaling
        logits = logits.div(temperature)

        # Compute entropy: -sum(p * log(p))
        probs = F.softmax(logits, dim=-1)

        logp = F.log_softmax(logits, dim=-1)
        logpy = torch.gather(logp, dim=-1, index=labels_remapped.unsqueeze(-1))
        logpy = logpy.squeeze(-1)
        entropy = -(probs * logp).sum(dim=-1)  # (batch, seq_len)

        # if torch.distributed.get_rank() == 0:
        #     logger.info(f"rollout_log_probs {logp.shape}, rollout_log_probs {logp}")

        outputs.logits = logits
        outputs.entropy = entropy
        outputs.logprobs = logpy
        return outputs

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save VLA model to directory"""
        self.model.save_pretrained(save_directory, **kwargs)
        if self.processor is not None:
            self.processor.save_pretrained(save_directory)
        logger.info(f"Saved VLA model to {save_directory}")

    @classmethod
    def preprocess_hf_config(cls, cosmos_config: CosmosConfig):
        vla_type = cosmos_config.vla.vla_type
        if vla_type == "openvla-oft":
            from cosmos_rl.policy.model.vla.openvla_oft.configuration_prismatic import (
                OpenVLAConfig,
            )
            from cosmos_rl.policy.model.vla.openvla_oft.processing_prismatic import (
                PrismaticProcessor,
            )
        else:
            from cosmos_rl.policy.model.vla.openvla.configuration_prismatic import (
                OpenVLAConfig,
            )
            from cosmos_rl.policy.model.vla.openvla.processing_prismatic import (
                PrismaticProcessor,
            )
        AutoConfig.register("openvla", OpenVLAConfig)
        name_or_path = cosmos_config.policy.model_name_or_path
        hf_config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=True)
        processor = PrismaticProcessor.from_pretrained(
            name_or_path, trust_remote_code=True
        )
        tokenizer = processor.tokenizer

        cosmos_default_dtype = util.str2torch_dtype(
            cosmos_config.train.master_dtype
            if cosmos_config.train.master_dtype is not None
            else cosmos_config.train.param_dtype
        )
        hf_config.torch_dtype = cosmos_default_dtype

        override_hf_config_kwargs = {
            "use_proprio": cosmos_config.vla.use_proprio,
            "proprio_dim": cosmos_config.vla.proprio_dim,
            "num_images_in_input": cosmos_config.vla.num_images_in_input,
            "vla_type": vla_type,
        }
        if tokenizer:
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = (
                    tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
                )
            override_hf_config_kwargs["bos_token_id"] = tokenizer.bos_token_id
            override_hf_config_kwargs["eos_token_id"] = tokenizer.eos_token_id
            override_hf_config_kwargs["pad_token_id"] = tokenizer.pad_token_id

        norm_stats = hf_config.norm_stats if hasattr(hf_config, "norm_stats") else None
        try:
            local_model_path = resolve_model_path(name_or_path)
            dataset_statistics_path = os.path.join(
                local_model_path, "dataset_statistics.json"
            )
            if not os.path.exists(dataset_statistics_path):
                from huggingface_hub import hf_hub_download

                dataset_statistics_path = hf_hub_download(
                    repo_id=name_or_path,
                    filename="dataset_statistics.json",
                    repo_type="model",
                )
            with open(dataset_statistics_path, "r") as f:
                norm_stats = json.load(f)
        except Exception as e:
            logger.warning(
                f"Failed to download dataset statistics from HuggingFace: {e}. "
                "This may cause incorrect action normalization."
            )

        action_dim = cosmos_config.vla.proprio_dim
        normalized_stats = {}
        for dataset_key, dataset_stats in norm_stats.items():
            if "action" not in dataset_stats:
                normalized_stats[dataset_key] = dataset_stats
                continue

            action_stats = dataset_stats["action"].copy()

            # Check and fix shape of action statistics
            for stat_key in ["min", "max", "q01", "q99", "mask"]:
                import numpy as np

                if stat_key not in action_stats:
                    continue

                stat_array = np.array(action_stats[stat_key])

                # If stats are concatenated across chunks (e.g., 14 = 2 * 7), slice to first action_dim
                # Only normalize if array is significantly larger (at least 1.5x)
                if (
                    len(stat_array.shape) == 1
                    and stat_array.shape[0] > action_dim * 1.5
                ):
                    logger.info(
                        f"Normalizing {dataset_key}.action.{stat_key}: "
                        f"shape {stat_array.shape} -> ({action_dim},) [concatenated chunks detected]"
                    )
                    action_stats[stat_key] = stat_array[:action_dim].tolist()
                else:
                    # Keep as-is (already correct shape)
                    action_stats[stat_key] = stat_array.tolist()

            # Reconstruct dataset stats with normalized action stats
            normalized_stats[dataset_key] = {**dataset_stats, "action": action_stats}

        override_hf_config_kwargs["norm_stats"] = normalized_stats

        for key, value in override_hf_config_kwargs.items():
            setattr(hf_config, key, value)

        return hf_config

    @classmethod
    def from_pretrained(
        cls,
        cosmos_config: CosmosConfig,
        hf_config: AutoConfig,
        model_name_or_path: str,
        max_position_embeddings: Optional[int] = None,
    ) -> "OpenVLA":
        """
        Initialize a VLA model from a pretrained model (following cosmos-rl pattern)

        Args:
            hf_config: HuggingFace configuration
            model_name_or_path: Model name or path to the pretrained model
            max_position_embeddings: Override max position embeddings

        Returns:
            VLAModel: VLA model instance with weights loaded
        """
        # Set max position embeddings if provided
        if max_position_embeddings is not None:
            hf_config.max_position_embeddings = max_position_embeddings
        model = cls(model_name_or_path, hf_config)
        return model

    @property
    def parallelize_fn(self):
        """Get parallelization function for VLA model"""
        from cosmos_rl.policy.model.vla.parallelize import parallelize_vla_model

        return parallelize_vla_model, self

    def apply_pipeline_split(self, pp_rank: int, pp_size: int):
        """Apply pipeline parallelism split"""
        if pp_size <= 1:
            return
        logger.warning("Pipeline parallelism not yet implemented for VLA models")

    def post_to_empty_hook(self, cosmos_config):
        """Post-processing hook after moving to empty device

        Ensures all VLA parameters are trainable (unlike MoE models that freeze gate weights).
        VLA models train vision_backbone, projector, language_model, and action_head.

        For rollout workers: weights will be synced via P2R
        """
        pass

    def _set_fsdp_reshard_after_forward(self, policy_str: str = "default"):
        reshard_after_forward = False if policy_str == "never" else True
        for m in self.model.modules():
            if isinstance(m, torch.distributed.fsdp.FSDPModule):
                m.set_reshard_after_forward(reshard_after_forward)

    def _replace_rope_modules_float32(self):
        """Replace RoPE modules with fresh float32 versions.

        After model.to(dtype=bfloat16), RoPE buffers are incorrectly converted to bfloat16.
        This method simply replaces the entire RoPE module with a fresh one that has
        correct float32 buffers. Much simpler than selective dtype conversion!
        """
        try:
            from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

            if not (
                hasattr(self.model, "language_model")
                and hasattr(self.model.language_model, "model")
            ):
                logger.debug("No language_model found, skipping RoPE replacement")
                return

            device = next(self.model.parameters()).device
            llm_config = self.model.language_model.model.config

            # Create fresh RoPE module with float32 buffers
            new_rope = LlamaRotaryEmbedding(config=llm_config, device=device)

            replaced_count = 0

            # Replace the top-level rotary_emb
            if hasattr(self.model.language_model.model, "rotary_emb"):
                old_dtype = self.model.language_model.model.rotary_emb.inv_freq.dtype
                self.model.language_model.model.rotary_emb = new_rope
                logger.debug(
                    f"Replaced top-level rotary_emb: {old_dtype} → {new_rope.inv_freq.dtype}"
                )
                replaced_count += 1

            # Also check for RoPE in each layer (some models have per-layer RoPE)
            if hasattr(self.model.language_model.model, "layers"):
                for i, layer in enumerate(self.model.language_model.model.layers):
                    if hasattr(layer, "self_attn") and hasattr(
                        layer.self_attn, "rotary_emb"
                    ):
                        # Share the same RoPE instance across all layers
                        layer.self_attn.rotary_emb = new_rope
                        replaced_count += 1

        except Exception as e:
            logger.error(f"Failed to replace RoPE modules: {e}")
            import traceback

            traceback.print_exc()

    def get_position_ids(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Get position IDs for input"""
        position_ids = None
        input_ids = kwargs.get("input_ids")
        seq_dim_idx = 1
        return position_ids, input_ids, seq_dim_idx

    def load_hf_weights(
        self,
        model_name_or_path: str,
        parallel_dims: Optional[ParallelDims],
        device: torch.device,
        revision: Optional[str] = None,
    ):
        """
        Load weights from a HuggingFace model (following cosmos-rl pattern)

        Args:
            model_name_or_path: Path to the HuggingFace model
            parallel_dims: Parallel dimensions definition (optional, not used for loading)
            device: Target device
            revision: Model revision/branch
        """
        logger.info(f"Loading VLA weights from {model_name_or_path}")

        # Load the reference model to get weights
        if self.hf_config.vla_type == "openvla-oft":
            from cosmos_rl.policy.model.vla.openvla_oft.modeling_prismatic import (
                OpenVLAForActionPrediction,
            )
            from cosmos_rl.policy.model.vla.openvla_oft.processing_prismatic import (
                PrismaticProcessor,
            )
        else:
            from cosmos_rl.policy.model.vla.openvla.modeling_prismatic import (
                OpenVLAForActionPrediction,
            )
            from cosmos_rl.policy.model.vla.openvla.processing_prismatic import (
                PrismaticProcessor,
            )

        # Load full model with weights to CPU
        kwargs = {
            "torch_dtype": self.hf_config.torch_dtype,
            "device_map": "cpu",  # Load to CPU first
        }
        if revision:
            kwargs["revision"] = revision

        try:
            # Load state dict from checkpoint (better for TIMM models)
            # This ensures TIMM vision backbone weights are properly loaded
            from safetensors import safe_open
            from pathlib import Path

            # Find safetensors or pytorch_model.bin files
            model_path = Path(model_name_or_path)
            if not model_path.exists():
                # Download from HF if needed
                from huggingface_hub import snapshot_download

                model_path = Path(
                    snapshot_download(repo_id=model_name_or_path, revision=revision)
                )

            # Try safetensors first (preferred for VLA models)
            safetensor_files = list(model_path.glob("*.safetensors"))
            if safetensor_files:
                state_dict = {}
                for st_file in safetensor_files:
                    with safe_open(st_file, framework="pt", device=str(device)) as f:
                        for key in f.keys():
                            state_dict[key] = f.get_tensor(key)
            else:
                # Fallback to pytorch_model.bin
                pt_files = list(model_path.glob("pytorch_model*.bin"))
                if pt_files:
                    state_dict = {}
                    for pt_file in pt_files:
                        state_dict.update(torch.load(pt_file, map_location=device))
                else:
                    raise FileNotFoundError(
                        f"No safetensors or pytorch_model.bin found in {model_path}"
                    )

            # Load state dict into model using FSDP-compatible method
            # Following HFModel's pattern: use weight converter to shard tensors
            from cosmos_rl.policy.model.vla.weight_converter import (
                convert_weight_from_hf,
            )

            with torch.no_grad():
                model_state_dict = self.model.state_dict()
                missing_keys = []
                loaded_keys = []

                for name, checkpoint_tensor in state_dict.items():
                    if name in model_state_dict:
                        target_param = model_state_dict[name]

                        # Check if parameter is a DTensor (FSDP-wrapped)
                        is_dist_tensor = isinstance(
                            target_param, torch.distributed.tensor.DTensor
                        )

                        # Get local view of the parameter
                        local_view = (
                            target_param.to_local() if is_dist_tensor else target_param
                        )

                        # All parameters are FSDP-sharded uniformly, so always use weight converter
                        _, checkpoint_shard = convert_weight_from_hf(
                            checkpoint_tensor, name, parallel_dims
                        )

                        # Copy sharded checkpoint to local view
                        try:
                            local_view.data.copy_(checkpoint_shard.to(device))
                            loaded_keys.append(name)
                        except Exception as copy_error:
                            logger.warning(
                                f"Failed to copy {name}: {copy_error} (local shape={local_view.shape}, shard shape={checkpoint_shard.shape})"
                            )
                            missing_keys.append(name)
                    else:
                        missing_keys.append(name)

                unexpected_keys = [
                    k for k in state_dict.keys() if k not in model_state_dict
                ]

            # Check requires_grad status after loading
            frozen_count = sum(
                1 for _, p in self.model.named_parameters() if not p.requires_grad
            )
            trainable_count = sum(
                1 for _, p in self.model.named_parameters() if p.requires_grad
            )
            logger.info(
                f"Frozen parameters count: {frozen_count}, trainable parameters count: {trainable_count}"
            )
            if frozen_count > 0:
                logger.warning(
                    f"⚠️  Found {frozen_count} frozen parameters after weight loading - these will be unfrozen in post_to_empty_hook"
                )
                # Sample some frozen vision backbone params
                frozen_vb_params = [
                    name
                    for name, p in self.model.named_parameters()
                    if not p.requires_grad and "vision_backbone" in name
                ]
                if frozen_vb_params:
                    logger.warning(
                        f"   Sample frozen vision_backbone params: {frozen_vb_params[:5]}"
                    )

            if missing_keys:
                logger.warning(f"⚠️  {len(missing_keys)} missing keys in checkpoint")
                logger.warning(f"   First 10: {missing_keys[:10]}")
            if unexpected_keys:
                logger.warning(
                    f"⚠️  {len(unexpected_keys)} unexpected keys in checkpoint"
                )
                logger.warning(f"   First 10: {unexpected_keys[:10]}")

        except Exception as e:
            logger.error(
                f"Failed to load VLA state dict, falling back to from_pretrained: {e}"
            )
            import traceback

            traceback.print_exc()

            # Fallback to from_pretrained method
            hf_model = OpenVLAForActionPrediction.from_pretrained(
                model_name_or_path, config=self.hf_config, **kwargs
            )

            # Copy weights using load_state_dict (proper way for TIMM models)
            hf_state_dict = hf_model.state_dict()
            missing_keys, unexpected_keys = self.model.load_state_dict(
                hf_state_dict, strict=False
            )

            del hf_model
            torch.cuda.empty_cache()

            if missing_keys:
                logger.warning(f"⚠️  {len(missing_keys)} missing keys")
                logger.warning(f"   First 10: {missing_keys[:10]}")
            if unexpected_keys:
                logger.warning(f"⚠️  {len(unexpected_keys)} unexpected keys")
                logger.warning(f"   First 10: {unexpected_keys[:10]}")

        # Setup VLA-specific features (only needed for from_pretrained path)
        # For direct state_dict loading, these features should already be in the checkpoint
        # self._setup_vla_specific_features(model_name_or_path, hf_model)

        # Load processor
        try:
            self.processor = PrismaticProcessor.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
            )
        except Exception as e:
            logger.warning(f"Failed to load VLA processor: {e}")

        # Load normalization stats
        self._load_vla_norm_stats(model_name_or_path)
        self._replace_rope_modules_float32()

    def _setup_vla_specific_features(self, model_name_or_path: str, hf_model):
        """Setup VLA-specific features after weight loading"""
        try:
            # OFT-specific setup
            if self.hf_config.use_proprio:
                if hasattr(hf_model, "load_proprio_projector_weights"):
                    hf_model.load_proprio_projector_weights(model_name_or_path)
                    logger.info("Loaded pre-trained proprio projector weights")

                if hasattr(hf_model, "vision_backbone") and hasattr(
                    hf_model.vision_backbone, "set_num_images_in_input"
                ):
                    num_images = self.hf_config.num_images_in_input
                    hf_model.vision_backbone.set_num_images_in_input(num_images)
                    logger.info(f"Set num_images_in_input to {num_images}")

            # Copy norm stats if available
            if hasattr(hf_model, "norm_stats"):
                self.norm_stats = hf_model.norm_stats

        except Exception as e:
            logger.warning(f"VLA-specific setup failed: {e}")

    def _load_vla_norm_stats(self, model_name_or_path: str):
        """
        Load VLA normalization statistics as fallback if not already in config.

        Note: Typically norm_stats should come from config (loaded in create_vla_config).
        This is a fallback for cases where config doesn't have it.
        """
        # Skip if norm_stats already loaded from config
        if self.norm_stats:
            logger.debug("norm_stats already loaded from config, skipping file loading")
            return

        try:
            import json

            # Try dataset_statistics.json first
            dataset_stats_path = os.path.join(
                model_name_or_path, "dataset_statistics.json"
            )
            if os.path.isfile(dataset_stats_path):
                with open(dataset_stats_path, "r") as f:
                    self.norm_stats = json.load(f)
                return

            # Try norm_stats.pt as fallback
            norm_stats_path = os.path.join(model_name_or_path, "norm_stats.pt")
            if os.path.exists(norm_stats_path):
                self.norm_stats = torch.load(norm_stats_path, map_location="cpu")
                return

            logger.warning(
                "⚠️  No norm_stats found in config or checkpoint files. "
                "This may cause issues with action normalization. "
                "Ignore if loading a base (not fine-tuned) VLA checkpoint."
            )

        except Exception as e:
            logger.warning(f"Could not load VLA norm_stats from files: {e}")

    def separate_model_parts(self) -> List[torch.nn.Module]:
        """Separate model into parts for parallelization"""
        parts = []
        parts.append(self.model)
        return parts

    def get_nparams_and_flops(self, seq_len: int) -> tuple[int, int]:
        """Calculate number of parameters and FLOPs for VLA model"""
        nparams = sum(p.numel() for p in self.parameters())

        # Vision component FLOPs
        vision_flops = 0
        if (
            hasattr(self.model, "vision_backbone")
            and self.model.vision_backbone is not None
        ):
            vision_params = sum(
                p.numel() for p in self.model.vision_backbone.parameters()
            )
            vision_flops = vision_params * 2  # Approximate

        # Language model FLOPs
        language_flops = 0
        if (
            hasattr(self.model, "language_model")
            and self.model.language_model is not None
        ):
            lm_params = sum(p.numel() for p in self.model.language_model.parameters())

            # Subtract embedding params from computation
            lm_embedding_params = 0
            if hasattr(self.model.language_model, "embed_tokens"):
                lm_embedding_params = sum(
                    p.numel()
                    for p in self.model.language_model.embed_tokens.parameters()
                )
            elif hasattr(self.model.language_model, "model") and hasattr(
                self.model.language_model.model, "embed_tokens"
            ):
                lm_embedding_params = sum(
                    p.numel()
                    for p in self.model.language_model.model.embed_tokens.parameters()
                )

            # Approximate FLOPs: 6 * non_embedding_params for forward pass
            language_flops = 6 * (lm_params - lm_embedding_params)

            # Add attention FLOPs
            try:
                if hasattr(self.hf_config, "text_config"):
                    text_config = self.hf_config.text_config
                elif hasattr(self.hf_config, "language_config"):
                    text_config = self.hf_config.language_config
                else:
                    text_config = self.hf_config

                if hasattr(text_config, "num_attention_heads") and hasattr(
                    text_config, "hidden_size"
                ):
                    layers = getattr(text_config, "num_hidden_layers", 32)
                    heads = text_config.num_attention_heads
                    head_dim = text_config.hidden_size // heads

                    # Attention FLOPs approximation
                    language_flops += 12 * layers * heads * head_dim * seq_len
            except Exception:
                # Fallback approximation
                language_flops += lm_params * 2 * seq_len / 1000

        # Action head FLOPs
        action_flops = 0
        if hasattr(self.model, "action_head") and self.model.action_head is not None:
            action_params = sum(p.numel() for p in self.model.action_head.parameters())
            action_flops = action_params * 2

        total_flops = vision_flops + language_flops + action_flops

        logger.debug(
            f"VLA model stats: {nparams} params, {total_flops} FLOPs (seq_len={seq_len})"
        )
        return nparams, int(total_flops)

    def process_input(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Process inputs for VLA model (matching SimpleVLA-RL's process_input)

        Args:
            inputs: List of observation dictionaries
            task_descriptions: List of task description strings

        Returns:
            Processed batch data for VLA model
        """
        full_images = inputs["full_images"]
        wrist_images = inputs["wrist_images"]
        task_descriptions = inputs["task_descriptions"]

        vla_type = self.hf_config.vla_type

        batchdata = {"input_ids": [], "attention_mask": [], "pixel_values": []}

        batch_size = full_images.shape[0]
        for i in range(batch_size):
            full_image = obs_to_vla_input(full_images[i])
            full_image = Image.fromarray(full_image).convert("RGB")
            full_image = center_crop_image(full_image)
            desp = task_descriptions[i]

            prompt = f"In: What action should the robot take to {desp.lower()}?\nOut:"
            batch_feature = self.processor(prompt, full_image)
            input_ids = batch_feature["input_ids"]
            attention_mask = batch_feature.get(
                "attention_mask", torch.ones_like(input_ids)
            )
            pixel_values = batch_feature["pixel_values"]

            if (
                hasattr(self.hf_config, "use_wrist_camera")
                and self.hf_config.use_wrist_camera
            ):
                wrist_image = obs_to_vla_input(wrist_images[i])
                wrist_image = Image.fromarray(wrist_images[i]).convert("RGB")
                wrist_image = center_crop_image(wrist_image)
                wrist_feature = self.processor(prompt, wrist_image)
                pixel_values = torch.cat(
                    [pixel_values, wrist_feature["pixel_values"]], dim=1
                )

            # Handle OpenVLA-OFT specific formatting
            if vla_type == "openvla-oft":
                # Add space token if needed (matching SimpleVLA-RL)
                space_token_id = 29871  # Space token for LLaMA-based models
                if not torch.all(input_ids[:, -1] == space_token_id):
                    input_ids = torch.cat(
                        (
                            input_ids,
                            torch.tensor(
                                [[space_token_id]],
                                dtype=input_ids.dtype,
                                device=input_ids.device,
                            ),
                        ),
                        dim=1,
                    )
                    attention_mask = torch.cat(
                        (
                            attention_mask,
                            torch.tensor(
                                [[True]],
                                dtype=attention_mask.dtype,
                                device=attention_mask.device,
                            ),
                        ),
                        dim=1,
                    )

            batchdata["input_ids"].append(input_ids)
            batchdata["attention_mask"].append(attention_mask)
            batchdata["pixel_values"].append(pixel_values)

        # Device placement
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if vla_type == "openvla-oft":
            # OpenVLA-OFT specific batch processing
            batchdata["input_ids"] = [x.transpose(0, 1) for x in batchdata["input_ids"]]
            batchdata["attention_mask"] = [
                x.transpose(0, 1) for x in batchdata["attention_mask"]
            ]

            batchdata["input_ids"] = (
                pad_sequence(
                    batchdata["input_ids"],
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id,
                )
                .squeeze(-1)
                .to(device)
            )
            batchdata["attention_mask"] = (
                pad_sequence(
                    batchdata["attention_mask"], batch_first=True, padding_value=0
                )
                .squeeze(-1)
                .to(device)
            )

            # Handle padding and sorting (matching SimpleVLA-RL)
            padding_mask = batchdata["input_ids"].ne(self.tokenizer.pad_token_id)
            padding_mask = ~padding_mask
            padding_mask = padding_mask.int()
            sorted_indices = torch.argsort(
                padding_mask, dim=1, descending=True, stable=True
            )
            batchdata["input_ids"] = torch.gather(
                batchdata["input_ids"], 1, sorted_indices
            )
            batchdata["attention_mask"] = torch.gather(
                batchdata["attention_mask"], 1, sorted_indices
            )

            batchdata["pixel_values"] = torch.cat(batchdata["pixel_values"], dim=0).to(
                device
            )
        else:
            # Standard batch processing
            for key in ["input_ids", "attention_mask", "pixel_values"]:
                batchdata[key] = torch.cat(batchdata[key], dim=0).to(device)
        return batchdata

    def generate_action(
        self,
        inputs: Dict[str, torch.Tensor],
        is_valid: bool = False,
        temperature: float = 0.0,
        unnorm_key: str = "libero_10_no_noops",
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate one step for OpenVLA-OFT (matching SimpleVLA-RL)"""
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]
        proprio = inputs.get("proprio", None)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            actions, responses, logprobs = self.model.generate_action(
                input_ids=input_ids,
                pixel_values=pixel_values,
                proprio=proprio,
                attention_mask=attention_mask,
                padding_idx=self.tokenizer.pad_token_id,
                do_sample=not is_valid,
                unnorm_key=unnorm_key,
                temperature=temperature,
            )

            actions = normalize_gripper_action(actions)
            actions = invert_gripper_action(actions)

            return {
                "action": actions,
                "responses": responses,
                "old_log_probs": logprobs,
            }
