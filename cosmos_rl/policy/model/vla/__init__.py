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
import torch.nn as nn
import os
import inspect
import warnings
from functools import cached_property
from typing import Tuple, List, Optional, Any, Dict
from dataclasses import dataclass, field
from transformers import AutoConfig, AutoProcessor
from safetensors import safe_open

from cosmos_rl.policy.model.base import BaseModel, ModelRegistry
from cosmos_rl.policy.model.vla.weight_mapper import VLAWeightMapper
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils import util
from cosmos_rl.utils.parallelism import ParallelDims


@dataclass
class VLAArgs:
    """VLA Model Arguments following cosmos-rl pattern"""
    vla_type: str = "openvla-oft"  # "openvla" or "openvla-oft"
    use_proprio: bool = False
    proprio_dim: int = 0
    num_images_in_input: int = 1
    hf_config: AutoConfig = None


@ModelRegistry.register(VLAWeightMapper)
class VLAModel(BaseModel):
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
    
    def __init__(self, vla_args: VLAArgs):
        """Initialize VLA model following cosmos-rl pattern"""
        super().__init__(vla_args.hf_config)
        self.config = vla_args
        self.hf_config = vla_args.hf_config
        
        # Create the actual VLA model instance directly (no AutoModel)
        if vla_args.vla_type == "openvla-oft":
            from cosmos_rl.policy.model.vla.openvla_oft.modeling_prismatic import OpenVLAForActionPrediction
            logger.info("Using OpenVLA-OFT direct implementation")
        else:  # openvla (default)
            from cosmos_rl.policy.model.vla.openvla.modeling_prismatic import OpenVLAForActionPrediction  
            logger.info("Using OpenVLA direct implementation")
        
        # Create model directly (structure only, no weights)
        logger.info(f"Creating VLA model structure with config: {self.hf_config.model_type}")
        self.model = OpenVLAForActionPrediction(self.hf_config)
        
        # Convert to proper dtype
        if hasattr(self.hf_config, 'torch_dtype'):
            self.model = self.model.to(dtype=self.hf_config.torch_dtype)
        
        # Initialize additional attributes  
        self.processor = None
        self.is_vlm = True
        
        # Initialize norm_stats from config if available
        if hasattr(self.hf_config, 'norm_stats') and self.hf_config.norm_stats is not None:
            self.norm_stats = self.hf_config.norm_stats
            logger.info(f"Initialized norm_stats from config with keys: {list(self.norm_stats.keys())}")
        else:
            self.norm_stats = {}
            logger.warning("No norm_stats in config, will try to load from checkpoint later")

        logger.info(f"✅ Initialized VLA model structure: {vla_args.vla_type}")

    @classmethod
    def from_model_args(cls, vla_args: VLAArgs) -> "VLAModel":
        """Initialize VLA model from VLAArgs object (following Qwen VL pattern)"""
        return cls(vla_args)

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
            k: v for k, v in kwargs.items() 
            if k in self.model_forward_valid_kwargs
        }
        
        # Prepare model inputs
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            **filtered_kwargs
        }
        
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
            
        if labels is not None:
            model_inputs["labels"] = labels
            
        # Remove None values
        model_inputs = {k: v for k, v in model_inputs.items() if v is not None}
        
        try:
            return self.model(**model_inputs)
        except Exception as e:
            logger.error(f"VLA model forward pass failed: {e}")
            logger.error(f"Input shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in model_inputs.items()]}")
            raise
    
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 512,
        do_sample: bool = True,
        temperature: float = 1.0,
        num_beams: int = 1,
        num_return_sequences: int = 1,
        **kwargs,
    ):
        """Generate sequences using VLA model"""
        model_inputs = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
            "max_length": max_length,
            "do_sample": do_sample,
            "temperature": temperature,
            "num_beams": num_beams,
            "num_return_sequences": num_return_sequences,
            **kwargs
        }
        
        # Remove None values
        model_inputs = {k: v for k, v in model_inputs.items() if v is not None}
        
        return self.model.generate(**model_inputs)

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save VLA model to directory"""
        self.model.save_pretrained(save_directory, **kwargs)
        if self.processor is not None:
            self.processor.save_pretrained(save_directory)
        logger.info(f"Saved VLA model to {save_directory}")
        
    @classmethod
    def from_pretrained(
        cls,
        hf_config: AutoConfig,
        model_name_or_path: str,
        max_position_embeddings: Optional[int] = None,
    ) -> "VLAModel":
        """
        Initialize a VLA model from a pretrained model (following cosmos-rl pattern)
        
        Args:
            hf_config: HuggingFace configuration
            model_name_or_path: Model name or path to the pretrained model
            max_position_embeddings: Override max position embeddings
            
        Returns:
            VLAModel: VLA model instance
        """
        logger.info(f"Creating VLA model from pretrained: {model_name_or_path}")
        
        # Infer VLA type if not present in config
        if not hasattr(hf_config, 'vla_type'):
            hf_config.vla_type = "openvla-oft"  # Default to OFT
            logger.info(f"Inferred VLA type: {hf_config.vla_type}")
        
        # Set max position embeddings if provided
        if max_position_embeddings is not None:
            hf_config.max_position_embeddings = max_position_embeddings
        
        # Create VLA args from HF config
        vla_args = VLAArgs(
            vla_type=hf_config.vla_type,
            use_proprio=getattr(hf_config, 'use_proprio', False),
            proprio_dim=getattr(hf_config, 'proprio_dim', 0),
            num_images_in_input=getattr(hf_config, 'num_images_in_input', 1),
            hf_config=hf_config
        )
        
        # Create model structure only (no weights loaded)
        return cls.from_model_args(vla_args)
    
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
        """Post-processing hook after moving to empty device"""
        pass
    
    def get_position_ids(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Get position IDs for input"""
        position_ids = None
        input_ids = kwargs.get("input_ids")
        seq_dim_idx = 1
        return position_ids, input_ids, seq_dim_idx
    
    def load_from_checkpoint(
        self,
        model_name_or_path: str,
        parallel_dims: Optional[ParallelDims] = None,
        device: Optional[torch.device] = None,
        revision: Optional[str] = None,
    ):
        """
        Load VLA model weights from checkpoint (simplified for single-GPU rollout)
        
        For rollout workers using 1 card per replica (data parallelism), we use
        HuggingFace's from_pretrained directly like RLinf does. This is simpler
        and faster than complex distributed loading.
        
        Args:
            model_name_or_path: Path to the HuggingFace model or local checkpoint
            parallel_dims: Ignored for single-GPU loading (kept for interface compatibility)
            device: Target device (defaults to current CUDA device)
            revision: Model revision/branch
        """
        if device is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        
        logger.info(f"Loading VLA model from checkpoint: {model_name_or_path}")
        logger.info(f"  Target device: {device}")
        logger.info(f"  Using single-GPU loading (like RLinf)")
        
        # Import the appropriate VLA model class
        if self.config.vla_type == "openvla-oft":
            from cosmos_rl.policy.model.vla.openvla_oft.modeling_prismatic import OpenVLAForActionPrediction
            from cosmos_rl.policy.model.vla.openvla_oft.processing_prismatic import PrismaticProcessor
        else:
            from cosmos_rl.policy.model.vla.openvla.modeling_prismatic import OpenVLAForActionPrediction
            from cosmos_rl.policy.model.vla.openvla.processing_prismatic import PrismaticProcessor
        
        kwargs = {
            "config": self.hf_config,  # Use our config with VLA-specific attributes
            "torch_dtype": self.hf_config.torch_dtype if hasattr(self.hf_config, 'torch_dtype') else torch.bfloat16,
            "device_map": str(device),  # Load directly to target device
            "trust_remote_code": True,
        }
        if revision:
            kwargs["revision"] = revision
        
        try:
            # Load full model with from_pretrained (simple approach)
            hf_model = OpenVLAForActionPrediction.from_pretrained(
                model_name_or_path,
                **kwargs
            )
            logger.info(f"✅ Loaded VLA model from HuggingFace: {type(hf_model)}")
            
            # Copy state dict to our model
            with torch.no_grad():
                self.model.load_state_dict(hf_model.state_dict())
            
            # Copy norm_stats if available
            if hasattr(hf_model, 'norm_stats') and hf_model.norm_stats is not None:
                self.norm_stats = hf_model.norm_stats
                self.model.norm_stats = hf_model.norm_stats
                logger.info(f"  Copied norm_stats with keys: {list(self.norm_stats.keys())}")
            
            # Move our model to device
            self.model = self.model.to(device)
            
            # Clean up HF model
            del hf_model
            torch.cuda.empty_cache()
            
            logger.info("✅ VLA checkpoint loading completed (single-GPU)")
            
        except Exception as e:
            logger.error(f"Failed to load VLA checkpoint from {model_name_or_path}: {e}")
            raise
        
        # Load processor
        try:
            self.processor = PrismaticProcessor.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
            )
            logger.info("✅ Loaded VLA processor")
        except Exception as e:
            logger.warning(f"Failed to load VLA processor: {e}")
    
    def load_hf_weights(
        self,
        model_name_or_path: str,
        parallel_dims: ParallelDims,
        device: torch.device,
        revision: Optional[str] = None,
    ):
        """
        Load weights from a HuggingFace model (following cosmos-rl pattern)
        
        This method is for distributed training with model parallelism.
        For single-GPU rollout, use load_from_checkpoint() instead.
        
        Args:
            model_name_or_path: Path to the HuggingFace model
            parallel_dims: Parallel dimensions definition  
            device: Target device
            revision: Model revision/branch
        """
        logger.info(f"Loading VLA weights from {model_name_or_path}")
        
        # Load the reference model to get weights
        if self.config.vla_type == "openvla-oft":
            from cosmos_rl.policy.model.vla.openvla_oft.modeling_prismatic import OpenVLAForActionPrediction
            from cosmos_rl.policy.model.vla.openvla_oft.processing_prismatic import PrismaticProcessor
        else:
            from cosmos_rl.policy.model.vla.openvla.modeling_prismatic import OpenVLAForActionPrediction
            from cosmos_rl.policy.model.vla.openvla.processing_prismatic import PrismaticProcessor
            
        # Load full model with weights to CPU
        kwargs = {
            "torch_dtype": self.hf_config.torch_dtype,
            "device_map": "cpu",  # Load to CPU first
        }
        if revision:
            kwargs["revision"] = revision
        
        try:
            # Try loading from pretrained using our config (with VLA attributes)
            hf_model = OpenVLAForActionPrediction.from_pretrained(
                model_name_or_path, 
                config=self.hf_config,  # ✅ Use our existing config with VLA attributes
                **kwargs
            )
            logger.info(f"✅ Loaded reference model from HF: {type(hf_model)}")
        except Exception as e:
            logger.error(f"Failed to load VLA model from {model_name_or_path}: {e}")
            raise
        
        # Copy weights from reference model to our model
        hf_state_dict = hf_model.state_dict()
        self_state_dict = self.model.state_dict()
        
        # Copy matching weights
        for name, tensor in hf_state_dict.items():
            if name in self_state_dict:
                target_tensor = self_state_dict[name]
                
                # Handle distributed tensors if present
                is_dist_tensor = hasattr(target_tensor, 'to_local')
                local_view = target_tensor.to_local() if is_dist_tensor else target_tensor
                
                # Copy data
                with torch.no_grad():
                    local_view.data.copy_(tensor.to(device))
            else:
                logger.warning(f"Weight {name} not found in VLA model")
        
        # Setup VLA-specific features
        self._setup_vla_specific_features(model_name_or_path, hf_model)
        
        # Load processor
        try:
            self.processor = PrismaticProcessor.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
            )
            logger.info("✅ Loaded VLA processor")
        except Exception as e:
            logger.warning(f"Failed to load VLA processor: {e}")
            
        # Load normalization stats
        self._load_vla_norm_stats(model_name_or_path)
        
        # Clean up reference model
        del hf_model
        logger.info("✅ VLA weight loading completed")
        
    def _setup_vla_specific_features(self, model_name_or_path: str, hf_model):
        """Setup VLA-specific features after weight loading"""
        try:
            # OFT-specific setup
            if self.hf_config.use_proprio:
                if hasattr(hf_model, 'load_proprio_projector_weights'):
                    hf_model.load_proprio_projector_weights(model_name_or_path)
                    logger.info("Loaded pre-trained proprio projector weights")
                
                if hasattr(hf_model, 'vision_backbone') and hasattr(hf_model.vision_backbone, 'set_num_images_in_input'):
                    num_images = self.config.num_images_in_input
                    hf_model.vision_backbone.set_num_images_in_input(num_images)
                    logger.info(f"Set num_images_in_input to {num_images}")
            
            # Copy norm stats if available
            if hasattr(hf_model, 'norm_stats'):
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
            dataset_stats_path = os.path.join(model_name_or_path, "dataset_statistics.json")
            if os.path.isfile(dataset_stats_path):
                with open(dataset_stats_path, "r") as f:
                    self.norm_stats = json.load(f)
                logger.info("✅ Loaded VLA normalization stats from dataset_statistics.json (fallback)")
                return
            
            # Try norm_stats.pt as fallback
            norm_stats_path = os.path.join(model_name_or_path, "norm_stats.pt")
            if os.path.exists(norm_stats_path):
                self.norm_stats = torch.load(norm_stats_path, map_location="cpu")
                logger.info("✅ Loaded VLA normalization stats from norm_stats.pt (fallback)")
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
        if hasattr(self.model, 'vision_backbone') and self.model.vision_backbone is not None:
            vision_params = sum(p.numel() for p in self.model.vision_backbone.parameters())
            vision_flops = vision_params * 2  # Approximate
        
        # Language model FLOPs
        language_flops = 0
        if hasattr(self.model, 'language_model') and self.model.language_model is not None:
            lm_params = sum(p.numel() for p in self.model.language_model.parameters())
            
            # Subtract embedding params from computation
            lm_embedding_params = 0
            if hasattr(self.model.language_model, 'embed_tokens'):
                lm_embedding_params = sum(p.numel() for p in self.model.language_model.embed_tokens.parameters())
            elif hasattr(self.model.language_model, 'model') and hasattr(self.model.language_model.model, 'embed_tokens'):
                lm_embedding_params = sum(p.numel() for p in self.model.language_model.model.embed_tokens.parameters())
            
            # Approximate FLOPs: 6 * non_embedding_params for forward pass
            language_flops = 6 * (lm_params - lm_embedding_params)
            
            # Add attention FLOPs
            try:
                if hasattr(self.hf_config, 'text_config'):
                    text_config = self.hf_config.text_config
                elif hasattr(self.hf_config, 'language_config'):
                    text_config = self.hf_config.language_config
                else:
                    text_config = self.hf_config
                    
                if hasattr(text_config, 'num_attention_heads') and hasattr(text_config, 'hidden_size'):
                    layers = getattr(text_config, 'num_hidden_layers', 32)
                    heads = text_config.num_attention_heads
                    head_dim = text_config.hidden_size // heads
                    
                    # Attention FLOPs approximation
                    language_flops += 12 * layers * heads * head_dim * seq_len
            except Exception:
                # Fallback approximation
                language_flops += lm_params * 2 * seq_len / 1000
        
        # Action head FLOPs
        action_flops = 0
        if hasattr(self.model, 'action_head') and self.model.action_head is not None:
            action_params = sum(p.numel() for p in self.model.action_head.parameters())
            action_flops = action_params * 2
        
        total_flops = vision_flops + language_flops + action_flops
        
        logger.debug(f"VLA model stats: {nparams} params, {total_flops} FLOPs (seq_len={seq_len})")
        return nparams, int(total_flops)


