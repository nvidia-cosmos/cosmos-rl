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
    
    def __init__(self, vla_args: VLAArgs, init_device: str = "cuda"):
        """
        Initialize VLA model following cosmos-rl pattern
        
        Args:
            vla_args: VLA configuration arguments
            init_device: Device to initialize model on ("cuda", "cpu", or "meta")
                        Note: VLA models use TIMM vision backbone which has issues with
                        meta tensors. For rollout workers, use "cuda" (policy worker does the same).
        """
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
        
        # Create model with specified device (use "meta" for fast initialization)
        logger.info(f"Creating VLA model structure on device: {init_device}")
        with torch.device(init_device):
            self.model = OpenVLAForActionPrediction(self.hf_config)
        
        # For meta device, allocate memory without initializing weights
        if init_device == "meta":
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
            logger.info(f"Allocating VLA model on {device} (weights uninitialized, for NCCL sync)")
            self.model.to_empty(device=device)
            # Set dtype after allocation
            if hasattr(self.hf_config, 'torch_dtype'):
                self.model = self.model.to(dtype=self.hf_config.torch_dtype)
        else:
            # Normal initialization with random weights
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
    def from_model_args(cls, vla_args: VLAArgs, init_device: str = "cuda") -> "VLAModel":
        """
        Initialize VLA model from VLAArgs object (following Qwen VL pattern)
        
        Args:
            vla_args: VLA configuration arguments
            init_device: Device to initialize model on ("cuda", "cpu", or "meta")
        """
        return cls(vla_args, init_device=init_device)

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
            
        # Remove None values
        model_inputs = {k: v for k, v in model_inputs.items() if v is not None}
        
        try:
            outputs = self.model(**model_inputs)
            
            # OpenVLA-OFT returns raw logits tensor, not output object
            # Wrap it in a simple namespace to provide .logits attribute
            if isinstance(outputs, torch.Tensor):
                from types import SimpleNamespace
                outputs = SimpleNamespace(logits=outputs)
            
            return outputs
        except Exception as e:
            logger.error(f"VLA model forward pass failed: {e}")
            logger.error(f"Input shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in model_inputs.items()]}")
            raise
    
    def forward_with_trajectory_structure(
        self,
        input_ids: torch.Tensor,  # (batch, num_steps, seq_len)
        pixel_values: torch.Tensor,  # (batch, num_steps, C, H, W)
        attention_mask: torch.Tensor,  # (batch, num_steps, seq_len)
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass for VLA training with per-step trajectory structure.
        
        Data packer already splits long trajectories into manageable sub-trajectories,
        so we just need to:
        1. Unpad sequences (remove padding tokens to save computation/memory)
        2. Flatten (batch, steps, ...) → (batch*steps, ...)
        3. Forward pass
        4. Reshape back to (batch, steps, output_len, vocab)
        
        Args:
            input_ids: (batch, num_steps, seq_len) - per-step input tokens
            pixel_values: (batch, num_steps, C, H, W) - per-step images
            attention_mask: (batch, num_steps, seq_len) - per-step masks
            
        Returns:
            Output with logits reshaped to (batch, num_steps, output_len, vocab_size)
            Note: output_len may differ from seq_len (openvla-oft returns only action tokens ~56)
        """
        batch_size, max_steps, seq_len = input_ids.shape
        _, _, C, H, W = pixel_values.shape
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # Flatten: (batch, steps, ...) → (batch*steps, ...)
            input_ids_flat = input_ids.reshape(batch_size * max_steps, seq_len)
            attention_mask_flat = attention_mask.reshape(batch_size * max_steps, seq_len)
            pixel_values_flat = pixel_values.reshape(batch_size * max_steps, C, H, W)
            
            logprob_masks_unpad = None
            if 'logprob_masks' in kwargs and kwargs['logprob_masks'] is not None:
                logprob_masks = kwargs['logprob_masks']
                if logprob_masks.ndim == 3:  # (batch, steps, action_seq_len)
                    # logprob_masks are already the right shape for model output (action tokens only)
                    # Just flatten for processing
                    action_seq_len = logprob_masks.shape[2]
                    logprob_masks_flat = logprob_masks.reshape(batch_size * max_steps, action_seq_len)
                    
                    # No unpadding needed - these match the model's output logits directly
                    logprob_masks_unpad = logprob_masks_flat.reshape(batch_size, max_steps, action_seq_len)
                    
                    # Log action token info
                    num_trainable = logprob_masks_unpad.sum().item()
                    logger.debug(
                        f"[VLA logprob_masks] Shape: {logprob_masks_unpad.shape}, "
                        f"trainable_tokens={num_trainable} (matches model output action positions)"
                    )
            
            outputs = self.forward(
                input_ids=input_ids_flat,
                pixel_values=pixel_values_flat,
                attention_mask=attention_mask_flat,
                position_ids=position_ids,
                labels=labels,
                **kwargs
            )
        
            # Reshape back: (batch*steps, output_len, vocab) → (batch, steps, output_len, vocab)
            # Note: output_len may differ from seq_len (openvla-oft returns only action tokens)
            flat_logits = outputs.logits  # (batch*steps, output_len, vocab)
            output_len = flat_logits.shape[1]
            vocab_size = flat_logits.shape[2]
        
            logits_reshaped = flat_logits.reshape(batch_size, max_steps, output_len, vocab_size)
        
            # Set the reshaped logits on the output object
            outputs.logits = logits_reshaped
            
            # Attach logprob_masks for trainer to use
            # For openvla-oft: logprob_masks (56 values) match output logits (56 tokens)
            if logprob_masks_unpad is not None:
                outputs.logprob_masks_unpad = logprob_masks_unpad
                logger.debug(f"[VLA Forward] Attached logprob_masks: {logprob_masks_unpad.shape} (matches output logits)")
            
            # Attach responses if available (needed for loss computation)
            # For openvla-oft: responses are the action tokens (56) to compare against output logits
            if 'responses' in kwargs and kwargs['responses'] is not None:
                outputs.responses = kwargs['responses']
                logger.debug("[VLA Forward] Attached responses for loss computation")
            
            return outputs
    
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
            VLAModel: VLA model instance with weights loaded
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
        
        # Policy workers: Create on CUDA with random init (TIMM can't use meta device)
        # Weights will be loaded later via load_hf_weights() after FSDP
        vla_model = cls.from_model_args(vla_args, init_device="cuda")
        
        return vla_model
    
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
        logger.info("🔧 VLA post_to_empty_hook called - checking parameter requires_grad status")
        
        # Count frozen vs trainable params BEFORE
        frozen_before = sum(1 for _, p in self.model.named_parameters() if not p.requires_grad)
        trainable_before = sum(1 for _, p in self.model.named_parameters() if p.requires_grad)
        logger.info(f"Before unfreezing: {trainable_before} trainable, {frozen_before} frozen params")
        
        # Ensure all parameters are trainable (unfreeze everything)
        # This is needed because checkpoint weights may have requires_grad=False
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                logger.debug(f"Unfreezing parameter: {name}")
            param.requires_grad = True
        
        # Count again AFTER
        frozen_after = sum(1 for _, p in self.model.named_parameters() if not p.requires_grad)
        trainable_after = sum(1 for _, p in self.model.named_parameters() if p.requires_grad)
        logger.info(f"After unfreezing: {trainable_after} trainable, {frozen_after} frozen params")
        logger.info("✅ VLA post_to_empty_hook completed")
    
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
        parallel_dims: Optional[ParallelDims],
        device: torch.device,
        revision: Optional[str] = None,
    ):
        """
        Load weights from a HuggingFace model (following cosmos-rl pattern)
        
        This method is for distributed training with model parallelism.
        For single-GPU rollout, use load_from_checkpoint() instead.
        
        Args:
            model_name_or_path: Path to the HuggingFace model
            parallel_dims: Parallel dimensions definition (optional, not used for loading)
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
            # Load state dict from checkpoint (better for TIMM models)
            # This ensures TIMM vision backbone weights are properly loaded
            logger.info(f"Loading VLA state dict from {model_name_or_path}...")
            from safetensors import safe_open
            from pathlib import Path
            import os
            
            # Find safetensors or pytorch_model.bin files
            model_path = Path(model_name_or_path)
            if not model_path.exists():
                # Download from HF if needed
                from huggingface_hub import snapshot_download
                model_path = Path(snapshot_download(repo_id=model_name_or_path, revision=revision))
            
            # Try safetensors first (preferred for VLA models)
            safetensor_files = list(model_path.glob("*.safetensors"))
            if safetensor_files:
                logger.info(f"Loading from safetensors: {[f.name for f in safetensor_files]}")
                state_dict = {}
                for st_file in safetensor_files:
                    with safe_open(st_file, framework="pt", device=str(device)) as f:
                        for key in f.keys():
                            state_dict[key] = f.get_tensor(key)
                logger.info(f"✅ Loaded {len(state_dict)} parameters from safetensors")
            else:
                # Fallback to pytorch_model.bin
                pt_files = list(model_path.glob("pytorch_model*.bin"))
                if pt_files:
                    logger.info(f"Loading from PyTorch checkpoints: {[f.name for f in pt_files]}")
                    state_dict = {}
                    for pt_file in pt_files:
                        state_dict.update(torch.load(pt_file, map_location=device))
                    logger.info(f"✅ Loaded {len(state_dict)} parameters from PyTorch checkpoints")
                else:
                    raise FileNotFoundError(f"No safetensors or pytorch_model.bin found in {model_path}")
            
            # Load state dict into model using FSDP-compatible method
            # Following HFModel's pattern: use weight converter to shard tensors
            from cosmos_rl.policy.model.vla.weight_converter import convert_weight_from_hf
            
            logger.info("Loading weights with FSDP-compatible method...")
            
            with torch.no_grad():
                model_state_dict = self.model.state_dict()
                missing_keys = []
                loaded_keys = []
                
                for name, checkpoint_tensor in state_dict.items():
                    if name in model_state_dict:
                        target_param = model_state_dict[name]
                        
                        # Check if parameter is a DTensor (FSDP-wrapped)
                        is_dist_tensor = isinstance(target_param, torch.distributed.tensor.DTensor)
                        
                        # Get local view of the parameter
                        local_view = target_param.to_local() if is_dist_tensor else target_param
                        
                        # All parameters are FSDP-sharded uniformly, so always use weight converter
                        _, checkpoint_shard = convert_weight_from_hf(
                            checkpoint_tensor, 
                            name, 
                            parallel_dims
                        )
                        
                        # Copy sharded checkpoint to local view
                        try:
                            local_view.data.copy_(checkpoint_shard.to(device))
                            loaded_keys.append(name)
                        except Exception as copy_error:
                            logger.warning(f"Failed to copy {name}: {copy_error} (local shape={local_view.shape}, shard shape={checkpoint_shard.shape})")
                            missing_keys.append(name)
                    else:
                        missing_keys.append(name)
                
                unexpected_keys = [k for k in state_dict.keys() if k not in model_state_dict]
                logger.info(f"✅ Successfully loaded {len(loaded_keys)}/{len(state_dict)} parameters")
            
            # Check requires_grad status after loading
            frozen_count = sum(1 for _, p in self.model.named_parameters() if not p.requires_grad)
            trainable_count = sum(1 for _, p in self.model.named_parameters() if p.requires_grad)
            logger.info(f"After load_hf_weights: {trainable_count} trainable, {frozen_count} frozen params")
            if frozen_count > 0:
                logger.warning(f"⚠️  Found {frozen_count} frozen parameters after weight loading - these will be unfrozen in post_to_empty_hook")
                # Sample some frozen vision backbone params
                frozen_vb_params = [name for name, p in self.model.named_parameters() 
                                   if not p.requires_grad and 'vision_backbone' in name]
                if frozen_vb_params:
                    logger.warning(f"   Sample frozen vision_backbone params: {frozen_vb_params[:5]}")
            
            if missing_keys:
                logger.warning(f"⚠️  {len(missing_keys)} missing keys in checkpoint")
                logger.warning(f"   First 10: {missing_keys[:10]}")
            if unexpected_keys:
                logger.warning(f"⚠️  {len(unexpected_keys)} unexpected keys in checkpoint")
                logger.warning(f"   First 10: {unexpected_keys[:10]}")
                
        except Exception as e:
            logger.error(f"Failed to load VLA state dict, falling back to from_pretrained: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to from_pretrained method
            hf_model = OpenVLAForActionPrediction.from_pretrained(
                model_name_or_path, 
                config=self.hf_config,
                **kwargs
            )
            logger.info(f"✅ Loaded reference model from HF: {type(hf_model)}")
            
            # Copy weights using load_state_dict (proper way for TIMM models)
            hf_state_dict = hf_model.state_dict()
            missing_keys, unexpected_keys = self.model.load_state_dict(hf_state_dict, strict=False)
            
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
            logger.info("✅ Loaded VLA processor")
        except Exception as e:
            logger.warning(f"Failed to load VLA processor: {e}")
            
        # Load normalization stats
        self._load_vla_norm_stats(model_name_or_path)
        
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


