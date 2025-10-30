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
Weight conversion utilities for VLA models

This module provides utilities for converting VLA model weights between
different formats (HuggingFace, vLLM, etc.) and handling weight initialization.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

from cosmos_rl.utils.logging import logger


def convert_vla_weights_from_hf(
    hf_state_dict: Dict[str, torch.Tensor],
    vla_type: str = "openvla"
) -> Dict[str, torch.Tensor]:
    """
    Convert HuggingFace VLA weights to cosmos-rl format
    
    Args:
        hf_state_dict: HuggingFace state dictionary
        vla_type: Type of VLA model ("openvla" or "openvla-oft")
        
    Returns:
        Converted state dictionary
    """
    
    converted_state_dict = {}
    
    for name, tensor in hf_state_dict.items():
        # Convert weight names to cosmos-rl convention
        converted_name = _convert_weight_name_hf_to_cosmos(name, vla_type)
        converted_state_dict[converted_name] = tensor
        
    logger.info(f"Converted {len(hf_state_dict)} VLA weights from HuggingFace format")
    return converted_state_dict


def _convert_weight_name_hf_to_cosmos(weight_name: str, vla_type: str) -> str:
    """
    Convert HuggingFace weight name to cosmos-rl convention
    
    Args:
        weight_name: Original HuggingFace weight name
        vla_type: Type of VLA model
        
    Returns:
        Converted weight name
    """
    
    # VLA models typically keep the same naming convention
    # as HuggingFace for simplicity, but we can add conversions here
    # if needed for specific components
    
    converted_name = weight_name
    
    # Example conversions (add as needed):
    # if "vision_backbone" in weight_name:
    #     converted_name = weight_name.replace("vision_backbone.", "vision_encoder.")
    
    return converted_name


def initialize_vla_weights(
    model: nn.Module,
    init_method: str = "normal",
    init_std: float = 0.02
) -> None:
    """
    Initialize VLA model weights
    
    Args:
        model: VLA model to initialize
        init_method: Initialization method ("normal", "xavier", "kaiming")
        init_std: Standard deviation for normal initialization
    """
    
    logger.info(f"Initializing VLA weights with method: {init_method}")
    
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            if init_method == "normal":
                torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)
            elif init_method == "xavier":
                torch.nn.init.xavier_uniform_(module.weight)
            elif init_method == "kaiming":
                torch.nn.init.kaiming_uniform_(module.weight)
            
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)
            
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    model.apply(_init_weights)
    logger.info("VLA weight initialization completed")


def load_vla_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load VLA model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: VLA model instance
        strict: Whether to strictly enforce weight names match
        
    Returns:
        Checkpoint metadata dictionary
    """
    
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading VLA checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract state dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    # Load into model
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    
    if missing_keys:
        logger.warning(f"Missing keys when loading checkpoint: {missing_keys[:10]}...")
    if unexpected_keys:
        logger.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys[:10]}...")
    
    # Return metadata
    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }
    
    # Add training metadata if available
    for key in ["epoch", "step", "optimizer_state_dict", "lr_scheduler_state_dict"]:
        if key in checkpoint:
            metadata[key] = checkpoint[key]
    
    logger.info(f"Successfully loaded VLA checkpoint with {len(state_dict)} parameters")
    return metadata


def save_vla_checkpoint(
    model: nn.Module,
    save_path: str,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    optimizer_state: Optional[Dict] = None,
    lr_scheduler_state: Optional[Dict] = None,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save VLA model checkpoint
    
    Args:
        model: VLA model to save
        save_path: Path to save checkpoint
        epoch: Training epoch
        step: Training step
        optimizer_state: Optimizer state dict
        lr_scheduler_state: LR scheduler state dict  
        metadata: Additional metadata to save
    """
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving VLA checkpoint to {save_path}")
    
    # Prepare checkpoint data
    checkpoint_data = {
        "state_dict": model.state_dict(),
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }
    
    # Add training state
    if epoch is not None:
        checkpoint_data["epoch"] = epoch
    if step is not None:
        checkpoint_data["step"] = step
    if optimizer_state is not None:
        checkpoint_data["optimizer_state_dict"] = optimizer_state
    if lr_scheduler_state is not None:
        checkpoint_data["lr_scheduler_state_dict"] = lr_scheduler_state
    
    # Add metadata
    if metadata is not None:
        checkpoint_data["metadata"] = metadata
    
    # Save checkpoint
    torch.save(checkpoint_data, save_path)
    logger.info(f"Successfully saved VLA checkpoint ({len(checkpoint_data['state_dict'])} parameters)")


def verify_vla_model_compatibility(
    model_path: str,
    expected_vla_type: str
) -> bool:
    """
    Verify that a VLA model is compatible with expected type
    
    Args:
        model_path: Path to VLA model
        expected_vla_type: Expected VLA type ("openvla" or "openvla-oft")
        
    Returns:
        True if compatible, False otherwise
    """
    
    try:
        from transformers import AutoConfig
        
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # Check model type
        model_type = getattr(config, "model_type", "")
        if model_type != "openvla":
            logger.warning(f"Expected model_type 'openvla', got '{model_type}'")
            return False
        
        # Check architecture compatibility
        if expected_vla_type == "openvla-oft":
            # Check for OFT-specific components
            if not hasattr(config, "use_oft") or not config.use_oft:
                logger.warning("Expected OFT model but use_oft=False or missing")
                return False
        
        logger.info(f"VLA model compatibility verified for type: {expected_vla_type}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to verify VLA model compatibility: {e}")
        return False

