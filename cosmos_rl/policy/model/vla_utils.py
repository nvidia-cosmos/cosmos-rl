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
VLA Configuration Utilities

This module provides utility functions for creating and configuring VLA models.
It is designed to be imported by both trainer and model modules without causing
circular dependencies.

Key Functions:
    - create_vla_config: Main interface for creating VLA configurations
    - set_pad_token_id: Helper to set pad_token_id for tokenizers
    - update_model_config: Helper to update model configs with override kwargs
"""

import warnings
import os
import numpy as np
from typing import Optional, Any, Tuple
from transformers import AutoTokenizer, AutoConfig, AutoProcessor, PretrainedConfig
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.util import resolve_model_path


def set_pad_token_id(tokenizer):
    """
    Set pad_token_id for tokenizer if not present.
    
    Args:
        tokenizer: HuggingFace tokenizer instance
        
    Note:
        If pad_token_id is None, it will be set to eos_token_id if available,
        otherwise defaults to 0.
    """
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info(f"Set pad_token_id to eos_token_id: {tokenizer.pad_token_id}")
        else:
            tokenizer.pad_token_id = 0
            logger.info(f"Set pad_token_id to default: {tokenizer.pad_token_id}")
    else:
        logger.debug(f"pad_token_id already set: {tokenizer.pad_token_id}")


def normalize_norm_stats(norm_stats: dict, action_dim: int = 7) -> dict:
    """
    Normalize norm_stats to ensure correct shape for action statistics.
    
    VLA models predict multiple action chunks (temporal predictions), but all chunks
    use the SAME action space normalization. This function ensures norm_stats are
    stored per-dimension, not concatenated across chunks.
    
    Args:
        norm_stats: Dictionary of normalization statistics from dataset_statistics.json
        action_dim: Dimensionality of action space (default: 7 for LIBERO)
                   If 0 or not specified, will infer from the smallest stat array size
        
    Returns:
        Normalized norm_stats with correct shapes
        
    Example:
        If norm_stats["dataset"]["action"]["min"] has shape (14,) for 2 chunks of 7 dims,
        this will slice it to (7,) since all chunks use the same normalization.
    """
    # Auto-detect action_dim if not provided or is 0
    if action_dim <= 0:
        # Find the smallest action stat array size across all datasets
        min_size = float('inf')
        for dataset_stats in norm_stats.values():
            if 'action' in dataset_stats:
                for stat_key in ['min', 'max', 'q01', 'q99']:
                    if stat_key in dataset_stats['action']:
                        size = len(dataset_stats['action'][stat_key])
                        min_size = min(min_size, size)
        
        if min_size != float('inf') and min_size > 0:
            action_dim = min_size
            logger.info(f"Auto-detected action_dim = {action_dim} from norm_stats")
        else:
            action_dim = 7  # Default fallback for LIBERO
            logger.warning(f"Could not auto-detect action_dim, using default: {action_dim}")
    
    normalized_stats = {}
    
    for dataset_key, dataset_stats in norm_stats.items():
        if 'action' not in dataset_stats:
            normalized_stats[dataset_key] = dataset_stats
            continue
            
        action_stats = dataset_stats['action'].copy()
        
        # Check and fix shape of action statistics
        for stat_key in ['min', 'max', 'q01', 'q99', 'mask']:
            if stat_key not in action_stats:
                continue
                
            stat_array = np.array(action_stats[stat_key])
            
            # If stats are concatenated across chunks (e.g., 14 = 2 * 7), slice to first action_dim
            # Only normalize if array is significantly larger (at least 1.5x)
            if len(stat_array.shape) == 1 and stat_array.shape[0] > action_dim * 1.5:
                logger.info(
                    f"Normalizing {dataset_key}.action.{stat_key}: "
                    f"shape {stat_array.shape} -> ({action_dim},) [concatenated chunks detected]"
                )
                action_stats[stat_key] = stat_array[:action_dim].tolist()
            else:
                # Keep as-is (already correct shape)
                action_stats[stat_key] = stat_array.tolist()
        
        # Reconstruct dataset stats with normalized action stats
        normalized_stats[dataset_key] = {**dataset_stats, 'action': action_stats}
    
    return normalized_stats


def update_model_config(config: PretrainedConfig, override_config_kwargs: Optional[dict] = None):
    """
    Update model config with override kwargs.
    
    This function dynamically adds attributes to the config object using setattr.
    This is the approach used by SimpleVLA-RL to add custom parameters like
    use_proprio, proprio_dim, etc. to the model configuration.
    
    Args:
        config: HuggingFace PretrainedConfig instance
        override_config_kwargs: Dictionary of config attributes to set
        
    Example:
        >>> config = AutoConfig.from_pretrained("model_name")
        >>> update_model_config(config, {"use_proprio": True, "proprio_dim": 7})
    """
    if override_config_kwargs:
        for key, val in override_config_kwargs.items():
            setattr(config, key, val)
            logger.debug(f"Updated config.{key} = {val}")


def create_vla_config(
    name_or_path: str,
    cosmos_config=None,
    correct_pad_token: bool = True,
    correct_gemma2: bool = True,
    **kwargs
) -> Tuple[PretrainedConfig, Optional[Any], AutoTokenizer]:
    """
    Create VLA configuration with processor and tokenizer.
    
    This is the main interface for VLA configuration creation, used by both
    the trainer initialization and model base classes.
    
    Args:
        name_or_path: HuggingFace model name or local path
        cosmos_config: CosmosConfig instance (optional)
        correct_pad_token: Whether to auto-correct pad_token_id if missing
        correct_gemma2: Whether to apply Gemma-2 specific corrections
        **kwargs: Additional keyword arguments for tokenizer/config creation
        
    Returns:
        Tuple of (vla_config, processor, tokenizer)
        - vla_config: HuggingFace config with VLA-specific parameters added
        - processor: VLA processor (for openvla/openvla-oft) or None
        - tokenizer: HuggingFace tokenizer instance
        
    Example:
        >>> from cosmos_rl.policy.config import Config
        >>> config = Config.from_toml("config.toml")
        >>> vla_config, processor, tokenizer = create_vla_config(
        ...     "openvla/openvla-7b",
        ...     cosmos_config=config,
        ...     model="openvla-oft"
        ... )
        
    Note:
        This function avoids circular dependencies by not importing from
        base.py or vla/__init__.py
    """
    # Import cosmos config only if needed (to avoid import at module level)
    if cosmos_config is not None:
        from cosmos_rl.policy.config import Config as CosmosConfig
    
    # Apply Gemma-2 specific corrections
    if correct_gemma2 and isinstance(name_or_path, str) and 'gemma-2-2b-it' in name_or_path:
        warnings.warn('Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.')
        kwargs['eos_token'] = '<end_of_turn>'
        kwargs['eos_token_id'] = 107
    
    # Determine VLA model type from cosmos config or kwargs
    if cosmos_config is not None and hasattr(cosmos_config, 'vla'):
        model = cosmos_config.vla.vla_type or kwargs.get("model", None)
    else:
        model = kwargs.get("model", None)
    
    # Create processor and tokenizer based on VLA type
    
    if model == "openvla-oft":
        from cosmos_rl.policy.model.vla.openvla_oft.configuration_prismatic import OpenVLAConfig
        from cosmos_rl.policy.model.vla.openvla_oft.processing_prismatic import PrismaticProcessor
        logger.info(f"Creating OpenVLA-OFT processor from {name_or_path}")
        AutoConfig.register("openvla", OpenVLAConfig)
        processor = PrismaticProcessor.from_pretrained(name_or_path, trust_remote_code=True)
        tokenizer = processor.tokenizer
        
    elif model == "openvla":
        from cosmos_rl.policy.model.vla.openvla.configuration_prismatic import OpenVLAConfig
        from cosmos_rl.policy.model.vla.openvla.processing_prismatic import PrismaticProcessor
        logger.info(f"Creating OpenVLA processor from {name_or_path}")
        AutoConfig.register("openvla", OpenVLAConfig)
        processor = PrismaticProcessor.from_pretrained(name_or_path, trust_remote_code=True)
        tokenizer = processor.tokenizer
        
    else:
        # Standard HuggingFace tokenizer for non-VLA or unknown types
        logger.info(f"Creating standard tokenizer from {name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True, **kwargs)
        processor = None
    
    # Correct pad_token_id if requested
    if correct_pad_token and tokenizer:
        set_pad_token_id(tokenizer)
    
    # Load base VLA config
    logger.info(f"Loading VLA config from {name_or_path}")
    vla_config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=True)
    
    # Check if config already has norm_stats (from model's config.json)
    config_has_norm_stats = hasattr(vla_config, 'norm_stats') and vla_config.norm_stats is not None
    if config_has_norm_stats:
        logger.info("Config already contains norm_stats (from model's config.json)")
    
    # Load norm_stats from dataset_statistics.json (required for VLA models)
    norm_stats = None
    if model in ["openvla", "openvla-oft"]:
        import json
        
        # Try to load dataset_statistics.json
        # First check if model is already downloaded locally
        try:
            local_model_path = resolve_model_path(name_or_path)
            dataset_stats_path = os.path.join(local_model_path, "dataset_statistics.json")
            
            if os.path.isfile(dataset_stats_path):
                # Load from local path
                with open(dataset_stats_path, "r") as f:
                    norm_stats = json.load(f)
                logger.info("✅ Loaded norm_stats from local dataset_statistics.json")
            else:
                logger.info("Model not fully downloaded, attempting to fetch dataset_statistics.json only...")
                # Model not downloaded yet - download just the JSON file
                from huggingface_hub import hf_hub_download
                try:
                    json_path = hf_hub_download(
                        repo_id=name_or_path,
                        filename="dataset_statistics.json",
                        repo_type="model"
                    )
                    with open(json_path, "r") as f:
                        norm_stats = json.load(f)
                    logger.info("✅ Loaded norm_stats from HuggingFace Hub (single file download)")
                except Exception as e:
                    logger.warning(
                        f"Failed to download dataset_statistics.json from HuggingFace Hub: {e}. "
                        "This may cause issues with action normalization. "
                        "You can ignore this if loading a base (not fine-tuned) VLA checkpoint."
                    )
        except Exception as e:
            logger.warning(f"Failed to load dataset_statistics.json: {e}")
    
    # Extract VLA-specific parameters from cosmos config or kwargs
    if cosmos_config is not None and hasattr(cosmos_config, 'vla'):
        use_proprio = cosmos_config.vla.use_proprio
        proprio_dim = cosmos_config.vla.action_dim
        num_images_in_input = cosmos_config.vla.num_images_in_input
        logger.info(
            f"Using VLA params from cosmos config: "
            f"use_proprio={use_proprio}, proprio_dim={proprio_dim}, "
            f"num_images={num_images_in_input}"
        )
    else:
        # Fallback to kwargs for backward compatibility
        use_proprio = kwargs.get("use_proprio", False)
        proprio_dim = kwargs.get("action_dim", 0)
        num_images_in_input = kwargs.get("num_images_in_input", 1)
        logger.info(
            f"Using VLA params from kwargs: "
            f"use_proprio={use_proprio}, proprio_dim={proprio_dim}, "
            f"num_images={num_images_in_input}"
        )
    
    # Build override config kwargs
    override_config_kwargs = {}
    
    # Add tokenizer IDs if tokenizer is available
    if tokenizer:
        override_config_kwargs.update({
            'bos_token_id': tokenizer.bos_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'pad_token_id': tokenizer.pad_token_id,
        })
    
    # Add VLA-specific parameters
    override_config_kwargs["use_proprio"] = use_proprio
    override_config_kwargs["proprio_dim"] = proprio_dim
    override_config_kwargs["num_images_in_input"] = num_images_in_input
    override_config_kwargs["vla_type"] = model
    
    # Handle norm_stats from multiple sources and normalize them
    # Priority: dataset_statistics.json > config.norm_stats
    final_norm_stats = None
    if norm_stats is not None:
        # Loaded from dataset_statistics.json
        logger.info("Using norm_stats from dataset_statistics.json")
        final_norm_stats = norm_stats
    elif config_has_norm_stats:
        # Use norm_stats from model's config.json
        logger.info("Using norm_stats from model's config.json")
        final_norm_stats = vla_config.norm_stats
    
    # Normalize norm_stats to ensure correct shape (fix root cause of shape mismatches)
    if final_norm_stats is not None:
        final_norm_stats = normalize_norm_stats(final_norm_stats, action_dim=proprio_dim)
        override_config_kwargs["norm_stats"] = final_norm_stats
        logger.info(f"✅ Normalized and added norm_stats to config with keys: {list(final_norm_stats.keys())}")
    else:
        logger.warning("⚠️  No norm_stats available from any source!")
    
    # Apply overrides to config
    update_model_config(vla_config, override_config_kwargs=override_config_kwargs)
    
    logger.info(f"✅ Created VLA config for type: {model}")
    logger.debug(f"VLA config attributes: {override_config_kwargs}")
    
    return vla_config, processor, tokenizer


def get_vla_processor(name_or_path: str, vla_type: str):
    """
    Get VLA processor for a specific VLA type.
    
    Args:
        name_or_path: HuggingFace model name or local path
        vla_type: Type of VLA model ("openvla" or "openvla-oft")
        
    Returns:
        PrismaticProcessor instance or None
        
    Raises:
        ValueError: If vla_type is not supported
    """
    if vla_type == "openvla-oft":
        from cosmos_rl.policy.model.vla.openvla_oft.processing_prismatic import PrismaticProcessor
        logger.info(f"Loading OpenVLA-OFT processor from {name_or_path}")
        return PrismaticProcessor.from_pretrained(name_or_path, trust_remote_code=True)
        
    elif vla_type == "openvla":
        from cosmos_rl.policy.model.vla.openvla.processing_prismatic import PrismaticProcessor
        logger.info(f"Loading OpenVLA processor from {name_or_path}")
        return PrismaticProcessor.from_pretrained(name_or_path, trust_remote_code=True)
        
    elif vla_type is None or vla_type == "":
        logger.warning("No VLA type specified, returning None for processor")
        return None
        
    else:
        raise ValueError(f"Unsupported VLA type: {vla_type}. Expected 'openvla' or 'openvla-oft'")


def validate_vla_config(config: PretrainedConfig, vla_type: str) -> bool:
    """
    Validate that a VLA config has all required attributes.
    
    Args:
        config: HuggingFace PretrainedConfig instance
        vla_type: Type of VLA model
        
    Returns:
        True if config is valid, False otherwise
        
    Note:
        Logs warnings for missing attributes but does not raise exceptions.
    """
    required_attrs = ["use_proprio", "proprio_dim", "num_images_in_input"]
    
    is_valid = True
    for attr in required_attrs:
        if not hasattr(config, attr):
            logger.warning(f"VLA config missing required attribute: {attr}")
            is_valid = False
    
    # Check model type consistency
    if hasattr(config, "vla_type") and config.vla_type != vla_type:
        logger.warning(
            f"VLA type mismatch: config has '{config.vla_type}' but expected '{vla_type}'"
        )
    
    if is_valid:
        logger.debug(f"VLA config validation passed for type: {vla_type}")
    
    return is_valid


