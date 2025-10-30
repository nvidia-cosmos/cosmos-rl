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
    processor = None
    
    if model == "openvla-oft":
        from cosmos_rl.policy.model.vla.openvla_oft.processing_prismatic import PrismaticProcessor
        logger.info(f"Creating OpenVLA-OFT processor from {name_or_path}")
        processor = PrismaticProcessor.from_pretrained(name_or_path, trust_remote_code=True)
        tokenizer = processor.tokenizer
        
    elif model == "openvla":
        from cosmos_rl.policy.model.vla.openvla.processing_prismatic import PrismaticProcessor
        logger.info(f"Creating OpenVLA processor from {name_or_path}")
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
    
    # Load norm_stats from dataset_statistics.json (required for VLA models)
    norm_stats = None
    if model in ["openvla", "openvla-oft"]:
        import json
        
        # Resolve the actual local path (handles HuggingFace downloads)
        try:
            local_model_path = resolve_model_path(name_or_path)
            logger.debug(f"Resolved model path: {name_or_path} -> {local_model_path}")
        except Exception as e:
            logger.warning(f"Failed to resolve model path: {e}, using original path")
            local_model_path = name_or_path
        
        dataset_stats_path = os.path.join(local_model_path, "dataset_statistics.json")
        if os.path.isfile(dataset_stats_path):
            try:
                with open(dataset_stats_path, "r") as f:
                    norm_stats = json.load(f)
                logger.info("✅ Loaded norm_stats from dataset_statistics.json")
            except Exception as e:
                logger.warning(f"Failed to load dataset_statistics.json: {e}")
        else:
            logger.warning(
                "No dataset_statistics.json found. This may cause issues with action normalization. "
                "You can ignore this if loading a base (not fine-tuned) VLA checkpoint."
            )
    
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
    
    # Add norm_stats to config if loaded (critical for action normalization)
    if norm_stats is not None:
        override_config_kwargs["norm_stats"] = norm_stats
        logger.info(f"Added norm_stats to config with keys: {list(norm_stats.keys())}")
    
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

