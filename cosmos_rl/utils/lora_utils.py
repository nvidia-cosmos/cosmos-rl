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
Shared LoRA utilities for Cosmos-RL.

This module provides common functionality for merging LoRA adapters with base models,
used across evaluation, inference, and other components.
"""

import os
import json
import traceback
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def merge_lora_model(lora_path: str, base_model_path: Optional[str] = None) -> str:
    """
    Merge LoRA weights with base model.

    This function handles the complete LoRA merging process:
    1. Checks if a merged model already exists (caching)
    2. Loads the base model and LoRA adapter
    3. Merges the weights and saves the combined model
    4. Handles processor/tokenizer saving
    5. Provides proper error handling and memory cleanup

    Args:
        lora_path: Path to the LoRA model directory
        base_model_path: Path to the base model (optional, can be inferred from adapter config)

    Returns:
        Path to the merged model directory

    Raises:
        ValueError: If base_model_path cannot be determined
        ImportError: If required libraries (transformers, peft) are not available
    """
    logger.info(f"Merging LoRA model: {lora_path}")

    # Check if already merged
    merged_path = lora_path.replace("safetensors", "merged")
    if os.path.exists(merged_path) and os.path.isdir(merged_path):
        logger.info(f"Merged model already exists at: {merged_path}")
        return merged_path

    try:
        # Import required libraries for LoRA merging
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from peft import PeftModel
        import torch

        # Use provided base model path or infer from LoRA config
        if not base_model_path:
            # Try to read base model path from adapter config
            adapter_config_path = os.path.join(lora_path, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                    base_model_path = adapter_config.get('base_model_name_or_path')

            if not base_model_path:
                raise ValueError(
                    "Base model path not provided and could not be inferred from adapter config. "
                    "Please provide base_model_path parameter or ensure adapter_config.json exists "
                    f"in {lora_path} with 'base_model_name_or_path' field."
                )

        logger.info(f"Loading base model: {base_model_path}")
        logger.info(f"Loading LoRA adapter: {lora_path}")

        # Load base model
        logger.info("Step 1/4: Loading base model...")
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_model_path,
                torch_dtype="auto"
            )
            logger.info(f"Base model loaded successfully. Type: {type(model)}")
        except Exception as e:
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            logger.error(f"Failed to load base model from {base_model_path}: {e}")
            raise RuntimeError(f"Base model loading failed: {e}") from e

        # Load LoRA adapter
        logger.info("Step 2/4: Loading LoRA adapter...")
        logger.info(f"  Base model device: {next(model.parameters()).device if model.parameters() else 'N/A'}")
        try:
            peft_model = PeftModel.from_pretrained(model, lora_path)
            logger.info(f"LoRA adapter loaded successfully. Type: {type(peft_model)}")
        except Exception as e:
            logger.error(f"Failed to load LoRA adapter from {lora_path}: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise RuntimeError(f"LoRA adapter loading failed: {e}") from e

        # Merge and unload
        logger.info("Step 3/4: Merging LoRA weights with base model...")
        try:
            merged_model = peft_model.merge_and_unload()
            logger.info(f"LoRA weights merged successfully. Type: {type(merged_model)}")
        except Exception as e:
            logger.error(f"Failed during merge_and_unload: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise RuntimeError(f"LoRA merge failed: {e}") from e

        # Save merged model
        logger.info(f"Step 4/4: Saving merged model to: {merged_path}")
        os.makedirs(merged_path, exist_ok=True)
        merged_model.save_pretrained(merged_path)

        # Also save the processor/tokenizer
        try:
            processor = AutoProcessor.from_pretrained(base_model_path)
            processor.save_pretrained(merged_path)
            logger.info("Saved processor to merged model directory")
        except Exception as e:
            logger.warning(f"Failed to save processor: {e}")

        # Clean up GPU memory
        del model, peft_model, merged_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"LoRA merging completed successfully: {merged_path}")
        return merged_path

    except ImportError as e:
        logger.error(f"Required libraries not available for LoRA merging: {e}")
        logger.error("Please ensure 'transformers' and 'peft' are installed")
        raise

    except Exception as e:
        logger.error(f"LoRA merging failed: {e}")
        logger.warning(f"Falling back to original model path: {lora_path}")
        return lora_path


def should_enable_lora(config: dict, enable_lora_flag: Optional[bool] = None) -> bool:
    """
    Determine if LoRA should be enabled based on configuration and flags.

    Args:
        config: Configuration dictionary that may contain LoRA settings
        enable_lora_flag: Explicit LoRA enable flag (overrides config)

    Returns:
        True if LoRA should be enabled, False otherwise
    """
    # Explicit flag takes precedence
    if enable_lora_flag is not None:
        return enable_lora_flag

    # Check various config paths for LoRA settings
    model_config = config.get("model", {})
    return model_config.get("enable_lora", False)


def get_base_model_path(config: dict, explicit_path: Optional[str] = None) -> Optional[str]:
    """
    Get base model path from configuration or explicit parameter.

    Args:
        config: Configuration dictionary that may contain base model path
        explicit_path: Explicitly provided base model path

    Returns:
        Base model path if found, None otherwise
    """
    # Explicit path takes precedence
    if explicit_path:
        return explicit_path

    # Check config for base model path
    model_config = config.get("model", {})
    return model_config.get("base_model_path")
