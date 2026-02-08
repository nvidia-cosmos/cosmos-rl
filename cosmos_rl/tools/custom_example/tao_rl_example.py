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

"""TAO-compatible GRPO/RL example with custom logger and hooks.

This script demonstrates how to use custom_logger_fns and hook_fns for TAO-compatible
status logging in RL/GRPO training. The TAOStatusLogger writes to status.json in the
format expected by TAO/NVAIE.

Key differences from SFT:
- Uses reward functions for GRPO training
- Validation monitors reward metrics (val/reward_avg) instead of loss
- Supports epoch-based validation for TAO AutoML hyperparameter optimization

Usage:
    cosmos-rl --config spec.toml cosmos_rl/tools/custom_example/tao_rl_example.py

Environment Variables for TAO logging:
    TAO_API_JOB_ID: Job ID for status file path
    TAO_API_RESULTS_DIR: Results directory (defaults to /results)

The status file is written to: {TAO_API_RESULTS_DIR}/{TAO_API_JOB_ID}/status.json
"""

import argparse
import json
import os
import re
import warnings
from pathlib import Path
from typing import List, Optional, Union

import cosmos_rl.launcher.worker_entry
import cosmos_rl.policy.config
import pydantic
import toml
import torch.utils.data
from cosmos_rl.dispatcher.data.packer import HFVLMDataPacker
from cosmos_rl.utils.logging import logger

# Import TAO status logger utilities
from cosmos_rl.tools.custom_hooks import TAOStatusLogger

# Optional: Import bert_score for open-ended QA evaluation
try:
    from bert_score import score as bert_score_fn

    HAS_BERT_SCORE = True
except ImportError:
    HAS_BERT_SCORE = False
    logger.warning("bert_score not found - open-ended QA will use simple string matching")

# Import TAO core logging for STARTED/SUCCESS/FAILURE status
try:
    from nvidia_tao_core.loggers.logging import (
        Status,
        StatusLogger,
        Verbosity,
        set_status_logger,
    )

    HAS_TAO_CORE = True
except ImportError:
    HAS_TAO_CORE = False
    logger.warning("nvidia_tao_core not found - job lifecycle status logging disabled")

# Optional: Import cosmos_reason1_utils if available
try:
    from cosmos_reason1_utils.text import create_conversation
    from cosmos_reason1_utils.vision import VisionConfig

    HAS_COSMOS_REASON1_UTILS = True
except ImportError:
    HAS_COSMOS_REASON1_UTILS = False
    logger.warning("cosmos_reason1_utils not found, using fallback conversation format")

    class VisionConfig(pydantic.BaseModel):
        fps: int = 1
        max_pixels: int = 81920


class CustomDatasetConfig(pydantic.BaseModel):
    """Dataset configuration - matches SFT example structure."""

    annotation_path: str = pydantic.Field()
    """Dataset annotation path."""
    media_path: str = pydantic.Field(default="")
    """Dataset media path."""


class RewardConfig(pydantic.BaseModel):
    """Configuration for reward functions."""

    accuracy_weight: float = pydantic.Field(default=1.0)
    """Weight for accuracy reward."""
    format_weight: float = pydantic.Field(default=0.2)
    """Weight for format reward (default 0.2 like original)."""


class CustomConfig(pydantic.BaseModel):
    """Custom config - matches SFT example structure."""

    train_dataset: CustomDatasetConfig = pydantic.Field()
    """Training dataset config."""

    val_dataset: Optional[CustomDatasetConfig] = pydantic.Field(default=None)
    """Validation dataset config (optional but recommended for TAO AutoML)."""

    system_prompt: str = pydantic.Field(default="")
    """System prompt."""

    vision: VisionConfig = pydantic.Field(
        default=VisionConfig(
            fps=1,
            max_pixels=81920,
        )
    )
    """Vision processor config."""

    reward: RewardConfig = pydantic.Field(default_factory=RewardConfig)
    """Reward function configuration."""


class CosmosGRPODataset(torch.utils.data.Dataset):
    """Dataset for GRPO training with VLM data.

    Uses eager initialization pattern matching SFT - data is loaded in __init__.

    Note: For GRPO, __getitem__ returns just the conversation list (prompt).
    The reference answer is accessed via get_reference_answer(idx) method,
    which is called by the framework for reward computation.
    """

    def __init__(
        self,
        config: cosmos_rl.policy.config.Config,
        custom_config: CustomConfig,
        annotation_path: str,
        media_path: str,
    ):
        """Initialize dataset with data loaded eagerly.

        Args:
            config: Cosmos-RL configuration
            custom_config: Custom configuration
            annotation_path: Path to annotation JSON file
            media_path: Path to media files
        """
        self.annotation = json.load(open(annotation_path))
        self.media_path = media_path
        self.system_prompt = custom_config.system_prompt
        self.config = config
        self.custom_config = custom_config
        self.vision_kwargs = custom_config.vision.model_dump(exclude_none=True)

    def setup(self, config, tokenizer):
        """Setup method required by the GRPO trainer.

        For our custom dataset, we don't need additional setup beyond __init__.
        """
        pass

    def __len__(self):
        return len(self.annotation)

    def get_reference_answer(self, idx: int) -> str:
        """Get the reference answer for reward computation.

        This method is called by the framework (RLDataset wrapper) to get
        the ground truth answer for computing rewards during GRPO training.

        Args:
            idx: Index of the sample

        Returns:
            Reference answer string for the sample
        """
        return self.annotation[idx]["conversations"][1]["value"]

    def __getitem__(self, idx: int) -> list:
        """Get conversation for a sample.

        Returns just the conversation list (not a dict) because the framework
        expects the prompt to be passed directly to the VLM data packer.
        Reference answer is accessed separately via get_reference_answer().

        Args:
            idx: Index of the sample

        Returns:
            List of conversation messages for rollout generation
        """
        sample = self.annotation[idx]

        user_prompt = sample["conversations"][0]["value"]
        images = sample.get("image", None) or sample.get("images", None)
        if images and isinstance(images, str):
            images = [images]
        videos = sample.get("video", None)
        if videos and isinstance(videos, str):
            videos = [videos]

        # If self.media_path is not empty, join it with each image/video path
        if self.media_path != "":
            if images:
                images = [os.path.join(self.media_path, img) for img in images]
            if videos:
                videos = [os.path.join(self.media_path, vid) for vid in videos]

        # Remove image and video tags from user prompt
        user_prompt = re.sub(r"(\n)?</?(image|video)>(\n)?", "", user_prompt)

        if HAS_COSMOS_REASON1_UTILS:
            conversations = create_conversation(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                images=images,
                videos=videos,
                vision_kwargs=self.vision_kwargs,
            )
        else:
            # Fallback conversation format with image/video support
            conversations = []
            if self.system_prompt:
                conversations.append({"role": "system", "content": self.system_prompt})

            # Build user content with media (images/videos) and vision_kwargs
            if images or videos:
                user_content = []
                if images:
                    for img in images:
                        user_content.append(
                            {"type": "image", "image": img, **self.vision_kwargs}
                        )
                if videos:
                    for vid in videos:
                        user_content.append(
                            {"type": "video", "video": vid, **self.vision_kwargs}
                        )
                user_content.append({"type": "text", "text": user_prompt})
                conversations.append({"role": "user", "content": user_content})
            else:
                conversations.append({"role": "user", "content": user_prompt})

        return conversations


class CosmosGRPOValDataset(CosmosGRPODataset):
    """Validation dataset for GRPO - same as training but may use different split."""

    pass


def _get_results_dir() -> str:
    """Get the results directory based on TAO environment variables."""
    job_id = os.environ.get("TAO_API_JOB_ID")
    if job_id:
        results_base = os.environ.get("TAO_API_RESULTS_DIR", "/results")
        return os.path.join(results_base, job_id)
    return "./results"


def _is_master_rank() -> bool:
    """Check if current process is the master rank for status logging."""
    cosmos_role = os.environ.get("COSMOS_ROLE", "")
    node_rank = int(os.environ.get("NODE_RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    is_worker = cosmos_role != "Controller"
    return is_worker and (node_rank == 0) and (local_rank == 0)


def monitor_status(experiment_name: str = "Cosmos-RL GRPO"):
    """Decorator to monitor job status (STARTED/SUCCESS/FAILURE).

    Only logs status from master rank to minimize memory overhead.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            s_logger = None

            # Only setup logger on master rank
            if HAS_TAO_CORE and _is_master_rank():
                results_dir = _get_results_dir()
                os.makedirs(results_dir, exist_ok=True)
                status_file = os.path.join(results_dir, "status.json")

                s_logger = StatusLogger(
                    filename=status_file,
                    is_master=True,
                    verbosity=Verbosity.INFO,
                    append=True,
                )
                set_status_logger(s_logger)
                logger.info(f"Job lifecycle status will be logged to: {status_file}")

                # Log STARTED
                s_logger.write(
                    status_level=Status.STARTED,
                    message=f"Starting {experiment_name} training",
                )
                logger.info(f"Job STARTED: {experiment_name}")

            try:
                result = func(*args, **kwargs)

                # Log SUCCESS
                if s_logger:
                    s_logger.write(
                        status_level=Status.RUNNING,
                        message=f"{experiment_name} training completed successfully",
                    )
                    logger.info(f"Job SUCCESS: {experiment_name}")

                return result

            except (KeyboardInterrupt, SystemExit) as e:
                if s_logger:
                    try:
                        s_logger.write(
                            status_level=Status.FAILURE,
                            verbosity_level=Verbosity.WARNING,
                            message=f"{experiment_name} training was interrupted: {str(e)}",
                        )
                    except Exception:
                        pass
                    logger.warning(f"Job INTERRUPTED: {experiment_name}")
                raise

            except Exception as e:
                if s_logger:
                    try:
                        s_logger.write(
                            status_level=Status.FAILURE,
                            verbosity_level=Verbosity.ERROR,
                            message=f"{experiment_name} training failed: {str(e)}",
                        )
                    except Exception:
                        pass
                    logger.error(f"Job FAILED: {experiment_name} - {str(e)}")
                raise

        return wrapper

    return decorator


# ============================================================================
# Reward Functions for GRPO (matches original example format)
# ============================================================================


def accuracy_reward_fn(
    to_be_evaluated: str,
    reference: Union[str, None],
    **kwargs,
) -> float:
    """Reward function that checks if the completion is correct.

    Handles 3 types of QA:
    1. MCQ: Ends with </think>\n\n[Option] -> Exact match of option.
    2. Binary: Contains </think>\n\nYes. or No. -> Exact match of Yes/No.
    3. Open-ended: Other formats -> BertScore F1 (or simple matching if unavailable).

    Args:
        to_be_evaluated: Model-generated response
        reference: Ground truth reference answer

    Returns:
        Reward between 0.0 and 1.0
    """
    if reference is None:
        return 0.0

    reward = 0.0

    try:
        # Determine type based on reference
        is_mcq = False
        is_binary = False

        # Normalize reference for type detection
        ref_stripped = reference.strip()

        # Check for MCQ: single letter A-Z, or ends with </think>\n\n[Letter]
        if re.match(r"^[A-Za-z]$", ref_stripped):
            # Simple single-letter MCQ answer like "A", "B", "C", "D"
            is_mcq = True
        elif re.search(r"</think>\s*\n\s*[A-Z]\s*$", reference):
            # Format with think tag: <think>...</think>\n\nA
            is_mcq = True
        # Check for Binary: Yes/No (simple or with think tag)
        elif re.match(r"^(Yes|No)\.?$", ref_stripped, re.IGNORECASE):
            # Simple Yes/No answer
            is_binary = True
        elif re.search(r"</think>\s*\n\s*(Yes|No)\.", reference):
            # Format with think tag
            is_binary = True

        if is_mcq or is_binary:

            def extract_final_answer(s: str) -> str:
                """Extract final answer after </think> tag."""
                if "</think>" in s:
                    after_think = s.split("</think>", 1)[1].strip()
                else:
                    after_think = s.strip()
                # Find first non-empty line
                for line in after_think.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    # Return the first word (letters)
                    match = re.match(r"^([A-Za-z]+)", line)
                    if match:
                        return match.group(1).strip()
                    return line
                return ""

            def normalize_answer(s: str) -> str:
                """Normalize answer for comparison."""
                s = s.lower().strip()
                if s.endswith("."):
                    s = s[:-1].rstrip()
                return s

            ground_truth = extract_final_answer(reference)
            student_answer = extract_final_answer(to_be_evaluated)

            norm_gt = normalize_answer(ground_truth)
            norm_student = normalize_answer(student_answer)

            if norm_student == norm_gt:
                reward = 1.0
        else:
            # Open-ended QA -> BertScore or simple matching

            def extract_content(s: str) -> str:
                """Extract content after </think> tag."""
                if "</think>" in s:
                    return s.split("</think>", 1)[1].strip()
                return s.strip()

            ref_content = extract_content(reference)
            cand_content = extract_content(to_be_evaluated)

            # Avoid empty strings
            if not ref_content:
                ref_content = " "
            if not cand_content:
                cand_content = " "

            if HAS_BERT_SCORE:
                # Use BertScore for semantic similarity
                # Force CPU to avoid CUDA re-initialization error in forked subprocess
                P, R, F1 = bert_score_fn(
                    [cand_content],
                    [ref_content],
                    lang="en",
                    verbose=False,
                    rescale_with_baseline=True,
                    device="cpu",
                )
                reward = F1.item()
            else:
                # Fallback: simple string matching
                if cand_content.lower().strip() == ref_content.lower().strip():
                    reward = 1.0
                else:
                    reward = 0.0

    except Exception as e:
        logger.warning(f"Error in accuracy_reward_fn: {e}")
        reward = 0.0

    return reward


def format_reward_fn(
    to_be_evaluated: str,
    reference: Union[str, None] = None,
    **kwargs,
) -> float:
    """Check if response matches expected format: <think>...</think>\n\n(answer).

    Args:
        to_be_evaluated: Model-generated response
        reference: Not used, kept for interface consistency

    Returns:
        1.0 if format is correct, 0.0 otherwise
    """
    try:
        pattern = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n\n([\s\S]+)$"
        match = re.search(pattern, to_be_evaluated, re.DOTALL)
        if match is None or len(match.groups()) != 2:
            return 0.0
        else:
            return 1.0
    except Exception as e:
        logger.debug(f"Exception in format_reward_fn: {e}")
        return 0.0


def create_reward_functions(reward_config: RewardConfig) -> List:
    """Create combined reward function matching original example.

    The combined reward = accuracy + format * format_weight

    Args:
        reward_config: Configuration for reward weights

    Returns:
        List with single combined reward function
    """

    def custom_reward_fn(
        to_be_evaluated: str,
        reference: Union[str, None] = None,
        **kwargs,
    ) -> float:
        """Combined reward function matching original example."""
        return sum(
            [
                accuracy_reward_fn(to_be_evaluated, reference, **kwargs)
                * reward_config.accuracy_weight,
                format_reward_fn(to_be_evaluated, reference, **kwargs)
                * reward_config.format_weight,
            ]
        )

    return [custom_reward_fn]


# ============================================================================
# Main Entry Point
# ============================================================================


@monitor_status("Cosmos-RL GRPO")
def main():
    # Suppress warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config file."
    )
    args = parser.parse_known_args()[0]

    # Load config
    with open(args.config, encoding="utf-8") as f:
        config_kwargs = toml.load(f)
    config = cosmos_rl.policy.config.Config.from_dict(config_kwargs)
    custom_config = CustomConfig.model_validate(config_kwargs.get("custom", {}))

    # Save config if controller
    role = os.environ.get("COSMOS_ROLE")
    is_controller = role == "Controller"
    if is_controller:
        output_dir = Path(config.train.output_dir).resolve().parent
        output_dir.mkdir(parents=True, exist_ok=True)

        config_kwargs_to_save = config.model_dump()
        config_kwargs_to_save["custom"] = custom_config.model_dump()
        config_path = output_dir / "config.toml"
        config_path.write_text(toml.dumps(config_kwargs_to_save))
        logger.info(f"Saved config to {config_path}")

    # Factory function for training dataset (matches SFT pattern)
    def get_train_dataset(
        config: cosmos_rl.policy.config.Config,
    ) -> torch.utils.data.Dataset:
        """Factory function to create GRPO training dataset."""
        custom_cfg = CustomConfig.model_validate(config.model_dump().get("custom", {}))

        logger.info(
            f"Creating GRPO training dataset from: {custom_cfg.train_dataset.annotation_path}"
        )
        return CosmosGRPODataset(
            config=config,
            custom_config=custom_cfg,
            annotation_path=custom_cfg.train_dataset.annotation_path,
            media_path=custom_cfg.train_dataset.media_path,
        )

    # Factory function for validation dataset
    def get_val_dataset(
        config: cosmos_rl.policy.config.Config,
    ) -> torch.utils.data.Dataset:
        """Factory function to create GRPO validation dataset."""
        custom_cfg = CustomConfig.model_validate(config.model_dump().get("custom", {}))

        if not custom_cfg.val_dataset:
            logger.warning(
                "No validation dataset specified. "
                "For TAO AutoML, validation dataset is recommended for monitoring metrics."
            )
            return None

        logger.info(
            f"Creating GRPO validation dataset from: {custom_cfg.val_dataset.annotation_path}"
        )
        return CosmosGRPOValDataset(
            config=config,
            custom_config=custom_cfg,
            annotation_path=custom_cfg.val_dataset.annotation_path,
            media_path=custom_cfg.val_dataset.media_path,
        )

    # Create reward functions
    reward_fns = create_reward_functions(custom_config.reward)
    logger.info(
        f"Created {len(reward_fns)} reward functions with weights: "
        f"accuracy={custom_config.reward.accuracy_weight}, "
        f"format={custom_config.reward.format_weight}"
    )

    # Setup TAO logging if enabled via logging.logger config and TAO_API_JOB_ID is set
    custom_logger_fns = []
    hook_fns = {}

    # Check if TAO logging is enabled via logging.logger config
    # Expected format: logging.logger = ["console", "tao"]
    loggers = config.logging.logger if hasattr(config.logging, "logger") else []
    tao_logging_enabled = (
        "tao" in loggers if isinstance(loggers, list) else loggers == "tao"
    )

    if tao_logging_enabled and os.environ.get("TAO_API_JOB_ID"):
        logger.info(
            "TAO logging enabled via logging.logger config - will write to status.json"
        )

        tao_logger = TAOStatusLogger(
            experiment_name=config.logging.experiment_name or "Cosmos-RL GRPO Training",
            # For GRPO, monitor validation reward instead of loss
            monitor_metric="val/reward_avg",
        )

        custom_logger_fns.append(tao_logger.log_status)
        hook_fns = tao_logger.get_hooks()

        logger.info(
            f"TAO status will be logged to: {tao_logger._get_status_file_path()}"
        )
        logger.info("TAO monitoring metric: val/reward_avg (validation reward average)")
    elif tao_logging_enabled:
        logger.info(
            "TAO logging enabled but TAO_API_JOB_ID not set - "
            "skipping TAO status logging"
        )

    # Launch worker with factory functions, reward functions, data packers, and TAO logging
    val_dataset_factory = get_val_dataset if custom_config.val_dataset else None

    cosmos_rl.launcher.worker_entry.main(
        dataset=get_train_dataset,
        val_dataset=val_dataset_factory,
        reward_fns=reward_fns,
        val_reward_fns=reward_fns,  # Use same reward fns for validation
        data_packer=HFVLMDataPacker(),
        val_data_packer=HFVLMDataPacker() if custom_config.val_dataset else None,
        custom_logger_fns=custom_logger_fns,
        hook_fns=hook_fns,
    )


if __name__ == "__main__":
    main()
