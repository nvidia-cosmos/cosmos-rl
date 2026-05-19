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

"""TAO-compatible SFT example with custom logger and hooks.

This script demonstrates how to use custom_logger_fns and hook_fns for TAO-compatible
status logging. The TAOStatusLogger writes to status.json in the format expected by TAO/NVAIE.

For a general-purpose llava-format SFT script without TAO logging, see:
    scripts/llava_sft.py

Usage:
    cosmos-rl --config spec.toml cosmos_rl/tools/custom_example/tao_sft_example.py

Environment Variables for TAO logging:
    TAO_API_JOB_ID: Job ID for status file path
    TAO_API_RESULTS_DIR: Results directory (defaults to /results)

The status file is written to: {TAO_API_RESULTS_DIR}/{TAO_API_JOB_ID}/status.json
"""

import argparse
import json
import os
import re
from pathlib import Path

import cosmos_rl.launcher.worker_entry
import cosmos_rl.policy.config
import pydantic
import toml
import torch.utils.data
from cosmos_rl.utils.logging import logger

# Import TAO status logger utilities
from cosmos_rl.tools.custom_hooks import TAOStatusLogger

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
    annotation_path: str = pydantic.Field()
    """Dataset annotation path."""
    media_path: str = pydantic.Field(default="")
    """Dataset media path."""


class CustomConfig(pydantic.BaseModel):
    train_dataset: CustomDatasetConfig = pydantic.Field()
    """Training dataset config."""

    val_dataset: CustomDatasetConfig = pydantic.Field(default=None)
    """Validation dataset config (optional)."""

    system_prompt: str = pydantic.Field(default="")
    """System prompt."""

    vision: VisionConfig = pydantic.Field(
        default=VisionConfig(
            fps=1,
            max_pixels=81920,
        )
    )
    """Vision processor config."""


class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config: cosmos_rl.policy.config.Config,
        custom_config: CustomConfig,
        annotation_path: str,
        media_path: str,
    ):
        self.annotation = json.load(open(annotation_path))
        self.media_path = media_path
        self.system_prompt = custom_config.system_prompt
        self.config = config
        self.custom_config = custom_config
        self.vision_kwargs = custom_config.vision.model_dump(exclude_none=True)

    def setup(self, config, tokenizer):
        """Setup method required by the SFT trainer."""
        # This method is called by the trainer to initialize the dataset
        # For our custom dataset, we don't need additional setup beyond __init__
        pass

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx: int) -> list[dict]:
        sample = self.annotation[idx]

        user_prompt = sample["conversations"][0]["value"]
        response = sample["conversations"][1]["value"]
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
                response=response,
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

            conversations.append({"role": "assistant", "content": response})

        return conversations


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


def monitor_status(experiment_name: str = "Cosmos-RL finetuning"):
    """Decorator to monitor job status (STARTED/SUCCESS/FAILURE).

    Only logs status from master rank to minimize memory overhead.
    This decorator handles TAO status logging without interfering with
    the main function's logic.

    Usage:
        @monitor_status("My Experiment")
        def main():
            # your code here
            pass
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


@monitor_status("Cosmos-RL SFT")
def main():
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

    # Factory function for training dataset
    def get_train_dataset(
        config: cosmos_rl.policy.config.Config,
    ) -> torch.utils.data.Dataset:
        """Factory function to create training dataset."""
        custom_cfg = CustomConfig.model_validate(config.model_dump().get("custom", {}))

        logger.info(
            f"Creating training dataset from: {custom_cfg.train_dataset.annotation_path}"
        )
        return CustomDataset(
            config=config,
            custom_config=custom_cfg,
            annotation_path=custom_cfg.train_dataset.annotation_path,
            media_path=custom_cfg.train_dataset.media_path,
        )

    # Factory function for validation dataset (optional)
    def get_val_dataset(
        config: cosmos_rl.policy.config.Config,
    ) -> torch.utils.data.Dataset:
        """Factory function to create validation dataset."""
        custom_cfg = CustomConfig.model_validate(config.model_dump().get("custom", {}))

        if not custom_cfg.val_dataset:
            logger.info("No validation dataset specified, skipping validation dataset")
            return None

        logger.info(
            f"Creating validation dataset from: {custom_cfg.val_dataset.annotation_path}"
        )
        return CustomDataset(
            config=config,
            custom_config=custom_cfg,
            annotation_path=custom_cfg.val_dataset.annotation_path,
            media_path=custom_cfg.val_dataset.media_path,
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
            experiment_name=config.logging.experiment_name or "Cosmos-RL SFT Training"
        )

        custom_logger_fns.append(tao_logger.log_status)
        hook_fns = tao_logger.get_hooks()

        logger.info(
            f"TAO status will be logged to: {tao_logger._get_status_file_path()}"
        )
    elif tao_logging_enabled:
        logger.info(
            "TAO logging enabled but TAO_API_JOB_ID not set - skipping TAO status logging"
        )

    # Launch worker with factory functions and TAO logging
    val_dataset_factory = get_val_dataset if custom_config.val_dataset else None

    cosmos_rl.launcher.worker_entry.main(
        dataset=get_train_dataset,
        val_dataset=val_dataset_factory,
        custom_logger_fns=custom_logger_fns,
        hook_fns=hook_fns,
    )


if __name__ == "__main__":
    main()
