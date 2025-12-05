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

"""SFT adapter for llava-format datasets with support for separate training and validation datasets."""

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

from cosmos_reason1_utils.text import create_conversation
from cosmos_reason1_utils.vision import VisionConfig


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

        conversations = create_conversation(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            response=response,
            images=images,
            videos=videos,
            vision_kwargs=self.vision_kwargs,
        )
        return conversations


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
    custom_config = CustomConfig.model_validate(config_kwargs["custom"])

    # Log
    role = os.environ.get("COSMOS_ROLE")
    is_controller = role == "Controller"
    if is_controller:
        output_dir = Path(config.train.output_dir).resolve().parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_kwargs = config.model_dump()
        config_kwargs["custom"] = custom_config.model_dump()
        config_path = output_dir / "config.toml"
        config_path.write_text(toml.dumps(config_kwargs))
        logger.info(f"Saved config to {config_path}")

    # Factory function for training dataset
    def get_train_dataset(config: cosmos_rl.policy.config.Config) -> torch.utils.data.Dataset:
        """Factory function to create training dataset."""
        custom_config = CustomConfig.model_validate(config.model_dump().get("custom", {}))

        logger.info(f"Creating training dataset from: {custom_config.train_dataset.annotation_path}")
        return CustomDataset(
            config=config,
            custom_config=custom_config,
            annotation_path=custom_config.train_dataset.annotation_path,
            media_path=custom_config.train_dataset.media_path,
        )

    # Factory function for validation dataset (optional)
    def get_val_dataset(config: cosmos_rl.policy.config.Config) -> torch.utils.data.Dataset:
        """Factory function to create validation dataset."""
        custom_config = CustomConfig.model_validate(config.model_dump().get("custom", {}))

        # Only create validation dataset if validation dataset config is specified
        if not custom_config.val_dataset:
            logger.info("No validation dataset specified, skipping validation dataset")
            return None

        logger.info(f"Creating validation dataset from: {custom_config.val_dataset.annotation_path}")
        return CustomDataset(
            config=config,
            custom_config=custom_config,
            annotation_path=custom_config.val_dataset.annotation_path,
            media_path=custom_config.val_dataset.media_path,
        )

    # Launch worker with factory functions
    val_dataset_factory = get_val_dataset if custom_config.val_dataset else None

    cosmos_rl.launcher.worker_entry.main(
        dataset=get_train_dataset,
        val_dataset=val_dataset_factory,
    )


if __name__ == "__main__":
    main()
