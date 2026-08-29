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
    TAO_API_RESULTS_DIR: User-supplied results directory
    TAO_STATUS_FILE: Explicit status file path (preferred for direct launches)

The status file is written to: {TAO_API_RESULTS_DIR}/{TAO_API_JOB_ID}/status.json
"""

import argparse
import json
import os
import re
from pathlib import Path
from collections import OrderedDict
from typing import Iterator, Literal, Optional

import cosmos_rl.launcher.worker_entry
import cosmos_rl.policy.config
import pydantic
import toml
import torch.utils.data
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.system_pyav_video_reader import register_system_pyav_video_reader

if os.environ.get("FORCE_QWENVL_VIDEO_READER") != "pynvvideocodec":
    register_system_pyav_video_reader()

# Import TAO status logger utilities
from cosmos_rl.tools.custom_hooks import TAOStatusLogger
from cosmos_rl.tools.custom_hooks.lifecycle_status import (
    append_terminal_status,
    is_lifecycle_status_owner,
)

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

    val_dataset: Optional[CustomDatasetConfig] = pydantic.Field(default=None)
    """Validation dataset config (optional)."""

    system_prompt: str = pydantic.Field(default="")
    """System prompt."""

    video_decoder: pydantic.StrictStr = "pynvvideocodec"
    # The validated hardware-video throughput profile avoids retaining
    # processed frame tensors while keeping a small native decoder-session LRU.
    # Both caches populate only on demand during ordinary training.
    video_cache_size: int = pydantic.Field(default=0, ge=0)
    video_decoder_cache_size: int = pydantic.Field(default=4, ge=1)
    video_override_map: str | None = None

    validation_shard_strategy: Literal["stride", "media_grouped"] = "stride"
    """Validation sharding policy.

    ``media_grouped`` preserves DistributedSampler's exact padded multiset and
    per-rank sample count, but orders records by media before splitting the
    stream into equal contiguous rank slices.  This keeps repeated questions
    for a video on as few ranks as possible so the on-demand video cache can be
    effective without changing validation coverage or loss weighting.
    """

    validation_cache_frontload_batch_size: int = pydantic.Field(default=0, ge=0)
    validation_cache_frontload_unique_per_batch: int = pydantic.Field(
        default=0, ge=0
    )
    """Optional staged population of the validation video-feature cache.

    When both values are positive, each early validation batch introduces at
    most ``validation_cache_frontload_unique_per_batch`` unseen rank-local
    media groups.  Remaining slots use already introduced groups.  This keeps
    decoder work pipelineable with model forward while preserving the exact
    validation index multiset and loss weighting.
    """

    vision: VisionConfig = pydantic.Field(
        default=VisionConfig(
            fps=1,
            max_pixels=81920,
        )
    )
    """Vision processor config."""


def configure_video_decoder(custom_config: CustomConfig) -> dict[str, object]:
    """Activate the decoder selected by the TAO video-conversation contract."""
    if (
        custom_config.video_decoder == "pynvvideocodec"
        and os.environ.get("COSMOS_ROLE") != "Controller"
    ):
        os.environ.pop("COSMOS_DATALOADER_VIDEO_DECODER", None)
        from cosmos_rl.utils.pynv_video_reader import register_pynv_video_reader

        return register_pynv_video_reader(
            cache_size=custom_config.video_cache_size,
            decoder_cache_size=custom_config.video_decoder_cache_size,
            video_override_map=custom_config.video_override_map,
        )
    if custom_config.video_decoder == "torchvision":
        if os.environ.get("FORCE_QWENVL_VIDEO_READER") != "torchvision":
            raise RuntimeError(
                "custom.video_decoder=torchvision requires "
                "FORCE_QWENVL_VIDEO_READER=torchvision"
            )
        os.environ["COSMOS_DATALOADER_VIDEO_DECODER"] = "system_pyav"
        register_system_pyav_video_reader()
        return {"backend": "torchvision", "implementation": "system_pyav_sparse"}
    if (
        custom_config.video_decoder == "cpu"
        or os.environ.get("COSMOS_ROLE") == "Controller"
    ):
        return {"backend": custom_config.video_decoder}
    raise ValueError("custom.video_decoder must be pynvvideocodec, torchvision, or cpu")


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

    def media_key(self, idx: int) -> tuple[str, ...]:
        """Return the resolved media identity used by validation sharding."""
        sample = self.annotation[idx]
        paths: list[str] = []
        for singular, plural in (("video", "videos"), ("image", "images")):
            values = sample.get(singular, None) or sample.get(plural, None)
            if not values:
                continue
            if isinstance(values, str):
                values = [values]
            for value in values:
                path = (
                    os.path.join(self.media_path, value) if self.media_path else value
                )
                paths.append(os.path.normpath(path))
        # Text-only records must remain independent instead of collapsing into
        # one artificial group.
        return tuple(paths) if paths else (f"__sample__:{idx}",)

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


class MediaGroupedDistributedSampler(torch.utils.data.Sampler[int]):
    """Equal-length deterministic validation shards with media locality.

    The initial index multiset is deliberately identical to PyTorch's
    ``DistributedSampler(shuffle=False, drop_last=False)``: records are padded
    with the leading indices to ``ceil(N / replicas) * replicas``.  The only
    change is ordering.  Records are grouped by media in first-seen order and
    the grouped stream is cut into equal contiguous rank slices.  At most one
    media group is split at each rank boundary.  Inside each rank, one record
    from every assigned media group is front-loaded before the remaining
    records.  This populates an on-demand feature cache in the fewest batches,
    allowing later validation batches to skip the vision encoder collectively.
    """

    cache_frontload_batch_size = 0
    cache_frontload_unique_per_batch = 0

    def __init__(
        self,
        dataset,
        num_replicas: int,
        rank: int,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> None:
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank} is outside [0, {num_replicas})")
        if shuffle:
            raise ValueError("media-grouped validation does not support shuffle=True")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        dataset_size = len(dataset)
        if drop_last and dataset_size % num_replicas:
            self.num_samples = dataset_size // num_replicas
        else:
            self.num_samples = (dataset_size + num_replicas - 1) // num_replicas
        self.total_size = self.num_samples * num_replicas
        self._indices, stats = self._build_indices()
        logger.info(
            "Media-grouped validation shard: rank=%s/%s samples=%s "
            "logical_records=%s padded_records=%s media_groups=%s "
            "rank_media_groups=%s cache_frontloaded_records=%s",
            self.rank,
            self.num_replicas,
            len(self._indices),
            len(self.dataset),
            self.total_size - len(self.dataset),
            stats["media_groups"],
            stats["rank_media_groups"],
            stats["cache_frontloaded_records"],
        )

    def _media_dataset(self):
        current = self.dataset
        seen: set[int] = set()
        while not hasattr(current, "media_key"):
            identity = id(current)
            if identity in seen or not hasattr(current, "dataset"):
                raise TypeError(
                    "media_grouped validation requires a dataset exposing media_key(index)"
                )
            seen.add(identity)
            current = current.dataset
        return current

    def _build_indices(self) -> tuple[list[int], dict[str, int]]:
        indices = list(range(len(self.dataset)))
        if self.drop_last:
            indices = indices[: self.total_size]
        else:
            padding = self.total_size - len(indices)
            if padding > 0:
                if padding <= len(indices):
                    indices += indices[:padding]
                else:
                    repeats = (padding + len(indices) - 1) // len(indices)
                    indices += (indices * repeats)[:padding]

        media_dataset = self._media_dataset()
        groups: OrderedDict[tuple[str, ...], list[int]] = OrderedDict()
        for index in indices:
            key = tuple(media_dataset.media_key(index))
            groups.setdefault(key, []).append(index)
        ordered = [index for group in groups.values() for index in group]
        if len(ordered) != self.total_size:
            raise RuntimeError(
                f"media-grouped sampler produced {len(ordered)} indices; "
                f"expected {self.total_size}"
            )
        start = self.rank * self.num_samples
        rank_indices = ordered[start : start + self.num_samples]
        rank_groups: OrderedDict[tuple[str, ...], list[int]] = OrderedDict()
        for index in rank_indices:
            key = tuple(media_dataset.media_key(index))
            rank_groups.setdefault(key, []).append(index)
        cache_frontloaded = [group[0] for group in rank_groups.values()]
        cache_remainder = [index for group in rank_groups.values() for index in group[1:]]
        frontload_batch_size = int(self.cache_frontload_batch_size)
        frontload_unique_per_batch = int(self.cache_frontload_unique_per_batch)
        if frontload_batch_size or frontload_unique_per_batch:
            if (
                frontload_batch_size <= 0
                or frontload_unique_per_batch <= 0
                or frontload_unique_per_batch > frontload_batch_size
            ):
                raise ValueError(
                    "staged validation cache frontloading requires positive "
                    "batch and unique counts with unique <= batch"
                )
            rank_indices = self._staged_cache_frontload(
                rank_groups,
                frontload_batch_size,
                frontload_unique_per_batch,
            )
        else:
            rank_indices = cache_frontloaded + cache_remainder
        rank_media_groups = len(rank_groups)
        if len(rank_indices) != self.num_samples:
            raise RuntimeError(
                f"media-grouped rank sampler produced {len(rank_indices)} indices; "
                f"expected {self.num_samples}"
            )
        return rank_indices, {
            "media_groups": len(groups),
            "rank_media_groups": rank_media_groups,
            "cache_frontloaded_records": len(cache_frontloaded),
        }

    @staticmethod
    def _staged_cache_frontload(
        rank_groups: OrderedDict[tuple[str, ...], list[int]],
        batch_size: int,
        unique_per_batch: int,
    ) -> list[int]:
        """Introduce bounded unseen media per early batch, deterministically."""
        remaining = OrderedDict((key, list(group)) for key, group in rank_groups.items())
        group_keys = list(remaining)
        active_keys: list[tuple[str, ...]] = []
        ordered: list[int] = []

        for start in range(0, len(group_keys), unique_per_batch):
            new_keys = group_keys[start : start + unique_per_batch]
            active_keys.extend(new_keys)
            batch = []
            for key in new_keys:
                batch.append(remaining[key].pop(0))

            while len(batch) < batch_size:
                made_progress = False
                for key in active_keys:
                    if remaining[key]:
                        batch.append(remaining[key].pop(0))
                        made_progress = True
                        if len(batch) == batch_size:
                            break
                if not made_progress:
                    break
            ordered.extend(batch)

        for group in remaining.values():
            ordered.extend(group)
        return ordered

    def __iter__(self) -> Iterator[int]:
        return iter(self._indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        # Validation ordering is intentionally invariant across epochs.
        del epoch


def _get_results_dir() -> str | None:
    """Get the results directory based on TAO environment variables."""
    job_id = os.environ.get("TAO_API_JOB_ID")
    results_base = os.environ.get("TAO_API_RESULTS_DIR")
    if job_id and results_base:
        return os.path.join(results_base, job_id)
    return None


def _is_master_rank() -> bool:
    """Check if this entrypoint process owns terminal lifecycle logging."""
    return is_lifecycle_status_owner()


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
            status_file = None

            # Only setup logger on master rank
            if HAS_TAO_CORE and _is_master_rank() and _get_results_dir() is not None:
                results_dir = _get_results_dir()
                assert results_dir is not None
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
                if status_file:
                    append_terminal_status(
                        status_file,
                        "SUCCESS",
                        f"{experiment_name} training completed successfully",
                    )
                    logger.info(f"Job SUCCESS: {experiment_name}")

                return result

            except (KeyboardInterrupt, SystemExit) as e:
                if status_file:
                    try:
                        append_terminal_status(
                            status_file,
                            "FAILURE",
                            f"{experiment_name} training was interrupted: {str(e)}",
                        )
                    except Exception:
                        pass
                    logger.warning(f"Job INTERRUPTED: {experiment_name}")
                raise

            except Exception as e:
                if status_file:
                    try:
                        append_terminal_status(
                            status_file,
                            "FAILURE",
                            f"{experiment_name} training failed: {str(e)}",
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
    decoder_info = configure_video_decoder(custom_config)
    logger.info("Video decoder configured: %s", decoder_info)

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

    # Release rank-local validation features before checkpointing or the next
    # training epoch.  Compose with the TAO status hook instead of replacing
    # its validation-complete event.
    existing_post_validation_hook = hook_fns.get("post_validation_hook")

    def post_validation_and_clear_feature_cache(worker, report_data):
        try:
            if existing_post_validation_hook is not None:
                existing_post_validation_hook(worker, report_data=report_data)
        finally:
            clear_cache = getattr(
                worker.trainer.forward_model,
                "clear_validation_video_feature_cache",
                None,
            )
            if callable(clear_cache):
                clear_cache()

    hook_fns["post_validation_hook"] = post_validation_and_clear_feature_cache

    # Launch worker with factory functions and TAO logging
    if custom_config.val_dataset:
        val_dataset_factory = get_val_dataset
        logger.info(
            "Using custom validation dataset from "
            f"{custom_config.val_dataset.annotation_path}"
        )
    else:
        val_dataset_factory = None
        if config.validation.enable:
            logger.info(
                "No custom validation dataset specified. Cosmos-RL will use "
                "validation.dataset if configured, otherwise split the training "
                "dataset for validation."
            )
        else:
            logger.info("Validation is disabled; no validation dataset is required.")

    if custom_config.validation_shard_strategy == "media_grouped":
        MediaGroupedDistributedSampler.cache_frontload_batch_size = (
            custom_config.validation_cache_frontload_batch_size
        )
        MediaGroupedDistributedSampler.cache_frontload_unique_per_batch = (
            custom_config.validation_cache_frontload_unique_per_batch
        )

    cosmos_rl.launcher.worker_entry.main(
        dataset=get_train_dataset,
        val_dataset=val_dataset_factory,
        val_sampler=(
            MediaGroupedDistributedSampler
            if custom_config.validation_shard_strategy == "media_grouped"
            else None
        ),
        custom_logger_fns=custom_logger_fns,
        hook_fns=hook_fns,
    )


if __name__ == "__main__":
    main()
