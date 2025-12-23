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

import os
import math
import numpy as np
import libero.libero.benchmark as benchmark
from typing import Dict

from cosmos_rl.utils.logging import logger


def get_libero_dummy_action(num_envs: int) -> list:
    dummy_actions = np.zeros((num_envs, 7))
    dummy_actions[:, -1] = -1.0
    return dummy_actions


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def get_benchmark_overridden(benchmark_name) -> benchmark.Benchmark:
    """
    Return the Benchmark class for a given name.
    For "libero_all": return a dynamically aggregated class from all suites.
    For others: delegate to the original LIBERO get_benchmark.

    Args:
        benchmark_name: Name of the benchmark to get

    Returns:
        Benchmark class
    """
    name = str(benchmark_name).lower()
    if name != "libero_all":
        return benchmark.get_benchmark(benchmark_name)

    libreo_cls = benchmark.BENCHMARK_MAPPING.get("libero_all", None)
    if libreo_cls is not None:
        return libreo_cls

    # Build aggregated task map once, preserving order and de-duplicating by task name
    aggregated_task_map: dict[str, benchmark.Task] = {}
    for suite_name in getattr(benchmark, "libero_suites", []):
        suite_map = benchmark.task_maps.get(suite_name, {})
        for task_name, task in suite_map.items():
            if task_name not in aggregated_task_map:
                aggregated_task_map[task_name] = task

    class LIBERO_ALL(benchmark.Benchmark):
        def __init__(self, task_order_index=0):
            super().__init__(task_order_index=task_order_index)
            self.name = "libero_all"
            self._make_benchmark()

        def _make_benchmark(self):
            tasks = list(aggregated_task_map.values())
            self.tasks = tasks
            self.n_tasks = len(self.tasks)

    # Register for discoverability/help
    benchmark.BENCHMARK_MAPPING["libero_all"] = LIBERO_ALL
    return LIBERO_ALL


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    Normalize gripper action from [0,1] to [-1,+1] range.

    This is necessary for some environments because the dataset wrapper
    standardizes gripper actions to [0,1]. Note that unlike the other action
    dimensions, the gripper action is not normalized to [-1,+1] by default.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1

    Args:
        action: Action array with gripper action in the last dimension
        binarize: Whether to binarize gripper action to -1 or +1

    Returns:
        np.ndarray: Action array with normalized gripper action
    """
    # Create a copy to avoid modifying the original
    normalized_action = action.copy()

    # Normalize the last action dimension to [-1,+1]
    orig_low, orig_high = 0.0, 1.0
    normalized_action[..., -1] = (
        2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1
    )

    if binarize:
        # Binarize to -1 or +1
        normalized_action[..., -1] = np.sign(normalized_action[..., -1])

    return normalized_action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """
    Flip the sign of the gripper action (last dimension of action vector).

    This is necessary for environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.

    Args:
        action: Action array with gripper action in the last dimension

    Returns:
        np.ndarray: Action array with inverted gripper action
    """
    # Create a copy to avoid modifying the original
    inverted_action = action.copy()

    # Invert the gripper action
    inverted_action[..., -1] = inverted_action[..., -1] * -1.0

    return inverted_action


def save_rollout_video(
    rollout_images, rollout_dir: str, task_name: str, step_idx: int, success: bool
) -> str:
    """
    Saves an MP4 replay of an episode.

    Args:
        rollout_images: List of images (numpy arrays) to save as video
        exp_name: Experiment name for organizing videos
        task_name: Task identifier
        step_idx: Current training step index
        success: Whether the episode was successful

    Returns:
        str: Path to the saved video file
    """
    import random

    try:
        import imageio
    except ImportError:
        logger.warning(
            "imageio not installed, cannot save rollout videos. Install with: pip install imageio imageio-ffmpeg"
        )
        return ""

    # Create rollout directory
    os.makedirs(rollout_dir, exist_ok=True)

    # Generate unique filename
    ran_id = random.randint(1, 10000)
    mp4_path = f"{rollout_dir}/step={step_idx}--task={task_name}--success={success}--ran={ran_id}.mp4"

    # Write video
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()

    logger.debug(f"✅ Saved {len(rollout_images)} frames to: {mp4_path}")
    return mp4_path


def obs_to_vla_input(obs: Dict, is_robotwin: bool = False) -> Dict:
    """
    Convert environment observation to VLA model input format

    Args:
        obs: Raw observation dict from environment
        is_robotwin: Whether this is RoboTwin format (default: False for LIBERO)

    Returns:
        Dict with 'full_image' (and 'state' for RoboTwin)
    """

    def resize_image(img: np.ndarray, resolution: int = 224) -> np.ndarray:
        if img.shape[0] != resolution or img.shape[1] != resolution:
            from PIL import Image

            pil_img = Image.fromarray(img.astype(np.uint8))
            pil_img = pil_img.resize((resolution, resolution), Image.Resampling.LANCZOS)
            img = np.array(pil_img, dtype=np.uint8)
        return img

    return resize_image(obs, 224)
