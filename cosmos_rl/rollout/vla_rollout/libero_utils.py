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
LIBERO utility functions for environment interaction and action processing.

Reimplemented from verl/utils/libero_utils.py to remove external dependencies.
"""

import os
import numpy as np
from typing import Dict
from cosmos_rl.utils.logging import logger

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def get_libero_env(task, model_family: str, resolution: int = 256):
    """
    Initialize and return the LIBERO environment, along with the task description.
    
    Args:
        task: LIBERO task object with language, problem_folder, and bddl_file attributes
        model_family: Model family name (e.g., 'openvla')
        resolution: Image resolution for camera
    
    Returns:
        Tuple[env, task_description]: Environment and task description string
    """

    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed affects object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str) -> list:
    """
    Get dummy/no-op action, used to roll out the simulation while the robot does nothing.
    
    Args:
        model_family: Model family name (e.g., 'openvla')
    
    Returns:
        list: Dummy action [x, y, z, rx, ry, rz, gripper]
    """
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


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
    normalized_action[..., -1] = 2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1
    
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
    rollout_images,
    exp_name: str,
    task_name: str,
    step_idx: int,
    success: bool
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
        logger.warning("imageio not installed, cannot save rollout videos. Install with: pip install imageio imageio-ffmpeg")
        return ""
    
    # Create rollout directory
    rollout_dir = f"./rollouts/{exp_name}"
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
    if is_robotwin:
        # RoboTwin format
        return {
            'full_image': obs.get('agentview_image', obs.get('image', np.zeros((256, 256, 3)))),
            'state': obs.get('robot0_proprio', obs.get('state', np.zeros(32)))
        }
    else:
        # LIBERO format
        # IMPORTANT: Resize to 224x224 to match SimpleVLA-RL's get_libero_image()
        img = obs.get('agentview_image', np.zeros((256, 256, 3)))
        
        # Resize using TensorFlow (EXACT match to SimpleVLA-RL's resize_image function)
        if img.shape[0] != 224 or img.shape[1] != 224:
            import tensorflow as tf
            # Exactly match SimpleVLA-RL's resize_image implementation:
            # - Encode as JPEG (as done in RLDS dataset builder)
            # - Decode back
            # - Resize with lanczos3 and antialias
            # - Clip and round
            img = tf.image.encode_jpeg(img)
            img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
            img = tf.image.resize(img, (224, 224), method="lanczos3", antialias=True)
            img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
            img = img.numpy()
        
        return {
            'full_image': img
        }

