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

import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from typing import Dict


def center_crop_image(image: Image.Image, crop_size: int = 256) -> Image.Image:
    """
    Center crop image with 0.9 scale then resize (matching SimpleVLA-RL)

    This function mimics SimpleVLA-RL's TensorFlow-based center crop:
    - Crops to 90% of the center (zoom in effect)
    - Resizes back to 224x224

    Replaced TensorFlow with torchvision for better compatibility.
    """

    crop_scale = 0.9  # Match SimpleVLA-RL

    # Get original image dimensions
    width, height = image.size

    # Calculate crop dimensions (sqrt of scale to match TF implementation)
    crop_ratio = np.sqrt(crop_scale)  # ~0.9487
    crop_height = int(height * crop_ratio)
    crop_width = int(width * crop_ratio)

    # Calculate offsets for center crop
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2

    # Perform center crop
    cropped_image = TF.crop(image, top, left, crop_height, crop_width)

    # Resize to 224x224 (matching SimpleVLA-RL)
    result_image = TF.resize(
        cropped_image, [224, 224], interpolation=TF.InterpolationMode.BILINEAR
    )

    # Ensure RGB format
    result_image = result_image.convert("RGB")

    return result_image


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
