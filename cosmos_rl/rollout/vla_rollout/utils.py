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
Utility functions for VLA rollout operations in cosmos-rl
"""

import os
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import torch

from cosmos_rl.utils.logging import logger


def encode_observation(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Encode environment observation for VLA model processing
    
    Args:
        obs: Raw observation from environment
        
    Returns:
        Encoded observation suitable for VLA model
    """
    encoded_obs = {}
    
    # Handle image observations
    if 'agentview_image' in obs:
        encoded_obs['image'] = _encode_image(obs['agentview_image'])
    
    if 'wrist_image' in obs:
        encoded_obs['wrist_image'] = _encode_image(obs['wrist_image'])
    
    # Handle proprioception
    if 'robot0_proprio' in obs:
        encoded_obs['proprioception'] = obs['robot0_proprio']
    
    # Handle other modalities
    for key, value in obs.items():
        if key not in ['agentview_image', 'wrist_image', 'robot0_proprio']:
            encoded_obs[key] = value
    
    return encoded_obs


def _encode_image(image: np.ndarray) -> np.ndarray:
    """Encode single image observation"""
    if isinstance(image, np.ndarray):
        # Ensure proper format for VLA processing
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Handle different image orientations/formats
        if len(image.shape) == 3 and image.shape[2] == 3:
            # RGB image, might need to flip for some environments
            image = image[::-1, ::-1]  # Flip both axes as done in SimpleVLA-RL
        
        return image
    
    return image


def create_vla_prompt(instruction: str, observation: Dict[str, Any], 
                     prompt_template: Optional[str] = None) -> str:
    """
    Create VLA prompt from instruction and observation
    
    Args:
        instruction: Task instruction
        observation: Current observation
        prompt_template: Optional custom prompt template
        
    Returns:
        Formatted VLA prompt
    """
    if prompt_template is None:
        # Default VLA prompt template
        prompt_template = "Instruction: {instruction}\n\nWhat action should the robot take next?"
    
    return prompt_template.format(instruction=instruction)


def parse_vla_response(response: str) -> Dict[str, Any]:
    """
    Parse VLA model response to extract actions and reasoning
    
    Args:
        response: Raw response from VLA model
        
    Returns:
        Parsed response with actions and metadata
    """
    parsed = {
        'raw_response': response,
        'actions': None,
        'reasoning': None
    }
    
    # TODO: Implement proper VLA response parsing
    # This depends on the specific VLA model format
    
    return parsed


def save_vla_episode(episode_data: Dict[str, Any], save_dir: str, 
                    episode_id: str) -> None:
    """
    Save VLA episode data for analysis and debugging
    
    Args:
        episode_data: Episode trajectory and metadata
        save_dir: Directory to save episode data
        episode_id: Unique episode identifier
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save episode metadata
    metadata_path = os.path.join(save_dir, f"{episode_id}_metadata.json")
    # TODO: Implement episode saving
    
    logger.info(f"Saved VLA episode data to {save_dir}")


def compute_vla_reward(episode_data: Dict[str, Any], reward_type: str = 'binary') -> float:
    """
    Compute reward for VLA episode
    
    Args:
        episode_data: Episode data containing success/failure information
        reward_type: Type of reward computation ('binary', 'sparse', 'dense')
        
    Returns:
        Computed reward value
    """
    if reward_type == 'binary':
        # Simple binary reward based on task success
        return 1.0 if episode_data.get('success', False) else 0.0
    
    elif reward_type == 'sparse':
        # Sparse reward with potential intermediate rewards
        reward = 0.0
        if episode_data.get('success', False):
            reward = 1.0
        # Add small negative reward for time steps to encourage efficiency
        reward -= 0.001 * episode_data.get('episode_length', 0)
        return max(reward, 0.0)
    
    elif reward_type == 'dense':
        # Dense reward based on progress and success
        # TODO: Implement dense reward computation
        return compute_vla_reward(episode_data, 'binary')
    
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")


def validate_vla_config(config: Dict[str, Any]) -> bool:
    """
    Validate VLA configuration parameters
    
    Args:
        config: VLA configuration dictionary
        
    Returns:
        True if configuration is valid
    """
    required_fields = ['task_suite', 'num_parallel_envs', 'action_dim']
    
    for field in required_fields:
        if field not in config:
            logger.error(f"Missing required VLA config field: {field}")
            return False
    
    # Validate task suite
    supported_suites = [
        'libero_10', 'libero_90', 'libero_spatial', 'libero_object', 'libero_goal',
        'robotwin2_*'  # RoboTwin 2.0 tasks
    ]
    
    task_suite = config['task_suite']
    if not any(suite in task_suite for suite in ['libero', 'robotwin']):
        logger.error(f"Unsupported task suite: {task_suite}")
        return False
    
    # Validate numeric parameters
    if config['num_parallel_envs'] <= 0:
        logger.error("num_parallel_envs must be positive")
        return False
    
    if config['action_dim'] <= 0:
        logger.error("action_dim must be positive")
        return False
    
    logger.info("VLA configuration validation passed")
    return True


def get_available_vla_tasks() -> Dict[str, List[str]]:
    """
    Get list of available VLA tasks for different benchmarks
    
    Returns:
        Dictionary mapping benchmark names to available tasks
    """
    tasks = {
        'libero': [
            'libero_10',
            'libero_90', 
            'libero_spatial',
            'libero_object',
            'libero_goal'
        ],
        'robotwin': [
            'robotwin2_click_bell',
            'robotwin2_move_can_pot',
            'robotwin2_place_phone_stand',
            'robotwin2_place_a2b_left',
            'robotwin2_place_a2b_right',
            'robotwin2_handover_mic',
            'robotwin2_pick_dual_bottles',
            'robotwin2_lift_pot',
            'robotwin2_put_bottles_dustbin',
            'robotwin2_stack_blocks_two',
            'robotwin2_stack_bowls_two',
            'robotwin2_handover_block',
            'robotwin2_place_empty_cup',
            'robotwin2_shake_bottle',
            'robotwin2_move_stapler_pad',
            'robotwin2_place_container_plate',
            'robotwin2_blocks_ranking_rgb',
            'robotwin2_beat_block_hammer',
            'robotwin2_place_mouse_pad',
            'robotwin2_place_shoe',
            'robotwin2_move_pillbottle_pad'
        ]
    }
    
    return tasks


def create_default_vla_config(task_suite: str = 'libero_10') -> Dict[str, Any]:
    """
    Create default VLA configuration for a given task suite
    
    Args:
        task_suite: Target task suite
        
    Returns:
        Default VLA configuration
    """
    config = {
        'task_suite': task_suite,
        'num_parallel_envs': 4,
        'action_dim': 7,
        'max_episode_length': 512,
        'image_size': 256,
        'use_multi_view': False,
        'use_wrist_camera': False,
        'action_normalization': 'minmax',
        'reward_type': 'binary',
        'env_config': {
            'headless': True,
            'resolution': 256,
            'model_family': 'openvla'
        }
    }
    
    # Task-specific adjustments
    if 'robotwin' in task_suite:
        config['robotwin_version'] = '2.0'
        config['instruction_type'] = 'language'
    
    return config
