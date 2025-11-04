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
LIBERO environment integration for VLA training in cosmos-rl.

This module implements the actual LIBERO simulation steps and environment interaction.
"""

import os
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import traceback

from cosmos_rl.utils.logging import logger


def setup_libero_environment(task_suite: str, task_id: int, trial_id: int) -> Tuple[Any, str]:
    """
    Setup LIBERO environment for the specified task
    
    Args:
        task_suite: Name of the task suite (e.g., 'libero_10')
        task_id: ID of the specific task
        trial_id: Trial ID for reproducibility
        
    Returns:
        Tuple of (environment, task_description)
    """
    try:
        # Import LIBERO modules
        from libero.libero import benchmark
        
        logger.info(f"Setting up LIBERO environment: {task_suite}, task {task_id}, trial {trial_id}")
        
        # Get benchmark and task
        benchmark_dict = benchmark.get_benchmark_dict()
        if task_suite not in benchmark_dict:
            raise ValueError(f"Unknown LIBERO task suite: {task_suite}")
        
        task_suite_obj = benchmark_dict[task_suite]()
        task = task_suite_obj.get_task(task_id)
        
        # Get initial states
        initial_states = task_suite_obj.get_task_init_states(task_id)
        initial_state = initial_states[trial_id % len(initial_states)]
        
        # Create environment using OffScreenRenderEnv (matches SimpleVLA-RL)
        env, task_description = create_libero_env_for_vla(task, resolution=256)
        
        # Reset and set initial state
        env.reset()
        obs = env.set_init_state(initial_state)
        
        # Wait for environment to stabilize (as done in SimpleVLA-RL)
        # This is CRITICAL - LIBERO cameras need several steps to initialize properly
        num_wait_steps = 10  # Increased from 10 to ensure proper camera initialization
        
        for step in range(num_wait_steps):
            dummy_action = get_libero_dummy_action()
            obs, _, _, _ = env.step(dummy_action)
            
        # IMPORTANT: Flip LIBERO images (they come upside-down and mirrored)
        # This matches SimpleVLA-RL behavior
        if 'agentview_image' in obs:
            obs['agentview_image'] = obs['agentview_image'][::-1, ::-1]
        if 'robot0_eye_in_hand_image' in obs:
            obs['robot0_eye_in_hand_image'] = obs['robot0_eye_in_hand_image'][::-1, ::-1]
        
        logger.info(f"LIBERO environment setup completed: {task_description}")
        return env, task_description, obs
        
    except ImportError as e:
        logger.error(f"LIBERO not available: {e}")
        raise ImportError("LIBERO is required for LIBERO tasks. Please install LIBERO.")
    except Exception as e:
        logger.error(f"Failed to setup LIBERO environment: {e}")
        traceback.print_exc()
        raise


def create_libero_env_for_vla(task, resolution: int = 256) -> Tuple[Any, str]:
    """
    Create LIBERO environment configured for VLA training
    
    Direct implementation using LIBERO's OffScreenRenderEnv (matches SimpleVLA-RL)
    """
    try:
        # Import LIBERO components
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        
        logger.info(f"Creating LIBERO environment for task: {task.name}")
        
        # Get task description (following SimpleVLA-RL pattern)
        task_description = task.language
        logger.info(f"Task description: {task_description}")
        
        # Create BDDL file path (following SimpleVLA-RL pattern)
        task_bddl_file = os.path.join(
            get_libero_path("bddl_files"), 
            task.problem_folder, 
            task.bddl_file
        )
        logger.info(f"BDDL file: {task_bddl_file}")
        
        # Create environment args (following SimpleVLA-RL pattern)
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": resolution,
            "camera_widths": resolution
        }
        
        # Create OffScreenRenderEnv directly (as in SimpleVLA-RL)
        env = OffScreenRenderEnv(**env_args)
        
        # Set seed (IMPORTANT: as noted in SimpleVLA-RL)
        env.seed(0)  # seed affects object positions even with fixed initial state
        
        logger.info(f"LIBERO OffScreenRenderEnv created successfully: {task.name}")
        return env, task_description
        
    except Exception as e:
        logger.error(f"Failed to create LIBERO environment: {e}")
        raise


def create_libero_env_manual(task, resolution: int = 256) -> Tuple[Any, str]:
    """
    Manual LIBERO environment creation using OffScreenRenderEnv
    
    This is the fallback when the primary method fails
    """
    logger.info(f"Creating LIBERO environment manually for task: {task.name}")
    
    try:
        # Import LIBERO components
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        
        # Get task description (language instruction)
        task_description = task.language
        
        # Create BDDL file path
        task_bddl_file = os.path.join(
            get_libero_path("bddl_files"), 
            task.problem_folder, 
            task.bddl_file
        )
        
        # Basic environment configuration (minimal args like SimpleVLA-RL)
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": resolution,
            "camera_widths": resolution
        }
        
        # Create OffScreenRenderEnv
        env = OffScreenRenderEnv(**env_args)
        env.seed(0)
        
        logger.info(f"LIBERO environment created manually: {task.name} -> '{task_description}'")
        return env, task_description
        
    except Exception as e:
        logger.error(f"Failed to create LIBERO environment manually: {e}")
        raise


def _configure_obs_space_for_vla(env):
    """Configure observation space for VLA training"""
    try:
        # Test observation to ensure proper format
        obs = env._get_observations() if hasattr(env, '_get_observations') else env.reset()
        
        expected_keys = ['agentview_image', 'robot0_proprio']
        missing_keys = [key for key in expected_keys if key not in obs]
        
        if missing_keys:
            logger.warning(f"Missing expected observation keys: {missing_keys}")
            logger.info(f"Available observation keys: {list(obs.keys())}")
        else:
            logger.info("✅ Environment observation space properly configured for VLA")
        
        return env
        
    except Exception as e:
        logger.warning(f"Could not verify observation space configuration: {e}")
        return env


def get_libero_dummy_action() -> List[float]:
    """
    Get dummy action for LIBERO environment stabilization
    
    Returns 7-DOF action: [x, y, z, rx, ry, rz, gripper] for delta position control
    """
    # LIBERO uses 7-DOF delta position control
    # [delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, gripper]
    # IMPORTANT: Use -1.0 for gripper to match SimpleVLA-RL (gripper closed)
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]  # No movement, gripper closed


def process_libero_action(action: np.ndarray) -> np.ndarray:
    """
    Process VLA action for LIBERO environment execution
    
    LIBERO uses 7-DOF delta position control:
    [delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, gripper]
    
    Args:
        action: Raw action from VLA model (normalized to [-1, 1])
        
    Returns:
        Processed action ready for LIBERO environment
    """
    # Ensure action is correct shape
    if action.ndim > 1:
        action = action[0]  # Take first action if batch
    
    processed_action = action.copy()
    
    # Ensure action is right shape (7 dims: x,y,z,rx,ry,rz,gripper)
    if len(processed_action) != 7:
        logger.warning(f"Expected 7D action, got {len(processed_action)}D, adjusting")
        if len(processed_action) < 7:
            # Pad with zeros
            processed_action = np.pad(processed_action, (0, 7 - len(processed_action)))
        else:
            # Truncate to 7 dimensions
            processed_action = processed_action[:7]
    
    # Scale position deltas (LIBERO expects small delta movements)
    # VLA models output normalized actions [-1, 1], scale to reasonable deltas
    position_scale = 0.05  # 5cm max delta movement
    rotation_scale = 0.1   # Small rotation deltas
    
    processed_action[:3] *= position_scale  # x, y, z deltas
    processed_action[3:6] *= rotation_scale  # rx, ry, rz deltas
    
    # Process gripper action for LIBERO
    # LIBERO gripper: 1.0 = open, -1.0 = close
    gripper_value = processed_action[6]
    if gripper_value > 0.5:
        processed_action[6] = 1.0   # Open gripper
    elif gripper_value < -0.5:
        processed_action[6] = -1.0  # Close gripper  
    else:
        processed_action[6] = 0.0   # No change
    
    # Apply action normalization and inversion as done in SimpleVLA-RL
    # This binarizes the gripper and applies the inversion
    processed_action = normalize_gripper_action_libero(processed_action, binarize=True)
    processed_action = invert_gripper_action_libero(processed_action)
    
    return processed_action


def normalize_gripper_action_libero(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    Normalize gripper action for LIBERO (reimplemented from SimpleVLA-RL)
    
    Args:
        action: 7-DOF action array
        binarize: Whether to binarize gripper action
        
    Returns:
        Normalized action
    """
    normalized_action = action.copy()
    
    if binarize and len(normalized_action) >= 7:
        # Binarize gripper: > 0 -> 1.0 (open), <= 0 -> -1.0 (close)
        gripper_val = normalized_action[6]
        normalized_action[6] = 1.0 if gripper_val > 0 else -1.0
    
    return normalized_action


def invert_gripper_action_libero(action: np.ndarray) -> np.ndarray:
    """
    Invert gripper action for LIBERO (reimplemented from SimpleVLA-RL)
    
    LIBERO gripper convention might need inversion depending on the model
    
    Args:
        action: 7-DOF action array
        
    Returns:
        Action with potentially inverted gripper
    """
    inverted_action = action.copy()
    
    # The gripper inversion logic from SimpleVLA-RL
    # This might be model/task specific, but generally:
    # VLA model outputs: positive = close, negative = open
    # LIBERO expects: positive = open, negative = close
    if len(inverted_action) >= 7:
        inverted_action[6] = -inverted_action[6]
    
    return inverted_action


def extract_libero_observation(obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Extract and process observation from LIBERO environment
    
    Args:
        obs: Raw observation from LIBERO environment
        
    Returns:
        Processed observation for VLA model
    """
    processed_obs = {}
    
    # Extract main camera image
    if 'agentview_image' in obs:
        image = obs['agentview_image']
        # Flip image as done in SimpleVLA-RL
        processed_obs['agentview_image'] = image[::-1, ::-1]
    
    # Extract wrist camera if available
    if 'robot0_eye_in_hand_image' in obs:
        wrist_image = obs['robot0_eye_in_hand_image']
        processed_obs['wrist_image'] = wrist_image[::-1, ::-1]
    
    # Extract proprioception
    if 'robot0_proprio' in obs:
        processed_obs['robot0_proprio'] = obs['robot0_proprio']
    
    # Extract other relevant observations
    for key, value in obs.items():
        if key not in ['agentview_image', 'robot0_eye_in_hand_image', 'robot0_proprio']:
            processed_obs[key] = value
    
    return processed_obs


def check_libero_success(env) -> bool:
    """
    Check if LIBERO task was completed successfully
    
    Args:
        env: LIBERO environment
        
    Returns:
        True if task completed successfully
    """
    try:
        # Check if environment has success evaluation
        if hasattr(env, 'is_success'):
            return env.is_success()
        elif hasattr(env, '_check_success'):
            return env._check_success()
        else:
            # Fallback - assume no success detection available
            logger.warning("No success detection method found in LIBERO environment")
            return False
    
    except Exception as e:
        logger.error(f"Error checking LIBERO success: {e}")
        return False


def run_libero_episode(env, vla_model_fn, task_description: str, max_steps: int = 512) -> Dict[str, Any]:
    """
    Run a complete LIBERO episode with VLA model
    
    Args:
        env: LIBERO environment
        vla_model_fn: Function that takes (observation, instruction) and returns action
        task_description: Task instruction
        max_steps: Maximum steps per episode
        
    Returns:
        Episode data dictionary
    """
    logger.info(f"Starting LIBERO episode: {task_description}")
    
    episode_data = {
        'task_description': task_description,
        'observations': [],
        'actions': [],
        'rewards': [],
        'success': False,
        'episode_length': 0,
        'total_reward': 0.0
    }
    
    # Get initial observation
    obs = env.get_obs() if hasattr(env, 'get_obs') else env._get_obs()
    processed_obs = extract_libero_observation(obs)
    episode_data['observations'].append(processed_obs)
    
    step_count = 0
    
    try:
        while step_count < max_steps:
            # Generate VLA action
            action = vla_model_fn(processed_obs, task_description)
            
            # Process action for LIBERO
            libero_action = process_libero_action(action)
            
            # Execute action
            obs, reward, done, info = env.step(libero_action.tolist())
            
            # Process new observation
            processed_obs = extract_libero_observation(obs)
            
            # Store step data
            episode_data['actions'].append(action)
            episode_data['observations'].append(processed_obs)
            episode_data['rewards'].append(reward)
            
            step_count += 1
            
            # Check termination
            if done:
                episode_data['success'] = check_libero_success(env)
                logger.info(f"LIBERO episode terminated at step {step_count}, success: {episode_data['success']}")
                break
        
        episode_data['episode_length'] = step_count
        episode_data['total_reward'] = 1.0 if episode_data['success'] else 0.0  # Binary reward
        
        logger.info(f"LIBERO episode completed: {step_count} steps, success: {episode_data['success']}")
        
    except Exception as e:
        logger.error(f"Error during LIBERO episode: {e}")
        traceback.print_exc()
        episode_data['success'] = False
        episode_data['total_reward'] = 0.0
    
    return episode_data


def close_libero_environment(env):
    """Safely close LIBERO environment"""
    try:
        if env is not None:
            env.close()
            logger.info("LIBERO environment closed")
    except Exception as e:
        logger.error(f"Error closing LIBERO environment: {e}")
    
    # Clear GPU memory
    torch.cuda.empty_cache()
