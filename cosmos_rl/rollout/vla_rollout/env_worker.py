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
Environment worker process for LIBERO environments.

This module implements the multiprocessing pattern from SimpleVLA-RL to avoid
shared OpenGL/MuJoCo state issues when running multiple LIBERO environments.
"""

import gc
import torch
from typing import Any, Dict
from cosmos_rl.utils.logging import logger


def libero_env_worker(
    task_suite: str,
    task_id: int,
    trial_id: int,
    input_queue,
    output_queue,
    save_video: bool,
    global_steps: int,
    max_steps: int,
    resolution: int = 256
):
    """
    Worker process for LIBERO environments.
    
    Runs in a separate process to provide complete memory isolation,
    avoiding shared OpenGL/MuJoCo rendering state issues.
    
    Args:
        task_suite: LIBERO task suite name (e.g., 'libero_10')
        task_id: Task ID within the suite
        trial_id: Trial ID for initial state selection
        input_queue: Queue to receive actions from main process
        output_queue: Queue to send observations to main process
        save_video: Whether to save video frames
        global_steps: Current training step (for video naming)
        max_steps: Maximum steps per episode
        resolution: Image resolution
    """
    from libero.libero import benchmark
    from cosmos_rl.rollout.vla_rollout.libero_utils import (
        get_libero_dummy_action,
        normalize_gripper_action,
        invert_gripper_action
    )
    
    # Get LIBERO task
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_obj = benchmark_dict[task_suite]()
    task = task_suite_obj.get_task(task_id)
    initial_states = task_suite_obj.get_task_init_states(task_id)
    initial_state = initial_states[trial_id % len(initial_states)]
    
    # Initialize environment with retry logic
    env = None
    while True:
        try:
            from cosmos_rl.rollout.vla_rollout.libero_utils import get_libero_env
            env, task_description = get_libero_env(task, 'openvla', resolution=resolution)
            break
        except Exception as e:
            logger.error(f"Environment initialization failed: {e}")
            if env is not None:
                try:
                    env.close()
                except Exception as close_error:
                    logger.error(f"Error closing env: {close_error}")
            torch.cuda.empty_cache()
            gc.collect()
    
    # Reset and set initial state
    env.reset()
    obs = env.set_init_state(initial_state)
    
    # Warmup steps (CRITICAL for LIBERO camera initialization)
    num_wait_steps = 10
    for _ in range(num_wait_steps):
        obs, _, _, _ = env.step(get_libero_dummy_action('openvla'))
    
    # IMPORTANT: Flip LIBERO images (they come upside-down and mirrored from MuJoCo)
    if 'agentview_image' in obs:
        obs['agentview_image'] = obs['agentview_image'][::-1, ::-1].copy()
    if 'robot0_eye_in_hand_image' in obs:
        obs['robot0_eye_in_hand_image'] = obs['robot0_eye_in_hand_image'][::-1, ::-1].copy()
    
    # Collect initial frame for video (obs is already flipped now)
    valid_images = []
    if save_video and 'agentview_image' in obs:
        valid_images.append(obs['agentview_image'].copy())
    
    # Send initial observation to main process
    output_queue.put({
        'type': 'init',
        'obs': obs,
        'task_description': task_description,
        'valid_images': valid_images,
        'task_file_name': f"{task_suite}_task_{task_id}_trial_{trial_id}",
        'active': True,
        'complete': False,
        'finish_step': 0
    })
    
    # Episode execution loop
    active = True
    complete = False
    finish_step = 0
    
    while True:
        # Wait for action from main process (blocking)
        action = input_queue.get()
        
        # None signal means terminate
        if action is None:
            env.close()
            output_queue.put({'type': 'terminate'})
            break
        
        # Execute action chunk
        step_images = []
        for i in range(len(action)):
            single_action = action[i]
            
            # Process action (normalize and invert gripper)
            normalized_action = normalize_gripper_action(single_action, binarize=True)
            inverted_action = invert_gripper_action(normalized_action)
            
            # Step environment
            obs, reward, done, info = env.step(inverted_action.tolist())
            
            # IMPORTANT: Flip LIBERO images (they come upside-down and mirrored from MuJoCo)
            if 'agentview_image' in obs:
                obs['agentview_image'] = obs['agentview_image'][::-1, ::-1].copy()
            if 'robot0_eye_in_hand_image' in obs:
                obs['robot0_eye_in_hand_image'] = obs['robot0_eye_in_hand_image'][::-1, ::-1].copy()
            
            # Collect frame for video (obs is already flipped now)
            if save_video and 'agentview_image' in obs:
                step_images.append(obs['agentview_image'].copy())
                logger.debug(f"[Worker] Collected frame {i}: shape={obs['agentview_image'].shape}, mean={obs['agentview_image'].mean():.2f}")
            
            finish_step += 1
            
            # Check termination
            if done or finish_step >= max_steps:
                active = False
                complete = done
                break
        
        # Send result back to main process
        output_data = {
            'type': 'step',
            'obs': obs,
            'active': active,
            'complete': complete,
            'finish_step': finish_step,
            'valid_images': step_images if save_video else []
        }
        output_queue.put(output_data)


def robotwin_env_worker(
    task_name: str,
    task_id: int,
    trial_id: int,
    input_queue,
    output_queue,
    save_video: bool,
    global_steps: int,
    max_steps: int
):
    """
    Worker process for RoboTwin environments.
    
    Similar to libero_env_worker but for RoboTwin tasks.
    
    Args:
        task_name: RoboTwin task name
        task_id: Task ID
        trial_id: Trial ID
        input_queue: Queue to receive actions
        output_queue: Queue to send observations
        save_video: Whether to save video frames
        global_steps: Current training step
        max_steps: Maximum steps per episode
    """
    # TODO: Implement RoboTwin worker when needed
    raise NotImplementedError("RoboTwin worker not yet implemented")

