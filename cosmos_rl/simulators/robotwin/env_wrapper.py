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
import gym
import numpy as np
from typing import Union, List, Any, Optional, Dict
from dataclasses import dataclass
import torch

from cosmos_rl.simulators.robotwin.venv import VectorEnv
from cosmos_rl.simulators.utils import save_rollout_video
from cosmos_rl.simulators.robotwin.venv import get_robotwin_task_suite


@dataclass
class EnvStates:
    env_idx: int
    task_id: int = -1
    trial_id: int = -1
    active: bool = True
    complete: bool = False
    step: int = 0
    language: str = ""

    current_obs: Optional[Any] = None
    do_validation: bool = False
    valid_pixels: Optional[Dict[str, Any]] = None


class RoboTwinEnvWrapper(gym.Env):
    """High-level wrapper for RoboTwin environments.

    This wrapper provides a unified interface for managing multiple RoboTwin
    environments with support for partial reset/step operations, observation
    extraction, and validation video recording.
    """

    def __init__(self, cfg, *args, skip_instruction_setup=False, **kwargs):
        """Initialize RoboTwinEnvWrapper.

        Args:
            cfg: Configuration object with attributes:
                - num_envs: Number of parallel environments
                - seed: Random seed
                - task_config: Task configuration dictionary
                - max_steps: Maximum steps per episode (default: 512)
        """
        self.num_envs = getattr(cfg, "num_envs", 1)
        self.seed = getattr(cfg, "seed", 0)
        self.task_config = getattr(cfg, "task_config", {})
        self.max_steps = getattr(cfg, "max_steps", 512)
        self.skip_instruction_setup = skip_instruction_setup
        self.task_suite = get_robotwin_task_suite()

        # Initialize environment states
        self.env_states = [EnvStates(env_idx=i) for i in range(self.num_envs)]

        self._init_env()

    def _init_env(self):
        """Initialize the vector environment."""
        # Ensure RoboTwin is in path
        robotwin_path = os.getenv("ROBOTWIN_PATH")
        if robotwin_path:
            import sys

            if robotwin_path not in sys.path:
                sys.path.insert(0, robotwin_path)

        # Generate seeds for each environment
        env_seeds = [self.seed + i for i in range(self.num_envs)]

        # Create vector environment with task configuration
        # Note: task_config will be merged with defaults from demo_randomized.yml
        self.env = VectorEnv(
            n_envs=self.num_envs,
            env_seeds=env_seeds,
            instruction_type=self.task_config.get("instruction_type", "seen"),
        )

    def _extract_image_and_state(self, obs_list):
        """Extract and format observations from raw environment observations.

        Args:
            obs_list: List of observation dictionaries from environments

        Returns:
            Dictionary with keys:
                - full_images: Array of full camera images
                - left_wrist_images: Array of left wrist camera images (if available)
                - right_wrist_images: Array of right wrist camera images (if available)
                - states: Array of robot states
        """
        full_images = []
        left_wrist_images = []
        right_wrist_images = []
        states = []

        for obs in obs_list:
            if obs is None:
                continue

            # Extract full image (head camera)
            if "full_image" in obs:
                full_images.append(obs["full_image"])

            # Extract wrist images
            if "left_wrist_image" in obs and obs["left_wrist_image"] is not None:
                left_wrist_images.append(obs["left_wrist_image"])

            if "right_wrist_image" in obs and obs["right_wrist_image"] is not None:
                right_wrist_images.append(obs["right_wrist_image"])

            # Extract state
            if "state" in obs:
                states.append(obs["state"])

        result = {
            "full_images": np.stack(full_images) if full_images else None,
            "states": np.stack(states) if states else None,
        }

        if left_wrist_images:
            result["left_wrist_images"] = np.stack(left_wrist_images)

        if right_wrist_images:
            result["right_wrist_images"] = np.stack(right_wrist_images)

        return result

    def _setup_task(
        self, env_ids: List[int], task_ids: List[int], trial_ids: List[int]
    ):
        """Setup tasks for specified environments.

        Args:
            env_ids: List of environment indices to setup
            task_ids: List of task IDs (index into task_suite)
            trial_ids: List of trial/seed IDs

        Returns:
            List of task descriptions/instructions
        """
        # Reset environments with Libero-style interface
        obs_list = self.env.reset(
            env_ids=env_ids, task_ids=task_ids, trial_ids=trial_ids
        )

        # Extract instructions from observations
        task_descriptions = []
        for i, env_id in enumerate(env_ids):
            instruction = obs_list[i].get("instruction", "") if obs_list[i] else ""

            self.env_states[env_id] = EnvStates(
                env_idx=env_id,
                task_id=task_ids[i] if i < len(task_ids) else 0,
                trial_id=trial_ids[i] if i < len(trial_ids) else 0,
                active=True,
                complete=False,
                step=0,
                language=instruction,
            )
            task_descriptions.append(instruction)

        return task_descriptions

    def reset(
        self,
        env_ids: Union[int, List[int]],
        task_ids: Union[int, List[int]],
        trial_ids: Union[int, List[int]],
        do_validation: Union[bool, List[bool]],
    ):
        """Reset specified environments.

        Args:
            env_ids: Environment indices to reset
            task_ids: Task IDs for each environment
            trial_ids: Trial/seed IDs for each environment
            do_validation: Whether to record validation videos

        Returns:
            Tuple of (observations_dict, task_descriptions)
        """
        # Normalize inputs to lists
        if isinstance(env_ids, int):
            env_ids = [env_ids]
        if isinstance(task_ids, int):
            task_ids = [task_ids] * len(env_ids)
        if isinstance(trial_ids, int):
            trial_ids = [trial_ids] * len(env_ids)
        if isinstance(do_validation, bool):
            do_validation = [do_validation] * len(env_ids)

        # Setup tasks (this also resets and gets initial observations)
        task_descriptions = self._setup_task(env_ids, task_ids, trial_ids)

        # Get initial observations again for proper formatting
        obs_list = self.env.get_obs(env_ids)
        images_and_states = self._extract_image_and_state(obs_list)

        # Setup validation tracking and store initial observations
        for i, env_id in enumerate(env_ids):
            self.env_states[env_id].do_validation = do_validation[i]
            if do_validation[i]:
                self.env_states[env_id].valid_pixels = {
                    "full_images": [],
                    "left_wrist_images": [],
                    "right_wrist_images": [],
                }

                # Record initial frame
                if images_and_states.get("full_images") is not None:
                    self.env_states[env_id].valid_pixels["full_images"].append(
                        images_and_states["full_images"][i]
                    )

            # Store current observation
            self.env_states[env_id].current_obs = {
                k: v[i] if v is not None else None for k, v in images_and_states.items()
            }

        return images_and_states, task_descriptions

    def reset_async(
        self,
        env_ids: Union[int, List[int]],
        task_ids: Union[int, List[int]],
        trial_ids: Union[int, List[int]],
        do_validation: Union[bool, List[bool]],
    ):
        """Start asynchronous reset of specified environments.

        Args:
            env_ids: Environment indices to reset
            task_ids: Task IDs for each environment
            trial_ids: Trial/seed IDs for each environment
            do_validation: Whether to record validation videos
        """
        # Normalize inputs to lists
        if isinstance(env_ids, int):
            env_ids = [env_ids]
        if isinstance(task_ids, int):
            task_ids = [task_ids] * len(env_ids)
        if isinstance(trial_ids, int):
            trial_ids = [trial_ids] * len(env_ids)
        if isinstance(do_validation, bool):
            do_validation = [do_validation] * len(env_ids)

        # Store pending reset info and do validation setup
        for i, env_id in enumerate(env_ids):
            self.env_states[env_id].task_id = task_ids[i]
            self.env_states[env_id].trial_id = trial_ids[i]
            self.env_states[env_id].do_validation = do_validation[i]
            if do_validation[i]:
                self.env_states[env_id].valid_pixels = {
                    "full_images": [],
                    "left_wrist_images": [],
                    "right_wrist_images": [],
                }

        # Setup tasks and get descriptions
        task_descriptions = self._setup_task(env_ids, task_ids, trial_ids)

        # Store descriptions in env states
        for i, env_id in enumerate(env_ids):
            self.env_states[env_id].language = task_descriptions[i]

    def reset_wait(self, env_ids: List[int]):
        """Wait for asynchronous reset to complete.

        Args:
            env_ids: Environment indices to wait for

        Returns:
            Tuple of (observations_dict, task_descriptions)
        """
        # Get observations after reset
        obs_list = self.env.get_obs(env_ids)
        images_and_states = self._extract_image_and_state(obs_list)

        # Update environment states
        task_descriptions = []
        for i, env_id in enumerate(env_ids):
            self.env_states[env_id].active = True
            self.env_states[env_id].complete = False
            self.env_states[env_id].step = 0

            # Store current observation
            self.env_states[env_id].current_obs = {
                k: v[i] if v is not None else None for k, v in images_and_states.items()
            }

            task_descriptions.append(self.env_states[env_id].language)

        return images_and_states, task_descriptions

    def step(self, env_ids: List[int], action):
        """Execute one step in specified environments.

        Args:
            env_ids: List of environment indices to step
            action: Actions for each environment

        Returns:
            Dictionary with keys:
                - full_images, left_wrist_images, right_wrist_images, states
                - complete: Boolean array indicating episode completion
                - active: Boolean array indicating if environment is still active
                - finish_step: Array of step counts
        """
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        # Filter for active environments
        active_indices = [
            i for i, env_id in enumerate(env_ids) if self.env_states[env_id].active
        ]

        if active_indices:
            active_env_ids = [env_ids[i] for i in active_indices]
            active_action = action[active_indices]

            # Execute step
            obs_list, rewards, terminated, truncated, infos = self.env.step(
                active_action, env_ids=active_env_ids
            )
            images_and_states = self._extract_image_and_state(obs_list)

            # Update environment states
            for i, env_id in enumerate(active_env_ids):
                self.env_states[env_id].step += 1

                # Check if episode is done
                done = terminated[i] or truncated[i]
                if done or self.env_states[env_id].step >= self.max_steps:
                    self.env_states[env_id].complete = terminated[i] or infos[i].get(
                        "success", False
                    )
                    self.env_states[env_id].active = False

                # Update current observation
                for k, v in images_and_states.items():
                    if v is not None:
                        self.env_states[env_id].current_obs[k] = v[i]

                # Record validation frames
                if self.env_states[env_id].do_validation:
                    if images_and_states.get("full_images") is not None:
                        self.env_states[env_id].valid_pixels["full_images"].append(
                            images_and_states["full_images"][i]
                        )

        # Collect results from all environments (including inactive ones)
        completes = np.array([self.env_states[env_id].complete for env_id in env_ids])
        active = np.array([self.env_states[env_id].active for env_id in env_ids])
        finish_steps = np.array([self.env_states[env_id].step for env_id in env_ids])

        # Build full observation dict
        full_images_and_states = {}
        for key in ["full_images", "left_wrist_images", "right_wrist_images", "states"]:
            obs_list = []
            for env_id in env_ids:
                if (
                    self.env_states[env_id].current_obs
                    and key in self.env_states[env_id].current_obs
                ):
                    obs_list.append(self.env_states[env_id].current_obs[key])

            if obs_list and obs_list[0] is not None:
                full_images_and_states[key] = np.stack(obs_list)

        return {
            **full_images_and_states,
            "complete": completes,
            "active": active,
            "finish_step": finish_steps,
        }

    def chunk_step(self, env_ids: List[int], actions: torch.Tensor):
        """Execute multiple steps with a sequence of actions.

        Args:
            env_ids: List of environment indices
            actions: Tensor of shape (num_envs, num_steps, action_dim)

        Returns:
            Results from the final step
        """
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        num_steps = actions.shape[1]
        for step in range(num_steps):
            results = self.step(env_ids, actions[:, step])

        return results

    def get_env_states(self, env_ids: List[int]):
        """Get environment states for specified environments.

        Args:
            env_ids: List of environment indices

        Returns:
            List of EnvStates objects
        """
        return [self.env_states[env_id] for env_id in env_ids]

    def save_validation_videos(self, rollout_dir: str, env_ids: List[int]):
        """Save validation videos for specified environments.

        Args:
            rollout_dir: Directory to save videos
            env_ids: List of environment indices to save videos for
        """
        for env_id in env_ids:
            state = self.env_states[env_id]
            if not state.do_validation:
                continue

            if not state.valid_pixels or not state.valid_pixels.get("full_images"):
                continue

            task_name = self.task_config.get("task_name", "robotwin_task")
            video_name = f"{task_name}_task_{state.task_id}_trial_{state.trial_id}"

            save_rollout_video(
                state.valid_pixels["full_images"],
                rollout_dir,
                video_name,
                state.complete,
            )

    def close(self, clear_cache: bool = True):
        """Close all environments and free resources.

        Args:
            clear_cache: Whether to clear rendering cache
        """
        if hasattr(self, "env") and self.env is not None:
            self.env.close(clear_cache=clear_cache)
