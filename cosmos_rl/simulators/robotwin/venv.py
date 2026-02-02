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
Vectorized RoboTwin environment wrapper with thread-based parallelization.

This module provides SubEnv and VectorEnv classes for managing multiple
RoboTwin simulation environments in parallel using threading (required for CUDA).
Supports partial reset/step operations and Libero-style uniform interfaces.
"""

import importlib
import os
import gc
import logging
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Union, List, Optional, Dict, Any
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml

# Monkey-patch to disable OIDN denoiser on incompatible GPUs (H200/Hopper)
_original_set_denoiser = None


def _disable_oidn_denoiser():
    """Disable OIDN denoiser to avoid GPU compatibility issues on H200/Hopper."""
    try:
        import sapien.render

        global _original_set_denoiser
        if _original_set_denoiser is None:
            _original_set_denoiser = sapien.render.set_ray_tracing_denoiser

        def patched_set_denoiser(denoiser_type: str):
            if denoiser_type.lower() == "oidn":
                logging.info(
                    "OIDN denoiser disabled for GPU compatibility (H200/Hopper)"
                )
                return
            _original_set_denoiser(denoiser_type)

        sapien.render.set_ray_tracing_denoiser = patched_set_denoiser
    except ImportError:
        pass  # SAPIEN not imported yet, will be patched when needed


_disable_oidn_denoiser()


def _class_decorator(task_name: str):
    """Dynamically import and instantiate a RoboTwin task class.

    Args:
        task_name: Name of the task class to import

    Returns:
        Instance of the task class

    Raises:
        SystemExit: If task not found
    """
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except AttributeError:
        raise SystemExit(f"No Task: {task_name}")
    return env_instance


def _update_obs(observation: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and format observation from RoboTwin environment.

    Args:
        observation: Raw observation dict from RoboTwin

    Returns:
        Formatted observation with images and state
    """
    full_image = observation["observation"]["head_camera"]["rgb"]
    left_wrist_image = (
        observation["observation"].get("left_camera", {}).get("rgb", None)
    )
    right_wrist_image = (
        observation["observation"].get("right_camera", {}).get("rgb", None)
    )
    state = observation["joint_action"]["vector"]

    return {
        "full_image": full_image,
        "left_wrist_image": left_wrist_image,
        "right_wrist_image": right_wrist_image,
        "state": state,
    }


def get_robotwin_task_suite() -> List[str]:
    """Automatically discover all available RoboTwin tasks.

    Scans the RoboTwin/envs directory for task files and returns a sorted list
    of task names, excluding special files like __init__, _base_task, etc.

    Returns:
        Sorted list of task names

    Raises:
        RuntimeError: If ROBOTWIN_PATH not set or envs directory not found
    """
    robotwin_path = os.getenv("ROBOTWIN_PATH")
    if not robotwin_path:
        raise RuntimeError(
            "ROBOTWIN_PATH environment variable not set. "
            "Please set: export ROBOTWIN_PATH=/path/to/RoboTwin"
        )

    envs_dir = Path(robotwin_path) / "envs"
    if not envs_dir.exists():
        raise RuntimeError(f"RoboTwin envs directory not found: {envs_dir}")

    # Find all .py files in envs directory
    task_files = envs_dir.glob("*.py")

    # Exclude special files
    exclude = {"__init__", "_base_task", "_GLOBAL_CONFIGS", "reward"}

    task_names = []
    for task_file in task_files:
        task_name = task_file.stem
        if task_name not in exclude and not task_name.startswith("_"):
            task_names.append(task_name)

    return sorted(task_names)


class SubEnv:
    """Single RoboTwin environment instance with task metadata caching.

    Manages a single simulated environment with thread-safe operations.
    Caches task metadata (episode info) to avoid redundant setup calls.
    """

    def __init__(
        self,
        task_args: Dict[str, Any],
        env_id: int = 0,
        env_seed: Optional[int] = None,
        instruction_type: str = "seen",
        global_lock: Optional[threading.Lock] = None,
        task_metadata_cache: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a SubEnv for RoboTwin.

        Args:
            task_args: Task configuration arguments
            env_id: Environment ID for logging purposes
            env_seed: Random seed (defaults to env_id if None)
            instruction_type: Type of instruction to generate ("seen" or "unseen")
            global_lock: Shared lock for coordinating environment creation
            task_metadata_cache: Shared cache for task metadata (episode info)
        """
        self.env_id = env_id
        self.task_args = task_args.copy()
        self.env_seed = env_seed if env_seed is not None else env_id
        self.instruction_type = instruction_type
        self.instruction = None
        self.task = None
        self.episode_info_list = None
        self.current_task_name = None

        # Shared cache for task metadata to avoid redundant setup_task calls
        self.task_metadata_cache = (
            task_metadata_cache if task_metadata_cache is not None else {}
        )

        self.global_lock = global_lock if global_lock else threading.Lock()
        self.local_lock = threading.Lock()

    def setup_task(self, task_name: str):
        """Setup task metadata by running a test episode.

        This creates a temporary task instance to extract episode info
        for instruction generation. Results are cached per task_name.
        """
        # Check cache first to avoid redundant setup
        if task_name in self.task_metadata_cache:
            self.episode_info_list = self.task_metadata_cache[task_name]
            logging.info(f"Using cached metadata for task: {task_name}")
            return

        # Double-check cache after acquiring lock
        if task_name in self.task_metadata_cache:
            self.episode_info_list = self.task_metadata_cache[task_name]
            return

        trial_seed = self.env_seed
        is_valid = False
        max_retries = 10
        retry_count = 0
        last_error = None

        logging.info(f"Setting up task metadata for: {task_name}")

        while not is_valid and retry_count < max_retries:
            try:
                task = _class_decorator(task_name)
                task.setup_demo(
                    now_ep_num=trial_seed,
                    seed=trial_seed,
                    is_test=True,
                    **self.task_args,
                )
                episode_info = task.get_info()
                is_valid = True
            except KeyError as e:
                # KeyError likely means config issue, not seed issue - fail immediately
                logging.error(f"SubEnv setup_task KeyError (config issue): {e}")
                if hasattr(task, "close_env"):
                    task.close_env()
                raise RuntimeError(
                    f"Configuration error in setup_task: Missing key {e}. "
                    f"Check that task_config has all required fields."
                )
            except Exception as e:
                # Other errors might be seed-related, retry with different seed
                last_error = e
                error_str = str(e)
                logging.warning(
                    f"SubEnv setup_task failed with seed {trial_seed}: {error_str[:100]}"
                )
                if hasattr(task, "close_env"):
                    task.close_env()
                trial_seed += 1
                retry_count += 1
                continue

            if hasattr(task, "close_env"):
                task.close_env()

        if not is_valid:
            raise RuntimeError(
                f"SubEnv failed to setup after {max_retries} attempts. "
                f"Last error: {last_error}"
            )

        self.episode_info_list = [episode_info]
        # Cache the result
        self.task_metadata_cache[task_name] = self.episode_info_list
        logging.info(f"Cached metadata for task: {task_name}")

    def create_instruction(self, task_name: str) -> str:
        """Generate task instruction/description.

        Returns:
            Task instruction string
        """
        # Import here to avoid circular dependencies
        try:
            from description.utils.generate_episode_instructions import (
                generate_episode_descriptions,
            )

            task_descriptions = generate_episode_descriptions(
                task_name, self.episode_info_list, 1, self.env_seed
            )
            instruction = np.random.choice(task_descriptions[0][self.instruction_type])
            return instruction
        except ImportError:
            logging.warning(
                "Could not import generate_episode_descriptions, using default instruction"
            )
            return "Complete the {} task".format(task_name)

    def step(self, action):
        """Execute one step in the environment.

        Args:
            action: Action to take in the environment (1D array of action_dim)

        Returns:
            Dictionary containing obs, reward, terminated, truncated, and info
        """
        if self.task is None:
            error_msg = (
                f"SubEnv {self.env_id} task not initialized. "
                f"Call reset() with task_name first."
            )
            logging.error(error_msg)
            raise RuntimeError(error_msg)

        with self.local_lock:
            try:
                # RoboTwin's gen_sparse_reward_data expects chunk_actions with shape (num_steps, action_dim)
                # For single-step execution, wrap action in an extra dimension
                if action.ndim == 1:
                    chunk_action = action[
                        np.newaxis, :
                    ]  # (action_dim,) -> (1, action_dim)
                else:
                    chunk_action = action

                reward, terminated, truncated, info = self.task.gen_sparse_reward_data(
                    chunk_action
                )
                obs = _update_obs(self.task.get_obs())
                obs["instruction"] = self.task.get_instruction()

                return {
                    "obs": obs,
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "info": info,
                }
            except Exception as e:
                logging.error(f"SubEnv {self.env_id} step error: {e}")
                raise

    def reset(self, task_name: str, env_seed: Optional[int] = None):
        """Reset the environment with a new seed.

        Args:
            task_name: Name of the task to reset to
            env_seed: New seed for reset (uses current seed if None)

        Returns:
            Initial observation
        """

        # Close existing task
        if self.task is not None and hasattr(self.task, "close_env"):
            try:
                self.task.close_env()
            except Exception as e:
                logging.warning(f"Error closing task: {e}")

        # Update seed if provided
        if env_seed is not None:
            self.env_seed = env_seed

        # Only setup task metadata if task changed (cached otherwise)
        if task_name != self.current_task_name:
            try:
                self.setup_task(task_name)
                self.instruction = self.create_instruction(task_name)
                self.current_task_name = task_name
                logging.info(f"SubEnv {self.env_id} task changed to: {task_name}")
            except Exception as e:
                logging.error(
                    f"SubEnv {self.env_id} failed to setup task {task_name}: {e}"
                )
                raise

        self.task_args["instruction"] = self.instruction
        logging.debug(
            f"SubEnv {self.env_id} resetting with task {task_name}, seed {self.env_seed}"
        )

        # Try to create and setup task with retries
        trial_seed = self.env_seed
        is_valid = False
        max_retries = 10
        retry_count = 0

        while not is_valid and retry_count < max_retries:
            try:
                self.task = _class_decorator(task_name)
                logging.info(
                    f"SubEnv {self.env_id} setup_demo with task {task_name}, seed {trial_seed}, retry {retry_count}"
                )
                self.task.setup_demo(
                    now_ep_num=trial_seed, seed=trial_seed, **self.task_args
                )
                self.task.step_lim = self.task_args.get("step_lim", 512)
                self.task.run_steps = 0
                self.task.reward_step = 0
                is_valid = True
            except Exception as e:
                logging.warning(
                    f"SubEnv {self.env_id} reset failed with seed {trial_seed}: {e}, retrying..."
                )
                if self.task is not None and hasattr(self.task, "close_env"):
                    self.task.close_env()
                trial_seed += 1
                retry_count += 1
                continue

        if not is_valid:
            raise RuntimeError(
                f"SubEnv {self.env_id} failed to reset after {max_retries} attempts"
            )

        # Get initial observation
        obs = _update_obs(self.task.get_obs())
        obs["instruction"] = self.task.get_instruction()
        return obs

    def get_obs(self):
        """Get current observation from the environment.

        Returns:
            Current observation with instruction
        """
        if self.task is None:
            return None

        with self.local_lock:
            obs = _update_obs(self.task.get_obs())
            obs["instruction"] = self.task.get_instruction()
            return obs

    def get_instruction(self):
        """Get current task instruction.

        Returns:
            Task instruction string or None if task not initialized
        """
        with self.local_lock:
            if self.task is None:
                return None
            return self.instruction

    def close(self, clear_cache: bool = True):
        """Close environment and free resources.

        Args:
            clear_cache: Whether to clear rendering cache
        """
        if self.task is not None:
            with self.local_lock:
                try:
                    self.task.close_env(clear_cache=clear_cache)
                except Exception as e:
                    logging.warning(f"Error closing SubEnv {self.env_id}: {e}")


class VectorEnv(gym.Env):
    """Vectorized environment manager for multiple RoboTwin environments.

    Manages multiple SubEnv instances in parallel using threading (required for CUDA).
    Supports Libero-style uniform interface with task_suite-based task selection.
    """

    def __init__(
        self,
        task_suite: Optional[Union[str, List[str]]] = None,
        task_config: Optional[Dict[str, Any]] = None,
        n_envs: int = 1,
        env_seeds: Optional[List[int]] = None,
        instruction_type: str = "seen",
        max_workers: Optional[int] = None,
    ):
        """Initialize VectorEnv with task suite and configuration.

        Args:
            task_suite: Task name, list of task names, or None (all RoboTwin tasks)
            task_config: Task configuration dict (merged with demo_randomized.yml)
            n_envs: Number of parallel environments
            env_seeds: Random seeds for each environment
            instruction_type: Type of instruction ("seen" or "unseen")
            max_workers: Max worker threads (defaults to n_envs)
        """
        self.num_envs = n_envs

        # Setup task suite
        if task_suite is None:
            self.task_suite = get_robotwin_task_suite()
        elif isinstance(task_suite, str):
            self.task_suite = [task_suite]
        else:
            self.task_suite = list(task_suite)

        # Setup seeds
        if env_seeds is not None:
            assert (
                len(env_seeds) == n_envs
            ), f"env_seeds length ({len(env_seeds)}) must match n_envs ({n_envs})"
            self.env_seeds = list(env_seeds)
        else:
            self.env_seeds = list(range(n_envs))

        # Load fully processed default config, merge with user overrides
        self.default_config = self._load_default_config()
        if task_config:
            # Merge user config with defaults
            self.default_config.update(task_config)

        self.global_lock = threading.Lock()

        # Shared cache for task metadata across all SubEnvs
        # This avoids redundant setup_task() calls when multiple envs use the same task
        self.task_metadata_cache: Dict[str, Any] = {}

        # Create sub-environments (each can have different task_name)
        self.envs: List[SubEnv] = []
        for i in range(n_envs):
            sub_env = SubEnv(
                task_args=self.default_config,
                env_id=i,
                env_seed=self.env_seeds[i],
                instruction_type=instruction_type,
                global_lock=self.global_lock,
                task_metadata_cache=self.task_metadata_cache,
            )
            self.envs.append(sub_env)

        # Thread pool for parallel operations
        self.max_workers = max_workers if max_workers else self.num_envs
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)

    def _load_default_config(self) -> Dict[str, Any]:
        """Load and process default configuration from RoboTwin.

        Following SimpleVLA-RL pattern: load yaml, process embodiment/camera configs,
        return fully processed default config ready to merge with user overrides.

        Returns:
            Fully processed default configuration with embodiment and camera configs

        Raises:
            RuntimeError: If ROBOTWIN_PATH not set or config files not found
        """

        # Get paths
        robotwin_path = os.getenv("ROBOTWIN_PATH")
        if not robotwin_path:
            raise RuntimeError(
                "ROBOTWIN_PATH environment variable not set. "
                "RoboTwin configuration files are required. "
                "Please set: export ROBOTWIN_PATH=/path/to/RoboTwin"
            )
        os.environ["ASSETS_PATH"] = robotwin_path

        from envs._GLOBAL_CONFIGS import CONFIGS_PATH

        # Load base config file
        config_file = os.path.join(CONFIGS_PATH, "demo_randomized.yml")
        if not os.path.exists(config_file):
            raise RuntimeError(
                f"RoboTwin config file not found: {config_file}\n"
                f"Please ensure RoboTwin is properly installed and ROBOTWIN_PATH is correct."
            )

        with open(config_file, "r", encoding="utf-8") as f:
            args = yaml.load(f.read(), Loader=yaml.FullLoader)

        # Validate required fields
        required = ["embodiment", "camera", "data_type"]
        missing = [f for f in required if f not in args]
        if missing:
            raise RuntimeError(
                f"RoboTwin config missing required fields: {missing}\n"
                f"Config file: {config_file}"
            )

        # Process embodiment configs (like SimpleVLA-RL does)
        embodiment_type = args["embodiment"]
        embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

        with open(embodiment_config_path, "r", encoding="utf-8") as f:
            _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

        def get_embodiment_file(emb_type):
            robot_file = _embodiment_types[emb_type]["file_path"]
            if robot_file is None:
                raise RuntimeError(f"No embodiment file for {emb_type}")
            return robot_file

        def get_embodiment_config(robot_file):
            robot_config_file = os.path.join(robot_file, "config.yml")
            with open(robot_config_file, "r", encoding="utf-8") as f:
                embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
            return embodiment_args

        assets_path = os.getenv("ASSETS_PATH")

        if len(embodiment_type) == 1:
            args["left_robot_file"] = os.path.join(
                assets_path, get_embodiment_file(embodiment_type[0])
            )
            args["right_robot_file"] = os.path.join(
                assets_path, get_embodiment_file(embodiment_type[0])
            )
            args["dual_arm_embodied"] = True
        elif len(embodiment_type) == 3:
            args["left_robot_file"] = os.path.join(
                assets_path, get_embodiment_file(embodiment_type[0])
            )
            args["right_robot_file"] = os.path.join(
                assets_path, get_embodiment_file(embodiment_type[1])
            )
            args["embodiment_dis"] = embodiment_type[2]
            args["dual_arm_embodied"] = False
        else:
            raise RuntimeError("embodiment items should be 1 or 3")

        args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
        args["right_embodiment_config"] = get_embodiment_config(
            args["right_robot_file"]
        )

        if len(embodiment_type) == 1:
            embodiment_name = str(embodiment_type[0])
        else:
            embodiment_name = str(embodiment_type[0]) + "_" + str(embodiment_type[1])
        args["embodiment_name"] = embodiment_name

        # Process camera configs
        camera_config_path = os.path.join(CONFIGS_PATH, "_camera_config.yml")
        with open(camera_config_path, "r", encoding="utf-8") as f:
            _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

        head_camera_type = args["camera"].get("head_camera_type", "D435")
        args["head_camera_h"] = _camera_config[head_camera_type]["h"]
        args["head_camera_w"] = _camera_config[head_camera_type]["w"]

        # Add runtime-specific defaults
        args.setdefault("planner_backend", "mplib")  # H200 GPU compatibility
        args.setdefault("eval_mode", True)
        args.setdefault("eval_video_log", False)
        args.setdefault("render_freq", 0)

        return args

    def setup_tasks(self, timeout: int = 600):
        """Setup tasks for all environments (slow, run once at initialization).

        This pre-populates the task metadata cache by running setup_task for each env.
        Only needed if you want instructions available immediately.

        Args:
            timeout: Timeout in seconds for each env setup (default: 10 minutes)

        Raises:
            RuntimeError: If any environment fails to setup
        """
        logging.info(
            f"Setting up tasks for {self.num_envs} environments (this may take several minutes)..."
        )

        # For now, we assume all envs start with the same task (task_suite[0])
        # In future, could be customized per env
        default_task = self.task_suite[0] if self.task_suite else None

        if not default_task:
            logging.warning("No tasks in task_suite, skipping setup_tasks")
            return

        futures = {}
        for i in range(self.num_envs):
            future = self.thread_pool.submit(self.envs[i].setup_task, default_task)
            futures[i] = future

        for i in range(self.num_envs):
            try:
                futures[i].result(timeout=timeout)
                logging.info(f"SubEnv {i} setup completed")
            except Exception as e:
                logging.error(f"SubEnv {i} setup_task error: {e}")
                raise RuntimeError(f"SubEnv {i} setup_task error: {e}")

        logging.info("All environments setup completed")

    def reset(
        self,
        env_ids: Union[int, List[int]],
        task_ids: Union[int, List[int]],
        trial_ids: Union[int, List[int]],
    ):
        """Reset specified environments with Libero-style interface.

        Maps task_id -> task_name via self.task_suite and trial_id -> env_seed.

        Args:
            env_ids: Environment indices to reset
            task_ids: Task IDs (index into self.task_suite for task_name)
            trial_ids: Trial IDs (used as env_seed for reproducibility)

        Returns:
            List of observations from reset environments
        """
        # Normalize inputs to lists
        if isinstance(env_ids, int):
            env_ids = [env_ids]
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        if isinstance(trial_ids, int):
            trial_ids = [trial_ids]

        # Validate lengths match
        if not (len(env_ids) == len(task_ids) == len(trial_ids)):
            raise ValueError(
                f"env_ids ({len(env_ids)}), task_ids ({len(task_ids)}), "
                f"and trial_ids ({len(trial_ids)}) must have same length"
            )

        # Reset sequentially (global_lock in SubEnv.reset() serializes anyway)
        observations = []
        for idx, env_id in enumerate(env_ids):
            if 0 <= env_id < self.num_envs:
                # Map task_id to task_name via task_suite
                task_id = task_ids[idx]
                if not (0 <= task_id < len(self.task_suite)):
                    raise ValueError(
                        f"task_id {task_id} out of range [0, {len(self.task_suite)})"
                    )
                task_name = self.task_suite[task_id]

                # Use trial_id as env_seed
                env_seed = trial_ids[idx]

                logging.info(
                    f"Resetting SubEnv {env_id}: task={task_name}, seed={env_seed}"
                )
                try:
                    # Note: global_lock in SubEnv.reset() ensures sequential execution
                    # First reset takes: setup_task (~60-90s) + setup_demo (~60-90s)
                    # Subsequent resets: only setup_demo (~60-90s) due to caching
                    logging.info(
                        f"SubEnv {env_id} reset starting (this may take 2-4 minutes)..."
                    )
                    obs = self.envs[env_id].reset(
                        task_name=task_name, env_seed=env_seed
                    )
                    logging.info(f"SubEnv {env_id} reset completed successfully")
                    observations.append(obs)
                except Exception as e:
                    error_details = traceback.format_exc()
                    logging.error(f"SubEnv {env_id} reset error:\n{error_details}")
                    raise RuntimeError(
                        f"SubEnv {env_id} reset error: {type(e).__name__}: {e}\n"
                        f"Full traceback:\n{error_details}"
                    )

        return observations

    def get_obs(self, env_ids: Optional[Union[int, List[int]]] = None):
        """Get current observations from specified environments.

        Args:
            env_ids: Environment indices (None = all environments)

        Returns:
            List of observations
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        elif isinstance(env_ids, int):
            env_ids = [env_ids]

        observations = []
        for env_id in env_ids:
            if 0 <= env_id < self.num_envs:
                obs = self.envs[env_id].get_obs()
                observations.append(obs)

        return observations

    def step(self, actions, env_ids: Optional[Union[int, List[int]]] = None):
        """Execute one step in specified environments.

        Args:
            actions: Actions for each environment
            env_ids: Environment indices to step (None = all environments)

        Returns:
            Tuple of (observations, rewards, terminated, truncated, infos)
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        elif isinstance(env_ids, int):
            env_ids = [env_ids]

        # Convert actions to numpy if needed
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        # Submit step tasks
        step_futures = {}
        for i, env_id in enumerate(env_ids):
            if 0 <= env_id < self.num_envs:
                future = self.thread_pool.submit(self.envs[env_id].step, actions[i])
                step_futures[env_id] = future

        # Collect results
        observations = []
        rewards = []
        terminated = []
        truncated = []
        infos = []

        for env_id in env_ids:
            if env_id in step_futures:
                try:
                    result = step_futures[env_id].result(timeout=120)
                    observations.append(result["obs"])
                    rewards.append(result["reward"])
                    terminated.append(result["terminated"])
                    truncated.append(result["truncated"])
                    infos.append(result["info"])
                except Exception as e:
                    logging.error(f"SubEnv {env_id} step error: {e}")
                    raise RuntimeError(f"SubEnv {env_id} step error: {e}")

        return observations, rewards, terminated, truncated, infos

    def close(self, clear_cache: bool = True):
        """Close all environments and free resources.

        Args:
            clear_cache: Whether to clear rendering cache
        """
        # Close all sub-environments
        for env in self.envs:
            env.close(clear_cache=clear_cache)

        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)

        # Clear environment list
        self.envs = []

        # Force garbage collection
        if clear_cache:
            gc.collect()
            torch.cuda.empty_cache()
