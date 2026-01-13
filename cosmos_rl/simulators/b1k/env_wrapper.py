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

import json
import os
import gymnasium as gym
import numpy as np
import torch
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

# Disable torch compilation to avoid typing_extensions compatibility issues
# This must be done before importing omnigibson
torch._dynamo.config.disable = True
import torch._dynamo

torch._dynamo.reset()

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.tasks.behavior_task import BehaviorTask
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.learning.utils.eval_utils import (
    TASK_NAMES_TO_INDICES,
    PROPRIOCEPTION_INDICES,
    generate_basic_environment_config,
)
from gello.robots.sim_robot.og_teleop_utils import (
    load_available_tasks,
    generate_robot_config,
)

gm.HEADLESS = True
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True

from cosmos_rl.simulators.utils import save_rollout_video
from cosmos_rl.simulators.b1k.utils import get_b1k_task_descriptions
from .venv import VectorEnvironment


@dataclass
class EnvStates:
    env_idx: int
    task_id: str = ""
    trial_id: int = 0
    active: bool = True
    complete: bool = False
    step: int = 0
    max_steps: int = 0

    current_obs: Optional[Any] = None
    do_validation: bool = False
    valid_pixels: Optional[Dict[str, Any]] = None
    task_description: Optional[str] = None


class B1KEnvWrapper(gym.Env):
    def __init__(self, cfg, *args, **kwargs):
        self.cfg = cfg
        self.num_envs = getattr(cfg, "num_envs", 1)
        self.env_states = [EnvStates(env_idx=i) for i in range(self.cfg.num_envs)]
        self.height = getattr(cfg, "height", 256)
        self.width = getattr(cfg, "width", 256)

        self.task_cfgs = load_available_tasks()
        self.total_tasks = len(self.task_cfgs)
        self.task_names = ["" for _ in range(self.total_tasks)]
        self.task_lengths = [0 for _ in range(self.total_tasks)]
        for task_name, idx in TASK_NAMES_TO_INDICES.items():
            self.task_names[idx] = task_name

        with open(
            os.path.join(
                gm.DATA_PATH,
                "2025-challenge-task-instances",
                "metadata",
                "episodes.jsonl",
            ),
            "r",
        ) as f:
            episodes = [json.loads(line) for line in f]
            human_stats_cnt = [0 for _ in range(self.total_tasks)]
        for episode in episodes:
            task_idx = int(episode["episode_index"] // 1e4)
            self.task_lengths[task_idx] += episode["length"]
            human_stats_cnt[task_idx] += 1

        for task_idx, cnt in enumerate(human_stats_cnt):
            if cnt > 0:
                self.task_lengths[task_idx] = int(self.task_lengths[task_idx] / cnt)
            else:
                self.task_lengths[task_idx] = 2000

        self.task_description_map = get_b1k_task_descriptions()
        self.default_cfg, _ = self._load_env_cfg(0)
        self.env = VectorEnvironment(
            num_envs=self.cfg.num_envs, config=self.default_cfg
        )

    def _load_env_cfg(self, task_id: int):
        task_name = self.task_names[task_id]
        task_cfg = self.task_cfgs[task_name][0]
        env_cfg = generate_basic_environment_config(
            task_name=task_name, task_cfg=task_cfg
        )
        env_cfg["robots"] = [
            generate_robot_config(
                task_name=task_name,
                task_cfg=task_cfg,
            )
        ]
        env_cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb"]
        env_cfg["robots"][0]["proprio_obs"] = list(
            PROPRIOCEPTION_INDICES["R1Pro"].keys()
        )
        env_cfg["task"]["termination_config"]["max_steps"] = int(
            self.task_lengths[task_id] * 2
        )
        env_cfg["task"]["include_obs"] = False
        task_description = self.task_description_map[task_name]
        return env_cfg, task_description

    def _load_task_instance(self, task_id, instance_id):
        env_cfg, task_description = self._load_env_cfg(task_id)
        scene_model = env_cfg["scene"]["scene_model"]
        tro_filename = BehaviorTask.get_cached_activity_scene_filename(
            scene_model=scene_model,
            activity_name=env_cfg["task"]["activity_name"],
            activity_definition_id=env_cfg["task"]["activity_definition_id"],
            activity_instance_id=instance_id,
        )
        tro_file_path = os.path.join(
            get_task_instance_path(scene_model),
            f"json/{scene_model}_task_{env_cfg['task']['activity_name']}_instances/{tro_filename}-tro_state.json",
        )
        with open(tro_file_path, "r") as f:
            tro_state = json.load(f)
        return env_cfg, task_description, tro_state

    def _extract_image_and_state(self, obs_list):
        full_images = []
        wrist_images = []
        for i, obs in enumerate(obs_list):
            for data in obs.values():
                assert isinstance(data, dict)
                for k, v in data.items():
                    if "left_realsense_link:Camera:0" in k:
                        left_image = v["rgb"]
                    elif "right_realsense_link:Camera:0" in k:
                        right_image = v["rgb"]
                    elif "zed_link:Camera:0" in k:
                        zed_image = v["rgb"]

            full_images.append(zed_image)
            wrist_images.append(torch.stack([left_image, right_image], axis=0))

        # full_images: [N_ENV, H, W, C]
        # wrist_images: [N_ENV, N_IMG, H, W, C]
        return {
            "full_images": np.stack(full_images, axis=0),
            "wrist_images": np.stack(wrist_images, axis=0),
        }

    def reset(
        self,
        env_ids: List[int],
        task_ids: List[int],
        trial_ids: List[int],
        do_validation: bool = False,
    ):
        env_cfgs, tro_states = [], []
        for env_id, task_id, trial_id in zip(env_ids, task_ids, trial_ids):
            env_cfg, task_description, tro_state = self._load_task_instance(
                task_id, trial_id
            )
            # og_cfg, task_description = self._load_full_config(task_id)
            self.env_states[env_id].task_id = task_id
            self.env_states[env_id].trial_id = trial_id
            self.env_states[env_id].task_description = task_description
            self.env_states[env_id].active = True
            self.env_states[env_id].complete = False
            self.env_states[env_id].step = 0
            self.env_states[env_id].do_validation = do_validation
            self.env_states[env_id].max_steps = self.task_lengths[task_id] * 2

            env_cfgs.append(env_cfg)
            tro_states.append(tro_state)

        raw_obs, _ = self.env.reset(env_ids, env_cfgs, tro_states)

        images_and_states = self._extract_image_and_state(raw_obs)

        # Store initial observations in env states
        for i, env_id in enumerate(env_ids):
            self.env_states[env_id].current_obs = {
                "full_images": images_and_states["full_images"][i],
                "wrist_images": images_and_states["wrist_images"][i],
                "states": np.zeros(0),  # Placeholder, adjust if you have state info
            }
            if do_validation:
                self.env_states[env_id].valid_pixels = {
                    "full_images": [images_and_states["full_images"][i]],
                    "wrist_images": [images_and_states["wrist_images"][i]],
                }
            else:
                self.env_states[env_id].valid_pixels = None

        task_descriptions = [
            self.env_states[env_id].task_description for env_id in env_ids
        ]
        return images_and_states, task_descriptions

    def step(self, env_ids: List[int], actions: Any):
        raw_obs, rewards, terminations, truncations, infos = self.env.step(
            env_ids, actions
        )
        images_and_states = self._extract_image_and_state(raw_obs)

        for i, env_id in enumerate(env_ids):
            self.env_states[env_id].step += 1
            if terminations[i]:
                self.env_states[env_id].complete = True
                self.env_states[env_id].active = False
            if (
                truncations[i]
                or self.env_states[env_id].step >= self.env_states[env_id].max_steps
            ):
                self.env_states[env_id].complete = False
                self.env_states[env_id].active = False

            # Update current observations
            if self.env_states[env_id].current_obs is None:
                self.env_states[env_id].current_obs = {}
            for k, v in images_and_states.items():
                self.env_states[env_id].current_obs[k] = v[i]
            # Add states placeholder if not present
            if "states" not in self.env_states[env_id].current_obs:
                self.env_states[env_id].current_obs["states"] = np.zeros(0)

            if self.env_states[env_id].do_validation:
                for img_key in ["full_images", "wrist_images"]:
                    self.env_states[env_id].valid_pixels[img_key].append(
                        images_and_states[img_key][i]
                    )

        completes = np.array([self.env_states[env_id].complete for env_id in env_ids])
        active = np.array([self.env_states[env_id].active for env_id in env_ids])
        finish_steps = np.array([self.env_states[env_id].step for env_id in env_ids])

        full_images_and_states = {}
        for key in ["full_images", "wrist_images", "states"]:
            full_images_and_states[key] = np.stack(
                [self.env_states[env_id].current_obs[key] for env_id in env_ids]
            )

        return {
            **full_images_and_states,
            "complete": completes,
            "active": active,
            "finish_step": finish_steps,
        }

    def chunk_step(self, env_ids: List[int], actions: torch.Tensor):
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        steps = actions.shape[1]
        for step in range(steps):
            results = self.step(env_ids, actions[:, step])
        return results

    def get_env_states(self, env_ids: List[int]):
        return [self.env_states[env_id] for env_id in env_ids]

    def save_validation_videos(self, rollout_dir: str, env_ids: List[int]):
        for env_id in env_ids:
            state = self.env_states[env_id]
            if not state.do_validation:
                continue
            # Use task name from the environment state
            task_name = self.task_names[state.task_id]
            save_rollout_video(
                state.valid_pixels["full_images"],
                rollout_dir,
                f"{task_name}_env{env_id}_trial{state.trial_id}",
                state.complete,
            )

    def close(self):
        """Clean up the environment and its scenes.

        This performs proper cleanup by removing all objects and systems from scenes,
        then stopping the simulator. We do this in close() where it belongs, but with
        careful handling to avoid viewport race conditions.
        """
        if not hasattr(og, "sim") or og.sim is None:
            return

        # Stop the simulator to halt rendering before cleanup
        if not og.sim.is_stopped():
            og.sim.stop()

        # Now safely clear all scenes
        for scene in og.sim.scenes:
            scene.clear()
        # Clear the scenes list
        og.sim.scenes.clear()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()
