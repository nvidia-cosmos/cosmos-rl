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
from libero.libero import get_libero_path
import torch

from cosmos_rl.simulators.libero.venv import ReconfigureSubprocEnv
from cosmos_rl.simulators.libero.utils import (
    get_benchmark_overridden,
    get_libero_dummy_action,
    quat2axisangle,
)


@dataclass
class EnvStates:
    env_idx: int
    task_id: int = -1
    trial_id: int = -1
    active: bool = True
    complete: bool = False
    step: int = 0

    current_obs: Optional[Any] = None
    do_validation: bool = False
    valid_pixels: Optional[Dict[str, Any]] = None


class LiberoEnvWrapper(gym.Env):
    def __init__(self, cfg, *args, **kwargs):
        self.num_envs = getattr(cfg, "num_envs", 1)
        self.seed = getattr(cfg, "seed", 0)
        self.task_suite_name = getattr(cfg, "task_suite_name", "libero_all")
        self.height = getattr(cfg, "height", 256)
        self.width = getattr(cfg, "width", 256)
        self.max_steps = getattr(cfg, "max_steps", 512)

        self.task_suite = get_benchmark_overridden(self.task_suite_name)()
        self.env_states = [EnvStates(env_idx=i) for i in range(self.num_envs)]

        self._init_env()

    def _init_env(self):
        # lazy init the env fns
        dummy_env_fns = []
        for _ in range(self.num_envs):

            def dummy_env_fn():
                return None

            dummy_env_fns.append(dummy_env_fn)
        self.env = ReconfigureSubprocEnv(dummy_env_fns)

    def _get_fn_params(self, task_id: int):
        task = self.task_suite.get_task(task_id)
        task_bddl_file = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        fn_params = {
            "camera_heights": self.height,
            "camera_widths": self.width,
            "bddl_file_name": task_bddl_file,
            "seed": self.seed,
        }
        return task.language, fn_params

    def _extract_image_and_state(self, obs):
        full_images = []
        wrist_images = []
        states = []
        for env_id in range(len(obs)):
            full_images.append(obs[env_id]["agentview_image"][::-1, ::-1])
            wrist_images.append(obs[env_id]["robot0_eye_in_hand_image"][::-1, ::-1])
            states.append(
                np.concatenate(
                    [
                        obs[env_id]["robot0_eef_pos"],
                        quat2axisangle(obs[env_id]["robot0_eef_quat"]),
                        obs[env_id]["robot0_gripper_qpos"],
                    ]
                )
            )
        return {
            "full_images": np.stack(full_images),
            "wrist_images": np.stack(wrist_images),
            "states": np.stack(states),
        }

    def _reconfigure(
        self, env_ids: List[int], task_ids: List[int], trial_ids: List[int]
    ):
        env_fn_params = [None for _ in range(len(env_ids))]
        task_descriptions = [None for _ in range(len(env_ids))]
        init_state = [None for _ in range(len(env_ids))]

        for i, env_id in enumerate(env_ids):
            desp, fn_params = self._get_fn_params(task_ids[i])
            task_descriptions[i] = desp
            env_fn_params[i] = fn_params
            n_trial_ids = len(self.task_suite.get_task_init_states(task_ids[i]))
            init_state[i] = self.task_suite.get_task_init_states(task_ids[i])[
                trial_ids[i] % n_trial_ids
            ]

            self.env_states[env_id] = EnvStates(
                env_idx=env_id,
                task_id=task_ids[i],
                trial_id=trial_ids[i],
            )
        self.env.reconfigure_env_fns(env_fn_params, env_ids)
        self.env.set_init_state(init_state, env_ids)

        dummy_actions = get_libero_dummy_action(len(env_ids))
        for _ in range(15):
            obs, _, _, _ = self.env.step(dummy_actions, env_ids)

        images_and_states = self._extract_image_and_state(obs)
        for i, env_id in enumerate(env_ids):
            self.env_states[env_id].current_obs = {
                "full_images": images_and_states["full_images"][i],
                "wrist_images": images_and_states["wrist_images"][i],
                "states": images_and_states["states"][i],
            }
        return images_and_states, task_descriptions

    def reset(
        self,
        env_ids: Union[int, List[int]],
        task_ids: Union[int, List[int]],
        trial_ids: Union[int, List[int]],
        do_validataion: Union[bool, List[bool]],
    ):
        if isinstance(env_ids, int):
            env_ids = [env_ids]
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        if isinstance(trial_ids, int):
            trial_ids = [trial_ids]
        if isinstance(do_validataion, bool):
            do_validataion = [do_validataion]

        images_and_states, task_descriptions = self._reconfigure(
            env_ids, task_ids, trial_ids
        )
        for i, env_id in enumerate(env_ids):
            self.env_states[env_id].do_validation = do_validataion[i]
            if do_validataion[i]:
                self.env_states[env_id].valid_pixels = {
                    "full_images": [],
                    "wrist_images": [],
                }

        return images_and_states, task_descriptions

    def step(self, env_ids: List[int], action):
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        active_indices = [
            i for i, env_id in enumerate(env_ids) if self.env_states[env_id].active
        ]
        if active_indices:
            active_env_ids = [env_ids[i] for i in active_indices]
            active_action = action[active_indices]

            obs, reward, done, info = self.env.step(active_action, active_env_ids)
            images_and_states = self._extract_image_and_state(obs)
            for i, env_id in enumerate(active_env_ids):
                self.env_states[env_id].step += 1
                if done[i] or self.env_states[env_id].step >= self.max_steps:
                    self.env_states[env_id].complete = done[i]
                    self.env_states[env_id].active = False
                for k, v in images_and_states.items():
                    self.env_states[env_id].current_obs[k] = v[i]

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

    def get_valid_pixels(self, env_ids: List[int]):
        results = {}
        for keys in ["full_images", "wrist_images"]:
            results[keys] = np.stack(
                [self.env_states[env_id].valid_pixels[keys] for env_id in env_ids]
            )
        return results
