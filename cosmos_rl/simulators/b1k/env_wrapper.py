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
import yaml
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
from omnigibson.envs import VectorEnvironment
from omnigibson.macros import gm

gm.HEADLESS = True
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True

from cosmos_rl.simulators.utils import save_rollout_video


@dataclass
class EnvStates:
    env_idx: int
    task_name: str = ""
    active: bool = True
    complete: bool = False
    step: int = 0

    current_obs: Optional[Any] = None
    do_validation: bool = False
    valid_pixels: Optional[Dict[str, Any]] = None


class B1KEnvWrapper(gym.Env):
    def __init__(self, cfg, *args, **kwargs):
        self.cfg = cfg
        self._init_env()
        self.env_states = [EnvStates(env_idx=i) for i in range(self.cfg.num_envs)]

    def _init_env(self):
        og_cfg_file = os.path.join(og.example_config_path, "r1pro_behavior.yaml")
        with open(og_cfg_file, "r") as f:
            self.og_cfg = yaml.load(f, Loader=yaml.FullLoader)

        task_description_path = os.path.join(
            os.path.dirname(__file__), "behavior_task.jsonl"
        )
        with open(task_description_path, "r") as f:
            text = f.read()
            task_description = [json.loads(x) for x in text.strip().split("\n") if x]

        self.task_description_map = {
            task_description[i]["task_name"]: task_description[i]["task"]
            for i in range(len(task_description))
        }
        self.task_description = self.task_description_map[self.cfg.task_name]
        self.og_cfg["task"]["activity_name"] = self.cfg.task_name
        self.og_cfg["robots"][0]["sensor_config"]["VisionSensor"]["sensor_kwargs"][
            "image_height"
        ] = 768
        self.og_cfg["robots"][0]["sensor_config"]["VisionSensor"]["sensor_kwargs"][
            "image_width"
        ] = 768
        self.env = VectorEnvironment(num_envs=self.cfg.num_envs, config=self.og_cfg)

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

    def reset(self, do_validation: bool = False):
        raw_obs, _ = self.env.reset()

        images_and_states = self._extract_image_and_state(raw_obs)
        task_descriptions = [self.task_description for i in range(self.cfg.num_envs)]

        for i in range(self.cfg.num_envs):
            # Store individual observation for each env
            self.env_states[i].current_obs = {
                "full_images": images_and_states["full_images"][i],
                "wrist_images": images_and_states["wrist_images"][i],
            }
            self.env_states[i].task_name = self.cfg.task_name
            self.env_states[i].active = True
            self.env_states[i].complete = False
            self.env_states[i].step = 0
            self.env_states[i].do_validation = do_validation

            # Initialize validation pixels if needed
            if do_validation:
                self.env_states[i].valid_pixels = {
                    "full_images": [],
                    "wrist_images": [],
                }
            else:
                self.env_states[i].valid_pixels = None

        return images_and_states, task_descriptions

    def step(self, action):
        raw_obs, rewards, terminations, truncations, infos = self.env.step(action)
        images_and_states = self._extract_image_and_state(raw_obs)

        # Update all environment states
        for i in range(self.cfg.num_envs):
            # Store individual observation for each env
            self.env_states[i].current_obs = {
                "full_images": images_and_states["full_images"][i],
                "wrist_images": images_and_states["wrist_images"][i],
            }
            self.env_states[i].step += 1

            # Check for completion
            if terminations[i] or truncations[i]:
                self.env_states[i].complete = True
                self.env_states[i].active = False

            # Collect validation pixels if enabled
            if self.env_states[i].do_validation:
                for img_key in ["full_images", "wrist_images"]:
                    self.env_states[i].valid_pixels[img_key].append(
                        images_and_states[img_key][i]
                    )

        # Prepare return values for all environments
        env_ids = list(range(self.cfg.num_envs))
        completes = np.array([self.env_states[env_id].complete for env_id in env_ids])
        active = np.array([self.env_states[env_id].active for env_id in env_ids])
        finish_steps = np.array([self.env_states[env_id].step for env_id in env_ids])

        return {
            **images_and_states,
            "complete": completes,
            "active": active,
            "finish_step": finish_steps,
        }

    def chunk_step(self, actions: torch.Tensor):
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        steps = actions.shape[1]
        for step in range(steps):
            results = self.step(actions[:, step])
        return results

    def get_env_states(self, env_ids: List[int]):
        return [self.env_states[env_id] for env_id in env_ids]

    def save_validation_videos(self, rollout_dir: str, env_ids: List[int]):
        for env_id in env_ids:
            state = self.env_states[env_id]
            if not state.do_validation:
                continue
            task_name = f"{self.cfg.task_name}"
            save_rollout_video(
                state.valid_pixels["full_images"],
                rollout_dir,
                task_name,
                state.complete,
            )

    def close(self):
        """Clean up the environment and its scenes.
        
        This performs proper cleanup by removing all objects and systems from scenes,
        then stopping the simulator. We do this in close() where it belongs, but with
        careful handling to avoid viewport race conditions.
        """
        try:
            if hasattr(self, "env") and self.env is not None:
                # VectorEnvironment.close() is a no-op, but call it for consistency
                if hasattr(self.env, "close"):
                    self.env.close()
                
                # Now clean up the scenes to prevent accumulation
                if og.sim is not None and hasattr(og.sim, 'scenes') and len(og.sim.scenes) > 0:
                    # Stop the simulator to halt rendering before cleanup
                    if not og.sim.is_stopped():
                        og.sim.stop()
                    
                    # Now safely clear all scenes
                    for scene in og.sim.scenes:
                        try:
                            # Remove all objects from the scene
                            if hasattr(scene, 'objects') and len(scene.objects) > 0:
                                og.sim.batch_remove_objects(list(scene.objects))
                            
                            # Clear all systems
                            if hasattr(scene, 'active_systems') and len(scene.active_systems) > 0:
                                for system_name in list(scene.active_systems.keys()):
                                    try:
                                        scene.clear_system(system_name=system_name)
                                    except Exception:
                                        pass  # Some systems may already be cleared
                        except Exception as e:
                            print(f"Warning during scene cleanup: {e}")
                    
                    # Clear the scenes list
                    og.sim.scenes.clear()
                
                # Clear the reference to allow garbage collection
                self.env = None
        except Exception as e:
            print(f"Warning: Error during environment cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()
