# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import omnigibson as og
from typing import List, Any
from omnigibson.learning.utils.eval_utils import ROBOT_CAMERA_NAMES, HEAD_RESOLUTION, WRIST_RESOLUTION


class VectorEnvironment:
    def __init__(self, num_envs, config):
        self.num_envs = num_envs
        if og.sim is not None:
            og.sim.stop()

        # First we create the environments. We can't let DummyVecEnv do this for us because of the play call
        # needing to happen before spaces are available for it to read things from.
        self.envs = [og.Environment(config, in_vec_env=True) for _ in range(num_envs)]

        # Play, and finish loading all the envs
        og.sim.play()
        for env in self.envs:
            env.post_play_load()
            self._apply_camera_resolutions(env)

    def _apply_camera_resolutions(self, env):
        """Apply the correct camera resolutions to match eval.py behavior.
        
        This sets:
        - Head camera: 720x720 with 40.0 horizontal aperture
        - Wrist cameras: 480x480
        """
        robot = env.robots[0]
        for camera_id, camera_name in ROBOT_CAMERA_NAMES["R1Pro"].items():
            sensor_name = camera_name.split("::")[1]
            if camera_id == "head":
                robot.sensors[sensor_name].horizontal_aperture = 40.0
                robot.sensors[sensor_name].image_height = HEAD_RESOLUTION[0]
                robot.sensors[sensor_name].image_width = HEAD_RESOLUTION[1]
            else:
                robot.sensors[sensor_name].image_height = WRIST_RESOLUTION[0]
                robot.sensors[sensor_name].image_width = WRIST_RESOLUTION[1]
        # Reload observation space to reflect the new camera resolutions
        env.load_observation_space()

    def step(self, env_ids: List[int], actions: Any):
        observations, rewards, terminates, truncates, infos = [], [], [], [], []
        for i, env_id in enumerate(env_ids):
            self.envs[env_id]._pre_step(actions[i])
        og.sim.step()
        for i, env_id in enumerate(env_ids):
            obs, reward, terminated, truncated, info = self.envs[env_id]._post_step(
                actions[i]
            )
            observations.append(obs)
            rewards.append(reward)
            terminates.append(terminated)
            truncates.append(truncated)
            infos.append(info)
        return observations, rewards, terminates, truncates, infos

    def reset(
        self, env_ids: List[int], env_cfgs: List[Any], tro_states: List[Any], **kwargs
    ):
        observations, infos = [], []
        og.sim.stop()
        for env_id, env_cfg in zip(env_ids, env_cfgs):
            env = self.envs[env_id]
            cur_scene_model = env.task.scene_name
            new_scene_model = env_cfg["scene"]["scene_model"]
            if cur_scene_model != new_scene_model:
                env.scene_config["model"] = env_cfg["scene"]["scene_model"]
                env.task_config.clear()
                env.task_config.update(env_cfg["task"])
                env.load()
        og.sim.play()

        for env_id, tro_state in zip(env_ids, tro_states):
            env = self.envs[env_id]
            env.post_play_load()
            self._apply_camera_resolutions(env)

            for tro_key, tro_state in tro_state.items():
                if tro_key == "robot_poses":
                    presampled_robot_poses = tro_state
                    robot_pos = presampled_robot_poses["R1Pro"][0]["position"]
                    robot_quat = presampled_robot_poses["R1Pro"][0]["orientation"]
                    robot = env.scene.object_registry("name", "robot_r1")
                    robot.set_position_orientation(robot_pos, robot_quat)
                    # Write robot poses to scene metadata
                    env.scene.write_task_metadata(key=tro_key, data=tro_state)
                else:
                    env.task.object_scope[tro_key].load_state(
                        tro_state, serialized=False
                    )

        for _ in range(25):
            og.sim.step_physics()
            for entity in env.task.object_scope.values():
                if not entity.is_system and entity.exists:
                    entity.keep_still()
        for env_id in env_ids:
            env = self.envs[env_id]
            env.scene.update_initial_file()
            env.scene.reset()
            
        # Warm up rendering to sync camera buffers
        # OmniGibson rendering is async and takes 3-4 render calls to sync
        for _ in range(4):
            og.sim.render()
            
        for env_id in env_ids:
            env = self.envs[env_id]
            obs, info = env.get_obs()
            observations.append(obs)
            infos.append(info)
        return observations, infos

    def close(self):
        pass

    def __len__(self):
        return self.num_envs
