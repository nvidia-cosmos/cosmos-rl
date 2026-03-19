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

import gymnasium as gym
import numpy as np
import torch
from typing import Union, List, Any, Optional, Dict
from dataclasses import dataclass

from cosmos_rl.simulators.utils import save_rollout_video
from cosmos_rl.utils.logging import logger


MANISKILL_TASK_SUITES = {
    "bridge": [
        "PutCarrotOnPlateInScene-v1",
        "PutSpoonOnTableClothInScene-v1",
        "StackGreenCubeOnYellowCubeBakedTexInScene-v1",
        "PutEggplantInBasketScene-v1",
    ],
    "tabletop": [
        "PickCube-v1",
        "StackCube-v1",
        "PushCube-v1",
        "PlugCharger-v1",
        "PegInsertionSide-v1",
    ],
    "mobile": [
        "OpenCabinetDoor-v1",
        "OpenCabinetDrawer-v1",
        "TurnFaucet-v1",
    ],
}

# Priority order for auto-detecting the main (third-person) camera
_MAIN_CAMERA_CANDIDATES = [
    "overhead_camera",
    "3rd_view_camera",
    "base_camera",
    "front_camera",
]

_WRIST_CAMERA_CANDIDATES = [
    "hand_camera",
    "wrist_camera",
]


def get_maniskill_task_suite(suite_name: str) -> List[str]:
    """Resolve a task suite name to a list of ManiSkill3 gym env IDs.

    Args:
        suite_name: A key in MANISKILL_TASK_SUITES, or a comma-separated
                    list of gym env IDs (e.g. "PickCube-v1,StackCube-v1").
    """
    if suite_name in MANISKILL_TASK_SUITES:
        return list(MANISKILL_TASK_SUITES[suite_name])
    return [s.strip() for s in suite_name.split(",")]


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


class ManiSkillEnvWrapper(gym.Env):
    """Wrapper that adapts ManiSkill3 vectorized environments to the
    cosmos-rl EnvWrapper interface (same as RoboTwin/Libero wrappers).

    ManiSkill3 natively supports GPU-accelerated vectorized simulation via
    ``gym.make(env_id, num_envs=N)``, so no separate VectorEnv/SubEnv layer
    is needed.  Because all parallel envs inside one ``gym.make`` call share
    the *same* task, this wrapper asserts **full reset/step** – i.e. every
    call must address all ``num_envs`` environments with a single, uniform
    ``task_id``.
    """

    def __init__(self, cfg, *args, **kwargs):
        self.num_envs = getattr(cfg, "num_envs", 1)
        self.seed = getattr(cfg, "seed", 0)
        self.max_steps = getattr(cfg, "max_steps", 200)
        self.obs_mode = getattr(cfg, "obs_mode", "rgb")
        self.shader = getattr(cfg, "shader", "default")
        self.env_kwargs: dict = getattr(cfg, "env_kwargs", {})

        task_suite_name = getattr(cfg, "task_suite_name", "PickCube-v1")
        self.task_suite = get_maniskill_task_suite(task_suite_name)

        # Camera names – auto-detected on first env creation if left as None.
        self.main_camera: Optional[str] = getattr(cfg, "main_camera", None)
        self.wrist_camera: Optional[str] = getattr(cfg, "wrist_camera", None)

        self.env = None
        self._current_task_env_id: Optional[str] = None
        self.env_states = [EnvStates(env_idx=i) for i in range(self.num_envs)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_env(self, env_id: str):
        """Create (or recreate) the ManiSkill3 vectorized env for *env_id*."""
        if self.env is not None and self._current_task_env_id == env_id:
            return

        if self.env is not None:
            self.env.close()
            self.env = None

        import mani_skill.envs  # noqa: F401 – registers ManiSkill3 gym envs

        make_kwargs: dict = dict(
            obs_mode=self.obs_mode,
            num_envs=self.num_envs,
            **self.env_kwargs,
        )
        if self.shader != "default":
            make_kwargs.setdefault("sensor_configs", {})["shader_pack"] = self.shader

        logger.info(f"Creating ManiSkill env: {env_id}  num_envs={self.num_envs}")
        self.env = gym.make(env_id, **make_kwargs)
        self._current_task_env_id = env_id

        if self.main_camera is None:
            self._detect_cameras()

    def _detect_cameras(self):
        """Run a throwaway reset to discover available camera names."""
        obs, _ = self.env.reset()
        camera_names = list(obs["sensor_data"].keys())

        self.main_camera = camera_names[0]
        for name in _MAIN_CAMERA_CANDIDATES:
            if name in camera_names:
                self.main_camera = name
                break

        for name in _WRIST_CAMERA_CANDIDATES:
            if name in camera_names:
                self.wrist_camera = name
                break

        logger.info(
            f"Auto-detected cameras – main: {self.main_camera}, wrist: {self.wrist_camera}"
        )

    def _extract_image_and_state(self, obs) -> Dict[str, Optional[np.ndarray]]:
        """Convert ManiSkill3 torch observation into numpy arrays.

        Returns dict with ``full_images`` (B,H,W,C uint8),
        ``wrist_images`` (same or None), ``states`` (B,D float or None).
        """
        full_images = obs["sensor_data"][self.main_camera]["rgb"]
        if isinstance(full_images, torch.Tensor):
            full_images = full_images.cpu().numpy()
        full_images = full_images.astype(np.uint8)

        wrist_images = None
        if self.wrist_camera and self.wrist_camera in obs.get("sensor_data", {}):
            wi = obs["sensor_data"][self.wrist_camera]["rgb"]
            if isinstance(wi, torch.Tensor):
                wi = wi.cpu().numpy()
            wrist_images = wi.astype(np.uint8)

        states = None
        if "agent" in obs:
            parts = []
            for key in sorted(obs["agent"].keys()):
                val = obs["agent"][key]
                if isinstance(val, torch.Tensor):
                    val = val.cpu().numpy()
                if isinstance(val, np.ndarray) and val.ndim >= 1:
                    parts.append(val.reshape(val.shape[0], -1))
            if parts:
                states = np.concatenate(parts, axis=-1)

        return {
            "full_images": full_images,
            "wrist_images": wrist_images,
            "states": states,
        }

    def _get_language_instructions(self) -> List[str]:
        try:
            instructions = self.env.unwrapped.get_language_instruction()
            if isinstance(instructions, str):
                return [instructions] * self.num_envs
            return list(instructions)
        except (AttributeError, NotImplementedError):
            return [self._current_task_env_id or ""] * self.num_envs

    @property
    def _device(self) -> torch.device:
        return self.env.unwrapped.device

    # ------------------------------------------------------------------
    # Public interface  (mirrors RoboTwin / Libero wrappers)
    # ------------------------------------------------------------------

    def reset(
        self,
        env_ids: Union[int, List[int]],
        task_ids: Union[int, List[int]],
        trial_ids: Union[int, List[int]],
        do_validation: Union[bool, List[bool]],
    ):
        if isinstance(env_ids, int):
            env_ids = [env_ids]
        if isinstance(task_ids, int):
            task_ids = [task_ids] * len(env_ids)
        if isinstance(trial_ids, int):
            trial_ids = [trial_ids] * len(env_ids)
        if isinstance(do_validation, bool):
            do_validation = [do_validation] * len(env_ids)

        assert len(env_ids) == self.num_envs, (
            f"ManiSkill wrapper requires full reset "
            f"(got {len(env_ids)}, expected {self.num_envs})"
        )
        assert len(set(task_ids)) == 1, (
            f"ManiSkill vectorized env requires uniform task_ids, got {set(task_ids)}"
        )

        task_id = task_ids[0]
        assert 0 <= task_id < len(self.task_suite), (
            f"task_id {task_id} out of range [0, {len(self.task_suite)})"
        )
        env_id = self.task_suite[task_id]

        self._create_env(env_id)

        seed = self.seed + trial_ids[0]
        episode_ids = torch.tensor(
            [self.seed + tid for tid in trial_ids], dtype=torch.int64
        )
        obs, _ = self.env.reset(seed=seed, options={"episode_id": episode_ids})

        instructions = self._get_language_instructions()
        images_and_states = self._extract_image_and_state(obs)

        task_descriptions = []
        for i, eid in enumerate(env_ids):
            instruction = instructions[i] if i < len(instructions) else ""
            self.env_states[eid] = EnvStates(
                env_idx=eid,
                task_id=task_ids[i],
                trial_id=trial_ids[i],
                active=True,
                complete=False,
                step=0,
                language=instruction,
                do_validation=do_validation[i],
            )

            self.env_states[eid].current_obs = {
                k: v[i] if v is not None else None for k, v in images_and_states.items()
            }

            if do_validation[i]:
                self.env_states[eid].valid_pixels = {
                    "full_images": [],
                    "wrist_images": [],
                }
                if images_and_states["full_images"] is not None:
                    self.env_states[eid].valid_pixels["full_images"].append(
                        images_and_states["full_images"][i]
                    )

            task_descriptions.append(instruction)

        return images_and_states, task_descriptions

    def reset_async(
        self,
        env_ids: Union[int, List[int]],
        task_ids: Union[int, List[int]],
        trial_ids: Union[int, List[int]],
        do_validation: Union[bool, List[bool]],
    ):
        raise NotImplementedError(
            "ManiSkill wrapper does not support async reset. Use reset() instead."
        )

    def reset_wait(self, env_ids: List[int]):
        raise NotImplementedError(
            "ManiSkill wrapper does not support async reset. Use reset() instead."
        )

    def step(self, env_ids: List[int], action):
        assert len(env_ids) == self.num_envs, (
            f"ManiSkill wrapper requires full step "
            f"(got {len(env_ids)}, expected {self.num_envs})"
        )

        if isinstance(action, torch.Tensor):
            action = action.detach().to(self._device)
        elif isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float().to(self._device)

        obs, reward, terminated, truncated, info = self.env.step(action)
        images_and_states = self._extract_image_and_state(obs)

        if isinstance(terminated, torch.Tensor):
            terminated = terminated.cpu().numpy()
        if isinstance(truncated, torch.Tensor):
            truncated = truncated.cpu().numpy()

        success = info.get("success", terminated)
        if isinstance(success, torch.Tensor):
            success = success.cpu().numpy()

        for i, eid in enumerate(env_ids):
            if not self.env_states[eid].active:
                continue

            self.env_states[eid].step += 1

            done = bool(terminated[i]) or bool(truncated[i])
            if done or self.env_states[eid].step >= self.max_steps:
                self.env_states[eid].complete = bool(success[i])
                self.env_states[eid].active = False

            self.env_states[eid].current_obs = {
                k: v[i] if v is not None else None for k, v in images_and_states.items()
            }

            if self.env_states[eid].do_validation:
                if images_and_states["full_images"] is not None:
                    self.env_states[eid].valid_pixels["full_images"].append(
                        images_and_states["full_images"][i]
                    )

        completes = np.array([self.env_states[eid].complete for eid in env_ids])
        active = np.array([self.env_states[eid].active for eid in env_ids])
        finish_steps = np.array([self.env_states[eid].step for eid in env_ids])

        full_images_and_states: Dict[str, Optional[np.ndarray]] = {}
        for key in ["full_images", "wrist_images", "states"]:
            obs_list = []
            for eid in env_ids:
                curr = self.env_states[eid].current_obs
                if curr and curr.get(key) is not None:
                    obs_list.append(curr[key])
            if obs_list and len(obs_list) == len(env_ids):
                full_images_and_states[key] = np.stack(obs_list)
            else:
                full_images_and_states[key] = None

        return {
            **full_images_and_states,
            "complete": completes,
            "active": active,
            "finish_step": finish_steps,
        }

    def chunk_step(self, env_ids: List[int], actions: torch.Tensor):
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        steps = actions.shape[1] if actions.ndim > 2 else 1
        if actions.ndim == 2:
            actions = actions[:, None, :]

        for step_idx in range(steps):
            results = self.step(env_ids, actions[:, step_idx])

        return results

    def get_env_states(self, env_ids: List[int]):
        return [self.env_states[eid] for eid in env_ids]

    def save_validation_videos(self, rollout_dir: str, env_ids: List[int]):
        for eid in env_ids:
            state = self.env_states[eid]
            if not state.do_validation:
                continue
            if not state.valid_pixels or not state.valid_pixels.get("full_images"):
                continue

            task_name = (
                f"maniskill_{self._current_task_env_id}"
                f"_task_{state.task_id}_trial_{state.trial_id}"
            )
            save_rollout_video(
                state.valid_pixels["full_images"],
                rollout_dir,
                task_name,
                state.complete,
            )

    def close(self, clear_cache: bool = True):
        if self.env is not None:
            self.env.close()
            self.env = None
