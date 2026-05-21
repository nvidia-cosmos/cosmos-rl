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

"""Gymnasium-driven rollout engine.

Drives a user-supplied :class:`gymnasium.Env` factory and produces
trajectory dicts in the canonical
:class:`~cosmos_rl.dispatcher.data.packer.tensor_data_packer.TensorDataPacker`
format so the trainer can consume them directly.

This module intentionally does **not** subclass
:class:`cosmos_rl.rollout.rollout_base.RolloutBase` so the example can
be exercised standalone (without the full controller / dispatcher
boot).  A thin wrapper that registers the engine with cosmos-rl's
``RolloutRegistry`` is provided in the package README.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

from cosmos_rl.dispatcher.data.packer.tensor_data_packer import (
    ACTIONS,
    EPISODE_LENGTH,
    OBSERVATIONS,
    REWARDS,
    TERMINATED,
    TRUNCATED,
)
from cosmos_rl.tools.gym_example.gym_policy import GymPolicy
from cosmos_rl.utils.logging import logger


# Type alias for an env factory: takes nothing and returns a fresh env.
EnvFactory = Callable[[], Any]


def rollout_episode(
    env: Any,
    policy: GymPolicy,
    *,
    max_steps: int,
    seed: Optional[int] = None,
    deterministic: bool = False,
) -> Dict[str, np.ndarray]:
    """Run a single episode and return a trajectory dict.

    The trajectory dict has fixed-length numpy arrays sized
    ``max_steps`` so it lines up with the UCXX schema layout used by
    other transports.  The actual valid prefix length is recorded in
    ``episode_length``; positions ``[ep_len:]`` are zero padding.

    Args:
        env: A constructed :class:`gymnasium.Env`.
        policy: A :class:`GymPolicy` whose ``act()`` returns
            ``(action, log_prob)``.
        max_steps: Hard cap on rollout length (also the padded shape).
        seed: Optional reset seed for reproducibility.
        deterministic: If True, ignores the policy's stochastic head and
            takes the argmax / mean.

    Returns:
        Dict with keys ``observations``, ``actions``, ``rewards``,
        ``terminated``, ``truncated``, ``episode_length``.
    """
    obs_dim = policy.config.obs_dim
    action_dim = policy.config.action_dim
    discrete = policy.config.discrete

    # Honor the device the policy lives on (CPU in unit tests, but cuda:0
    # under the colocated launcher, where the policy worker has already
    # moved params onto the device).  ``next(...)`` is safe because
    # GymPolicy always has at least one parameter.
    policy_device = next(policy.parameters()).device

    if discrete:
        actions_buf = np.zeros((max_steps,), dtype=np.int64)
    else:
        actions_buf = np.zeros((max_steps, action_dim), dtype=np.float32)
    obs_buf = np.zeros((max_steps, obs_dim), dtype=np.float32)
    rew_buf = np.zeros((max_steps,), dtype=np.float32)
    term_buf = np.zeros((max_steps,), dtype=np.bool_)
    trunc_buf = np.zeros((max_steps,), dtype=np.bool_)

    reset_kwargs = {} if seed is None else {"seed": int(seed)}
    obs, _info = env.reset(**reset_kwargs)
    ep_len = 0
    with torch.no_grad():
        for step in range(max_steps):
            obs_buf[step] = np.asarray(obs, dtype=np.float32)
            obs_t = torch.from_numpy(obs_buf[step]).unsqueeze(0).to(policy_device)
            if deterministic:
                logits = policy(obs_t)
                if discrete:
                    action = int(logits.argmax(dim=-1).item())
                else:
                    mean = logits[..., :action_dim]
                    action = mean.squeeze(0).cpu().numpy()
            else:
                action_t, _logp = policy.act(obs_t)
                if discrete:
                    action = int(action_t.item())
                else:
                    action = action_t.squeeze(0).cpu().numpy()

            if discrete:
                actions_buf[step] = action
                step_action = action
            else:
                actions_buf[step] = action
                step_action = action

            next_obs, reward, terminated, truncated, _info = env.step(step_action)
            rew_buf[step] = float(reward)
            term_buf[step] = bool(terminated)
            trunc_buf[step] = bool(truncated)
            ep_len += 1
            if terminated or truncated:
                break
            obs = next_obs

    return {
        OBSERVATIONS: obs_buf,
        ACTIONS: actions_buf,
        REWARDS: rew_buf,
        TERMINATED: term_buf,
        TRUNCATED: trunc_buf,
        EPISODE_LENGTH: np.array([ep_len], dtype=np.int64),
    }


class GymRolloutEngine:
    """Standalone rollout engine that drives a Gymnasium env factory.

    Usage::

        import gymnasium as gym

        engine = GymRolloutEngine(
            env_factory=lambda: gym.make("CartPole-v1"),
            policy=policy,
            max_steps=200,
        )
        traj = engine.run({"seed": 42})

    The wrapper that adapts this to cosmos-rl's
    :class:`~cosmos_rl.rollout.rollout_base.RolloutBase` interface is
    documented in the package README; it amounts to a
    ``rollout_generation()`` method that calls :meth:`run` per
    payload.
    """

    def __init__(
        self,
        *,
        env_factory: EnvFactory,
        policy: GymPolicy,
        max_steps: int = 500,
    ):
        self.env_factory = env_factory
        self.policy = policy
        self.max_steps = max_steps
        self._env = None

    def _ensure_env(self):
        if self._env is None:
            self._env = self.env_factory()
        return self._env

    def run(self, init: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        """Generate one trajectory.

        Args:
            init: Optional dict with ``seed`` (int) and / or
                ``deterministic`` (bool) overrides.  Anything else is
                ignored with a debug log so users can extend the
                contract without breakage.

        Returns:
            Trajectory dict from :func:`rollout_episode`.
        """
        env = self._ensure_env()
        seed = None
        deterministic = False
        if init:
            seed = init.get("seed")
            deterministic = bool(init.get("deterministic", False))
            unknown = set(init) - {"seed", "deterministic"}
            if unknown:
                logger.debug(
                    f"[GymRolloutEngine] Ignoring unknown init keys: {sorted(unknown)}"
                )
        return rollout_episode(
            env,
            self.policy,
            max_steps=self.max_steps,
            seed=seed,
            deterministic=deterministic,
        )

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

    @staticmethod
    def init_from_prompt(prompt: Any) -> Dict[str, Any]:
        """Decode a dataset prompt into an :meth:`run` ``init`` dict."""
        if isinstance(prompt, dict):
            return prompt
        if not prompt:
            return {}
        try:
            return json.loads(prompt)
        except (TypeError, json.JSONDecodeError):
            return {}


__all__ = ["EnvFactory", "GymRolloutEngine", "rollout_episode"]
