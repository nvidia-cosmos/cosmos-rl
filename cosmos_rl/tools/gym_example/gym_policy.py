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

"""Minimal MLP policy for the Gymnasium Classic Control demo.

Two flavors are supported through a single :class:`GymPolicy` module:

* **Discrete actions** (CartPole) — outputs categorical logits.
* **Continuous actions** (Pendulum) — outputs a Gaussian (mean, log_std).

The model is intentionally tiny (~2k params for CartPole) so the demo
fits comfortably on CPU and exercises the cosmos-rl pipeline without
GPU contention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

try:
    from transformers import PretrainedConfig
except ImportError:  # pragma: no cover - transformers is a required dep of cosmos_rl
    PretrainedConfig = object  # type: ignore[assignment, misc]


_GYM_MODEL_TYPE = "cosmos_rl_gym_mlp"


@dataclass
class GymMLPConfigSpec:
    """Plain-dataclass mirror of :class:`GymMLPConfig` — used by tests
    and by the local model-config loader to avoid pulling in the heavy
    HuggingFace ``PretrainedConfig`` machinery."""

    obs_dim: int = 4
    action_dim: int = 2
    hidden_dim: int = 64
    discrete: bool = True


class GymMLPConfig(PretrainedConfig):  # type: ignore[misc]
    """HuggingFace-shaped config so cosmos-rl's ModelRegistry can key
    on ``model_type``.

    The fields mirror :class:`GymMLPConfigSpec` and are populated either
    from a TOML file (via the local model-config loader registered in
    :func:`register_gym_policy`) or directly via kwargs in tests.
    """

    model_type = _GYM_MODEL_TYPE

    def __init__(
        self,
        obs_dim: int = 4,
        action_dim: int = 2,
        hidden_dim: int = 64,
        discrete: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.discrete = discrete


class GymPolicy(nn.Module):
    """Two-layer MLP with a discrete- or continuous-action head.

    For discrete actions the head emits ``action_dim`` logits; for
    continuous actions it emits ``2 * action_dim`` numbers (mean and
    log_std stacked).  ``act()`` samples and returns
    ``(action, log_prob)`` tensors so the rollout engine can record
    log-probs alongside the trajectory.
    """

    @staticmethod
    def supported_model_types():
        # Required by ``ModelRegistry.register_model`` so that
        # ``WeightMapper.get_weight_mapper(model_type)`` resolves to
        # :class:`IdentityWeightMapper` instead of falling back to
        # ``HFModelWeightMapper`` (which is LLM-specific and rejects
        # the gym MLP config because it lacks ``kv_head_ratio``).
        return [_GYM_MODEL_TYPE]

    def __init__(self, config: GymMLPConfig):
        super().__init__()
        self.config = config
        self.discrete = config.discrete

        head_out = config.action_dim if config.discrete else 2 * config.action_dim
        self.net = nn.Sequential(
            nn.Linear(config.obs_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, head_out),
        )
        # A separate value head is useful for actor-critic; we expose it
        # so a future PPO trainer can plug in without restructuring the
        # policy.
        self.value_head = nn.Linear(config.hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        # All but the last layer; useful for the value head.
        x = obs
        for layer in list(self.net)[:-1]:
            x = layer(x)
        return x

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.features(obs)).squeeze(-1)

    def act(self, obs: torch.Tensor):
        """Sample an action and return (action, log_prob).

        Returns:
            For discrete: ``(action: int64 tensor, log_prob: float tensor)``.
            For continuous: ``(action: float tensor, log_prob: float tensor)``.
        """
        out = self.forward(obs)
        if self.discrete:
            dist = Categorical(logits=out)
            action = dist.sample()
            return action, dist.log_prob(action)

        action_dim = self.config.action_dim
        mean, log_std = out.split(action_dim, dim=-1)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.sample()
        # Sum log-probs across action dims for a single per-step value.
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob


def register_gym_policy(
    *,
    model_type: str = _GYM_MODEL_TYPE,
    config_path_predicate: Optional[callable] = None,
) -> None:
    """Register the Gym MLP policy with cosmos-rl's registries.

    Wires up:

    * :class:`~cosmos_rl.utils.no_op_tokenizer.NoOpTokenizer` for any
      ``policy.model_name_or_path`` ending in ``.toml``.
    * Local model-config loader so the same ``.toml`` resolves to a
      :class:`GymMLPConfig`.
    * :class:`GymPolicy` -> :class:`IdentityWeightMapper` in
      :class:`ModelRegistry`, so the rollout worker's
      ``WeightMapper.get_weight_mapper(model_type)`` resolves to the
      identity mapper instead of falling back to the LLM-specific
      ``HFModelWeightMapper`` (which would reject the gym MLP config
      because it lacks ``kv_head_ratio`` / ``head_dim``).

    Args:
        model_type: Identifier matched against the TOML's ``model_type``
            field (default: ``"cosmos_rl_gym_mlp"``).
        config_path_predicate: Optional override for the predicate used
            to detect "this is a Gym TOML config" paths.  Defaults to a
            simple ``.toml`` extension check.
    """
    import toml

    from cosmos_rl.policy.model.base import IdentityWeightMapper, ModelRegistry
    from cosmos_rl.utils.model_config import register_local_model_config
    from cosmos_rl.utils.no_op_tokenizer import NoOpTokenizer
    from cosmos_rl.utils.util import register_tokenizer_loader

    predicate = config_path_predicate or (lambda p: p.endswith(".toml"))

    def _load_gym_config(path: str) -> GymMLPConfig:
        data = toml.load(path)
        # Nested under [model] for clarity in TOML; flat fallback for
        # convenience in tests.
        section = data.get("model", data)
        return GymMLPConfig(
            obs_dim=int(section.get("obs_dim", 4)),
            action_dim=int(section.get("action_dim", 2)),
            hidden_dim=int(section.get("hidden_dim", 64)),
            discrete=bool(section.get("discrete", True)),
        )

    register_tokenizer_loader(predicate=predicate, loader=lambda _p: NoOpTokenizer())
    register_local_model_config(predicate=predicate, factory=_load_gym_config)

    # Idempotent: ``register_gym_policy`` may be called multiple times
    # in a single process (e.g. controller + worker via gym_entry.py).
    if model_type not in ModelRegistry._MODEL_REGISTRY:
        ModelRegistry.register_model(GymPolicy, IdentityWeightMapper)


__all__ = [
    "GymMLPConfig",
    "GymMLPConfigSpec",
    "GymPolicy",
    "register_gym_policy",
]
