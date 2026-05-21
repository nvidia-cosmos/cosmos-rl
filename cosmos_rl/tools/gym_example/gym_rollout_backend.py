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

"""Cosmos-RL :class:`RolloutBase` adapter for :class:`GymRolloutEngine`.

Registered as ``"gym"`` so a config can resolve
``[rollout].backend = "gym"``. The adapter is deliberately thin: it
holds a :class:`GymRolloutEngine`, handles per-payload episode
generation, and reports the underlying ``nn.Module`` through
:meth:`get_underlying_model` for colocated weight sharing.

Real disaggregated weight-sync between rollout and policy ranks is
**not** in scope for the toy demo (see the gym example README and the
trajectory-iteration feature doc); when a future PR wants to scale
this example up, the missing pieces (a working ``WeightMapper``
round-trip, NCCL bring-up) will live in that PR.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch

from cosmos_rl.dispatcher.data.data_fetcher import DataFetcherBase
from cosmos_rl.dispatcher.data.packer.base import BaseDataPacker
from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.rollout.rollout_base import RolloutBase, RolloutRegistry
from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.tools.gym_example.gym_policy import GymMLPConfig, GymPolicy
from cosmos_rl.tools.gym_example.gym_rollout import GymRolloutEngine
from cosmos_rl.utils.logging import logger


_DEFAULT_ENV_NAME = "CartPole-v1"
_DEFAULT_MAX_STEPS = 500


def _resolve_custom(config: Any) -> Dict[str, Any]:
    """Pull ``config.custom`` as a plain dict (handles both dict and pydantic)."""
    custom = getattr(config, "custom", None) or {}
    if isinstance(custom, dict):
        return custom
    try:
        return custom.model_dump()
    except Exception:
        return {}


def _build_default_policy(config: Any) -> GymPolicy:
    """Build a fresh :class:`GymPolicy` from the policy TOML, with sensible
    fallbacks if the config can't be resolved (e.g. unit-test path)."""
    path = getattr(getattr(config, "policy", None), "model_name_or_path", None)
    if path:
        try:
            import toml

            data = toml.load(path)
            section = data.get("model", data)
            cfg = GymMLPConfig(
                obs_dim=int(section.get("obs_dim", 4)),
                action_dim=int(section.get("action_dim", 2)),
                hidden_dim=int(section.get("hidden_dim", 64)),
                discrete=bool(section.get("discrete", True)),
            )
            return GymPolicy(cfg)
        except Exception as e:  # pragma: no cover - exercised in launch path
            logger.warning(
                f"[GymRolloutBackend] Failed to load policy config from {path!r}: {e}; "
                "falling back to default GymMLPConfig (CartPole shape)."
            )
    return GymPolicy(GymMLPConfig())


@RolloutRegistry.register(rollout_type="gym")
class GymRolloutBackend(RolloutBase):
    """:class:`RolloutBase` wrapper around :class:`GymRolloutEngine`.

    Holds the env factory, the underlying policy module, and a single
    :class:`GymRolloutEngine` instance. Each call to
    :meth:`rollout_generation` runs one episode per payload and packs
    the resulting trajectory dict into a :class:`RolloutResult`.

    Init parameters (read from ``config.custom``):

    * ``env_name`` (str, default ``"CartPole-v1"``): forwarded to
      ``gym.make`` to construct the env.
    * ``max_steps`` (int, default 500): hard cap on episode length;
      also the padded shape of the trajectory buffers.

    Tests can pass a pre-built ``policy`` and / or ``env_factory`` via
    :meth:`post_init_hook` kwargs to bypass the gymnasium import and
    config-file resolution.
    """

    def post_init_hook(self, **kwargs: Any) -> None:
        """Resolve the policy and env factory, but defer engine bringup
        to :meth:`init_engine` (matching the LLM rollout backends'
        contract)."""
        custom = _resolve_custom(self.config)
        self._env_name: str = str(custom.get("env_name", _DEFAULT_ENV_NAME))
        self._max_steps: int = int(custom.get("max_steps", _DEFAULT_MAX_STEPS))

        self._policy: GymPolicy = kwargs.get("policy") or _build_default_policy(
            self.config
        )
        self._env_factory = kwargs.get("env_factory")
        self._engine: Optional[GymRolloutEngine] = None
        self._model_param_map: Dict[str, torch.Tensor] = {}

    def init_engine(
        self,
        quantization: Optional[str] = None,
        seed: Optional[int] = None,
        load_format: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Lazily construct the :class:`GymRolloutEngine`.

        Honors ``env_factory`` from :meth:`post_init_hook` if provided
        (test path); otherwise imports gymnasium and builds via
        ``gym.make(env_name)``. Quantization / load_format are accepted
        for API compatibility but ignored — the toy MLP doesn't have
        a quantization story.
        """
        if self._engine is not None:
            self._engine_initialized = True
            return

        env_factory = self._env_factory
        if env_factory is None:
            try:
                import gymnasium as gym
            except ImportError as e:  # pragma: no cover - covered by skipif in tests
                raise RuntimeError(
                    "[GymRolloutBackend] gymnasium is not installed. "
                    'Install with `pip install "cosmos_rl[gym]"` or pass '
                    "an explicit env_factory in tests."
                ) from e

            env_name = self._env_name

            def env_factory() -> Any:
                return gym.make(env_name)

        self._engine = GymRolloutEngine(
            env_factory=env_factory,
            policy=self._policy,
            max_steps=self._max_steps,
        )
        self._engine_initialized = True

    def rollout_generation(
        self,
        payloads: Union[List[RLPayload], List[Dict[str, torch.Tensor]]],
        stream: Optional[torch.cuda.Stream] = None,
        data_packer: Optional[BaseDataPacker] = None,
        data_fetcher: Optional[DataFetcherBase] = None,
        is_validation: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> List[RolloutResult]:
        """Run one episode per payload and pack as :class:`RolloutResult`.

        The trajectory dict is placed in ``RolloutResult.completions[0]``
        so :class:`GymDataPacker` can read it back via the trainer-side
        ``Rollout.completion`` field. The ``stream`` /
        ``data_packer`` / ``data_fetcher`` arguments are accepted for
        API compatibility; the toy backend doesn't need any of them.
        """
        assert self._engine is not None, (
            "[GymRolloutBackend] init_engine() must be called before rollout_generation()."
        )

        results: List[RolloutResult] = []
        for payload in payloads:
            init = self._init_for_payload(payload, data_packer)
            traj = self._engine.run(init)
            prompt = init  # echo back so RolloutResult.prompt is informative
            results.append(RolloutResult(prompt=prompt, completions=[traj]))
        return results

    def get_underlying_model(self) -> torch.nn.Module:
        return self._policy

    def set_underlying_model(self, model: torch.nn.Module) -> None:
        """Used in colocated mode to share the trainer's policy module
        directly. Replaces the engine's policy reference too so the next
        ``rollout_generation`` uses the updated weights."""
        if not isinstance(model, GymPolicy):
            raise TypeError(
                "[GymRolloutBackend] set_underlying_model expects a GymPolicy, "
                f"got {type(model).__name__}."
            )
        self._policy = model
        if self._engine is not None:
            self._engine.policy = model

    def shutdown(self) -> None:
        """Release the gym env held by the engine."""
        if self._engine is not None:
            self._engine.close()
            self._engine = None
        self._engine_initialized = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _init_for_payload(
        payload: Any, data_packer: Optional[BaseDataPacker]
    ) -> Dict[str, Any]:
        """Resolve an ``init`` dict for :class:`GymRolloutEngine.run` from a payload.

        Tries the data packer's :meth:`get_rollout_input` first (the
        canonical path: ``GymDataPacker`` JSON-decodes the prompt
        there). Falls back to :meth:`GymRolloutEngine.init_from_prompt`
        on the raw payload prompt when no packer is provided (test
        path).
        """
        prompt = getattr(payload, "prompt", None)
        if data_packer is not None:
            try:
                return data_packer.get_rollout_input({"prompt": prompt})
            except Exception:  # pragma: no cover - belt-and-braces
                pass
        return GymRolloutEngine.init_from_prompt(prompt)


__all__ = ["GymRolloutBackend"]
