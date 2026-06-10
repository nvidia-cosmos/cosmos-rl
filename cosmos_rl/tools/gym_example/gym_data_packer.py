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

"""Gym-flavored data packer.

Subclasses :class:`TensorDataPacker` to:

* Parse the dataset's ``prompt`` field as JSON-encoded initial
  conditions (typically just ``{"seed": 42}``) before handing it to
  the rollout engine.
* Echo the trajectory dict produced by the rollout engine straight
  through to the trainer (no transport-side translation needed for
  the in-process / Redis demo).

Also implements the :class:`TrajectoryPacker` Protocol so trainers
that compose :class:`TrajectoryExpansionMixin` can iterate the
trajectory at the rollout, chunk, or transition scope. ``iter_chunks``
and ``iter_rollouts`` come for free from the protocol's default bodies
(via subclassing); only ``num_transitions`` and ``iter_transitions``
need real implementations because they're packer-payload-specific.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator

from cosmos_rl.dispatcher.data.packer.tensor_data_packer import (
    ACTIONS,
    EPISODE_LENGTH,
    OBSERVATIONS,
    REWARDS,
    TensorDataPacker,
)
from cosmos_rl.dispatcher.data.packer.trajectory_packer import TrajectoryPacker
from cosmos_rl.dispatcher.data.schema import Rollout
from cosmos_rl.utils.logging import logger


def _trajectory_from_rollout(rollout: Rollout) -> Dict[str, Any]:
    """Resolve the trajectory dict for ``rollout``.

    The Gym rollout engine writes the trajectory to ``Rollout.completion``
    (single completion per rollout). ``extra_info`` is checked as a
    fallback so transport-aware subclasses that stash the trajectory
    there in :meth:`get_rollout_output` keep working without overriding
    this helper.
    """
    completion = rollout.completion
    if isinstance(completion, dict) and OBSERVATIONS in completion:
        return completion
    extra = rollout.extra_info or {}
    if isinstance(extra, dict) and OBSERVATIONS in extra:
        return extra
    raise ValueError(
        "[GymDataPacker] Rollout does not carry a trajectory dict in "
        "completion or extra_info; "
        f"completion type={type(completion).__name__}."
    )


def _episode_length(traj: Dict[str, Any]) -> int:
    """Decode ``EPISODE_LENGTH`` to a Python int.

    The rollout engine emits ``np.array([ep_len], dtype=int64)`` but a
    plain ``int`` or a 0-d tensor is also accepted so manually-built
    trajectories (e.g. in tests) work without ceremony.
    """
    ep_len = traj.get(EPISODE_LENGTH)
    if ep_len is None:
        obs = traj.get(OBSERVATIONS)
        if hasattr(obs, "shape") and len(obs.shape) > 0:
            return int(obs.shape[0])
        return 0
    if hasattr(ep_len, "item"):
        try:
            return int(ep_len.item())
        except Exception:
            pass
    if hasattr(ep_len, "__len__") and len(ep_len) == 1:
        return int(ep_len[0])
    return int(ep_len)


class GymDataPacker(TensorDataPacker, TrajectoryPacker):
    """Tensor data packer for Gymnasium-driven rollouts.

    Implements :class:`TrajectoryPacker` so the gym example can compose
    :class:`TrajectoryExpansionMixin` on the trainer side. Default
    ``iter_chunks`` / ``iter_rollouts`` bodies are inherited from the
    protocol; ``num_transitions`` and ``iter_transitions`` are
    payload-specific and implemented here.
    """

    def get_rollout_input(self, item: Any, **kwargs) -> Dict[str, Any]:
        """Decode the dataset ``prompt`` JSON into a dict of init conditions.

        Cosmos-RL conventionally puts ``{"prompt": "..."}`` on the
        dataset; for a non-text task we treat the prompt string as a
        small JSON document carrying e.g. the env seed and any
        environment-specific overrides.  Plain strings (or missing
        prompts) fall back to an empty dict so the rollout engine can
        use sensible defaults.
        """
        prompt = item.get("prompt") if isinstance(item, dict) else None
        if not prompt:
            return {}
        if isinstance(prompt, dict):
            return prompt
        try:
            return json.loads(prompt)
        except (TypeError, json.JSONDecodeError):
            logger.debug(
                f"[GymDataPacker] prompt is not JSON, passing through verbatim: "
                f"{prompt!r}"
            )
            return {"prompt": prompt}

    def policy_compute_max_len(self, processed_samples) -> int:
        """Same logic as the base class but defaults to the OBS-shape
        fallback first since the Gym example always emits
        ``OBSERVATIONS`` and ``EPISODE_LENGTH`` is sometimes a Python
        int rather than a tensor.  Inherits the safe-floor of 1."""
        max_len = 1
        for item in processed_samples:
            if not isinstance(item, dict):
                continue
            ep_len = item.get(EPISODE_LENGTH)
            if ep_len is None:
                obs = item.get(OBSERVATIONS)
                if hasattr(obs, "shape") and len(obs.shape) > 0:
                    ep_len = int(obs.shape[0])
            if ep_len is None:
                continue
            try:
                ep_len_int = int(ep_len.item() if hasattr(ep_len, "item") else ep_len)
            except (TypeError, ValueError):
                continue
            if ep_len_int > max_len:
                max_len = ep_len_int
        return max_len

    # ------------------------------------------------------------------
    # TrajectoryPacker conformance
    # ------------------------------------------------------------------

    def num_transitions(self, rollout: Rollout) -> int:
        """Number of valid transitions in ``rollout``.

        Reads :data:`EPISODE_LENGTH` from the trajectory dict the rollout
        engine produced. Padded positions ``[ep_len:]`` are not counted.
        """
        return _episode_length(_trajectory_from_rollout(rollout))

    def iter_transitions(self, rollout: Rollout) -> Iterator[Dict[str, Any]]:
        """Yield single-step dicts for the valid prefix of the trajectory.

        Each yielded dict has keys ``{observation, action, reward}``,
        sliced from the (possibly padded) per-episode buffers. Padded
        positions are not yielded. The yielded values keep whatever
        backing type the rollout engine produced (numpy array, torch
        tensor, scalar) — the consumer is responsible for any
        device / dtype conversion.
        """
        traj = _trajectory_from_rollout(rollout)
        observations = traj[OBSERVATIONS]
        actions = traj[ACTIONS]
        rewards = traj[REWARDS]
        for t in range(_episode_length(traj)):
            yield {
                "observation": observations[t],
                "action": actions[t],
                "reward": rewards[t],
            }


__all__ = ["GymDataPacker"]
