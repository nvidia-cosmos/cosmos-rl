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

"""``TensorDataPacker`` — a generic data packer for non-text (e.g.
Gymnasium) RL trajectories.

Cosmos-RL's default ``DataPacker`` assumes the rollout output is text
that requires tokenization.  For RL workloads where the rollout produces
tensor trajectories (observations, actions, rewards, terminated /
truncated flags, episode length), this packer handles the
rollout↔trainer plumbing without invoking a real tokenizer.

It is a thin, transport-agnostic base class:

* :meth:`get_rollout_input` simply forwards the dataset sample (which
  for gym-style tasks typically carries seed / initial-condition
  metadata).
* :meth:`get_policy_input` accepts a trajectory dict and passes it
  through unchanged so the trainer can consume the tensors directly.
* :meth:`policy_collate_fn` returns the list of per-episode dicts
  unchanged; collation across episodes is left to the trainer because
  RL trajectories typically have variable lengths and the trainer
  knows the right batching/padding strategy for its algorithm.

Subclasses can override :meth:`get_rollout_output` /
:meth:`get_policy_input` to add transport-specific behavior (e.g.
inserting a UCXX reference dict in place of the raw tensors and
materializing it on the policy side).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from cosmos_rl.dispatcher.data.packer.base import BaseDataPacker
from cosmos_rl.policy.config import Config
from cosmos_rl.utils.logging import logger


# Canonical field names used by ``TensorDataPacker``-compatible
# trajectories.  Subclasses and rollout engines should agree on these
# keys so the analyzer / trainer can introspect trajectories uniformly.
OBSERVATIONS = "observations"
ACTIONS = "actions"
REWARDS = "rewards"
TERMINATED = "terminated"
TRUNCATED = "truncated"
EPISODE_LENGTH = "episode_length"

TRAJECTORY_KEYS = (
    OBSERVATIONS,
    ACTIONS,
    REWARDS,
    TERMINATED,
    TRUNCATED,
    EPISODE_LENGTH,
)


class TensorDataPacker(BaseDataPacker):
    """Generic non-text data packer for tensor trajectories.

    Cosmos-RL invokes the data packer in three phases:

    1. Rollout setup — :meth:`get_rollout_input` converts a dataset
       sample to the input the rollout engine consumes.  For Gymnasium
       tasks this is typically the seed / initial-condition dict.
    2. Rollout output — :meth:`get_rollout_output` post-processes the
       trajectory dicts produced by the rollout engine.  The default
       implementation is a passthrough; subclasses can replace
       ``completions`` with reference handles (e.g. UCXX slot ids) and
       resolve them on the policy side.
    3. Policy input — :meth:`get_policy_input` converts each rollout
       output to the trainer's per-episode payload.  The default
       implementation passes the trajectory dict through unchanged so
       the trainer sees the same shape it produced.

    Subclassing pattern::

        class MyDataPacker(TensorDataPacker):
            def get_rollout_input(self, sample, **kwargs):
                # e.g. parse JSON initial conditions
                return {"prompt": sample.get("prompt", "{}")}
    """

    # The base ``BaseDataPacker`` interface signals "no tokenizer
    # required" simply by inheriting from BaseDataPacker (rather than
    # the LLM-flavored ``DataPacker``).  ``setup`` therefore stays
    # minimal.
    def setup(self, config: Config, *args, **kwargs) -> None:
        super().setup(config, *args, **kwargs)
        logger.info(
            f"[TensorDataPacker] {type(self).__name__} setup complete "
            "(no tokenizer required for tensor RL)"
        )

    # ------------------------------------------------------------------
    # Rollout-side hooks
    # ------------------------------------------------------------------

    def get_rollout_input(
        self,
        item: Any,
        **kwargs,
    ) -> Any:
        """Convert a dataset sample to rollout-engine input.

        Default: passthrough.  Override to e.g. parse a JSON ``prompt``
        field carrying initial conditions for a Gymnasium env.
        """
        return item

    def rollout_collate_fn(self, items: List[Any]) -> List[Any]:
        # The dispatcher passes mini-batches of rollout inputs; the
        # default behavior of preserving the list works fine for tensor
        # tasks since the rollout engine can iterate per-episode.
        return items

    def get_rollout_output(
        self,
        completions: List[Any],
        completed_conversations: List[Any],
        logprobs: List[Any],
        token_ids: List[Any],
        **kwargs,
    ):
        """Post-process rollout outputs.

        For tensor trajectories, ``completions`` is the list of
        per-episode trajectory dicts produced by the rollout engine.
        We pass them through unchanged.  Subclasses that swap the
        completions for transport handles (e.g. UCXX references) should
        override this method.
        """
        return (
            completions,
            completed_conversations,
            logprobs,
            token_ids,
            kwargs,
        )

    # ------------------------------------------------------------------
    # Policy-side hooks
    # ------------------------------------------------------------------

    def get_policy_input(
        self,
        sample: Any = None,
        rollout_output: Optional[Any] = None,
        n_ignore_prefix_tokens: int = 0,
        **kwargs,
    ) -> Any:
        """Return the per-episode payload the trainer consumes.

        The default implementation assumes ``rollout_output`` is already
        a trajectory dict produced by the rollout engine.  Subclasses
        may resolve transport-side references here (e.g. fetch tensors
        over UCXX) before returning.
        """
        return rollout_output

    def policy_compute_max_len(self, processed_samples: List[Any]) -> int:
        """Maximum trajectory length across the mini-batch.

        Falls back to ``1`` when episode-length metadata is unavailable
        — this value is only used to pre-allocate buffers, so a
        conservative answer is safe.
        """
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
            if hasattr(ep_len, "item"):
                try:
                    ep_len = int(ep_len.item())
                except Exception:
                    continue
            try:
                ep_len = int(ep_len)
            except (TypeError, ValueError):
                continue
            if ep_len > max_len:
                max_len = ep_len
        return max_len

    def policy_collate_fn(
        self,
        processed_samples: List[Any],
        computed_max_len: int,
    ) -> Dict[str, Any]:
        """Return the per-episode list as-is.

        RL trajectories are typically variable-length and ragged
        across episodes; the trainer is the right place to decide
        whether to concatenate (policy gradient with returns) or
        pad-and-stack (Q-learning with mini-batches).  Returning the
        list keeps the data packer transport- and algorithm-agnostic.
        """
        return {"trajectories": processed_samples}

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @staticmethod
    def is_trajectory(value: Any) -> bool:
        """Return True if ``value`` looks like a tensor trajectory dict.

        Used by transport-aware subclasses (and the analyzer) to detect
        whether to apply tensor-specific handling.  A trajectory is a
        dict that contains at least ``observations`` and ``actions``.
        """
        return (
            isinstance(value, dict)
            and OBSERVATIONS in value
            and ACTIONS in value
        )


__all__ = [
    "ACTIONS",
    "EPISODE_LENGTH",
    "OBSERVATIONS",
    "REWARDS",
    "TERMINATED",
    "TRAJECTORY_KEYS",
    "TRUNCATED",
    "TensorDataPacker",
]
