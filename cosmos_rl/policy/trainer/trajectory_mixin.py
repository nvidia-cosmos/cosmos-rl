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

"""
Trajectory-expansion mixin for trainers whose rollouts contain multiple
trainable transitions.

The mixin provides a three-phase ``step_training`` skeleton
(``begin -> per-rollout-or-per-chunk -> finalize``) that composing trainers
plug into by implementing either :meth:`TrajectoryExpansionMixin._train_one_rollout`
(rollout-level) or :meth:`TrajectoryExpansionMixin._train_one_chunk`
(chunk-level). The two iteration shapes are selected via the
``chunk_size`` class attribute. The outer rollout loop is delegated to
the packer via :meth:`TrajectoryPacker.iter_rollouts`, which lets packers
group, reorder, or filter rollouts (default body: yield once with all of
them). It is deliberately thin: it does **not** own advantage assignment,
loss math, length normalization, or cross-rollout batching. Those are the
composing trainer's concern.
"""

from typing import Any, Dict, List, Optional, Sequence

from cosmos_rl.dispatcher.data.packer.trajectory_packer import TrajectoryPacker
from cosmos_rl.dispatcher.data.schema import Rollout


class TrajectoryExpansionMixin:
    """For trainers whose rollouts contain multiple trainable transitions.

    Two iteration shapes, selected via the ``chunk_size`` class attribute:

    * ``chunk_size = None`` (default): rollout-level. Override
      :meth:`_train_one_rollout`. Identical to the PR-1 contract.
    * ``chunk_size = N >= 1``: chunk-level. Override :meth:`_train_one_chunk`.
      The mixin walks ``packer.iter_chunks(rollout, chunk_size)`` and calls
      the chunk hook once per chunk. For per-transition iteration, set
      ``chunk_size = 1``; the chunk hook receives a one-element list.

    The composing trainer overrides exactly one of ``_train_one_rollout`` or
    ``_train_one_chunk`` (whichever matches its ``chunk_size``), plus
    optionally ``_begin_training_step`` and ``_finalize_training_step``.
    Failing to override the matching hook raises :class:`NotImplementedError`
    with a diagnostic message naming the trainer class.

    Provides a three-phase ``step_training``::

        begin -> for each rollout: (train_one_rollout | per-chunk train_one_chunk) -> finalize

    Loss math, advantage handling, per-rollout/per-chunk gradient
    accumulation, and metrics computation are entirely the subclass's
    concern. This mixin does not interpret rollouts; it only orders the
    outer loops and the phase hooks.

    Compose as (mixin leftmost so its ``step_training`` wins)::

        class MyEmbodiedTrainer(TrajectoryExpansionMixin, GRPOTrainer):
            chunk_size = 8  # or None for rollout-level

            def _train_one_chunk(self, rollout, chunk_data, *args, **kwargs):
                # forward + loss + backward over a chunk of <= chunk_size
                # transitions
                ...

    Requires ``self.data_packer`` to implement the
    :class:`~cosmos_rl.dispatcher.data.packer.trajectory_packer.TrajectoryPacker`
    protocol (``num_transitions``, ``iter_transitions``, ``iter_chunks``,
    ``iter_rollouts``). The mixin asserts this on every ``step_training``
    call; the check is cheap and catches the common misconfiguration
    where the mixin is composed onto a trainer whose packer was never
    updated to produce trajectory-shaped rollouts.

    The composing trainer is free to read
    ``self.data_packer.num_transitions(rollout)`` for length normalization or
    other within-trajectory scaling, and may pre-compute aggregates over the
    full ``rollouts`` list in :meth:`_begin_training_step`. The mixin offers
    no opinion on which scheme the trainer picks.
    """

    chunk_size: Optional[int] = None

    def step_training(
        self,
        rollouts: Sequence[Rollout],
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Orchestrate the three-phase training step.

        Walks ``rollouts`` once via ``packer.iter_rollouts`` (which yields
        rollouts in packer-defined batches; the default body is a single
        batch containing all rollouts), dispatching per rollout to
        :meth:`_train_one_rollout` when ``chunk_size`` is ``None`` and to
        :meth:`_train_one_chunk` (over ``packer.iter_chunks``) when
        ``chunk_size`` is a positive int. Batch boundaries do not surface
        as trainer hooks; a future consumer that wants
        ``_begin_batch`` / ``_finalize_batch`` can add them without
        changing the packer signature. The composing trainer overrides
        the hook matching its ``chunk_size``; the default implementations
        of the unused hook raise :class:`NotImplementedError`.
        ``_begin_training_step`` is a no-op and ``_finalize_training_step``
        returns an empty dict; trainers that need ``zero_grad``, metric
        reset, ``lr.step()``, grad-reduce, or a ``report_data`` return
        value should override the relevant hook.
        """
        assert isinstance(self.data_packer, TrajectoryPacker), (
            f"{type(self).__name__} composes TrajectoryExpansionMixin but "
            f"its data_packer ({type(self.data_packer).__name__}) does not "
            f"implement the TrajectoryPacker protocol "
            f"(num_transitions, iter_transitions, iter_chunks, iter_rollouts)."
        )
        self._begin_training_step(rollouts, *args, **kwargs)
        for batch in self.data_packer.iter_rollouts(rollouts):
            if self.chunk_size is None:
                for rollout in batch:
                    self._train_one_rollout(rollout, *args, **kwargs)
            else:
                for rollout in batch:
                    for chunk_data in self.data_packer.iter_chunks(
                        rollout, self.chunk_size
                    ):
                        self._train_one_chunk(rollout, chunk_data, *args, **kwargs)
        return self._finalize_training_step(rollouts, *args, **kwargs)

    def _train_one_rollout(
        self,
        rollout: Rollout,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Per-rollout training body (used when ``chunk_size is None``).

        Typically iterates ``self.data_packer.iter_transitions(rollout)``
        and runs forward + loss + backward per transition. The subclass
        owns all of this: loss function, weighting, length normalization,
        per-rollout optimizer interactions, etc.
        """
        raise NotImplementedError(
            f"{type(self).__name__} composes TrajectoryExpansionMixin with "
            f"chunk_size=None (rollout mode) but does not override "
            f"_train_one_rollout."
        )

    def _train_one_chunk(
        self,
        rollout: Rollout,
        chunk_data: List[Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Per-chunk training body (used when ``chunk_size`` is a positive int).

        ``chunk_data`` is the list yielded by ``packer.iter_chunks``: up to
        ``chunk_size`` consecutive transition payloads from the same
        rollout. Typically the subclass collates ``chunk_data`` into a
        batch tensor, runs forward + loss + backward, and accumulates
        gradients before the next chunk.
        """
        raise NotImplementedError(
            f"{type(self).__name__} composes TrajectoryExpansionMixin with "
            f"chunk_size={self.chunk_size} (chunk mode) but does not "
            f"override _train_one_chunk."
        )

    def _begin_training_step(
        self,
        rollouts: Sequence[Rollout],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Hook for ``zero_grad``, metric resets, precomputing aggregates
        over the full ``rollouts`` list (e.g. total transitions for length
        normalization), etc. Default is a no-op so subclasses can keep their
        existing pattern; override as needed."""

    def _finalize_training_step(
        self,
        rollouts: Sequence[Rollout],
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Hook for ``lr.step()``, grad reduce, metric collection, and the
        return value. Default returns an empty dict; override to produce the
        trainer's ``report_data``."""
        return {}


__all__ = ["TrajectoryExpansionMixin"]
