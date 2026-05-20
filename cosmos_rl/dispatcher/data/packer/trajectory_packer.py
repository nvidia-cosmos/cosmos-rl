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
Trajectory-iteration contract for trajectory-shaped rollouts.

A trajectory-shaped rollout encodes ``N >= 1`` trainable transitions that share
one :class:`~cosmos_rl.dispatcher.data.schema.Rollout` envelope: one
``prompt_idx``, one group-relative advantage, one wire-level admission unit.
Packers that produce this shape implement :class:`TrajectoryPacker` so that
trainers composing :class:`~cosmos_rl.policy.trainer.trajectory_mixin.TrajectoryExpansionMixin`
can iterate across rollouts (in packer-defined batches), per-rollout, or
per-chunk in a uniform way.

This contract is purely additive and opt-in. ``BaseDataPacker`` is unchanged;
existing packers (LLM, VLM, diffusion, PI05) need not implement this protocol.
"""

from typing import Any, Iterator, List, Protocol, Sequence, runtime_checkable

from cosmos_rl.dispatcher.data.schema import Rollout


@runtime_checkable
class TrajectoryPacker(Protocol):
    """Packer-side contract for trajectory-shaped rollouts.

    A trajectory-shaped rollout encodes ``N >= 1`` trainable transitions that
    share one :class:`~cosmos_rl.dispatcher.data.schema.Rollout` envelope (one
    ``prompt_idx``, one group-relative advantage). Packers that produce this
    shape implement four methods covering three iteration scopes:

    * ``num_transitions(rollout)`` -- how many transitions a rollout has.
    * ``iter_transitions(rollout)`` -- yield per-transition payloads.
    * ``iter_chunks(rollout, chunk_size)`` -- yield per-chunk payloads.
    * ``iter_rollouts(rollouts)`` -- yield rollouts in packer-defined batches.

    ``iter_chunks`` and ``iter_rollouts`` ship with inline default bodies, so
    a packer with no packer-specific efficient layout for those scopes only
    needs to implement ``num_transitions`` and ``iter_transitions``.
    Subclassing :class:`TrajectoryPacker` lets a packer inherit the defaults
    for free; structural implementers must provide their own implementations
    of both default-bearing methods to satisfy the runtime ``isinstance``
    check.

    Existing packers (LLM, VLM, diffusion, PI05) need not implement this
    protocol. Adoption is purely opt-in by the new trainer's packer.

    Implementations must:

    * Be **pure** with respect to advantages and loss-weights.
      ``iter_transitions`` and ``iter_chunks`` yield per-step *structure*;
      they must not silently scale ``rollout.advantage`` or apply length
      normalization. Loss aggregation belongs to the trainer.
    * Have ``num_transitions(rollout) == sum(1 for _ in iter_transitions(rollout))``
      for every rollout the packer produces. The mixin and any future
      cross-rollout batcher rely on this equality.
    * For overrides of ``iter_chunks``: yielded chunks must concatenate (in
      order) to the same per-transition sequence as ``iter_transitions``.
      The default body satisfies this trivially.
    * For overrides of ``iter_rollouts``: yielded batches partition (or
      filter) the input ``rollouts``. The mixin walks each batch in order
      and walks rollouts within a batch in the order yielded; it does not
      currently surface batch boundaries to the trainer (no
      ``_begin_batch`` / ``_finalize_batch`` hooks). A future consumer
      that wants batch-level hooks can add them without changing this
      signature.
    * Be safe to call once per training step; transitions, chunks, and
      batches yielded need not be re-iterable (callers do a single pass).
    """

    def num_transitions(self, rollout: Rollout) -> int:
        """Number of trainable transitions in this rollout."""

    def iter_transitions(self, rollout: Rollout) -> Iterator[Any]:
        """Yield each trainable transition payload in iteration order.

        Return type is intentionally opaque (``Any``): the trainer subclass
        produced this packer and knows what it yields.
        """

    def iter_chunks(self, rollout: Rollout, chunk_size: int) -> Iterator[List[Any]]:
        """Yield chunks of up to ``chunk_size`` consecutive transitions.

        Default body collects ``iter_transitions`` into fixed-size buckets;
        the last chunk may be shorter when ``num_transitions`` is not a
        multiple of ``chunk_size``. Override for packer-specific efficient
        chunking (e.g. pre-stacked tensor slices that avoid Python-level
        per-transition overhead).

        ``chunk_size = 1`` yields one-element chunks and is the recommended
        way to iterate per-transition through the chunk hook.
        """
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        chunk: List[Any] = []
        for transition in self.iter_transitions(rollout):
            chunk.append(transition)
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    def iter_rollouts(self, rollouts: Sequence[Rollout]) -> Iterator[Sequence[Rollout]]:
        """Yield rollouts in packer-defined batches.

        Default body yields a single batch containing all input rollouts
        (or zero batches if ``rollouts`` is empty, matching the ``empty
        in -> nothing yielded`` convention used by ``iter_chunks``).
        Override to group rollouts (e.g. by GRPO group, by length bucket,
        or by any other packer-side criterion), to reorder, or to filter.

        The mixin walks each yielded batch in order; within each batch it
        walks rollouts in the order yielded; per rollout it dispatches to
        either ``_train_one_rollout`` or ``_train_one_chunk`` depending on
        the trainer's ``chunk_size``. Batch boundaries are advisory: they
        do not currently surface as trainer-side hooks. A future consumer
        that wants ``_begin_batch`` / ``_finalize_batch`` hooks can add
        them without changing this signature.

        Filtering (yielding fewer rollouts than were passed in) is
        permitted; rollouts not yielded simply do not surface to the
        trainer.
        """
        rollouts_list = list(rollouts)
        if rollouts_list:
            yield rollouts_list


__all__ = ["TrajectoryPacker"]
