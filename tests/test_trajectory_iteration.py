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
Unit tests for the trajectory-iteration contract:

* :class:`cosmos_rl.dispatcher.data.packer.trajectory_packer.TrajectoryPacker`
  -- structural ``@runtime_checkable`` behaviour and the default
  ``iter_chunks`` / ``iter_rollouts`` bodies inherited via subclassing.
* :class:`cosmos_rl.policy.trainer.trajectory_mixin.TrajectoryExpansionMixin`
  -- three-phase ``step_training`` ordering for both rollout-level
  (``chunk_size = None``) and chunk-level (``chunk_size >= 1``) iteration,
  the packer-contract assertion, and the ``NotImplementedError`` raised when
  the trainer fails to override the hook matching its ``chunk_size``.

These are pure unit tests: no torch, no GPU, no dispatcher. They pin the
shape of the contract so future refactors of either file get caught locally.
"""

import unittest
from typing import Any, Iterator, List, Sequence
from unittest.mock import MagicMock

from cosmos_rl.dispatcher.data.packer.trajectory_packer import TrajectoryPacker
from cosmos_rl.policy.trainer.trajectory_mixin import TrajectoryExpansionMixin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_rollout(prompt_idx: int) -> Any:
    """Return a stand-in for ``Rollout`` that is enough for these tests.

    The mixin and protocol do not introspect rollouts; they only pass them
    through. Using a ``MagicMock`` with a stable ``prompt_idx`` keeps the
    tests free of Pydantic-schema coupling.
    """
    r = MagicMock(name=f"Rollout(prompt_idx={prompt_idx})")
    r.prompt_idx = prompt_idx
    return r


class _GoodPacker(TrajectoryPacker):
    """Implements ``num_transitions`` + ``iter_transitions`` and inherits
    the default ``iter_chunks`` and ``iter_rollouts`` bodies from
    :class:`TrajectoryPacker`.

    Subclassing the protocol is the recommended pattern for packers
    without packer-specific efficient layouts: it gives them the
    default-bearing methods for free.
    """

    def __init__(self, transitions_per_rollout: int = 3) -> None:
        self._n = transitions_per_rollout

    def num_transitions(self, rollout: Any) -> int:
        return self._n

    def iter_transitions(self, rollout: Any) -> Iterator[Any]:
        for i in range(self._n):
            yield (rollout.prompt_idx, i)


class _StructuralFullPacker:
    """Implements all four methods structurally (no inheritance).

    Verifies that a class satisfies the ``@runtime_checkable`` protocol
    purely by method presence, even without subclassing. Used to guard
    against accidentally tightening the contract to require inheritance.
    """

    def num_transitions(self, rollout: Any) -> int:
        return 2

    def iter_transitions(self, rollout: Any) -> Iterator[Any]:
        yield (rollout.prompt_idx, 0)
        yield (rollout.prompt_idx, 1)

    def iter_chunks(self, rollout: Any, chunk_size: int) -> Iterator[List[Any]]:
        all_t = list(self.iter_transitions(rollout))
        for i in range(0, len(all_t), chunk_size):
            yield all_t[i : i + chunk_size]

    def iter_rollouts(self, rollouts: Sequence[Any]) -> Iterator[Sequence[Any]]:
        rollouts_list = list(rollouts)
        if rollouts_list:
            yield rollouts_list


class _OverriddenChunksPacker(TrajectoryPacker):
    """Overrides ``iter_chunks`` with a sentinel-tagged body. Inherits
    the default ``iter_rollouts``.

    Used to verify that the mixin actually delegates to the packer's
    ``iter_chunks`` (rather than re-implementing chunking inside the
    mixin). Each chunk carries a ``("chunk", chunk_idx, ...)`` tag so
    tests can assert provenance unambiguously.
    """

    def __init__(self, transitions_per_rollout: int = 5) -> None:
        self._n = transitions_per_rollout

    def num_transitions(self, rollout: Any) -> int:
        return self._n

    def iter_transitions(self, rollout: Any) -> Iterator[Any]:
        for i in range(self._n):
            yield (rollout.prompt_idx, i)

    def iter_chunks(self, rollout: Any, chunk_size: int) -> Iterator[List[Any]]:
        all_t = list(self.iter_transitions(rollout))
        for chunk_idx, start in enumerate(range(0, len(all_t), chunk_size)):
            yield [("chunk", chunk_idx, t) for t in all_t[start : start + chunk_size]]


class _GroupedRolloutsPacker(TrajectoryPacker):
    """Overrides ``iter_rollouts`` to yield user-defined groups.

    Used to verify the mixin delegates the outer loop to the packer's
    ``iter_rollouts`` and that batch-internal rollout order is preserved.
    Inherits the default ``iter_chunks``. Rollouts whose ``prompt_idx``
    is not listed in ``groups`` are filtered out and never surface to
    the trainer.
    """

    def __init__(
        self,
        groups: Sequence[Sequence[int]],
        transitions_per_rollout: int = 2,
    ) -> None:
        self._groups = [list(g) for g in groups]
        self._n = transitions_per_rollout

    def num_transitions(self, rollout: Any) -> int:
        return self._n

    def iter_transitions(self, rollout: Any) -> Iterator[Any]:
        for i in range(self._n):
            yield (rollout.prompt_idx, i)

    def iter_rollouts(self, rollouts: Sequence[Any]) -> Iterator[Sequence[Any]]:
        by_idx = {r.prompt_idx: r for r in rollouts}
        for group in self._groups:
            batch = [by_idx[i] for i in group if i in by_idx]
            if batch:
                yield batch


class _MissingIter:
    """Implements ``num_transitions`` only -- should NOT satisfy the protocol."""

    def num_transitions(self, rollout: Any) -> int:
        return 1


class _MissingNum:
    """Implements ``iter_transitions`` only -- should NOT satisfy the protocol."""

    def iter_transitions(self, rollout: Any) -> Iterator[Any]:
        yield rollout


class _MissingChunks:
    """Implements ``num_transitions`` + ``iter_transitions`` structurally
    but no ``iter_chunks`` -- should NOT satisfy the protocol.

    Confirms that the protocol's default ``iter_chunks`` body does NOT
    leak into structural-only implementers; they must either subclass
    :class:`TrajectoryPacker` to inherit the default or provide their
    own.
    """

    def num_transitions(self, rollout: Any) -> int:
        return 1

    def iter_transitions(self, rollout: Any) -> Iterator[Any]:
        yield rollout


class _MissingRollouts:
    """Implements ``num_transitions`` + ``iter_transitions`` +
    ``iter_chunks`` structurally but no ``iter_rollouts`` -- should NOT
    satisfy the protocol.

    Symmetric to ``_MissingChunks``: pins that the default
    ``iter_rollouts`` body does NOT leak into structural-only
    implementers either.
    """

    def num_transitions(self, rollout: Any) -> int:
        return 1

    def iter_transitions(self, rollout: Any) -> Iterator[Any]:
        yield rollout

    def iter_chunks(self, rollout: Any, chunk_size: int) -> Iterator[List[Any]]:
        yield list(self.iter_transitions(rollout))


# ---------------------------------------------------------------------------
# TrajectoryPacker protocol tests
# ---------------------------------------------------------------------------


class TestTrajectoryPackerProtocol(unittest.TestCase):
    def test_subclass_inherits_default_chunks(self) -> None:
        """A subclass that implements ``num_transitions`` and
        ``iter_transitions`` and inherits the default ``iter_chunks`` body
        passes ``isinstance``."""
        self.assertIsInstance(_GoodPacker(), TrajectoryPacker)

    def test_structural_full_implementer_satisfies_protocol(self) -> None:
        """A class that implements all four methods structurally (no
        inheritance) also passes ``isinstance``: the protocol is
        ``@runtime_checkable`` and method-presence-based."""
        self.assertIsInstance(_StructuralFullPacker(), TrajectoryPacker)

    def test_missing_iter_transitions_fails(self) -> None:
        """A stub missing ``iter_transitions`` fails the runtime check."""
        self.assertNotIsInstance(_MissingIter(), TrajectoryPacker)

    def test_missing_num_transitions_fails(self) -> None:
        """A stub missing ``num_transitions`` fails the runtime check."""
        self.assertNotIsInstance(_MissingNum(), TrajectoryPacker)

    def test_missing_iter_chunks_structural_fails(self) -> None:
        """A structural stub missing ``iter_chunks`` fails the runtime
        check; the protocol's default body is only available via
        inheritance, not structural implementation."""
        self.assertNotIsInstance(_MissingChunks(), TrajectoryPacker)

    def test_missing_iter_rollouts_structural_fails(self) -> None:
        """A structural stub missing ``iter_rollouts`` (but providing all
        three other methods) fails the runtime check. Symmetric to the
        ``iter_chunks`` case: default bodies don't leak into structural-
        only implementers."""
        self.assertNotIsInstance(_MissingRollouts(), TrajectoryPacker)

    def test_unrelated_object_fails(self) -> None:
        """Plain objects without the methods do not accidentally satisfy
        the protocol."""
        self.assertNotIsInstance(object(), TrajectoryPacker)


class TestTrajectoryPackerDefaultIterChunks(unittest.TestCase):
    """Default ``iter_chunks`` body inherited from :class:`TrajectoryPacker`.

    Pins the contract that overrides must respect: chunks concatenate to
    the same per-transition sequence as ``iter_transitions``, the last
    chunk may be shorter, and ``chunk_size = 1`` yields one-element chunks
    (the recommended way to get per-transition iteration through the chunk
    hook).
    """

    def test_default_chunks_partition_transitions_evenly(self) -> None:
        packer = _GoodPacker(transitions_per_rollout=6)
        rollout = _make_fake_rollout(7)
        chunks = list(packer.iter_chunks(rollout, chunk_size=2))
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(len(c) == 2 for c in chunks))
        flattened = [t for c in chunks for t in c]
        self.assertEqual(flattened, list(packer.iter_transitions(rollout)))

    def test_default_chunks_short_tail_when_uneven(self) -> None:
        """``num_transitions = 5`` with ``chunk_size = 2`` produces chunks
        of length ``[2, 2, 1]``."""
        packer = _GoodPacker(transitions_per_rollout=5)
        rollout = _make_fake_rollout(0)
        chunks = list(packer.iter_chunks(rollout, chunk_size=2))
        self.assertEqual([len(c) for c in chunks], [2, 2, 1])
        flattened = [t for c in chunks for t in c]
        self.assertEqual(flattened, list(packer.iter_transitions(rollout)))

    def test_default_chunks_chunk_size_one(self) -> None:
        """``chunk_size = 1`` yields one-element chunks, one per
        transition. This is the documented way to get per-transition
        iteration through the chunk hook."""
        packer = _GoodPacker(transitions_per_rollout=4)
        rollout = _make_fake_rollout(3)
        chunks = list(packer.iter_chunks(rollout, chunk_size=1))
        self.assertEqual(len(chunks), 4)
        self.assertTrue(all(len(c) == 1 for c in chunks))
        flattened = [t for c in chunks for t in c]
        self.assertEqual(flattened, list(packer.iter_transitions(rollout)))

    def test_default_chunks_chunk_size_larger_than_n(self) -> None:
        """``chunk_size`` larger than ``num_transitions`` yields a single
        chunk containing all transitions."""
        packer = _GoodPacker(transitions_per_rollout=3)
        rollout = _make_fake_rollout(0)
        chunks = list(packer.iter_chunks(rollout, chunk_size=100))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), 3)

    def test_default_chunks_zero_transitions(self) -> None:
        """A rollout with zero transitions yields zero chunks (no
        empty-chunk artefact)."""
        packer = _GoodPacker(transitions_per_rollout=0)
        rollout = _make_fake_rollout(0)
        chunks = list(packer.iter_chunks(rollout, chunk_size=4))
        self.assertEqual(chunks, [])

    def test_default_chunks_chunk_size_equals_n(self) -> None:
        """``chunk_size == num_transitions`` yields exactly one full
        chunk with no short tail. Boundary case between the
        ``chunk_size < n`` and ``chunk_size > n`` branches."""
        packer = _GoodPacker(transitions_per_rollout=4)
        rollout = _make_fake_rollout(0)
        chunks = list(packer.iter_chunks(rollout, chunk_size=4))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), 4)
        self.assertEqual(chunks[0], list(packer.iter_transitions(rollout)))

    def test_default_chunks_zero_chunk_size_raises(self) -> None:
        """``chunk_size = 0`` raises ``ValueError`` with a diagnostic
        message naming the offending parameter."""
        packer = _GoodPacker()
        with self.assertRaises(ValueError) as ctx:
            list(packer.iter_chunks(_make_fake_rollout(0), chunk_size=0))
        self.assertIn("chunk_size", str(ctx.exception))

    def test_default_chunks_negative_chunk_size_raises(self) -> None:
        """``chunk_size < 0`` is symmetrically rejected. Pinned to guard
        against the validation accidentally being narrowed to ``== 0``."""
        packer = _GoodPacker()
        with self.assertRaises(ValueError) as ctx:
            list(packer.iter_chunks(_make_fake_rollout(0), chunk_size=-1))
        self.assertIn("chunk_size", str(ctx.exception))

    def test_default_chunks_yield_independent_lists(self) -> None:
        """The default body must not alias an internal buffer across
        yields: mutating one yielded chunk must not corrupt other
        already-yielded chunks. Pins the buffer-reset (``chunk = []``
        rather than ``chunk.clear()``) in the implementation."""
        packer = _GoodPacker(transitions_per_rollout=4)
        rollout = _make_fake_rollout(0)
        chunks = list(packer.iter_chunks(rollout, chunk_size=2))
        self.assertEqual(len(chunks), 2)
        chunks[0].append(("mutated",))
        self.assertEqual(chunks[1], [(0, 2), (0, 3)])
        self.assertNotIn(("mutated",), chunks[1])


class TestTrajectoryPackerDefaultIterRollouts(unittest.TestCase):
    """Default ``iter_rollouts`` body inherited from :class:`TrajectoryPacker`.

    Pins the contract: a non-empty input yields exactly one batch
    containing all input rollouts in the input order; an empty input
    yields zero batches (matching the ``empty in -> nothing yielded``
    convention used by ``iter_chunks``).
    """

    def test_default_rollouts_yields_single_batch(self) -> None:
        """A non-empty input yields exactly one batch, in input order."""
        packer = _GoodPacker()
        rollouts = [_make_fake_rollout(i) for i in (3, 1, 4, 1, 5)]
        batches = list(packer.iter_rollouts(rollouts))
        self.assertEqual(len(batches), 1)
        self.assertEqual(list(batches[0]), rollouts)

    def test_default_rollouts_empty_input_yields_zero_batches(self) -> None:
        """Empty input yields zero batches, matching the ``iter_chunks``
        zero-transition convention. Pinned so the default body is not
        accidentally changed to yield an empty batch."""
        packer = _GoodPacker()
        batches = list(packer.iter_rollouts([]))
        self.assertEqual(batches, [])

    def test_default_rollouts_yields_independent_list(self) -> None:
        """The yielded batch is a fresh list, not aliased to the input:
        mutating the yielded batch must not corrupt the input. Pins the
        ``list(rollouts)`` defensive copy in the implementation."""
        packer = _GoodPacker()
        rollouts = [_make_fake_rollout(i) for i in (0, 1, 2)]
        batches = list(packer.iter_rollouts(rollouts))
        self.assertIsNot(batches[0], rollouts)
        list(batches[0]).append(_make_fake_rollout(99))
        self.assertEqual(len(rollouts), 3)


# ---------------------------------------------------------------------------
# TrajectoryExpansionMixin tests
# ---------------------------------------------------------------------------


class _RecordingTrainer(TrajectoryExpansionMixin):
    """Minimal composing trainer used to record phase-order semantics.

    Records every phase call into ``self.events`` with enough metadata to
    verify both ordering and the rollout argument identity.
    """

    def __init__(self, data_packer: Any) -> None:
        self.data_packer = data_packer
        self.events: List[Any] = []

    def _begin_training_step(self, rollouts, *args, **kwargs):  # type: ignore[override]
        self.events.append(("begin", len(rollouts)))

    def _train_one_rollout(self, rollout, *args, **kwargs):  # type: ignore[override]
        self.events.append(("train", rollout.prompt_idx))

    def _finalize_training_step(self, rollouts, *args, **kwargs):  # type: ignore[override]
        self.events.append(("finalize", len(rollouts)))
        return {"num_rollouts": len(rollouts)}


class TestTrajectoryExpansionMixinOrdering(unittest.TestCase):
    def test_three_phase_order_with_in_order_rollouts(self) -> None:
        """``step_training`` must emit ``begin``, then one ``train`` per
        rollout in input order, then ``finalize`` exactly once."""
        trainer = _RecordingTrainer(_GoodPacker())
        rollouts = [_make_fake_rollout(i) for i in (5, 2, 9, 7)]

        result = trainer.step_training(rollouts)

        self.assertEqual(
            trainer.events,
            [
                ("begin", 4),
                ("train", 5),
                ("train", 2),
                ("train", 9),
                ("train", 7),
                ("finalize", 4),
            ],
        )
        self.assertEqual(result, {"num_rollouts": 4})

    def test_empty_rollouts_still_runs_begin_and_finalize(self) -> None:
        """With zero rollouts, the begin/finalize bookends still fire but no
        ``train`` event is recorded. The mixin's contract is to be a pure
        outer loop; degenerate inputs must not crash."""
        trainer = _RecordingTrainer(_GoodPacker())
        result = trainer.step_training([])
        self.assertEqual(trainer.events, [("begin", 0), ("finalize", 0)])
        self.assertEqual(result, {"num_rollouts": 0})

    def test_forwards_args_and_kwargs(self) -> None:
        """Positional and keyword arguments are forwarded unchanged to all
        three hooks so trainers can thread state (e.g. an optimizer step
        index) through ``step_training``."""

        seen: List[Any] = []

        class _ArgRecorder(TrajectoryExpansionMixin):
            def __init__(self, data_packer: Any) -> None:
                self.data_packer = data_packer

            def _begin_training_step(self, rollouts, *args, **kwargs):
                seen.append(("begin", args, kwargs))

            def _train_one_rollout(self, rollout, *args, **kwargs):
                seen.append(("train", args, kwargs))

            def _finalize_training_step(self, rollouts, *args, **kwargs):
                seen.append(("finalize", args, kwargs))
                return {}

        trainer = _ArgRecorder(_GoodPacker())
        rollouts = [_make_fake_rollout(0)]

        trainer.step_training(rollouts, 42, mode="train")

        self.assertEqual(
            seen,
            [
                ("begin", (42,), {"mode": "train"}),
                ("train", (42,), {"mode": "train"}),
                ("finalize", (42,), {"mode": "train"}),
            ],
        )

    def test_default_finalize_returns_empty_dict(self) -> None:
        """A subclass that does not override ``_finalize_training_step``
        gets the default empty-dict return -- this is the documented
        baseline that trainers extend."""

        class _Bare(TrajectoryExpansionMixin):
            def __init__(self, data_packer: Any) -> None:
                self.data_packer = data_packer

            def _train_one_rollout(self, rollout, *args, **kwargs):
                pass

        trainer = _Bare(_GoodPacker())
        result = trainer.step_training([_make_fake_rollout(0)])
        self.assertEqual(result, {})


class TestTrajectoryExpansionMixinPackerAssertion(unittest.TestCase):
    def test_assertion_when_packer_missing_protocol(self) -> None:
        """If the composing trainer's ``data_packer`` does not implement
        ``TrajectoryPacker``, the mixin raises ``AssertionError`` with a
        diagnostic message that names both the trainer class and the
        offending packer class."""
        trainer = _RecordingTrainer(_MissingIter())

        with self.assertRaises(AssertionError) as ctx:
            trainer.step_training([_make_fake_rollout(0)])

        msg = str(ctx.exception)
        self.assertIn("TrajectoryExpansionMixin", msg)
        self.assertIn("_RecordingTrainer", msg)
        self.assertIn("_MissingIter", msg)
        self.assertIn("TrajectoryPacker", msg)

    def test_assertion_fires_before_begin_hook(self) -> None:
        """The packer-contract assertion must run before any subclass hook,
        so a misconfigured trainer cannot e.g. ``zero_grad`` and then crash
        with a half-mutated optimizer state."""
        trainer = _RecordingTrainer(_MissingNum())

        with self.assertRaises(AssertionError):
            trainer.step_training([_make_fake_rollout(0)])

        self.assertEqual(trainer.events, [])


# ---------------------------------------------------------------------------
# Chunk-level mixin tests
# ---------------------------------------------------------------------------


class _ChunkRecordingTrainer(TrajectoryExpansionMixin):
    """Minimal chunk-mode composing trainer used to record phase-order
    semantics, chunk content, and chunk count.

    The class attribute ``chunk_size`` is set per-test by overriding on
    instantiation through a subclass; see the chunk-mode tests below.
    """

    chunk_size = 2

    def __init__(self, data_packer: Any) -> None:
        self.data_packer = data_packer
        self.events: List[Any] = []

    def _begin_training_step(self, rollouts, *args, **kwargs):  # type: ignore[override]
        self.events.append(("begin", len(rollouts)))

    def _train_one_chunk(self, rollout, chunk_data, *args, **kwargs):  # type: ignore[override]
        self.events.append(("chunk", rollout.prompt_idx, list(chunk_data)))

    def _finalize_training_step(self, rollouts, *args, **kwargs):  # type: ignore[override]
        self.events.append(("finalize", len(rollouts)))
        return {"num_rollouts": len(rollouts)}


class TestTrajectoryExpansionMixinChunkMode(unittest.TestCase):
    def test_chunk_mode_phase_order_with_default_chunking(self) -> None:
        """``chunk_size = 2`` over a packer producing 3 transitions per
        rollout yields chunks of length ``[2, 1]`` per rollout, in order,
        bracketed by ``begin`` and ``finalize``."""
        trainer = _ChunkRecordingTrainer(_GoodPacker(transitions_per_rollout=3))
        rollouts = [_make_fake_rollout(i) for i in (5, 9)]

        result = trainer.step_training(rollouts)

        self.assertEqual(
            trainer.events,
            [
                ("begin", 2),
                ("chunk", 5, [(5, 0), (5, 1)]),
                ("chunk", 5, [(5, 2)]),
                ("chunk", 9, [(9, 0), (9, 1)]),
                ("chunk", 9, [(9, 2)]),
                ("finalize", 2),
            ],
        )
        self.assertEqual(result, {"num_rollouts": 2})

    def test_chunk_mode_uses_packer_iter_chunks_override(self) -> None:
        """The mixin must delegate to ``packer.iter_chunks`` (not
        re-implement chunking internally). A packer that overrides
        ``iter_chunks`` with a sentinel-tagged body shows up in the
        recorded events."""

        class _SentinelTrainer(_ChunkRecordingTrainer):
            chunk_size = 2

        trainer = _SentinelTrainer(_OverriddenChunksPacker(transitions_per_rollout=3))
        rollouts = [_make_fake_rollout(7)]

        trainer.step_training(rollouts)

        self.assertEqual(
            trainer.events,
            [
                ("begin", 1),
                (
                    "chunk",
                    7,
                    [("chunk", 0, (7, 0)), ("chunk", 0, (7, 1))],
                ),
                ("chunk", 7, [("chunk", 1, (7, 2))]),
                ("finalize", 1),
            ],
        )

    def test_chunk_size_one_covers_per_transition_iteration(self) -> None:
        """A trainer that wants per-transition iteration sets
        ``chunk_size = 1`` and unwraps the one-element chunk."""

        class _PerTransitionTrainer(_ChunkRecordingTrainer):
            chunk_size = 1

        trainer = _PerTransitionTrainer(_GoodPacker(transitions_per_rollout=3))
        rollouts = [_make_fake_rollout(0)]

        trainer.step_training(rollouts)

        self.assertEqual(
            trainer.events,
            [
                ("begin", 1),
                ("chunk", 0, [(0, 0)]),
                ("chunk", 0, [(0, 1)]),
                ("chunk", 0, [(0, 2)]),
                ("finalize", 1),
            ],
        )

    def test_chunk_mode_empty_rollouts(self) -> None:
        """With zero rollouts in chunk mode the begin/finalize bookends
        still fire and no ``chunk`` event is recorded."""
        trainer = _ChunkRecordingTrainer(_GoodPacker())
        result = trainer.step_training([])
        self.assertEqual(trainer.events, [("begin", 0), ("finalize", 0)])
        self.assertEqual(result, {"num_rollouts": 0})

    def test_chunk_mode_packer_assertion(self) -> None:
        """The packer-protocol assertion fires in chunk mode the same way
        it fires in rollout mode: a packer missing any of the three
        methods cannot be used with the mixin."""
        trainer = _ChunkRecordingTrainer(_MissingChunks())
        with self.assertRaises(AssertionError) as ctx:
            trainer.step_training([_make_fake_rollout(0)])
        msg = str(ctx.exception)
        self.assertIn("_MissingChunks", msg)
        self.assertIn("TrajectoryPacker", msg)

    def test_chunk_mode_forwards_args_and_kwargs(self) -> None:
        """In chunk mode, positional and keyword arguments are forwarded
        unchanged to all three hooks (``begin``, ``train_one_chunk``,
        ``finalize``). Parity with the rollout-mode forwarding test."""
        seen: List[Any] = []

        class _ArgChunkRecorder(TrajectoryExpansionMixin):
            chunk_size = 2

            def __init__(self, data_packer: Any) -> None:
                self.data_packer = data_packer

            def _begin_training_step(self, rollouts, *args, **kwargs):
                seen.append(("begin", args, kwargs))

            def _train_one_chunk(self, rollout, chunk_data, *args, **kwargs):
                seen.append(("chunk", args, kwargs))

            def _finalize_training_step(self, rollouts, *args, **kwargs):
                seen.append(("finalize", args, kwargs))
                return {}

        trainer = _ArgChunkRecorder(_GoodPacker(transitions_per_rollout=2))
        rollouts = [_make_fake_rollout(0)]

        trainer.step_training(rollouts, 17, mode="train")

        self.assertEqual(
            seen,
            [
                ("begin", (17,), {"mode": "train"}),
                ("chunk", (17,), {"mode": "train"}),
                ("finalize", (17,), {"mode": "train"}),
            ],
        )

    def test_chunk_mode_default_finalize_returns_empty_dict(self) -> None:
        """A chunk-mode subclass that does not override
        ``_finalize_training_step`` gets the default empty-dict return.
        Parity with the rollout-mode default-finalize test."""

        class _BareChunk(TrajectoryExpansionMixin):
            chunk_size = 1

            def __init__(self, data_packer: Any) -> None:
                self.data_packer = data_packer

            def _train_one_chunk(self, rollout, chunk_data, *args, **kwargs):
                pass

        trainer = _BareChunk(_GoodPacker(transitions_per_rollout=2))
        result = trainer.step_training([_make_fake_rollout(0)])
        self.assertEqual(result, {})

    def test_chunk_mode_variable_length_rollouts(self) -> None:
        """``iter_chunks`` yields rollout-aligned chunks: each rollout's
        chunks are walked completely before the next rollout starts, and
        rollouts of different ``num_transitions`` produce different
        per-rollout chunk counts. This is the realistic shape PI05 will
        consume."""

        class _VariablePacker(TrajectoryPacker):
            """``num_transitions`` is keyed off ``rollout.prompt_idx`` so
            different rollouts produce different chunk counts."""

            def num_transitions(self, rollout: Any) -> int:
                return rollout.prompt_idx

            def iter_transitions(self, rollout: Any) -> Iterator[Any]:
                for i in range(rollout.prompt_idx):
                    yield (rollout.prompt_idx, i)

        trainer = _ChunkRecordingTrainer(_VariablePacker())
        rollouts = [_make_fake_rollout(i) for i in (1, 4, 3)]

        trainer.step_training(rollouts)

        chunk_events = [e for e in trainer.events if e[0] == "chunk"]
        self.assertEqual(
            chunk_events,
            [
                ("chunk", 1, [(1, 0)]),
                ("chunk", 4, [(4, 0), (4, 1)]),
                ("chunk", 4, [(4, 2), (4, 3)]),
                ("chunk", 3, [(3, 0), (3, 1)]),
                ("chunk", 3, [(3, 2)]),
            ],
        )
        self.assertEqual(trainer.events[0], ("begin", 3))
        self.assertEqual(trainer.events[-1], ("finalize", 3))

    def test_chunk_data_is_a_list(self) -> None:
        """``chunk_data`` is documented as a ``List[Any]``. Pinned so a
        future refactor of ``iter_chunks`` cannot silently switch to a
        tuple, generator, or other iterable that would break trainer
        consumers (e.g. those that index ``chunk_data[0]`` for the lead
        transition)."""
        captured: List[Any] = []

        class _CaptureTrainer(TrajectoryExpansionMixin):
            chunk_size = 2

            def __init__(self, data_packer: Any) -> None:
                self.data_packer = data_packer

            def _train_one_chunk(self, rollout, chunk_data, *args, **kwargs):
                captured.append(chunk_data)

        trainer = _CaptureTrainer(_GoodPacker(transitions_per_rollout=3))
        trainer.step_training([_make_fake_rollout(0)])

        self.assertEqual(len(captured), 2)
        for chunk in captured:
            self.assertIsInstance(chunk, list)

    def test_chunk_size_can_be_set_at_instance_level(self) -> None:
        """A trainer that wants to choose ``chunk_size`` at instantiation
        time (e.g. from config) can set it as an instance attribute;
        Python's attribute lookup picks up the instance attribute over
        the class default."""

        class _ConfigurableChunkTrainer(_ChunkRecordingTrainer):
            chunk_size = 99  # class default; should be shadowed below

            def __init__(self, data_packer: Any, chunk_size: int) -> None:
                super().__init__(data_packer)
                self.chunk_size = chunk_size

        trainer = _ConfigurableChunkTrainer(
            _GoodPacker(transitions_per_rollout=4), chunk_size=2
        )
        trainer.step_training([_make_fake_rollout(0)])

        chunk_events = [e for e in trainer.events if e[0] == "chunk"]
        self.assertEqual([len(e[2]) for e in chunk_events], [2, 2])


class TestTrajectoryExpansionMixinIterRolloutsDelegation(unittest.TestCase):
    """The mixin's outer loop is delegated to ``packer.iter_rollouts``.

    Pins three properties:
      * Default (single-batch) ``iter_rollouts`` produces input-order
        ``_train_one_rollout`` / ``_train_one_chunk`` calls (i.e., the
        delegation is invisible when the default body is in effect).
      * A packer that overrides ``iter_rollouts`` to yield multiple
        batches is honoured: rollouts are emitted batch-by-batch,
        rollout-by-rollout-within-batch.
      * A packer that overrides ``iter_rollouts`` to filter rollouts is
        honoured: skipped rollouts produce zero trainer-side events.

    Batch boundaries do NOT surface as trainer hooks (no
    ``_begin_batch`` / ``_finalize_batch``); the ``begin``/``finalize``
    bookends fire exactly once around the whole step regardless of how
    many batches the packer emits.
    """

    def test_grouped_rollouts_in_rollout_mode_preserves_batch_order(self) -> None:
        """``iter_rollouts`` yields ``[[r5, r2], [r9]]``; mixin emits
        ``train`` events for r5, r2, r9 in that exact order, with one
        ``begin`` and one ``finalize`` total (not per-batch)."""
        packer = _GroupedRolloutsPacker(groups=[(5, 2), (9,)])
        trainer = _RecordingTrainer(packer)
        rollouts = [_make_fake_rollout(i) for i in (5, 2, 9)]

        trainer.step_training(rollouts)

        self.assertEqual(
            trainer.events,
            [
                ("begin", 3),
                ("train", 5),
                ("train", 2),
                ("train", 9),
                ("finalize", 3),
            ],
        )

    def test_grouped_rollouts_in_chunk_mode_preserves_batch_order(self) -> None:
        """Same as above in chunk mode. Per-rollout chunks still come
        from ``iter_chunks``; batch boundaries do not interfere with
        chunking."""
        packer = _GroupedRolloutsPacker(
            groups=[(5,), (9, 2)], transitions_per_rollout=2
        )
        trainer = _ChunkRecordingTrainer(packer)
        rollouts = [_make_fake_rollout(i) for i in (5, 2, 9)]

        trainer.step_training(rollouts)

        self.assertEqual(
            [e[0] for e in trainer.events],
            ["begin", "chunk", "chunk", "chunk", "finalize"],
        )
        self.assertEqual(
            [e[1] for e in trainer.events if e[0] == "chunk"],
            [5, 9, 2],
        )

    def test_iter_rollouts_filtering_skips_rollouts(self) -> None:
        """A packer that omits rollouts from ``iter_rollouts`` makes
        them invisible to the trainer entirely. ``r2`` is in the input
        but the packer's groups do not list it; the trainer never sees
        a ``train`` event for it."""
        packer = _GroupedRolloutsPacker(groups=[(5,), (9,)])
        trainer = _RecordingTrainer(packer)
        rollouts = [_make_fake_rollout(i) for i in (5, 2, 9)]

        trainer.step_training(rollouts)

        prompt_idxs_seen = [e[1] for e in trainer.events if e[0] == "train"]
        self.assertEqual(prompt_idxs_seen, [5, 9])
        self.assertNotIn(2, prompt_idxs_seen)
        # Bookends still fire and report the original input length.
        self.assertEqual(trainer.events[0], ("begin", 3))
        self.assertEqual(trainer.events[-1], ("finalize", 3))

    def test_iter_rollouts_empty_input_no_train_events(self) -> None:
        """The default ``iter_rollouts`` body yields zero batches on an
        empty input; the mixin still fires ``begin`` and ``finalize``
        bookends but no ``train`` event. Symmetric to the existing
        empty-rollouts tests; pins that adding the outer ``for batch in
        iter_rollouts`` loop did not regress empty handling."""
        trainer = _RecordingTrainer(_GoodPacker())
        result = trainer.step_training([])
        self.assertEqual(trainer.events, [("begin", 0), ("finalize", 0)])
        self.assertEqual(result, {"num_rollouts": 0})

    def test_default_iter_rollouts_in_rollout_mode_invisible(self) -> None:
        """With the default ``iter_rollouts`` body (single batch), the
        mixin's behaviour is observably identical to the pre-PR-2 path
        (one ``train`` per rollout in input order). Sanity: introducing
        the outer ``for batch in iter_rollouts`` loop is a no-op when
        the default body is in effect."""
        trainer = _RecordingTrainer(_GoodPacker())
        rollouts = [_make_fake_rollout(i) for i in (5, 2, 9, 7)]

        trainer.step_training(rollouts)

        self.assertEqual(
            trainer.events,
            [
                ("begin", 4),
                ("train", 5),
                ("train", 2),
                ("train", 9),
                ("train", 7),
                ("finalize", 4),
            ],
        )


class TestTrajectoryExpansionMixinHookMisuse(unittest.TestCase):
    def test_rollout_mode_without_train_one_rollout_raises(self) -> None:
        """``chunk_size = None`` (the default) but no override of
        ``_train_one_rollout`` raises ``NotImplementedError`` with a
        diagnostic message naming the trainer class."""

        class _BareRollout(TrajectoryExpansionMixin):
            def __init__(self, data_packer: Any) -> None:
                self.data_packer = data_packer

        trainer = _BareRollout(_GoodPacker())
        with self.assertRaises(NotImplementedError) as ctx:
            trainer.step_training([_make_fake_rollout(0)])
        msg = str(ctx.exception)
        self.assertIn("_BareRollout", msg)
        self.assertIn("chunk_size=None", msg)
        self.assertIn("_train_one_rollout", msg)

    def test_chunk_mode_without_train_one_chunk_raises(self) -> None:
        """``chunk_size`` set but no override of ``_train_one_chunk``
        raises ``NotImplementedError`` with a diagnostic message naming
        both the trainer class and the configured ``chunk_size``."""

        class _BareChunk(TrajectoryExpansionMixin):
            chunk_size = 4

            def __init__(self, data_packer: Any) -> None:
                self.data_packer = data_packer

        trainer = _BareChunk(_GoodPacker())
        with self.assertRaises(NotImplementedError) as ctx:
            trainer.step_training([_make_fake_rollout(0)])
        msg = str(ctx.exception)
        self.assertIn("_BareChunk", msg)
        self.assertIn("chunk_size=4", msg)
        self.assertIn("_train_one_chunk", msg)

    def test_default_chunk_size_is_none(self) -> None:
        """The mixin's class-level default for ``chunk_size`` is ``None``
        (rollout mode). Pinned so future refactors do not silently flip
        the default and change PR-1's semantics."""
        self.assertIsNone(TrajectoryExpansionMixin.chunk_size)


class TestTrajectoryExpansionMixinStateless(unittest.TestCase):
    """The mixin must be stateless across ``step_training`` calls.

    Calling ``step_training`` twice on the same trainer must run the full
    three-phase sequence twice, in order, with no state leaked between
    calls. Trainers can carry their own state (e.g. step counters) but
    the mixin itself contributes none.
    """

    def test_repeated_calls_run_full_sequence_each_time_rollout_mode(self) -> None:
        trainer = _RecordingTrainer(_GoodPacker(transitions_per_rollout=1))
        trainer.step_training([_make_fake_rollout(0)])
        trainer.step_training([_make_fake_rollout(1)])

        self.assertEqual(
            trainer.events,
            [
                ("begin", 1),
                ("train", 0),
                ("finalize", 1),
                ("begin", 1),
                ("train", 1),
                ("finalize", 1),
            ],
        )

    def test_repeated_calls_run_full_sequence_each_time_chunk_mode(self) -> None:
        trainer = _ChunkRecordingTrainer(_GoodPacker(transitions_per_rollout=2))
        trainer.step_training([_make_fake_rollout(0)])
        trainer.step_training([_make_fake_rollout(1)])

        self.assertEqual(
            [e[0] for e in trainer.events],
            ["begin", "chunk", "finalize", "begin", "chunk", "finalize"],
        )

    def test_rollout_and_chunk_subclasses_independent(self) -> None:
        """A rollout-mode trainer and a chunk-mode trainer composed from
        the same mixin do not interfere: each picks the branch matching
        its own ``chunk_size`` regardless of what other subclasses do."""
        rollout_trainer = _RecordingTrainer(_GoodPacker(transitions_per_rollout=1))
        chunk_trainer = _ChunkRecordingTrainer(_GoodPacker(transitions_per_rollout=2))

        rollout_trainer.step_training([_make_fake_rollout(0)])
        chunk_trainer.step_training([_make_fake_rollout(0)])

        self.assertEqual(
            [e[0] for e in rollout_trainer.events],
            ["begin", "train", "finalize"],
        )
        self.assertEqual(
            [e[0] for e in chunk_trainer.events],
            ["begin", "chunk", "finalize"],
        )


if __name__ == "__main__":
    unittest.main()
