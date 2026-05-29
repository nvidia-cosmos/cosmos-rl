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

"""Tests for ``cosmos_rl.rollout.generation_mixin.RolloutGenerationMixin``.

Covers, in order of importance:

* Template-method dispatch (each hook called exactly once per
  ``rollout_generation``, in the documented order).
* Inline path (prefetch off): ``_prepare_sample`` runs on the calling
  thread; behavior is identical to a hand-written
  ``rollout_generation``.
* Prefetch path (prefetch on): ``submit_setup`` schedules
  ``_prepare_sample`` on the bg worker; ``_gather_prepared_samples``
  awaits the matching future; cold-start payloads (no future for a
  key) fall back to inline preparation.
* Overlap: ``_prepare_sample`` for batch B+1 starts before
  ``_generate`` for batch B finishes.
* Error propagation: an exception inside ``_prepare_sample`` flows
  through the future, is re-raised in the consumer thread, and is
  caught by the template's ``except`` block which routes through
  ``_on_generation_error``.
* Lifecycle: ``setup_generation`` is idempotent and
  ``shutdown_generation`` joins cleanly even with pending futures.
* Synthetic-throughput assertion: with ``sleep``-stubbed hooks, the
  prefetch path's wall-clock is reduced by approximately
  ``(n - 1) * min(p, g)`` versus the inline path.
"""

from __future__ import annotations

import logging
import threading
import time
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from cosmos_rl.rollout.generation_mixin import RolloutGenerationMixin
from cosmos_rl.rollout.rollout_base import RolloutBase


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakePayload:
    """Minimal payload that satisfies the mixin's default ``_payload_key``."""

    def __init__(self, idx: int, prompt: str = "p") -> None:
        self.prompt_idx = idx
        self.prompt = prompt


def _fake_config(*, prefetch: bool) -> Any:
    """Build the smallest object that ``setup_generation`` reads."""
    return SimpleNamespace(rollout=SimpleNamespace(prefetch_rollout=prefetch))


class _FakeBackend(RolloutGenerationMixin, RolloutBase):
    """Test double that records hook invocations and supports controllable hook delays.

    Avoids the heavy ``RolloutBase.__init__`` chain by overriding the
    constructor; the mixin doesn't need any of ``RolloutBase``'s
    abstract-method machinery for these tests.  We register the
    abstract surface (``post_init_hook``, ``init_engine``,
    ``get_underlying_model``) trivially.
    """

    def __init__(
        self,
        *,
        prefetch: bool = False,
        prep_ms: int = 0,
        gen_ms: int = 0,
        fail_prepare_for: Optional[set] = None,
    ) -> None:
        self.config = _fake_config(prefetch=prefetch)
        self.parallel_dims = None
        self.device = None
        self._engine_initialized = True  # skip init_engine for unit tests
        self._prep_ms = prep_ms
        self._gen_ms = gen_ms
        self._fail_prepare_for = fail_prepare_for or set()
        self.events: List[str] = []
        self._events_lock = threading.Lock()
        self.thread_names: Dict[str, List[str]] = {
            "_prepare_sample": [],
            "_collate_batch": [],
            "_generate": [],
            "_postprocess": [],
        }
        self.setup_generation(thread_name="FakeBackendPrefetch")

    # The mixin's ``rollout_generation`` is final; the abstract method
    # on ``RolloutBase`` is satisfied via MRO.

    def post_init_hook(self, **kwargs: Any) -> None:
        # Unused: bypassed by our custom __init__.
        pass

    def init_engine(self, *args: Any, **kwargs: Any) -> None:
        self._engine_initialized = True

    def get_underlying_model(self) -> Any:
        return None

    # ------------------------------------------------------------------
    # Hook overrides
    # ------------------------------------------------------------------

    def _record(self, hook: str) -> None:
        with self._events_lock:
            self.events.append(hook)
            self.thread_names[hook].append(threading.current_thread().name)

    def _prepare_sample(
        self, payload, *, data_packer=None, data_fetcher=None, is_validation=False
    ):
        self._record("_prepare_sample")
        if payload.prompt_idx in self._fail_prepare_for:
            raise RuntimeError(
                f"_prepare_sample boom for prompt_idx={payload.prompt_idx}"
            )
        if self._prep_ms:
            time.sleep(self._prep_ms / 1000.0)
        return ("prepared", payload.prompt_idx)

    def _collate_batch(self, samples, *, data_packer=None, is_validation=False):
        self._record("_collate_batch")
        return ("collated", tuple(samples))

    def _generate(self, batch, *, stream=None, is_validation=False):
        self._record("_generate")
        if self._gen_ms:
            time.sleep(self._gen_ms / 1000.0)
        return ("generated", batch)

    def _postprocess(self, raw, payloads, *, is_validation=False):
        self._record("_postprocess")
        return [("result", p.prompt_idx) for p in payloads]


# ---------------------------------------------------------------------------
# Dispatch / inline path
# ---------------------------------------------------------------------------


class TestTemplateMethodDispatch(unittest.TestCase):
    """Each hook fires exactly once per ``rollout_generation``, in the
    documented order, regardless of prefetch state."""

    def _run_dispatch_check(self, prefetch: bool) -> None:
        backend = _FakeBackend(prefetch=prefetch)
        try:
            payloads = [_FakePayload(idx=0), _FakePayload(idx=1)]
            if prefetch:
                backend.submit_setup(payloads)
            results = backend.rollout_generation(payloads)
            self.assertEqual(
                results,
                [("result", 0), ("result", 1)],
                msg="postprocess output should be 1:1 with payloads in order",
            )
            self.assertEqual(
                backend.events,
                # 2 prepare + 1 collate + 1 generate + 1 postprocess.
                [
                    "_prepare_sample",
                    "_prepare_sample",
                    "_collate_batch",
                    "_generate",
                    "_postprocess",
                ],
            )
        finally:
            backend.shutdown_generation()

    def test_dispatch_order_inline(self) -> None:
        self._run_dispatch_check(prefetch=False)

    def test_dispatch_order_prefetch(self) -> None:
        self._run_dispatch_check(prefetch=True)


class TestInlinePath(unittest.TestCase):
    """Prefetch off: every hook (including prepare) runs on the calling thread."""

    def test_prepare_runs_on_caller_thread_when_prefetch_off(self) -> None:
        backend = _FakeBackend(prefetch=False)
        try:
            payloads = [_FakePayload(idx=0), _FakePayload(idx=1)]
            backend.rollout_generation(payloads)
            caller = threading.current_thread().name
            for hook, names in backend.thread_names.items():
                for n in names:
                    self.assertEqual(
                        n,
                        caller,
                        msg=f"hook {hook} ran on {n!r}, expected {caller!r}",
                    )
        finally:
            backend.shutdown_generation()

    def test_submit_setup_is_noop_when_prefetch_off(self) -> None:
        backend = _FakeBackend(prefetch=False)
        try:
            # Should not crash and should not record any prepare events.
            backend.submit_setup([_FakePayload(idx=0)])
            self.assertEqual(backend.events, [])
        finally:
            backend.shutdown_generation()


# ---------------------------------------------------------------------------
# Prefetch path
# ---------------------------------------------------------------------------


class TestPrefetchPath(unittest.TestCase):
    """Prefetch on: ``_prepare_sample`` runs on the bg worker; the main thread
    only sees collate / generate / postprocess."""

    def test_prepare_runs_on_background_thread(self) -> None:
        backend = _FakeBackend(prefetch=True)
        try:
            payloads = [_FakePayload(idx=0), _FakePayload(idx=1)]
            backend.submit_setup(payloads)
            backend.rollout_generation(payloads)
            caller = threading.current_thread().name
            for n in backend.thread_names["_prepare_sample"]:
                self.assertNotEqual(
                    n,
                    caller,
                    msg=f"_prepare_sample ran on caller thread {n!r}; "
                    "expected the bg worker thread",
                )
                self.assertEqual(n, "FakeBackendPrefetch")
            # collate / generate / postprocess always run on the caller.
            for hook in ("_collate_batch", "_generate", "_postprocess"):
                self.assertEqual(backend.thread_names[hook], [caller])
        finally:
            backend.shutdown_generation()

    def test_cold_start_payload_falls_back_to_inline_prepare(self) -> None:
        """A payload arriving at ``rollout_generation`` without a matching
        ``submit_setup`` runs ``_prepare_sample`` inline."""
        backend = _FakeBackend(prefetch=True)
        try:
            submitted = [_FakePayload(idx=0)]
            cold = _FakePayload(idx=99)
            backend.submit_setup(submitted)
            backend.rollout_generation([submitted[0], cold])
            caller = threading.current_thread().name
            # Two prepare invocations: one bg (for idx=0), one inline (for idx=99).
            self.assertEqual(len(backend.thread_names["_prepare_sample"]), 2)
            self.assertIn(caller, backend.thread_names["_prepare_sample"])
            self.assertIn(
                "FakeBackendPrefetch", backend.thread_names["_prepare_sample"]
            )
        finally:
            backend.shutdown_generation()

    def test_resubmission_replaces_pending_future(self) -> None:
        """Submitting the same key twice replaces the prior future; the
        consumer awaits the latest one."""
        backend = _FakeBackend(prefetch=True)
        try:
            p = _FakePayload(idx=42)
            backend.submit_setup([p])
            backend.submit_setup([p])  # replace
            results = backend.rollout_generation([p])
            self.assertEqual(results, [("result", 42)])
            # At least one prepare ran (the cancelled prior one may or may
            # not have raced through depending on timing).
            self.assertGreaterEqual(len(backend.thread_names["_prepare_sample"]), 1)
        finally:
            backend.shutdown_generation()


class TestOverlap(unittest.TestCase):
    """The bg worker keeps making progress while the main thread is in
    ``_generate``."""

    def test_prepare_for_next_batch_overlaps_with_generate(self) -> None:
        # Slow generate (200ms), fast prepare (50ms each).  Submit batch B+1
        # *before* calling rollout_generation(B), then verify B+1's prepare
        # finished while B's generate was running.
        backend = _FakeBackend(prefetch=True, prep_ms=50, gen_ms=200)
        try:
            b1 = [_FakePayload(idx=0)]
            b2 = [_FakePayload(idx=1)]
            backend.submit_setup(b1)
            backend.submit_setup(b2)

            # Run B1; meanwhile bg is preparing B1[0] -> B2[0].
            t0 = time.monotonic()
            backend.rollout_generation(b1)
            t1 = time.monotonic()

            # By now, bg should have done both prepares (50ms + 50ms = 100ms,
            # well within B1's gather (50ms) + generate (200ms) = 250ms).
            with backend._setup_futures_lock:
                self.assertIn(("idx", 1), backend._setup_futures)
                self.assertTrue(
                    backend._setup_futures[("idx", 1)].done(),
                    msg="B2[0] prepare should be done by the time B1 finishes",
                )

            # Sanity: B1's wall clock should still cover the gather + generate
            # (~250ms), not double that (which would imply the bg never
            # progressed during generate).
            self.assertGreater(t1 - t0, 0.20)
            self.assertLess(t1 - t0, 0.50)
        finally:
            backend.shutdown_generation()


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


class TestErrorPropagation(unittest.TestCase):
    """Exceptions in the data-shaping hooks flow to ``_on_generation_error``;
    ``rollout_generation`` returns ``[]`` instead of crashing the worker.

    Engine-not-initialized is *not* swallowed; that's a programming
    error and propagates as ``RuntimeError``.
    """

    def test_prepare_failure_returns_empty_inline(self) -> None:
        backend = _FakeBackend(prefetch=False, fail_prepare_for={1})
        try:
            payloads = [_FakePayload(idx=0), _FakePayload(idx=1)]
            results = backend.rollout_generation(payloads)
            self.assertEqual(results, [])
            # _generate / _postprocess should NOT have run.
            self.assertEqual(backend.thread_names["_generate"], [])
            self.assertEqual(backend.thread_names["_postprocess"], [])
        finally:
            backend.shutdown_generation()

    def test_prepare_failure_returns_empty_prefetch(self) -> None:
        backend = _FakeBackend(prefetch=True, fail_prepare_for={1})
        try:
            payloads = [_FakePayload(idx=0), _FakePayload(idx=1)]
            backend.submit_setup(payloads)
            results = backend.rollout_generation(payloads)
            self.assertEqual(results, [])
        finally:
            backend.shutdown_generation()

    def test_engine_not_initialized_propagates(self) -> None:
        backend = _FakeBackend(prefetch=False)
        backend._engine_initialized = False
        try:
            with self.assertRaises(RuntimeError):
                backend.rollout_generation([_FakePayload(idx=0)])
        finally:
            backend.shutdown_generation()

    def test_subclass_preflight_check_propagates(self) -> None:
        """Backends extending ``_preflight_check`` get loud propagation
        outside the template's ``try``/``except``.
        """

        class _Picky(_FakeBackend):
            def _preflight_check(self) -> None:
                super()._preflight_check()
                raise RuntimeError("policy not attached")

        backend = _Picky(prefetch=False)
        try:
            with self.assertRaisesRegex(RuntimeError, "policy not attached"):
                backend.rollout_generation([_FakePayload(idx=0)])
            # _generate / _postprocess must not have run.
            self.assertEqual(backend.thread_names["_generate"], [])
            self.assertEqual(backend.thread_names["_postprocess"], [])
        finally:
            backend.shutdown_generation()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle(unittest.TestCase):
    def test_setup_generation_is_idempotent(self) -> None:
        backend = _FakeBackend(prefetch=True)
        try:
            first_thread = backend._setup_thread
            backend.setup_generation()  # second call
            backend.setup_generation()  # third call
            self.assertIs(backend._setup_thread, first_thread)
            self.assertTrue(backend._setup_thread.is_alive())
        finally:
            backend.shutdown_generation()

    def test_shutdown_generation_is_safe_when_not_initialized(self) -> None:
        backend = _FakeBackend.__new__(_FakeBackend)
        # Don't call any setup; just confirm shutdown doesn't blow up.
        backend.shutdown_generation()  # no-op

    def test_shutdown_generation_clears_pending_futures(self) -> None:
        backend = _FakeBackend(prefetch=True, prep_ms=200)
        # Submit but do NOT consume; then shut down.
        payloads = [_FakePayload(idx=i) for i in range(5)]
        backend.submit_setup(payloads)
        backend.shutdown_generation()
        # After shutdown, no futures should remain in the map.
        self.assertEqual(backend._setup_futures, {})
        # And the worker thread should have exited.
        self.assertFalse(backend._setup_thread.is_alive())


# ---------------------------------------------------------------------------
# Synthetic-throughput assertion
# ---------------------------------------------------------------------------


class TestPrefetchThroughputSynthetic(unittest.TestCase):
    """Stub-hook microbenchmark: prefetch should reduce wall-clock by
    approximately ``(n - 1) * min(p, g)`` compared to the inline path,
    where ``n`` is the number of batches and ``p`` / ``g`` are the
    per-payload prepare time and per-batch generate time.

    The assertion is loose (>= 30% of the theoretical max) to tolerate
    GIL contention, thread-startup overhead, and CI scheduler noise.
    The point is to catch a *regression* of the prefetch mechanism (e.g.
    accidentally running prepare on the main thread), not to claim a
    specific real-world speedup.
    """

    def _measure(
        self, prefetch: bool, n_batches: int, batch_size: int, prep_ms: int, gen_ms: int
    ) -> float:
        backend = _FakeBackend(prefetch=prefetch, prep_ms=prep_ms, gen_ms=gen_ms)
        try:
            batches = [
                [_FakePayload(idx=b * batch_size + i) for i in range(batch_size)]
                for b in range(n_batches)
            ]
            if prefetch:
                # Mirror what the rollout worker's _prefetch_loop does:
                # submit every batch up-front so the bg worker has work
                # queued ahead of the consumer.
                for batch in batches:
                    backend.submit_setup(batch)
            t0 = time.monotonic()
            for batch in batches:
                backend.rollout_generation(batch)
            return time.monotonic() - t0
        finally:
            backend.shutdown_generation()

    def test_prefetch_reduces_wallclock(self) -> None:
        n_batches = 4
        batch_size = 1
        prep_ms = 50
        gen_ms = 50
        # Theoretical: inline = n*(p+g); prefetch = (p+g) + (n-1)*max(p, g).
        # Win = (n-1)*min(p, g) = 3 * 50ms = 150ms out of 400ms.
        inline = self._measure(False, n_batches, batch_size, prep_ms, gen_ms)
        prefetch = self._measure(True, n_batches, batch_size, prep_ms, gen_ms)
        win = inline - prefetch
        theoretical_max_win = (n_batches - 1) * min(prep_ms, gen_ms) / 1000.0
        # Require at least 30% of theoretical max to absorb noise.
        self.assertGreater(
            win,
            0.30 * theoretical_max_win,
            msg=(
                f"prefetch did not noticeably reduce wall-clock: "
                f"inline={inline * 1000:.1f}ms, prefetch={prefetch * 1000:.1f}ms, "
                f"win={win * 1000:.1f}ms, theoretical_max={theoretical_max_win * 1000:.1f}ms"
            ),
        )


class _ListHandler(logging.Handler):
    """Minimal handler that just appends formatted records to a list.
    Avoids ``assertLogs`` pitfalls when interacting with cosmos's
    pre-configured default logger."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record.getMessage())


class TestTraceEmission(unittest.TestCase):
    """The per-batch ``[mixin-trace]`` DEBUG line is the contract that
    downstream analyzers parse to prove prefetch is overlapping.  Any
    change to its shape or fields is a public-surface break.
    """

    def _wire_logger(self, backend, level: int = logging.DEBUG) -> _ListHandler:
        """Replace backend._gen_logger with a clean isolated logger so
        we don't fight cosmos's default logger / handler config.
        """
        test_logger = logging.getLogger(
            f"test_mixin_trace.{self._testMethodName}.{id(backend)}"
        )
        test_logger.handlers.clear()
        test_logger.setLevel(level)
        test_logger.propagate = False
        handler = _ListHandler()
        test_logger.addHandler(handler)
        backend._gen_logger = test_logger
        return handler

    def test_trace_line_present_at_debug_level(self) -> None:
        backend = _FakeBackend(prefetch=False, prep_ms=5, gen_ms=5)
        handler = self._wire_logger(backend, logging.DEBUG)
        try:
            backend.rollout_generation([_FakePayload(idx=0), _FakePayload(idx=1)])
        finally:
            backend.shutdown_generation()
        mixin_lines = [m for m in handler.records if "[mixin-trace]" in m]
        self.assertEqual(
            len(mixin_lines),
            1,
            f"expected exactly one trace line, got: {mixin_lines!r} (all records: {handler.records!r})",
        )

    def test_trace_line_silent_above_debug(self) -> None:
        backend = _FakeBackend(prefetch=False, prep_ms=0, gen_ms=0)
        handler = self._wire_logger(backend, logging.INFO)
        try:
            backend.rollout_generation([_FakePayload(idx=0)])
        finally:
            backend.shutdown_generation()
        mixin_lines = [m for m in handler.records if "[mixin-trace]" in m]
        self.assertEqual(
            mixin_lines,
            [],
            f"trace line leaked above DEBUG level: {mixin_lines!r}",
        )

    def test_trace_fields_present_inline(self) -> None:
        backend = _FakeBackend(prefetch=False, prep_ms=5, gen_ms=5)
        handler = self._wire_logger(backend, logging.DEBUG)
        try:
            backend.rollout_generation([_FakePayload(idx=0), _FakePayload(idx=1)])
        finally:
            backend.shutdown_generation()
        mixin_lines = [m for m in handler.records if "[mixin-trace]" in m]
        self.assertEqual(len(mixin_lines), 1, f"records: {handler.records!r}")
        line = mixin_lines[0]
        for field in (
            "batch_size=2",
            "prefetch=False",
            "gather_ms=",
            "collate_ms=",
            "generate_ms=",
            "postprocess_ms=",
            "inline_count=2",
            "inline_ms=",
            "wait_count=0",
            "wait_ms=0.00",
        ):
            self.assertIn(field, line, f"missing {field!r} in trace line: {line}")

    def test_trace_fields_present_prefetch(self) -> None:
        backend = _FakeBackend(prefetch=True, prep_ms=5, gen_ms=5)
        handler = self._wire_logger(backend, logging.DEBUG)
        try:
            payloads = [_FakePayload(idx=i) for i in range(2)]
            backend.submit_setup(payloads)
            # Let the bg worker drain so wait_ms ≈ 0 (full overlap).
            time.sleep(0.05)
            backend.rollout_generation(payloads)
        finally:
            backend.shutdown_generation()
        mixin_lines = [m for m in handler.records if "[mixin-trace]" in m]
        self.assertEqual(len(mixin_lines), 1, f"records: {handler.records!r}")
        line = mixin_lines[0]
        # With prefetch on AND bg worker drained before
        # rollout_generation, every payload's future should be ready
        # so wait_count=2 but wait_ms should be tiny.  inline_count=0.
        for field in (
            "batch_size=2",
            "prefetch=True",
            "inline_count=0",
            "wait_count=2",
        ):
            self.assertIn(field, line, f"missing {field!r} in trace line: {line}")


if __name__ == "__main__":
    unittest.main()
