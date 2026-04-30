# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Tests for the transport-agnostic PrefetchDataPackerMixin (MR5).

These tests stub out :meth:`_fetch_batch` with an in-memory function so
the entire scheduler / double-buffer / dispatch state machine can be
exercised without any real transport (UCXX, NCCL, …).  When a future
NCCLDataPackerMixin lands it will pick up the same scheduling
guarantees, and these tests guard the contract.
"""

import time
import unittest
from typing import Any, Dict, List

from cosmos_rl.utils.payload_transport.prefetch_mixin import (
    PrefetchDataPackerMixin,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _StubDataPacker:
    """Records get_policy_input calls so we can verify what was forwarded."""

    def __init__(self):
        self.calls: List[dict] = []

    def get_policy_input(
        self,
        sample: Any = None,
        rollout_output: Any = None,
        n_ignore_prefix_tokens: int = 0,
        **kwargs,
    ) -> Any:
        self.calls.append(
            {
                "sample": sample,
                "rollout_output": rollout_output,
                "n_ignore_prefix_tokens": n_ignore_prefix_tokens,
                "kwargs": kwargs,
            }
        )
        return rollout_output


class _PassthroughPacker(PrefetchDataPackerMixin, _StubDataPacker):
    """No subclass overrides -> base mixin is a transparent pass-through."""


class _FakeTransportPacker(PrefetchDataPackerMixin, _StubDataPacker):
    """A minimal transport plug-in.

    Intercept rule: rollout_output is a dict carrying ``{"_fake": True}``.
    Cache key:      the dict's ``"id"`` field.
    Fetch:          returns ``{"resolved": id, "from": "prefetch"}`` for
                    each task.  Counts invocations so tests can assert
                    background activity.
    """

    def __init__(self):
        super().__init__()
        self.fetch_calls = 0
        self.fetch_should_raise: BaseException | None = None
        self.sync_returns: Dict[str, Any] = {}
        self.resolve_failed_calls: List[tuple] = []
        self.complete_calls: List[tuple] = []

    def _should_intercept(self, rollout_output: Any) -> bool:
        return isinstance(rollout_output, dict) and bool(rollout_output.get("_fake"))

    def _cache_key(self, rollout_output: Any) -> str:
        return str(rollout_output.get("id"))

    def _fetch_batch(self, tasks: List[Any]) -> Dict[str, Any]:
        self.fetch_calls += 1
        if self.fetch_should_raise is not None:
            raise self.fetch_should_raise
        out: Dict[str, Any] = {}
        for _idx, ref in tasks:
            key = self._cache_key(ref)
            out[key] = {"resolved": key, "from": "prefetch"}
        return out

    def _sync_fetch(self, rollout_output: Any) -> Any:
        return self.sync_returns.get(self._cache_key(rollout_output))

    def _on_prefetch_complete(self, batch_id: int, n_results: int, fetch_ms: float):
        self.complete_calls.append((batch_id, n_results, fetch_ms))

    def _on_resolve_failed(self, rollout_output: Any, cache_key: str) -> None:
        self.resolve_failed_calls.append((rollout_output, cache_key))


# ---------------------------------------------------------------------------
# Pure pass-through (no subclass overrides) -- regression guard for the
# "transport-agnostic" promise
# ---------------------------------------------------------------------------


class TestBaseIsPassThrough(unittest.TestCase):
    def setUp(self):
        self.p = _PassthroughPacker()

    def test_get_policy_input_passes_dict_through(self):
        traj = {"observations": [1, 2, 3]}
        out = self.p.get_policy_input(rollout_output=traj)
        self.assertIs(out, traj)
        self.assertEqual(len(self.p.calls), 1)
        self.assertIs(self.p.calls[0]["rollout_output"], traj)

    def test_get_policy_input_passes_string_through(self):
        out = self.p.get_policy_input(rollout_output="raw")
        self.assertEqual(out, "raw")

    def test_get_policy_input_passes_none_through(self):
        # rollout_output=None -> should call super() with None, never
        # try to intercept.
        self.p.get_policy_input(rollout_output=None)
        self.assertEqual(len(self.p.calls), 1)
        self.assertIsNone(self.p.calls[0]["rollout_output"])

    def test_cold_start_initially_true(self):
        self.assertTrue(self.p.is_cold_start)
        self.assertIsNone(self.p.prefetch_buffer)

    def test_start_prefetch_noop_when_setup_not_called(self):
        # Background thread isn't running; start should be silent no-op.
        self.p.start_prefetch([{"_fake": True, "id": 1}])
        # no exception, buffer untouched
        self.assertIsNone(self.p.prefetch_buffer)


# ---------------------------------------------------------------------------
# Subclass-hook contract enforcement
# ---------------------------------------------------------------------------


class TestSubclassHookContract(unittest.TestCase):
    def test_cache_key_must_be_overridden_when_intercepting(self):
        class _Bad(PrefetchDataPackerMixin, _StubDataPacker):
            def _should_intercept(self, ro):  # noqa: D401
                return True

            # _cache_key intentionally not overridden

        with self.assertRaises(NotImplementedError):
            _Bad().get_policy_input(rollout_output={"x": 1})

    def test_fetch_batch_must_be_implemented(self):
        class _Bad(PrefetchDataPackerMixin, _StubDataPacker):
            pass

        # Direct invocation triggers the NotImplementedError; this is
        # what the worker thread would hit too.
        with self.assertRaises(NotImplementedError):
            _Bad()._fetch_batch([("idx", "ref")])


# ---------------------------------------------------------------------------
# Double-buffer / early-ack state machine (without background thread)
# ---------------------------------------------------------------------------


class TestDoubleBufferStateMachine(unittest.TestCase):
    """The defer/collect cycle is pure data-flow and works even when the
    background thread isn't running, because cold-start defer doesn't
    issue any wait."""

    def setUp(self):
        self.p = _PassthroughPacker()

    def test_defer_seeds_buffer_on_cold_start(self):
        rollouts = ["r0", "r1"]
        self.p.defer_prefetch(rollouts)
        self.assertFalse(self.p.is_cold_start)
        self.assertEqual(self.p.prefetch_buffer, rollouts)
        self.assertFalse(self.p._prefetch_pending)

    def test_defer_after_seed_marks_pending(self):
        self.p.defer_prefetch(["r0"])
        self.p.defer_prefetch(["r1"])
        self.assertTrue(self.p._prefetch_pending)
        self.assertEqual(self.p._prefetch_rollouts, ["r1"])

    def test_collect_returns_none_when_cold(self):
        self.assertIsNone(self.p.collect_prefetch())

    def test_collect_returns_buffer_when_seeded(self):
        self.p.defer_prefetch(["a"])
        self.assertEqual(self.p.collect_prefetch(), ["a"])


# ---------------------------------------------------------------------------
# Scheduler with live background thread (the main integration test)
# ---------------------------------------------------------------------------


class TestSchedulerWithBackgroundThread(unittest.TestCase):
    def setUp(self):
        self.p = _FakeTransportPacker()
        # Short timeout so a misbehaving test fails fast rather than
        # waiting for the default 5min.
        self.p._setup_prefetch(prefetch_timeout=2.0, thread_name="TestPrefetch")

    def tearDown(self):
        self.p.shutdown_prefetch()

    def _wait_for(self, predicate, timeout=2.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return False

    def test_start_then_wait_populates_cache(self):
        rollouts = [
            {"_fake": True, "id": "a"},
            {"_fake": True, "id": "b"},
            {"_other": True},  # should be filtered out by _should_intercept
        ]
        self.p.start_prefetch(rollouts)
        self.p.wait_prefetch()
        self.assertEqual(self.p._prefetch_cache.keys(), {"a", "b"})
        self.assertEqual(self.p._prefetch_cache["a"]["from"], "prefetch")
        self.assertEqual(self.p.fetch_calls, 1)
        self.assertEqual(len(self.p.complete_calls), 1)

    def test_get_policy_input_uses_cache(self):
        ref = {"_fake": True, "id": "x"}
        self.p.start_prefetch([ref])
        self.p.wait_prefetch()
        out = self.p.get_policy_input(rollout_output=ref)
        # The base mixin replaces the ref with the resolved payload
        # before delegating to super().
        self.assertEqual(out, {"resolved": "x", "from": "prefetch"})

    def test_get_policy_input_falls_back_to_sync_fetch(self):
        ref = {"_fake": True, "id": "y"}
        self.p.sync_returns["y"] = {"resolved": "y", "from": "sync"}
        # No prefetch issued -> cache miss -> _sync_fetch path.
        out = self.p.get_policy_input(rollout_output=ref)
        self.assertEqual(out["from"], "sync")
        self.assertEqual(self.p.resolve_failed_calls, [])

    def test_get_policy_input_resolve_failed_hook_fires(self):
        ref = {"_fake": True, "id": "z"}
        # No prefetch + no sync return -> resolution fails.
        out = self.p.get_policy_input(rollout_output=ref)
        self.assertIsNone(out)
        self.assertEqual(len(self.p.resolve_failed_calls), 1)
        self.assertEqual(self.p.resolve_failed_calls[0][1], "z")

    def test_fetch_exception_is_captured_as_error_payload(self):
        self.p.fetch_should_raise = RuntimeError("boom")
        self.p.start_prefetch([{"_fake": True, "id": "q"}])
        self.p.wait_prefetch()
        # Cache cleared because the worker reported an error.
        self.assertEqual(self.p._prefetch_cache, {})

    def test_wait_prefetch_timeout_clears_cache(self):
        # No request submitted -> wait should time out and reset cache.
        self.p._prefetch_cache = {"stale": 1}
        self.p._prefetch_timeout_s = 0.05
        self.p.wait_prefetch()
        self.assertEqual(self.p._prefetch_cache, {})

    def test_defer_then_collect_drives_full_cycle(self):
        # Cycle 1: cold start -> seed buffer.
        ref0 = {"_fake": True, "id": "iter0"}
        self.p.start_prefetch([ref0])
        self.p.wait_prefetch()  # populates cache for iter0
        self.p.defer_prefetch([ref0])  # cold-start defer just seeds buffer
        self.assertEqual(self.p.prefetch_buffer, [ref0])
        self.assertFalse(self.p._prefetch_pending)

        # Cycle 2: prefetch next, defer (now pending), collect rotates.
        ref1 = {"_fake": True, "id": "iter1"}
        self.p.start_prefetch([ref1])
        self.p.defer_prefetch([ref1])
        self.assertTrue(self.p._prefetch_pending)
        out = self.p.collect_prefetch()
        self.assertEqual(out, [ref1])
        self.assertEqual(self.p._prefetch_cache.keys(), {"iter1"})

    def test_setup_prefetch_is_idempotent(self):
        # Already set up in setUp.  A second setup must not start a
        # second thread or wipe the existing cache.
        self.p._prefetch_cache = {"keep_me": 1}
        original_thread = self.p._prefetch_thread
        self.p._setup_prefetch(prefetch_timeout=10.0)
        self.assertIs(self.p._prefetch_thread, original_thread)
        self.assertEqual(self.p._prefetch_cache, {"keep_me": 1})
        # But the timeout was refreshed.
        self.assertEqual(self.p._prefetch_timeout_s, 10.0)

    def test_filter_default_only_includes_intercepted(self):
        # Default _filter_prefetch_tasks delegates to _should_intercept.
        rollouts = [
            {"_fake": True, "id": "yes1"},
            {"_other": True, "id": "no"},
            {"_fake": True, "id": "yes2"},
        ]
        tasks = self.p._filter_prefetch_tasks(rollouts)
        self.assertEqual([t[1]["id"] for t in tasks], ["yes1", "yes2"])


# ---------------------------------------------------------------------------
# Shutdown semantics
# ---------------------------------------------------------------------------


class TestShutdown(unittest.TestCase):
    def test_shutdown_is_idempotent_when_never_set_up(self):
        p = _PassthroughPacker()
        # Calling shutdown without a prior setup must not raise.
        p.shutdown_prefetch()
        p.shutdown_prefetch()  # twice for good measure

    def test_shutdown_stops_thread(self):
        p = _FakeTransportPacker()
        p._setup_prefetch(prefetch_timeout=1.0)
        thread = p._prefetch_thread
        self.assertTrue(thread.is_alive())
        p.shutdown_prefetch()
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())
        self.assertFalse(p._prefetch_enabled)


if __name__ == "__main__":
    unittest.main()
