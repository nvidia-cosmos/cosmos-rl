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

"""Composed-transport seam: does the scheduler actually route through a strategy?

These drive the real scheduling layer -- background worker, cache, early-ack --
rather than asserting that methods exist, because the failure this guards
against is a hook that is *defined* but never *reached*: the packer would
silently behave as if no transport were attached and every payload would take
the fallback path.
"""

import threading
import unittest
import unittest.mock
from typing import Any, Dict, List, Optional

from cosmos_rl.utils.payload_transport.prefetch_mixin import PrefetchDataPackerMixin
from cosmos_rl.utils.payload_transport.registry import make_transport_strategy
from cosmos_rl.utils.payload_transport.strategy import PayloadTransportStrategy


class _RecordingStrategy(PayloadTransportStrategy):
    """Minimal strategy that records which hooks the scheduler invoked."""

    def __init__(self, *, sync_result: Any = None):
        self.calls: List[str] = []
        self.payloads: Dict[str, Any] = {}
        self.sync_result = sync_result
        self.completed: List[tuple] = []
        self.failed: List[str] = []
        self.before_join_calls = 0
        self.fetch_started = threading.Event()

    # -- recognition ------------------------------------------------------
    def should_intercept(self, rollout_output: Any) -> bool:
        self.calls.append("should_intercept")
        return isinstance(rollout_output, str) and rollout_output.startswith("ref:")

    def cache_key(self, rollout_output: Any) -> str:
        self.calls.append("cache_key")
        return f"key::{rollout_output}"

    # -- fetching ---------------------------------------------------------
    def fetch_batch(self, tasks: List[Any]) -> Dict[str, Any]:
        self.calls.append("fetch_batch")
        self.fetch_started.set()
        out = {}
        for _idx, ro in tasks:
            out[self.cache_key(ro)] = self.payloads.get(ro, {"payload": ro})
        return out

    def sync_fetch(self, rollout_output: Any) -> Optional[Any]:
        self.calls.append("sync_fetch")
        return self.sync_result

    # -- telemetry --------------------------------------------------------
    def on_prefetch_complete(self, batch_id, n_results, fetch_ms, step) -> None:
        self.calls.append("on_prefetch_complete")
        self.completed.append((batch_id, n_results, step))

    def on_resolve_failed(self, rollout_output: Any, cache_key: str) -> None:
        self.calls.append("on_resolve_failed")
        self.failed.append(cache_key)

    def before_join(self) -> None:
        self.before_join_calls += 1


class _BasePacker:
    """Stands in for the real DataPacker below the mixin.

    The mixin resolves a reference and hands the *resolved payload* down via
    ``super().get_policy_input``, so tagging what arrives here is how we tell
    "resolved through the strategy" from "passed through untouched".
    """

    def get_policy_input(
        self, sample=None, rollout_output=None, n_ignore_prefix_tokens=0, **kw
    ):
        return {"received": rollout_output}


class _Packer(PrefetchDataPackerMixin, _BasePacker):
    pass


class TestStrategyDelegation(unittest.TestCase):
    def setUp(self):
        self.packer = _Packer()
        self.strategy = _RecordingStrategy()
        self.packer.set_transport_strategy(self.strategy)

    def tearDown(self):
        self.packer.shutdown_prefetch(join_timeout=5.0)

    def test_prefetched_payload_is_served_from_the_strategy(self):
        """The whole point: a ref resolves via the strategy, not the fallback."""
        self.packer._setup_prefetch(prefetch_timeout=10.0)
        self.packer.start_prefetch(["ref:a", "ref:b"])
        self.packer.wait_prefetch()

        out = self.packer.get_policy_input(rollout_output="ref:a")
        # Resolved by the strategy, then handed to the base packer.
        self.assertEqual(out, {"received": {"payload": "ref:a"}})
        self.assertIn("fetch_batch", self.strategy.calls)

    def test_non_matching_payload_falls_through_untouched(self):
        """should_intercept gates the whole path; a plain payload must pass by."""
        self.packer._setup_prefetch(prefetch_timeout=10.0)
        out = self.packer.get_policy_input(rollout_output="plain-text")
        # Reached the base packer unresolved -- exactly as handed in.
        self.assertEqual(out, {"received": "plain-text"})
        self.assertNotIn("fetch_batch", self.strategy.calls)

    def test_cache_miss_falls_back_to_strategy_sync_fetch(self):
        self.strategy.sync_result = {"payload": "from-sync"}
        self.packer._setup_prefetch(prefetch_timeout=10.0)

        # Never prefetched, so the cache cannot serve it.
        out = self.packer.get_policy_input(rollout_output="ref:never-prefetched")
        self.assertEqual(out, {"received": {"payload": "from-sync"}})
        self.assertIn("sync_fetch", self.strategy.calls)

    def test_unresolvable_reference_reports_through_on_resolve_failed(self):
        self.strategy.sync_result = None  # both cache and sync come up empty
        self.packer._setup_prefetch(prefetch_timeout=10.0)

        self.packer.get_policy_input(rollout_output="ref:gone")
        self.assertEqual(self.strategy.failed, ["key::ref:gone"])

    def test_on_prefetch_complete_receives_the_iteration_counter(self):
        """step is passed in rather than read off the packer -- verify it advances."""
        self.packer._setup_prefetch(prefetch_timeout=10.0)
        for _ in range(2):
            self.packer.start_prefetch(["ref:a"])
            self.packer.wait_prefetch()

        steps = [step for (_bid, _n, step) in self.strategy.completed]
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps, sorted(steps))
        self.assertGreater(steps[-1], 0)

    def test_shutdown_calls_strategy_before_join_by_default(self):
        """Teardown must unwedge transport I/O without the caller arranging it."""
        self.packer._setup_prefetch(prefetch_timeout=10.0)
        self.packer.shutdown_prefetch(join_timeout=5.0)
        self.assertEqual(self.strategy.before_join_calls, 1)

    def test_explicit_before_join_overrides_the_strategy(self):
        self.packer._setup_prefetch(prefetch_timeout=10.0)
        called = []
        self.packer.shutdown_prefetch(
            join_timeout=5.0, before_join=lambda: called.append(1)
        )
        self.assertEqual(called, [1])
        self.assertEqual(self.strategy.before_join_calls, 0)


class TestWithoutStrategy(unittest.TestCase):
    """No strategy attached: the mixin must stay an inert pass-through."""

    def setUp(self):
        self.packer = _Packer()

    def test_never_intercepts(self):
        self.assertFalse(self.packer._should_intercept("ref:a"))
        out = self.packer.get_policy_input(rollout_output="ref:a")
        self.assertEqual(out, {"received": "ref:a"})

    def test_cache_key_and_fetch_batch_still_raise(self):
        with self.assertRaises(NotImplementedError):
            self.packer._cache_key("ref:a")
        with self.assertRaises(NotImplementedError):
            self.packer._fetch_batch([(0, "ref:a")])

    def test_detaching_restores_pass_through(self):
        self.packer.set_transport_strategy(_RecordingStrategy())
        self.assertTrue(self.packer._should_intercept("ref:a"))
        self.packer.set_transport_strategy(None)
        self.assertFalse(self.packer._should_intercept("ref:a"))


class TestSubclassPathStillWins(unittest.TestCase):
    """The mixin path predates strategies and must keep working unchanged."""

    def test_subclass_override_beats_an_attached_strategy(self):
        class _Overriding(_Packer):
            def _should_intercept(self, rollout_output):
                return rollout_output == "only-this"

        packer = _Overriding()
        packer.set_transport_strategy(_RecordingStrategy())

        # The strategy would accept "ref:a"; the override must win.
        self.assertFalse(packer._should_intercept("ref:a"))
        self.assertTrue(packer._should_intercept("only-this"))

    def test_subclass_hooks_work_with_no_strategy_at_all(self):
        class _SubclassTransport(_Packer):
            def _should_intercept(self, rollout_output):
                return rollout_output == "mine"

            def _cache_key(self, rollout_output):
                return "k"

            def _fetch_batch(self, tasks):
                return {"k": {"payload": "via-subclass"}}

        packer = _SubclassTransport()
        packer._setup_prefetch(prefetch_timeout=10.0)
        try:
            packer.start_prefetch(["mine"])
            packer.wait_prefetch()
            self.assertEqual(
                packer.get_policy_input(rollout_output="mine"),
                {"received": {"payload": "via-subclass"}},
            )
        finally:
            packer.shutdown_prefetch(join_timeout=5.0)


class TestFactorySelectsFromConfig(unittest.TestCase):
    """Selecting a transport from config is the whole point of the seam."""

    @staticmethod
    def _config(mode, tp=1, cp=1, pp=1):
        class _Parallelism:
            tp_size, cp_size, pp_size = tp, cp, pp

        class _Policy:
            parallelism = _Parallelism()

        class _Config:
            policy = _Policy()
            custom = {} if mode is None else {"payload_transfer": mode}

        return _Config()

    def test_builds_the_transport_the_config_names(self):
        from cosmos_rl.utils.payload_transport.nccl.strategy import (
            NCCLTransportStrategy,
        )
        from cosmos_rl.utils.payload_transport.ucxx.strategy import (
            UCXXTransportStrategy,
        )

        self.assertIsInstance(
            make_transport_strategy(self._config("nccl")), NCCLTransportStrategy
        )
        self.assertIsInstance(
            make_transport_strategy(self._config("ucxx")), UCXXTransportStrategy
        )

    def test_redis_needs_no_strategy(self):
        """redis moves payloads through the control plane -- nothing to compose.

        None is the right answer rather than a null object: an unattached packer
        already passes everything through, which is exactly redis's behaviour.
        """
        self.assertIsNone(make_transport_strategy(self._config(None)))

    def test_rejects_a_topology_neither_transport_can_serve(self):
        """The guard must fire here too, not just on the mixin path.

        Both transports deliver a payload to exactly one receiver, so a split
        topology has to fail loudly at selection rather than hang mid-run.
        """
        for mode in ("nccl", "ucxx"):
            for kw in ({"tp": 2}, {"cp": 2}, {"pp": 2}):
                with self.subTest(mode=mode, **kw):
                    with self.assertRaises(ValueError):
                        make_transport_strategy(self._config(mode, **kw))

    def test_a_factory_built_strategy_drives_the_packer(self):
        """End-to-end: config -> strategy -> attached -> intercepts."""
        packer = _Packer()
        strategy = make_transport_strategy(self._config("nccl"))
        packer.set_transport_strategy(strategy)
        # NCCL's wire format, recognised without any subclassing.
        self.assertTrue(packer._should_intercept({"_nccl": True}))
        self.assertFalse(packer._should_intercept("plain-text"))


class TestComposedPackerAttaches(unittest.TestCase):
    """A packer with NO transport ancestry must still get wired by attach.

    This is the payoff of the whole refactor: a consumer keeps one packer class
    and lets config pick the transport.  It only works if attach_data_packer
    recognises a packer that merely schedules -- otherwise the strategy is
    never set up (device and the redis client exist only at attach time) and
    every payload silently takes the fallback path.
    """

    def test_ucxx_attach_wires_a_plain_scheduling_packer(self):
        from cosmos_rl.utils.payload_transport.ucxx import transport as ucxx_transport

        packer = _Packer()  # PrefetchDataPackerMixin + a plain base, no UCXX ancestry
        self.assertFalse(hasattr(packer, "_setup_ucxx_data_packer"))

        captured = {}

        def _fake_compose(pk, **kw):
            captured["packer"] = pk
            captured["kwargs"] = kw

        with unittest.mock.patch(
            "cosmos_rl.utils.payload_transport.ucxx.strategy.compose_ucxx_transport",
            _fake_compose,
        ):
            ucxx_transport.UCXXPayloadTransport().attach_data_packer(
                packer, config=_cfg(), device="cpu"
            )

        self.assertIs(captured.get("packer"), packer)
        # Tunables still come from the shared resolution, not a second copy.
        self.assertIn("prefetch_timeout", captured.get("kwargs", {}))
        self.assertIn("read_timeout", captured.get("kwargs", {}))

    def test_a_transport_mixin_packer_still_takes_the_mixin_path(self):
        """Back-compat: subclassing must keep winning over composition."""
        from cosmos_rl.utils.payload_transport.ucxx.data_packer_mixin import (
            UCXXDataPackerMixin,
        )

        class _MixinPacker(UCXXDataPackerMixin, _BasePacker):
            pass

        packer = _MixinPacker()
        self.assertTrue(callable(packer._setup_ucxx_data_packer))


def _cfg():
    class _Config:
        custom = {"ucxx_prefetch_timeout": 12.0, "ucxx_read_timeout": 3.0}

    return _Config()


class TestStrategyTelemetryCountsTheBatch(unittest.TestCase):
    """The real strategies must accumulate the BATCH's figures.

    `on_prefetch_complete` receives `n_results`, which the base passes as
    `len(self._prefetch_cache)` -- the whole double-buffered cache, not this
    batch. Accumulating that over-counts every step, and reading a byte total
    that was never set silently reports 0.0 MB forever. Both are invisible
    unless something asserts on the counters, so this does.
    """

    def _nccl(self):
        from cosmos_rl.utils.payload_transport.nccl.strategy import (
            NCCLTransportStrategy,
        )

        return NCCLTransportStrategy()

    def _ucxx(self):
        from cosmos_rl.utils.payload_transport.ucxx.strategy import (
            UCXXTransportStrategy,
        )

        return UCXXTransportStrategy()

    def test_nccl_totals_track_the_batch_not_the_cache(self):
        s = self._nccl()
        s._last_count, s._last_bytes = 3, 4096
        # n_results is deliberately larger: it is the cache size, not the batch.
        s.on_prefetch_complete(batch_id=1, n_results=50, fetch_ms=12.0, step=1)
        self.assertEqual(s._total_nccl, 3)
        self.assertEqual(s._total_bytes, 4096)

        s._last_count, s._last_bytes = 2, 1024
        s.on_prefetch_complete(batch_id=2, n_results=50, fetch_ms=8.0, step=2)
        self.assertEqual(s._total_nccl, 5)
        self.assertEqual(s._total_bytes, 5120)

    def test_ucxx_totals_track_the_batch_not_the_cache(self):
        s = self._ucxx()
        s._last_count, s._last_bytes = 3, 4096
        s.on_prefetch_complete(batch_id=1, n_results=50, fetch_ms=12.0, step=1)
        self.assertEqual(s._total_ucxx, 3)
        self.assertEqual(s._total_bytes, 4096)

    def test_bytes_are_not_silently_zero(self):
        """The failure mode was a byte total that never left 0."""
        for s in (self._nccl(), self._ucxx()):
            s._last_count, s._last_bytes = 1, 8192
            s.on_prefetch_complete(batch_id=1, n_results=1, fetch_ms=1.0, step=1)
            self.assertGreater(
                s._total_bytes, 0, f"{type(s).__name__} reported 0 bytes"
            )


if __name__ == "__main__":
    unittest.main()
