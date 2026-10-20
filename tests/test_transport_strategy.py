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
        with self.assertRaises(RuntimeError):
            self.packer.set_transport_strategy(None)
        self.packer.close_transport()
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
    """Count actual fetches, not later cache promotions or preparation work."""

    def _nccl(self):
        from cosmos_rl.utils.payload_transport.nccl.strategy import (
            NCCLTransportStrategy,
        )

        return NCCLTransportStrategy()

    def _ucxx(self):
        from cosmos_rl.utils.payload_transport.ucxx.strategy import (
            UCXXTransportStrategy,
        )

        strategy = UCXXTransportStrategy()
        self.addCleanup(strategy.shutdown)
        return strategy

    def _batch(self, strategy, sizes):
        """Real reference validation/mapping, with only native I/O mocked."""
        import torch
        from cosmos_rl.utils.payload_transport.layout import WIRE_VERSION

        data = {
            index: {"tensor": torch.zeros(size, dtype=torch.uint8)}
            for index, size in enumerate(sizes)
        }
        rollouts = []
        for index in range(len(sizes)):
            reference = {
                "_wire_version": WIRE_VERSION,
                "_layout_id": "a" * 32,
                "_sender_incarnation": "test-incarnation",
            }
            if type(strategy).__name__ == "NCCLTransportStrategy":
                reference.update(
                    _nccl=True,
                    _transfer_id=f"0:telemetry-{index}",
                    _rollout_idx=0,
                    _sender_rank=0,
                    _sender_replica="producer",
                )
            else:
                reference.update(
                    _ucxx=True,
                    _worker_ip="localhost",
                    _ucxx_port=1234,
                    _slot=index,
                    _slot_generation=1,
                )
            rollouts.append(reference)
        if type(strategy).__name__ == "NCCLTransportStrategy":
            strategy._resolve_layout = unittest.mock.Mock(return_value=object())
            strategy._fetch_all = unittest.mock.Mock(
                return_value=(data, sum(sizes), 5.0)
            )
        else:
            strategy._fetch_all = unittest.mock.AsyncMock(return_value=(data, 5.0, 2.0))
        return rollouts

    def _fetch(self, strategy, sizes, duration=12.0):
        rollouts = self._batch(strategy, sizes)
        with unittest.mock.patch(
            type(strategy).__module__ + ".get_trace_time",
            side_effect=[100.0, 100.0 + duration],
        ):
            strategy.fetch_batch(strategy.filter_prefetch_tasks(rollouts))

    def _totals(self, strategy):
        count = getattr(strategy, "_total_nccl", getattr(strategy, "_total_ucxx", 0))
        return count, strategy._total_bytes, strategy._total_latency_ms, strategy._steps

    def test_nccl_totals_track_the_batch_not_the_cache(self):
        s = self._nccl()
        self._fetch(s, [2048, 1024, 1024])
        # n_results is deliberately larger: it is the cache size, not the batch.
        s.on_prefetch_complete(batch_id=1, n_results=50, fetch_ms=12.0, step=1)
        self.assertEqual(self._totals(s), (3, 4096, 12.0, 1))

        self._fetch(s, [512, 512], duration=8.0)
        s.on_prefetch_complete(batch_id=2, n_results=50, fetch_ms=8.0, step=2)
        self.assertEqual(self._totals(s), (5, 5120, 20.0, 2))

    def test_ucxx_totals_track_the_batch_not_the_cache(self):
        s = self._ucxx()
        self._fetch(s, [2048, 1024, 1024])
        s.on_prefetch_complete(batch_id=1, n_results=50, fetch_ms=12.0, step=1)
        self.assertEqual(self._totals(s), (3, 4096, 12.0, 1))

    def test_bytes_are_not_silently_zero(self):
        """The failure mode was a byte total that never left 0."""
        for s in (self._nccl(), self._ucxx()):
            self._fetch(s, [8192])
            s.on_prefetch_complete(batch_id=1, n_results=1, fetch_ms=1.0, step=1)
            self.assertGreater(
                s._total_bytes, 0, f"{type(s).__name__} reported 0 bytes"
            )

    def test_delayed_or_repeated_legacy_hooks_cannot_count_another_fetch(self):
        for s in (self._nccl(), self._ucxx()):
            self._fetch(s, [1024], duration=3.0)
            self._fetch(s, [2048, 2048], duration=7.0)
            # The latest diagnostics are not the earlier batch's statistics.
            s._last_count, s._last_bytes = 999, 999999
            for _ in range(2):
                s.on_prefetch_complete(1, 50, 500.0, 1)
            self.assertEqual(self._totals(s), (3, 5120, 10.0, 2))

    def test_failed_fetch_is_not_a_completed_batch(self):
        for s in (self._nccl(), self._ucxx()):
            rollouts = self._batch(s, [1024])
            s._fetch_all.side_effect = RuntimeError("receive failed")
            with self.assertRaisesRegex(RuntimeError, "receive failed"):
                s.fetch_batch(s.filter_prefetch_tasks(rollouts))
            self.assertEqual(self._totals(s), (0, 0, 0.0, 0))

    def test_empty_result_does_not_invent_native_payloads(self):
        for s in (self._nccl(), self._ucxx()):
            self._fetch(s, [], duration=4.0)
            s.on_prefetch_complete(1, 50, 100.0, 1)
            self.assertEqual(self._totals(s), (0, 0, 4.0, 1))

    def test_received_count_is_not_reduced_by_cache_key_coalescing(self):
        from cosmos_rl.utils.payload_transport.receive_memory import ReceivedBatch

        for s in (self._nccl(), self._ucxx()):
            rollouts = self._batch(s, [1024, 1024])
            rollouts[1] = dict(rollouts[0])
            values, *timing = s._fetch_all.return_value
            received = ReceivedBatch(values)
            self.addCleanup(received.release)
            s._fetch_all.return_value = (received, *timing)
            result = s.fetch_batch(s.filter_prefetch_tasks(rollouts))
            self.assertEqual(len(result), 1)
            self.assertEqual(self._totals(s)[:2], (2, 2048))

    def test_both_scheduler_paths_record_once(self):
        for factory in (self._nccl, self._ucxx):
            for managed in (False, True):
                with self.subTest(backend=factory.__name__, managed=managed):
                    s = factory()
                    rollouts = self._batch(s, [1024, 2048])
                    s.on_prefetch_complete = unittest.mock.Mock(
                        wraps=s.on_prefetch_complete
                    )
                    packer = _Packer()
                    packer.set_transport_strategy(s)
                    packer._setup_prefetch(prefetch_timeout=10.0)
                    self.addCleanup(packer.shutdown_prefetch, join_timeout=5.0)
                    if managed:
                        future = packer.prefetch_payload_batch(rollouts)
                        future.result(timeout=5.0)
                        packer.consume_payload_batch(future)
                        packer.finish_payload_batch()
                        s.on_prefetch_complete.assert_not_called()
                    else:
                        packer.start_prefetch(rollouts)
                        packer.wait_prefetch()
                        s.on_prefetch_complete.assert_called_once()
                    self.assertEqual(self._totals(s)[:2], (2, 3072))
                    self.assertEqual(s._steps, 1)
                    # Stop this worker before the next subtest's loop/resources.
                    packer.shutdown_prefetch(join_timeout=5.0)

    def test_managed_fetch_has_a_shutdown_summary_without_legacy_hook(self):
        for s in (self._nccl(), self._ucxx()):
            self._fetch(s, [8192])
            with unittest.mock.patch(type(s).__module__ + ".logger.info") as info:
                s.shutdown()
            summaries = [
                call for call in info.call_args_list if "Final:" in call.args[0]
            ]
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0].args[1:4], (1, 1, 0))


if __name__ == "__main__":
    unittest.main()
