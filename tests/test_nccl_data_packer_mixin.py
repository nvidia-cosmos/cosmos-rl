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

"""Tests for :class:`NCCLDataPackerMixin` (CPU; no NCCL/Redis traffic).

Mirrors ``test_ucxx_data_packer_mixin.py``: MRO, the inherited double-
buffer state machine, ``get_policy_input`` dispatch (plain vs NCCL ref,
string- and dict-form, cache hit/miss), the cache-key helper, and the
transport-driven ``_setup_nccl_data_packer`` invocation.
"""

import unittest
from types import SimpleNamespace
from typing import Any, List
from unittest import mock

from cosmos_rl.utils.payload_transport.nccl.data_packer_mixin import (
    NCCLDataPackerMixin,
)


class _StubDataPacker:
    def __init__(self):
        self.calls: List[dict] = []

    def get_policy_input(
        self,
        sample: Any = None,
        rollout_output: Any = None,
        n_ignore_prefix_tokens: int = 0,
        **kwargs,
    ) -> Any:
        self.calls.append({"rollout_output": rollout_output})
        return rollout_output


class _Packer(NCCLDataPackerMixin, _StubDataPacker):
    """MRO: NCCLDataPackerMixin first, then _StubDataPacker."""


class TestMro(unittest.TestCase):
    def test_mixin_intercepts_first(self):
        mro = [c.__name__ for c in _Packer.__mro__]
        self.assertLess(mro.index("NCCLDataPackerMixin"), mro.index("_StubDataPacker"))


class TestPrefetchStateMachine(unittest.TestCase):
    def setUp(self):
        self.p = _Packer()

    def test_initial_cold_start(self):
        self.assertTrue(self.p.is_cold_start)
        self.assertIsNone(self.p.prefetch_buffer)

    def test_defer_seeds_buffer_on_cold_start(self):
        self.p.defer_prefetch(["r0", "r1"])
        self.assertFalse(self.p.is_cold_start)
        self.assertEqual(self.p.prefetch_buffer, ["r0", "r1"])
        self.assertFalse(self.p._prefetch_pending)

    def test_defer_after_seed_marks_pending(self):
        self.p.defer_prefetch(["r0"])
        self.p.defer_prefetch(["r1"])
        self.assertTrue(self.p._prefetch_pending)

    def test_collect_returns_buffer_after_seed(self):
        self.p.defer_prefetch(["r0", "r1"])
        self.assertEqual(self.p.collect_prefetch(), ["r0", "r1"])

    def test_start_prefetch_noop_when_disabled(self):
        before = self.p.prefetch_buffer
        self.p.start_prefetch(["r0"])
        self.assertEqual(self.p.prefetch_buffer, before)


class TestShouldIntercept(unittest.TestCase):
    def setUp(self):
        self.p = _Packer()

    def test_nccl_string(self):
        self.assertTrue(self.p._should_intercept("nccl:0:abc"))

    def test_nccl_dict(self):
        self.assertTrue(
            self.p._should_intercept({"_nccl": True, "_transfer_id": "0:a"})
        )

    def test_plain_string_and_dict(self):
        self.assertFalse(self.p._should_intercept("plain completion"))
        self.assertFalse(self.p._should_intercept({"observations": [1, 2]}))
        self.assertFalse(self.p._should_intercept(None))


class TestGetPolicyInputDispatch(unittest.TestCase):
    def setUp(self):
        self.p = _Packer()

    def test_plain_dict_delegates_to_super(self):
        traj = {"observations": [1, 2, 3]}
        out = self.p.get_policy_input(rollout_output=traj)
        self.assertIs(out, traj)
        self.assertEqual(len(self.p.calls), 1)

    def test_plain_string_delegates_to_super(self):
        out = self.p.get_policy_input(rollout_output="hello")
        self.assertEqual(out, "hello")
        self.assertEqual(len(self.p.calls), 1)

    def test_nccl_string_cache_miss_skips_episode(self):
        # No setup -> _fetch_all returns nothing -> sync fetch None -> skip.
        out = self.p.get_policy_input(rollout_output="nccl:0:abc")
        self.assertIsNone(out)
        self.assertEqual(len(self.p.calls), 0)

    def test_nccl_dict_cache_miss_skips_episode(self):
        ref = {"_nccl": True, "_transfer_id": "0:abc", "_sender_rank": 0}
        out = self.p.get_policy_input(rollout_output=ref)
        self.assertIsNone(out)
        self.assertEqual(len(self.p.calls), 0)

    def test_nccl_string_cache_hit_delegates_to_super(self):
        resolved = {"observations": [1, 2, 3]}
        key = NCCLDataPackerMixin._nccl_dp_cache_key("nccl:0:abc")
        self.p._nccl_dp_prefetch_cache = {key: resolved}
        out = self.p.get_policy_input(rollout_output="nccl:0:abc")
        self.assertIs(out, resolved)
        self.assertEqual(len(self.p.calls), 1)
        self.assertIs(self.p.calls[0]["rollout_output"], resolved)

    def test_nccl_dict_cache_hit_delegates_to_super(self):
        resolved = {"observations": [9]}
        ref = {"_nccl": True, "_transfer_id": "1:xyz"}
        key = NCCLDataPackerMixin._nccl_dp_cache_key(ref)
        self.p._nccl_dp_prefetch_cache = {key: resolved}
        out = self.p.get_policy_input(rollout_output=ref)
        self.assertIs(out, resolved)


class TestCacheKey(unittest.TestCase):
    def test_string_form(self):
        self.assertEqual(
            NCCLDataPackerMixin._nccl_dp_cache_key("nccl:0:abcdef"), "0:abcdef"
        )

    def test_dict_form(self):
        self.assertEqual(
            NCCLDataPackerMixin._nccl_dp_cache_key(
                {"_nccl": True, "_transfer_id": "3:deadbeef"}
            ),
            "3:deadbeef",
        )

    def test_non_ref_falls_back_to_str(self):
        self.assertEqual(NCCLDataPackerMixin._nccl_dp_cache_key(42), "42")


class _FakeRv:
    """Records the timeout each ``initiate`` is given; returns a fixed status."""

    def __init__(self, status):
        self._status = status
        self.timeouts: List[float] = []

    def initiate(self, *, need_uid, timeout, **kwargs):
        from cosmos_rl.utils.payload_transport.nccl.rendezvous import (
            RendezvousResult,
        )

        self.timeouts.append(timeout)
        return RendezvousResult(self._status, [1, 2, 3] if need_uid else None)


def _consumer(rv, cache):
    from cosmos_rl.utils.trajectory import (
        build_trajectory_schema,
    )

    p = _Packer()
    p._nccl_dp_rendezvous = rv
    p._nccl_dp_comm_cache = cache
    p._nccl_dp_receiver_rank = 0
    p._nccl_dp_receiver_replica = "pol-A"
    p._nccl_dp_prefix = "pfx"
    p._nccl_dp_max_attempts = 2
    p._nccl_dp_recv_timeout = 5.0
    p._nccl_dp_first_transfer_timeout = 30.0
    p._nccl_dp_warm_pairs = set()
    p._nccl_dp_device = None
    p._nccl_dp_schema = build_trajectory_schema(
        {"max_steps": 4, "obs_dim": 2, "action_dim": 1}
    )
    return p


def _ref():
    return {"sender_replica": "rA", "sender_rank": 0, "transfer_id": "0:x"}


class TestQuarantineClearsWarmMarker(unittest.TestCase):
    """Comm-generation recovery (receiver side): quarantining a warm endpoint
    must also demote its pair back to 'warming', so the post-cooldown rebuild
    gets the generous first_transfer_timeout budget instead of the tight
    recv_timeout that caused the original failure."""

    def test_quarantine_endpoint_demotes_pair_to_warming(self):
        quarantined = []

        class _RecordingCache:
            def quarantine(self, health_key):
                quarantined.append(health_key)

        p = _consumer(_FakeRv(None), _RecordingCache())
        pair = ("rA", 0, 0)
        p._nccl_dp_warm_pairs.add(pair)
        p._quarantine_endpoint(p._nccl_dp_comm_cache, ("rA", 0), pair)
        self.assertEqual(quarantined, [("rA", 0)])  # endpoint quarantined
        self.assertNotIn(pair, p._nccl_dp_warm_pairs)  # demoted to warming

    def test_quarantine_survives_cache_raising(self):
        class _BoomCache:
            def quarantine(self, health_key):
                raise RuntimeError("abort failed")

        p = _consumer(_FakeRv(None), _BoomCache())
        pair = ("rA", 0, 0)
        p._nccl_dp_warm_pairs.add(pair)
        p._quarantine_endpoint(p._nccl_dp_comm_cache, ("rA", 0), pair)  # no raise
        self.assertNotIn(pair, p._nccl_dp_warm_pairs)  # still demoted


class TestSyncFetchContainment(unittest.TestCase):
    """Gap 3: the cache-miss sync fallback must convert a fetch error to None
    (clean degrade to the packer's own path) rather than propagating -- the
    base mixin calls _sync_fetch inside get_policy_input without its own
    try/except, so a raised rendezvous/recv error would crash the train step."""

    def _packer(self):
        from cosmos_rl.utils.trajectory import (
            build_trajectory_schema,
        )

        p = _Packer()
        p._nccl_dp_schema = build_trajectory_schema(
            {"max_steps": 4, "obs_dim": 2, "action_dim": 1}
        )
        return p

    def test_returns_none_on_fetch_error(self):
        p = self._packer()

        def boom(refs):
            raise RuntimeError("rendezvous exploded")

        p._fetch_all = boom
        # "nccl:0:x" parses to a real ref, so _fetch_all IS reached and raises.
        self.assertIsNone(p._sync_fetch("nccl:0:x"))

    def test_returns_none_on_unparseable_ref(self):
        p = self._packer()
        # Not an NCCL reference -> None without touching _fetch_all.
        self.assertIsNone(p._sync_fetch(12345))

    def test_returns_none_on_malformed_schema(self):
        # A corrupt dict ref raises from _parse_ref/deserialize_schema, which is
        # now INSIDE the containment (base get_policy_input has none above it).
        p = self._packer()
        bad = {"_nccl": True, "_transfer_id": "0:x", "_schema": "not-a-schema"}
        self.assertIsNone(p._sync_fetch(bad))

    def test_returns_result_on_success(self):
        p = self._packer()
        p._fetch_all = lambda refs: ({0: {"ok": 1}}, 4, 1.0)
        self.assertEqual(p._sync_fetch("nccl:0:x"), {"ok": 1})


class TestColdStartTolerance(unittest.TestCase):
    """A slow cold-start pair must get the long budget and NOT be quarantined;
    an established comm that times out MUST be quarantined."""

    def _cache(self, aborted):
        from cosmos_rl.utils.payload_transport.nccl.comm_cache import CommCache

        idx = {"n": 40}

        def build(u, r):
            idx["n"] += 1
            return idx["n"]

        return CommCache(build_fn=build, abort_fn=lambda i: aborted.append(i))

    def test_cold_start_uses_first_transfer_budget_no_quarantine(self):
        from cosmos_rl.utils.payload_transport.nccl.rendezvous import (
            TransferStatus,
        )

        aborted = []
        cache = self._cache(aborted)
        rv = _FakeRv(TransferStatus.CANCELLED)
        p = _consumer(rv, cache)

        out = p._rendezvous_one(_ref(), None)
        self.assertIsNone(out)
        # Both attempts used the long first-transfer budget (pair never cached).
        self.assertEqual(rv.timeouts, [30.0, 30.0])
        # Cold-start timeout must NOT quarantine a healthy-but-warming sender.
        self.assertFalse(cache.is_quarantined(("rA", 0)))

    def test_warm_comm_timeout_quarantines(self):
        from cosmos_rl.utils.payload_transport.nccl.rendezvous import (
            TransferStatus,
        )

        aborted = []
        cache = self._cache(aborted)
        built = cache.get_or_create(("rA", 0, 0), uid_chars=[1], local_rank=1)
        rv = _FakeRv(TransferStatus.CANCELLED)
        p = _consumer(rv, cache)
        # Mark the pair WARM (has transferred before) -> tight timeout + a
        # timeout now is a genuine failure.
        p._nccl_dp_warm_pairs.add(("rA", 0, 0))

        out = p._rendezvous_one(_ref(), None)
        self.assertIsNone(out)
        self.assertEqual(rv.timeouts, [5.0, 5.0])  # warm -> tight budget
        self.assertTrue(cache.is_quarantined(("rA", 0)))  # quarantine + abort
        self.assertEqual(aborted, [built])

    def test_warming_built_comm_not_quarantined_or_aborted(self):
        # The regression case: a comm was built but the pair has NOT transferred
        # yet (warming).  A timeout must NOT quarantine/abort it -- keep it and
        # retry, else the cold-start storm churns it to 0 MB.
        from cosmos_rl.utils.payload_transport.nccl.rendezvous import (
            TransferStatus,
        )

        aborted = []
        cache = self._cache(aborted)
        cache.get_or_create(("rA", 0, 0), uid_chars=[1], local_rank=1)  # built
        rv = _FakeRv(TransferStatus.CANCELLED)
        p = _consumer(rv, cache)  # warm_pairs empty -> warming

        out = p._rendezvous_one(_ref(), None)
        self.assertIsNone(out)
        self.assertEqual(rv.timeouts, [30.0, 30.0])  # warming -> long budget
        self.assertFalse(cache.is_quarantined(("rA", 0)))  # NOT quarantined
        self.assertIn(("rA", 0, 0), cache)  # comm KEPT
        self.assertEqual(aborted, [])  # NOT aborted


class TestRecvFailureIsolation(unittest.TestCase):
    """Recvs are issued standalone (no cross-comm group).  A failed recv is
    isolated to its pair: a WARM pair is quarantined+aborted (genuine); a
    WARMING pair is kept (retry)."""

    def _run(self, *, warm):
        import torch

        import cosmos_rl.utils.pynccl as pynccl_mod
        from cosmos_rl.utils.payload_transport.nccl.comm_cache import CommCache

        aborted = []
        cache = CommCache(
            build_fn=lambda u, r: 55, abort_fn=lambda i: aborted.append(i)
        )
        cache.get_or_create(("rA", 0, 0), uid_chars=[1], local_rank=1)

        p = _Packer()
        p._nccl_dp_rendezvous = object()
        p._nccl_dp_comm_cache = cache
        p._nccl_dp_device = None
        p._nccl_dp_streams = None
        p._nccl_dp_receiver_rank = 0
        p._nccl_dp_recv_timeout = 5.0
        p._nccl_dp_first_transfer_timeout = 30.0
        p._nccl_dp_warm_pairs = {("rA", 0, 0)} if warm else set()
        # Skip the real rendezvous; hand _fetch_all one accepted recv.
        p._rendezvous_one = lambda ref, pynccl: (0, torch.zeros(4, dtype=torch.uint8))

        # Ungrouped path: recvs are issued standalone, no nccl_group_start/end.
        with mock.patch.object(
            pynccl_mod, "nccl_recv", mock.Mock(side_effect=RuntimeError("boom"))
        ):
            results, nbytes, _ = p._fetch_all([(0, _ref())])
        return results, cache, aborted

    def test_warm_pair_failure_isolated_and_quarantined(self):
        results, cache, aborted = self._run(warm=True)
        self.assertEqual(results, {})
        self.assertTrue(cache.is_quarantined(("rA", 0)))  # warm -> quarantined
        self.assertEqual(aborted, [55])  # its comm aborted

    def test_warming_pair_failure_keeps_comm(self):
        results, cache, aborted = self._run(warm=False)
        self.assertEqual(results, {})
        self.assertFalse(cache.is_quarantined(("rA", 0)))  # warming -> kept
        self.assertIn(("rA", 0, 0), cache)  # comm NOT aborted
        self.assertEqual(aborted, [])


class TestRecvLaunchSerialized(unittest.TestCase):
    """Fix C: the recv LAUNCH must run under ``_nccl_dp_recv_lock`` (mirroring the
    producer's ``_nccl_send_lock``) so a trainer-thread ``_sync_fetch`` and the
    prefetch worker can't fire concurrent multi-comm recv launches on one device
    -> native launch deadlock the recv timeout can't rescue.  The lock must be
    released before the blocking synchronize() so real transfers still overlap."""

    def test_recv_launch_holds_lock_then_releases(self):
        import threading

        import torch

        import cosmos_rl.utils.pynccl as pynccl_mod
        from cosmos_rl.utils.payload_transport.nccl.comm_cache import CommCache

        cache = CommCache(build_fn=lambda u, r: 7, abort_fn=lambda i: None)
        cache.get_or_create(("rA", 0, 0), uid_chars=[1], local_rank=1)

        p = _Packer()
        p._nccl_dp_rendezvous = object()
        p._nccl_dp_comm_cache = cache
        p._nccl_dp_device = None
        p._nccl_dp_streams = None
        p._nccl_dp_receiver_rank = 0
        p._nccl_dp_recv_timeout = 5.0
        p._nccl_dp_first_transfer_timeout = 30.0
        p._nccl_dp_warm_pairs = set()  # warming -> failing recv just retries
        p._nccl_dp_recv_lock = threading.Lock()
        p._rendezvous_one = lambda ref, pynccl: (0, torch.zeros(4, dtype=torch.uint8))

        held = []

        def _recv(*a, **k):
            held.append(p._nccl_dp_recv_lock.locked())
            # Raise so posted stays empty -> return before the unpack loop; the
            # lock must already be held at the point of the NCCL launch.
            raise RuntimeError("stop after lock check")

        with mock.patch.object(pynccl_mod, "nccl_recv", mock.Mock(side_effect=_recv)):
            p._fetch_all([(0, _ref())])

        self.assertEqual(held, [True])  # launch happened WITH the lock held
        self.assertFalse(p._nccl_dp_recv_lock.locked())  # released afterwards

    def test_fetch_all_lazily_creates_lock_without_setup(self):
        # A bare harness that skips _setup_nccl_data_packer leaves the lock None;
        # _fetch_all must lazily create it rather than `with None:` -> TypeError.
        import torch

        import cosmos_rl.utils.pynccl as pynccl_mod
        from cosmos_rl.utils.payload_transport.nccl.comm_cache import CommCache

        cache = CommCache(build_fn=lambda u, r: 1, abort_fn=lambda i: None)
        cache.get_or_create(("rA", 0, 0), uid_chars=[1], local_rank=1)

        p = _Packer()
        p._nccl_dp_rendezvous = object()
        p._nccl_dp_comm_cache = cache
        p._nccl_dp_device = None
        p._nccl_dp_streams = None
        p._nccl_dp_receiver_rank = 0
        p._nccl_dp_recv_timeout = 5.0
        p._nccl_dp_first_transfer_timeout = 30.0
        p._nccl_dp_warm_pairs = set()
        self.assertIsNone(p._nccl_dp_recv_lock)  # setup skipped
        p._rendezvous_one = lambda ref, pynccl: (0, torch.zeros(4, dtype=torch.uint8))

        with mock.patch.object(
            pynccl_mod, "nccl_recv", mock.Mock(side_effect=RuntimeError("x"))
        ):
            p._fetch_all([(0, _ref())])  # must not raise TypeError
        self.assertIsNotNone(p._nccl_dp_recv_lock)  # lazily created


class TestRecvSyncFailureContained(unittest.TestCase):
    """Fix D: a recv that ENQUEUES cleanly but whose peer never sends surfaces at
    the completion synchronize(), not at the enqueue try/except.  An uncaught
    raise there unwinds _fetch_all -> the prefetch worker marks the WHOLE batch
    failed -> wait_prefetch wipes the cache -> every episode drops to fallback AND
    the offending pair is never quarantined.  It must instead quarantine the
    posted warm pair(s) and drop only this batch."""

    def test_sync_failure_quarantines_posted_and_returns_empty(self):
        import threading

        import torch

        import cosmos_rl.utils.pynccl as pynccl_mod
        from cosmos_rl.utils.payload_transport.nccl import data_packer_mixin as dpm
        from cosmos_rl.utils.payload_transport.nccl.comm_cache import CommCache

        aborted = []
        cache = CommCache(build_fn=lambda u, r: 9, abort_fn=lambda i: aborted.append(i))
        cache.get_or_create(("rA", 0, 0), uid_chars=[1], local_rank=1)

        p = _Packer()
        p._nccl_dp_rendezvous = object()
        p._nccl_dp_comm_cache = cache
        p._nccl_dp_device = None
        p._nccl_dp_streams = None
        p._nccl_dp_receiver_rank = 0
        p._nccl_dp_recv_timeout = 5.0
        p._nccl_dp_first_transfer_timeout = 30.0
        p._nccl_dp_warm_pairs = {("rA", 0, 0)}  # warm -> eligible for quarantine
        p._nccl_dp_recv_lock = threading.Lock()
        p._rendezvous_one = lambda ref, pynccl: (0, torch.zeros(4, dtype=torch.uint8))

        bad_stream = mock.Mock()
        bad_stream.synchronize.side_effect = RuntimeError("peer never sent")

        with (
            mock.patch.object(pynccl_mod, "nccl_recv", mock.Mock()),
            mock.patch.object(dpm, "record_event", lambda stream=None: object()),
            mock.patch.object(dpm, "wait_event", lambda s, e: None),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.current_stream", return_value=bad_stream),
        ):
            results, nbytes, _ = p._fetch_all([(0, _ref())])

        self.assertEqual(results, {})  # batch dropped to fallback, NO raise
        self.assertEqual(nbytes, 0)
        self.assertTrue(cache.is_quarantined(("rA", 0)))  # dead pair quarantined
        self.assertEqual(aborted, [9])  # its comm aborted
        self.assertNotIn(("rA", 0, 0), cache)  # comm removed


class TestReceiverRenegotiation(unittest.TestCase):
    """On NEED_UID the receiver must drop its stale comm and retry WITH a uid
    (else it waits forever on a comm the sender never rejoins)."""

    def test_need_uid_aborts_stale_comm_and_retries_with_uid(self):
        from cosmos_rl.utils.payload_transport.nccl.comm_cache import CommCache
        from cosmos_rl.utils.payload_transport.nccl.rendezvous import (
            RendezvousResult,
            TransferStatus,
        )

        class _SeqRv:
            def __init__(self, statuses):
                self._statuses = list(statuses)
                self.need_uids = []

            def initiate(self, *, need_uid, timeout, **kwargs):
                idx = min(len(self.need_uids), len(self._statuses) - 1)
                self.need_uids.append(need_uid)
                st = self._statuses[idx]
                return RendezvousResult(st, [1, 2, 3] if need_uid else None)

        cache = CommCache(build_fn=lambda u, r: 9, abort_fn=lambda i: None)
        # Pre-cache the receiver pair (sender_replica, sender_rank, recv_rank)
        # so attempt 1 sends need_uid=False.
        cache.get_or_create(("rA", 0, 0), uid_chars=[1], local_rank=1)
        rv = _SeqRv([TransferStatus.NEED_UID, TransferStatus.ACCEPTED])
        p = _consumer(rv, cache)
        ref = {
            "sender_replica": "rA",
            "sender_rank": 0,
            "transfer_id": "0:x",
            "schema": p._nccl_dp_schema,
        }

        out = p._rendezvous_one(ref, None)
        # attempt 1: cached -> need_uid=False -> NEED_UID -> abort + retry;
        # attempt 2: aborted -> need_uid=True -> ACCEPTED -> comm rebuilt.
        self.assertEqual(rv.need_uids, [False, True])
        self.assertIsNotNone(out)


class TestSetupViaTransportAttach(unittest.TestCase):
    def test_attach_via_transport_invokes_setup(self):
        from cosmos_rl.utils.payload_transport.nccl.transport import (
            NcclPayloadTransport,
        )
        from cosmos_rl.utils.payload_transport.registry import RedisEndpoint

        captured = {}

        class _AttachPacker(NCCLDataPackerMixin):
            pass

        config = SimpleNamespace(
            custom={
                "nccl_prefetch_timeout": 12.0,
                "nccl_read_max_attempts": 5,
                "nccl_recv_timeout": 3.0,
                "nccl_first_transfer_timeout": 45.0,
            }
        )
        fake_client = object()

        def _fake_setup(self, **kwargs):
            captured.update(kwargs)

        with (
            mock.patch.object(
                NCCLDataPackerMixin, "_setup_nccl_data_packer", _fake_setup
            ),
            mock.patch(
                "cosmos_rl.utils.payload_transport.nccl.transport._build_redis_client",
                return_value=fake_client,
            ),
        ):
            NcclPayloadTransport().attach_data_packer(
                _AttachPacker(),
                config=config,
                device="cuda:1",
                redis_endpoint=RedisEndpoint("h", 6379),
            )

        self.assertEqual(captured["device"], "cuda:1")
        self.assertIs(captured["redis_client"], fake_client)
        # prefetch_timeout is floored to cover batch_hint x max_attempts x
        # first_transfer_timeout (8 x 5 x 45 = 1800 > the configured 12).
        self.assertEqual(captured["prefetch_timeout"], 1800.0)
        self.assertEqual(captured["max_attempts"], 5)
        self.assertEqual(captured["recv_timeout"], 3.0)
        self.assertEqual(captured["first_transfer_timeout"], 45.0)

    def test_attach_defaults_when_custom_missing(self):
        from cosmos_rl.utils.payload_transport.nccl.transport import (
            NcclPayloadTransport,
        )
        from cosmos_rl.utils.payload_transport.registry import RedisEndpoint

        captured = {}

        class _AttachPacker(NCCLDataPackerMixin):
            pass

        with (
            mock.patch.object(
                NCCLDataPackerMixin,
                "_setup_nccl_data_packer",
                lambda self, **kw: captured.update(kw),
            ),
            mock.patch(
                "cosmos_rl.utils.payload_transport.nccl.transport._build_redis_client",
                return_value=object(),
            ),
        ):
            NcclPayloadTransport().attach_data_packer(
                _AttachPacker(),
                config=SimpleNamespace(custom={}),
                device=None,
                redis_endpoint=RedisEndpoint("h", 6379),
            )
        # Floored to batch_hint(8, default) x max_attempts(2) x
        # first_transfer_timeout(30) = 480 > the default 30.
        self.assertEqual(captured["prefetch_timeout"], 480.0)
        self.assertEqual(captured["max_attempts"], 2)
        self.assertEqual(captured["recv_timeout"], 5.0)
        self.assertEqual(captured["first_transfer_timeout"], 30.0)  # default

    def test_attach_floors_prefetch_timeout_to_cover_retry_budget(self):
        # Codex: prefetch wait must cover the BATCH's sequential cold-start cost
        # (batch_hint x max_attempts x first_transfer_timeout), else it expires
        # mid cold-start and the late result is mis-consumed.  batch_hint pinned
        # to 1 here isolates the max_attempts x first_transfer_timeout factor.
        from cosmos_rl.utils.payload_transport.nccl.transport import (
            NcclPayloadTransport,
        )
        from cosmos_rl.utils.payload_transport.registry import RedisEndpoint

        captured = {}
        config = SimpleNamespace(
            custom={
                "nccl_prefetch_timeout": 10.0,  # too small
                "nccl_read_max_attempts": 3,
                "nccl_first_transfer_timeout": 20.0,  # 1 x 3 x 20 = 60
                "nccl_prefetch_batch_hint": 1,
            }
        )
        with (
            mock.patch.object(
                NCCLDataPackerMixin,
                "_setup_nccl_data_packer",
                lambda self, **kw: captured.update(kw),
            ),
            mock.patch(
                "cosmos_rl.utils.payload_transport.nccl.transport._build_redis_client",
                return_value=object(),
            ),
        ):
            NcclPayloadTransport().attach_data_packer(
                NCCLDataPackerMixin(),
                config=config,
                device=None,
                redis_endpoint=RedisEndpoint("h", 6379),
            )
        self.assertEqual(captured["prefetch_timeout"], 60.0)  # 1 x 3 x 20

    def test_prefetch_batch_hint_scales_floor(self):
        # The batch_hint multiplies the cold-start floor to cover a batch of
        # sequential first-transfers.
        from cosmos_rl.utils.payload_transport.nccl.transport import (
            NcclPayloadTransport,
        )
        from cosmos_rl.utils.payload_transport.registry import RedisEndpoint

        captured = {}
        config = SimpleNamespace(
            custom={
                "nccl_prefetch_timeout": 10.0,
                "nccl_read_max_attempts": 2,
                "nccl_first_transfer_timeout": 20.0,
                "nccl_prefetch_batch_hint": 4,  # 4 x 2 x 20 = 160
            }
        )
        with (
            mock.patch.object(
                NCCLDataPackerMixin,
                "_setup_nccl_data_packer",
                lambda self, **kw: captured.update(kw),
            ),
            mock.patch(
                "cosmos_rl.utils.payload_transport.nccl.transport._build_redis_client",
                return_value=object(),
            ),
        ):
            NcclPayloadTransport().attach_data_packer(
                NCCLDataPackerMixin(),
                config=config,
                device=None,
                redis_endpoint=RedisEndpoint("h", 6379),
            )
        self.assertEqual(captured["prefetch_timeout"], 160.0)


class TestUidTtlCoversColdStart(unittest.TestCase):
    """Codex P2: the per-pair UID TTL must outlive the cold-start budget, or a
    request queued behind the init storm reads an expired UID and the two comm
    halves never join."""

    def test_rendezvous_uid_ttl_covers_first_transfer_timeout(self):
        captured = {}

        class _RvSpy:
            def __init__(self, redis_client, prefix, *, uid_ttl_s=60, **kw):
                captured["uid_ttl_s"] = uid_ttl_s

        packer = NCCLDataPackerMixin()
        with (
            mock.patch(
                "cosmos_rl.utils.payload_transport.nccl.data_packer_mixin.NcclRendezvous",
                _RvSpy,
            ),
            mock.patch.object(packer, "_setup_prefetch", lambda **kw: None),
            mock.patch(
                "cosmos_rl.utils.payload_transport.nccl.data_packer_mixin."
                "get_transfer_stream_pool",
                lambda **kw: None,
            ),
        ):
            packer._setup_nccl_data_packer(
                device=None,
                redis_client=object(),
                config=SimpleNamespace(custom={}),
                first_transfer_timeout=90.0,
            )
        self.assertGreaterEqual(captured["uid_ttl_s"], 90.0)


if __name__ == "__main__":
    unittest.main()
