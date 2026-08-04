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

"""Tests for the producer-side :class:`NCCLRolloutMixin` (CPU-only).

The pack path and the request/cleanup handlers are exercised without any
NCCL traffic: buffers are plain CPU tensors, ``_send`` is stubbed, and the
rendezvous is a fake that records replies.
"""

import contextlib
import threading
import time
import unittest
from unittest import mock

import torch

from cosmos_rl.utils.payload_transport.nccl.buffer_registry import SendBufferRegistry
from cosmos_rl.utils.payload_transport.nccl.mixins import NCCLRolloutMixin
from cosmos_rl.utils.payload_transport.nccl.rendezvous import TransferStatus
from cosmos_rl.utils.trajectory import (
    build_trajectory_schema,
    schema_layout,
)


class _FakeRendezvous:
    def __init__(self):
        self.replies = []

    def respond(self, *, resp_key, status):
        self.replies.append((resp_key, status))

    def read_uid(self, uid_key):
        return [1, 2, 3]


def _make_producer(capacity=8):
    p = NCCLRolloutMixin()
    p._nccl_enabled = True
    p._nccl_replica_id = "rollout-test-0"
    p._nccl_rollout_idx = 0
    p._nccl_sender_rank = 0
    p._nccl_device = None  # CPU tensors
    p._nccl_schema = build_trajectory_schema(
        {"max_steps": 10, "obs_dim": 4, "action_dim": 2}
    )
    p._nccl_offsets, p._nccl_entry_size = schema_layout(p._nccl_schema)
    p._nccl_registry = SendBufferRegistry(capacity=capacity, on_free=p._on_buffer_free)
    p._nccl_rendezvous = _FakeRendezvous()
    # Real cache so _handle_request's cache.get(pair) works; tests that need a
    # cached pair or a failing build override this.
    from cosmos_rl.utils.payload_transport.nccl.comm_cache import CommCache

    p._nccl_comm_cache = CommCache(build_fn=lambda u, r: 7, abort_fn=lambda i: None)
    p._nccl_streams = None
    p._nccl_send_lock = threading.Lock()
    return p


class TestWriteToBuffer(unittest.TestCase):
    def test_metadata_shape_and_registration(self):
        p = _make_producer()
        traj = {
            "observations": torch.zeros(6, 4),
            "actions": torch.zeros(6, 2),
            "rewards": torch.ones(6),
            "episode_length": 6,
        }
        meta = p.write_to_buffer(traj)
        self.assertIsNotNone(meta)
        self.assertTrue(meta["_nccl"])
        # Globally-unique sender identity is carried for multi-replica addressing.
        self.assertEqual(meta["_sender_replica"], "rollout-test-0")
        self.assertEqual(meta["_sender_rank"], 0)
        self.assertEqual(meta["_rollout_idx"], 0)
        self.assertEqual(meta["episode_length"], 6)
        # Completion string carries the nccl: prefix + transfer id.
        self.assertTrue(meta["completion"].startswith("nccl:"))
        self.assertEqual(meta["completion"], "nccl:" + meta["_transfer_id"])
        # Schema was serialized for the consumer.
        self.assertEqual(
            [s["name"] for s in meta["_schema"]],
            [s.name for s in p._nccl_schema],
        )
        # Buffer registered under the transfer id.
        self.assertIn(meta["_transfer_id"], p._nccl_registry)

    def test_disabled_returns_none(self):
        p = _make_producer()
        p._nccl_enabled = False
        self.assertIsNone(p.write_to_buffer({"observations": torch.zeros(3, 4)}))


class TestHandleRequest(unittest.TestCase):
    def test_missing_buffer_replies_missing(self):
        p = _make_producer()
        p._handle_request(
            {
                "transfer_id": "0:absent",
                "resp_key": "rk",
                "receiver_rank": 1,
                "uid_key": "uk",
            }
        )
        self.assertEqual(p._nccl_rendezvous.replies, [("rk", TransferStatus.MISSING)])

    def test_present_buffer_acks_then_sends(self):
        p = _make_producer()
        # Register a buffer for the transfer.
        p._nccl_registry.register(
            "0:present", torch.zeros(p._nccl_entry_size, dtype=torch.uint8)
        )
        sent = []
        p._send = lambda entry, rr, uk, rrep=None: sent.append(
            (entry.transfer_id, rr, uk, rrep)
        )

        p._handle_request(
            {
                "transfer_id": "0:present",
                "resp_key": "rk",
                "receiver_rank": 1,
                "receiver_replica": "policy-A",
                "uid_key": "uk",
            }
        )
        # ACCEPTED must be sent before the (stubbed) send runs.
        self.assertEqual(p._nccl_rendezvous.replies, [("rk", TransferStatus.ACCEPTED)])
        # receiver_replica from the request is threaded through to the send.
        self.assertEqual(sent, [("0:present", 1, "uk", "policy-A")])

    def test_expired_request_is_dropped(self):
        # Bilateral cancellation: a request whose receiver deadline has passed
        # must be dropped WITHOUT replying ACCEPTED or launching a send (which
        # would be unmatched and pin the send lock -> the N_POLICY>=4 cascade).
        p = _make_producer()
        p._nccl_registry.register(
            "0:x", torch.zeros(p._nccl_entry_size, dtype=torch.uint8)
        )
        sent = []
        p._send = lambda *a, **k: sent.append(a)
        p._handle_request(
            {
                "transfer_id": "0:x",
                "resp_key": "rk",
                "receiver_rank": 1,
                "receiver_replica": "pol",
                "uid_key": "uk",
                "req_deadline": time.time() - 5.0,  # receiver already gave up
            }
        )
        self.assertEqual(p._nccl_rendezvous.replies, [])  # no late ACCEPTED
        self.assertEqual(sent, [])  # no unmatched send
        # No lease taken (drop happens before acquire).
        self.assertEqual(p._nccl_registry.get("0:x").inflight, 0)

    def test_live_request_within_deadline_is_served(self):
        p = _make_producer()
        p._nccl_registry.register(
            "0:x", torch.zeros(p._nccl_entry_size, dtype=torch.uint8)
        )
        sent = []
        p._send = lambda entry, rr, uk, rrep=None: sent.append(entry.transfer_id)
        p._handle_request(
            {
                "transfer_id": "0:x",
                "resp_key": "rk",
                "receiver_rank": 1,
                "receiver_replica": "pol",
                "uid_key": "uk",
                "req_deadline": time.time() + 30.0,  # still waiting
            }
        )
        self.assertEqual(p._nccl_rendezvous.replies, [("rk", TransferStatus.ACCEPTED)])
        self.assertEqual(sent, ["0:x"])

    def test_lease_released_on_setup_failure(self):
        # A failure BETWEEN acquire() and _send (here: an unhashable
        # receiver_replica making cache.get(pair) raise) must release the lease,
        # or the buffer would be pinned un-reapable.
        from cosmos_rl.utils.payload_transport.nccl.comm_cache import CommCache

        p = _make_producer()
        p._nccl_comm_cache = CommCache(build_fn=lambda u, r: 7, abort_fn=lambda i: None)
        p._nccl_registry.register(
            "0:x", torch.zeros(p._nccl_entry_size, dtype=torch.uint8)
        )
        p._handle_request(
            {
                "transfer_id": "0:x",
                "resp_key": "rk",
                "receiver_rank": 1,
                "receiver_replica": [],  # unhashable -> cache.get(pair) raises
                "uid_key": None,
            }
        )
        self.assertEqual(p._nccl_registry.get("0:x").inflight, 0)

    def test_lease_released_when_send_build_fails(self):
        # A real _send that raises during comm build must release the lease via
        # its own try/finally (and _handle_request must NOT double-abandon).
        from cosmos_rl.utils.payload_transport.nccl.comm_cache import CommCache

        def boom_build(u, r):
            raise RuntimeError("comm build failed")

        p = _make_producer()
        p._nccl_comm_cache = CommCache(build_fn=boom_build, abort_fn=lambda i: None)
        p._nccl_registry.register(
            "0:x", torch.zeros(p._nccl_entry_size, dtype=torch.uint8)
        )
        p._handle_request(
            {
                "transfer_id": "0:x",
                "resp_key": "rk",
                "receiver_rank": 1,
                "receiver_replica": "pol",
                "uid_key": "uk",  # truthy -> skips renegotiation, reaches _send
            }
        )
        self.assertEqual(p._nccl_registry.get("0:x").inflight, 0)


class TestHandleCleanup(unittest.TestCase):
    def test_cleanup_frees_buffer(self):
        p = _make_producer()
        p._nccl_registry.register("0:xyz", torch.zeros(4, dtype=torch.uint8))
        self.assertIn("0:xyz", p._nccl_registry)
        p._handle_cleanup('{"transfer_id": "0:xyz"}')
        self.assertNotIn("0:xyz", p._nccl_registry)

    def test_cleanup_ignores_malformed(self):
        p = _make_producer()
        p._nccl_registry.register("0:xyz", torch.zeros(4, dtype=torch.uint8))
        p._handle_cleanup("not json")
        p._handle_cleanup('{"no_id": 1}')
        # Buffer untouched.
        self.assertIn("0:xyz", p._nccl_registry)

    def test_pack_roundtrip_bytes(self):
        """Packed buffer round-trips through the schema layout on CPU."""
        p = _make_producer()
        obs = torch.arange(10 * 4, dtype=torch.float32).reshape(10, 4)
        traj = {
            "observations": obs,
            "actions": torch.zeros(10, 2),
            "rewards": torch.zeros(10),
            "episode_length": 10,
        }
        buf, _ = p._pack(traj, 10)
        self.assertEqual(buf.numel(), p._nccl_entry_size)
        # Slice observations back out and compare.
        off = p._nccl_offsets["observations"]
        nbytes = p._nccl_schema[0].nbytes
        recovered = buf[off : off + nbytes].view(torch.float32).reshape(10, 4)
        self.assertTrue(torch.equal(recovered, obs))


class TestRenegotiation(unittest.TestCase):
    """If the receiver omits a uid (expecting a cached comm) but the sender
    evicted its side, the sender must reply NEED_UID (not build a doomed comm
    the receiver never joins)."""

    def _real_cache(self, comm_idx=7):
        from cosmos_rl.utils.payload_transport.nccl.comm_cache import CommCache

        return CommCache(build_fn=lambda u, r: comm_idx, abort_fn=lambda i: None)

    def test_no_uid_and_no_cached_comm_replies_need_uid(self):
        from cosmos_rl.utils.payload_transport.nccl.rendezvous import TransferStatus

        p = _make_producer()
        p._nccl_comm_cache = self._real_cache()
        p._nccl_registry.register(
            "0:x", torch.zeros(p._nccl_entry_size, dtype=torch.uint8)
        )
        sent = []
        p._send = lambda *a, **k: sent.append(a)
        p._handle_request(
            {
                "transfer_id": "0:x",
                "resp_key": "rk",
                "receiver_rank": 1,
                "receiver_replica": "pol",
                "uid_key": None,
            }
        )
        self.assertEqual(p._nccl_rendezvous.replies, [("rk", TransferStatus.NEED_UID)])
        self.assertEqual(sent, [])  # no doomed send
        # The acquire() lease taken at the top of _handle_request must be
        # released on the NEED_UID early-return -- otherwise the buffer would
        # be pinned un-reapable forever.
        self.assertEqual(p._nccl_registry.get("0:x").inflight, 0)

    def test_no_uid_but_cached_comm_accepts(self):
        from cosmos_rl.utils.payload_transport.nccl.comm_cache import CommCache
        from cosmos_rl.utils.payload_transport.nccl.mixins import _producer_pair_key
        from cosmos_rl.utils.payload_transport.nccl.rendezvous import TransferStatus

        p = _make_producer()
        cache = CommCache(build_fn=lambda u, r: 7, abort_fn=lambda i: None)
        # Sender already has the pair cached -> can reuse it without a uid.
        cache.get_or_create(
            _producer_pair_key(p._nccl_sender_rank, "pol", 1),
            uid_chars=[1],
            local_rank=0,
        )
        p._nccl_comm_cache = cache
        p._nccl_registry.register(
            "0:x", torch.zeros(p._nccl_entry_size, dtype=torch.uint8)
        )
        sent = []
        p._send = lambda entry, rr, uk, rrep=None: sent.append((rr, rrep))
        p._handle_request(
            {
                "transfer_id": "0:x",
                "resp_key": "rk",
                "receiver_rank": 1,
                "receiver_replica": "pol",
                "uid_key": None,
            }
        )
        self.assertEqual(p._nccl_rendezvous.replies, [("rk", TransferStatus.ACCEPTED)])
        self.assertEqual(sent, [(1, "pol")])

    def test_uid_key_present_but_unreadable_replies_need_uid(self):
        # The receiver sent a uid_key but its UID publish was lost / expired
        # (read_uid -> None) and we must BUILD the comm.  Building from an empty
        # UID would desync -> ask for a fresh UID instead of a doomed transfer.
        from cosmos_rl.utils.payload_transport.nccl.comm_cache import CommCache
        from cosmos_rl.utils.payload_transport.nccl.rendezvous import TransferStatus

        p = _make_producer()
        p._nccl_comm_cache = CommCache(build_fn=lambda u, r: 7, abort_fn=lambda i: None)
        p._nccl_rendezvous.read_uid = lambda uid_key: None  # UID unreadable
        p._nccl_registry.register(
            "0:x", torch.zeros(p._nccl_entry_size, dtype=torch.uint8)
        )
        sent = []
        p._send = lambda *a, **k: sent.append(a)
        p._handle_request(
            {
                "transfer_id": "0:x",
                "resp_key": "rk",
                "receiver_rank": 1,
                "receiver_replica": "pol",
                "uid_key": "uk",  # present, but read_uid returns None
            }
        )
        self.assertEqual(p._nccl_rendezvous.replies, [("rk", TransferStatus.NEED_UID)])
        self.assertEqual(sent, [])  # no doomed empty-UID send
        self.assertEqual(p._nccl_registry.get("0:x").inflight, 0)  # lease released

    def test_uid_toctou_between_precheck_and_build_fails_fast(self):
        # UID readable at the pre-ACCEPTED precheck but gone by the time _send
        # builds (expired / overwritten): the comm-cache empty-UID guard must
        # fail FAST (release lease + quarantine) instead of wedging 600s in
        # create_nccl_comm from an all-zero UID.
        from cosmos_rl.utils.payload_transport.nccl.comm_cache import CommCache

        p = _make_producer()
        built = []
        p._nccl_comm_cache = CommCache(
            build_fn=lambda u, r: built.append(u) or 7, abort_fn=lambda i: None
        )
        # read_uid: OK on the precheck, empty on _send's re-read (the TOCTOU).
        reads = iter([[1, 2, 3], []])
        p._nccl_rendezvous.read_uid = lambda uid_key: next(reads)
        p._nccl_registry.register(
            "0:x", torch.zeros(p._nccl_entry_size, dtype=torch.uint8)
        )
        p._handle_request(
            {
                "transfer_id": "0:x",
                "resp_key": "rk",
                "receiver_rank": 1,
                "receiver_replica": "pol",
                "uid_key": "uk",
            }
        )
        self.assertEqual(built, [])  # guard fired before build_fn ran
        self.assertEqual(p._nccl_registry.get("0:x").inflight, 0)  # lease released


class TestSendLaunchSerialized(unittest.TestCase):
    """The producer's NCCL launch (group_start -> send -> group_end) must be
    serialized across the sender-thread pool -- concurrent multi-comm launches
    on one GPU deadlock natively at N_POLICY>=2 (Codex root cause)."""

    def test_concurrent_sends_do_not_overlap_group(self):
        import cosmos_rl.utils.payload_transport.nccl.mixins as mixins_mod
        from cosmos_rl.utils.payload_transport.nccl.buffer_registry import (
            SendBufferEntry,
        )

        p = _make_producer()

        class _Cache:  # comm build is allowed concurrent; return a dummy idx
            def __init__(self):
                self.leased_pairs = []

            def get_or_create(self, pair, **kw):
                return 1

            @contextlib.contextmanager
            def leased(self, pair, **kw):
                # The producer must LEASE (pin) the comm for the duration of the
                # send, not merely fetch it, so LRU eviction cannot abort it
                # mid-collective.
                self.leased_pairs.append(pair)
                yield self.get_or_create(pair, **kw)

        p._nccl_comm_cache = _Cache()

        st = {"in_group": 0, "peak": 0}
        lk = threading.Lock()

        def g_start(c):
            with lk:
                st["in_group"] += 1
                st["peak"] = max(st["peak"], st["in_group"])

        def g_send(*a, **k):
            time.sleep(0.02)  # hold the launch open to expose interleaving

        def g_end(c):
            with lk:
                st["in_group"] -= 1

        eA = SendBufferEntry("0:a", buffer=object())
        eB = SendBufferEntry("0:b", buffer=object())

        with (
            mock.patch.object(mixins_mod, "wait_event", lambda s, e: None),
            mock.patch.object(mixins_mod, "record_event", lambda s: object()),
            mock.patch("cosmos_rl.utils.pynccl.nccl_group_start", g_start),
            mock.patch("cosmos_rl.utils.pynccl.nccl_send", g_send),
            mock.patch("cosmos_rl.utils.pynccl.nccl_group_end", g_end),
        ):
            tA = threading.Thread(target=p._send, args=(eA, 1, None, "polA"))
            tB = threading.Thread(target=p._send, args=(eB, 1, None, "polB"))
            tA.start()
            tB.start()
            tA.join()
            tB.join()

        # Never two threads inside a group at once -> launches serialized.
        self.assertEqual(st["peak"], 1)
        # The send must LEASE the comm (pin it against LRU eviction), not just
        # fetch it.  Without this assertion the fake's get_or_create would keep
        # a reverted `cache.get_or_create(...)` green.
        self.assertEqual(len(p._nccl_comm_cache.leased_pairs), 2)
        self.assertEqual(st["in_group"], 0)


class TestOnBufferFree(unittest.TestCase):
    """The registry callback must wait for every send to drain before dropping
    the tensor (use-after-free), but the wait must be BOUNDED so a send that
    never completes (receiver crash) can't hang teardown (Codex P1)."""

    def test_waits_for_all_events_then_releases(self):
        from cosmos_rl.utils.payload_transport.nccl.buffer_registry import (
            SendBufferEntry,
        )

        p = _make_producer()
        queried = []

        class _Ev:
            def query(self):
                queried.append(True)
                return True  # already complete

        entry = SendBufferEntry(
            transfer_id="0:x", buffer=object(), done_events=[_Ev(), _Ev()]
        )
        p._on_buffer_free(entry)
        self.assertEqual(len(queried), 2)  # polled both receivers' events
        self.assertIsNone(entry.buffer)  # then released

    def test_bounded_wait_when_event_never_completes(self):
        from cosmos_rl.utils.payload_transport.nccl.buffer_registry import (
            SendBufferEntry,
        )

        p = _make_producer()
        p._nccl_send_timeout_ms = 30  # tiny bound so the test is fast

        class _StuckEv:
            def query(self):
                return False  # never completes

        entry = SendBufferEntry(
            transfer_id="0:x", buffer=object(), done_events=[_StuckEv()]
        )
        t0 = time.time()
        p._on_buffer_free(entry)  # must return within ~the bound, not hang
        self.assertLess(time.time() - t0, 5.0)
        self.assertIsNone(entry.buffer)  # released anyway

    def test_no_events_releases_immediately(self):
        from cosmos_rl.utils.payload_transport.nccl.buffer_registry import (
            SendBufferEntry,
        )

        p = _make_producer()
        entry = SendBufferEntry(transfer_id="0:x", buffer=object(), done_events=[])
        p._on_buffer_free(entry)  # must not raise
        self.assertIsNone(entry.buffer)

    def test_waits_for_inflight_lease_before_release(self):
        """The Gap-1 UAF fix: a buffer evicted/freed while a send is LEASED
        (acquired, not yet event-recorded) must not be released until the lease
        clears -- otherwise the storage is dropped before the send reads it."""
        from cosmos_rl.utils.payload_transport.nccl.buffer_registry import (
            SendBufferEntry,
        )

        p = _make_producer()
        p._nccl_send_timeout_ms = 5000  # generous: the lease clears well within
        # inflight=1 models a send that acquired the buffer but hasn't yet
        # reached add_done_event (still blocked on the launch lock / ready_event).
        entry = SendBufferEntry(transfer_id="0:x", buffer=object(), inflight=1)
        observed = []

        def finish_send():
            time.sleep(0.05)
            # While the lease is open, _on_buffer_free must NOT have freed yet.
            observed.append(entry.buffer is None)
            p._nccl_registry.add_done_event(entry, None)  # record + drop lease

        th = threading.Thread(target=finish_send)
        th.start()
        p._on_buffer_free(entry)  # blocks until inflight hits 0
        th.join(timeout=5.0)
        self.assertEqual(observed, [False])  # buffer still alive during the send
        self.assertIsNone(entry.buffer)  # released only after the lease cleared

    def test_bounded_wait_when_lease_never_clears(self):
        """A send that never records/abandons its lease (crashed sender) must
        not hang teardown -- the inflight wait is bounded by the send timeout."""
        from cosmos_rl.utils.payload_transport.nccl.buffer_registry import (
            SendBufferEntry,
        )

        p = _make_producer()
        p._nccl_send_timeout_ms = 30  # tiny bound so the test is fast
        entry = SendBufferEntry(transfer_id="0:x", buffer=object(), inflight=1)
        t0 = time.time()
        p._on_buffer_free(entry)  # must return within ~the bound, not hang
        self.assertLess(time.time() - t0, 5.0)
        self.assertIsNone(entry.buffer)  # released anyway


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class TestGpuPackUnpackRoundtrip(unittest.TestCase):
    """Real-GPU byte fidelity: producer ``_pack`` -> consumer ``_unpack``.

    Exercises the actual device packing/unpacking (incl. padding + episode
    truncation) that the 2-rank E2E depends on, without needing a second
    rank — so it runs on a single GPU.
    """

    def test_pack_then_unpack_on_device(self):
        from cosmos_rl.utils.payload_transport.nccl.strategy import _unpack

        device = torch.device("cuda:0")
        p = _make_producer()
        p._nccl_device = device

        ep_len = 6  # shorter than max_steps=10 -> exercises pad + truncate
        obs = torch.randn(ep_len, 4, device=device)
        actions = torch.randn(ep_len, 2, device=device)
        rewards = torch.arange(ep_len, dtype=torch.float32, device=device)
        traj = {
            "observations": obs,
            "actions": actions,
            "rewards": rewards,
            "episode_length": ep_len,
        }

        buf, ready_event = p._pack(traj, ep_len)
        self.assertEqual(buf.device.type, "cuda")
        self.assertEqual(buf.numel(), p._nccl_entry_size)

        out = _unpack(buf, p._nccl_schema, device)
        # Unpacked tensors live on the GPU and are truncated to the episode.
        self.assertEqual(out["observations"].device.type, "cuda")
        self.assertEqual(out["observations"].shape, (ep_len, 4))
        self.assertTrue(torch.allclose(out["observations"], obs))
        self.assertTrue(torch.allclose(out["actions"], actions))
        self.assertTrue(torch.allclose(out["rewards"], rewards))
        self.assertEqual(int(out["episode_length"][0].item()), ep_len)


if __name__ == "__main__":
    unittest.main()
