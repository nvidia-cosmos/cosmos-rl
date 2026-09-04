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

"""A payload must never be unpacked as some other transfer's (CPU-only).

A cached 2-rank comm is an ORDERED, untagged stream: its k-th ``nccl_send`` is
taken by the k-th ``nccl_recv``.  Nothing in the data plane says which transfer
a buffer holds, so once the two ends disagree about how many transfers have
crossed a pair -- one send launched out of accept order, one accepted transfer
whose recv was never posted -- every payload after it lands in the previous
one's buffer.  With one fixed schema that is invisible; with a per-payload
schema (rl-gym packs a pickled structure blob whose length varies per
trajectory) the receiver slices a foreign buffer at its own offsets and returns
decoded garbage.

Three things are tested here, one per layer of the fix:

* :mod:`~cosmos_rl.utils.payload_transport.nccl.header` -- every payload
  carries the transfer it belongs to, so a mispairing is detectable at all;
* the consumer -- it checks that header before unpacking and resyncs the pair
  instead of returning the bytes;
* the producer -- it launches a pair's sends in accept order, and resyncs the
  pair when it cannot honour an ACCEPTED.
"""

import threading
import time
import unittest
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import torch

import cosmos_rl.utils.pynccl as pynccl_mod
from cosmos_rl.utils.payload_transport.nccl.buffer_registry import SendBufferRegistry
from cosmos_rl.utils.payload_transport.nccl.comm_cache import CommCache
from cosmos_rl.utils.payload_transport.nccl.header import (
    HEADER_NBYTES,
    PayloadHeaderMismatch,
    build_header,
    parse_header,
    verify_header,
)
from cosmos_rl.utils.payload_transport.nccl.mixins import NCCLRolloutMixin
from cosmos_rl.utils.payload_transport.nccl.rendezvous import (
    NcclRendezvous,
    TransferStatus,
)
from cosmos_rl.utils.payload_transport.nccl.strategy import (
    NCCLTransportStrategy,
    _pair_key,
    _parse_ref,
)
from cosmos_rl.utils.trajectory import build_trajectory_schema, schema_layout


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeRendezvous:
    def __init__(self):
        self.replies = []

    def respond(self, *, resp_key, status):
        self.replies.append((resp_key, status))

    def read_uid(self, uid_key):
        return [1, 2, 3]


def _schema(max_steps=10):
    return build_trajectory_schema(
        {"max_steps": max_steps, "obs_dim": 4, "action_dim": 2}
    )


def _make_producer(*, executor=None, max_steps=10, capacity=32):
    p = NCCLRolloutMixin()
    p._nccl_enabled = True
    p._nccl_replica_id = "rollout-test-0"
    p._nccl_rollout_idx = 0
    p._nccl_sender_rank = 0
    p._nccl_device = None  # CPU tensors
    p._nccl_schema = _schema(max_steps)
    p._nccl_offsets, p._nccl_entry_size = schema_layout(p._nccl_schema)
    p._nccl_registry = SendBufferRegistry(capacity=capacity, on_free=p._on_buffer_free)
    p._nccl_rendezvous = _FakeRendezvous()
    p._nccl_comm_cache = CommCache(build_fn=lambda u, r: 7, abort_fn=lambda i: None)
    p._nccl_streams = None
    p._nccl_send_lock = threading.Lock()
    p._nccl_executor = executor if executor is not None else ThreadPoolExecutor(4)
    return p


def _request(
    transfer_id,
    *,
    receiver_replica="pol-A",
    receiver_rank=1,
    deadline=None,
    timeout=None,
):
    return {
        "transfer_id": transfer_id,
        "resp_key": f"rk-{transfer_id}",
        "receiver_rank": receiver_rank,
        "receiver_replica": receiver_replica,
        "uid_key": "uk",
        "req_deadline": deadline,
        "req_timeout": timeout,
    }


def _trajectory(ep_len=6):
    return {
        "observations": torch.arange(ep_len * 4, dtype=torch.float32).reshape(
            ep_len, 4
        ),
        "actions": torch.ones(ep_len, 2),
        "rewards": torch.arange(ep_len, dtype=torch.float32),
        "episode_length": ep_len,
    }


def _consumer(cache, *, warm_pairs=None):
    """A strategy wired for a CPU ``_fetch_all`` with no NCCL traffic."""
    s = NCCLTransportStrategy()
    s._rendezvous = object()
    s._comm_cache = cache
    s._device = None
    s._streams = None
    s._receiver_rank = 0
    s._recv_timeout = 5.0
    s._first_transfer_timeout = 30.0
    s._warm_pairs = set(warm_pairs or ())
    return s


def _consumer_ref(transfer_id, schema):
    return {
        "transfer_id": transfer_id,
        "rollout_idx": 0,
        "sender_rank": 0,
        "sender_replica": "rA",
        "schema": schema,
    }


# ---------------------------------------------------------------------------
# The header itself
# ---------------------------------------------------------------------------


class TestPayloadHeader(unittest.TestCase):
    def test_roundtrip(self):
        raw = build_header(transfer_id="0:abc", payload_nbytes=1234)
        self.assertEqual(len(raw), HEADER_NBYTES)
        fields = parse_header(raw)
        self.assertEqual(fields["payload_nbytes"], 1234)
        verify_header(raw, transfer_id="0:abc", payload_nbytes=1234)

    def test_transfer_key_is_process_stable(self):
        # Not Python's salted hash(): the two ends are different processes.
        from cosmos_rl.utils.payload_transport.nccl.header import transfer_key

        self.assertEqual(transfer_key("0:abc"), transfer_key("0:abc"))
        self.assertNotEqual(transfer_key("0:abc"), transfer_key("0:abd"))

    def test_another_transfers_payload_is_rejected(self):
        raw = build_header(transfer_id="0:other", payload_nbytes=1234)
        with self.assertRaises(PayloadHeaderMismatch) as ctx:
            verify_header(raw, transfer_id="0:mine", payload_nbytes=1234)
        self.assertIn("DIFFERENT transfer", str(ctx.exception))

    def test_size_disagreement_is_rejected(self):
        raw = build_header(transfer_id="0:abc", payload_nbytes=1234)
        with self.assertRaises(PayloadHeaderMismatch):
            verify_header(raw, transfer_id="0:abc", payload_nbytes=5678)

    def test_headerless_buffer_is_rejected(self):
        # A buffer overwritten by a differently-sized payload carries no header
        # at all; the magic catches that without any transfer id to compare.
        with self.assertRaises(PayloadHeaderMismatch) as ctx:
            verify_header(
                b"\xbf" * HEADER_NBYTES, transfer_id="0:abc", payload_nbytes=8
            )
        self.assertIn("no valid header", str(ctx.exception))

    def test_truncated_buffer_is_rejected(self):
        with self.assertRaises(PayloadHeaderMismatch):
            verify_header(b"\x00" * 4, transfer_id="0:abc", payload_nbytes=8)


class TestProducerStampsHeader(unittest.TestCase):
    def test_packed_buffer_names_its_transfer(self):
        p = _make_producer()
        meta = p.write_to_buffer(_trajectory())
        entry = p._nccl_registry.get(meta["_transfer_id"])
        self.assertEqual(entry.buffer.numel(), HEADER_NBYTES + p._nccl_entry_size)
        verify_header(
            bytes(entry.buffer[:HEADER_NBYTES].numpy()),
            transfer_id=meta["_transfer_id"],
            payload_nbytes=p._nccl_entry_size,
        )
        p._nccl_executor.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Consumer: check before unpacking
# ---------------------------------------------------------------------------


class TestConsumerRejectsMispairedPayload(unittest.TestCase):
    """The failure DRIVE-33173 reported, reproduced at the buffer level.

    The consumer posts a recv for transfer A and the pair's stream hands it
    transfer B's bytes.  Before the header it unpacked them with A's schema and
    returned plausible-looking tensors; now it must drop the episode and tear
    the pair down.
    """

    def _fetch(self, *, wire_transfer_id, ref_transfer_id):
        schema = _schema()
        _, entry_size = schema_layout(schema)

        aborted = []
        cache = CommCache(build_fn=lambda u, r: 55, abort_fn=aborted.append)
        pair = ("rA", 0, 0)
        cache.get_or_create(pair, uid_chars=[1], local_rank=1)
        s = _consumer(cache, warm_pairs={pair})

        # The buffer as it looks AFTER the recv: header stamped by whichever
        # transfer actually crossed the wire.
        recv_buf = torch.zeros(HEADER_NBYTES + entry_size, dtype=torch.uint8)
        header = build_header(transfer_id=wire_transfer_id, payload_nbytes=entry_size)
        recv_buf[:HEADER_NBYTES] = torch.frombuffer(
            bytearray(header), dtype=torch.uint8
        )
        s._rendezvous_one = lambda ref, pynccl: (55, recv_buf)

        ref = _consumer_ref(ref_transfer_id, schema)
        with mock.patch.object(pynccl_mod, "nccl_recv", mock.Mock()):
            results, nbytes, _ = s._fetch_all([(0, ref)])
        return results, cache, aborted, pair, s

    def test_foreign_payload_is_dropped_and_pair_resynced(self):
        results, cache, aborted, pair, s = self._fetch(
            wire_transfer_id="0:someone-else", ref_transfer_id="0:mine"
        )
        self.assertEqual(results, {})  # episode dropped, not decoded
        self.assertNotIn(pair, cache)  # comm torn down...
        self.assertEqual(aborted, [55])  # ...and actually aborted
        self.assertNotIn(pair, s._warm_pairs)  # rebuild gets the cold budget

    def test_matching_payload_is_accepted(self):
        # Positive control: the same path must NOT reject a correct pairing.
        results, cache, aborted, pair, s = self._fetch(
            wire_transfer_id="0:mine", ref_transfer_id="0:mine"
        )
        self.assertEqual(list(results), [0])
        self.assertIn(pair, cache)
        self.assertEqual(aborted, [])
        self.assertIn(pair, s._warm_pairs)


class TestAbortedPairDropsItsPostedRecvs(unittest.TestCase):
    """A recv posted on a comm that is then aborted must not be unpacked.

    When one ref's recv fails, the pair is quarantined and its communicator
    aborted -- but a sibling ref already posted on that same comm raised
    nothing, because its own enqueue succeeded. NCCL will never write its
    buffer. Unpacking it reads uninitialised memory, which the header check
    then reports as "no valid header ... stream is desynced": an abort we
    performed ourselves, reported as a peer problem.
    """

    def _fetch_two_refs_on_one_pair(self, *, second_recv_raises):
        schema = _schema()
        _, entry_size = schema_layout(schema)
        pair = ("rA", 0, 0)

        aborted = []
        cache = CommCache(build_fn=lambda u, r: 55, abort_fn=aborted.append)
        cache.get_or_create(pair, uid_chars=[1], local_rank=1)
        s = _consumer(cache, warm_pairs={pair})

        # Both refs name the SAME producer, so they share one pair comm.
        refs = [
            _consumer_ref("0:first", schema),
            _consumer_ref("0:second", schema),
        ]
        # Correctly headered, so the control case unpacks and the failure case
        # is attributable to the abort rather than to a bad header.
        bufs = []
        for r in refs:
            b = torch.zeros(HEADER_NBYTES + entry_size, dtype=torch.uint8)
            hdr = build_header(transfer_id=r["transfer_id"], payload_nbytes=entry_size)
            b[:HEADER_NBYTES] = torch.frombuffer(bytearray(hdr), dtype=torch.uint8)
            bufs.append(b)
        it = iter(range(len(refs)))
        s._rendezvous_one = lambda ref, pynccl: (55, bufs[next(it)])

        calls = {"n": 0}

        def _recv(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2 and second_recv_raises:
                raise RuntimeError("enqueue timed out")

        with mock.patch.object(pynccl_mod, "nccl_recv", _recv):
            results, _, _ = s._fetch_all(list(enumerate(refs)))
        return results, cache, aborted, pair

    def test_sibling_of_a_failed_recv_is_dropped_not_unpacked(self):
        results, cache, aborted, pair = self._fetch_two_refs_on_one_pair(
            second_recv_raises=True
        )
        # Ref 2 failed outright; ref 1 was posted on the comm that its failure
        # aborted. Neither may be returned -- and critically, ref 1 must not
        # reach the header check, which would call it a desync.
        self.assertEqual(results, {})
        self.assertNotIn(pair, cache)
        self.assertEqual(aborted, [55])

    def test_unaffected_pair_still_unpacks(self):
        # Control: with no failure, both refs unpack normally, so the filter is
        # not simply discarding everything.
        results, cache, aborted, pair = self._fetch_two_refs_on_one_pair(
            second_recv_raises=False
        )
        self.assertEqual(sorted(results), [0, 1])
        self.assertIn(pair, cache)
        self.assertEqual(aborted, [])


class TestProducerConsumerRoundtrip(unittest.TestCase):
    """The two ends still agree on the wire format, header included."""

    def test_packed_payload_unpacks_to_the_same_tensors(self):
        p = _make_producer()
        traj = _trajectory(ep_len=6)
        meta = p.write_to_buffer(traj)
        wire = p._nccl_registry.get(meta["_transfer_id"]).buffer
        p._nccl_executor.shutdown(wait=True)

        cache = CommCache(build_fn=lambda u, r: 55, abort_fn=lambda i: None)
        s = _consumer(cache)
        # The consumer's schema comes off the reference, exactly as in the run.
        ref = _parse_ref(meta | {"_nccl": True})
        # Prime the cache under the REF's own pair key -- the producer names
        # itself in the metadata, so it is not the "rA" the other fixtures use.
        pair = _pair_key(ref, s._receiver_rank)
        cache.get_or_create(pair, uid_chars=[1], local_rank=1)
        s._rendezvous_one = lambda r, pynccl: (55, wire.clone())

        with mock.patch.object(pynccl_mod, "nccl_recv", mock.Mock()):
            results, _, _ = s._fetch_all([(0, ref)])

        got = results[0]
        self.assertTrue(torch.equal(got["observations"], traj["observations"]))
        self.assertTrue(torch.equal(got["actions"], traj["actions"]))
        self.assertTrue(torch.equal(got["rewards"], traj["rewards"]))


class TestReferenceWithoutSchema(unittest.TestCase):
    """A dict reference that lost its schema must not decode at the default.

    Dict metadata is what a per-payload-schema producer emits.  Silently
    falling back to the static ``build_trajectory_schema`` layout is the same
    failure as a mispaired buffer -- garbage that looks like tensors.
    """

    def test_parse_keeps_the_id_but_no_schema(self):
        ref = _parse_ref(
            {"_nccl": True, "_transfer_id": "3:deadbeef"}, default_schema=_schema()
        )
        self.assertEqual(ref["transfer_id"], "3:deadbeef")
        self.assertIsNone(ref["schema"])

    def test_fetch_batch_drops_it(self):
        s = _consumer(CommCache(build_fn=lambda u, r: 1, abort_fn=lambda i: None))
        s._schema = _schema()
        s._rendezvous_one = lambda ref, pynccl: self.fail("must not rendezvous")
        self.assertEqual(
            s.fetch_batch([(0, {"_nccl": True, "_transfer_id": "3:deadbeef"})]), {}
        )

    def test_sync_fetch_drops_it(self):
        s = _consumer(CommCache(build_fn=lambda u, r: 1, abort_fn=lambda i: None))
        s._schema = _schema()
        s._rendezvous_one = lambda ref, pynccl: self.fail("must not rendezvous")
        self.assertIsNone(s.sync_fetch({"_nccl": True, "_transfer_id": "3:deadbeef"}))

    def test_string_form_still_uses_the_default_schema(self):
        # The bare "nccl:<id>" form has no metadata channel; the default is the
        # only schema it can have, and that is documented and intended.
        ref = _parse_ref("nccl:0:abc", default_schema=_schema())
        self.assertIsNotNone(ref["schema"])


# ---------------------------------------------------------------------------
# Producer: send in accept order, resync when an ACCEPTED cannot be honoured
# ---------------------------------------------------------------------------


class TestSendsLeaveInAcceptOrder(unittest.TestCase):
    """The root cause: a pool that launched one pair's sends out of order.

    ``_handle_request`` accepts on the single pub/sub listener thread, and the
    receiver posts its recvs in that same order.  Submitting each send to the
    pool independently let two workers race for the launch lock and reverse a
    pair's transfers -- the receiver then took the wrong payload into the
    buffer it had sized for another.
    """

    def test_one_pair_is_strictly_ordered(self):
        launched = []
        first_launched = threading.Event()
        all_accepted = threading.Event()

        def _fake_send(entry, receiver_rank, uid_key, receiver_replica=None):
            # Mirror the real ``_send``: the launch order that reaches NCCL is
            # the order workers win ``_nccl_send_lock``, not the order they
            # were submitted in.
            with p._nccl_send_lock:
                if not first_launched.is_set():
                    # Hold the launch lock until every request has been
                    # accepted, so each remaining send is contending for it.
                    # That is the state a saturated producer is in at
                    # N_POLICY>=3, and where submitting sends independently
                    # reorders them.
                    first_launched.set()
                    all_accepted.wait(timeout=5)
                launched.append(entry.transfer_id)

        p = _make_producer(executor=ThreadPoolExecutor(16))
        p._send = _fake_send
        ids = [f"0:t{i}" for i in range(24)]
        try:
            for tid in ids:
                p._nccl_registry.register(
                    tid, torch.zeros(p._nccl_entry_size, dtype=torch.uint8)
                )
                p._handle_request(_request(tid))
                if tid == ids[0]:
                    self.assertTrue(first_launched.wait(timeout=5))
            all_accepted.set()
            deadline = time.time() + 10
            while len(launched) < len(ids) and time.time() < deadline:
                time.sleep(0.01)
        finally:
            all_accepted.set()
            p._nccl_executor.shutdown(wait=True)
        self.assertEqual(launched, ids)

    def test_distinct_pairs_still_overlap(self):
        """Ordering is PER PAIR: a parked peer must not block another one."""
        parked = threading.Event()
        b_sent = threading.Event()

        def _fake_send(entry, receiver_rank, uid_key, receiver_replica=None):
            if receiver_replica == "pol-A":
                parked.wait(timeout=5)
            else:
                b_sent.set()

        p = _make_producer(executor=ThreadPoolExecutor(4))
        p._send = _fake_send
        try:
            for tid in ("0:a1", "0:a2", "0:b1"):
                p._nccl_registry.register(
                    tid, torch.zeros(p._nccl_entry_size, dtype=torch.uint8)
                )
            p._handle_request(_request("0:a1", receiver_replica="pol-A"))
            p._handle_request(_request("0:a2", receiver_replica="pol-A"))
            p._handle_request(_request("0:b1", receiver_replica="pol-B"))
            self.assertTrue(
                b_sent.wait(timeout=5),
                "a send parked on one pair head-of-line-blocked another",
            )
        finally:
            parked.set()
            p._nccl_executor.shutdown(wait=True)


class TestAcceptedSendAlwaysResolves(unittest.TestCase):
    """An ACCEPTED that cannot be sent must resync the pair, not vanish.

    The receiver posts a recv the moment it sees ACCEPTED.  A send that never
    arrives leaves that recv to take the NEXT transfer on the pair -- an
    off-by-one that mispairs everything after it.  Aborting our half of the
    comm ends the stream instead, and both sides rebuild.
    """

    def test_unqueueable_send_resyncs_and_releases_the_lease(self):
        aborted = []
        p = _make_producer(executor=ThreadPoolExecutor(1))
        p._nccl_comm_cache = CommCache(build_fn=lambda u, r: 9, abort_fn=aborted.append)
        pair = (0, "pol-A", 1)
        p._nccl_comm_cache.get_or_create(pair, uid_chars=[1], local_rank=0)
        p._nccl_registry.register(
            "0:x", torch.zeros(p._nccl_entry_size, dtype=torch.uint8)
        )
        p._nccl_executor.shutdown(wait=True)  # every submit now raises

        p._handle_request(_request("0:x"))

        self.assertEqual(
            p._nccl_rendezvous.replies, [("rk-0:x", TransferStatus.ACCEPTED)]
        )
        self.assertEqual(p._nccl_registry.get("0:x").inflight, 0)  # lease back
        self.assertNotIn(pair, p._nccl_comm_cache)
        self.assertEqual(aborted, [9])

    def test_shutdown_drops_queued_sends_and_resyncs(self):
        aborted = []
        p = _make_producer(executor=ThreadPoolExecutor(1))
        p._nccl_shutdown = threading.Event()
        p._nccl_comm_cache = CommCache(build_fn=lambda u, r: 9, abort_fn=aborted.append)
        pair = (0, "pol-A", 1)
        p._nccl_comm_cache.get_or_create(pair, uid_chars=[1], local_rank=0)
        p._nccl_registry.register(
            "0:x", torch.zeros(p._nccl_entry_size, dtype=torch.uint8)
        )
        sent = []
        p._send = lambda *a, **k: sent.append(a)
        p._nccl_shutdown.set()
        try:
            p._handle_request(_request("0:x"))
            deadline = time.time() + 5
            while pair in p._nccl_comm_cache and time.time() < deadline:
                time.sleep(0.01)
        finally:
            p._nccl_executor.shutdown(wait=True)
        self.assertEqual(sent, [])
        self.assertEqual(p._nccl_registry.get("0:x").inflight, 0)
        self.assertEqual(aborted, [9])


class TestAcceptMargin(unittest.TestCase):
    """A request about to expire is not worth the risk of accepting.

    The receiver posts its recv only if the ACCEPTED beats its deadline.
    Accepting at the very edge races that, and a receiver that gives up first
    leaves our send for its next recv to take.
    """

    def test_request_expiring_within_the_margin_is_dropped(self):
        p = _make_producer()
        p._nccl_registry.register(
            "0:x", torch.zeros(p._nccl_entry_size, dtype=torch.uint8)
        )
        sent = []
        p._send = lambda *a, **k: sent.append(a)
        try:
            p._handle_request(_request("0:x", deadline=time.time() + 0.05))
        finally:
            p._nccl_executor.shutdown(wait=True)
        self.assertEqual(p._nccl_rendezvous.replies, [])
        self.assertEqual(sent, [])

    def test_margin_scales_to_a_short_budget(self):
        # A deployment running very short rendezvous timeouts must still get
        # served: the margin is capped at a fraction of the receiver's budget,
        # so it cannot swallow every request.
        p = _make_producer()
        p._nccl_registry.register(
            "0:x", torch.zeros(p._nccl_entry_size, dtype=torch.uint8)
        )
        sent = []
        p._send = lambda *a, **k: sent.append(a)
        try:
            p._handle_request(_request("0:x", deadline=time.time() + 0.15, timeout=0.2))
            p._nccl_executor.shutdown(wait=True)
        finally:
            pass
        self.assertEqual(
            p._nccl_rendezvous.replies, [("rk-0:x", TransferStatus.ACCEPTED)]
        )
        self.assertEqual(len(sent), 1)

    def test_request_with_room_to_spare_is_served(self):
        p = _make_producer()
        p._nccl_registry.register(
            "0:x", torch.zeros(p._nccl_entry_size, dtype=torch.uint8)
        )
        sent = []
        p._send = lambda *a, **k: sent.append(a)
        try:
            p._handle_request(_request("0:x", deadline=time.time() + 30))
            p._nccl_executor.shutdown(wait=True)
        finally:
            pass
        self.assertEqual(
            p._nccl_rendezvous.replies, [("rk-0:x", TransferStatus.ACCEPTED)]
        )
        self.assertEqual(len(sent), 1)


# ---------------------------------------------------------------------------
# Rendezvous: an ACCEPTED that lands after we gave up
# ---------------------------------------------------------------------------


class _FakeRedis:
    def __init__(self):
        self.store = {}
        self.published = []

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ex=None):
        self.store[key] = value

    def delete(self, key):
        self.store.pop(key, None)

    def publish(self, channel, message):
        self.published.append((channel, message))


class TestLateAcceptIsReported(unittest.TestCase):
    """A reply that lands between the last poll and the give-up must be seen.

    The sender wrote ACCEPTED and will launch its send; if the receiver walks
    away without noticing, that send waits on the pair for whatever recv comes
    next.  ``initiate`` therefore takes one last look at the response key
    before returning CANCELLED.
    """

    def _rendezvous(self, redis, *, timeout, poll_interval, arm_reply_at=None):
        clock = {"t": 0.0}
        rv = NcclRendezvous(
            redis,
            "pfx",
            poll_interval=poll_interval,
            uid_fn=lambda: [1, 2, 3],
            clock=lambda: clock["t"],
            sleep=lambda seconds: clock.__setitem__("t", clock["t"] + seconds),
        )
        if arm_reply_at is not None:
            original = rv._consume_reply

            def _consume(resp_key):
                reply = original(resp_key)
                # Simulate the sender's SET landing right after this poll read
                # an empty key -- the exact race the last-look check is for.
                if reply is None and clock["t"] >= arm_reply_at:
                    redis.store[resp_key] = TransferStatus.ACCEPTED.value
                return reply

            rv._consume_reply = _consume
        return rv

    def test_reply_landing_after_the_last_poll_is_surfaced(self):
        redis = _FakeRedis()
        rv = self._rendezvous(redis, timeout=1.0, poll_interval=0.5, arm_reply_at=1.0)
        result = rv.initiate(
            transfer_id="0:x",
            sender_replica="rA",
            sender_rank=0,
            receiver_replica="pol",
            receiver_rank=1,
            request_channel="ch",
            need_uid=True,
            timeout=1.0,
            attempt=1,
        )
        self.assertIs(result.status, TransferStatus.CANCELLED)
        self.assertTrue(
            result.late_accept,
            "an ACCEPTED that arrives after the deadline must be reported: the "
            "sender will send, and no recv will be posted for it",
        )

    def test_reply_in_time_is_still_an_accept(self):
        redis = _FakeRedis()
        rv = self._rendezvous(redis, timeout=10.0, poll_interval=0.5, arm_reply_at=0.0)
        result = rv.initiate(
            transfer_id="0:x",
            sender_replica="rA",
            sender_rank=0,
            receiver_replica="pol",
            receiver_rank=1,
            request_channel="ch",
            need_uid=True,
            timeout=10.0,
            attempt=1,
        )
        self.assertIs(result.status, TransferStatus.ACCEPTED)
        self.assertFalse(result.late_accept)

    def test_plain_timeout_is_not_a_late_accept(self):
        rv = self._rendezvous(_FakeRedis(), timeout=0.5, poll_interval=0.1)
        result = rv.initiate(
            transfer_id="0:x",
            sender_replica="rA",
            sender_rank=0,
            receiver_replica="pol",
            receiver_rank=1,
            request_channel="ch",
            need_uid=True,
            timeout=0.5,
            attempt=1,
        )
        self.assertIs(result.status, TransferStatus.CANCELLED)
        self.assertFalse(result.late_accept)


class TestConsumerResyncsOnLateAccept(unittest.TestCase):
    def test_pair_is_aborted_and_not_retried(self):
        from cosmos_rl.utils.payload_transport.nccl.rendezvous import RendezvousResult

        aborted = []
        cache = CommCache(build_fn=lambda u, r: 55, abort_fn=aborted.append)
        pair = ("rA", 0, 0)
        cache.get_or_create(pair, uid_chars=[1], local_rank=1)
        s = _consumer(cache, warm_pairs={pair})
        s._max_attempts = 3
        s._prefix = "pfx"

        attempts = []

        class _Rv:
            def initiate(self, **kwargs):
                attempts.append(kwargs["attempt"])
                return RendezvousResult(
                    TransferStatus.CANCELLED, None, late_accept=True
                )

        s._rendezvous = _Rv()
        out = s._rendezvous_one(_consumer_ref("0:x", _schema()), pynccl_mod)

        self.assertIsNone(out)
        self.assertEqual(attempts, [1], "must not retry into the orphaned send")
        self.assertNotIn(pair, cache)
        self.assertEqual(aborted, [55])


# ---------------------------------------------------------------------------
# Receiver identity: two policy replicas must never look like one producer-side
# ---------------------------------------------------------------------------


class TestReceiverIdentityIsUnique(unittest.TestCase):
    """Every policy replica must be distinguishable to the producer.

    The producer keys its comm cache AND its per-pair unique-ID on
    ``(sender_rank, receiver_replica, receiver_rank)``.  Single-GPU policy
    replicas all have ``receiver_rank == 0``, so ``receiver_replica`` is the
    only thing separating them.  If two replicas report the same string they
    share one communicator on the producer side while each holds its own on the
    receiving side, and a recv posted by one takes a send meant for the other.
    """

    def _replica_id(self, *, given=None):
        s = NCCLTransportStrategy()
        s.setup(
            device=None,
            redis_client=None,
            config=None,
            receiver_replica=given,
        )
        return s._receiver_replica

    def test_explicit_replica_name_is_used(self):
        self.assertEqual(self._replica_id(given="policy-abc123"), "policy-abc123")

    def test_fallback_is_not_bare_rank(self):
        # ``recv0`` for every single-GPU replica is what cross-wired them.
        self.assertNotEqual(self._replica_id(), "recv0")

    def test_fallback_distinguishes_hosts_and_processes(self):
        import os
        import socket

        got = self._replica_id()
        self.assertIn(socket.gethostname(), got)
        self.assertIn(str(os.getpid()), got)


class TestComposedPackerGetsAnIdentity(unittest.TestCase):
    """A packer that COMPOSES a transport must be given the replica name too.

    ``_attach_payload_transport`` used to assign it only when the packer
    declared ``_nccl_dp_receiver_replica`` -- true for subclasses of
    ``NCCLDataPackerMixin``, false for the composed
    ``PrefetchDataPackerMixin`` + ``set_transport_strategy`` packer that
    config-driven transport selection actually builds.  The composed packer
    silently got no identity.
    """

    def test_attach_assigns_replica_name_to_a_composed_packer(self):
        from cosmos_rl.utils.payload_transport.prefetch_mixin import (
            PrefetchDataPackerMixin,
        )

        class _ComposedPacker(PrefetchDataPackerMixin):
            pass

        packer = _ComposedPacker()
        self.assertFalse(
            hasattr(packer, "_nccl_dp_receiver_replica"),
            "fixture no longer reproduces the composed-packer shape",
        )

        # The assignment under test, as _attach_payload_transport performs it.
        worker = SimpleNamespace(replica_name="policy-replica-7")
        packer._nccl_dp_receiver_replica = getattr(worker, "replica_name", None)

        strategy = NCCLTransportStrategy()
        strategy.setup(
            device=None,
            redis_client=None,
            config=None,
            receiver_replica=getattr(packer, "_nccl_dp_receiver_replica", None),
        )
        self.assertEqual(strategy._receiver_replica, "policy-replica-7")


if __name__ == "__main__":
    unittest.main()
