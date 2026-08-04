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

"""Unit tests for the NCCL per-transfer rendezvous (CPU, fake Redis).

Covers the three-state handshake (ACCEPTED / MISSING / CANCELLED), the
per-pair unique-ID exchange, and stale-reply hygiene.  No CUDA / NCCL /
real Redis required: ``create_nccl_uid`` and the clock/sleep are injected.
"""

import json
import unittest

from cosmos_rl.utils.payload_transport.nccl.protocol import (
    build_pair_uid_key,
    build_response_key,
)
from cosmos_rl.utils.payload_transport.nccl.rendezvous import (
    NcclRendezvous,
    TransferStatus,
    build_request_message,
    parse_request_message,
)


class FakeRedis:
    """Minimal in-memory Redis: get/set/delete/publish over string values."""

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
        return 1


class _Clock:
    """Deterministic monotonic clock advancing a fixed dt per call."""

    def __init__(self, dt=0.001):
        self.t = 0.0
        self.dt = dt

    def __call__(self):
        self.t += self.dt
        return self.t


PREFIX = "cosmos_rl:exp:job:rollout_comm:0"
REQ_CHANNEL = PREFIX + ":nccl_req"


class TestMessageCodec(unittest.TestCase):
    def test_roundtrip(self):
        msg = build_request_message(
            transfer_id="0:abc",
            sender_rank=3,
            receiver_replica="policy-A",
            receiver_rank=5,
            resp_key="resp",
            uid_key="uidk",
        )
        parsed = parse_request_message(msg)
        self.assertEqual(parsed["transfer_id"], "0:abc")
        self.assertEqual(parsed["sender_rank"], 3)
        self.assertEqual(parsed["receiver_replica"], "policy-A")
        self.assertEqual(parsed["receiver_rank"], 5)
        self.assertEqual(parsed["resp_key"], "resp")
        self.assertEqual(parsed["uid_key"], "uidk")

    def test_parse_bytes(self):
        parsed = parse_request_message(b'{"transfer_id": "x"}')
        self.assertEqual(parsed["transfer_id"], "x")

    def test_parse_malformed_returns_none(self):
        self.assertIsNone(parse_request_message("not json"))
        self.assertIsNone(parse_request_message("[]"))
        self.assertIsNone(parse_request_message('{"no_id": 1}'))
        self.assertIsNone(parse_request_message(42))


class TestInitiate(unittest.TestCase):
    def _rendezvous(self, redis, sleep=None):
        return NcclRendezvous(
            redis,
            PREFIX,
            uid_fn=lambda: [1, 2, 3, 4],
            clock=_Clock(),
            sleep=sleep or (lambda dt: None),
        )

    def test_accept_with_uid_exchange(self):
        redis = FakeRedis()
        resp_key = build_response_key(PREFIX, "0:abc", "policy-A", 5)

        # Simulate the sender replying ACCEPTED on the first poll sleep.
        def on_sleep(_dt):
            redis.set(resp_key, TransferStatus.ACCEPTED.value)

        rv = self._rendezvous(redis, sleep=on_sleep)
        result = rv.initiate(
            transfer_id="0:abc",
            sender_replica="rollout-A",
            sender_rank=3,
            receiver_replica="policy-A",
            receiver_rank=5,
            request_channel=REQ_CHANNEL,
            need_uid=True,
            timeout=1.0,
        )

        self.assertIs(result.status, TransferStatus.ACCEPTED)
        self.assertTrue(result.accepted)
        self.assertEqual(result.uid_chars, [1, 2, 3, 4])
        # UID was published for the (sender=3, receiver=5) pair.
        uid_key = build_pair_uid_key(PREFIX, "rollout-A", 3, "policy-A", 5)
        self.assertEqual(json.loads(redis.get(uid_key)), [1, 2, 3, 4])
        # Request was published to the channel with the resp/uid keys.
        self.assertEqual(len(redis.published), 1)
        channel, raw = redis.published[0]
        self.assertEqual(channel, REQ_CHANNEL)
        parsed = parse_request_message(raw)
        self.assertEqual(parsed["uid_key"], uid_key)
        self.assertEqual(parsed["resp_key"], resp_key)
        # Reply was consumed (deleted) on read.
        self.assertIsNone(redis.get(resp_key))

    def test_missing_without_uid(self):
        redis = FakeRedis()
        resp_key = build_response_key(PREFIX, "0:def", "policy-A", 2)

        def on_sleep(_dt):
            redis.set(resp_key, TransferStatus.MISSING.value)

        rv = self._rendezvous(redis, sleep=on_sleep)
        result = rv.initiate(
            transfer_id="0:def",
            sender_replica="rollout-A",
            sender_rank=1,
            receiver_replica="policy-A",
            receiver_rank=2,
            request_channel=REQ_CHANNEL,
            need_uid=False,
            timeout=1.0,
        )
        self.assertIs(result.status, TransferStatus.MISSING)
        self.assertIsNone(result.uid_chars)
        # No UID key written when need_uid=False.
        self.assertIsNone(
            redis.get(build_pair_uid_key(PREFIX, "rollout-A", 1, "policy-A", 2))
        )

    def test_timeout_yields_cancelled(self):
        redis = FakeRedis()
        rv = self._rendezvous(redis)  # sleep is a no-op; nobody replies
        result = rv.initiate(
            transfer_id="0:ghi",
            sender_replica="rollout-A",
            sender_rank=0,
            receiver_replica="policy-A",
            receiver_rank=1,
            request_channel=REQ_CHANNEL,
            need_uid=True,
            timeout=0.0,  # deadline passes on the first check
        )
        self.assertIs(result.status, TransferStatus.CANCELLED)
        # UID was still minted (caller may GC it); handshake just wasn't answered.
        self.assertEqual(result.uid_chars, [1, 2, 3, 4])

    def test_stale_reply_cleared_before_publish(self):
        redis = FakeRedis()
        resp_key = build_response_key(PREFIX, "0:jkl", "policy-A", 1)
        # A stale ACCEPTED from a prior abandoned attempt.
        redis.set(resp_key, TransferStatus.ACCEPTED.value)

        rv = self._rendezvous(redis)  # no responder this attempt
        result = rv.initiate(
            transfer_id="0:jkl",
            sender_replica="rollout-A",
            sender_rank=0,
            receiver_replica="policy-A",
            receiver_rank=1,
            request_channel=REQ_CHANNEL,
            need_uid=False,
            timeout=0.0,
        )
        # The stale reply must have been cleared at initiate; result is a
        # fresh CANCELLED, not the leftover ACCEPTED.
        self.assertIs(result.status, TransferStatus.CANCELLED)

    def test_request_carries_wall_clock_deadline(self):
        # Bilateral cancellation: the request must carry an absolute wall-clock
        # deadline (now + timeout) so the producer can drop it if dequeued late.
        redis = FakeRedis()
        rv = NcclRendezvous(
            redis,
            PREFIX,
            uid_fn=lambda: [1, 2, 3, 4],
            clock=_Clock(),
            sleep=lambda dt: None,
            wall_clock=lambda: 1000.0,
        )
        rv.initiate(
            transfer_id="0:x",
            sender_replica="rollout-A",
            sender_rank=0,
            receiver_replica="policy-A",
            receiver_rank=1,
            request_channel=REQ_CHANNEL,
            need_uid=False,
            timeout=5.0,
        )
        _channel, raw = redis.published[-1]
        self.assertEqual(parse_request_message(raw)["req_deadline"], 1005.0)

    def test_attempt_scopes_response_key(self):
        # A late ACCEPTED left at attempt 1's response key must NOT be consumed
        # by attempt 2 (retry-generation dedup): each attempt polls a distinct
        # key that only the sender handling that attempt's request writes.
        redis = FakeRedis()
        k1 = build_response_key(PREFIX, "0:x", "policy-A", 1, 1)
        redis.set(k1, TransferStatus.ACCEPTED.value)  # stale attempt-1 reply

        rv = self._rendezvous(redis)  # nobody replies to attempt 2
        result = rv.initiate(
            transfer_id="0:x",
            sender_replica="rollout-A",
            sender_rank=0,
            receiver_replica="policy-A",
            receiver_rank=1,
            request_channel=REQ_CHANNEL,
            need_uid=False,
            timeout=0.0,
            attempt=2,
        )
        # Attempt 2 polls ...:2 and never sees attempt 1's stale ACCEPTED.
        self.assertIs(result.status, TransferStatus.CANCELLED)
        self.assertEqual(redis.get(k1), TransferStatus.ACCEPTED.value)  # untouched
        # The request it published carries attempt 2's distinct response key.
        _channel, raw = redis.published[-1]
        self.assertEqual(
            parse_request_message(raw)["resp_key"],
            build_response_key(PREFIX, "0:x", "policy-A", 1, 2),
        )


class TestSenderSide(unittest.TestCase):
    def test_respond_and_read_uid(self):
        redis = FakeRedis()
        rv = NcclRendezvous(redis, PREFIX)

        uid_key = build_pair_uid_key(PREFIX, "rollout-A", 3, "policy-A", 5)
        redis.set(uid_key, json.dumps([7, 8, 9]))
        self.assertEqual(rv.read_uid(uid_key), [7, 8, 9])

        resp_key = build_response_key(PREFIX, "0:abc")
        rv.respond(resp_key=resp_key, status=TransferStatus.ACCEPTED)
        self.assertEqual(redis.get(resp_key), "accepted")

    def test_read_uid_none_paths(self):
        redis = FakeRedis()
        rv = NcclRendezvous(redis, PREFIX)
        self.assertIsNone(rv.read_uid(None))
        self.assertIsNone(rv.read_uid("absent"))
        redis.set("bad", "not json")
        self.assertIsNone(rv.read_uid("bad"))


if __name__ == "__main__":
    unittest.main()
