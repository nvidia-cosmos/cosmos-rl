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

"""Multi-replica addressing: distinct rollout replicas must not collide.

The bug this guards against: the comm-cache pair key and the rendezvous
request routing were keyed on ``sender_rank`` alone, which is the rank
*within* a rollout replica (0 for single-GPU replicas).  With >1 rollout
replica, two different senders share ``sender_rank==0`` and would collide:
one producer's comm/request would be reused for another's payload.  The fix
threads a globally-unique ``sender_replica`` identity through the pair key,
the request channel, and the per-pair UID key.
"""

import unittest

from cosmos_rl.utils.payload_transport.nccl.strategy import (
    _pair_key,
    _parse_ref,
)
from cosmos_rl.utils.payload_transport.nccl.mixins import _producer_pair_key
from cosmos_rl.utils.payload_transport.nccl.protocol import (
    build_nccl_prefix,
    build_pair_uid_key,
    build_sender_request_channel,
)
from cosmos_rl.utils.payload_transport.nccl.rendezvous import (
    build_request_message,
    parse_request_message,
)


def _ref(replica, rank, tid):
    return _parse_ref(
        {
            "_nccl": True,
            "_transfer_id": tid,
            "_sender_replica": replica,
            "_sender_rank": rank,
        }
    )


class TestMultiReplicaAddressing(unittest.TestCase):
    def setUp(self):
        self.prefix = build_nccl_prefix(experiment_name="exp", job_id="job")
        # Two rollout replicas, BOTH with per-replica rank 0 (single-GPU
        # replicas) -- the exact collision case.
        self.a = _ref("rollout-A", 0, "0:aaaa")
        self.b = _ref("rollout-B", 0, "1:bbbb")

    def test_refs_carry_sender_replica(self):
        self.assertEqual(self.a["sender_replica"], "rollout-A")
        self.assertEqual(self.b["sender_replica"], "rollout-B")
        self.assertEqual(self.a["sender_rank"], 0)
        self.assertEqual(self.b["sender_rank"], 0)

    def test_pair_keys_distinct_despite_same_rank(self):
        pk_a = _pair_key(self.a, receiver_rank=5)
        pk_b = _pair_key(self.b, receiver_rank=5)
        self.assertNotEqual(pk_a, pk_b, (pk_a, pk_b))

    def test_same_sender_same_pair_key(self):
        # Same replica + rank + receiver -> identical pair key (comm reuse).
        again = _ref("rollout-A", 0, "0:cccc")
        self.assertEqual(_pair_key(self.a, 5), _pair_key(again, 5))

    def test_request_channels_distinct_per_replica(self):
        ch_a = build_sender_request_channel(self.prefix, self.a)
        ch_b = build_sender_request_channel(self.prefix, self.b)
        self.assertNotEqual(ch_a, ch_b)

    def test_uid_keys_distinct_per_replica(self):
        # Same sender_rank + same receiver identity, different rollout replica.
        k_a = build_pair_uid_key(self.prefix, "rollout-A", 0, "policy-A", 5)
        k_b = build_pair_uid_key(self.prefix, "rollout-B", 0, "policy-A", 5)
        self.assertNotEqual(k_a, k_b)

    def test_string_completion_ref_has_no_replica_but_still_parses(self):
        # Bare "nccl:<id>" string (no dict metadata) -- sender_replica falls
        # back to the rollout-idx-derived id so it still resolves single-node.
        ref = _parse_ref("nccl:2:deadbeef")
        self.assertIsNotNone(ref)
        self.assertIn("sender_replica", ref)


class TestMultiPolicyReplicaAddressing(unittest.TestCase):
    """Symmetric mirror of the sender fix: distinct POLICY (receiver) replicas
    must not collide.

    With >1 policy replica each is a separate distributed world, so
    ``receiver_rank`` (== ``dist.get_rank()``) restarts at 0 per replica.
    The producer keyed its comm cache by ``(sender_rank, receiver_rank)``
    alone, so two policy replicas with ``receiver_rank==0`` requesting the
    same sender would share one comm.  The fix threads a globally-unique
    ``receiver_replica`` (the policy's ``replica_name``) through the request
    message, the producer pair key, and the per-pair UID key.
    """

    def setUp(self):
        self.prefix = build_nccl_prefix(experiment_name="exp", job_id="job")

    def test_producer_pair_keys_distinct_per_receiver_replica(self):
        # Same sender_rank + same receiver_rank, different policy replica --
        # the exact collision case (two single-GPU policy replicas).
        pk_a = _producer_pair_key(
            sender_rank=0, receiver_replica="policy-A", receiver_rank=0
        )
        pk_b = _producer_pair_key(
            sender_rank=0, receiver_replica="policy-B", receiver_rank=0
        )
        self.assertNotEqual(pk_a, pk_b, (pk_a, pk_b))

    def test_producer_pair_key_stable_for_same_receiver(self):
        # Same receiver replica + rank -> identical key (comm reuse).
        self.assertEqual(
            _producer_pair_key(0, "policy-A", 0),
            _producer_pair_key(0, "policy-A", 0),
        )

    def test_uid_keys_distinct_per_receiver_replica(self):
        # Same sender + same receiver_rank, different receiver replica.
        k_a = build_pair_uid_key(self.prefix, "rollout-A", 0, "policy-A", 0)
        k_b = build_pair_uid_key(self.prefix, "rollout-A", 0, "policy-B", 0)
        self.assertNotEqual(k_a, k_b)

    def test_uid_keys_unique_across_all_four_axes(self):
        # sender_replica x sender_rank x receiver_replica x receiver_rank ->
        # every one of the 16 combinations must be a distinct key.
        keys = {
            build_pair_uid_key(self.prefix, sr, srank, rr, rrank)
            for sr in ("rollout-A", "rollout-B")
            for srank in (0, 1)
            for rr in ("policy-A", "policy-B")
            for rrank in (0, 1)
        }
        self.assertEqual(len(keys), 16)

    def test_response_keys_distinct_per_receiver(self):
        # TP/PP ranks sharing a DP id request the SAME transfer_id; the resp
        # key must be scoped by receiver identity so acks aren't cross-consumed.
        from cosmos_rl.utils.payload_transport.nccl.protocol import (
            build_response_key,
        )

        a = build_response_key(self.prefix, "0:x", "policy-A", 0)
        b = build_response_key(self.prefix, "0:x", "policy-B", 0)  # other replica
        c = build_response_key(self.prefix, "0:x", "policy-A", 1)  # other rank
        self.assertEqual(len({a, b, c}), 3)

    def test_request_message_round_trips_receiver_replica(self):
        raw = build_request_message(
            transfer_id="0:x",
            sender_rank=0,
            receiver_replica="policy-B",
            receiver_rank=0,
            resp_key="rk",
            uid_key="uk",
        )
        msg = parse_request_message(raw)
        self.assertEqual(msg["receiver_replica"], "policy-B")
        self.assertEqual(msg["receiver_rank"], 0)


if __name__ == "__main__":
    unittest.main()
