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

"""Tests for the UCXX consumer fetch engine.

``_fetch_batch`` / ``_ucxx_dp_fetch_all`` / ``_read_one`` / ``_to_gpu`` decide
what happens when a remote read fails, and they were entirely untested -- the
NCCL side has a direct equivalent for each of these behaviours.

The engine has two independent retry layers, and the interesting behaviour is
how they interact:

* ``_read_one`` retries ``max_attempts`` times **within** a round, but only for
  errors named in ``_TRANSIENT_UCXX_ERRORS``.  Classification is by
  ``type(e).__name__``, so the fakes below define exception classes with those
  exact names.
* ``_ucxx_dp_fetch_all`` retries whole rounds up to ``_MAX_FETCH_ROUNDS``, but
  only for slots the first layer marked retryable.  A non-transient failure
  (a stale slot) must be dropped immediately -- retrying it burns an RTT per
  round on a slot that cannot come back.

Everything runs on the ``_fetch_batch`` entry point where possible, since that
is what the prefetch worker actually calls, and it spins its own event loop.
"""

import asyncio
import unittest
from typing import Any, Dict, List

import numpy as np
import torch

from cosmos_rl.utils.payload_transport.ucxx.data_packer_mixin import (
    UCXXDataPackerMixin,
)

# The fetch engine and its round cap moved to the strategy that the mixin
# now composes; these tests drive that engine directly.
from cosmos_rl.utils.payload_transport.ucxx.strategy import (
    _MAX_FETCH_ROUNDS,
    UCXXTransportStrategy,
)

# Imported from the data-packer layer, which owns the trajectory field names
# and exposes them on every branch (the transport packages mirror them).
from cosmos_rl.dispatcher.data.packer.tensor_data_packer import (
    ACTIONS,
    EPISODE_LENGTH,
    OBSERVATIONS,
    REWARDS,
)


# ``_TRANSIENT_UCXX_ERRORS`` matches on the class NAME, so these stand in for
# the real ucxx exceptions without needing the extra installed.
class UCXXCanceledError(Exception):
    pass


class UCXXCloseError(Exception):
    pass


class StaleSlotError(Exception):
    """Not in the transient set -- the slot was overwritten; it cannot return."""


class _StubPacker:
    def get_policy_input(self, sample=None, rollout_output=None, *a, **kw):
        return rollout_output


class _Packer(UCXXDataPackerMixin, _StubPacker):
    pass


class _FakeClient:
    """Scripts ``read`` per slot: a list of outcomes consumed in order.

    An outcome is either an exception instance (raised) or a payload dict
    (returned), so a test can say "fail twice then succeed".
    """

    def __init__(self, script: Dict[int, List[Any]]):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: List[int] = []
        self.returned_pinned: List[Any] = []

    async def read(self, worker_ip, port, slot, schema, ports=None, timeout=None):
        self.calls.append(slot)
        outcomes = self.script.get(slot)
        if not outcomes:
            raise AssertionError(f"unscripted read of slot {slot}")
        out = outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]
        if isinstance(out, Exception):
            raise out
        return dict(out)

    def return_pinned(self, buf):
        self.returned_pinned.append(buf)


def _payload(n_steps=3, max_steps=5):
    """A decoded slot: numpy arrays padded to max_steps, as the server sends."""
    obs = np.zeros((max_steps, 2), dtype=np.float32)
    obs[:n_steps] = 1.0
    act = np.zeros((max_steps, 1), dtype=np.float32)
    act[:n_steps] = 2.0
    rew = np.zeros((max_steps,), dtype=np.float32)
    rew[:n_steps] = 3.0
    return {
        OBSERVATIONS: obs,
        ACTIONS: act,
        REWARDS: rew,
        EPISODE_LENGTH: np.array([n_steps], dtype=np.int64),
    }


def _meta(slot=1, ip="10.0.0.1", port=7000):
    return {"_ucxx": True, "_worker_ip": ip, "_ucxx_port": port, "_slot": slot}


def _make_consumer(client, *, max_attempts=2, device="cpu"):
    p = UCXXTransportStrategy()
    p._client = client
    p._device = device
    p._max_attempts = max_attempts
    p._read_timeout = 0.01
    return p


def _fetch(p, metas):
    """Drive the real entry point: [(idx, metadata), ...] -> {cache_key: data}."""
    return p.fetch_batch([(i, m) for i, m in enumerate(metas)])


class TestTransientRetryWithinARound(unittest.TestCase):
    def test_retries_and_succeeds_within_max_attempts(self):
        client = _FakeClient({1: [UCXXCanceledError("flaky"), _payload()]})
        p = _make_consumer(client, max_attempts=2)
        out = _fetch(p, [_meta(slot=1)])
        self.assertEqual(len(out), 1)
        self.assertEqual(client.calls, [1, 1])  # one retry, same round

    def test_gives_up_after_max_attempts_but_stays_retryable(self):
        # Exhausting attempts is NOT terminal: the slot goes back in the pool
        # for the next round, so the total call count spans all rounds.
        client = _FakeClient({1: [UCXXCloseError("down")]})
        p = _make_consumer(client, max_attempts=2)
        out = _fetch(p, [_meta(slot=1)])
        self.assertEqual(out, {})
        self.assertEqual(len(client.calls), 2 * _MAX_FETCH_ROUNDS)


class TestNonTransientIsDroppedImmediately(unittest.TestCase):
    """The point of the retryable flag: a stale slot must not be re-read."""

    def test_no_retry_within_the_round(self):
        client = _FakeClient({1: [StaleSlotError("overwritten")]})
        p = _make_consumer(client, max_attempts=3)
        out = _fetch(p, [_meta(slot=1)])
        self.assertEqual(out, {})
        self.assertEqual(client.calls, [1], "a stale slot was retried")

    def test_no_retry_across_rounds(self):
        # If the round loop treated it as retryable it would cost
        # _MAX_FETCH_ROUNDS extra RTTs on a slot that cannot come back.
        client = _FakeClient({1: [StaleSlotError("overwritten")]})
        p = _make_consumer(client, max_attempts=1)
        _fetch(p, [_meta(slot=1)])
        self.assertEqual(len(client.calls), 1)


class TestRoundLevelRetry(unittest.TestCase):
    def test_succeeds_on_a_later_round(self):
        # max_attempts=1 so each round is exactly one call: fail, fail, succeed.
        client = _FakeClient(
            {1: [UCXXCanceledError("a"), UCXXCanceledError("b"), _payload()]}
        )
        p = _make_consumer(client, max_attempts=1)
        out = _fetch(p, [_meta(slot=1)])
        self.assertEqual(len(out), 1)
        self.assertEqual(len(client.calls), 3)

    def test_exhausting_rounds_drops_the_episode_without_raising(self):
        # A permanently-dead slot must degrade to a missing cache entry, not an
        # exception -- the prefetch worker treats a raise as a whole-batch
        # failure and wipes the cache.
        client = _FakeClient({1: [UCXXCanceledError("dead")]})
        p = _make_consumer(client, max_attempts=1)
        self.assertEqual(_fetch(p, [_meta(slot=1)]), {})


class TestFailureIsolation(unittest.TestCase):
    def test_one_bad_slot_does_not_lose_the_batch(self):
        client = _FakeClient(
            {
                1: [_payload()],
                2: [StaleSlotError("gone")],
                3: [_payload()],
            }
        )
        p = _make_consumer(client)
        out = _fetch(p, [_meta(slot=1), _meta(slot=2), _meta(slot=3)])
        self.assertEqual(len(out), 2, "a single stale slot took down the batch")
        self.assertIn("10.0.0.1:7000:1", out)
        self.assertIn("10.0.0.1:7000:3", out)
        self.assertNotIn("10.0.0.1:7000:2", out)


class TestMetadataFiltering(unittest.TestCase):
    def test_incomplete_metadata_is_skipped_not_read(self):
        client = _FakeClient({1: [_payload()]})
        p = _make_consumer(client)
        bad = {"_ucxx": True, "_worker_ip": "10.0.0.1"}  # no port, no slot
        out = _fetch(p, [bad, _meta(slot=1)])
        self.assertEqual(len(out), 1)
        self.assertEqual(client.calls, [1])

    def test_slot_zero_is_a_valid_slot(self):
        # The guard is `slot is not None`, not truthiness -- ring-buffer slot 0
        # is real, and a truthy check would silently drop every episode landing
        # there.
        client = _FakeClient({0: [_payload()]})
        p = _make_consumer(client)
        out = _fetch(p, [_meta(slot=0)])
        self.assertEqual(len(out), 1)
        self.assertIn("10.0.0.1:7000:0", out)


class TestEpisodeLengthTruncation(unittest.TestCase):
    def test_varlen_fields_are_cut_to_the_episode_length(self):
        client = _FakeClient({1: [_payload(n_steps=3, max_steps=5)]})
        p = _make_consumer(client)
        data = _fetch(p, [_meta(slot=1)])["10.0.0.1:7000:1"]
        self.assertEqual(data[OBSERVATIONS].shape[0], 3)
        self.assertEqual(data[ACTIONS].shape[0], 3)
        self.assertEqual(data[REWARDS].shape[0], 3)
        # Only the padding is removed; the live rows survive intact.
        self.assertTrue(torch.all(data[OBSERVATIONS] == 1.0))

    def test_no_truncation_when_the_episode_fills_the_buffer(self):
        client = _FakeClient({1: [_payload(n_steps=5, max_steps=5)]})
        p = _make_consumer(client)
        data = _fetch(p, [_meta(slot=1)])["10.0.0.1:7000:1"]
        self.assertEqual(data[OBSERVATIONS].shape[0], 5)


class _ExplodingPinned:
    """A pinned buffer whose bulk device copy fails."""

    def to(self, *a, **kw):
        raise RuntimeError("bulk D2H failed")


class TestToGpuFallback(unittest.TestCase):
    def test_falls_back_to_per_tensor_copy_and_still_returns_the_pinned_buffer(
        self,
    ):
        # The bulk path is an optimisation; when it fails the episode must still
        # be delivered -- and the pinned buffer must go back to the pool either
        # way, or the client leaks one per failed fetch.
        pinned = _ExplodingPinned()
        payload = _payload()
        payload["_pinned_buf"] = pinned
        client = _FakeClient({1: [payload]})
        p = _make_consumer(client)

        data = _fetch(p, [_meta(slot=1)])["10.0.0.1:7000:1"]
        self.assertEqual(data[OBSERVATIONS].shape[0], 3)  # delivered anyway
        self.assertEqual(client.returned_pinned, [pinned], "pinned buffer leaked")


class TestSyncFetchContainment(unittest.TestCase):
    """``_sync_fetch`` runs on the trainer thread during a cache miss, so a
    raise there would propagate into training rather than skipping an
    episode."""

    def test_returns_none_on_failure(self):
        client = _FakeClient({1: [StaleSlotError("gone")]})
        p = _make_consumer(client)
        self.assertIsNone(p.sync_fetch(_meta(slot=1)))

    def test_returns_none_when_the_client_is_absent(self):
        p = _make_consumer(None)
        self.assertIsNone(p.sync_fetch(_meta(slot=1)))

    def test_returns_data_on_success(self):
        client = _FakeClient({1: [_payload()]})
        p = _make_consumer(client)
        got = p.sync_fetch(_meta(slot=1))
        self.assertIsNotNone(got)
        self.assertIn(OBSERVATIONS, got)


class TestFetchAllReportsTiming(unittest.TestCase):
    def test_returns_results_and_both_timings(self):
        client = _FakeClient({1: [_payload()]})
        p = _make_consumer(client)
        results, transfer_ms, copy_ms = asyncio.run(p._fetch_all([(0, _meta(slot=1))]))
        self.assertIn(0, results)
        self.assertGreaterEqual(transfer_ms, 0.0)
        self.assertGreaterEqual(copy_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
