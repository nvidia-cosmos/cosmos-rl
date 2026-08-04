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

"""Tests for the producer-side GPU send-buffer registry (CPU-only).

Buffers are opaque here (plain sentinels), so backpressure, idempotency,
and eviction are all exercised without a GPU.
"""

import threading
import time
import unittest

from cosmos_rl.utils.payload_transport.nccl.buffer_registry import SendBufferRegistry


class TestBasics(unittest.TestCase):
    def test_register_get_free(self):
        reg = SendBufferRegistry(capacity=4)
        reg.register("t0", "buf0", nbytes=10)
        self.assertIn("t0", reg)
        self.assertEqual(len(reg), 1)
        entry = reg.get("t0")
        self.assertEqual(entry.buffer, "buf0")
        self.assertEqual(entry.nbytes, 10)

        self.assertTrue(reg.free("t0"))
        self.assertNotIn("t0", reg)
        self.assertIsNone(reg.get("t0"))

    def test_free_is_idempotent(self):
        reg = SendBufferRegistry(capacity=4)
        reg.register("t0", "buf0")
        self.assertTrue(reg.free("t0"))
        # Second free: no-op, returns False, no crash (defensive TODO-8).
        self.assertFalse(reg.free("t0"))
        self.assertFalse(reg.free("never-registered"))

    def test_overwrite_frees_old_buffer(self):
        freed = []
        reg = SendBufferRegistry(capacity=4, on_free=lambda e: freed.append(e.buffer))
        reg.register("t0", "old")
        reg.register("t0", "new")  # idempotent overwrite
        self.assertEqual(freed, ["old"])
        self.assertEqual(reg.get("t0").buffer, "new")
        self.assertEqual(len(reg), 1)

    def test_bookkeeping_on_detached_entry_is_safe(self):
        # An in-flight send holds the entry OBJECT.  If the entry is freed /
        # evicted out of the registry mid-send, the sender must still be able to
        # balance its lease on that detached object without raising -- that is
        # what lets _on_buffer_free wait out the send before releasing storage.
        reg = SendBufferRegistry(capacity=4)
        reg.register("t0", "buf0")
        leased = reg.acquire("t0")  # sender leases (inflight=1)
        self.assertEqual(leased.inflight, 1)
        reg.free("t0")  # concurrent discard detaches + frees the slot
        self.assertIsNone(reg.acquire("t0"))  # no NEW lease after detach
        # The in-flight sender still records/clears its lease on the object.
        reg.add_done_event(leased, object())
        self.assertEqual(leased.inflight, 0)
        reg.abandon_inflight(leased)  # defensive double-release: no underflow
        self.assertEqual(leased.inflight, 0)
        self.assertNotIn("t0", reg)

    def test_acquire_leases_and_none_when_recycled(self):
        reg = SendBufferRegistry(capacity=2, block_timeout=0.0)
        reg.register("t0", "b0")
        e0 = reg.acquire("t0")
        self.assertEqual(e0.inflight, 1)
        # A leased entry is NOT reaped as delivered even with a fired event,
        # because a lease means a send is still being launched.
        self.assertIsNone(reg.acquire("missing"))

    def test_clear_frees_all(self):
        freed = []
        reg = SendBufferRegistry(
            capacity=4, on_free=lambda e: freed.append(e.transfer_id)
        )
        reg.register("t0", "b0")
        reg.register("t1", "b1")
        reg.clear()
        self.assertEqual(len(reg), 0)
        self.assertEqual(sorted(freed), ["t0", "t1"])


class TestBackpressure(unittest.TestCase):
    def test_drop_oldest_when_full_and_nonblocking(self):
        freed = []
        reg = SendBufferRegistry(
            capacity=2,
            block_timeout=0.0,  # never block -> evict oldest immediately
            on_free=lambda e: freed.append(e.transfer_id),
        )
        reg.register("t0", "b0")
        reg.register("t1", "b1")
        reg.register("t2", "b2")  # full -> evict oldest (t0)
        self.assertEqual(len(reg), 2)
        self.assertEqual(freed, ["t0"])
        self.assertNotIn("t0", reg)
        self.assertIn("t1", reg)
        self.assertIn("t2", reg)
        stats = reg.stats()
        self.assertEqual(stats["evicted"], 1)
        self.assertEqual(stats["registered"], 3)

    def test_blocking_register_unblocks_on_free(self):
        reg = SendBufferRegistry(capacity=1, block_timeout=5.0)
        reg.register("t0", "b0")  # registry now full

        done = threading.Event()

        def producer():
            # Blocks until the consumer frees t0, then registers t1.
            reg.register("t1", "b1")
            done.set()

        th = threading.Thread(target=producer)
        th.start()
        # Give the producer a moment to reach the blocking wait.
        time.sleep(0.1)
        self.assertFalse(done.is_set(), "register should be blocked while full")

        reg.free("t0")  # signals capacity
        self.assertTrue(done.wait(timeout=5.0), "register should unblock after free")
        th.join(timeout=5.0)
        self.assertIn("t1", reg)
        self.assertNotIn("t0", reg)
        # Nothing was evicted; the block path drained cleanly.
        self.assertEqual(reg.stats()["evicted"], 0)


class _DoneEvent:
    """Fake send-complete event: ``query()`` reports drained-or-not."""

    def __init__(self, drained=True):
        self._drained = drained

    def query(self):
        return self._drained


class TestReapCompleted(unittest.TestCase):
    """Delivered buffers (done_event fired) are retired under capacity
    pressure BEFORE blocking/evicting -- so a healthy run never hits the
    30s block that made every write stall after 64 sends."""

    def test_reap_retires_delivered_before_eviction(self):
        freed = []
        reg = SendBufferRegistry(
            capacity=2, block_timeout=0.0, on_free=lambda e: freed.append(e.transfer_id)
        )
        reg.register("t0", "b0")
        reg.register("t1", "b1")
        # Both delivered (their sends drained).
        reg.add_done_event(reg.get("t0"), _DoneEvent(True))
        reg.add_done_event(reg.get("t1"), _DoneEvent(True))
        # Registering a 3rd hits capacity -> reap frees the 2 delivered ones,
        # no eviction, no block.
        reg.register("t2", "b2")
        self.assertEqual(sorted(freed), ["t0", "t1"])
        self.assertIn("t2", reg)
        stats = reg.stats()
        self.assertEqual(stats["evicted"], 0)  # reaped, not evicted
        self.assertEqual(stats["freed"], 2)

    def test_reap_skips_unsent_and_evicts_when_none_completed(self):
        freed = []
        reg = SendBufferRegistry(
            capacity=1, block_timeout=0.0, on_free=lambda e: freed.append(e.transfer_id)
        )
        reg.register("t0", "b0")  # un-sent: no done_event -> not reapable
        reg.register("t1", "b1")  # capacity -> reap finds nothing -> evict t0
        self.assertEqual(freed, ["t0"])
        self.assertIn("t1", reg)
        self.assertEqual(reg.stats()["evicted"], 1)

    def test_reap_only_completed_kept_unsent(self):
        freed = []
        reg = SendBufferRegistry(
            capacity=2, block_timeout=0.0, on_free=lambda e: freed.append(e.transfer_id)
        )
        reg.register("t0", "b0")
        reg.register("t1", "b1")
        reg.add_done_event(reg.get("t0"), _DoneEvent(True))  # delivered
        reg.add_done_event(reg.get("t1"), _DoneEvent(False))  # send still in flight
        reg.register("t2", "b2")  # reap frees only t0; t1 kept
        self.assertEqual(freed, ["t0"])
        self.assertIn("t1", reg)
        self.assertIn("t2", reg)


class TestSendDrained(unittest.TestCase):
    """Multi-receiver drain: a transfer_id served to several receivers is
    'drained' only when NO send is still in flight AND every recorded event
    fired (Codex P1: a single done_event could free the shared tensor while
    another rank's send was mid-flight)."""

    def test_drain_requires_all_events_and_no_inflight(self):
        from cosmos_rl.utils.payload_transport.nccl.buffer_registry import (
            SendBufferEntry,
            _send_drained,
        )

        e = SendBufferEntry("t0", "b")
        self.assertFalse(_send_drained(e))  # never sent -> not reapable
        e.done_events = [_DoneEvent(True)]
        self.assertTrue(_send_drained(e))  # single receiver, delivered
        e.done_events = [_DoneEvent(True), _DoneEvent(False)]
        self.assertFalse(_send_drained(e))  # a second receiver still in flight
        e.done_events = [_DoneEvent(True), _DoneEvent(True)]
        e.inflight = 1
        self.assertFalse(_send_drained(e))  # a send started but not yet recorded
        e.inflight = 0
        self.assertTrue(_send_drained(e))  # all done, none pending

    def test_reap_skips_entry_with_inflight_send(self):
        freed = []
        reg = SendBufferRegistry(
            capacity=2, block_timeout=0.0, on_free=lambda e: freed.append(e.transfer_id)
        )
        reg.register("t0", "b0")
        reg.add_done_event(reg.get("t0"), _DoneEvent(True))  # delivered
        reg.register("t1", "b1")
        reg.acquire("t1")  # send leased, not yet recorded
        reg.register("t2", "b2")  # capacity -> reap frees t0 only; t1 kept
        self.assertEqual(freed, ["t0"])
        self.assertIn("t1", reg)
        self.assertIn("t2", reg)


if __name__ == "__main__":
    unittest.main()
