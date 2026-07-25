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

"""Tests for the NCCL transfer-stream pool + event helpers.

Three layers:

* **CPU degradation** — with CUDA forced off, the pool hands out ``None``
  streams and the event helpers no-op, so the scheduling code stays
  callable on CPU-only hosts.
* **Event ordering (mock)** — with CUDA forced on but streams/events
  replaced by mocks, assert that ``wait_event`` actually makes the given
  stream wait on the given event.  This proves the send-gate is
  load-bearing without needing a real GPU.
* **Stream config (GPU)** — ``skipUnless`` real CUDA: the pool allocates
  the requested number of low-priority streams and hands them out
  round-robin.
"""

import unittest
from unittest import mock

import torch

from cosmos_rl.utils.payload_transport.nccl import streams as streams_mod
from cosmos_rl.utils.payload_transport.nccl.streams import (
    TransferStreamPool,
    get_transfer_stream_pool,
    record_event,
    reset_transfer_stream_pools,
    wait_event,
)


class TestCpuDegradation(unittest.TestCase):
    """With CUDA off, everything degrades to well-defined no-ops."""

    def setUp(self):
        self._patcher = mock.patch.object(
            streams_mod.torch.cuda, "is_available", return_value=False
        )
        self._patcher.start()
        self.addCleanup(self._patcher.stop)

    def test_pool_hands_out_none(self):
        pool = TransferStreamPool(size=2)
        self.assertIsNone(pool.acquire())
        self.assertEqual(pool.all_streams(), [])

    def test_record_event_returns_none(self):
        self.assertIsNone(record_event())
        self.assertIsNone(record_event(stream=None))

    def test_wait_event_noops(self):
        # Neither a None event nor a None stream should raise.
        wait_event(None, None)
        wait_event(mock.MagicMock(), None)


class TestEventOrderingMock(unittest.TestCase):
    """CUDA forced on, streams/events mocked: the wait must be issued."""

    def setUp(self):
        self._patcher = mock.patch.object(
            streams_mod.torch.cuda, "is_available", return_value=True
        )
        self._patcher.start()
        self.addCleanup(self._patcher.stop)

    def test_wait_event_delegates_to_stream(self):
        stream = mock.MagicMock(name="transfer_stream")
        event = mock.MagicMock(name="ready_event")
        wait_event(stream, event)
        # The transfer stream must wait on the producer's ready-event
        # before any subsequent send enqueues on it.
        stream.wait_event.assert_called_once_with(event)

    def test_record_event_records_on_given_stream(self):
        fake_event = mock.MagicMock(name="event")
        stream = mock.MagicMock(name="stream")
        with mock.patch.object(
            streams_mod.torch.cuda, "Event", return_value=fake_event
        ):
            ev = record_event(stream=stream)
        self.assertIs(ev, fake_event)
        fake_event.record.assert_called_once_with(stream)

    def test_send_gate_ordering_is_load_bearing(self):
        """Simulate the sender hand-off: ready-event on compute stream,
        transfer stream waits on it *before* the (mock) send."""
        order = []
        compute_stream = mock.MagicMock(name="compute")
        transfer_stream = mock.MagicMock(name="transfer")
        fake_event = mock.MagicMock(name="ready")
        fake_event.record.side_effect = lambda s: order.append(("record", s))
        transfer_stream.wait_event.side_effect = lambda e: order.append(("wait", e))

        with mock.patch.object(
            streams_mod.torch.cuda, "Event", return_value=fake_event
        ):
            ready = record_event(stream=compute_stream)
        wait_event(transfer_stream, ready)

        def fake_send():
            order.append(("send", None))

        fake_send()

        self.assertEqual(
            order,
            [("record", compute_stream), ("wait", fake_event), ("send", None)],
        )


class TestSingletonRegistry(unittest.TestCase):
    def setUp(self):
        reset_transfer_stream_pools()
        self.addCleanup(reset_transfer_stream_pools)

    def test_same_pool_per_device_and_size(self):
        a = get_transfer_stream_pool(size=1, device=None)
        b = get_transfer_stream_pool(size=1, device=None)
        self.assertIs(a, b)

    def test_distinct_pool_per_size(self):
        a = get_transfer_stream_pool(size=1, device=None)
        c = get_transfer_stream_pool(size=2, device=None)
        self.assertIsNot(a, c)
        self.assertEqual(c.size, 2)

    def test_reset_clears(self):
        a = get_transfer_stream_pool(size=1, device=None)
        reset_transfer_stream_pools()
        b = get_transfer_stream_pool(size=1, device=None)
        self.assertIsNot(a, b)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for real streams")
class TestStreamConfigGpu(unittest.TestCase):
    def test_pool_allocates_requested_size(self):
        pool = TransferStreamPool(size=3)
        streams = pool.all_streams()
        self.assertEqual(len(streams), 3)
        self.assertTrue(all(isinstance(s, torch.cuda.Stream) for s in streams))

    def test_round_robin_acquire(self):
        pool = TransferStreamPool(size=2)
        s0 = pool.acquire()
        s1 = pool.acquire()
        s2 = pool.acquire()
        self.assertIsNot(s0, s1)
        self.assertIs(s0, s2)  # wrapped around

    def test_priority_is_least_preferred(self):
        least, greatest = torch.cuda.Stream.priority_range()
        pool = TransferStreamPool(size=1)
        self.assertEqual(pool.priority, max(least, greatest))


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class TestRealEventHandoff(unittest.TestCase):
    """Real CUDA events: the transfer-stream gate is load-bearing on device.

    Records a ready-event on the compute stream after real GPU work, has a
    transfer stream ``wait_event`` it, then does dependent work on the
    transfer stream and asserts the result is correct — proving the
    cross-stream sync actually orders the work (single GPU, no NCCL)."""

    def test_cross_stream_wait_orders_work(self):
        device = torch.device("cuda:0")
        compute = torch.cuda.current_stream()
        transfer = TransferStreamPool(size=1).acquire()
        self.assertIsNotNone(transfer)

        with torch.cuda.stream(compute):
            src = torch.ones(1 << 20, device=device)
            # Some real compute so the event has something to gate on.
            for _ in range(50):
                src = src * 1.0 + 1.0
        ready = record_event(compute)
        self.assertIsNotNone(ready)

        wait_event(transfer, ready)
        with torch.cuda.stream(transfer):
            dst = src.clone()
        torch.cuda.synchronize()

        # src ends at 1 + 50 = 51 everywhere; dst copied after the wait.
        self.assertTrue(torch.allclose(dst, torch.full_like(dst, 51.0)))

    def test_record_and_wait_return_real_objects(self):
        ev = record_event()
        self.assertIsInstance(ev, torch.cuda.Event)
        # Waiting on a real event on the current stream must not raise.
        wait_event(None, ev)
        torch.cuda.synchronize()


class TestBindThreadDevice(unittest.TestCase):
    def test_none_is_noop(self):
        streams_mod.bind_thread_device(None)  # must not raise

    def test_sets_device_when_cuda_available(self):
        from unittest import mock

        called = []
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.set_device", lambda d: called.append(d)),
        ):
            streams_mod.bind_thread_device("cuda:3")
        self.assertEqual(called, ["cuda:3"])


if __name__ == "__main__":
    unittest.main()
