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

"""CPU tests for ``UCXXPayloadProducer`` -- the UCXX producer.

This module had no tests at all: every existing UCXX test targets the ring
buffer, the client, or the consumer mixin, and all of the interesting producer
tests would need the optional ``ucxx-cu12`` extra to reach ``setup_ucxx``.

The way around that is the same one ``test_nccl_payload_producer`` uses: bypass
setup and inject the producer state directly, with a fake buffer standing in
for the SHM ring.  That leaves the part worth testing -- the pack path, which
produces the on-wire bytes -- fully exercised without ucxx installed.

The pack path is a *wire contract*: the consumer slices the flat buffer by
``spec.nbytes`` and reinterprets each slot with ``spec.dtype``, so these tests
unpack exactly that way rather than trusting the producer's own view of what it
wrote.
"""

import asyncio
import threading
import unittest
from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest.mock import patch
from unittest import mock

import numpy as np
import torch

from cosmos_rl.utils.payload_transport.ucxx import producer as producer_mod
from cosmos_rl.utils.payload_transport.ucxx.producer import UCXXPayloadProducer
from cosmos_rl.utils.payload_transport.ucxx.ucxx_buffer import UCXXBuffer
from cosmos_rl.utils.payload_transport.layout import TensorLayout
from cosmos_rl.utils.payload_transport.context import PayloadContext, PayloadRef
from cosmos_rl.utils.trajectory import (
    ACTIONS,
    EPISODE_LENGTH,
    OBSERVATIONS,
    REWARDS,
    TERMINATED,
    TRUNCATED,
    build_trajectory_schema,
)

MAX_STEPS, OBS_DIM, ACTION_DIM = 6, 3, 2
DIMS = {"max_steps": MAX_STEPS, "obs_dim": OBS_DIM, "action_dim": ACTION_DIM}

_NP_TO_TORCH = {
    np.dtype("float32"): torch.float32,
    np.dtype("int64"): torch.int64,
    np.dtype("bool"): torch.bool,
}


class _FakeBuffer:
    """Stands in for ``UCXXBuffer`` -- records what the producer handed it."""

    def __init__(self, *, raises: Optional[Exception] = None):
        self.raw_writes = []
        self.dict_writes = []
        self.ports = [7000, 7001]
        self._raises = raises
        self.published_slots = []
        self.abandoned_slots = []
        self.direct_writes = []
        self._reservations = {}

    def write_raw(self, mv) -> int:
        if self._raises is not None:
            raise self._raises
        self.raw_writes.append(bytes(mv))  # copy; the pinned buffer is reused
        return 3

    def write(self, data: Dict[str, Any]) -> int:
        if self._raises is not None:
            raise self._raises
        self.dict_writes.append({k: np.array(v, copy=True) for k, v in data.items()})
        return 4

    def get_handle(self):
        return {"name": "fake-shm", "layout": self.layout.descriptor()}

    def get_slot_generation(self, slot):
        return 1

    def acquire_write_slot(self):
        if self._raises is not None:
            raise self._raises
        raw = np.empty(
            self.layout.payload_nbytes + self.layout.alignment, dtype=np.uint8
        )
        shift = (-raw.ctypes.data) % self.layout.alignment
        storage = raw[shift : shift + self.layout.payload_nbytes]
        self._reservations[(2, 7)] = storage
        return 2, 7, storage

    def publish_write_slot(self, slot, generation):
        self.published_slots.append((slot, generation))
        self.direct_writes.append(bytes(self._reservations[(slot, generation)]))

    def abandon_write_slot(self, slot, generation):
        self.abandoned_slots.append((slot, generation))


def _make_producer(*, buffer: Optional[_FakeBuffer] = None) -> UCXXPayloadProducer:
    p = UCXXPayloadProducer()
    p._ucxx_enabled = True
    p._ucxx_replica_id = "rollout-ucxx-0"
    p._ucxx_ip = "10.0.0.7"
    p._ucxx_port = 7000
    p._ucxx_max_steps = MAX_STEPS
    p._ucxx_obs_dim = OBS_DIM
    p._ucxx_action_dim = ACTION_DIM
    p._ucxx_schema = build_trajectory_schema(DIMS)
    p._ucxx_layout = TensorLayout.compile(p._ucxx_schema)
    p._ucxx_tensor_offsets = p._ucxx_layout.offsets()
    p._ucxx_entry_data_size = p._ucxx_layout.payload_nbytes
    p._ucxx_sender_incarnation = "sender-test"
    p._ucxx_packed_cpu = torch.empty(p._ucxx_entry_data_size, dtype=torch.uint8)
    p._ucxx_gpu_staging = {}
    p._ucxx_buffer = buffer if buffer is not None else _FakeBuffer()
    p._ucxx_buffer.layout = p._ucxx_layout
    return p


def _unpack(raw: bytes, layout: TensorLayout) -> Dict[str, torch.Tensor]:
    """Read the flat payload back the way a consumer does: by schema only."""
    assert len(raw) == layout.payload_nbytes
    buf = torch.empty(layout.payload_nbytes, dtype=torch.uint8)
    buf.copy_(torch.frombuffer(bytearray(raw), dtype=torch.uint8))
    return {name: tensor.clone() for name, tensor in layout.views(buf).items()}


def _trajectory(n_steps, *, device=None, include_episode_length=True):
    def t(*shape, dtype=torch.float32):
        return torch.arange(int(np.prod(shape)), dtype=dtype, device=device).reshape(
            *shape
        )

    traj = {
        OBSERVATIONS: t(n_steps, OBS_DIM),
        ACTIONS: t(n_steps, ACTION_DIM),
        REWARDS: t(n_steps),
        TERMINATED: torch.zeros(n_steps, dtype=torch.bool, device=device),
        TRUNCATED: torch.zeros(n_steps, dtype=torch.bool, device=device),
    }
    if include_episode_length:
        traj[EPISODE_LENGTH] = torch.tensor([n_steps], dtype=torch.int64, device=device)
    return traj


_CUDA = torch.cuda.is_available()
requires_cuda = unittest.skipUnless(_CUDA, "coalesced GPU pack path needs CUDA")


class TestGuards(unittest.TestCase):
    def test_returns_none_when_disabled(self):
        p = _make_producer()
        p._ucxx_enabled = False
        self.assertIsNone(p.publish(_trajectory(3)))

    def test_returns_none_without_a_buffer(self):
        p = _make_producer()
        p._ucxx_buffer = None
        self.assertIsNone(p.publish(_trajectory(3)))

    def test_write_failure_is_contained(self):
        # A failed write must not propagate into the rollout loop -- the
        # producer degrades to "no UCXX metadata" and the caller falls back.
        p = _make_producer(buffer=_FakeBuffer(raises=RuntimeError("shm full")))
        self.assertIsNone(p.publish(_trajectory(3)))


class TestCpuPackPath(unittest.TestCase):
    """CPU tensors are packed directly into a reserved shared-memory slot."""

    def test_writes_reserved_slot_and_pads_varlen_fields(self):
        buf = _FakeBuffer()
        p = _make_producer(buffer=buf)
        out = p.publish(_trajectory(4))  # 4 of MAX_STEPS=6

        self.assertIsNotNone(out)
        self.assertEqual(buf.published_slots, [(2, 7)])
        self.assertEqual(buf.dict_writes, [])
        self.assertEqual(buf.raw_writes, [])
        written = _unpack(buf.direct_writes[0], p._ucxx_layout)
        for name, width in (
            (OBSERVATIONS, OBS_DIM),
            (ACTIONS, ACTION_DIM),
        ):
            self.assertEqual(written[name].shape, (MAX_STEPS, width))
            self.assertTrue((written[name][4:] == 0).all(), f"{name} tail not zeroed")
        for name in (REWARDS, TERMINATED, TRUNCATED):
            self.assertEqual(written[name].shape, (MAX_STEPS,))

    def test_episode_length_is_written_as_int64(self):
        buf = _FakeBuffer()
        p = _make_producer(buffer=buf)
        p.publish(_trajectory(4))
        ep = _unpack(buf.direct_writes[0], p._ucxx_layout)[EPISODE_LENGTH]
        self.assertEqual(ep.dtype, torch.int64)
        self.assertEqual(int(ep[0]), 4)

    def test_strict_batched_policy_uses_shared_memory_slot(self):
        buf = _FakeBuffer()
        p = _make_producer(buffer=buf)

        out = p.publish(_trajectory(4), copy_policy="strict_batched")

        self.assertIsNotNone(out)
        self.assertEqual(buf.published_slots, [(2, 7)])
        self.assertEqual(p._ucxx_payload_metrics["adapter_fallback_fields"], 0)

    def test_trace_reports_the_aligned_payload_size(self):
        p = _make_producer()
        with mock.patch.object(producer_mod.logger, "debug") as debug:
            out = p.publish(_trajectory(4))

        self.assertIsNotNone(out)
        message = debug.call_args.args[0]
        self.assertIn(f"bytes={p._ucxx_layout.payload_nbytes}", message)


class TestReturnedMetadata(unittest.TestCase):
    """The dict the producer returns IS the reference the consumer resolves,
    so its shape is part of the protocol."""

    def test_metadata_fields(self):
        buf = _FakeBuffer()
        p = _make_producer(buffer=buf)
        out = p.publish(_trajectory(4))

        self.assertTrue(out["_ucxx"])
        # The consumer's kill-switch: absent would silently mean "enabled".
        self.assertTrue(out["_ucxx_enabled"])
        self.assertEqual(out["_worker_ip"], "10.0.0.7")
        self.assertEqual(out["_ucxx_port"], 7000)
        self.assertEqual(out["_ports"], [7000, 7001])
        self.assertEqual(out["_slot"], 2)
        self.assertEqual(out["_replica_id"], "rollout-ucxx-0")
        self.assertNotIn("_buffer_handle", out)
        self.assertEqual(out["_slot_generation"], 7)
        self.assertEqual(out["_layout_id"], p._ucxx_layout.layout_id)
        self.assertEqual(out["_sender_incarnation"], "sender-test")
        self.assertEqual(out[EPISODE_LENGTH], 4)
        self.assertEqual(out[REWARDS], [0.0, 1.0, 2.0, 3.0])

    def test_cache_key_fields_are_present(self):
        # Endpoint, producer incarnation, slot and generation isolate every
        # publication from slot reuse and producer restarts.
        out = _make_producer().publish(_trajectory(3))
        for key in (
            "_worker_ip",
            "_ucxx_port",
            "_sender_incarnation",
            "_slot",
            "_slot_generation",
        ):
            self.assertIn(key, out)
            self.assertIsNotNone(out[key])

    def test_context_direct_storage_commits_the_reserved_slot(self):
        buf = _FakeBuffer()
        producer = _make_producer(buffer=buf)
        context = PayloadContext("ucxx", producer, producer._ucxx_layout)
        payload = context.acquire()
        for tensor in payload.tensors.values():
            tensor.zero_()
        payload.tensors[EPISODE_LENGTH].fill_(2)
        payload.tensors[REWARDS][:2] = torch.tensor([4.0, 5.0])
        payload.mark_all_written()

        reference = context.publish(payload)

        self.assertIsInstance(reference, PayloadRef)
        self.assertEqual(buf.published_slots, [(2, 7)])
        self.assertEqual(buf.abandoned_slots, [])
        self.assertEqual(reference["_slot_generation"], 7)
        self.assertNotIn("_buffer_handle", reference)


class TestUCXXServerPorts(unittest.TestCase):
    def setUp(self):
        # Port-policy tests own fake listeners, not a native UCXX worker.
        for name in ("_acquire_ucxx_context", "_release_ucxx_context"):
            self.enterContext(
                patch(f"cosmos_rl.utils.payload_transport.ucxx.ucxx_buffer.{name}")
            )

    @staticmethod
    def _server(base_port, n_threads=4):
        server = UCXXBuffer.__new__(UCXXBuffer)
        server._base_port = base_port
        server._n_threads = n_threads
        server._local_ip = "127.0.0.1"
        server._server_threads = []
        server._shutdown_flag = threading.Event()
        server._server_ready_count = 0
        server._server_ready_lock = threading.Lock()
        server._server_ready_event = threading.Event()
        server._ports = []
        server._listeners = []
        server._server_loops = []
        server._handler_tasks_per_thread = []
        server._active_endpoints = []
        server.config = SimpleNamespace(send_timeout=1.0)
        server._buffer = SimpleNamespace(close=lambda: None)
        return server

    def test_auto_port_is_requested_independently_by_every_listener(self):
        server = self._server(0)
        requested = []

        def bind(thread_idx, port):
            requested.append(port)
            with server._server_ready_lock:
                server._ports.append(41000 + thread_idx)
                server._server_ready_count += 1
                if server._server_ready_count == server._n_threads:
                    server._ports.sort()
                    server._server_ready_event.set()

        server._run_server_loop = bind
        with patch(
            "cosmos_rl.utils.payload_transport.ucxx.ucxx_buffer.UCXX_AVAILABLE",
            True,
        ):
            server.start_server(timeout=1)

        self.assertEqual(requested, [0, 0, 0, 0])
        self.assertEqual(server.ports, [41000, 41001, 41002, 41003])

    def test_explicit_base_port_remains_consecutive(self):
        server = self._server(31000)
        requested = []

        def bind(thread_idx, port):
            requested.append(port)
            with server._server_ready_lock:
                server._ports.append(port)
                server._server_ready_count += 1
                if server._server_ready_count == server._n_threads:
                    server._ports.sort()
                    server._server_ready_event.set()

        server._run_server_loop = bind
        with patch(
            "cosmos_rl.utils.payload_transport.ucxx.ucxx_buffer.UCXX_AVAILABLE",
            True,
        ):
            server.start_server(timeout=1)

        self.assertEqual(sorted(requested), [31000, 31001, 31002, 31003])
        self.assertEqual(server.ports, sorted(requested))

    def test_auto_port_publishes_listener_actual_port(self):
        server = self._server(0, n_threads=1)
        server._listeners = [None]
        server._server_loops = [None]
        server._handler_tasks_per_thread = [[]]
        server._thread_metrics = {}
        server._thread_metrics_lock = threading.Lock()
        server._active_endpoints = []
        server._endpoints_lock = threading.Lock()
        server._shutdown_flag.set()
        listener = SimpleNamespace(port=42017, close=lambda: None, active_clients=0)
        fake_ucxx = SimpleNamespace(
            init=lambda: None,
            create_listener=lambda callback, port: listener,
        )

        with patch(
            "cosmos_rl.utils.payload_transport.ucxx.ucxx_buffer.ucxx", fake_ucxx
        ):
            asyncio.run(server._async_server_main(0, 0))

        self.assertIsNone(server._listeners[0])
        self.assertEqual(server.ports, [42017])

    def test_partial_startup_is_unwound_before_timeout_is_reported(self):
        server = self._server(0, n_threads=2)
        released = threading.Event()

        def bind(thread_idx, port):
            if thread_idx == 0:
                server._listeners[thread_idx] = SimpleNamespace(close=lambda: None)
                server._shutdown_flag.wait(timeout=5)
                server._listeners[thread_idx] = None
                released.set()

        server._run_server_loop = bind
        with (
            patch(
                "cosmos_rl.utils.payload_transport.ucxx.ucxx_buffer.UCXX_AVAILABLE",
                True,
            ),
            self.assertRaisesRegex(RuntimeError, "failed to start 2 threads"),
        ):
            server.start_server(timeout=0.05)

        self.assertTrue(released.is_set())
        self.assertFalse(any(t.is_alive() for t in server._server_threads))
        self.assertEqual(server._listeners, [])
        self.assertEqual(server._ports, [])

    def test_thread_start_failure_unwinds_already_started_threads(self):
        server = self._server(0, n_threads=2)
        released = threading.Event()

        def serve(thread_idx, port):
            server._shutdown_flag.wait(timeout=5)
            released.set()

        server._run_server_loop = serve
        real_start = threading.Thread.start
        start_count = 0

        def fail_second_start(thread):
            nonlocal start_count
            start_count += 1
            if start_count == 2:
                raise RuntimeError("thread start failed")
            return real_start(thread)

        with (
            patch(
                "cosmos_rl.utils.payload_transport.ucxx.ucxx_buffer.UCXX_AVAILABLE",
                True,
            ),
            patch.object(threading.Thread, "start", fail_second_start),
            self.assertRaisesRegex(RuntimeError, "thread start failed"),
        ):
            server.start_server(timeout=1)

        self.assertTrue(released.is_set())
        self.assertFalse(any(t.is_alive() for t in server._server_threads))
        self.assertEqual(server._listeners, [])


@requires_cuda
class TestCoalescedGpuPackPath(unittest.TestCase):
    """The fast path builds the flat payload itself, so it is the one that
    must satisfy the wire contract byte-for-byte."""

    def _write_and_unpack(self, traj):
        buf = _FakeBuffer()
        p = _make_producer(buffer=buf)
        out = p.publish(traj)
        self.assertIsNotNone(out, "write_to_buffer failed")
        self.assertEqual(len(buf.raw_writes), 1, "did not take the coalesced path")
        return _unpack(buf.raw_writes[0], p._ucxx_layout), out

    def test_full_length_roundtrip(self):
        traj = _trajectory(MAX_STEPS, device="cuda")
        got, _ = self._write_and_unpack(traj)
        for name in (OBSERVATIONS, ACTIONS, REWARDS):
            torch.testing.assert_close(got[name], traj[name].cpu())

    def test_short_episode_is_zero_padded(self):
        traj = _trajectory(2, device="cuda")
        got, _ = self._write_and_unpack(traj)
        # Live prefix preserved ...
        torch.testing.assert_close(got[OBSERVATIONS][:2], traj[OBSERVATIONS].cpu())
        # ... and the padded tail is zero, not stale bytes from a prior write.
        self.assertTrue((got[OBSERVATIONS][2:] == 0).all())
        self.assertTrue((got[ACTIONS][2:] == 0).all())
        self.assertTrue((got[REWARDS][2:] == 0).all())

    def test_episode_length_slot_matches_the_metadata(self):
        got, out = self._write_and_unpack(_trajectory(3, device="cuda"))
        self.assertEqual(int(got[EPISODE_LENGTH][0]), 3)
        self.assertEqual(out[EPISODE_LENGTH], 3)

    def test_payload_is_exactly_one_schema_entry(self):
        buf = _FakeBuffer()
        p = _make_producer(buffer=buf)
        p.publish(_trajectory(MAX_STEPS, device="cuda"))
        self.assertEqual(len(buf.raw_writes[0]), p._ucxx_entry_data_size)

    def test_gpu_staging_allocation_is_reused(self):
        buf = _FakeBuffer()
        p = _make_producer(buffer=buf)
        p.publish(_trajectory(2, device="cuda"))
        first = next(iter(p._ucxx_gpu_staging.values()))
        p.publish(_trajectory(3, device="cuda"))
        second = next(iter(p._ucxx_gpu_staging.values()))
        self.assertEqual(first.data_ptr(), second.data_ptr())
        self.assertEqual(len(p._ucxx_gpu_staging), 1)
        self.assertEqual(len(buf.raw_writes), 2)

    def test_failed_gpu_pack_quarantines_all_async_owners(self):
        p = _make_producer()
        trajectory = _trajectory(2, device="cuda")

        def fail_after_enqueue(destination, *args, **kwargs):
            destination.zero_()
            raise RuntimeError("pack failed")

        with mock.patch.object(
            producer_mod, "pack_trajectory_into", side_effect=fail_after_enqueue
        ):
            self.assertIsNone(p.publish(trajectory))

        self.assertEqual(p._ucxx_gpu_staging, {})
        self.assertIsNone(p._ucxx_packed_cpu)
        retained = producer_mod._FAILED_PACK_OWNERS.pop()
        self.assertIs(retained[1], trajectory)
        self.assertTrue(retained[2].is_cuda)
        self.assertFalse(retained[3].is_cuda)

    def test_fields_do_not_bleed_into_neighbouring_slots(self):
        # Every field is written at its schema offset; a wrong offset or a
        # wrong-width write corrupts the NEXT field rather than failing.
        traj = _trajectory(MAX_STEPS, device="cuda")
        traj[TERMINATED] = torch.ones(MAX_STEPS, dtype=torch.bool, device="cuda")
        got, _ = self._write_and_unpack(traj)
        self.assertTrue(got[TERMINATED].all())
        self.assertFalse(got[TRUNCATED].any())  # neighbour untouched
        self.assertEqual(int(got[EPISODE_LENGTH][0]), MAX_STEPS)


@requires_cuda
class TestPackSemanticsMatchNccl(unittest.TestCase):
    """Two behaviours where the UCXX pack path used to disagree with NCCL's.

    Both were pinned as expectedFailure when this file was written; unifying
    the pack loops onto the shared helper fixed them, so they now assert
    directly.
    """

    def test_episode_length_written_even_when_absent_from_the_trajectory(self):
        # `if raw is None: continue` runs BEFORE the EPISODE_LENGTH branch, so
        # a trajectory without the key leaves zeros in that slot -- even though
        # ep_len was already resolved from the observation rows.  The consumer
        # then truncates the episode to length 0.
        buf = _FakeBuffer()
        p = _make_producer(buffer=buf)
        traj = _trajectory(4, device="cuda", include_episode_length=False)
        p.publish(traj)
        got = _unpack(buf.raw_writes[0], p._ucxx_layout)
        self.assertEqual(int(got[EPISODE_LENGTH][0]), 4)

    def test_source_tensors_are_coerced_to_the_schema_dtype(self):
        # The schema IS the wire format: the consumer slices by spec.nbytes and
        # views with spec.dtype.  Packing a float64 observation (what a gym env
        # commonly yields) against a float32 spec writes twice the bytes the
        # slot holds, running into the neighbouring field.
        buf = _FakeBuffer()
        p = _make_producer(buffer=buf)
        traj = _trajectory(MAX_STEPS, device="cuda")
        traj[OBSERVATIONS] = traj[OBSERVATIONS].double()
        out = p.publish(traj)
        self.assertIsNotNone(out, "float64 observations were rejected outright")
        got = _unpack(buf.raw_writes[0], p._ucxx_layout)
        torch.testing.assert_close(got[OBSERVATIONS], traj[OBSERVATIONS].float().cpu())


class TestCleanup(unittest.TestCase):
    def test_cleanup_is_safe_without_a_buffer(self):
        p = _make_producer()
        p._ucxx_buffer = None
        p.close()  # must not raise
        self.assertFalse(p._ucxx_enabled)


if __name__ == "__main__":
    unittest.main()
