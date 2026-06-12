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

"""Tests for the UCXX payload transport (MR3b).

The UCXX network path requires the optional ``ucxx-cu12`` extra and a
machine with UCX + (ideally) RDMA hardware, so the test suite focuses
on what we can validate hermetically:

* ``TensorSpec`` semantics (shape / dtype / nbytes / contains).
* ``SharedRingBuffer`` round-trip and slot state machine -- this is the
  core data structure underlying both same-node and cross-node UCXX
  transfers and is pure POSIX shared memory + numpy.
* ``UCXXPayloadTransport`` registration with the
  :class:`PayloadTransportRegistry` (string mode resolution +
  ``attach_data_packer`` invocation + ``completion_prefix=None`` skips
  controller cleanup).
* ``ucxx_buffer`` import gracefully degrades when ``ucxx-cu12`` is not
  installed (i.e. ``UCXX_AVAILABLE`` is False rather than ImportError).
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from cosmos_rl.utils.payload_transport import (
    PAYLOAD_TRANSFER_KEY,
    PayloadTransportRegistry,
    get_payload_transfer_mode,
)

# Import side-effect: registers the UCXX backend.
import cosmos_rl.utils.payload_transport.ucxx as ucxx_pkg
from cosmos_rl.utils.payload_transport.ucxx import (
    UCXX_AVAILABLE,
    BufferConfig,
    SharedRingBuffer,
    SlotError,
    SlotState,
    TensorSpec,
    UCXXPayloadTransport,
)


# ---------------------------------------------------------------------------
# TensorSpec
# ---------------------------------------------------------------------------


class TestTensorSpec(unittest.TestCase):
    def test_dtype_normalized(self):
        spec = TensorSpec(shape=(4,), dtype=np.float32)
        self.assertIsInstance(spec.dtype, np.dtype)
        self.assertEqual(spec.dtype, np.dtype(np.float32))

    def test_nbytes(self):
        # 4 floats * 4 bytes each = 16 bytes.
        spec = TensorSpec(shape=(4,), dtype=np.float32)
        self.assertEqual(spec.nbytes, 16)
        # Multi-dim: (2, 3) of int64 = 6 * 8 = 48 bytes.
        spec2 = TensorSpec(shape=(2, 3), dtype=np.int64)
        self.assertEqual(spec2.nbytes, 48)

    def test_contains_matches(self):
        spec = TensorSpec(shape=(4,), dtype=np.float32, name="obs")
        ok = np.zeros(4, dtype=np.float32)
        bad_shape = np.zeros(5, dtype=np.float32)
        bad_dtype = np.zeros(4, dtype=np.float64)
        self.assertTrue(spec.contains(ok))
        self.assertFalse(spec.contains(bad_shape))
        self.assertFalse(spec.contains(bad_dtype))


# ---------------------------------------------------------------------------
# SharedRingBuffer
# ---------------------------------------------------------------------------


def _make_schema(max_steps: int = 4, obs_dim: int = 3) -> list:
    return [
        TensorSpec(name="observations", shape=(max_steps, obs_dim), dtype=np.float32),
        TensorSpec(name="actions", shape=(max_steps,), dtype=np.int64),
        TensorSpec(name="rewards", shape=(max_steps,), dtype=np.float32),
        TensorSpec(name="episode_length", shape=(1,), dtype=np.int64),
    ]


class _BufferTestBase(unittest.TestCase):
    """Shared setup/teardown that ensures the SHM segment is unlinked
    even when a test fails -- /dev/shm leakage poisons subsequent runs."""

    def setUp(self) -> None:
        self.schema = _make_schema()
        # Per-test buffer name avoids cross-test SHM collisions when
        # pytest is invoked in parallel mode.
        name = f"cosmos_rl_test_{os.getpid()}_{id(self)}"
        config = BufferConfig(
            buffer_name=name,
            max_entries=4,
            schema=self.schema,
        )
        self.buf = SharedRingBuffer(config, create=True)

    def tearDown(self) -> None:
        try:
            self.buf.unlink()
        except Exception:
            pass
        try:
            self.buf.close()
        except Exception:
            pass


class TestSharedRingBufferBasics(_BufferTestBase):
    def test_initial_state_all_free(self):
        states = self.buf.get_slot_states()
        self.assertEqual(len(states), 4)
        for s in states.values():
            self.assertEqual(s, SlotState.FREE)

    def test_write_then_read_roundtrip(self):
        data = {
            "observations": np.arange(12, dtype=np.float32).reshape(4, 3),
            "actions": np.array([1, 2, 3, 4], dtype=np.int64),
            "rewards": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            "episode_length": np.array([3], dtype=np.int64),
        }
        slot = self.buf.write(data)
        self.assertEqual(self.buf.get_slot_state(slot), SlotState.READY)
        out = self.buf.read(slot)
        np.testing.assert_array_equal(out["observations"], data["observations"])
        np.testing.assert_array_equal(out["actions"], data["actions"])
        np.testing.assert_array_equal(out["rewards"], data["rewards"])
        np.testing.assert_array_equal(out["episode_length"], data["episode_length"])
        # After read, slot should be in READING state until mark_consumed.
        self.assertEqual(self.buf.get_slot_state(slot), SlotState.READING)

    def test_write_raw_path(self):
        # Pre-pack a contiguous byte buffer matching the schema layout.
        data = {
            "observations": np.zeros(12, dtype=np.float32).reshape(4, 3),
            "actions": np.zeros(4, dtype=np.int64),
            "rewards": np.zeros(4, dtype=np.float32),
            "episode_length": np.array([2], dtype=np.int64),
        }
        # Use the regular write to populate, then exercise read_raw to
        # confirm raw bytes round-trip.
        slot = self.buf.write(data)
        raw = self.buf.read_raw(slot)
        # raw should have entry_data_size bytes total.
        expected_size = sum(s.nbytes for s in self.schema)
        self.assertEqual(raw.nbytes, expected_size)
        self.buf.mark_consumed(slot)

    def test_write_raw_size_mismatch_raises(self):
        with self.assertRaises(SlotError):
            self.buf.write_raw(b"too short")

    def test_mark_consumed_returns_slot_to_free(self):
        data = {
            "observations": np.zeros((4, 3), dtype=np.float32),
            "actions": np.zeros(4, dtype=np.int64),
            "rewards": np.zeros(4, dtype=np.float32),
            "episode_length": np.array([1], dtype=np.int64),
        }
        slot = self.buf.write(data)
        self.buf.read(slot)
        self.assertEqual(self.buf.get_slot_state(slot), SlotState.READING)
        self.buf.mark_consumed(slot)
        self.assertEqual(self.buf.get_slot_state(slot), SlotState.FREE)

    def test_mark_consumed_defensive_against_stale_caller(self):
        """Regression: ``mark_consumed`` must NOT clobber a non-READING slot.

        Pre-fix behaviour: a stale reader exiting after the writer had
        already recycled the slot would unconditionally write
        SlotState.FREE -- silently dropping a READY payload (or worse,
        racing a WRITING transition).  The defensive guard turns the
        call into a logged no-op when the slot is not in READING
        state, so any orphan / out-of-order reader exit is harmless.
        """
        data = {
            "observations": np.zeros((4, 3), dtype=np.float32),
            "actions": np.zeros(4, dtype=np.int64),
            "rewards": np.zeros(4, dtype=np.float32),
            "episode_length": np.array([1], dtype=np.int64),
        }

        # Case 1: slot is FREE (the worst-case orphan exit -- the
        # writer has already recycled and the buffer is empty).
        # mark_consumed must be a no-op, NOT transition anything.
        self.assertEqual(self.buf.get_slot_state(0), SlotState.FREE)
        self.buf.mark_consumed(0)
        self.assertEqual(self.buf.get_slot_state(0), SlotState.FREE)

        # Case 2: slot is READY (writer published, no reader yet).
        # A stale mark_consumed must NOT silently drop the payload.
        slot = self.buf.write(data)
        self.assertEqual(self.buf.get_slot_state(slot), SlotState.READY)
        self.buf.mark_consumed(slot)  # stale call from a prior reader
        self.assertEqual(
            self.buf.get_slot_state(slot),
            SlotState.READY,
            "stale mark_consumed must NOT clobber a READY payload",
        )
        # Sanity: the data is still readable.
        out = self.buf.read(slot)
        np.testing.assert_array_equal(out["observations"], data["observations"])
        self.buf.mark_consumed(slot)  # legitimate consumer; READING -> FREE
        self.assertEqual(self.buf.get_slot_state(slot), SlotState.FREE)

    def test_read_view_is_a_view(self):
        # ``read_view`` returns numpy arrays that point into shared
        # memory.  Modifying them should be visible to read_raw.
        data = {
            "observations": np.ones((4, 3), dtype=np.float32),
            "actions": np.zeros(4, dtype=np.int64),
            "rewards": np.zeros(4, dtype=np.float32),
            "episode_length": np.array([4], dtype=np.int64),
        }
        slot = self.buf.write(data)
        view = self.buf.read_view(slot)
        view["observations"][0, 0] = 42.0
        # Re-read fresh to confirm shared memory was updated.
        # (NB: read() requires READY state, so first put back via release_reading.)
        self.buf.release_reading(slot)
        out = self.buf.read(slot)
        self.assertEqual(out["observations"][0, 0], 42.0)
        self.buf.mark_consumed(slot)

    def test_release_reading_back_to_ready(self):
        data = {
            "observations": np.zeros((4, 3), dtype=np.float32),
            "actions": np.zeros(4, dtype=np.int64),
            "rewards": np.zeros(4, dtype=np.float32),
            "episode_length": np.array([1], dtype=np.int64),
        }
        slot = self.buf.write(data)
        self.buf.read(slot)
        self.assertEqual(self.buf.get_slot_state(slot), SlotState.READING)
        self.buf.release_reading(slot)
        self.assertEqual(self.buf.get_slot_state(slot), SlotState.READY)
        self.buf.read(slot)
        self.buf.mark_consumed(slot)

    def test_get_ready_indices_and_count(self):
        for _ in range(2):
            self.buf.write(
                {
                    "observations": np.zeros((4, 3), dtype=np.float32),
                    "actions": np.zeros(4, dtype=np.int64),
                    "rewards": np.zeros(4, dtype=np.float32),
                    "episode_length": np.array([1], dtype=np.int64),
                }
            )
        ready = self.buf.get_ready_indices()
        self.assertEqual(sorted(ready), [0, 1])
        self.assertEqual(self.buf.get_ready_count(), 2)

    def test_get_ready_count_reflects_consumption_after_saturation(self):
        """Regression: ``get_ready_count`` must count live READY slots,
        not the (saturating) header ``entry_count``.

        Before the fix, the header field was returned directly: it
        increments on every write up to ``max_entries`` and never
        decrements, so after the first ring lap ``get_ready_count``
        was permanently equal to ``max_entries`` regardless of how
        many slots had actually been consumed.
        """
        payload = {
            "observations": np.zeros((4, 3), dtype=np.float32),
            "actions": np.zeros(4, dtype=np.int64),
            "rewards": np.zeros(4, dtype=np.float32),
            "episode_length": np.array([1], dtype=np.int64),
        }
        # Saturate the 4-slot buffer.
        for _ in range(4):
            self.buf.write(payload)
        self.assertEqual(self.buf.get_ready_count(), 4)

        # Consume one slot; ready count must drop accordingly.
        self.buf.read(0)
        self.buf.mark_consumed(0)
        self.assertEqual(self.buf.get_ready_count(), 3)
        self.assertNotIn(0, self.buf.get_ready_indices())

    def test_overwrite_when_full(self):
        # Fill the 4-slot buffer, then write a 5th entry: oldest should
        # be overwritten and drops_total bumped.
        for _ in range(5):
            self.buf.write(
                {
                    "observations": np.zeros((4, 3), dtype=np.float32),
                    "actions": np.zeros(4, dtype=np.int64),
                    "rewards": np.zeros(4, dtype=np.float32),
                    "episode_length": np.array([1], dtype=np.int64),
                }
            )
        metrics = self.buf.get_metrics()
        self.assertEqual(metrics.writes_total, 5)
        self.assertEqual(metrics.drops_total, 1)

    def test_write_raises_when_full_and_overwrite_disabled(self):
        for _ in range(4):
            self.buf.write(
                {
                    "observations": np.zeros((4, 3), dtype=np.float32),
                    "actions": np.zeros(4, dtype=np.int64),
                    "rewards": np.zeros(4, dtype=np.float32),
                    "episode_length": np.array([1], dtype=np.int64),
                }
            )
        with self.assertRaises(SlotError):
            self.buf.write(
                {
                    "observations": np.zeros((4, 3), dtype=np.float32),
                    "actions": np.zeros(4, dtype=np.int64),
                    "rewards": np.zeros(4, dtype=np.float32),
                    "episode_length": np.array([1], dtype=np.int64),
                },
                overwrite_if_full=False,
            )

    def test_handle_roundtrip_attaches(self):
        data = {
            "observations": np.full((4, 3), 7.0, dtype=np.float32),
            "actions": np.array([1, 2, 3, 4], dtype=np.int64),
            "rewards": np.zeros(4, dtype=np.float32),
            "episode_length": np.array([4], dtype=np.int64),
        }
        slot = self.buf.write(data)

        handle = self.buf.get_handle()
        attached = SharedRingBuffer.from_handle(handle)
        try:
            out = attached.read(slot)
            np.testing.assert_array_equal(out["observations"], data["observations"])
        finally:
            attached.close()

    def test_metrics_track_writes_and_reads(self):
        data = {
            "observations": np.zeros((4, 3), dtype=np.float32),
            "actions": np.zeros(4, dtype=np.int64),
            "rewards": np.zeros(4, dtype=np.float32),
            "episode_length": np.array([1], dtype=np.int64),
        }
        for _ in range(3):
            slot = self.buf.write(data)
            self.buf.read(slot)
            self.buf.mark_consumed(slot)
        m = self.buf.get_metrics()
        self.assertEqual(m.writes_total, 3)
        self.assertEqual(m.reads_total, 3)
        self.assertEqual(m.drops_total, 0)


# ---------------------------------------------------------------------------
# UCXX transport registration
# ---------------------------------------------------------------------------


class TestUCXXTransportRegistration(unittest.TestCase):
    def test_registered_with_ucxx_name(self):
        transport = PayloadTransportRegistry.get("ucxx")
        self.assertIsInstance(transport, UCXXPayloadTransport)

    def test_completion_prefix_is_none(self):
        # UCXX intentionally uses dict-shaped completion metadata, not
        # a string prefix.  ``completion_prefix=None`` makes
        # ``handle_discarded`` skip UCXX cleanly when partitioning
        # discards by prefix.  See transport.py module docstring for
        # the longer rationale.
        transport = PayloadTransportRegistry.get("ucxx")
        self.assertIsNone(transport.completion_prefix)

    def test_active_for_completion_does_not_match_ucxx(self):
        # No completion string should ever resolve to UCXX (since UCXX
        # does not stamp a prefix).  Even an obviously "ucxx-style"
        # string must not match.
        result = PayloadTransportRegistry.active_for_completion("ucxx:host:7000:42")
        self.assertNotIsInstance(result, UCXXPayloadTransport)

    def test_publish_cleanup_inherited_returns_zero(self):
        # UCXX inherits the base no-op (returns 0): the producer-side
        # ring buffer auto-recycles slots so the controller has nothing
        # to publish.
        transport = UCXXPayloadTransport()
        n = transport.publish_cleanup_for_discarded(
            transfer_ids=["host:7000:1", "host:7000:2"],
            config=None,
            redis_client=None,
        )
        self.assertEqual(n, 0)

    def test_handle_discarded_skips_ucxx(self):
        # End-to-end: a discard whose completion looks UCXX-ish should
        # NOT route to UCXXPayloadTransport.publish_cleanup_for_discarded,
        # because completion_prefix=None excludes UCXX from the dispatch.
        rollouts = [
            SimpleNamespace(completion={"_ucxx": True, "_slot": 1}),
            SimpleNamespace(completion="ucxx:host:7000:1"),
        ]
        with mock.patch.object(
            UCXXPayloadTransport,
            "publish_cleanup_for_discarded",
            return_value=99,
        ) as patched:
            published = PayloadTransportRegistry.handle_discarded(
                rollouts, [], config=SimpleNamespace(), redis_client=None
            )
        self.assertEqual(published, 0)
        patched.assert_not_called()

    def test_get_payload_transfer_mode_with_ucxx(self):
        config = SimpleNamespace(custom={PAYLOAD_TRANSFER_KEY: "ucxx"})
        self.assertEqual(get_payload_transfer_mode(config), "ucxx")

    def test_ucxx_completion_prefix_constant_removed(self):
        # Regression guard: the dead ``UCXX_COMPLETION_PREFIX`` constant
        # was removed when ``completion_prefix`` flipped to None.  Make
        # sure it is not silently re-introduced from either the
        # transport submodule or the package surface.
        from cosmos_rl.utils.payload_transport.ucxx import transport as transport_mod

        self.assertFalse(
            hasattr(transport_mod, "UCXX_COMPLETION_PREFIX"),
            "UCXX_COMPLETION_PREFIX must not be re-introduced; UCXX "
            "uses completion_prefix=None and dict metadata",
        )
        self.assertFalse(
            hasattr(ucxx_pkg, "UCXX_COMPLETION_PREFIX"),
            "UCXX_COMPLETION_PREFIX must not be re-exported from the ucxx package",
        )

    def test_module_exports(self):
        # The package __init__ should re-export everything callers need
        # (minus the deliberately-removed UCXX_COMPLETION_PREFIX and the
        # deprecated ``UCXXTrainerMixin``).
        for symbol in [
            "TensorSpec",
            "SharedRingBuffer",
            "UCXXBuffer",
            "UCXXClient",
            "UCXX_AVAILABLE",
            "UCXXPayloadTransport",
            "UCXXRolloutMixin",
            "UCXXDataPackerMixin",
        ]:
            self.assertTrue(
                hasattr(ucxx_pkg, symbol),
                f"public re-export missing: {symbol}",
            )
        # Negative assertion: the deprecated ``UCXXTrainerMixin`` was
        # removed.  Guard against a future commit re-exporting it
        # (e.g. by accidentally restoring the old import line).
        self.assertFalse(
            hasattr(ucxx_pkg, "UCXXTrainerMixin"),
            "UCXXTrainerMixin was removed -- new code should use "
            "UCXXDataPackerMixin instead",
        )


class TestUCXXAttachDataPacker(unittest.TestCase):
    """The unified ``attach_data_packer`` hook drives per-packer setup."""

    def test_attach_invokes_setup_with_resolved_args(self):
        # When the packer exposes ``_setup_ucxx_data_packer`` (added by
        # MR5's UCXXDataPackerMixin), attach_data_packer must invoke it
        # with the device + tunables resolved from config.custom.
        captured = {}

        class _Packer:
            def _setup_ucxx_data_packer(self, **kwargs):
                captured.update(kwargs)

        config = SimpleNamespace(
            custom={
                "ucxx_prefetch_timeout": 12.5,
                "ucxx_read_max_attempts": 5,
                "ucxx_read_timeout": 30.0,
            }
        )
        UCXXPayloadTransport().attach_data_packer(
            _Packer(),
            config=config,
            device="cuda:0",
        )
        self.assertEqual(captured["device"], "cuda:0")
        self.assertEqual(captured["prefetch_timeout"], 12.5)
        self.assertEqual(captured["max_attempts"], 5)
        self.assertEqual(captured["read_timeout"], 30.0)
        # ``n_chunks`` is no longer part of the UCXX surface (single-
        # chunk per slot); the attach hook must not pass it along.
        self.assertNotIn("n_chunks", captured)

    def test_attach_uses_defaults_when_config_missing(self):
        captured = {}

        class _Packer:
            def _setup_ucxx_data_packer(self, **kwargs):
                captured.update(kwargs)

        UCXXPayloadTransport().attach_data_packer(_Packer(), config=SimpleNamespace())
        self.assertEqual(captured["prefetch_timeout"], 30.0)
        self.assertEqual(captured["max_attempts"], 2)
        self.assertEqual(captured["read_timeout"], 5.0)
        self.assertNotIn("n_chunks", captured)

    def test_attach_noop_when_setup_method_missing(self):
        # Packers that do NOT subclass UCXXDataPackerMixin should be
        # left untouched -- attach must not raise.
        class _PlainPacker:
            pass

        # Should silently no-op.  Just assert it returns without raising.
        UCXXPayloadTransport().attach_data_packer(
            _PlainPacker(), config=SimpleNamespace()
        )

    def test_attach_passes_device_through(self):
        captured = {}

        class _Packer:
            def _setup_ucxx_data_packer(self, **kwargs):
                captured.update(kwargs)

        for device in ("cuda:1", None, "cpu"):
            UCXXPayloadTransport().attach_data_packer(
                _Packer(), config=SimpleNamespace(), device=device
            )
            self.assertEqual(captured["device"], device)

    def test_attach_falls_back_when_config_values_invalid(self):
        # Garbled custom values should fall back to defaults rather
        # than raising at attach time.
        captured = {}

        class _Packer:
            def _setup_ucxx_data_packer(self, **kwargs):
                captured.update(kwargs)

        config = SimpleNamespace(
            custom={
                "ucxx_prefetch_timeout": "not-a-float",
                "ucxx_read_max_attempts": "bad",
                "ucxx_read_timeout": None,
            }
        )
        UCXXPayloadTransport().attach_data_packer(_Packer(), config=config)
        self.assertEqual(captured["prefetch_timeout"], 30.0)
        self.assertEqual(captured["max_attempts"], 2)
        self.assertEqual(captured["read_timeout"], 5.0)
        self.assertNotIn("n_chunks", captured)

    def test_attach_clamps_max_attempts_to_at_least_one(self):
        # max_attempts < 1 is nonsensical (no read would ever happen);
        # the transport should clamp it to 1 rather than disabling reads.
        captured = {}

        class _Packer:
            def _setup_ucxx_data_packer(self, **kwargs):
                captured.update(kwargs)

        for raw in (0, -1, -100):
            config = SimpleNamespace(custom={"ucxx_read_max_attempts": raw})
            UCXXPayloadTransport().attach_data_packer(_Packer(), config=config)
            self.assertEqual(
                captured["max_attempts"],
                1,
                f"max_attempts={raw} should clamp to 1",
            )


# ---------------------------------------------------------------------------
# Optional-extra import contract
# ---------------------------------------------------------------------------


class TestOptionalUcxxExtra(unittest.TestCase):
    """When ``ucxx-cu12`` is not installed, importing the buffer module
    must still succeed and surface ``UCXX_AVAILABLE = False``.  Attempts
    to actually start a server should raise ``RuntimeError`` rather than
    failing at module import."""

    def test_ucxx_available_is_bool(self):
        self.assertIn(UCXX_AVAILABLE, (True, False))

    def test_starting_server_without_ucxx_raises(self):
        if UCXX_AVAILABLE:
            self.skipTest("ucxx-cu12 is installed; skip the negative path")
        from cosmos_rl.utils.payload_transport.ucxx import (
            UCXXBuffer,
            UCXXBufferConfig,
        )

        cfg = UCXXBufferConfig(
            buffer_name=f"cosmos_rl_unavail_{os.getpid()}",
            max_entries=2,
            schema=_make_schema(),
        )
        try:
            buf = UCXXBuffer(cfg)
            with self.assertRaises(RuntimeError):
                buf.start_server()
        finally:
            try:
                buf._buffer.unlink()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# UCXXClient.read port rotation + on-failure fallback
#
# These tests exercise the load-balancing / resilience contract of
# ``UCXXClient.read`` without spinning up a real UCXX server.  We
# instantiate a real ``UCXXClient`` (so ``ucxx.init()`` runs), then
# monkeypatch ``_read_chunk`` with a stub that records every call's
# ``target_port`` and either succeeds or raises a synthetic transport
# error.  Behaviour we lock in:
#
# 1. *Rotation across calls.*  Every call advances the per-client RR
#    counter.  With ``n_chunks=1`` and 4 ports, four consecutive calls
#    must touch ports 0, 1, 2, 3 (in some starting offset, but each port
#    exactly once).
# 2. *Disjoint fallback on transport failures.*  On an error in
#    ``_PORT_ROTATABLE_ERRORS``, the retry uses ports shifted by
#    ``n_chunks`` so a single wedged thread cannot poison the fallback.
# 3. *No retry on slot/protocol errors.*  ``StaleSlotError`` and
#    server-side ``RuntimeError`` propagate immediately.
# ---------------------------------------------------------------------------


class TestUCXXClientPortRotation(unittest.TestCase):
    """Per-call port rotation + transport-class fallback in ``UCXXClient.read``."""

    def setUp(self):
        if not UCXX_AVAILABLE:
            self.skipTest("ucxx-cu12 not installed; client cannot init")
        from cosmos_rl.utils.payload_transport.ucxx.ucxx_buffer import UCXXClient

        self.client = UCXXClient()
        # Tiny schema so ``_acquire_pinned`` allocates a few bytes.
        self.schema = [TensorSpec(shape=(4,), dtype=np.float32, name="x")]
        self.ports = [13620, 13621, 13622, 13623]
        self.worker_ip = "127.0.0.1"

    def _install_recording_stub(self, *, fail_on_ports=None, fail_with=None):
        """Replace ``_read_slot`` with an async stub.

        ``fail_on_ports`` is an iterable of port numbers; calls whose
        ``target_port`` is in that set raise ``fail_with`` (default:
        ``asyncio.TimeoutError``).  All other calls succeed silently.
        Returns a list to which the stub appends every call's
        ``(port, slot)`` tuple in invocation order.
        """
        import asyncio as _asyncio

        recorded: list[tuple] = []
        bad_ports = set(fail_on_ports or ())
        exc = fail_with or _asyncio.TimeoutError("synthetic timeout")

        async def _stub(
            self_unused,
            worker_ip,
            port,
            slot,
            recv_buf,
            timeout,
        ):
            recorded.append((port, slot))
            if port in bad_ports:
                raise exc

        # Bind to *this* client only (other instances unaffected).
        from cosmos_rl.utils.payload_transport.ucxx.ucxx_buffer import UCXXClient

        self._patcher = mock.patch.object(UCXXClient, "_read_slot", _stub)
        self._patcher.start()
        self.addCleanup(self._patcher.stop)
        return recorded

    def _run(self, coro):
        import asyncio as _asyncio

        return _asyncio.new_event_loop().run_until_complete(coro)

    # -- rotation across calls --------------------------------------------

    def test_rotates_starting_port_across_calls(self):
        recorded = self._install_recording_stub()
        # Force counter to a known starting value so the assertion
        # is deterministic regardless of test ordering.
        self.client._rr_counter = 0

        for slot in range(4):
            self._run(
                self.client.read(
                    self.worker_ip,
                    self.ports[0],
                    slot=slot,
                    schema=self.schema,
                    ports=self.ports,
                )
            )

        used_ports = [p for (p, _) in recorded]
        self.assertEqual(
            used_ports,
            self.ports,
            "calls 0..3 should hit ports[0..3] in rotation order",
        )

    # -- fallback on transport failure ------------------------------------

    def test_falls_back_to_disjoint_port_on_timeout(self):
        # Wedge port[0] only; the retry should use a different port and
        # the read should succeed without raising.
        recorded = self._install_recording_stub(fail_on_ports=[self.ports[0]])
        self.client._rr_counter = 0  # forces first call to start at port[0]

        self._run(
            self.client.read(
                self.worker_ip,
                self.ports[0],
                slot=42,
                schema=self.schema,
                ports=self.ports,
            )
        )

        # Two calls: the failing initial attempt on port[0], then a
        # retry on a different (healthy) port.
        self.assertEqual(len(recorded), 2)
        self.assertEqual(recorded[0][0], self.ports[0], "first attempt -> wedged port")
        self.assertNotEqual(
            recorded[1][0],
            self.ports[0],
            "retry must avoid the wedged port",
        )

    # -- non-transport errors propagate immediately -----------------------

    def test_stale_slot_error_does_not_retry(self):
        from cosmos_rl.utils.payload_transport.ucxx.ucxx_buffer import (
            StaleSlotError,
        )

        recorded = self._install_recording_stub(
            fail_on_ports=[self.ports[0]],
            fail_with=StaleSlotError("Slot 42 unavailable (stale reference)"),
        )
        self.client._rr_counter = 0

        with self.assertRaises(StaleSlotError):
            self._run(
                self.client.read(
                    self.worker_ip,
                    self.ports[0],
                    slot=42,
                    schema=self.schema,
                    ports=self.ports,
                )
            )
        # Exactly one attempt: stale slots are not retryable on a
        # different port.
        self.assertEqual(len(recorded), 1)

    # -- two attempts cap -------------------------------------------------

    def test_propagates_when_both_attempts_fail(self):
        # Wedge two ports such that both attempt-0 and attempt-1 land
        # on a wedged port.  With 4 ports and rotation=0, attempt-0
        # hits port[0]; attempt-1 hits port[1]; if we wedge both, the
        # read must fail-fast after the second attempt rather than
        # spinning.
        #
        # NB: the skip-list filter is computed once at the *start* of
        # ``read()`` (see :meth:`UCXXClient._healthy_ports`), so within
        # a single call attempt-1's rotation still draws from the
        # original ``available_ports``.  The skip-list affects future
        # calls (covered by ``test_quarantine_excludes_failed_port_*``).
        import asyncio as _asyncio

        recorded = self._install_recording_stub(
            fail_on_ports=[self.ports[0], self.ports[1]]
        )
        self.client._rr_counter = 0

        with self.assertRaises(_asyncio.TimeoutError):
            self._run(
                self.client.read(
                    self.worker_ip,
                    self.ports[0],
                    slot=99,
                    schema=self.schema,
                    ports=self.ports,
                )
            )
        self.assertEqual(
            len(recorded),
            2,
            "expected exactly two attempts before giving up",
        )

    # -- per-port skip-list (health-aware rotation) -----------------------
    #
    # Two contracts exercised below:
    # 1. *Quarantine on transport failure.*  A ``_PORT_ROTATABLE_ERRORS``
    #    failure adds the offending ``(worker_ip, port)`` to the
    #    skip-list; the next ``read()`` rotates around it.
    # 2. *Cooldown expiry restores eligibility.*  An expired entry
    #    (timestamp <= now) is treated as "absent"; the port re-enters
    #    rotation without any explicit cleanup.

    def test_quarantine_excludes_failed_port_on_next_call(self):
        # Wedge ports[0] for the FIRST call only; the existing rotation
        # fallback (test_falls_back_to_disjoint_port_on_timeout) makes
        # the call succeed.  After that, ports[0] is in the skip-list,
        # and a SECOND call's rotation must skip it entirely (no
        # first-attempt request lands on ports[0]).
        recorded = self._install_recording_stub(fail_on_ports=[self.ports[0]])
        self.client._rr_counter = 0

        # Call 1: triggers quarantine of ports[0] via failure + healthy
        # retry.  Don't assert on details here -- that's covered by
        # test_falls_back_to_disjoint_port_on_timeout.
        self._run(
            self.client.read(
                self.worker_ip,
                self.ports[0],
                slot=0,
                schema=self.schema,
                ports=self.ports,
            )
        )
        skip_ts = self.client._port_skip_until.get((self.worker_ip, self.ports[0]), 0.0)
        self.assertGreater(
            skip_ts,
            0.0,
            "ports[0] should be in skip-list after a transport failure",
        )

        # Call 2: stub now succeeds on every port (no fail set on
        # remaining slots).  Healthy = [ports[1..3]]; rotation must
        # land somewhere in that subset.
        first_call_calls = len(recorded)
        for slot in range(1, 5):
            self._run(
                self.client.read(
                    self.worker_ip,
                    self.ports[0],
                    slot=slot,
                    schema=self.schema,
                    ports=self.ports,
                )
            )

        post_quarantine = recorded[first_call_calls:]
        used_ports = {p for (p, _) in post_quarantine}
        self.assertNotIn(
            self.ports[0],
            used_ports,
            "ports[0] is quarantined; rotation must skip it on subsequent calls",
        )
        # Sanity: rotation actually distributed over the remaining 3
        # healthy ports.
        self.assertEqual(used_ports, set(self.ports[1:]))

    def test_quarantine_expires_and_port_re_eligible(self):
        # Inject a stale skip-list entry whose deadline is already in
        # the past; the port should be eligible immediately without
        # any explicit cleanup.  This avoids depending on real
        # ``_PORT_QUARANTINE_SEC`` elapsing in the test.
        import time as _time

        self.client._port_skip_until[(self.worker_ip, self.ports[0])] = (
            _time.monotonic() - 1.0
        )  # cooldown already expired

        recorded = self._install_recording_stub()
        self.client._rr_counter = 0

        # Round through all 4 ports; every port must be exercised
        # (including the previously-quarantined one) iff its expiry
        # has lapsed.
        for slot in range(4):
            self._run(
                self.client.read(
                    self.worker_ip,
                    self.ports[0],
                    slot=slot,
                    schema=self.schema,
                    ports=self.ports,
                )
            )
        used_ports = {p for (p, _) in recorded}
        self.assertEqual(
            used_ports,
            set(self.ports),
            "expired skip-list entry must not exclude the port",
        )


# ---------------------------------------------------------------------------
# UCXXClient endpoint pool stale-eviction
#
# When the server-side handler exits after its idle window
# (``_HANDLER_MAX_IDLE_CYCLES * _HANDLER_RECV_TIMEOUT`` = 120 s by
# default) but the client's pool still holds the corresponding
# endpoint, the next checkout would otherwise hand out a dead
# endpoint -- the first send would fail as a transport-class error
# and quarantine an otherwise-healthy port for 30 s.
#
# The fix preemptively closes any pooled endpoint older than
# ``_POOL_ENDPOINT_MAX_AGE_S`` (kept strictly less than the server's
# eviction window) and falls back to ``ucxx.create_endpoint`` for a
# fresh connection.  This test class locks that contract in.
# ---------------------------------------------------------------------------


class TestUCXXClientPoolStaleEviction(unittest.TestCase):
    """Pooled endpoints must not survive past the server's idle window."""

    def setUp(self):
        if not UCXX_AVAILABLE:
            self.skipTest("ucxx-cu12 not installed; client cannot init")
        from cosmos_rl.utils.payload_transport.ucxx.ucxx_buffer import UCXXClient

        self.client = UCXXClient()
        self.worker_ip = "127.0.0.1"
        self.port = 13700

    def _run(self, coro):
        import asyncio as _asyncio

        return _asyncio.new_event_loop().run_until_complete(coro)

    def _make_mock_endpoint(self, name: str):
        """Endpoint stub that satisfies the single-chunk wire protocol.

        ``recv(status)`` returns status=0 (success); ``recv(payload)``
        is a no-op (the test passes a small zero-init buffer); any
        ``close`` call flips ``closed`` so the test can assert the
        stale endpoint was actually closed before being replaced.
        """

        class _MockEndpoint:
            def __init__(self_inner):
                self_inner.name = name
                self_inner.closed = False
                self_inner.send_calls = 0
                self_inner.recv_calls = 0

            async def send(self_inner, arr):
                self_inner.send_calls += 1

            async def recv(self_inner, arr):
                self_inner.recv_calls += 1
                # First recv = status byte; subsequent = payload.
                if arr.dtype == np.uint8 and arr.size == 1:
                    arr[0] = 0

            async def close(self_inner):
                self_inner.closed = True

        return _MockEndpoint()

    def test_stale_pooled_endpoint_is_closed_and_replaced(self):
        """Endpoint older than ``_POOL_ENDPOINT_MAX_AGE_S`` must be evicted."""
        import collections
        import time as _time

        from cosmos_rl.utils.payload_transport.ucxx.ucxx_buffer import (
            _POOL_ENDPOINT_MAX_AGE_S,
        )

        stale_ep = self._make_mock_endpoint("stale")
        # Last-use older than the threshold: must be evicted.
        stale_ep._pool_last_use = _time.monotonic() - _POOL_ENDPOINT_MAX_AGE_S - 5.0
        key = (self.worker_ip, self.port)
        self.client._pool[key] = collections.deque([stale_ep])

        fresh_ep = self._make_mock_endpoint("fresh")

        async def _fake_create_endpoint(*args, **kwargs):
            return fresh_ep

        recv_buf = np.zeros(16, dtype=np.uint8)
        with mock.patch(
            "cosmos_rl.utils.payload_transport.ucxx.ucxx_buffer.ucxx.create_endpoint",
            side_effect=_fake_create_endpoint,
        ):
            self._run(
                self.client._read_slot(
                    self.worker_ip,
                    self.port,
                    slot=0,
                    recv_buf=recv_buf,
                    timeout=5.0,
                )
            )

        self.assertTrue(
            stale_ep.closed,
            "stale pooled endpoint must be closed on checkout",
        )
        self.assertGreater(
            fresh_ep.send_calls,
            0,
            "fresh endpoint must have served the request",
        )
        # After successful read, the fresh endpoint is returned to
        # the pool with a current timestamp.
        pool_after = self.client._pool.get(key)
        self.assertIsNotNone(pool_after)
        self.assertEqual(len(pool_after), 1)
        self.assertIs(pool_after[0], fresh_ep)
        self.assertGreater(
            getattr(fresh_ep, "_pool_last_use", 0.0),
            0.0,
            "returned endpoint must be timestamped for next age check",
        )

    def test_fresh_pooled_endpoint_is_reused(self):
        """Endpoint younger than the threshold must NOT be evicted."""
        import collections
        import time as _time

        recent_ep = self._make_mock_endpoint("recent")
        recent_ep._pool_last_use = _time.monotonic()  # just used
        key = (self.worker_ip, self.port)
        self.client._pool[key] = collections.deque([recent_ep])

        # If this were called we'd allocate a fresh endpoint -- the
        # test asserts the recent one was reused instead by failing
        # the create-endpoint call.
        async def _fail_create_endpoint(*args, **kwargs):
            raise AssertionError(
                "create_endpoint must not be called when a fresh "
                "pooled endpoint is available"
            )

        recv_buf = np.zeros(16, dtype=np.uint8)
        with mock.patch(
            "cosmos_rl.utils.payload_transport.ucxx.ucxx_buffer.ucxx.create_endpoint",
            side_effect=_fail_create_endpoint,
        ):
            self._run(
                self.client._read_slot(
                    self.worker_ip,
                    self.port,
                    slot=0,
                    recv_buf=recv_buf,
                    timeout=5.0,
                )
            )

        self.assertFalse(
            recent_ep.closed,
            "fresh pooled endpoint must not be closed",
        )
        self.assertGreater(recent_ep.send_calls, 0)


if __name__ == "__main__":
    unittest.main()
