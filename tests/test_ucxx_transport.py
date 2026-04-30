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
        # (minus the deliberately-removed UCXX_COMPLETION_PREFIX).
        for symbol in [
            "TensorSpec",
            "SharedRingBuffer",
            "UCXXBuffer",
            "UCXXClient",
            "UCXX_AVAILABLE",
            "UCXXPayloadTransport",
            "UCXXRolloutMixin",
            "UCXXTrainerMixin",
        ]:
            self.assertTrue(
                hasattr(ucxx_pkg, symbol),
                f"public re-export missing: {symbol}",
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
                "ucxx_n_chunks": 4,
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
        self.assertEqual(captured["n_chunks"], 4)
        self.assertEqual(captured["max_attempts"], 5)
        self.assertEqual(captured["read_timeout"], 30.0)

    def test_attach_uses_defaults_when_config_missing(self):
        captured = {}

        class _Packer:
            def _setup_ucxx_data_packer(self, **kwargs):
                captured.update(kwargs)

        UCXXPayloadTransport().attach_data_packer(_Packer(), config=SimpleNamespace())
        self.assertEqual(captured["prefetch_timeout"], 30.0)
        self.assertEqual(captured["n_chunks"], 2)
        self.assertEqual(captured["max_attempts"], 2)
        self.assertEqual(captured["read_timeout"], 60.0)

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
                "ucxx_n_chunks": object(),
                "ucxx_read_max_attempts": "bad",
                "ucxx_read_timeout": None,
            }
        )
        UCXXPayloadTransport().attach_data_packer(_Packer(), config=config)
        self.assertEqual(captured["prefetch_timeout"], 30.0)
        self.assertEqual(captured["n_chunks"], 2)
        self.assertEqual(captured["max_attempts"], 2)
        self.assertEqual(captured["read_timeout"], 60.0)

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


if __name__ == "__main__":
    unittest.main()
