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

"""CPU tests for ``UCXXRolloutMixin`` -- the UCXX producer.

This module had no tests at all: every existing UCXX test targets the ring
buffer, the client, or the consumer mixin, and all of the interesting producer
tests would need the optional ``ucxx-cu12`` extra to reach ``setup_ucxx``.

The way around that is the same one ``test_nccl_rollout_mixin`` uses: bypass
setup and inject the producer state directly, with a fake buffer standing in
for the SHM ring.  That leaves the part worth testing -- the pack path, which
produces the on-wire bytes -- fully exercised without ucxx installed.

The pack path is a *wire contract*: the consumer slices the flat buffer by
``spec.nbytes`` and reinterprets each slot with ``spec.dtype``, so these tests
unpack exactly that way rather than trusting the producer's own view of what it
wrote.
"""

import unittest
from typing import Any, Dict, Optional

import numpy as np
import torch

from cosmos_rl.utils.payload_transport.ucxx.mixins import UCXXRolloutMixin
from cosmos_rl.utils.trajectory import (
    ACTIONS,
    EPISODE_LENGTH,
    OBSERVATIONS,
    REWARDS,
    TERMINATED,
    TRUNCATED,
    build_trajectory_schema,
    schema_layout,
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
        return {"name": "fake-shm"}


def _make_producer(*, buffer: Optional[_FakeBuffer] = None) -> UCXXRolloutMixin:
    p = UCXXRolloutMixin()
    p._ucxx_enabled = True
    p._ucxx_replica_id = "rollout-ucxx-0"
    p._ucxx_ip = "10.0.0.7"
    p._ucxx_port = 7000
    p._ucxx_max_steps = MAX_STEPS
    p._ucxx_obs_dim = OBS_DIM
    p._ucxx_action_dim = ACTION_DIM
    p._ucxx_schema = build_trajectory_schema(DIMS)
    p._ucxx_tensor_offsets, p._ucxx_entry_data_size = schema_layout(p._ucxx_schema)
    p._ucxx_packed_cpu = torch.empty(p._ucxx_entry_data_size, dtype=torch.uint8)
    p._ucxx_buffer = buffer if buffer is not None else _FakeBuffer()
    return p


def _unpack(raw: bytes, schema) -> Dict[str, torch.Tensor]:
    """Read the flat payload back the way a consumer does: by schema only."""
    offsets, entry = schema_layout(schema)
    assert len(raw) == entry, f"payload is {len(raw)}B, schema says {entry}B"
    buf = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    out = {}
    for spec in schema:
        off = offsets[spec.name]
        chunk = buf[off : off + spec.nbytes].clone()
        out[spec.name] = chunk.view(_NP_TO_TORCH[np.dtype(spec.dtype)]).reshape(
            spec.shape
        )
    return out


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
        self.assertIsNone(p.write_to_buffer(_trajectory(3)))

    def test_returns_none_without_a_buffer(self):
        p = _make_producer()
        p._ucxx_buffer = None
        self.assertIsNone(p.write_to_buffer(_trajectory(3)))

    def test_write_failure_is_contained(self):
        # A failed write must not propagate into the rollout loop -- the
        # producer degrades to "no UCXX metadata" and the caller falls back.
        p = _make_producer(buffer=_FakeBuffer(raises=RuntimeError("shm full")))
        self.assertIsNone(p.write_to_buffer(_trajectory(3)))


class TestCpuFallbackPath(unittest.TestCase):
    """No GPU tensors -> the per-tensor path, which hands a dict to
    ``UCXXBuffer.write`` rather than raw bytes."""

    def test_writes_dict_and_pads_varlen_fields(self):
        buf = _FakeBuffer()
        p = _make_producer(buffer=buf)
        out = p.write_to_buffer(_trajectory(4))  # 4 of MAX_STEPS=6

        self.assertIsNotNone(out)
        self.assertEqual(len(buf.dict_writes), 1)
        self.assertEqual(buf.raw_writes, [])  # not the coalesced path
        written = buf.dict_writes[0]
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
        p.write_to_buffer(_trajectory(4))
        ep = buf.dict_writes[0][EPISODE_LENGTH]
        self.assertEqual(ep.dtype, np.dtype("int64"))
        self.assertEqual(int(ep[0]), 4)


class TestReturnedMetadata(unittest.TestCase):
    """The dict the producer returns IS the reference the consumer resolves,
    so its shape is part of the protocol."""

    def test_metadata_fields(self):
        buf = _FakeBuffer()
        p = _make_producer(buffer=buf)
        out = p.write_to_buffer(_trajectory(4))

        self.assertTrue(out["_ucxx"])
        # The consumer's kill-switch: absent would silently mean "enabled".
        self.assertTrue(out["_ucxx_enabled"])
        self.assertEqual(out["_worker_ip"], "10.0.0.7")
        self.assertEqual(out["_ucxx_port"], 7000)
        self.assertEqual(out["_ports"], [7000, 7001])
        self.assertEqual(out["_slot"], 4)  # slot id from the fallback write()
        self.assertEqual(out["_replica_id"], "rollout-ucxx-0")
        self.assertEqual(out["_buffer_handle"], {"name": "fake-shm"})
        self.assertEqual(out[EPISODE_LENGTH], 4)
        self.assertEqual(out[REWARDS], [0.0, 1.0, 2.0, 3.0])

    def test_cache_key_fields_are_present(self):
        # The consumer keys its prefetch cache on "ip:port:slot"; all three
        # must survive the round trip or every episode is a cache miss.
        out = _make_producer().write_to_buffer(_trajectory(3))
        for key in ("_worker_ip", "_ucxx_port", "_slot"):
            self.assertIn(key, out)
            self.assertIsNotNone(out[key])


@requires_cuda
class TestCoalescedGpuPackPath(unittest.TestCase):
    """The fast path builds the flat payload itself, so it is the one that
    must satisfy the wire contract byte-for-byte."""

    def _write_and_unpack(self, traj):
        buf = _FakeBuffer()
        p = _make_producer(buffer=buf)
        out = p.write_to_buffer(traj)
        self.assertIsNotNone(out, "write_to_buffer failed")
        self.assertEqual(len(buf.raw_writes), 1, "did not take the coalesced path")
        return _unpack(buf.raw_writes[0], p._ucxx_schema), out

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
        p.write_to_buffer(_trajectory(MAX_STEPS, device="cuda"))
        self.assertEqual(len(buf.raw_writes[0]), p._ucxx_entry_data_size)

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
        p.write_to_buffer(traj)
        got = _unpack(buf.raw_writes[0], p._ucxx_schema)
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
        out = p.write_to_buffer(traj)
        self.assertIsNotNone(out, "float64 observations were rejected outright")
        got = _unpack(buf.raw_writes[0], p._ucxx_schema)
        torch.testing.assert_close(got[OBSERVATIONS], traj[OBSERVATIONS].float().cpu())


class TestCleanup(unittest.TestCase):
    def test_cleanup_is_safe_without_a_buffer(self):
        p = _make_producer()
        p._ucxx_buffer = None
        p.cleanup_ucxx()  # must not raise
        self.assertFalse(p._ucxx_enabled)


if __name__ == "__main__":
    unittest.main()
