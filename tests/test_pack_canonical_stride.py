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

"""The payload packer must accept every legal layout of a supported tensor.

The layout that motivated these tests is an int64 column view --
``sampled_modes[:, idx]`` with ``shape=(1,)`` and ``stride=(8,)``.  PyTorch
calls it contiguous (size-1 dimensions may carry any stride), so the packer's
``.contiguous()`` was a no-op and the following ``view(torch.uint8)`` raised
"self.stride(-1) must be 1 to view Long as Byte".  The producer caught that,
returned ``None``, and every affected payload silently fell back to disk
instead of the NCCL fast path.
"""

import threading
import unittest
from unittest import mock

import numpy as np
import torch

from cosmos_rl.utils.payload_transport.nccl.buffer_registry import SendBufferRegistry
from cosmos_rl.utils.payload_transport.nccl.mixins import NCCLRolloutMixin
from cosmos_rl.utils.payload_transport.pack import (
    _canonical_for_byte_view,
    pack_trajectory_into,
    torch_dtype_for,
)
from cosmos_rl.utils.trajectory import (
    TensorSpec,
    build_trajectory_schema,
    schema_layout,
)


def _singleton_column(dtype=torch.int64):
    """``shape=(1,)``, non-unit last stride, yet ``is_contiguous()`` is True."""
    row = torch.arange(8, dtype=dtype).reshape(1, 8)
    return row[:, 0]


def _pack_and_recover(spec, tensor):
    """Pack one field through the real schema path and read it back."""
    schema = [spec]
    offsets, entry_size = schema_layout(schema)
    flat = torch.zeros(entry_size, dtype=torch.uint8)
    pack_trajectory_into(flat, {spec.name: tensor}, schema, offsets, ep_len=1)
    off = offsets[spec.name]
    return (
        flat[off : off + spec.nbytes]
        .view(torch_dtype_for(spec.dtype))
        .reshape(spec.shape)
    )


class TestCanonicalForByteView(unittest.TestCase):
    def test_reproducer_layout_is_the_bug(self):
        """Guard the premise: this layout really does defeat ``.contiguous()``."""
        value = _singleton_column()
        self.assertEqual(tuple(value.shape), (1,))
        self.assertEqual(value.stride(), (8,))
        self.assertTrue(value.is_contiguous())
        with self.assertRaises(RuntimeError):
            value.contiguous().view(torch.uint8)

    def test_canonicalized_tensor_accepts_the_byte_view(self):
        canonical = _canonical_for_byte_view(_singleton_column())
        self.assertEqual(canonical.stride(), (1,))
        self.assertEqual(canonical.view(torch.uint8).numel(), 8)
        self.assertTrue(torch.equal(canonical, _singleton_column()))

    def test_canonical_input_is_not_copied(self):
        """An already-canonical tensor keeps the existing zero-copy fast path."""
        tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        self.assertIs(_canonical_for_byte_view(tensor), tensor)

    def test_zero_dim_and_empty_tensors(self):
        scalar = torch.tensor(5, dtype=torch.int64)
        self.assertIs(_canonical_for_byte_view(scalar), scalar)
        empty = torch.empty((0,), dtype=torch.int64)
        self.assertIs(_canonical_for_byte_view(empty), empty)

    def test_non_contiguous_transpose_is_materialized(self):
        transposed = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
        canonical = _canonical_for_byte_view(transposed)
        self.assertTrue(canonical.is_contiguous())
        self.assertEqual(canonical.stride(-1), 1)
        self.assertTrue(torch.equal(canonical, transposed))


class TestPackSingletonStride(unittest.TestCase):
    def test_int64_singleton_column_round_trips(self):
        spec = TensorSpec(name="sampled_mode", shape=(1,), dtype=np.int64)
        value = _singleton_column()
        recovered = _pack_and_recover(spec, value)
        self.assertEqual(recovered.dtype, torch.int64)
        self.assertEqual(tuple(recovered.shape), (1,))
        self.assertTrue(torch.equal(recovered, value))

    def test_size_one_dimension_in_each_position(self):
        base = torch.arange(2 * 3 * 4, dtype=torch.int64).reshape(2, 3, 4)
        cases = {
            "leading": (base[:1], (1, 3, 4)),
            "middle": (base[:, :1], (2, 1, 4)),
            "trailing": (base[:, :, :1], (2, 3, 1)),
        }
        for name, (view, shape) in cases.items():
            with self.subTest(position=name):
                spec = TensorSpec(name=f"field_{name}", shape=shape, dtype=np.int64)
                recovered = _pack_and_recover(spec, view)
                self.assertTrue(torch.equal(recovered, view))

    def test_other_multibyte_dtypes(self):
        for np_dtype, torch_dtype in (
            (np.float32, torch.float32),
            (np.float64, torch.float64),
            (np.int32, torch.int32),
            (np.int16, torch.int16),
        ):
            with self.subTest(dtype=np_dtype):
                spec = TensorSpec(name="value", shape=(1,), dtype=np_dtype)
                value = _singleton_column(dtype=torch_dtype)
                self.assertNotEqual(value.stride(-1), 1)
                recovered = _pack_and_recover(spec, value)
                self.assertTrue(torch.equal(recovered, value))

    def test_single_byte_dtype_still_round_trips(self):
        spec = TensorSpec(name="flag", shape=(1,), dtype=np.uint8)
        value = torch.arange(8, dtype=torch.uint8).reshape(1, 8)[:, 0]
        self.assertTrue(torch.equal(_pack_and_recover(spec, value), value))

    @unittest.skipUnless(torch.cuda.is_available(), "requires a GPU")
    def test_gpu_singleton_column_round_trips(self):
        spec = TensorSpec(name="sampled_mode", shape=(1,), dtype=np.int64)
        value = _singleton_column().cuda()
        schema = [spec]
        offsets, entry_size = schema_layout(schema)
        flat = torch.zeros(entry_size, dtype=torch.uint8, device="cuda")
        pack_trajectory_into(
            flat, {spec.name: value}, schema, offsets, ep_len=1, device=value.device
        )
        recovered = flat[: spec.nbytes].view(torch.int64).reshape(spec.shape)
        self.assertTrue(torch.equal(recovered.cpu(), value.cpu()))


class TestPackErrorNamesTheField(unittest.TestCase):
    def test_failure_message_carries_field_identity(self):
        """A layout the packer cannot handle must say which field it was."""
        spec = TensorSpec(name="sampled_mode", shape=(1,), dtype=np.int64)
        schema = [spec]
        offsets, entry_size = schema_layout(schema)
        flat = torch.zeros(entry_size, dtype=torch.uint8)

        # Force the byte view to fail after canonicalization, standing in for
        # any future layout the canonical copy cannot repair.
        def _passthrough(tensor):
            return tensor

        with mock.patch(
            "cosmos_rl.utils.payload_transport.pack._canonical_for_byte_view",
            _passthrough,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                pack_trajectory_into(
                    flat,
                    {spec.name: _singleton_column()},
                    schema,
                    offsets,
                    ep_len=1,
                )
        message = str(ctx.exception)
        self.assertIn("sampled_mode", message)
        self.assertIn("torch.int64", message)
        self.assertIn("stride=(8,)", message)
        self.assertIn("shape=(1,)", message)


class TestProducerKeepsNcclPath(unittest.TestCase):
    """End-to-end producer check: no silent fallback to disk."""

    def _producer(self):
        p = NCCLRolloutMixin()
        p._nccl_enabled = True
        p._nccl_replica_id = "rollout-test-0"
        p._nccl_rollout_idx = 0
        p._nccl_sender_rank = 0
        p._nccl_device = None  # CPU tensors
        p._nccl_schema = build_trajectory_schema(
            {"max_steps": 4, "obs_dim": 2, "action_dim": 2}
        )
        # A producer-supplied extra field is what carries the offending layout.
        p._nccl_schema.append(
            TensorSpec(name="sampled_mode", shape=(1,), dtype=np.int64)
        )
        p._nccl_offsets, p._nccl_entry_size = schema_layout(p._nccl_schema)
        p._nccl_registry = SendBufferRegistry(capacity=4, on_free=p._on_buffer_free)
        p._nccl_streams = None
        p._nccl_send_lock = threading.Lock()
        return p

    def test_write_to_buffer_returns_a_reference(self):
        p = self._producer()
        meta = p.write_to_buffer(
            {
                "observations": torch.zeros(4, 2),
                "actions": torch.zeros(4, 2),
                "rewards": torch.zeros(4),
                "episode_length": 4,
                "sampled_mode": _singleton_column(),
            }
        )
        self.assertIsNotNone(meta, "packing must not fall back to the disk path")
        self.assertTrue(meta["_nccl"])
        self.assertIn(meta["_transfer_id"], p._nccl_registry)


if __name__ == "__main__":
    unittest.main()
