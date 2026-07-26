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

"""Tests for the shared trajectory format (``cosmos_rl.utils.trajectory``).

This module is the single definition of the on-the-wire payload layout, which
the NCCL and UCXX producers previously built independently.  The schema-order
test below is the regression guard that replaced that hand-maintained
agreement: offsets are derived from spec order, so a reorder silently breaks
every peer.
"""

import unittest

import numpy as np

from cosmos_rl.utils.trajectory import (
    ACTIONS,
    EPISODE_LENGTH,
    OBSERVATIONS,
    REWARDS,
    TERMINATED,
    TRUNCATED,
    build_trajectory_schema,
    deserialize_schema,
    episode_length,
    schema_layout,
    serialize_schema,
)

DIMS = {"max_steps": 8, "obs_dim": 4, "action_dim": 2}


class TestWireContract(unittest.TestCase):
    """Spec order and sizes ARE the wire format -- pin them."""

    def test_schema_order_is_stable(self):
        # schema_layout derives byte offsets from this order; reordering
        # silently desyncs producer pack from consumer unpack.
        self.assertEqual(
            [s.name for s in build_trajectory_schema(DIMS)],
            [OBSERVATIONS, ACTIONS, REWARDS, TERMINATED, TRUNCATED, EPISODE_LENGTH],
        )

    def test_schema_shapes_and_dtypes(self):
        by_name = {s.name: s for s in build_trajectory_schema(DIMS)}
        self.assertEqual(by_name[OBSERVATIONS].shape, (8, 4))
        self.assertEqual(by_name[ACTIONS].shape, (8, 2))
        self.assertEqual(by_name[REWARDS].shape, (8,))
        self.assertEqual(by_name[EPISODE_LENGTH].shape, (1,))
        self.assertEqual(by_name[OBSERVATIONS].dtype, np.dtype(np.float32))
        self.assertEqual(by_name[TERMINATED].dtype, np.dtype(np.bool_))
        self.assertEqual(by_name[TRUNCATED].dtype, np.dtype(np.bool_))
        self.assertEqual(by_name[EPISODE_LENGTH].dtype, np.dtype(np.int64))

    def test_layout_is_contiguous_and_ordered(self):
        schema = build_trajectory_schema(DIMS)
        offsets, entry_size = schema_layout(schema)
        running = 0
        for spec in schema:
            self.assertEqual(offsets[spec.name], running)
            running += spec.nbytes
        self.assertEqual(entry_size, running)
        # 8*4*4 + 8*2*4 + 8*4 + 8 + 8 + 8
        self.assertEqual(entry_size, 128 + 64 + 32 + 8 + 8 + 8)

    def test_serialize_round_trip(self):
        schema = build_trajectory_schema(DIMS)
        restored = deserialize_schema(serialize_schema(schema))
        self.assertEqual([s.name for s in restored], [s.name for s in schema])
        self.assertEqual([s.shape for s in restored], [s.shape for s in schema])
        self.assertEqual([s.dtype for s in restored], [s.dtype for s in schema])
        self.assertEqual(schema_layout(restored), schema_layout(schema))


class _FakeTensor:
    """Stands in for a torch tensor without importing torch."""

    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class TestEpisodeLength(unittest.TestCase):
    """One resolver replacing three near-copies (NCCL producer, UCXX
    producer, TensorDataPacker), so their fallbacks must all still hold."""

    def test_explicit_int(self):
        self.assertEqual(episode_length({EPISODE_LENGTH: 5}), 5)

    def test_explicit_tensor_like(self):
        # torch tensors are duck-typed via .item() so this module stays
        # torch-free.
        self.assertEqual(episode_length({EPISODE_LENGTH: _FakeTensor(7)}), 7)

    def test_falls_back_to_observation_rows(self):
        obs = np.zeros((6, 4), dtype=np.float32)
        self.assertEqual(episode_length({OBSERVATIONS: obs}), 6)

    def test_falls_back_to_len_for_unshaped_observations(self):
        self.assertEqual(episode_length({OBSERVATIONS: [[0.0], [1.0], [2.0]]}), 3)

    def test_default_precedes_schema(self):
        # The UCXX producer's historical fallback: its configured max_steps.
        schema = build_trajectory_schema(DIMS)
        self.assertEqual(episode_length({}, schema, default=99), 99)

    def test_schema_padded_extent_when_no_default(self):
        # The NCCL producer's historical fallback: the schema's padded max.
        schema = build_trajectory_schema(DIMS)
        self.assertEqual(episode_length({}, schema), DIMS["max_steps"])

    def test_zero_when_nothing_resolves(self):
        self.assertEqual(episode_length({}), 0)

    def test_explicit_zero_is_honoured_not_treated_as_missing(self):
        schema = build_trajectory_schema(DIMS)
        self.assertEqual(episode_length({EPISODE_LENGTH: 0}, schema, default=99), 0)


if __name__ == "__main__":
    unittest.main()
