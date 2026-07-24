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

"""Flat trajectory schema helpers shared by the NCCL sender + receiver.

Reuses :class:`~cosmos_rl.utils.payload_transport.ucxx.tensor_spec.TensorSpec`
(the transport-agnostic descriptor) so the on-wire byte layout is defined
once and both the producer (pack) and consumer (unpack) agree on offsets,
sizes, and the ``(name, shape, dtype)`` triples exchanged as JSON in the
dict metadata.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from cosmos_rl.utils.payload_transport.ucxx.tensor_spec import TensorSpec

__all__ = [
    "build_trajectory_schema",
    "schema_layout",
    "serialize_schema",
    "deserialize_schema",
]

# Canonical trajectory field names (mirrored from the tensor data packer so
# this module imports standalone).
OBSERVATIONS = "observations"
ACTIONS = "actions"
REWARDS = "rewards"
TERMINATED = "terminated"
TRUNCATED = "truncated"
EPISODE_LENGTH = "episode_length"


def build_trajectory_schema(dims: Dict[str, int]) -> List[TensorSpec]:
    """Build the fixed-shape trajectory schema from ``{max_steps, obs_dim,
    action_dim}``.  Matches the UCXX rollout mixin's schema so a run can
    switch transports without re-plumbing the payload layout."""
    max_steps = int(dims["max_steps"])
    obs_dim = int(dims["obs_dim"])
    action_dim = int(dims["action_dim"])
    return [
        TensorSpec(name=OBSERVATIONS, shape=(max_steps, obs_dim), dtype=np.float32),
        TensorSpec(name=ACTIONS, shape=(max_steps, action_dim), dtype=np.float32),
        TensorSpec(name=REWARDS, shape=(max_steps,), dtype=np.float32),
        TensorSpec(name=TERMINATED, shape=(max_steps,), dtype=np.bool_),
        TensorSpec(name=TRUNCATED, shape=(max_steps,), dtype=np.bool_),
        TensorSpec(name=EPISODE_LENGTH, shape=(1,), dtype=np.int64),
    ]


def schema_layout(schema: List[TensorSpec]) -> Tuple[Dict[str, int], int]:
    """Return ``(name -> byte offset, total entry size)`` for ``schema``."""
    offsets: Dict[str, int] = {}
    offset = 0
    for spec in schema:
        offsets[spec.name] = offset
        offset += spec.nbytes
    return offsets, offset


def serialize_schema(schema: List[TensorSpec]) -> List[Dict[str, Any]]:
    """JSON-friendly ``[{name, shape, dtype}]`` for the dict metadata."""
    return [
        {"name": s.name, "shape": list(s.shape), "dtype": np.dtype(s.dtype).str}
        for s in schema
    ]


def deserialize_schema(raw: List[Dict[str, Any]]) -> List[TensorSpec]:
    """Inverse of :func:`serialize_schema`."""
    return [
        TensorSpec(
            name=s["name"],
            shape=tuple(s["shape"]),
            dtype=np.dtype(s["dtype"]),
        )
        for s in raw
    ]
