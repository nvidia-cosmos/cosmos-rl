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

"""The trajectory payload format: field names, flat byte schema, and layout.

A trajectory is defined by the **data packer** layer
(:class:`~cosmos_rl.dispatcher.data.packer.tensor_data_packer.TensorDataPacker`),
not by any transport -- transports only move the bytes.  But the packer sits
above ``utils`` in the layering (it imports ``dispatcher`` and ``policy.config``),
so the transports cannot import it.  Historically each side worked around that
by re-declaring the format: the field names existed in three places and the
episode-length resolution in three more, with only a comment ("Mirrored from
tensor_data_packer", "Matches the UCXX rollout mixin's schema") keeping them in
step.

This module is the single definition, placed low enough that both the packer
and every transport can depend on it.  Ownership is unchanged -- the packer
still defines what a trajectory *means*; it just no longer redeclares it.

The byte layout is a wire contract: the producer's pack and the consumer's
unpack must agree, and a run must be able to switch transports without
re-plumbing the payload.  Spec ORDER is therefore load-bearing.

Deliberately dependency-light (numpy only).  In particular it does NOT import
torch, so the transport-agnostic scheduler can use it without pulling in a GPU
stack; :func:`episode_length` duck-types tensor-likes via ``.item()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "ACTIONS",
    "EPISODE_LENGTH",
    "OBSERVATIONS",
    "REWARDS",
    "TERMINATED",
    "TRAJECTORY_KEYS",
    "TRUNCATED",
    "VARLEN_FIELDS",
    "TensorSpec",
    "build_trajectory_schema",
    "deserialize_schema",
    "episode_length",
    "schema_layout",
    "serialize_schema",
]


# Canonical trajectory field names.
OBSERVATIONS = "observations"
ACTIONS = "actions"
REWARDS = "rewards"
TERMINATED = "terminated"
TRUNCATED = "truncated"
EPISODE_LENGTH = "episode_length"

#: Every field of a trajectory, in schema order.
TRAJECTORY_KEYS = (
    OBSERVATIONS,
    ACTIONS,
    REWARDS,
    TERMINATED,
    TRUNCATED,
    EPISODE_LENGTH,
)

#: Fields whose leading dimension is the (variable) episode length, and which
#: are therefore zero-padded up to the schema's ``max_steps`` when packed.
#: ``EPISODE_LENGTH`` is excluded -- it carries the true length.
VARLEN_FIELDS = (OBSERVATIONS, ACTIONS, REWARDS, TERMINATED, TRUNCATED)


@dataclass
class TensorSpec:
    """Fixed-shape tensor descriptor.

    Attributes:
        shape: Tuple of dimensions (e.g. ``(max_steps, obs_dim)``).
        dtype: Element type, accepted as a Python type (``np.float32``) or a
            :class:`numpy.dtype` instance.  Always normalized to
            :class:`numpy.dtype` after construction.
        name: Identifier used in flat schemas (e.g. ``"observations"``).
            Required when the spec participates in a schema.
    """

    shape: Tuple[int, ...]
    dtype: Union[type, np.dtype]
    name: str = ""

    def __post_init__(self) -> None:
        # ``dataclass`` is permissive about dtype; normalize once so downstream
        # code can rely on ``self.dtype.itemsize`` etc.
        self.dtype = np.dtype(self.dtype)

    @property
    def nbytes(self) -> int:
        """Byte size of one tensor matching this spec."""
        return int(np.prod(self.shape)) * self.dtype.itemsize

    def contains(self, tensor: np.ndarray) -> bool:
        """Return True if ``tensor`` has matching shape and dtype."""
        if tensor.shape != self.shape:
            return False
        if tensor.dtype != self.dtype:
            return False
        return True


def build_trajectory_schema(dims: Dict[str, int]) -> List[TensorSpec]:
    """Build the fixed-shape trajectory schema from ``{max_steps, obs_dim,
    action_dim}``.

    Spec ORDER is part of the wire contract -- :func:`schema_layout` derives
    byte offsets from it, so reordering silently breaks every peer.
    """
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


def schema_layout(schema: Sequence[TensorSpec]) -> Tuple[Dict[str, int], int]:
    """Return ``(name -> byte offset, total entry size)`` for ``schema``."""
    offsets: Dict[str, int] = {}
    offset = 0
    for spec in schema:
        offsets[spec.name] = offset
        offset += spec.nbytes
    return offsets, offset


def serialize_schema(schema: Sequence[TensorSpec]) -> List[Dict[str, Any]]:
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


def episode_length(
    trajectory: Dict[str, Any],
    schema: Optional[Sequence[TensorSpec]] = None,
    *,
    default: Optional[int] = None,
) -> int:
    """Resolve a trajectory's true (unpadded) episode length.

    Resolution order:

    1. an explicit ``episode_length`` field (tensor-like via ``.item()``, else
       ``int()``);
    2. the leading dimension of ``observations``;
    3. ``default`` when provided (the producer's configured ``max_steps``);
    4. the schema's padded ``observations`` extent;
    5. ``0``.

    Steps 3 and 4 are the same number in practice -- the schema is built from
    that same ``max_steps`` -- which is why the producers' historically
    different fallbacks converge here.
    """
    ep = trajectory.get(EPISODE_LENGTH)
    if ep is not None:
        item = getattr(ep, "item", None)
        return int(item()) if callable(item) else int(ep)

    obs = trajectory.get(OBSERVATIONS)
    if obs is not None:
        shape = getattr(obs, "shape", None)
        if shape is not None:
            return int(shape[0])
        try:
            return len(obs)
        except TypeError:  # pragma: no cover - unsized, fall through
            pass

    if default is not None:
        return int(default)

    for spec in schema or ():
        if spec.name == OBSERVATIONS:
            return int(spec.shape[0])
    return 0
