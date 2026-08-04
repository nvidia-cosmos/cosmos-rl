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

"""Pack a trajectory into the flat, schema-defined payload buffer.

Both producers write the same on-wire format -- a single contiguous ``uint8``
buffer whose field offsets come from :func:`~cosmos_rl.utils.trajectory.
schema_layout` -- and the consumer reads it back by schema alone, slicing at
``spec.nbytes`` and reinterpreting with ``spec.dtype``.  Keeping one
implementation of that loop is what makes the two ends agree.

Lives here rather than in :mod:`cosmos_rl.utils.trajectory` because it needs
torch, and that module is deliberately numpy-only so the transport-agnostic
scheduler can import it without pulling in a GPU stack.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Set, Tuple

import numpy as np
import torch

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.trajectory import EPISODE_LENGTH, VARLEN_FIELDS, TensorSpec

__all__ = ["NP_TO_TORCH", "pack_trajectory_into", "torch_dtype_for"]


NP_TO_TORCH = {
    np.dtype("float32"): torch.float32,
    np.dtype("float64"): torch.float64,
    np.dtype("float16"): torch.float16,
    np.dtype("int64"): torch.int64,
    np.dtype("int32"): torch.int32,
    np.dtype("int16"): torch.int16,
    np.dtype("int8"): torch.int8,
    np.dtype("uint8"): torch.uint8,
    np.dtype("bool"): torch.bool,
}

#: Fields already warned about, so a narrowing cast logs once per process
#: rather than once per trajectory.
_WARNED_NARROWING: Set[Tuple[str, str, str]] = set()


def torch_dtype_for(np_dtype: Any) -> torch.dtype:
    """Torch dtype for a schema spec's numpy dtype."""
    td = NP_TO_TORCH.get(np.dtype(np_dtype))
    if td is None:
        raise ValueError(f"unsupported schema dtype {np_dtype}")
    return td


def _coerce(tensor: torch.Tensor, spec: TensorSpec) -> torch.Tensor:
    """Cast ``tensor`` to the spec dtype, warning once if that narrows.

    The schema IS the wire format: the consumer slices ``spec.nbytes`` and
    views the result as ``spec.dtype``, so a tensor packed in its source dtype
    would write the wrong number of bytes and run into the neighbouring field.
    The cast is therefore mandatory -- but a narrowing one (a gym env's float64
    observations against a float32 schema) silently loses precision, so say so.
    """
    target = torch_dtype_for(spec.dtype)
    if tensor.dtype != target:
        src = np.dtype(str(tensor.dtype).removeprefix("torch."))
        if src.itemsize > np.dtype(spec.dtype).itemsize:
            key = (spec.name, str(tensor.dtype), str(target))
            if key not in _WARNED_NARROWING:
                _WARNED_NARROWING.add(key)
                logger.warning(
                    "[pack] '%s' supplied as %s but the payload schema declares "
                    "%s; narrowing the cast loses precision. Emit the field in "
                    "the schema dtype to avoid this.",
                    spec.name,
                    tensor.dtype,
                    target,
                )
        tensor = tensor.to(target)
    return tensor


def pack_trajectory_into(
    flat: torch.Tensor,
    trajectory: Dict[str, Any],
    schema: Sequence[TensorSpec],
    offsets: Dict[str, int],
    ep_len: int,
    device: Any = None,
) -> None:
    """Write ``trajectory`` into the preallocated ``flat`` uint8 buffer.

    ``flat`` must already be zeroed and sized to the schema's entry size --
    absent optional fields are left as zeros.

    ``episode_length`` is checked BEFORE the missing-field skip and is always
    written from the resolved ``ep_len``: the producer knows the true length
    even when the trajectory dict omits the key, and leaving that slot zero
    makes the consumer truncate the episode to nothing.
    """
    for spec in schema:
        raw = trajectory.get(spec.name)
        if spec.name == EPISODE_LENGTH:
            tensor = torch.tensor([ep_len], dtype=torch.int64, device=device)
        elif raw is None:
            continue
        else:
            tensor = raw if isinstance(raw, torch.Tensor) else torch.as_tensor(raw)
            if device is not None:
                tensor = tensor.to(device)
            tensor = _coerce(tensor, spec)
            if spec.name in VARLEN_FIELDS and tensor.shape[0] < spec.shape[0]:
                padded = torch.zeros(
                    spec.shape, dtype=tensor.dtype, device=tensor.device
                )
                padded[: tensor.shape[0]] = tensor
                tensor = padded
        tensor = tensor.reshape(spec.shape).contiguous()
        chunk = tensor.view(torch.uint8).reshape(-1)
        off = offsets[spec.name]
        flat[off : off + chunk.numel()] = chunk
