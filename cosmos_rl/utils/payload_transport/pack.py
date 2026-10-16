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

from typing import Any, Dict, Iterator, Sequence, Set, Tuple

import numpy as np
import torch

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.tensor_packing import canonical_byte_tensor
from cosmos_rl.utils.trajectory import (
    EPISODE_LENGTH,
    VARLEN_FIELDS,
    TensorSpec,
    schema_layout,
    validate_episode_length,
)

__all__ = [
    "NP_TO_TORCH",
    "normalize_trajectory",
    "pack_trajectory_into",
    "torch_dtype_for",
]


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
        if tensor.element_size() > np.dtype(spec.dtype).itemsize:
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


def _layout_error(spec: TensorSpec, tensor: torch.Tensor, exc: Exception) -> str:
    """Message naming the offending field, so a pack failure is actionable.

    ``write_to_buffer`` logs only the exception text before falling back to
    the plain trajectory; without the field identity that log says a payload
    failed but not which one.
    """
    return (
        f"[pack] field '{spec.name}' cannot be viewed as bytes: "
        f"dtype={tensor.dtype} shape={tuple(tensor.shape)} "
        f"stride={tuple(tensor.stride())} device={tensor.device} "
        f"schema_dtype={spec.dtype} schema_shape={tuple(spec.shape)}: {exc}"
    )


def _canonical_for_byte_view(tensor: torch.Tensor) -> torch.Tensor:
    """Return ``tensor`` in storage that ``view(torch.uint8)`` accepts.

    ``.contiguous()`` alone is NOT enough.  PyTorch's contiguity rules ignore
    the stride of a size-1 dimension, so an int64 column view such as
    ``sampled_modes[:, idx]`` -- ``shape=(1,), stride=(8,)`` -- reports
    ``is_contiguous() is True`` and ``.contiguous()`` returns it untouched.
    ``view(dtype)`` applies the stricter byte-layout rule and requires
    ``stride(-1) == 1`` when the element size changes, so that tensor raises
    "self.stride(-1) must be 1 to view Long as Byte".

    Tensors that already satisfy the byte-view rule are returned as-is, so the
    common case still packs without an extra copy.
    """
    return canonical_byte_tensor(tensor)


def _normalized_fields(
    trajectory: Dict[str, Any],
    schema: Sequence[TensorSpec],
    ep_len: int,
    device: Any = None,
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Normalize every field before publication, identically on CPU and GPU.

    Missing optional fields are zero. Present fields must match their schema;
    only sequence leading dimensions may be shorter and receive zero padding.
    Source values are detached, cast, and canonicalized without mutation.
    """
    schema_layout(schema)  # Validate the metadata wire contract too.
    capacities = [s.shape[0] for s in schema if s.name in VARLEN_FIELDS and s.shape]
    ep_len = validate_episode_length(ep_len, capacities)
    for spec in schema:
        raw = trajectory.get(spec.name)
        if spec.name == EPISODE_LENGTH:
            tensor = torch.full(spec.shape, ep_len, dtype=torch.int64, device=device)
        elif raw is None:
            tensor = torch.zeros(
                spec.shape, dtype=torch_dtype_for(spec.dtype), device=device
            )
        else:
            if isinstance(raw, np.ndarray) and any(
                stride < 0 for stride in raw.strides
            ):
                raw = raw.copy()
            tensor = (
                raw.detach() if isinstance(raw, torch.Tensor) else torch.as_tensor(raw)
            )
            if device is not None:
                tensor = tensor.to(device)
            tensor = _coerce(tensor, spec)
            if spec.name in VARLEN_FIELDS:
                if (
                    not spec.shape
                    or tensor.ndim != len(spec.shape)
                    or tuple(tensor.shape[1:]) != tuple(spec.shape[1:])
                    or not ep_len <= tensor.shape[0] <= spec.shape[0]
                ):
                    raise ValueError(
                        f"[pack] field '{spec.name}' shape {tuple(tensor.shape)} "
                        f"does not match schema {spec.shape} and episode_length {ep_len}"
                    )
                if tensor.shape[0] < spec.shape[0]:
                    padded = torch.zeros(
                        spec.shape, dtype=tensor.dtype, device=tensor.device
                    )
                    padded[: tensor.shape[0]].copy_(tensor)
                    tensor = padded
            elif tuple(tensor.shape) != tuple(spec.shape):
                raise ValueError(
                    f"[pack] field '{spec.name}' shape {tuple(tensor.shape)} "
                    f"does not match schema {spec.shape}"
                )
        yield spec.name, _canonical_for_byte_view(tensor)


def normalize_trajectory(
    trajectory: Dict[str, Any],
    schema: Sequence[TensorSpec],
    ep_len: int,
    device: Any = None,
) -> Dict[str, torch.Tensor]:
    """Validate all fields before claiming a shared-memory slot."""
    return dict(_normalized_fields(trajectory, schema, ep_len, device))


def pack_trajectory_into(
    flat: torch.Tensor,
    trajectory: Dict[str, Any],
    schema: Sequence[TensorSpec],
    offsets: Dict[str, int],
    ep_len: int,
    device: Any = None,
) -> None:
    """Write ``trajectory`` into the preallocated ``flat`` uint8 buffer.

    ``flat`` must be sized to the schema's entry size. Absent optional fields
    are written as zeros, including when the destination is reused. The caller
    must not publish it until this function returns successfully.

    ``episode_length`` is checked BEFORE the missing-field skip and is always
    written from the resolved ``ep_len``: the producer knows the true length
    even when the trajectory dict omits the key, and leaving that slot zero
    makes the consumer truncate the episode to nothing.
    """
    # Stream normalization into private storage, retaining at most one field's
    # cast/padding temporary rather than another whole payload on the GPU.
    for spec, (_, tensor) in zip(
        schema, _normalized_fields(trajectory, schema, ep_len, device)
    ):
        try:
            chunk = tensor.reshape(-1).view(torch.uint8)
        except RuntimeError as e:  # pragma: no cover - defensive
            raise RuntimeError(_layout_error(spec, tensor, e)) from e
        off = offsets[spec.name]
        flat[off : off + chunk.numel()] = chunk
