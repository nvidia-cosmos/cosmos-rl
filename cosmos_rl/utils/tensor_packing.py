# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small utilities for deterministic byte packing of ordered tensors."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

import torch


def canonical_byte_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Canonical storage, preserving shape; flatten *before* viewing as bytes.

    Contiguity ignores singleton strides. Resolve lazy view bits as well so
    wire bytes represent logical values, not their underlying storage.
    """
    tensor = tensor.resolve_conj().resolve_neg()
    if tensor.is_contiguous() and (tensor.ndim == 0 or tensor.stride(-1) == 1):
        return tensor
    canonical = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
    canonical.copy_(tensor)
    return canonical


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def iter_tensor_byte_buckets(
    tensors: Iterable[torch.Tensor], bucket_size_bytes: int
) -> Iterator[list[torch.Tensor]]:
    """Yield ordered buckets, leaving an oversized tensor in its own bucket."""
    if bucket_size_bytes <= 0:
        raise ValueError("tensor packing bucket size must be positive")
    pending = []
    pending_bytes = 0
    for tensor in tensors:
        nbytes = tensor_nbytes(tensor)
        if pending and pending_bytes + nbytes > bucket_size_bytes:
            yield pending
            pending = []
            pending_bytes = 0
        pending.append(tensor)
        pending_bytes += nbytes
        if pending_bytes >= bucket_size_bytes:
            yield pending
            pending = []
            pending_bytes = 0
    if pending:
        yield pending


def packed_nbytes(tensors: Sequence[torch.Tensor]) -> int:
    return sum(tensor_nbytes(tensor) for tensor in tensors)


def pack_tensors_into_buffer(
    tensors: Sequence[torch.Tensor], buffer: torch.Tensor
) -> torch.Tensor:
    """Copy contiguous tensors into the leading bytes of ``buffer``."""
    required = packed_nbytes(tensors)
    if buffer.dtype != torch.uint8 or buffer.numel() < required:
        raise ValueError("byte buffer is too small or has the wrong dtype")
    payload = buffer[:required]
    offset = 0
    for tensor in tensors:
        if not tensor.is_contiguous():
            raise ValueError("tensor packing requires contiguous tensors")
        tensor_bytes = canonical_byte_tensor(tensor).reshape(-1).view(torch.uint8)
        end = offset + tensor_bytes.numel()
        payload[offset:end].copy_(tensor_bytes)
        offset = end
    return payload


def unpack_tensors_from_buffer(
    buffer: torch.Tensor, tensors: Sequence[torch.Tensor]
) -> None:
    """Copy leading bytes from ``buffer`` into contiguous tensors in order."""
    required = packed_nbytes(tensors)
    if buffer.dtype != torch.uint8 or buffer.numel() < required:
        raise ValueError("byte buffer is too small or has the wrong dtype")
    offset = 0
    for tensor in tensors:
        if not tensor.is_contiguous():
            raise ValueError("tensor unpacking requires contiguous tensors")
        canonical = canonical_byte_tensor(tensor)
        tensor_bytes = canonical.reshape(-1).view(torch.uint8)
        end = offset + tensor_bytes.numel()
        tensor_bytes.copy_(buffer[offset:end])
        if canonical is not tensor:
            tensor.copy_(canonical)
        offset = end
