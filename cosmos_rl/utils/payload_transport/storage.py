# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit producer ownership for transport-backed payload storage."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import threading
from types import MappingProxyType
from typing import Any, Callable

import torch

from cosmos_rl.utils.payload_transport.layout import TensorLayout


@dataclass(frozen=True)
class PayloadStorageRequirements:
    """Physical allocation contract for a backend-compatible borrowed slot."""

    total_nbytes: int
    payload_offset: int
    alignment: int
    device: torch.device
    dtype: torch.dtype = torch.uint8


class BorrowedPayloadLease:
    """Keep application-owned storage alive until transport use is proven done."""

    def __init__(self, *, owner: Any, on_release: Callable[[], None]) -> None:
        self._owner = owner
        self._on_release = on_release
        self.released = False
        self.quarantined = False
        self._lock = threading.Lock()

    def release(self) -> None:
        with self._lock:
            if self.released or self.quarantined:
                return
            self.released = True
            callback = self._on_release
            self._on_release = lambda: None
            self._owner = None
        callback()

    def quarantine(self) -> None:
        with self._lock:
            if self.released or self.quarantined:
                return
            # Retain both the owner and callback. A failed native enqueue may
            # still dereference this storage, so only process exit is safe.
            self.quarantined = True


class WritablePayload:
    """Typed views over one exclusively owned publication slot.

    The views are mutable until publication. Publication transfers ownership to
    the transport; abandoning an unpublished payload returns its slot.
    """

    def __init__(
        self,
        *,
        owner: Any,
        layout: TensorLayout,
        storage: torch.Tensor,
        tensors: Mapping[str, torch.Tensor],
        opaque: Any,
        on_abandon: Callable[[], None],
    ) -> None:
        self._owner = owner
        self._layout = layout
        self._storage = storage
        self._tensors = MappingProxyType(dict(tensors))
        self._opaque = opaque
        self._on_abandon = on_abandon
        self._field_names = frozenset(tensors)
        self._written_fields: set[str] = set()
        self._submitted = False
        self._closed = False
        self._lock = threading.Lock()

    @property
    def layout(self) -> TensorLayout:
        return self._layout

    @property
    def storage(self) -> torch.Tensor:
        return self._storage

    @property
    def tensors(self) -> Mapping[str, torch.Tensor]:
        return self._tensors

    @property
    def missing_fields(self) -> frozenset[str]:
        """Return fields the caller has not attested were fully overwritten."""
        with self._lock:
            return frozenset(self._field_names - self._written_fields)

    def mark_written(self, *names: str) -> None:
        """Attest that every byte of the named tensor views was overwritten.

        Direct producers mutate tensor views in place, which PyTorch cannot
        observe at the mapping boundary. Requiring an explicit attestation
        prevents an omitted field from silently publishing bytes left by a
        prior use of the pooled slot.
        """
        if not names:
            raise ValueError("mark_written requires at least one field name")
        unknown = set(names) - self._field_names
        if unknown:
            raise KeyError(f"unknown payload fields: {sorted(unknown)!r}")
        with self._lock:
            if self._closed or self._submitted:
                raise RuntimeError("payload storage is no longer writable")
            self._written_fields.update(names)

    def mark_all_written(self) -> None:
        """Attest that every field view was fully overwritten by the producer."""
        with self._lock:
            if self._closed or self._submitted:
                raise RuntimeError("payload storage is no longer writable")
            self._written_fields.update(self._field_names)

    def _submit(self, owner: Any) -> Any:
        with self._lock:
            if owner is not self._owner:
                raise ValueError("payload storage belongs to a different producer")
            if self._closed or self._submitted:
                raise RuntimeError("payload storage is no longer publishable")
            missing = self._field_names - self._written_fields
            if missing:
                preview = sorted(missing)[:8]
                suffix = "" if len(missing) <= len(preview) else ", ..."
                raise RuntimeError(
                    "payload fields were not marked fully written: "
                    f"{', '.join(preview)}{suffix}"
                )
            self._submitted = True
            return self._opaque

    def close(self) -> None:
        with self._lock:
            if self._closed or self._submitted:
                return
            # Keep abandonment retryable. Backend release callbacks are
            # idempotent, and a callback exception is not proof that storage
            # ownership was resolved.
            self._on_abandon()
            self._closed = True

    def __enter__(self) -> "WritablePayload":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
