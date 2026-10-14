# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable tensor layouts shared by payload transport backends."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
import threading
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
import torch

from cosmos_rl.utils.payload_transport.pack import NP_TO_TORCH
from cosmos_rl.utils.trajectory import TensorSpec


WIRE_VERSION = 2
DEFAULT_ALIGNMENT = 64
MAX_LAYOUT_BYTES = (1 << 63) - 1
MAX_LAYOUT_DESCRIPTOR_BYTES = 16 * 1024 * 1024
_LAYOUT_ID_PATTERN = re.compile(r"[0-9a-f]{32}\Z")


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def identity_key(value: str) -> int:
    """Return a stable unsigned 64-bit identity digest for wire checks."""
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def validate_layout_id(value: Any) -> str:
    """Return a canonical layout identity or reject untrusted metadata."""
    if not isinstance(value, str) or _LAYOUT_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("tensor layout identity must be 32 lowercase hex characters")
    return value


def serialize_layout_descriptor(layout: "TensorLayout") -> str:
    """Encode one bounded canonical descriptor for control-plane storage."""
    encoded = json.dumps(layout.descriptor(), sort_keys=True, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > MAX_LAYOUT_DESCRIPTOR_BYTES:
        raise ValueError("tensor layout descriptor is too large")
    return encoded


def deserialize_layout_descriptor(raw: Any) -> Mapping[str, Any]:
    """Decode a bounded control-plane descriptor without trusting its fields."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise ValueError("tensor layout descriptor must be JSON text")
    if len(raw.encode("utf-8")) > MAX_LAYOUT_DESCRIPTOR_BYTES:
        raise ValueError("tensor layout descriptor is too large")

    def _unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"tensor layout descriptor repeats JSON field {key!r}")
            result[key] = value
        return result

    def _reject_nonfinite(value):
        raise ValueError(
            f"tensor layout descriptor contains nonfinite JSON value {value!r}"
        )

    descriptor = json.loads(
        raw,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_nonfinite,
    )
    if not isinstance(descriptor, Mapping):
        raise ValueError("tensor layout descriptor must be a mapping")
    return descriptor


def _checked_nbytes(shape: Tuple[int, ...], dtype: np.dtype) -> int:
    elements = math.prod(shape)
    if elements > MAX_LAYOUT_BYTES // dtype.itemsize:
        raise ValueError("tensor field exceeds the supported layout size")
    return elements * dtype.itemsize


@dataclass(frozen=True)
class TensorField:
    """One validated typed view inside a contiguous payload allocation."""

    name: str
    shape: Tuple[int, ...]
    dtype: np.dtype
    offset: int
    nbytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "dtype", np.dtype(self.dtype))

    @property
    def torch_dtype(self) -> torch.dtype:
        try:
            return NP_TO_TORCH[self.dtype]
        except KeyError as error:
            raise ValueError(f"unsupported tensor dtype {self.dtype}") from error

    def descriptor(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype.str,
            "offset": self.offset,
            "nbytes": self.nbytes,
        }


@dataclass(frozen=True)
class TensorLayout:
    """Validated physical layout for a reusable contiguous payload."""

    fields: Tuple[TensorField, ...]
    padding: Tuple[Tuple[int, int], ...]
    payload_nbytes: int
    alignment: int
    layout_id: str

    @classmethod
    def compile(
        cls,
        schema: Sequence[TensorSpec],
        *,
        alignment: int = DEFAULT_ALIGNMENT,
    ) -> "TensorLayout":
        if not isinstance(alignment, int) or isinstance(alignment, bool):
            raise ValueError("layout alignment must be an integer")
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("layout alignment must be a positive power of two")
        if alignment > DEFAULT_ALIGNMENT:
            raise ValueError(
                f"layout alignment cannot exceed the transport boundary "
                f"({DEFAULT_ALIGNMENT} bytes)"
            )

        placed_fields = []
        names = set()
        cursor = 0
        prepared = []
        for position, spec in enumerate(schema):
            try:
                dtype = np.dtype(spec.dtype)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"unsupported tensor dtype {spec.dtype} for {spec.name!r}"
                ) from error
            prepared.append((position, spec, dtype))
        # Ascending dtype alignment leaves the final extent aligned for reuse in
        # array/ring slots. Logical access remains name based.
        prepared.sort(key=lambda item: (item[2].itemsize, item[0]))
        padding = []
        for position, spec, dtype in prepared:
            name = spec.name
            shape = tuple(spec.shape)
            if not isinstance(name, str) or not name or name in names:
                raise ValueError(
                    "tensor layout field names must be strings, unique and nonempty"
                )
            if dtype not in NP_TO_TORCH or dtype.hasobject or dtype.itemsize <= 0:
                raise ValueError(f"unsupported tensor dtype {dtype} for {name!r}")
            if any(type(dimension) is not int or dimension < 0 for dimension in shape):
                raise ValueError(f"invalid tensor shape for {name!r}: {shape}")
            field_alignment = dtype.itemsize
            if field_alignment & (field_alignment - 1):
                raise ValueError(f"unsupported dtype alignment for {name!r}")
            if field_alignment > alignment:
                raise ValueError(
                    f"layout alignment {alignment} is smaller than the "
                    f"{field_alignment}-byte alignment required by {name!r}"
                )
            offset = _align(cursor, field_alignment)
            if offset > cursor:
                padding.append((cursor, offset - cursor))
            nbytes = _checked_nbytes(shape, dtype)
            if offset > MAX_LAYOUT_BYTES - nbytes:
                raise ValueError("tensor layout exceeds the supported size")
            placed_fields.append(
                (position, TensorField(name, shape, dtype, offset, nbytes))
            )
            names.add(name)
            cursor = offset + nbytes

        payload_nbytes = _align(cursor, alignment)
        if payload_nbytes > MAX_LAYOUT_BYTES:
            raise ValueError("tensor layout exceeds the supported size")
        if payload_nbytes > cursor:
            padding.append((cursor, payload_nbytes - cursor))
        fields = [field for _, field in sorted(placed_fields)]
        canonical = {
            "wire_version": WIRE_VERSION,
            "alignment": alignment,
            "payload_nbytes": payload_nbytes,
            "padding": [list(item) for item in padding],
            "fields": [field.descriptor() for field in fields],
        }
        encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        layout_id = hashlib.blake2b(encoded, digest_size=16).hexdigest()
        return cls(tuple(fields), tuple(padding), payload_nbytes, alignment, layout_id)

    @classmethod
    def from_descriptor(cls, descriptor: Mapping[str, Any]) -> "TensorLayout":
        expected_keys = {
            "wire_version",
            "layout_id",
            "alignment",
            "payload_nbytes",
            "padding",
            "fields",
        }
        if not isinstance(descriptor, Mapping) or set(descriptor) != expected_keys:
            raise ValueError("tensor layout descriptor is not canonical")
        if descriptor.get("wire_version") != WIRE_VERSION:
            raise ValueError("unsupported payload wire version")
        raw_fields = descriptor.get("fields")
        if not isinstance(raw_fields, list):
            raise ValueError("layout descriptor fields must be a list")
        schema = []
        for raw in raw_fields:
            if not isinstance(raw, Mapping):
                raise ValueError("layout field descriptor must be a mapping")
            try:
                schema.append(
                    TensorSpec(
                        name=raw.get("name", ""),
                        shape=tuple(raw.get("shape", ())),
                        dtype=np.dtype(raw.get("dtype")),
                    )
                )
            except (TypeError, ValueError) as error:
                raise ValueError("invalid tensor layout field descriptor") from error
        layout = cls.compile(schema, alignment=descriptor.get("alignment", 0))
        if layout.payload_nbytes != descriptor.get("payload_nbytes"):
            raise ValueError("layout descriptor payload size mismatch")
        if [field.descriptor() for field in layout.fields] != raw_fields:
            raise ValueError("layout descriptor offsets or sizes are not canonical")
        if descriptor.get("padding") != [list(item) for item in layout.padding]:
            raise ValueError("layout descriptor padding is not canonical")
        claimed_id = descriptor.get("layout_id")
        if validate_layout_id(claimed_id) != layout.layout_id:
            raise ValueError("layout descriptor identity mismatch")
        # Python mapping equality treats bool as an int (``False == 0``),
        # which is too permissive at a wire trust boundary. Compare canonical
        # JSON as well so type substitutions and other noncanonical encodings
        # cannot borrow a valid layout identity.
        encoded_input = dict(descriptor)
        encoded_input["fields"] = [dict(field) for field in raw_fields]
        try:
            encoded_input = json.dumps(
                encoded_input, sort_keys=True, separators=(",", ":")
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "tensor layout descriptor is not JSON canonical"
            ) from error
        if encoded_input != serialize_layout_descriptor(layout):
            raise ValueError("tensor layout descriptor is not canonical")
        return layout

    def descriptor(self) -> dict[str, Any]:
        return {
            "wire_version": WIRE_VERSION,
            "layout_id": self.layout_id,
            "alignment": self.alignment,
            "payload_nbytes": self.payload_nbytes,
            "padding": [list(item) for item in self.padding],
            "fields": [field.descriptor() for field in self.fields],
        }

    def schema(self) -> list[TensorSpec]:
        return [
            TensorSpec(name=field.name, shape=field.shape, dtype=field.dtype)
            for field in self.fields
        ]

    def offsets(self) -> dict[str, int]:
        return {field.name: field.offset for field in self.fields}

    def views(self, storage: torch.Tensor) -> dict[str, torch.Tensor]:
        if (
            storage.dtype != torch.uint8
            or storage.ndim != 1
            or not storage.is_contiguous()
        ):
            raise ValueError(
                "payload storage must be a contiguous one-dimensional uint8 tensor"
            )
        if storage.numel() < self.payload_nbytes:
            raise ValueError("payload storage is smaller than its layout")
        if storage.numel() and storage.data_ptr() % self.alignment:
            raise ValueError("payload storage does not satisfy its layout alignment")
        result = {}
        for field in self.fields:
            raw = storage.narrow(0, field.offset, field.nbytes)
            result[field.name] = raw.view(field.torch_dtype).reshape(field.shape)
        return result

    def zero_padding(self, storage: torch.Tensor) -> None:
        """Initialize only non-field bytes so pooled storage never leaks stale data."""
        if storage.dtype != torch.uint8 or storage.ndim != 1:
            raise ValueError("payload storage must be one-dimensional uint8")
        if storage.numel() < self.payload_nbytes:
            raise ValueError("payload storage is smaller than its layout")
        for offset, nbytes in self.padding:
            storage.narrow(0, offset, nbytes).zero_()


class LayoutCache:
    """Small bounded cache that validates descriptors before reuse."""

    def __init__(self, capacity: int = 64):
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
            raise ValueError("layout cache capacity must be a positive integer")
        self._capacity = capacity
        self._layouts: dict[str, TensorLayout] = {}
        self._order: list[str] = []
        self._lock = threading.RLock()

    def register(self, layout: TensorLayout) -> TensorLayout:
        with self._lock:
            current = self._layouts.get(layout.layout_id)
            if current is not None:
                if current != layout:
                    raise ValueError("layout identity collision")
                self._order.remove(layout.layout_id)
                self._order.append(layout.layout_id)
                return current
            self._layouts[layout.layout_id] = layout
            self._order.append(layout.layout_id)
            if len(self._order) > self._capacity:
                evicted = self._order.pop(0)
                del self._layouts[evicted]
            return layout

    def resolve(
        self, layout_id: str, descriptor: Mapping[str, Any] | None
    ) -> TensorLayout:
        layout_id = validate_layout_id(layout_id)
        with self._lock:
            current = self._layouts.get(layout_id)
            if current is not None:
                self._order.remove(layout_id)
                self._order.append(layout_id)
                return current
            if descriptor is None:
                raise KeyError(f"unknown tensor layout {layout_id}")
            layout = TensorLayout.from_descriptor(descriptor)
            if layout.layout_id != layout_id:
                raise ValueError("payload reference and descriptor layout IDs disagree")
            return self.register(layout)

    def __len__(self) -> int:
        with self._lock:
            return len(self._layouts)
