# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public composition API for producer-side payload transport."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
import threading
from types import MappingProxyType
from typing import Any, Callable, Optional

from cosmos_rl.utils.payload_transport.layout import TensorLayout
from cosmos_rl.utils.payload_transport.storage import (
    PayloadStorageRequirements,
    WritablePayload,
)
from cosmos_rl.utils.trajectory import TensorSpec


class PayloadRef(Mapping[str, Any]):
    """Immutable publication reference suitable for controller handoff."""

    def __init__(self, values: Mapping[str, Any]):
        self._values = MappingProxyType(deepcopy(dict(values)))

    def __getitem__(self, key: str) -> Any:
        return deepcopy(self._values[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached mutable representation for existing controllers."""
        return deepcopy(dict(self._values))


class PayloadContext:
    """Own one producer backend, registered layouts and transport teardown."""

    def __init__(
        self,
        backend_name: str,
        producer: Any,
        default_layout: TensorLayout,
        *,
        layout_cache_capacity: int = 64,
    ) -> None:
        self._backend_name = backend_name
        self._producer = producer
        self._default_layout = default_layout
        self._validate_layout_capacity(layout_cache_capacity)
        self._layout_capacity = layout_cache_capacity
        self._layouts = {default_layout.layout_id: default_layout}
        self._acquired: set[WritablePayload] = set()
        self._lock = threading.RLock()
        self._closing = False
        self._closed = False

    @classmethod
    def create(cls, backend: str, **kwargs: Any) -> "PayloadContext":
        """Construct and attach a producer backend selected by name."""
        layout_cache_capacity = kwargs.pop("layout_cache_capacity", 64)
        cls._validate_layout_capacity(layout_cache_capacity)
        if backend == "nccl":
            from cosmos_rl.utils.payload_transport.nccl.producer import (
                NCCLPayloadProducer,
            )

            producer = NCCLPayloadProducer()
            producer.setup(**kwargs)
            return cls(
                backend,
                producer,
                producer.default_layout,
                layout_cache_capacity=layout_cache_capacity,
            )
        if backend == "ucxx":
            from cosmos_rl.utils.payload_transport.ucxx.producer import (
                UCXXPayloadProducer,
            )

            producer = UCXXPayloadProducer()
            producer.setup(**kwargs)
            return cls(
                backend,
                producer,
                producer.default_layout,
                layout_cache_capacity=layout_cache_capacity,
            )
        raise ValueError(f"unsupported payload transport backend {backend!r}")

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def default_layout(self) -> TensorLayout:
        return self._default_layout

    def register_layout(self, schema: Sequence[TensorSpec]) -> TensorLayout:
        """Compile and cache an immutable physical layout."""
        with self._lock:
            self._ensure_open()
            layout = TensorLayout.compile(schema)
            current = self._layouts.get(layout.layout_id)
            if current is not None:
                if current != layout:
                    raise ValueError("layout identity collision")
                return current
            if len(self._layouts) >= self._layout_capacity:
                raise RuntimeError(
                    "payload layout registry capacity exhausted; reuse a registered "
                    "layout or increase layout_cache_capacity"
                )
            self._producer.register_layout(layout)
            self._layouts[layout.layout_id] = layout
            return layout

    def publish_tensors(
        self,
        tensors: Mapping[str, Any],
        *,
        layout: Optional[TensorLayout] = None,
        copy_policy: str = "batched",
    ) -> Optional[PayloadRef]:
        """Publish one tensor mapping through a batched packing adapter."""
        with self._lock:
            self._ensure_open()
            selected = layout or self._default_layout
            registered = self._registered_layout(selected)
            raw_ref = self._producer.publish(
                dict(tensors), layout=registered, copy_policy=copy_policy
            )
            return PayloadRef(raw_ref) if raw_ref is not None else None

    def acquire(self, *, layout: Optional[TensorLayout] = None) -> WritablePayload:
        """Reserve a transport-owned slot with typed writable views."""
        with self._lock:
            self._ensure_open()
            selected = layout or self._default_layout
            registered = self._registered_layout(selected)
            payload = self._producer.acquire(layout=registered)
            self._acquired.add(payload)
            return payload

    def storage_requirements(
        self, *, layout: Optional[TensorLayout] = None
    ) -> PayloadStorageRequirements:
        """Describe storage accepted by :meth:`borrow` without allocating it."""
        with self._lock:
            self._ensure_open()
            selected = layout or self._default_layout
            registered = self._registered_layout(selected)
            requirements = getattr(self._producer, "storage_requirements", None)
            if not callable(requirements):
                raise NotImplementedError(
                    f"{self._backend_name} does not accept borrowed payload storage"
                )
            return requirements(layout=registered)

    def borrow(
        self,
        storage: Any,
        *,
        layout: Optional[TensorLayout] = None,
        owner: Any = None,
        on_release: Optional[Callable[[], None]] = None,
    ) -> WritablePayload:
        """Lease a compatible application allocation to the transport.

        The backend retains ``owner`` (or ``storage`` when omitted) until every
        accepted native reader is proven complete. ``on_release`` runs exactly
        once on safe retirement or abandonment, and never on quarantine.
        """
        with self._lock:
            self._ensure_open()
            selected = layout or self._default_layout
            registered = self._registered_layout(selected)
            borrow = getattr(self._producer, "borrow", None)
            if not callable(borrow):
                raise NotImplementedError(
                    f"{self._backend_name} does not accept borrowed payload storage"
                )
            if on_release is not None and not callable(on_release):
                raise TypeError("on_release must be callable")
            payload = borrow(
                storage,
                layout=registered,
                owner=storage if owner is None else owner,
                on_release=on_release or (lambda: None),
            )
            self._acquired.add(payload)
            return payload

    def metrics(self) -> Mapping[str, Any]:
        """Return a detached backend metrics snapshot when available."""
        with self._lock:
            snapshot = getattr(self._producer, "metrics", None)
            return MappingProxyType(dict(snapshot() if callable(snapshot) else {}))

    def publish(
        self,
        payload: WritablePayload,
        *,
        ready_event: Any = None,
    ) -> PayloadRef:
        """Publish an acquired slot without repacking its tensor fields."""
        with self._lock:
            self._ensure_open()
            if payload not in self._acquired:
                raise ValueError("payload was not acquired from this context")
            try:
                raw_ref = self._producer.publish_storage(
                    payload, ready_event=ready_event
                )
                return PayloadRef(raw_ref)
            except BaseException:
                # If the backend failed before accepting ownership, return the
                # acquired slot. Once submitted, close is intentionally a no-op:
                # the backend owns completion, quarantine, or retirement.
                payload.close()
                raise
            finally:
                self._acquired.discard(payload)

    def close(self, timeout: float = 5.0) -> None:
        """Stop admission and release backend resources after accepted work settles."""
        with self._lock:
            if self._closed:
                return
            self._closing = True
            failures = []
            for payload in tuple(self._acquired):
                try:
                    payload.close()
                except BaseException as error:
                    failures.append(error)
                else:
                    self._acquired.discard(payload)
            if failures:
                raise RuntimeError(
                    "payload context could not abandon "
                    f"{len(failures)} unpublished operation(s)"
                ) from failures[0]
            try:
                self._producer.close(timeout=timeout)
            except BaseException as error:
                failures.append(error)
            if failures:
                raise RuntimeError(
                    f"payload context shutdown failed in {len(failures)} operation(s)"
                ) from failures[0]
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closing or self._closed:
            raise RuntimeError("payload context is closing or closed")

    def _registered_layout(self, layout: TensorLayout) -> TensorLayout:
        registered = self._layouts.get(layout.layout_id)
        if registered is None:
            raise KeyError(f"unknown tensor layout {layout.layout_id}")
        if registered != layout:
            raise ValueError("layout identity collision")
        return registered

    @staticmethod
    def _validate_layout_capacity(capacity: int) -> None:
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
            raise ValueError("layout registry capacity must be a positive integer")

    def __enter__(self) -> "PayloadContext":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
