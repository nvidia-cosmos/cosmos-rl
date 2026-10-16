# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Byte admission and explicit batch leases for bounded payload reception.

Reservations include the complete decoded batch and its transient workspace.
They are deliberately distinct from live tensor byte attribution.
"""

import math
import threading
import time
from collections.abc import Mapping
from collections import defaultdict, deque
from dataclasses import dataclass, field

import torch


class ReceiveMemoryError(RuntimeError):
    """A receive cannot safely proceed under the configured memory contract."""


class ReceiveBudget:
    def __init__(self, limit, timeout):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("nccl_receive_budget_bytes must be a positive integer")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("nccl_receive_admission_timeout must be positive")
        self.limit = limit
        self.timeout = timeout
        self.condition = threading.Condition()
        self.closed = False
        self.used = self.peak = self.waits = 0
        self.raw = self.decoded = self.peak_live = 0
        self.consumer_bytes = 0
        self.waiting_for_consumer = False
        self.consumer_wait_ended_at = 0.0

    def reserve(self, size):
        if size < 0:
            raise ValueError("negative receive reservation")
        if size > self.limit:
            raise ReceiveMemoryError(
                f"NCCL batch needs {size} bytes (decoded batch plus receive/decode "
                f"workspace), exceeding nccl_receive_budget_bytes={self.limit}; "
                "increase the budget or reduce batch/payload size. Disk spill is disabled."
            )
        deadline = time.monotonic() + self.timeout
        with self.condition:
            waited = False
            while self.used + size > self.limit and not self.closed:
                if not waited:
                    self.waits += 1
                    waited = True
                if self.consumer_bytes and self.used == self.consumer_bytes:
                    # No receive has started: the only owner is the training
                    # consumer. Its step duration is not a transport deadline.
                    self.waiting_for_consumer = True
                    try:
                        self.condition.wait()
                    finally:
                        self.waiting_for_consumer = False
                        self.consumer_wait_ended_at = time.monotonic()
                    deadline = time.monotonic() + self.timeout
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ReceiveMemoryError(
                        "NCCL receive admission timed out; final readers must call "
                        "release_prefetch before waiting for the next batch, or "
                        "increase nccl_receive_budget_bytes."
                    )
                self.condition.wait(remaining)
            if self.closed:
                raise ReceiveMemoryError("NCCL receive admission cancelled by shutdown")
            self.used += size
            self.peak = max(self.peak, self.used)

    def watchdog_delay(self, deadline, timeout):
        """Exclude proven pre-receive consumer backpressure, not native work."""
        with self.condition:
            if self.waiting_for_consumer:
                return timeout
            return max(
                0.0,
                max(deadline, self.consumer_wait_ended_at + timeout) - time.monotonic(),
            )

    def release(self, size):
        with self.condition:
            self.used -= size
            self.condition.notify_all()

    def reserve_available(self, size):
        """Grow an admitted workspace without waiting while holding capacity."""
        with self.condition:
            if self.closed:
                raise ReceiveMemoryError("NCCL receive admission cancelled by shutdown")
            granted = min(size, self.limit - self.used)
            self.used += granted
            self.peak = max(self.peak, self.used)
            return granted

    def attribute(self, *, raw=0, decoded=0):
        with self.condition:
            self.raw += raw
            self.decoded += decoded
            self.peak_live = max(self.peak_live, self.raw + self.decoded)

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return dict(
                budget_bytes=self.limit,
                reserved_bytes=self.used,
                peak_reserved_bytes=self.peak,
                raw_receive_bytes=self.raw,
                decoded_leased_bytes=self.decoded,
                admission_waits=self.waits,
                current_tensor_bytes=self.raw + self.decoded,
                peak_tensor_bytes=self.peak_live,
            )


class ReceivedBatch(dict):
    """Dictionary owning a reservation until the final consumer releases it.

    Callers must finish every reader, including tensor views, before release.
    Completion events retain both storage and budget until all streams finish.
    No implicit destructor release: forgetting a lease must not undercount memory.
    """

    def __init__(
        self,
        values,
        budget=None,
        nbytes=0,
        *,
        rejected_keys=(),
        owners=(),
        on_release=None,
    ):
        super().__init__(values)
        # Explicit safe outcomes, scoped to this batch's lifetime. Absence from
        # the values dictionary alone is not proof that a transfer was rejected.
        self.rejected_keys = set(rejected_keys)
        self.budget = budget
        self.nbytes = nbytes
        self.released = False
        self.consumer_owned = False
        self._owners = tuple(owners)
        self._on_release = on_release
        # The consumer can reshape, pop, or replace dictionary entries. Lease
        # ownership must survive those mutations until all readers complete.
        storages = {}
        for payload in self.values():
            for tensor in payload.values():
                storage = tensor.untyped_storage()
                storages[(str(tensor.device), storage.data_ptr())] = storage
        self._storages = tuple(storages.values())
        self._state_lock = threading.RLock()

    def claim_consumer(self):
        """Mark an actual handoff, not merely a queued or prepared result."""
        with self._state_lock:
            if self.budget is None:
                if self.released:
                    raise ReceiveMemoryError("Cannot consume a released payload batch")
                self.consumer_owned = True
                return
            with self.budget.condition:
                if self.released:
                    raise ReceiveMemoryError("Cannot consume a released payload batch")
                if not self.consumer_owned:
                    self.budget.consumer_bytes += self.nbytes
                    self.consumer_owned = True
                    self.budget.condition.notify_all()

    def check_returned_metrics(self, value):
        """A training report must not keep payload storage/autograd alive."""
        import torch

        if isinstance(value, torch.Tensor):
            if value.requires_grad or any(
                storage.device == value.device
                and storage.data_ptr() == value.untyped_storage().data_ptr()
                for storage in self._storages
            ):
                raise ReceiveMemoryError(
                    "Training metrics must be detached and not alias received payloads"
                )
        elif isinstance(value, Mapping):
            for item in value.values():
                self.check_returned_metrics(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                self.check_returned_metrics(item)

    def release(self, *, streams=()):
        with self._state_lock:
            if self.released:
                return
            # The current stream is always included. Additional reader streams must
            # be supplied by the consumer; record all events before waiting on any.
            devices = {
                storage.device
                for storage in self._storages
                if storage.device.type == "cuda"
            }
            readers = list(streams) + [torch.cuda.current_stream(d) for d in devices]
            events = []
            for stream in readers:
                event = torch.cuda.Event()
                event.record(stream)
                events.append(event)
            for event in events:
                event.synchronize()
            self.clear()
            self.rejected_keys.clear()
            self._storages = ()
            self._owners = ()
            if self._on_release is not None:
                self._on_release()
                self._on_release = None
            if self.budget is not None:
                with self.budget.condition:
                    if self.consumer_owned:
                        self.budget.consumer_bytes -= self.nbytes
                    self.budget.attribute(decoded=-self.nbytes)
                    self.budget.release(self.nbytes)
            self.released = True


class ReusableViewMap(dict):
    """Cached tensor-view mapping that repairs only keys changed by a consumer."""

    def __init__(self, base):
        self._base = base
        self._dirty = set()
        dict.__init__(self, base)

    def __setitem__(self, key, value):
        self._dirty.add(key)
        dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        self._dirty.add(key)
        dict.__delitem__(self, key)

    def clear(self):
        self._dirty.update(self)
        dict.clear(self)

    def pop(self, key, *default):
        self._dirty.add(key)
        return dict.pop(self, key, *default)

    def popitem(self):
        key, value = dict.popitem(self)
        self._dirty.add(key)
        return key, value

    def setdefault(self, key, default=None):
        if key not in self:
            self._dirty.add(key)
        return dict.setdefault(self, key, default)

    def update(self, *args, **kwargs):
        incoming = dict(*args, **kwargs)
        self._dirty.update(incoming)
        dict.update(self, incoming)

    def __ior__(self, other):
        self.update(other)
        return self

    def reset(self, names=()):
        self._dirty.update(names)
        for name in self._dirty:
            if name in self._base:
                dict.__setitem__(self, name, self._base[name])
            elif name in self:
                dict.__delitem__(self, name)
        self._dirty.clear()
        return self

    def retain(self, *, owner, release):
        """Attach the current storage owner without copying this mapping."""
        self.owner = owner
        self.release_storage = release
        return self


@dataclass
class _ReusableBuffer:
    tensor: torch.Tensor
    bindings: dict


@dataclass
class ReceiveBufferLease:
    """Exclusive ownership of one reusable receive allocation."""

    pool: "ReceiveBufferPool"
    key: tuple
    buffer: _ReusableBuffer
    generation: int
    released: bool = False
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    @property
    def tensor(self) -> torch.Tensor:
        return self.buffer.tensor

    def bind_views(self, layout, *, offset=0, reset_fields=()):
        """Reuse typed tensor wrappers for this backing allocation."""
        binding_key = (layout.layout_id, offset)
        binding = self.buffer.bindings.get(binding_key)
        if binding is None:
            storage = self.tensor.narrow(0, offset, layout.payload_nbytes)
            base_views = layout.views(storage)
            active_views = ReusableViewMap(base_views)
            binding = (base_views, active_views)
            self.buffer.bindings[binding_key] = binding
        _, active_views = binding
        return active_views.reset(reset_fields)

    def release(self) -> None:
        with self._lock:
            if self.released:
                return
            self.pool._release(self)
            self.released = True

    def quarantine(self) -> None:
        with self._lock:
            if self.released:
                return
            self.pool._quarantine(self)
            self.released = True


class ReceiveBufferPool:
    """Reuse exact-layout receive buffers after their final readers finish."""

    def __init__(
        self,
        *,
        max_cached_per_layout: int = 64,
        max_cached_buffers: int = 64,
        max_cached_bytes: int | None = None,
    ):
        if (
            isinstance(max_cached_per_layout, bool)
            or not isinstance(max_cached_per_layout, int)
            or max_cached_per_layout < 0
        ):
            raise ValueError("receive pool cache size must be a nonnegative integer")
        if (
            isinstance(max_cached_buffers, bool)
            or not isinstance(max_cached_buffers, int)
            or max_cached_buffers < 0
        ):
            raise ValueError("receive pool total cache size must be nonnegative")
        if max_cached_bytes is not None and (
            isinstance(max_cached_bytes, bool)
            or not isinstance(max_cached_bytes, int)
            or max_cached_bytes < 0
        ):
            raise ValueError("receive pool byte limit must be nonnegative")
        self._max_cached_per_layout = max_cached_per_layout
        self._max_cached_buffers = max_cached_buffers
        self._max_cached_bytes = max_cached_bytes
        self._free = defaultdict(deque)
        self._cached_buffers = 0
        self._cached_bytes = 0
        self._next_generation = 0
        self._lock = threading.Lock()
        self._closed = False
        self._quarantined = []

    def acquire(self, layout_id: str, nbytes: int, device) -> ReceiveBufferLease:
        if not layout_id or nbytes < 0:
            raise ValueError("receive allocation requires a layout and valid size")
        resolved_device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        key = (layout_id, str(resolved_device), nbytes)
        evicted = []
        with self._lock:
            if self._closed:
                raise ReceiveMemoryError("receive buffer pool is closed")
            self._next_generation += 1
            generation = self._next_generation
            free = self._free.get(key)
            buffer = free.popleft() if free else None
            if buffer is not None:
                self._cached_buffers -= 1
                self._cached_bytes -= buffer.tensor.untyped_storage().nbytes()
                if not free:
                    del self._free[key]
            else:
                while self._free and (
                    self._cached_buffers >= self._max_cached_buffers
                    or (
                        self._max_cached_bytes is not None
                        and self._cached_bytes + nbytes > self._max_cached_bytes
                    )
                ):
                    stale_key = next(iter(self._free))
                    stale = self._free[stale_key].pop()
                    if not self._free[stale_key]:
                        del self._free[stale_key]
                    self._cached_buffers -= 1
                    self._cached_bytes -= stale.tensor.untyped_storage().nbytes()
                    evicted.append(stale)
        # Drop evicted CUDA allocations outside the bookkeeping lock.
        evicted.clear()
        if buffer is None:
            buffer = _ReusableBuffer(
                torch.empty(nbytes, dtype=torch.uint8, device=resolved_device),
                {},
            )
        return ReceiveBufferLease(self, key, buffer, generation)

    def _release(self, lease: ReceiveBufferLease) -> None:
        with self._lock:
            nbytes = lease.buffer.tensor.untyped_storage().nbytes()
            if (
                self._closed
                or self._max_cached_per_layout == 0
                or self._cached_buffers >= self._max_cached_buffers
                or (
                    self._max_cached_bytes is not None
                    and self._cached_bytes + nbytes > self._max_cached_bytes
                )
            ):
                return
            free = self._free[lease.key]
            if len(free) < self._max_cached_per_layout:
                free.append(lease.buffer)
                self._cached_buffers += 1
                self._cached_bytes += nbytes

    def _quarantine(self, lease: ReceiveBufferLease) -> None:
        with self._lock:
            self._quarantined.append(lease.buffer)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._free.clear()
            self._cached_buffers = 0
            self._cached_bytes = 0


def storage_bytes(payloads):
    """Count full unique storage, including backing storage of truncated views."""
    storages = {}
    for payload in payloads:
        for tensor in payload.values():
            storage = tensor.untyped_storage()
            storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())
