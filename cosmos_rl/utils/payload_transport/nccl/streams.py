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

"""Per-process transfer-stream pool + CUDA event helpers for NCCL payload.

Both the rollout sender (:class:`NCCLRolloutMixin`) and the trainer
receiver (:class:`NCCLDataPackerMixin`) move payload bytes on **dedicated
CUDA streams**, separate from the compute stream *and* the weight-sync
stream, so payload NCCL kernels overlap compute and cannot false-serialize
against either.  This module owns:

* :class:`TransferStreamPool` — a small (default 1) pool of low-priority
  CUDA streams, allocated lazily per process/device.  Each 2-rank
  communicator's ops stay ordered on a single stream; different comms can
  use different streams for cross-pair overlap.  More than ~2 streams
  mostly adds scheduling overhead because NCCL kernels already contend
  for SMs / NIC, so the default is deliberately small.  The pool runs at
  **low priority** so compute kernels win at scheduling boundaries.

* Event helpers (:func:`record_event`, :func:`wait_event`) mirroring the
  ``activation_offloading`` s0/s1 hand-off.  Correctness is event-based:

  - **Sender**: record a ready-event on the *compute* stream once the
    trajectory tensor is produced; the transfer stream ``wait_event``\\ s
    it before ``nccl_send``; the send buffer is held alive until a
    send-complete event fires.
  - **Receiver**: after ``nccl_recv``, record a recv-complete event; the
    training consumer ``wait_event``\\ s it before reading.

Granularity is **per process/device**: CUDA streams live in a process's
device context and each cosmos-rl rank is one process on one GPU.

CPU-safety
----------
Every entry point degrades to a well-defined no-op when CUDA is
unavailable (``record_event`` returns ``None``; ``wait_event(None)`` is a
no-op; the pool hands out ``None`` "streams").  This keeps the scheduling
logic unit-testable on CPU-only hosts and lets callers write
transport code without sprinkling ``torch.cuda.is_available()`` guards.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

import torch

from cosmos_rl.utils.logging import logger

__all__ = [
    "TransferStreamPool",
    "get_transfer_stream_pool",
    "reset_transfer_stream_pools",
    "record_event",
    "wait_event",
    "bind_thread_device",
]


def bind_thread_device(device: Any) -> None:
    """Bind the calling worker/executor thread to ``device``.

    CUDA's current device is thread-local.  The producer's sender threads and
    the consumer's prefetch thread create communicators and issue sends/recvs
    off the main thread; if they never set the device, that work targets
    ``cuda:0`` and pynccl rejects a buffer on any other device with a device
    mismatch.  Idempotent + best-effort (no-op on CPU / non-CUDA device).
    """
    if device is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
    except Exception:  # pragma: no cover - defensive
        pass


# Default CUDA stream priority.  On CUDA, lower number == higher priority;
# the valid range is queried from the runtime.  We want payload transfer to
# be *lower* priority than compute, i.e. the least-preferred (highest number)
# priority the device exposes.
def _low_priority() -> int:
    if not torch.cuda.is_available():
        return 0
    try:
        least, greatest = torch.cuda.Stream.priority_range()
        # priority_range() returns (least_priority, greatest_priority) where
        # greatest_priority is the *most* preferred (most negative).  The least
        # preferred (largest, typically 0) is what we want for background I/O.
        return int(max(least, greatest))
    except Exception:
        return 0


class TransferStreamPool:
    """Lazily-allocated pool of low-priority CUDA transfer streams.

    Streams are created on first :meth:`acquire` (so constructing a pool
    is cheap and CPU-safe) and handed out round-robin.  Keep one pool per
    process/device; use :func:`get_transfer_stream_pool` for the shared
    per-device singleton.
    """

    def __init__(
        self,
        size: int = 1,
        priority: Optional[int] = None,
        device: Any = None,
    ) -> None:
        if size < 1:
            size = 1
        self._size = size
        self._priority = priority
        self._device = device
        self._streams: List[Any] = []
        self._next = 0
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        return self._size

    @property
    def priority(self) -> int:
        return self._low_priority_value()

    def _low_priority_value(self) -> int:
        return self._priority if self._priority is not None else _low_priority()

    def _ensure_streams(self) -> None:
        if self._streams or not torch.cuda.is_available():
            return
        prio = self._low_priority_value()
        for _ in range(self._size):
            self._streams.append(torch.cuda.Stream(device=self._device, priority=prio))
        logger.debug(
            "[TransferStreamPool] Allocated %d transfer stream(s) at priority %d "
            "(device=%s)",
            self._size,
            prio,
            self._device,
        )

    def acquire(self) -> Any:
        """Return the next transfer stream round-robin.

        Returns ``None`` when CUDA is unavailable (callers treat a ``None``
        stream as "run on the default/current stream").
        """
        with self._lock:
            self._ensure_streams()
            if not self._streams:
                return None
            stream = self._streams[self._next % len(self._streams)]
            self._next += 1
            return stream

    def all_streams(self) -> List[Any]:
        """Materialize + return every stream (empty on CPU).  For tests."""
        with self._lock:
            self._ensure_streams()
            return list(self._streams)


# ---------------------------------------------------------------------------
# Per-process singleton registry (keyed by device)
# ---------------------------------------------------------------------------

_POOLS: Dict[Tuple[Any, int], TransferStreamPool] = {}
_POOLS_LOCK = threading.Lock()


def get_transfer_stream_pool(
    *,
    size: int = 1,
    priority: Optional[int] = None,
    device: Any = None,
) -> TransferStreamPool:
    """Return the process-wide transfer-stream pool for ``device``.

    Idempotent per ``(device, size)`` key: the first caller fixes the pool
    configuration for that device; later callers with the same key get the
    same pool.  Different sizes on the same device yield distinct pools
    (the sender and receiver may want different pool sizes).
    """
    key = (device, size)
    with _POOLS_LOCK:
        pool = _POOLS.get(key)
        if pool is None:
            pool = TransferStreamPool(size=size, priority=priority, device=device)
            _POOLS[key] = pool
        return pool


def reset_transfer_stream_pools() -> None:
    """Drop all cached pools.  For teardown / tests."""
    with _POOLS_LOCK:
        _POOLS.clear()


# ---------------------------------------------------------------------------
# Event helpers (CPU-safe)
# ---------------------------------------------------------------------------


def record_event(stream: Any = None) -> Optional[torch.cuda.Event]:
    """Record and return a CUDA event on ``stream`` (or the current stream).

    Returns ``None`` when CUDA is unavailable so callers can pass the
    result straight to :func:`wait_event`, which no-ops on ``None``.
    """
    if not torch.cuda.is_available():
        return None
    event = torch.cuda.Event()
    if stream is not None:
        event.record(stream)
    else:
        event.record()
    return event


def wait_event(stream: Any, event: Optional[torch.cuda.Event]) -> None:
    """Make ``stream`` wait until ``event`` completes before later work.

    No-op when ``event`` is ``None`` (CPU path) or CUDA is unavailable.
    When ``stream`` is ``None`` the current stream waits.
    """
    if event is None or not torch.cuda.is_available():
        return
    if stream is not None:
        stream.wait_event(event)
    else:
        torch.cuda.current_stream().wait_event(event)
