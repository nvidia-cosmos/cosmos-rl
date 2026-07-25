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

"""Producer-side GPU send-buffer registry with bounded backpressure.

The rollout worker packs each trajectory into a fixed-schema GPU buffer
and registers it here keyed by ``transfer_id``.  The demand-driven sender
threads look the buffer up when the receiver requests it; the cleanup
subscriber frees it when the controller discards the rollout.

Why backpressure is mandatory
-----------------------------
Rollout generation prefetches *ahead* of demand, but NCCL transfer is
**demand-driven** — a buffer is only sent when a receiver asks for it.
Without a cap, generation would register buffers faster than they drain
and OOM the rollout GPU.  So the registry has a **bounded capacity**:
when full, :meth:`register` applies backpressure (block until a slot
frees, then — to guarantee it never wedges generation — drop the oldest
un-sent buffer).  This is the key cross-mixin constraint.

Idempotency
------------------------------------
There is no shared per-transfer reader state, so register/free are
idempotent: re-registering a ``transfer_id`` frees the prior buffer
first; :meth:`free` on an absent or already-freed id is a no-op.  A stale
cleanup message or a duplicate request therefore cannot double-free or
orphan a buffer.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from cosmos_rl.utils.logging import logger

__all__ = ["SendBufferEntry", "SendBufferRegistry"]


def _event_fired(ev: Any) -> bool:
    """True if a send-complete event has fired.

    A non-CUDA / fake event without ``query`` (or one that raises) is treated
    as complete so the bookkeeping is testable on CPU and never wedges here.
    """
    query = getattr(ev, "query", None)
    if query is None:
        return True
    try:
        return bool(query())
    except Exception:  # pragma: no cover - opaque event -> assume complete
        return True


def _send_drained(entry: "SendBufferEntry") -> bool:
    """True if EVERY send of ``entry`` has completed.

    A ``transfer_id`` may be served to more than one receiver (TP/PP/CP policy
    ranks sharing a DP id), so the buffer is drained only when no send is
    still in flight (``inflight == 0``) AND at least one send actually
    recorded a completion event AND all such events have fired.  An entry with
    no recorded events has not been sent yet, so it is NOT reapable via
    completion (only eviction / cleanup can reclaim it).
    """
    if entry.inflight > 0 or not entry.done_events:
        return False
    return all(_event_fired(ev) for ev in entry.done_events)


@dataclass
class SendBufferEntry:
    """One registered GPU send buffer + its per-receiver send bookkeeping.

    Attributes:
        transfer_id: Opaque per-transfer key (``<rollout_idx>:<uuid>``).
        buffer: The GPU tensor to send (held alive by this reference).
        ready_event: CUDA event recorded on the compute stream once the
            trajectory tensor is populated.  The sender's transfer stream
            waits on it before ``nccl_send``.  ``None`` on CPU / when the
            producer packed synchronously.
        nbytes: Payload size for capacity accounting / logging.
        inflight: Count of sends that have STARTED but not yet recorded a
            completion event.  A ``transfer_id`` may be sent to several
            receivers; the buffer must not be reaped while any send is
            mid-flight.
        done_events: One CUDA event per COMPLETED-enqueue send; the buffer
            must stay alive until every one fires.
        created_at: Monotonic timestamp (for oldest-first eviction).
    """

    transfer_id: str
    buffer: Any
    ready_event: Any = None
    nbytes: int = 0
    inflight: int = 0
    done_events: List[Any] = field(default_factory=list)
    created_at: float = field(default=0.0)


class SendBufferRegistry:
    """Bounded, thread-safe registry of GPU send buffers.

    Args:
        capacity: Maximum number of live buffers.  ``register`` blocks
            when the registry is full.
        block_timeout: Seconds to wait for a slot before falling back to
            dropping the oldest un-sent buffer (never-wedge guarantee).
            ``0`` disables blocking and drops immediately when full.
        on_free: Optional callback invoked with a :class:`SendBufferEntry`
            when it is evicted / freed, so the owner can free device
            memory or record telemetry.  Must be cheap and not re-enter
            the registry.
    """

    def __init__(
        self,
        *,
        capacity: int = 64,
        block_timeout: float = 30.0,
        on_free: Optional[Callable[[SendBufferEntry], None]] = None,
    ) -> None:
        if capacity < 1:
            capacity = 1
        self._capacity = capacity
        self._block_timeout = max(0.0, block_timeout)
        self._on_free = on_free
        self._entries: "OrderedDict[str, SendBufferEntry]" = OrderedDict()
        self._lock = threading.Lock()
        self._not_full = threading.Condition(self._lock)
        # Observability counters.
        self._n_registered = 0
        self._n_freed = 0
        self._n_evicted = 0

    # ------------------------------------------------------------------
    # Producer API
    # ------------------------------------------------------------------

    def register(
        self,
        transfer_id: str,
        buffer: Any,
        *,
        ready_event: Any = None,
        nbytes: int = 0,
    ) -> SendBufferEntry:
        """Register ``buffer`` under ``transfer_id`` with backpressure.

        If ``transfer_id`` is already present its prior buffer is freed
        first (idempotent overwrite).  If the registry is at capacity, first
        reap already-delivered buffers (whose send has drained), then block
        up to ``block_timeout`` for a concurrent :meth:`free`, then evict the
        oldest buffer so generation is never wedged.  Returns the entry.

        ``on_free`` callbacks run OUTSIDE the lock (they may wait on a send's
        completion event), so a slow-draining send never blocks the registry.
        """
        to_free: List[SendBufferEntry] = []
        with self._not_full:
            existing = self._entries.pop(transfer_id, None)
            if existing is not None:
                to_free.append(existing)  # idempotent overwrite

            if len(self._entries) >= self._capacity:
                # Retire delivered buffers first: a healthy run drains via
                # completion here, so steady state never blocks/evicts.
                to_free.extend(self._reap_completed_locked())
            if len(self._entries) >= self._capacity:
                to_free.extend(self._make_room_locked())

            entry = SendBufferEntry(
                transfer_id=transfer_id,
                buffer=buffer,
                ready_event=ready_event,
                nbytes=nbytes,
                created_at=time.monotonic(),
            )
            self._entries[transfer_id] = entry
            self._n_registered += 1
        for old in to_free:
            self._invoke_on_free(old)
        return entry

    def _reap_completed_locked(self) -> List[SendBufferEntry]:
        """Pop entries whose send has drained (``done_event`` fired).

        Caller holds the lock.  Returns the popped entries for the caller to
        free OUTSIDE the lock.  Signals ``register`` waiters that slots opened.
        """
        done_ids = [tid for tid, entry in self._entries.items() if _send_drained(entry)]
        reaped = [self._entries.pop(tid) for tid in done_ids]
        self._n_freed += len(reaped)
        if reaped:
            self._not_full.notify_all()
        return reaped

    def _make_room_locked(self) -> List[SendBufferEntry]:
        """Block for a free slot; fall back to oldest-first eviction.

        Caller holds the lock.  Returns entries to free OUTSIDE the lock.
        """
        to_free: List[SendBufferEntry] = []
        deadline = time.monotonic() + self._block_timeout
        while len(self._entries) >= self._capacity:
            # A send may have completed while we waited -> reap before evicting.
            reaped = self._reap_completed_locked()
            if reaped:
                to_free.extend(reaped)
                continue
            remaining = deadline - time.monotonic()
            if self._block_timeout > 0 and remaining > 0:
                # Wait for a concurrent free() to signal capacity.
                self._not_full.wait(timeout=remaining)
                continue
            # Timed out (or blocking disabled): evict oldest buffer so the
            # producer never deadlocks on an idle consumer.
            old_id, old_entry = self._entries.popitem(last=False)
            self._n_evicted += 1
            logger.warning(
                "[SendBufferRegistry] Capacity %d reached; evicting oldest "
                "buffer transfer_id=%s (nbytes=%d). Consumer is not draining "
                "fast enough.",
                self._capacity,
                old_id,
                old_entry.nbytes,
            )
            to_free.append(old_entry)
        return to_free

    # ------------------------------------------------------------------
    # Sender / cleanup API
    # ------------------------------------------------------------------

    def get(self, transfer_id: str) -> Optional[SendBufferEntry]:
        """Return the entry for ``transfer_id`` (or ``None`` if recycled).

        Read-only peek that does NOT lease the buffer -- a sender that intends
        to transmit must use :meth:`acquire` instead, or the entry can be
        evicted / freed out from under it before the send is recorded.
        """
        with self._lock:
            return self._entries.get(transfer_id)

    def acquire(self, transfer_id: str) -> Optional[SendBufferEntry]:
        """Look up ``transfer_id`` and atomically LEASE it for a send.

        Increments ``inflight`` under the lock, so from this point the entry
        cannot be reaped, evicted, or freed out from under the caller until the
        lease is balanced.  This closes the use-after-free window that a bare
        :meth:`get` followed by a later ``mark_inflight`` left open: between the
        two, capacity eviction (:meth:`_make_room_locked`) or a discard
        :meth:`free` could pop the entry and release its GPU storage while the
        send still pointed at it.

        The caller MUST balance every non-``None`` acquire with exactly one
        :meth:`add_done_event` (send enqueued -> event recorded) or
        :meth:`abandon_inflight` (send never recorded an event).  Returns
        ``None`` if the id was already recycled.
        """
        with self._lock:
            entry = self._entries.get(transfer_id)
            if entry is not None:
                entry.inflight += 1
            return entry

    def add_done_event(self, entry: SendBufferEntry, done_event: Any) -> None:
        """Record a send-complete event and release this send's lease.

        Operates on the entry OBJECT (not a dict lookup) so it correctly
        balances the :meth:`acquire` lease even if the entry was concurrently
        evicted / freed out of the registry mid-send.  The buffer's owner
        (:meth:`SendBufferRegistry.free` / eviction -> ``on_free``, or the
        reaper) then observes the recorded event and waits on it before
        releasing the GPU storage.  The buffer stays alive until every
        recorded event fires.
        """
        with self._lock:
            entry.done_events.append(done_event)
            if entry.inflight > 0:
                entry.inflight -= 1

    def abandon_inflight(self, entry: SendBufferEntry) -> None:
        """Release a send's :meth:`acquire` lease WITHOUT recording an event.

        For a send that failed, or was never attempted after acquiring (e.g. a
        renegotiation abort), so a dropped send never pins the buffer forever.
        Operates on the entry object (see :meth:`add_done_event`).
        """
        with self._lock:
            if entry.inflight > 0:
                entry.inflight -= 1

    def free(self, transfer_id: str) -> bool:
        """Release the buffer for ``transfer_id``.  Idempotent.

        Returns ``True`` if an entry was freed, ``False`` if it was
        absent / already freed.  Signals :meth:`register` waiters that a
        slot is now available.
        """
        with self._not_full:
            entry = self._entries.pop(transfer_id, None)
            if entry is None:
                return False
            self._n_freed += 1
            self._not_full.notify()
        # Release OUTSIDE the lock: _on_buffer_free may wait on the (bounded)
        # send-complete event, and we must not hold the registry lock while a
        # send drains.
        self._invoke_on_free(entry)
        return True

    def clear(self) -> None:
        """Free every buffer (teardown)."""
        with self._not_full:
            entries = list(self._entries.values())
            self._entries.clear()
            self._not_full.notify_all()
        for entry in entries:
            self._invoke_on_free(entry)

    def _invoke_on_free(self, entry: SendBufferEntry) -> None:
        if self._on_free is None:
            return
        try:
            self._on_free(entry)
        except Exception as exc:  # pragma: no cover - callback bug isolation
            logger.warning(
                "[SendBufferRegistry] on_free callback raised %s for %s; continuing",
                type(exc).__name__,
                entry.transfer_id,
            )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, transfer_id: str) -> bool:
        with self._lock:
            return transfer_id in self._entries

    @property
    def capacity(self) -> int:
        return self._capacity

    def stats(self) -> dict:
        with self._lock:
            return {
                "live": len(self._entries),
                "capacity": self._capacity,
                "registered": self._n_registered,
                "freed": self._n_freed,
                "evicted": self._n_evicted,
            }

    def transfer_ids(self) -> List[str]:
        with self._lock:
            return list(self._entries.keys())
