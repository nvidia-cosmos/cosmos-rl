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

    def __init__(self, values, budget, nbytes, *, rejected_keys=()):
        super().__init__(values)
        # Explicit safe outcomes, scoped to this batch's lifetime. Absence from
        # the values dictionary alone is not proof that a transfer was rejected.
        self.rejected_keys = set(rejected_keys)
        self.budget = budget
        self.nbytes = nbytes
        self.released = False
        self.consumer_owned = False
        # The consumer can reshape, pop, or replace dictionary entries. Lease
        # ownership must survive those mutations until all readers complete.
        storages = {}
        for payload in self.values():
            for tensor in payload.values():
                storage = tensor.untyped_storage()
                storages[(str(tensor.device), storage.data_ptr())] = storage
        self._storages = tuple(storages.values())

    def claim_consumer(self):
        """Mark an actual handoff, not merely a queued or prepared result."""
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
        if self.released:
            return
        import torch

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
        with self.budget.condition:
            if self.consumer_owned:
                self.budget.consumer_bytes -= self.nbytes
            self.budget.attribute(decoded=-self.nbytes)
            self.budget.release(self.nbytes)
        self.released = True


def storage_bytes(payloads):
    """Count full unique storage, including backing storage of truncated views."""
    storages = {}
    for payload in payloads:
        for tensor in payload.values():
            storage = tensor.untyped_storage()
            storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())
