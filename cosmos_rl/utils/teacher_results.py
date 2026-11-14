# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Local teacher-result ownership, including cancellation of skipped batches."""

from collections import deque
import logging
import threading
import time

from cosmos_rl.utils.teacher_channel import deadline_after


class TeacherResultInbox:
    def __init__(self, max_bytes=None):
        self._condition = threading.Condition()
        self._results = {}
        self._max_bytes = max_bytes
        self._bytes = 0
        self._pending = deque()
        self._waiting = set()
        self._overflow = set()
        self._retired = deque()
        self._prefetch_full = False

    def admit(self, identity):
        with self._condition:
            if identity not in self._results:
                self._results[identity] = None
                self._pending.append(identity)

    def pending(self, identity):
        with self._condition:
            return (
                identity in self._results
                and self._results[identity] is None
                and identity not in self._overflow
            )

    def next_pending(self):
        """Round-robin reads; a waiting update takes priority over prefetch."""
        with self._condition:
            if self._prefetch_full and not self._waiting:
                return None
            for _ in range(len(self._pending)):
                identity = self._pending.popleft()
                if not self.pending(identity):
                    continue
                self._pending.append(identity)
                if not self._waiting or identity in self._waiting:
                    return identity
        return None

    def complete(self, identity, value):
        with self._condition:
            if identity not in self._results:
                return False  # A skipped/retired update cannot recreate cache entries.
            previous = self._results[identity]
            size = len(value) - (len(previous) if previous is not None else 0)
            if self._max_bytes is not None and self._bytes + size > self._max_bytes:
                # Future replies remain in Redis, still charged to its budget.
                # Discard only their local copies so they cannot strand the
                # current update behind a full prefetch cache.
                if identity in self._waiting:
                    for other, body in self._results.items():
                        if other not in self._waiting and body is not None:
                            self._bytes -= len(body)
                            self._results[other] = None
                            self._pending.append(other)
                            if self._bytes + size <= self._max_bytes:
                                break
                if self._bytes + size > self._max_bytes:
                    self._prefetch_full = True
                    if identity in self._waiting or len(value) > self._max_bytes:
                        self._overflow.add(identity)
                        self._condition.notify_all()
                    return False
            self._results[identity] = value
            self._bytes += size
            self._overflow.discard(identity)
            if identity in self._pending:
                self._pending.remove(identity)
            self._condition.notify_all()
            return True

    def wait(self, identities, timeout):
        deadline = deadline_after(timeout)
        with self._condition:
            self._waiting = set(identities)
            try:
                while any(self.pending(identity) for identity in identities):
                    if self._waiting & self._overflow:
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(remaining)
                if self._waiting & self._overflow:
                    logging.getLogger(__name__).warning(
                        "Teacher update exceeds the learner inbox byte budget (%s)",
                        self._max_bytes,
                    )
                    return [None for _ in identities]
                return [self._results.get(identity) for identity in identities]
            finally:
                self._waiting.clear()

    def retire(self, identities):
        with self._condition:
            for identity in identities:
                if identity in self._results:
                    body = self._results.pop(identity)
                    self._bytes -= len(body) if body is not None else 0
                    self._overflow.discard(identity)
                    self._retired.append(identity)
            self._prefetch_full = False
            self._condition.notify_all()

    def release_retired(self, acknowledge):
        """Release channel capacity only after local retirement; retry lost IO."""
        with self._condition:
            count = len(self._retired)
        # Drain the retired window without a polling sleep per acknowledgment.
        # A failed IO yields back to target fetching and is retried next poll.
        for _ in range(count):
            with self._condition:
                if not self._retired:
                    return
                identity = self._retired.popleft()
            try:
                acknowledge(identity)
            except Exception:
                with self._condition:
                    self._retired.append(identity)
                raise
