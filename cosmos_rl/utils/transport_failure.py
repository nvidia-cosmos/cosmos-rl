# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend-neutral terminal transport failure, not ordinary transfer rejection.

Use only when native completion cannot be established and continuing could reuse
live storage or distributed state. No native cleanup runs on this path.
"""

import os
import math
import threading
import time
from typing import NoReturn


FATAL_TRANSPORT_EXIT_CODE = 86


class TransportUnusableError(RuntimeError):
    """The backend cannot establish completion; this worker must not continue."""


class TransportDeadline:
    """One accepted operation's budget, including queue and native completion.

    The timer never calls native cleanup or takes an application lock. A native
    call can hang before returning a handle; polling only in that caller cannot
    enforce its deadline. Timeout is terminal, not permission to recycle storage.
    """

    def __init__(self, timeout_s, context, *, fatal=None, clock=time.monotonic):
        if not math.isfinite(timeout_s):
            raise ValueError("Transport deadline must be finite")
        self._clock = clock
        self.deadline = clock() + max(0, timeout_s)
        self.context = context
        self._fatal = fatal or fail_transport
        self._lock = threading.Lock()
        self._state = "active"
        self._timer = threading.Timer(max(0, timeout_s), self._expire)
        self._timer.daemon = True
        self._timer.start()

    def _expire(self):
        self.fail("accepted operation deadline expired")

    @property
    def active(self):
        return self._state == "active"

    def fail(self, reason):
        with self._lock:
            if self._state != "active":
                return
            self._state = "expired"
        self._fatal(f"{self.context}: {reason}")

    def remaining_ms(self):
        remaining = self.deadline - self._clock()
        if remaining <= 0 or self._state == "expired":
            raise TransportUnusableError(f"{self.context}: operation deadline expired")
        return max(1, math.ceil(remaining * 1000))

    def close(self):
        with self._lock:
            expired = self._state == "expired" or self._clock() >= self.deadline
            self._state = "closed"
        self._timer.cancel()
        if expired:
            raise TransportUnusableError(f"{self.context}: completion after deadline")

    def wait_event(self, event, *, check_peer=None):
        while event is not None and not event.query():
            self.remaining_ms()
            if check_peer is not None:
                check_peer()
            time.sleep(0.001)
        self.remaining_ms()


def fail_transport(context: str) -> NoReturn:
    # Avoid logging-handler locks. Even a diagnostic write failure must exit.
    try:
        os.set_blocking(2, False)
        os.write(
            2,
            f"[Transport FATAL] {context[:2048]}; exiting without native cleanup\n".encode(),
        )
    finally:
        os._exit(FATAL_TRANSPORT_EXIT_CODE)
