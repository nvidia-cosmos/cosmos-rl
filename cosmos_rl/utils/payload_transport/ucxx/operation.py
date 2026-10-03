# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Own UCXX/native-copy resources until completion, never until cancellation.

UCXX 0.47's request waiter awaits a Python future (or polls native completion).
Cancelling that coroutine destroys the request wrapper, whose destructor requests
native cancellation. Neither action is proof that the request stopped using its
buffer. Keep the uncancelled task and its operands alive on the terminal path.
"""

import asyncio
import math

from cosmos_rl.utils.transport_failure import (
    TransportDeadline,
    TransportUnusableError,
    fail_transport,
)


# Process-lifetime ownership on uncertain completion. Never run native cleanup
# from the watchdog, and never interpret a caller catching the error as recovery.
_TERMINAL_OPERATIONS = []


class UCXXOperation:
    def __init__(self, timeout, context, *, owners=(), on_failure=None):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("UCXX operation timeout must be finite and positive")
        self.owners = list(owners)
        self.failure = None
        self._on_failure = on_failure
        self.deadline = TransportDeadline(timeout, context, fatal=self._terminal)

    def _terminal(self, reason):
        if self.failure is None:
            self.failure = reason
            _TERMINAL_OPERATIONS.append(self)
            if self._on_failure is not None:
                self._on_failure(reason)
            fail_transport(reason)

    def fail(self, reason):
        self.deadline.fail(reason)
        raise TransportUnusableError(self.failure or reason)

    async def wait(self, function, *args):
        """Wait without propagating cancellation into a native request waiter."""
        try:
            remaining = self.deadline.remaining_ms() / 1000
            # Retain operands BEFORE invoking even the synchronous part of the
            # function; that part can block inside native code as well.
            self.owners.extend((function, *args))
            task = asyncio.ensure_future(function(*args))
            self.owners.append(task)
            done, _ = await asyncio.wait((task,), timeout=remaining)
            if not done:
                self.fail("native request completion timed out")
            self.deadline.remaining_ms()
            return task.result()
        except (asyncio.CancelledError, TimeoutError, TransportUnusableError) as error:
            self.fail(f"native request completion uncertain: {error}")

    def complete(self):
        try:
            self.deadline.close()
        except TransportUnusableError as error:
            self._terminal(str(error))
            raise
        if self.failure is not None:
            raise TransportUnusableError(self.failure)
        self.owners.clear()
