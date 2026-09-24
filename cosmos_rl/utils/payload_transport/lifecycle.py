# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded ownership of backend teardown, including blocking native calls."""

import threading
import math
from collections.abc import Callable


_creation_lock = threading.Lock()


def get_close_operation(owner, attribute: str, cleanup: Callable[[], None]):
    """Install one operation even when multiple shutdown callers arrive."""
    with _creation_lock:
        operation = getattr(owner, attribute, None)
        if operation is None:
            operation = TransportClose(cleanup)
            setattr(owner, attribute, operation)
        return operation


class TransportClose:
    """Run a teardown sequence once; retain its ownership on timeout.

    A timeout does not cancel native code or authorize freeing its buffers.
    The same operation can be waited on again. A failed operation stays failed;
    callers must not reattach or pretend shutdown succeeded.
    """

    def __init__(self, cleanup: Callable[[], None]):
        self._cleanup = cleanup
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._thread = None
        self._error = None

    @property
    def completed(self) -> bool:
        return self._done.is_set() and self._error is None

    def close(self, timeout: float = 5.0) -> None:
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("Transport close timeout must be finite and nonnegative")
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="payload-transport-close", daemon=True
                )
                try:
                    self._thread.start()
                except BaseException as error:
                    # No cleanup ran, so resources remain owned. Preserve this
                    # failure for subsequent callers instead of timing out on
                    # an event that can never be signalled.
                    self._error = error
                    self._done.set()
        if not self._done.wait(timeout):
            raise TimeoutError(
                "Payload transport teardown is still running; resources remain "
                "owned and reattachment is forbidden"
            )
        if self._error is not None:
            raise RuntimeError("Payload transport teardown failed") from self._error

    def _run(self):
        try:
            self._cleanup()
        except BaseException as error:
            self._error = error
        finally:
            self._done.set()
