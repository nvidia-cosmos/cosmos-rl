# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Own worker producer threads until they stop, or terminate their process.

A failed producer cannot leave its consumer waiting for an impossible batch.
Unknown thread completion cannot authorize native-resource teardown. This is
local process containment, not distributed recovery or cross-node job abort.
"""

import math
import faulthandler
import os
import threading
import time


def fail_worker(reason):
    """Do not enter native cleanup, Python logging locks or atexit handlers."""
    try:
        os.set_blocking(2, False)
        os.write(2, f"[Worker FATAL] {reason[:2048]}\n".encode())
        faulthandler.dump_traceback(file=2, all_threads=True)
    finally:
        os._exit(1)


class OwnedWorkerThreads:
    def __init__(self, stop_event, *, fatal=fail_worker):
        self.stop_event = stop_event
        self.fatal = fatal
        self._lock = threading.Lock()
        self._threads = []
        self._error = None
        self._closed = False

    def _failed(self, name, error):
        with self._lock:
            if self._error is not None:
                return
            self._error = error
            self._closed = True
        self.stop_event.set()
        self.fatal(f"background task {name} failed: {type(error).__name__}: {error}")

    def check(self):
        if self._error is not None:
            raise RuntimeError("Worker background task failed") from self._error

    def start(self, name, target, args=()):
        def run():
            try:
                target(*args)
            except BaseException as error:
                self._failed(name, error)

        # Register and start atomically with closing admission. The object, not
        # Thread.start()'s return value, is the owned join handle.
        with self._lock:
            self.check()
            if self._closed or self.stop_event.is_set():
                raise RuntimeError("Worker thread admission is closed")
            thread = threading.Thread(target=run, name=name, daemon=True)
            self._threads.append(thread)
            try:
                thread.start()
            except BaseException:
                self._threads.remove(thread)
                raise
            return thread

    def close(self, timeout=45.0):
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError(
                "Worker thread stop timeout must be finite and nonnegative"
            )
        deadline = time.monotonic() + timeout
        with self._lock:
            self._closed = True
            threads = tuple(self._threads)
        self.stop_event.set()
        for thread in threads:
            if thread is threading.current_thread():
                self._failed(
                    thread.name, RuntimeError("cannot join current worker thread")
                )
                break
            thread.join(max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                self._failed(
                    thread.name, TimeoutError("thread still owns worker resources")
                )
                break
        self.check()


def start_worker_thread(worker, name, target, args=()):
    """Called by the serialized worker startup/main-loop thread."""
    owner = getattr(worker, "_owned_worker_threads", None)
    if owner is None:
        owner = worker._owned_worker_threads = OwnedWorkerThreads(
            worker.shutdown_signal
        )
    return owner.start(name, target, args)


def stop_worker_threads(worker):
    owner = getattr(worker, "_owned_worker_threads", None)
    if owner is not None:
        owner.close(float(os.getenv("COSMOS_WORKER_THREAD_STOP_TIMEOUT_S", "45")))
