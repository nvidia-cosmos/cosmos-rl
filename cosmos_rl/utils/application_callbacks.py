# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Role-owned application callbacks with explicit borrowed/telemetry boundaries."""

from dataclasses import dataclass, replace
import inspect
import os
from multiprocessing.util import Finalize
from queue import Empty, Full, Queue
import threading
import time
from typing import Callable, Literal

from cosmos_rl.utils.logging import logger


@dataclass(frozen=True)
class CallbackContext:
    role: str
    replica_name: str | None = None
    global_rank: int | None = None
    is_validation: bool = False
    process_id: int | None = None
    worker_id: str | None = None

    def current_worker(self):
        thread = threading.current_thread()
        return replace(
            self,
            process_id=os.getpid(),
            worker_id=f"{thread.name}:{id(thread)}",
        )


@dataclass(frozen=True)
class CallbackFactory:
    """Create a callable once in each executing process/thread's context."""

    create: Callable[[CallbackContext], Callable]


@dataclass(frozen=True)
class CallbackRegistration:
    handler: Callable | CallbackFactory
    policy: Literal["required", "best_effort"] = "required"
    queue_capacity: int = 64

    def __post_init__(self):
        if self.policy not in ("required", "best_effort"):
            raise ValueError("Callback policy must be required or best_effort")
        if type(self.queue_capacity) is not int or self.queue_capacity < 1:
            raise ValueError("Callback queue capacity must be a positive integer")
        if not isinstance(self.handler, CallbackFactory) and not callable(self.handler):
            raise TypeError("A callback must be callable or a CallbackFactory")


def _invoke_sync(callback, *args, **kwargs):
    result = callback(*args, **kwargs)
    if inspect.isawaitable(result):
        if inspect.iscoroutine(result):
            result.close()
        raise TypeError("Application callbacks must complete synchronously")
    return result


def _create_handler(handler, context):
    callback = (
        _invoke_sync(handler.create, context)
        if isinstance(handler, CallbackFactory)
        else handler
    )
    if not callable(callback):
        raise TypeError("CallbackFactory must return a callable")
    return callback


class _BestEffortWorker:
    def __init__(self, name, registration, context):
        self.name = name
        self.registration = registration
        self.context = context
        self.queue = Queue(maxsize=registration.queue_capacity)
        self.stopping = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name=f"ApplicationCallback:{name}", daemon=True
        )
        self.thread.start()

    def submit(self, args, kwargs):
        if self.stopping.is_set() or not self.thread.is_alive():
            logger.warning(
                "Dropping best-effort callback %s: worker unavailable", self.name
            )
            return
        try:
            self.queue.put_nowait((args, kwargs))
        except Full:
            logger.warning("Dropping best-effort callback %s: queue full", self.name)

    def _run(self):
        callback = None
        try:
            # Factory and handler code both run away from the controller lock.
            callback = _create_handler(
                self.registration.handler, self.context.current_worker()
            )
        except Exception:
            logger.exception("Best-effort callback %s initialization failed", self.name)
        try:
            while not self.stopping.is_set() or not self.queue.empty():
                try:
                    args, kwargs = self.queue.get(timeout=0.05)
                except Empty:
                    continue
                try:
                    if callback is not None:
                        _invoke_sync(callback, *args, **kwargs)
                except Exception:
                    logger.exception(
                        "Best-effort callback %s failed; event dropped", self.name
                    )
                finally:
                    self.queue.task_done()
                    del args, kwargs
        finally:
            close = getattr(callback, "close", None)
            if callable(close):
                try:
                    _invoke_sync(close)
                except Exception:
                    logger.exception("Best-effort callback %s close failed", self.name)

    def close(self, timeout_s):
        self.stopping.set()
        self.thread.join(timeout_s)
        if self.thread.is_alive():
            logger.warning(
                "Best-effort callback %s did not drain before close", self.name
            )


class CallbackRegistry:
    """Bind hooks to a role and lazily to each executing worker.

    Plain callables are required and may be shared across execution threads;
    applications must make them thread-safe. Factories create independent state
    once per hook/process/thread. Factories and registered callback objects must
    be serializable when the enclosing launcher uses process workers.

    Borrowed calls are synchronous and required. Best-effort calls accept only
    explicitly detached arguments and run on a bounded daemon queue. Owners must
    quiesce required callers before close; optional telemetry never owns payload
    or transport lifetimes. Required callable.close(), when provided, runs on the
    closing owner thread; best-effort close runs on its telemetry worker.
    A process finalizer is registered on first use, including after fork, to
    close child-owned instances at worker exit. Finalizer errors are reported by
    multiprocessing after tasks have completed; strict per-task payload cleanup
    must run in the synchronous reader callback, never in this finalizer.
    """

    def __init__(self, hook_fns, context):
        self.hook_fns = dict(hook_fns or {})
        self.context = context
        self._lock = threading.RLock()
        self._handlers = {}
        self._closed = False
        self._close_error = None
        self._finalizers = {}

    def has(self, name):
        return self.hook_fns.get(name) is not None

    def validate(self, name, *, borrowed=True, required=False):
        value = self.hook_fns.get(name)
        if value is None:
            return None
        registration = (
            value
            if isinstance(value, CallbackRegistration)
            else CallbackRegistration(value)
        )
        if (borrowed or required) and registration.policy != "required":
            raise ValueError(f"Callback {name} must be required for this boundary")
        return registration

    def invoke(self, name, *args, borrowed=True, required=False, **kwargs):
        registration = self.validate(name, borrowed=borrowed, required=required)
        if registration is None:
            return None
        key = (os.getpid(), threading.current_thread(), name)
        with self._lock:
            if self._closed:
                raise RuntimeError("Application callback registry is closed")
            if os.getpid() not in self._finalizers:
                # Constructor/setup can run before fork. Register in the process
                # that actually binds handlers so child-owned state is retired.
                self._finalizers[os.getpid()] = Finalize(
                    None, self.close, exitpriority=10
                )
            handler = self._handlers.get(key)
            if handler is None:
                handler = (
                    _BestEffortWorker(name, registration, self.context)
                    if registration.policy == "best_effort"
                    else _create_handler(
                        registration.handler, self.context.current_worker()
                    )
                )
                self._handlers[key] = handler
        if registration.policy == "best_effort":
            handler.submit(args, kwargs)
            return None
        return _invoke_sync(handler, *args, **kwargs)

    def close(self, timeout_s=1.0):
        if timeout_s < 0:
            raise ValueError("Callback close timeout must be non-negative")
        with self._lock:
            if self._closed:
                if self._close_error is not None:
                    raise self._close_error
                return
            self._closed = True
            finalizer = self._finalizers.pop(os.getpid(), None)
            if finalizer is not None:
                finalizer.cancel()
            handlers = [
                handler
                for (pid, _, _), handler in self._handlers.items()
                if pid == os.getpid()
            ]
        first_error = None
        deadline = time.monotonic() + timeout_s
        seen = set()
        for handler in handlers:
            if id(handler) in seen:
                continue
            seen.add(id(handler))
            try:
                if isinstance(handler, _BestEffortWorker):
                    handler.close(max(0, deadline - time.monotonic()))
                else:
                    close = getattr(handler, "close", None)
                    if callable(close):
                        _invoke_sync(close)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            self._close_error = first_error
            raise first_error
