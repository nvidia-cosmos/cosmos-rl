# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""A bounded, ordered controller outbox, independent of the HTTP event loop.

Controller state is sealed before submission. Only Redis I/O runs on the writer
thread: it never mutates membership, reports or accounting. Submission is not
delivery; uncertain publication is terminal, never silently retried as new work.
Worker-side RedisStreamHandler remains synchronous.
"""

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import threading

from cosmos_rl.utils.redis_publication import PublicationPlan


class ControllerPublisher:
    def __init__(self, transport, *, on_failure, max_pending=128):
        if max_pending <= 0:
            raise ValueError("Controller outbox capacity must be positive")
        self.transport = transport
        self.on_failure = on_failure
        self.max_pending = max_pending
        self._writer = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="redis-outbox"
        )
        self._lock = threading.Lock()
        self._pending = deque()
        self._failure = None
        self._closed = False

    def __getattr__(self, name):
        return getattr(self.transport, name)

    def _fail(self, error):
        with self._lock:
            if self._failure is not None:
                return
            self._failure = error
        # The controller must not keep serving after a potentially partial
        # dispatch. Production supplies its terminal-exit callback; tests can
        # observe it without terminating their fixture process.
        self.on_failure(error)

    def _finished(self, future):
        if not future.cancelled():
            error = future.exception()
            if error is not None:
                self._fail(error)

    def publish_plan(self, plan):
        try:
            with self._lock:
                if self._closed or self._failure is not None:
                    raise RuntimeError(
                        "Controller outbox is closed or terminally failed"
                    )
                while self._pending and self._pending[0].done():
                    self._pending.popleft()
                if len(self._pending) >= self.max_pending:
                    raise RuntimeError("Controller publication outbox is full")
                # One thread preserves stream/command ordering across plans.
                # The plan's deadline already includes time spent in this queue.
                future = self._writer.submit(self.transport.publish_plan, plan)
                self._pending.append(future)
            future.add_done_callback(self._finished)
            return future
        except Exception as error:
            self._fail(error)
            raise

    def publish_command(self, data, stream_name):
        return self.publish_plan(
            PublicationPlan.create(((stream_name + "_command", "command", data),))
        )

    def publish_rollout(self, data, stream_name):
        return self.publish_plan(
            PublicationPlan.create(((stream_name + "_rollout", "rollout", data),))
        )

    async def flush(self, timeout_s=65):
        with self._lock:
            pending = tuple(self._pending)
        try:
            if pending:
                await asyncio.wait_for(
                    asyncio.gather(*(asyncio.wrap_future(f) for f in pending)),
                    timeout=timeout_s,
                )
            if self._failure is not None:
                raise RuntimeError("Controller publication failed") from self._failure
        except Exception as error:
            self._fail(error)
            raise

    async def close(self, timeout_s=65):
        with self._lock:
            self._closed = True
        try:
            await self.flush(timeout_s)
        finally:
            # Unknown native completion is owned until terminal process exit;
            # executor shutdown is not a claim that cancellation stopped I/O.
            self._writer.shutdown(wait=False, cancel_futures=True)
