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

"""UCXX-based payload-transfer server / client.

UCX (and its Python binding UCXX) provides unified communication that
auto-optimizes the underlying transport:

* Same-node: shared-memory transport (~100 GB/s)
* Cross-node: RDMA (~12.5 GB/s) or TCP fallback

This module wraps :class:`SharedRingBuffer` with a UCXX server that
lets remote trainers read slot data directly from a worker's CPU
buffer without going through Redis.

The ``ucxx-cu12`` (or platform-equivalent) extra is **optional**.  When
it is not installed, importing this module still succeeds and
:data:`UCXX_AVAILABLE` is set to ``False``; attempts to start a server
or client will raise an explicit ``RuntimeError`` rather than failing
with an import error in random places.
"""

import asyncio
import collections
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.payload_transport.ucxx.shared_buffer import (
    BufferConfig,
    BufferMetrics,
    SharedRingBuffer,
    SlotError,
    SlotState,
)

# Optional UCXX import - graceful handling if not available
# Package: pip install ucxx-cu12 (for CUDA 12)
try:
    import ucxx

    UCXX_AVAILABLE = True
except ImportError:
    ucxx = None
    UCXX_AVAILABLE = False
    logger.warning(
        "[UCXXBuffer] ucxx not available. Install with: pip install ucxx-cu12. "
        "Cross-node UCXX will not work."
    )


class StaleSlotError(RuntimeError):
    """Raised when a client reads a slot that has already been consumed."""

    pass


# Errors for which retrying on a *different* server port can plausibly
# help.  These are transport / connectivity failures: the underlying
# server thread or its endpoint is unhealthy, but a sibling thread on
# the same worker reads the same SHM slot and is independent.
#
# Explicitly excluded:
#   * ``StaleSlotError`` -- the slot is gone everywhere; rotating ports
#     cannot resurrect it.
#   * ``RuntimeError`` from server status=2 ("Remote read failed: ...")
#     or status=unknown -- the server already replied; the failure is
#     in the data path or protocol, not the connection.
_PORT_ROTATABLE_ERRORS = frozenset(
    {
        "UCXXCanceledError",
        "UCXXConnectionResetError",
        "UCXXCloseError",
        "TimeoutError",
    }
)


# Cooldown after a ``(worker_ip, port)`` emits a transport-class
# failure before that port is re-eligible for rotation in
# :meth:`UCXXClient.read`.
#
# Picked at 30 s so that a typical trainer's ~1.5-fetch/sec stream
# spends ~50 fetches diverted from a flaky port before re-probing
# it -- long enough to ride through a transient network blip, short
# enough to recover quickly when the port heals.  The cost is a
# slightly less even load distribution while a port is quarantined;
# the benefit is that one flaky server thread cannot silently
# consume wall time.
_PORT_QUARANTINE_SEC = 30.0


# Maximum age of a pooled :class:`UCXXClient` endpoint before it is
# preemptively closed and replaced with a fresh connection on the
# next checkout.  Must be **strictly less than** the server-side
# handler idle eviction window
# (``UCXXBuffer._HANDLER_MAX_IDLE_CYCLES * _HANDLER_RECV_TIMEOUT``,
# currently 24 * 5 s = 120 s) so that we never hand out a pooled
# endpoint whose server-side handler has already exited.
#
# Rationale: without this, a long pause between fetches lets the
# server kill its handler while the client's pool still holds the
# endpoint.  The next read on that endpoint fails as a
# transport-class error, which would otherwise quarantine an
# otherwise-healthy port for ``_PORT_QUARANTINE_SEC``.  Preemptive
# replacement absorbs the common case (steady-state idle then
# resumed traffic) without any protocol change.  A microsecond-wide
# race remains where the server kills the handler between this
# check and the actual send -- caught by the existing port-rotation
# fallback at the cost of one transient quarantine, which the
# current data shows is acceptable.
_POOL_ENDPOINT_MAX_AGE_S = 100.0


@dataclass
class UCXXBufferConfig:
    """Configuration for UCXXBuffer."""

    # SharedRingBuffer config
    buffer_name: str = ""
    max_entries: int = 100
    entry_size_bytes: int = 65536
    schema: List[Any] = None  # List of TensorSpec

    # UCXX server config
    port: int = 13337
    n_server_threads: int = 4

    def __post_init__(self):
        if self.schema is None:
            self.schema = []

    def to_buffer_config(self) -> BufferConfig:
        """Convert to BufferConfig for SharedRingBuffer."""
        return BufferConfig(
            buffer_name=self.buffer_name,
            max_entries=self.max_entries,
            entry_size_bytes=self.entry_size_bytes,
            schema=self.schema,
        )


class UCXXBuffer:
    """
    CPU ring buffer with UCXX server for remote reads.

    Worker side: Creates a SharedRingBuffer and starts a UCXX listener
    that allows trainers to read slot data remotely.

    The UCXX server handles read requests:
    1. Trainer connects to worker's UCXX server
    2. Trainer sends slot index
    3. Worker reads from local buffer and sends data back
    4. UCX auto-selects transport (shm for same-node, RDMA for cross-node)

    Usage:
        # Worker side
        buffer = UCXXBuffer(config)
        await buffer.start_server()

        slot = buffer.write(rollout_data)
        metadata = buffer.get_metadata(slot)
        # Send metadata via Redis stream...

        # Cleanup
        await buffer.stop_server()
        buffer.close()
    """

    def __init__(self, config: UCXXBufferConfig, create: bool = True):
        """
        Initialize UCXXBuffer.

        Args:
            config: Buffer configuration
            create: If True, create new shared memory; if False, attach to existing
        """
        self.config = config
        self._base_port = config.port
        self._n_threads = max(1, config.n_server_threads)
        self._local_ip = self._get_local_ip()

        # Create underlying SharedRingBuffer
        buffer_config = config.to_buffer_config()
        self._buffer = SharedRingBuffer(buffer_config, create=create)

        # Multi-threaded UCXX server state (one per server thread)
        self._ports: List[int] = []
        self._listeners: List[Any] = []
        self._server_threads: List[threading.Thread] = []
        self._server_loops: List[Optional[asyncio.AbstractEventLoop]] = []
        self._shutdown_flag = threading.Event()
        self._active_endpoints: List[Any] = []
        self._endpoints_lock = threading.Lock()
        self._server_ready_count = 0
        self._server_ready_lock = threading.Lock()
        self._server_ready_event = threading.Event()
        self._handler_tasks_per_thread: List[List[asyncio.Task]] = []
        self._thread_metrics: Dict[str, Dict[str, float]] = {}
        self._thread_metrics_lock = threading.Lock()

        logger.info(
            f"[UCXXBuffer] Initialized '{config.buffer_name}' on {self._local_ip} "
            f"(n_server_threads={self._n_threads})"
        )

    @staticmethod
    def _get_local_ip() -> str:
        """Get local IP for UCXX listener, preferring RDMA interfaces.

        Delegates to mixins._get_local_ip() which checks rdma* interfaces
        first to avoid binding to the management network on IB clusters.
        """
        from .mixins import _get_local_ip

        return _get_local_ip()

    # =========================================================================
    # UCXX Server (Worker Side)
    # =========================================================================

    def start_server(self, timeout: float = 10.0) -> None:
        """Start N UCXX listeners on consecutive ports in background threads.

        This method is synchronous and blocks until all server threads are
        ready.  Each thread gets its own asyncio event loop and UCX worker.

        Args:
            timeout: Timeout in seconds to wait for all threads to start.

        Raises:
            RuntimeError: If UCXX is not available or server fails to start.
        """
        if not UCXX_AVAILABLE:
            raise RuntimeError(
                "UCXX is required for UCXXBuffer server. "
                "Install with: pip install ucxx-cu12"
            )

        if self._server_threads and any(t.is_alive() for t in self._server_threads):
            logger.warning("[UCXXBuffer] Server already running")
            return

        self._shutdown_flag.clear()
        self._server_ready_count = 0
        self._server_ready_event.clear()
        self._ports = []
        self._listeners = [None] * self._n_threads
        self._server_loops = [None] * self._n_threads
        self._handler_tasks_per_thread = [[] for _ in range(self._n_threads)]
        self._server_threads = []

        for i in range(self._n_threads):
            port = self._base_port + i
            t = threading.Thread(
                target=self._run_server_loop,
                args=(i, port),
                daemon=True,
                name=f"UCXXServer-{port}",
            )
            self._server_threads.append(t)
            t.start()

        if not self._server_ready_event.wait(timeout=timeout):
            raise RuntimeError(
                f"UCXX server failed to start {self._n_threads} threads "
                f"within {timeout}s (ports {self._base_port}–"
                f"{self._base_port + self._n_threads - 1})"
            )

        ucx_tls = os.environ.get("UCX_TLS", "(not set)")
        logger.info(
            f"[UCXXBuffer] Server started: {self._n_threads} threads on "
            f"{self._local_ip} ports {self._ports}"
        )
        logger.info(f"[UCXXBuffer] UCX_TLS={ucx_tls}")

    def _run_server_loop(self, thread_idx: int, port: int) -> None:
        """Run the UCXX server event loop in background thread."""
        with self._thread_metrics_lock:
            self._thread_metrics[threading.current_thread().name] = {
                "requests": 0,
                "total_read_ms": 0.0,
                "total_send_ms": 0.0,
            }
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._server_loops[thread_idx] = loop

            loop.run_until_complete(self._async_server_main(thread_idx, port))

        except Exception as e:
            logger.error(f"[UCXXBuffer] Server loop error (thread {thread_idx}): {e}")
            import traceback

            traceback.print_exc()
        finally:
            loop = self._server_loops[thread_idx]
            if loop:
                loop.close()
                self._server_loops[thread_idx] = None

    async def _async_server_main(self, thread_idx: int, port: int) -> None:
        """Async main function for one server thread.

        Each thread has its own event loop and UCX worker so that
        concurrent sends don't block each other (eliminates head-of-line
        blocking).
        """
        try:
            ucxx.init()
        except RuntimeError as e:
            if "already initiated" not in str(e):
                logger.error(f"[UCXXBuffer] Failed to init UCXX: {e}")
                return

        handler_tasks = self._handler_tasks_per_thread[thread_idx]

        def _dispatch(endpoint):
            task = asyncio.get_event_loop().create_task(
                self._handle_connection(endpoint)
            )
            handler_tasks.append(task)

        last_err: Optional[Exception] = None
        bound_port = port
        for attempt in range(self._PORT_RETRY_ATTEMPTS):
            candidate = port + attempt * self._n_threads
            try:
                listener = ucxx.create_listener(_dispatch, port=candidate)
                bound_port = candidate
                self._listeners[thread_idx] = listener
                if candidate != port:
                    logger.warning(
                        f"[UCXXBuffer] Thread {thread_idx}: port {port} busy, "
                        f"bound to {candidate} instead"
                    )
                break
            except Exception as e:
                last_err = e
                logger.debug(
                    f"[UCXXBuffer] Thread {thread_idx}: port {candidate} unavailable: {e}"
                )
        else:
            logger.error(
                f"[UCXXBuffer] Thread {thread_idx}: failed to bind after "
                f"{self._PORT_RETRY_ATTEMPTS} attempts: {last_err}"
            )
            return

        logger.info(f"[UCXXBuffer] Thread {thread_idx} listener on port {bound_port}")

        with self._server_ready_lock:
            self._ports.append(bound_port)
            self._server_ready_count += 1
            if self._server_ready_count >= self._n_threads:
                self._ports.sort()
                self._server_ready_event.set()

        logger.info(
            f"[UCXXBuffer] Server ready in thread {threading.current_thread().name}"
        )

        last_handler_log = time.perf_counter()
        while not self._shutdown_flag.is_set():
            handler_tasks[:] = [t for t in handler_tasks if not t.done()]
            now = time.perf_counter()
            if now - last_handler_log >= 10.0:
                logger.info(
                    f"[UCXXBuffer] Thread {thread_idx}: "
                    f"{len(handler_tasks)} active handlers"
                )
                last_handler_log = now
            # Sleep for a meaningful interval rather than ``sleep(0)``:
            # ``_dispatch`` schedules new handler tasks via the
            # listener callback, so this loop only needs to wake up
            # often enough to reap done tasks and notice
            # ``_shutdown_flag``.  ``sleep(0)`` busy-spins all server
            # threads at 100% CPU even with zero traffic; 50 ms gives
            # bounded shutdown latency and effectively zero idle CPU.
            await asyncio.sleep(0.05)

        with self._endpoints_lock:
            endpoints_to_close = list(self._active_endpoints)
            self._active_endpoints.clear()
        for ep in endpoints_to_close:
            try:
                await ep.close()
            except Exception:
                pass

        for task in handler_tasks:
            if not task.done():
                task.cancel()
        if handler_tasks:
            await asyncio.gather(*handler_tasks, return_exceptions=True)
        handler_tasks.clear()

    _HANDLER_RECV_TIMEOUT = 5.0  # seconds per recv wait cycle
    _HANDLER_MAX_IDLE_CYCLES = 24  # exit after 24 × 5s = 120s idle
    # NB: deliberately no per-send timeout.  ``endpoint.send()`` blocks
    # on UCX flow control as well as on transport health, so an
    # absolute send-time cap conflates "trainer is slow to drain" (a
    # legitimate consequence of prefetch-style consumer scheduling)
    # with "trainer half-closed mid-transfer" (the case we'd want to
    # abort).  Client-side per-call rotation + 5s read_timeout already
    # bounds wedge cost to ~5s of trainer-side latency, and
    # ``_HANDLER_RECV_TIMEOUT`` + ``_HANDLER_MAX_IDLE_CYCLES`` already
    # evict idle handlers.
    _PORT_RETRY_ATTEMPTS = 10

    async def _handle_connection(self, endpoint) -> None:
        """Handle incoming connection from trainer.

        Single-chunk-per-slot protocol:

        1. Receive: ``int64[1] = [slot]``.
        2. Send: status byte (``0`` = ok / ``1`` = stale slot /
           ``2`` = error).
        3. On status=0, send the entire raw SHM slot buffer.

        The handler is the *unique owner* of the slot's ``READING ->
        FREE`` (success) or ``READING -> READY`` (failure) transition
        for this read attempt -- there is no shared cross-chunk state
        to corrupt.  This is the structural property the prior
        multi-chunk ``_SlotReadGuard`` design tried (and failed) to
        provide via reference counting.
        """
        logger.debug("[UCXXBuffer] New connection from trainer")
        with self._endpoints_lock:
            self._active_endpoints.append(endpoint)

        idle_cycles = 0
        try:
            while not self._shutdown_flag.is_set():
                try:
                    slot_buf = np.empty(1, dtype=np.int64)
                    await asyncio.wait_for(
                        endpoint.recv(slot_buf), timeout=self._HANDLER_RECV_TIMEOUT
                    )
                    idle_cycles = 0
                    t_recv_done = time.perf_counter()
                    slot = int(slot_buf[0])

                    if not self._buffer.schema:
                        err = RuntimeError(
                            "Zero-pack protocol requires schema-based buffer"
                        )
                        await self._send_error_response(endpoint, err)
                        logger.warning(
                            f"[UCXXBuffer] Send error for slot {slot}: {err}"
                        )
                        continue

                    try:
                        raw_buf = self._buffer.read_raw(slot)
                    except SlotError as e:
                        await endpoint.send(np.array([1], dtype=np.uint8))
                        write_idx, _, entry_count = self._buffer._read_header()
                        logger.warning(
                            f"[UCXXBuffer] StaleSlot slot={slot} err='{e}' "
                            f"write_idx={write_idx} entry_count={entry_count} "
                            f"thread={threading.current_thread().name}"
                        )
                        continue

                    # ``read_raw`` succeeded -> slot is now in READING
                    # state, owned by this handler until we either
                    # mark_consumed (success) or release_reading
                    # (failure).  Both outcomes happen exactly once.
                    t_read_done = time.perf_counter()
                    sent_ok = False
                    try:
                        await endpoint.send(np.array([0], dtype=np.uint8))
                        await endpoint.send(raw_buf)
                        sent_ok = True
                    except Exception as e:
                        # Best-effort error report to the client; the
                        # ``finally`` block does the slot-state cleanup
                        # so it runs whether or not the error report
                        # itself succeeds.
                        await self._send_error_response(endpoint, e)
                        logger.warning(f"[UCXXBuffer] Send error for slot {slot}: {e}")
                    finally:
                        if sent_ok:
                            self._buffer.mark_consumed(slot)
                        else:
                            self._buffer.release_reading(slot)

                    if sent_ok:
                        t_send_done = time.perf_counter()
                        read_ms = (t_read_done - t_recv_done) * 1000
                        send_ms = (t_send_done - t_read_done) * 1000
                        total_ms = (t_send_done - t_recv_done) * 1000
                        # Per-request log is DEBUG: the trainer-side
                        # equivalent is also DEBUG, and at steady
                        # state these fire many times per second per
                        # server thread.  Aggregate counters live in
                        # ``self._thread_metrics`` for ops dashboards.
                        logger.debug(
                            f"[UCXXBuffer] req slot={slot} bytes={raw_buf.nbytes} "
                            f"read_ms={read_ms:.1f} send_ms={send_ms:.1f} "
                            f"total_ms={total_ms:.1f}"
                        )

                        tname = threading.current_thread().name
                        with self._thread_metrics_lock:
                            m = self._thread_metrics.setdefault(
                                tname,
                                {
                                    "requests": 0,
                                    "total_read_ms": 0.0,
                                    "total_send_ms": 0.0,
                                },
                            )
                            m["requests"] += 1
                            m["total_read_ms"] += read_ms
                            m["total_send_ms"] += send_ms

                except asyncio.TimeoutError:
                    idle_cycles += 1
                    if idle_cycles >= self._HANDLER_MAX_IDLE_CYCLES:
                        logger.debug(
                            f"[UCXXBuffer] Handler idle for "
                            f"{idle_cycles * self._HANDLER_RECV_TIMEOUT:.0f}s, closing"
                        )
                        break
                    continue
                except Exception as e:
                    # Connection closed by client is expected
                    if "canceled" in str(e).lower() or "reset" in str(e).lower():
                        logger.debug(f"[UCXXBuffer] Client disconnected: {e}")
                    else:
                        logger.warning(f"[UCXXBuffer] Connection error: {e}")
                    break
        finally:
            with self._endpoints_lock:
                if endpoint in self._active_endpoints:
                    self._active_endpoints.remove(endpoint)

    async def _send_error_response(self, endpoint, exc: BaseException) -> None:
        """Best-effort status=2 + (msg_len, msg) error report to the client.

        Failure to deliver the report is silent: the client is in some
        unknown state, and this handler's job is just to avoid pinning
        on a doomed transfer.  Slot-state cleanup is the caller's
        responsibility (the handler's ``finally`` calls
        ``release_reading`` / ``mark_consumed`` exactly once).
        """
        try:
            await endpoint.send(np.array([2], dtype=np.uint8))
            error_msg = str(exc).encode("utf-8")
            msg_len = np.array([len(error_msg)], dtype=np.int32)
            await endpoint.send(msg_len)
            await endpoint.send(np.frombuffer(error_msg, dtype=np.uint8))
        except Exception:
            pass

    def stop_server(self, timeout: float = 5.0) -> None:
        """Stop all UCXX server threads and wait for them to finish.

        Args:
            timeout: Timeout in seconds to wait for each server thread to stop.
        """
        self._shutdown_flag.set()

        for t in self._server_threads:
            if t is not None and t.is_alive():
                t.join(timeout=timeout)
                if t.is_alive():
                    logger.warning(
                        f"[UCXXBuffer] Server thread {t.name} did not stop cleanly"
                    )

        for listener in self._listeners:
            if listener is not None:
                try:
                    listener.close()
                except Exception:
                    pass

        self._listeners.clear()
        self._server_threads.clear()
        self._ports.clear()
        logger.info("[UCXXBuffer] Server stopped")

    def get_server_metrics(self) -> Dict[str, Dict[str, float]]:
        """Return per-thread server metrics (request count, cumulative timings)."""
        with self._thread_metrics_lock:
            return {k: dict(v) for k, v in self._thread_metrics.items()}

    # =========================================================================
    # Buffer Write Operations (Worker Side)
    # =========================================================================

    def write(self, data: Dict[str, Any], overwrite_if_full: bool = True) -> int:
        """
        Write data to buffer.

        Args:
            data: Dict of tensors/arrays matching schema.
            overwrite_if_full: If True, overwrite oldest unconsumed entry.

        Returns:
            Slot index where data was written.
        """
        return self._buffer.write(data, overwrite_if_full)

    def write_raw(self, buf: bytes, overwrite_if_full: bool = True) -> int:
        """Write a pre-packed contiguous buffer to the next slot.

        See :meth:`SharedRingBuffer.write_raw` for details.
        """
        return self._buffer.write_raw(buf, overwrite_if_full)

    def get_metadata(self, slot: int) -> Dict[str, Any]:
        """
        Get metadata for a slot (to be sent via Redis stream).

        Args:
            slot: Slot index.

        Returns:
            Metadata dict with worker_ip, ports, slot for trainer to connect.
        """
        return {
            "worker_ip": self._local_ip,
            "ports": list(self._ports) if self._ports else [self._base_port],
            "slot": slot,
            "buffer_name": self._buffer.buffer_name,
        }

    # =========================================================================
    # Buffer Read Operations (for local reads)
    # =========================================================================

    def read(self, index: int) -> Dict[str, Any]:
        """Read data from buffer (local access)."""
        return self._buffer.read(index)

    def try_read(self, index: int) -> Optional[Dict[str, Any]]:
        """Try to read data, return None if not ready."""
        return self._buffer.try_read(index)

    def mark_consumed(self, index: int) -> None:
        """Mark slot as consumed (for local reads)."""
        self._buffer.mark_consumed(index)

    def is_ready(self, index: int) -> bool:
        """Check if slot is ready to read."""
        return self._buffer.is_ready(index)

    def get_slot_state(self, index: int) -> SlotState:
        """Get current state of a slot."""
        return self._buffer.get_slot_state(index)

    # =========================================================================
    # Metrics and Info
    # =========================================================================

    def get_metrics(self) -> BufferMetrics:
        """Get buffer metrics."""
        return self._buffer.get_metrics()

    def get_handle(self) -> Dict[str, Any]:
        """Get serializable handle for buffer discovery."""
        handle = self._buffer.get_handle()
        handle["ucxx_ports"] = list(self._ports) if self._ports else [self._base_port]
        handle["worker_ip"] = self._local_ip
        return handle

    @property
    def buffer_name(self) -> str:
        """Get buffer name."""
        return self._buffer.buffer_name

    @property
    def local_ip(self) -> str:
        """Get local IP address."""
        return self._local_ip

    @property
    def port(self) -> int:
        """Get primary UCXX server port."""
        return self._ports[0] if self._ports else self._base_port

    @property
    def ports(self) -> List[int]:
        """Get all UCXX server ports."""
        return list(self._ports) if self._ports else [self._base_port]

    # =========================================================================
    # Cleanup
    # =========================================================================

    def close(self) -> None:
        """Close buffer (doesn't unlink shared memory)."""
        self._buffer.close()

    def unlink(self) -> None:
        """Unlink (delete) shared memory."""
        self._buffer.unlink()

    def __del__(self):
        self.close()


class UCXXClient:
    """UCXX client for reading from remote UCXXBuffer servers.

    Trainer side: connects to worker UCXX servers and reads slot data.
    Endpoints are cached per (worker_ip, port) using exclusive checkout
    (``dict.pop``) so concurrent reads to the same target never share an
    endpoint.  A successful read returns the endpoint to the cache for
    reuse; a failed read discards it.

    Usage::

        client = UCXXClient()
        data = await client.read(worker_ip="10.0.0.5", port=13337,
                                 slot=42, schema=schema)
        await client.close()
    """

    _PINNED_POOL_MAX = 8

    def __init__(self) -> None:
        if not UCXX_AVAILABLE:
            raise RuntimeError(
                "UCXX is required for UCXXClient. Install with: pip install ucxx-cu12"
            )

        self._pool: Dict[tuple, collections.deque] = {}
        self._pool_size = 2
        self._rr_counter = 0
        self._rr_lock = threading.Lock()

        # Per-(worker_ip, port) skip-list: maps to the ``time.monotonic()``
        # tick at which the port becomes re-eligible for rotation.  No
        # lock needed; CPython dict ops are atomic for single-key
        # access and the worst-case race -- one task reading a
        # one-tick-stale timestamp -- is harmless.
        self._port_skip_until: Dict[Tuple[str, int], float] = {}

        self._pinned_pool: collections.deque = collections.deque()
        self._pinned_buf_size: int = 0

        try:
            ucxx.init()
        except RuntimeError as e:
            if "already initiated" not in str(e):
                raise

    def _healthy_ports(self, worker_ip: str, ports: List[int]) -> List[int]:
        """Filter ``ports`` to the subset not currently quarantined.

        The skip-list (``self._port_skip_until``) records ports that
        recently emitted a transport-class failure.  Entries expire
        naturally after :data:`_PORT_QUARANTINE_SEC`; this method
        consults the expiry stamps lazily on each call.

        Falls back to the full ``ports`` list if every entry is
        quarantined, so a transient all-port outage never starves a
        read.  Healthy-only filtering is sufficient under any
        partial-failure mode where at least one server thread is
        alive.
        """
        now = time.monotonic()
        healthy = [
            p for p in ports if self._port_skip_until.get((worker_ip, p), 0.0) <= now
        ]
        return healthy if healthy else list(ports)

    def _quarantine_port(self, worker_ip: str, port: int) -> None:
        """Mark ``(worker_ip, port)`` unhealthy for the cooldown.

        The next ``_PORT_QUARANTINE_SEC`` of :meth:`read` calls will
        route chunks around this port via :meth:`_healthy_ports`
        until the timestamp expires.  Re-failure during the cooldown
        extends the deadline to a fresh ``now +
        _PORT_QUARANTINE_SEC`` (no exponential backoff -- if data
        shows that's needed, it's a one-line change here).
        """
        self._port_skip_until[(worker_ip, port)] = (
            time.monotonic() + _PORT_QUARANTINE_SEC
        )

    def _acquire_pinned(self, nbytes: int) -> torch.Tensor:
        """Get a pinned CPU buffer from the pool, or allocate a new one."""
        if self._pinned_buf_size == nbytes and self._pinned_pool:
            return self._pinned_pool.popleft()
        if self._pinned_buf_size != nbytes:
            self._pinned_pool.clear()
            self._pinned_buf_size = nbytes
        try:
            buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        except RuntimeError:
            buf = torch.empty(nbytes, dtype=torch.uint8)
            logger.warning("[UCXXClient] cudaHostAlloc failed, using pageable memory")
        return buf

    def return_pinned(self, buf: torch.Tensor) -> None:
        """Return a pinned buffer to the pool for reuse."""
        if len(self._pinned_pool) < self._PINNED_POOL_MAX:
            self._pinned_pool.append(buf)

    async def _read_slot(
        self,
        worker_ip: str,
        port: int,
        slot: int,
        recv_buf: np.ndarray,
        timeout: float,
    ) -> None:
        """Fetch the entire slot payload from one server thread on ``port``.

        Single-chunk protocol -- one connection, one ``send([slot])``,
        one ``recv(status)``, one ``recv(payload)``.  ``timeout`` bounds
        each individual ``send`` / ``recv`` await.  See
        :meth:`UCXXClient.read` for the rationale on the 5 s default.
        """
        key = (worker_ip, port)
        pool = self._pool.get(key)
        endpoint = None
        if pool:
            now = time.monotonic()
            # Drain any pooled endpoints that have aged past the
            # server's idle-eviction window.  Each entry's
            # ``_pool_last_use`` is stamped at return-to-pool below;
            # absence (defaults to 0.0) means "never returned"
            # which is older than any threshold and gets evicted.
            while pool:
                try:
                    candidate = pool.popleft()
                except IndexError:
                    break
                last_use = getattr(candidate, "_pool_last_use", 0.0)
                if now - last_use <= _POOL_ENDPOINT_MAX_AGE_S:
                    endpoint = candidate
                    break
                # Aged out -- close and try the next one.  Failures
                # here are silent because the endpoint is being
                # discarded anyway.
                try:
                    await candidate.close()
                except Exception:
                    pass
        if endpoint is None:
            endpoint = await asyncio.wait_for(
                ucxx.create_endpoint(worker_ip, port), timeout=timeout
            )

        ok = False
        try:
            slot_arr = np.array([slot], dtype=np.int64)
            await asyncio.wait_for(endpoint.send(slot_arr), timeout=timeout)

            status = np.empty(1, dtype=np.uint8)
            await asyncio.wait_for(endpoint.recv(status), timeout=timeout)

            if status[0] == 0:
                await asyncio.wait_for(endpoint.recv(recv_buf), timeout=timeout)
                ok = True
            elif status[0] == 1:
                # Stale slot is a clean protocol-level "no" -- the
                # endpoint stays healthy, so we can safely return it
                # to the pool before raising.
                ok = True
                raise StaleSlotError(f"Slot {slot} unavailable (stale reference)")
            elif status[0] == 2:
                msg_len = np.empty(1, dtype=np.int32)
                await asyncio.wait_for(endpoint.recv(msg_len), timeout=timeout)
                msg_buf = np.empty(int(msg_len[0]), dtype=np.uint8)
                await asyncio.wait_for(endpoint.recv(msg_buf), timeout=timeout)
                raise RuntimeError(
                    f"Remote read failed: {msg_buf.tobytes().decode('utf-8')}"
                )
            else:
                raise RuntimeError(f"Unknown response status: {status[0]}")
        finally:
            if ok:
                ep_pool = self._pool.setdefault(key, collections.deque())
                if len(ep_pool) < self._pool_size:
                    # Stamp last-use so the next checkout can age
                    # it out before the server's handler-idle
                    # eviction window expires.
                    endpoint._pool_last_use = time.monotonic()
                    ep_pool.append(endpoint)
                else:
                    try:
                        await endpoint.close()
                    except Exception:
                        pass
            else:
                try:
                    await endpoint.close()
                except Exception:
                    pass

    async def read(
        self,
        worker_ip: str,
        port: int,
        slot: int,
        schema: List[Any],
        timeout: float = 5.0,
        ports: Optional[List[int]] = None,
    ) -> Dict[str, np.ndarray]:
        """Read slot data from a remote worker buffer.

        Single-chunk semantics: one connection to one server port
        ferries the whole slot payload.  All N server threads on a
        worker mirror the same SHM, so any thread can serve any slot;
        we exploit that symmetry for load balancing and failure
        recovery without ever splitting a single read across threads.

        **Port rotation and fallback.**

        1. *Per-call rotation.*  Successive calls advance a round-
           robin counter so traffic spreads evenly across all healthy
           server threads instead of always hammering
           ``available_ports[0]``.  Pure load balancing.
        2. *On-failure fallback.*  If a read fails with a transport-
           class error (timeout, endpoint reset, etc. -- see
           :data:`_PORT_ROTATABLE_ERRORS`), the offending port is
           quarantined for :data:`_PORT_QUARANTINE_SEC` and we retry
           once on the next port in rotation.  A single wedged server
           thread therefore costs one timeout's worth of latency, not
           the whole job.  Non-transport errors (stale slot, server-
           side protocol error) propagate immediately.

        ``timeout`` defaults to 5 s -- p99 happy-path read of a ~500
        MB slot is ~1 s on RDMA / shared memory, so 5 s is ample
        headroom.  Larger values just delay rotation onto a healthy
        port when one server thread wedges.

        Returns a dict of tensor name -> numpy view into a pinned CPU
        buffer.  The pinned backing tensor is stored under the
        ``_pinned_buf`` key and must be returned to the pool via
        :meth:`return_pinned` after the caller has copied data to GPU.
        """
        if not schema:
            raise ValueError("Schema required for zero-pack protocol")

        # Health-aware rotation: skip ports that recently emitted a
        # transport-class failure (see :meth:`_healthy_ports`).
        all_ports = ports if ports and len(ports) > 1 else [port]
        available_ports = self._healthy_ports(worker_ip, all_ports)
        total_bytes = sum(spec.nbytes for spec in schema)
        pinned_buf = self._acquire_pinned(total_bytes)
        raw = pinned_buf.numpy()

        t_start = time.perf_counter()

        # Per-call rotation: each call advances the round-robin
        # counter so the next call lands on a different starting port.
        # Locked because UCXXClient may be shared across asyncio
        # tasks in the trainer prefetcher.
        with self._rr_lock:
            rotation = self._rr_counter
            self._rr_counter = (self._rr_counter + 1) % len(available_ports)

        # Two attempts max.  Attempt 1 lands on the next port in
        # rotation, so a single wedged thread costs one timeout, not
        # the whole job.
        last_exc: Optional[BaseException] = None
        target_port: Optional[int] = None
        for attempt in range(2):
            offset = (rotation + attempt) % len(available_ports)
            target_port = available_ports[offset]
            try:
                await self._read_slot(worker_ip, target_port, slot, raw, timeout)
                last_exc = None
                break  # success
            except BaseException as e:  # noqa: BLE001 -- re-raised below
                last_exc = e
                if attempt == 0 and type(e).__name__ in _PORT_ROTATABLE_ERRORS:
                    self._quarantine_port(worker_ip, target_port)
                    logger.warning(
                        f"[UCXXClient] read failed via {worker_ip} "
                        f"port={target_port} slot={slot}: "
                        f"{type(e).__name__}: {e}; rotating to next "
                        f"port and retrying"
                    )
                    continue
                raise
        if last_exc is not None:  # second attempt also failed
            raise last_exc

        t_done = time.perf_counter()
        total_ms = (t_done - t_start) * 1000
        mb = total_bytes / (1024 * 1024)
        bw_str = f", bw={mb / (total_ms / 1000):.0f} MB/s" if total_ms > 0 else ""
        logger.debug(
            f"[UCXXClient] read {worker_ip} slot={slot}: "
            f"{mb:.1f} MB in {total_ms:.1f} ms "
            f"(port={target_port}{bw_str})"
        )

        result: Dict[str, Any] = {}
        offset = 0
        for spec in schema:
            result[spec.name] = np.frombuffer(
                raw[offset : offset + spec.nbytes], dtype=spec.dtype
            ).reshape(spec.shape)
            offset += spec.nbytes
        result["_pinned_buf"] = pinned_buf
        return result

    async def close(self) -> None:
        """Drain and close all pooled endpoints."""
        for key in list(self._pool):
            pool = self._pool.pop(key, None)
            if pool:
                for ep in pool:
                    try:
                        await ep.close()
                    except Exception:
                        pass
