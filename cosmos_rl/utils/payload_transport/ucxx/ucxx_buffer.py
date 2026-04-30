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
from typing import Any, Dict, List, Optional

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
        self._slot_refcount: Dict[int, int] = {}
        self._slot_refcount_lock = threading.Lock()
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
            await asyncio.sleep(0)

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
    _PORT_RETRY_ATTEMPTS = 10

    async def _handle_connection(self, endpoint) -> None:
        """Handle incoming connection from trainer.

        Chunked protocol:
        1. Receive: int64[3] = [slot, chunk_idx, n_chunks]
        2. Send: status byte (1 byte)
        3. Send: chunk slice of the raw SHM slot
        """
        logger.debug("[UCXXBuffer] New connection from trainer")
        with self._endpoints_lock:
            self._active_endpoints.append(endpoint)

        idle_cycles = 0
        try:
            while not self._shutdown_flag.is_set():
                try:
                    slot_buf = np.empty(3, dtype=np.int64)
                    await asyncio.wait_for(
                        endpoint.recv(slot_buf), timeout=self._HANDLER_RECV_TIMEOUT
                    )
                    idle_cycles = 0
                    t_recv_done = time.perf_counter()
                    slot = int(slot_buf[0])
                    chunk_idx = int(slot_buf[1])
                    n_chunks = int(slot_buf[2])

                    acquired_ref = False
                    try:
                        if not self._buffer.schema:
                            raise RuntimeError(
                                "Zero-pack protocol requires schema-based buffer"
                            )

                        try:
                            raw_buf = self._buffer.read_raw(slot)
                        except SlotError as e:
                            status = np.array([1], dtype=np.uint8)
                            await endpoint.send(status)
                            with self._slot_refcount_lock:
                                rc_snap = dict(self._slot_refcount)
                            write_idx, _, entry_count = self._buffer._read_header()
                            logger.warning(
                                f"[UCXXBuffer] StaleSlot slot={slot} chunk={chunk_idx}/{n_chunks} "
                                f"err='{e}' write_idx={write_idx} entry_count={entry_count} "
                                f"refcount_snapshot={rc_snap} thread={threading.current_thread().name}"
                            )
                            continue
                        with self._slot_refcount_lock:
                            if slot not in self._slot_refcount:
                                self._slot_refcount[slot] = n_chunks
                            elif self._slot_refcount[slot] <= 0:
                                self._slot_refcount[slot] = n_chunks
                        acquired_ref = True

                        total = raw_buf.nbytes
                        chunk_size = (total + n_chunks - 1) // n_chunks
                        start = chunk_idx * chunk_size
                        end = min(start + chunk_size, total)
                        chunk_buf = raw_buf[start:end]
                        t_read_done = time.perf_counter()

                        status = np.array([0], dtype=np.uint8)
                        await endpoint.send(status)

                        await endpoint.send(chunk_buf)

                        with self._slot_refcount_lock:
                            self._slot_refcount[slot] -= 1
                            remaining = self._slot_refcount[slot]
                            if remaining <= 0:
                                self._slot_refcount.pop(slot, None)
                        acquired_ref = False
                        if remaining <= 0:
                            self._buffer.mark_consumed(slot)

                        t_send_done = time.perf_counter()
                        read_ms = (t_read_done - t_recv_done) * 1000
                        send_ms = (t_send_done - t_read_done) * 1000
                        total_ms = (t_send_done - t_recv_done) * 1000
                        logger.info(
                            f"[UCXXBuffer] req slot={slot} chunk={chunk_idx}/{n_chunks} "
                            f"bytes={chunk_buf.nbytes} read_ms={read_ms:.1f} "
                            f"send_ms={send_ms:.1f} total_ms={total_ms:.1f}"
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

                    except Exception as e:
                        if acquired_ref:
                            with self._slot_refcount_lock:
                                self._slot_refcount[slot] = (
                                    self._slot_refcount.get(slot, 0) - 1
                                )
                                remaining = self._slot_refcount[slot]
                                if remaining <= 0:
                                    self._slot_refcount.pop(slot, None)
                            if remaining <= 0:
                                self._buffer.release_reading(slot)
                                logger.warning(
                                    f"[UCXXBuffer] Released slot {slot} back "
                                    f"to READY after chunk {chunk_idx} failure"
                                )
                        try:
                            status = np.array([2], dtype=np.uint8)
                            await endpoint.send(status)
                            error_msg = str(e).encode("utf-8")
                            msg_len = np.array([len(error_msg)], dtype=np.int32)
                            await endpoint.send(msg_len)
                            await endpoint.send(
                                np.frombuffer(error_msg, dtype=np.uint8)
                            )
                        except Exception:
                            pass
                        logger.warning(f"[UCXXBuffer] Send error for slot {slot}: {e}")

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

        self._pinned_pool: collections.deque = collections.deque()
        self._pinned_buf_size: int = 0

        try:
            ucxx.init()
        except RuntimeError as e:
            if "already initiated" not in str(e):
                raise

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

    async def _read_chunk(
        self,
        worker_ip: str,
        port: int,
        slot: int,
        chunk_idx: int,
        n_chunks: int,
        recv_buf: np.ndarray,
        timeout: float,
    ) -> None:
        """Read a single chunk from the server into a slice of recv_buf."""
        key = (worker_ip, port)
        pool = self._pool.get(key)
        endpoint = None
        if pool:
            try:
                endpoint = pool.popleft()
            except IndexError:
                pass
        if endpoint is None:
            endpoint = await asyncio.wait_for(
                ucxx.create_endpoint(worker_ip, port), timeout=timeout
            )

        ok = False
        try:
            slot_arr = np.array([slot, chunk_idx, n_chunks], dtype=np.int64)
            await asyncio.wait_for(endpoint.send(slot_arr), timeout=timeout)

            status = np.empty(1, dtype=np.uint8)
            await asyncio.wait_for(endpoint.recv(status), timeout=timeout)

            if status[0] == 0:
                total = len(recv_buf)
                chunk_size = (total + n_chunks - 1) // n_chunks
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, total)
                chunk_view = recv_buf[start:end]
                await asyncio.wait_for(endpoint.recv(chunk_view), timeout=timeout)
                ok = True
            elif status[0] == 1:
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
        timeout: float = 60.0,
        ports: Optional[List[int]] = None,
        n_chunks: int = 1,
    ) -> Dict[str, np.ndarray]:
        """Read slot data from a remote worker buffer.

        When ``n_chunks > 1`` and multiple ``ports`` are available, the
        transfer is split into parallel chunk reads across different server
        threads for higher aggregate RDMA throughput.

        Returns a dict of tensor name -> numpy view into a pinned CPU buffer.
        The pinned backing tensor is stored under the ``_pinned_buf`` key and
        must be returned to the pool via :meth:`return_pinned` after the
        caller has finished copying data to GPU.
        """
        if not schema:
            raise ValueError("Schema required for zero-pack protocol")

        available_ports = ports if ports and len(ports) > 1 else [port]
        n_chunks = min(n_chunks, len(available_ports))
        total_bytes = sum(spec.nbytes for spec in schema)
        pinned_buf = self._acquire_pinned(total_bytes)
        raw = pinned_buf.numpy()

        t_start = time.perf_counter()

        if n_chunks > 1:
            coros = []
            for ci in range(n_chunks):
                target_port = available_ports[ci % len(available_ports)]
                coros.append(
                    self._read_chunk(
                        worker_ip, target_port, slot, ci, n_chunks, raw, timeout
                    )
                )
            results = await asyncio.gather(*coros, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    raise r
        else:
            target_port = available_ports[0]
            await self._read_chunk(worker_ip, target_port, slot, 0, 1, raw, timeout)

        t_done = time.perf_counter()
        total_ms = (t_done - t_start) * 1000
        mb = total_bytes / (1024 * 1024)
        bw_str = f", bw={mb / (total_ms / 1000):.0f} MB/s" if total_ms > 0 else ""
        logger.debug(
            f"[UCXXClient] read {worker_ip} slot={slot}: "
            f"{mb:.1f} MB in {total_ms:.1f} ms "
            f"(n_chunks={n_chunks}{bw_str})"
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
