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

"""UCXX mixins for rollout workers and trainers.

Active mixins:

1. :class:`UCXXRolloutMixin` -- rollout workers: write to SharedRingBuffer
   and serve over UCXX.
2. :class:`UCXXDataPackerMixin` (added in a follow-up MR under
   ``data_packer_mixin.py``) -- DataPackers: resolve UCXX pointers in
   ``get_policy_input()`` with prefetch + double-buffering.

Deprecated:

3. :class:`UCXXTrainerMixin` -- kept for reference; superseded by
   ``UCXXDataPackerMixin``.
"""

import asyncio
import fcntl
import queue
import socket
import struct
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.payload_transport.ucxx.ucxx_buffer import (
    UCXX_AVAILABLE,
    UCXXBuffer,
    UCXXBufferConfig,
    UCXXClient,
)

# Trace utility is provided by a sibling MR; fall back to a wall-clock
# stand-in when running against an older cosmos-rl that lacks it.  This
# keeps the UCXX MR independent of the trace MR's merge order.
try:
    from cosmos_rl.utils.trace import get_trace_time  # type: ignore
except ImportError:  # pragma: no cover - fallback path
    import time as _time

    def get_trace_time() -> float:  # type: ignore[no-redef]
        return _time.perf_counter() * 1000.0


_TRANSIENT_UCXX_ERRORS = frozenset(
    {
        "UCXXCanceledError",
        "UCXXConnectionResetError",
        "UCXXCloseError",
        "TimeoutError",
    }
)

_MAX_FETCH_ROUNDS = 3

# Canonical trajectory field names.  Mirrored from
# ``cosmos_rl.dispatcher.data.packer.tensor_data_packer`` so this module
# can be imported standalone without dragging in the dispatcher.
OBSERVATIONS = "observations"
ACTIONS = "actions"
REWARDS = "rewards"
TERMINATED = "terminated"
TRUNCATED = "truncated"
EPISODE_LENGTH = "episode_length"


def _get_iface_ip(iface: str) -> Optional[str]:
    """Get IPv4 address of a network interface via ioctl."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        addr = fcntl.ioctl(
            s.fileno(),
            0x8915,  # SIOCGIFADDR
            struct.pack("256s", iface.encode()),
        )
        s.close()
        return socket.inet_ntoa(addr[20:24])
    except OSError:
        return None


def _get_local_ip() -> str:
    """Get local IP for UCXX binding, preferring RDMA interfaces.

    On clusters with IB/RoCE, the hostname-resolved IP (e.g. 10.49.x.x) is on
    the management network and unreachable over the RDMA fabric.  UCXX must
    bind to an RDMA interface IP (e.g. 192.168.x.x on rdma0) for IB transport.

    Priority:
      1. First ``rdma*`` interface with an IPv4 address
      2. Hostname resolution (fallback for non-IB environments)
    """
    # Prefer RDMA interfaces for IB-capable clusters
    rdma_ifaces = sorted(p.name for p in Path("/sys/class/net").glob("rdma*"))
    for iface in rdma_ifaces:
        ip = _get_iface_ip(iface)
        if ip:
            logger.info(f"[UCXX] Binding to RDMA interface {iface} -> {ip}")
            return ip

    # Fallback: hostname resolution (works for non-IB setups)
    hostname = socket.getfqdn()
    ip = socket.gethostbyname(hostname)

    # UCX may not recognise non-standard loopback addresses (e.g. 127.0.1.1
    # from Docker hostname entries) for its shared-memory transport.  Normalise
    # anything in the 127.x.x.x range to the canonical 127.0.0.1.
    if ip.startswith("127.") and ip != "127.0.0.1":
        logger.info(f"[UCXX] Hostname resolved to {ip}, normalising to 127.0.0.1")
        ip = "127.0.0.1"
    else:
        logger.info(f"[UCXX] No RDMA interfaces found, using hostname -> {ip}")
    return ip


class UCXXRolloutMixin:
    """
    Mixin for rollout workers to enable UCXX-based data transfer.

    Features:
    - Writes data to SharedRingBuffer with automatic padding
    - Starts UCXX server to serve data to trainers
    - Auto-optimizes for local (shared memory) and remote (RDMA) access

    Usage:
        class MyWorker(UCXXRolloutMixin, BaseWorker):
            def post_init_hook(self):
                self.setup_ucxx(
                    replica_id=self.replica_name,
                    max_steps=100,
                    obs_dim=4,
                    action_dim=2,
                )

            def generate_rollout(self):
                trajectory = collect_trajectory()
                metadata = self.write_to_buffer(trajectory)
                return metadata or trajectory

            def cleanup(self):
                self.cleanup_ucxx()
    """

    _ucxx_buffer: Optional[UCXXBuffer] = None
    _ucxx_replica_id: str = ""
    _ucxx_enabled: bool = False
    _ucxx_ip: str = ""
    _ucxx_port: int = 0
    _ucxx_max_steps: int = 100
    _ucxx_obs_dim: int = 4
    _ucxx_action_dim: int = 2
    _ucxx_packed_cpu: Optional[torch.Tensor] = None  # Pinned CPU staging buffer
    _ucxx_tensor_offsets: Optional[Dict[str, int]] = None
    _ucxx_entry_data_size: int = 0

    def setup_ucxx(
        self,
        replica_id: str,
        max_steps: int,
        obs_dim: int,
        action_dim: int,
        port: int = 0,
        config: Optional[UCXXBufferConfig] = None,
    ) -> None:
        """
        Initialize UCXX buffer and server for this rollout worker.

        Args:
            replica_id: Unique identifier for this replica
            max_steps: Maximum episode length (for padding)
            obs_dim: Observation dimension
            action_dim: Action dimension
            port: Port for UCXX server (0 = auto-assign)
            config: Optional buffer configuration

        Raises:
            RuntimeError: If UCXX is not available or setup fails
        """
        if not UCXX_AVAILABLE:
            raise RuntimeError(
                "UCXX is required for UCXXRolloutMixin. "
                "Install with: pip install ucxx-cu12"
            )

        self._ucxx_replica_id = replica_id
        self._ucxx_max_steps = max_steps
        self._ucxx_obs_dim = obs_dim
        self._ucxx_action_dim = action_dim

        try:
            # Build schema for trajectory data (required for zero-pack protocol)
            from cosmos_rl.utils.payload_transport.ucxx.tensor_spec import (
                TensorSpec,
            )

            schema = [
                TensorSpec(
                    name=OBSERVATIONS, shape=(max_steps, obs_dim), dtype=np.float32
                ),
                TensorSpec(
                    name=ACTIONS, shape=(max_steps, action_dim), dtype=np.float32
                ),
                TensorSpec(name=REWARDS, shape=(max_steps,), dtype=np.float32),
                TensorSpec(name=TERMINATED, shape=(max_steps,), dtype=np.bool_),
                TensorSpec(name=TRUNCATED, shape=(max_steps,), dtype=np.bool_),
                TensorSpec(name=EPISODE_LENGTH, shape=(1,), dtype=np.int64),
            ]

            # Create UCXX buffer with server and schema
            buffer_config = config or UCXXBufferConfig(
                max_entries=1000,
                entry_size_bytes=65536,
            )
            buffer_config.buffer_name = f"ucxx_rollout_{replica_id}"
            buffer_config.port = port  # Set port in config
            buffer_config.schema = schema  # Schema required for zero-pack protocol

            self._ucxx_buffer = UCXXBuffer(buffer_config)

            # Pre-compute schema layout for coalesced writes
            self._ucxx_tensor_offsets = {}
            offset = 0
            for spec in schema:
                self._ucxx_tensor_offsets[spec.name] = offset
                offset += spec.nbytes
            self._ucxx_entry_data_size = offset
            self._ucxx_schema = schema

            # Pre-allocate pinned CPU staging buffer for bulk D2H + SHM copy.
            # Pinned memory enables DMA and avoids per-tensor cudaMemcpy overhead.
            self._ucxx_packed_cpu = torch.empty(
                self._ucxx_entry_data_size, dtype=torch.uint8, pin_memory=True
            )

            # Start UCXX server
            self._ucxx_buffer.start_server()
            self._ucxx_ip = self._ucxx_buffer.local_ip
            self._ucxx_port = self._ucxx_buffer.port

            self._ucxx_enabled = True
            logger.info(
                f"[UCXXRolloutMixin] Worker '{replica_id}' ready at "
                f"{self._ucxx_ip}:{self._ucxx_port} "
                f"(max_steps={max_steps}, obs_dim={obs_dim}, action_dim={action_dim}, "
                f"entry_size={self._ucxx_entry_data_size / 1e6:.1f} MB)"
            )

        except Exception as e:
            raise RuntimeError(f"UCXX setup failed: {e}") from e

    def write_to_buffer(self, trajectory: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Write trajectory to buffer with padding and return metadata for UCXX fetch.

        Uses a coalesced write strategy for large payloads:
        1. Pack all schema tensors into a single contiguous GPU buffer
        2. One bulk ``cudaMemcpy`` D2H into a pre-allocated pinned CPU buffer
        3. One bulk ``memoryview`` copy into the SHM slot

        This eliminates per-tensor Python loop overhead and reaches closer to
        hardware bandwidth limits (PCIe for D2H, DDR for SHM).

        Falls back to the per-tensor path when tensors are not on GPU.
        """
        if not self._ucxx_enabled or self._ucxx_buffer is None:
            return None

        try:
            ep_len_val = trajectory.get(EPISODE_LENGTH)
            if ep_len_val is None:
                obs = trajectory.get(OBSERVATIONS)
                if obs is not None:
                    ep_len = obs.shape[0] if hasattr(obs, "shape") else len(obs)
                else:
                    ep_len = self._ucxx_max_steps
            elif isinstance(ep_len_val, torch.Tensor):
                ep_len = int(ep_len_val.item())
            else:
                ep_len = int(ep_len_val)

            any_gpu = any(
                isinstance(v, torch.Tensor) and v.is_cuda for v in trajectory.values()
            )

            t_gpu2cpu_start = get_trace_time()

            if any_gpu and self._ucxx_packed_cpu is not None:
                # --- Fast path: coalesced GPU → pinned CPU → SHM ----
                device = None
                for v in trajectory.values():
                    if isinstance(v, torch.Tensor) and v.is_cuda:
                        device = v.device
                        break

                gpu_packed = torch.zeros(
                    self._ucxx_entry_data_size, dtype=torch.uint8, device=device
                )

                for spec in self._ucxx_schema:
                    raw = trajectory.get(spec.name)
                    if raw is None:
                        continue

                    if isinstance(raw, torch.Tensor):
                        tensor = raw
                    else:
                        tensor = torch.as_tensor(raw, device=device)

                    # Pad variable-length fields
                    if spec.name in (
                        OBSERVATIONS,
                        ACTIONS,
                        REWARDS,
                        TERMINATED,
                        TRUNCATED,
                    ):
                        if tensor.shape[0] < spec.shape[0]:
                            padded = torch.zeros(
                                spec.shape, dtype=tensor.dtype, device=device
                            )
                            padded[: tensor.shape[0]] = tensor
                            tensor = padded
                    elif spec.name == EPISODE_LENGTH:
                        tensor = torch.tensor(
                            [ep_len], dtype=torch.int64, device=device
                        )

                    tensor = tensor.reshape(spec.shape).contiguous()
                    flat = tensor.view(torch.uint8).reshape(-1)
                    off = self._ucxx_tensor_offsets[spec.name]
                    gpu_packed[off : off + flat.numel()] = flat

                # Single bulk D2H copy into pinned staging buffer
                self._ucxx_packed_cpu.copy_(gpu_packed, non_blocking=False)

                t_gpu2cpu_end = get_trace_time()

                # Single bulk SHM write
                slot = self._ucxx_buffer.write_raw(
                    memoryview(self._ucxx_packed_cpu.numpy())
                )
                t_shm_end = get_trace_time()
                total_bytes = self._ucxx_entry_data_size
            else:
                # --- Fallback: per-tensor CPU path (no GPU tensors) ---
                cpu_data = {}
                for key, value in trajectory.items():
                    if isinstance(value, torch.Tensor):
                        arr = value.cpu().numpy()
                    else:
                        arr = np.asarray(value)

                    if key in (OBSERVATIONS, ACTIONS, REWARDS, TERMINATED, TRUNCATED):
                        if len(arr.shape) > 0 and arr.shape[0] < self._ucxx_max_steps:
                            if key == OBSERVATIONS:
                                padded = np.zeros(
                                    (self._ucxx_max_steps, self._ucxx_obs_dim),
                                    dtype=arr.dtype,
                                )
                            elif key == ACTIONS:
                                padded = np.zeros(
                                    (self._ucxx_max_steps, self._ucxx_action_dim),
                                    dtype=arr.dtype,
                                )
                            else:
                                padded = np.zeros(
                                    (self._ucxx_max_steps,), dtype=arr.dtype
                                )
                            padded[: arr.shape[0]] = arr
                            arr = padded
                    elif key == EPISODE_LENGTH:
                        arr = np.array([ep_len], dtype=np.int64)

                    cpu_data[key] = arr

                t_gpu2cpu_end = get_trace_time()

                slot = self._ucxx_buffer.write(cpu_data)
                t_shm_end = get_trace_time()
                total_bytes = sum(
                    arr.nbytes for arr in cpu_data.values() if hasattr(arr, "nbytes")
                )

            gpu2cpu_ms = t_gpu2cpu_end - t_gpu2cpu_start
            shm_ms = t_shm_end - t_gpu2cpu_end
            thread_name = threading.current_thread().name
            logger.debug(
                f"[Trace] thread={thread_name} op=ucxx_write "
                f"start={t_gpu2cpu_start:.1f} end={t_shm_end:.1f} "
                f"gpu2cpu_ms={gpu2cpu_ms:.1f} shm_ms={shm_ms:.1f} "
                f"bytes={total_bytes}"
            )

            return {
                "_ucxx": True,
                "_ucxx_enabled": True,
                "_worker_ip": self._ucxx_ip,
                "_ucxx_port": self._ucxx_port,
                "_ports": self._ucxx_buffer.ports,
                "_slot": slot,
                "_buffer_handle": self._ucxx_buffer.get_handle(),
                "_replica_id": self._ucxx_replica_id,
                REWARDS: trajectory.get(REWARDS, torch.tensor([])).tolist(),
                EPISODE_LENGTH: ep_len,
            }
        except Exception as e:
            logger.error(f"[UCXXRolloutMixin] Write failed: {e}")
            return None

    def cleanup_ucxx(self) -> None:
        """Clean up UCXX resources."""
        if self._ucxx_buffer:
            self._ucxx_buffer.stop_server()
            self._ucxx_buffer.close()
            self._ucxx_buffer = None
        self._ucxx_enabled = False
        logger.info(f"[UCXXRolloutMixin] Worker '{self._ucxx_replica_id}' cleaned up")


class UCXXTrainerMixin:
    """Mixin for trainers to enable UCXX-based data fetching with background prefetch.

    .. deprecated::
        Superseded by :class:`UCXXDataPackerMixin` (in ``data_packer_mixin.py``)
        which places UCXX resolution in the DataPacker's ``get_policy_input()``
        rather than coupling it to a specific trainer class.  Kept here for
        backward-compatibility and reference; new code should use the mixin
        on the DataPacker instead.
    """

    _ucxx_trainer_client: Optional[UCXXClient] = None
    _ucxx_trainer_enabled: bool = False
    _ucxx_trainer_device: torch.device = None
    _ucxx_trainer_prefetch_thread: Optional[threading.Thread] = None
    _ucxx_trainer_request_queue: Optional[queue.Queue] = None
    _ucxx_trainer_result_queue: Optional[queue.Queue] = None
    _ucxx_trainer_shutdown: Optional[threading.Event] = None
    _ucxx_trainer_batch_id: int = 0
    _ucxx_trainer_step_count: int = 0
    _ucxx_trainer_total_ucxx: int = 0
    _ucxx_trainer_total_fallback: int = 0
    _ucxx_trainer_total_bytes: int = 0
    _ucxx_trainer_total_latency_ms: float = 0.0
    _ucxx_trainer_prefetch_timeout: float = 300.0
    _ucxx_trainer_max_attempts: int = 2
    _ucxx_trainer_read_timeout: float = 60.0
    _UCXX_LOG_INTERVAL: int = 50

    def setup_ucxx_trainer(
        self,
        device: torch.device,
        prefetch_timeout: float = 300.0,
        max_attempts: int = 2,
        read_timeout: float = 60.0,
    ) -> None:
        """Initialize UCXX client and background prefetch thread.

        All episodes for a training iteration are fetched concurrently
        (no chunking).  Results stream to GPU via ``asyncio.as_completed``
        so the CPU-to-GPU copy for each episode starts as soon as its
        network read finishes.

        Args:
            device: Target GPU device for fetched tensors.
            prefetch_timeout: Seconds to wait for prefetch results before
                raising a timeout error.
            max_attempts: Total attempts per remote slot read (initial +
                retries on transient UCX errors).  Defaults to 2 to match
                the historic "retry once" behaviour.  Set to 1 to disable
                retries entirely.
            read_timeout: Per-await timeout (seconds) inside a single
                ``UCXXClient.read`` call -- bounds one ``send`` / ``recv``
                operation, distinct from ``prefetch_timeout`` which bounds
                a whole batch.
        """
        if not UCXX_AVAILABLE:
            raise RuntimeError(
                "UCXX is required for UCXXTrainerMixin. "
                "Install with: pip install ucxx-cu12"
            )

        self._ucxx_trainer_device = device
        self._ucxx_trainer_prefetch_timeout = prefetch_timeout
        self._ucxx_trainer_max_attempts = max(1, max_attempts)
        self._ucxx_trainer_read_timeout = read_timeout
        self._ucxx_trainer_client = UCXXClient()

        self._ucxx_trainer_request_queue = queue.Queue()
        self._ucxx_trainer_result_queue = queue.Queue()
        self._ucxx_trainer_shutdown = threading.Event()
        self._ucxx_trainer_shutdown.clear()

        self._ucxx_trainer_prefetch_thread = threading.Thread(
            target=self._ucxx_trainer_prefetch_worker,
            name="UCXXTrainerPrefetch",
            daemon=True,
        )
        self._ucxx_trainer_prefetch_thread.start()
        self._ucxx_trainer_enabled = True

        logger.info(
            f"[UCXXTrainerMixin] Initialized with device={device}, "
            f"prefetch_timeout={prefetch_timeout}s"
        )

    def _ucxx_trainer_prefetch_worker(self) -> None:
        """Background worker that fetches UCXX data while training runs.

        All episodes for the batch are fetched concurrently.  Results
        stream to GPU via ``asyncio.as_completed`` so the CPU→GPU copy
        for each episode starts as soon as its network read finishes.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            while not self._ucxx_trainer_shutdown.is_set():
                try:
                    batch_id, ucxx_tasks = self._ucxx_trainer_request_queue.get(
                        timeout=0.1
                    )
                except queue.Empty:
                    continue

                results = {}
                total_transfer_ms = 0.0
                total_copy_ms = 0.0

                if ucxx_tasks:
                    targets = [
                        f"{m.get('_worker_ip')}:{m.get('_ucxx_port')}"
                        for _, m in ucxx_tasks
                    ]
                    logger.debug(
                        f"[UCXXTrainerMixin] Batch {batch_id} targets: {targets}"
                    )

                    try:
                        results, total_transfer_ms, total_copy_ms = (
                            loop.run_until_complete(self._fetch_all(ucxx_tasks))
                        )
                    except Exception as e:
                        error_msg = (
                            f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
                        )
                        logger.error(
                            f"[UCXXTrainerMixin] Batch {batch_id} failed: {error_msg} "
                            f"(targets: {targets})"
                        )
                        results = {"_error": error_msg}

                self._ucxx_trainer_result_queue.put(
                    (batch_id, results, total_transfer_ms, total_copy_ms)
                )

        except Exception as e:
            logger.error(f"[UCXXTrainerMixin] Prefetch worker error: {e}")
        finally:
            loop.close()
            logger.info("[UCXXTrainerMixin] Prefetch worker stopped")

    async def _fetch_all(self, ucxx_tasks: list) -> tuple:
        """Fetch all episodes concurrently with multi-round retry.

        Each round attempts all pending episodes concurrently.  Within a
        round, ``_read_one`` does one immediate retry (handles stale
        endpoints).  Failed episodes are collected and retried in the next
        round — this works because the server-side ``release_reading()``
        transitions the slot back to READY, preserving data for re-read.

        Returns ``(results_dict, transfer_ms, copy_ms)``.
        """
        client = self._ucxx_trainer_client
        device = self._ucxx_trainer_device

        async def _read_one(idx: int, metadata: dict):
            """Read a single episode; return (idx, None) on failure."""
            worker_ip = metadata.get("_worker_ip")
            ucxx_port = metadata.get("_ucxx_port")
            slot = metadata.get("_slot")
            handle = metadata.get("_buffer_handle")
            ports = metadata.get("_ports") or (
                handle.get("ucxx_ports") if handle else None
            )

            schema = None
            schema_info = handle.get("schema") if handle else None
            if schema_info:
                from cosmos_rl.utils.payload_transport.ucxx.tensor_spec import (
                    TensorSpec,
                )

                schema = [
                    TensorSpec(
                        name=s["name"],
                        shape=tuple(s["shape"]),
                        dtype=np.dtype(s["dtype"]),
                    )
                    for s in schema_info
                ]

            max_attempts = max(1, self._ucxx_trainer_max_attempts)
            read_timeout = self._ucxx_trainer_read_timeout
            data = None
            for attempt in range(1, max_attempts + 1):
                try:
                    data = await client.read(
                        worker_ip,
                        ucxx_port,
                        slot,
                        schema,
                        ports=ports,
                        timeout=read_timeout,
                    )
                    break
                except Exception as e:
                    if type(e).__name__ not in _TRANSIENT_UCXX_ERRORS:
                        logger.error(
                            f"[UCXXTrainerMixin] Non-transient error reading "
                            f"{worker_ip}:{ucxx_port} slot={slot}: "
                            f"{type(e).__name__}: {e}"
                        )
                        return idx, None
                    if attempt == max_attempts:
                        logger.warning(
                            f"[UCXXTrainerMixin] All {max_attempts} attempts "
                            f"failed for {worker_ip}:{ucxx_port} slot={slot}: "
                            f"{type(e).__name__}: {e}"
                        )
                        return idx, None
                    logger.warning(
                        f"[UCXXTrainerMixin] Transient error reading "
                        f"{worker_ip}:{ucxx_port} slot={slot} "
                        f"(attempt {attempt}/{max_attempts}): "
                        f"{type(e).__name__}, retrying"
                    )
            return idx, data

        def _to_gpu(result: dict) -> dict:
            gpu_data = {}
            for key, value in result.items():
                if hasattr(value, "shape"):
                    gpu_data[key] = torch.from_numpy(value).to(
                        device, non_blocking=True
                    )
                else:
                    gpu_data[key] = value

            ep_len_tensor = gpu_data.get(EPISODE_LENGTH)
            if ep_len_tensor is not None:
                ep_len = (
                    int(ep_len_tensor.item())
                    if ep_len_tensor.numel() == 1
                    else int(ep_len_tensor[0].item())
                )
                for key in (OBSERVATIONS, ACTIONS, REWARDS, TERMINATED, TRUNCATED):
                    if key in gpu_data and gpu_data[key].shape[0] > ep_len:
                        gpu_data[key] = gpu_data[key][:ep_len]
            return gpu_data

        # Build metadata lookup for retries
        meta_by_idx = {}
        for idx, metadata in ucxx_tasks:
            worker_ip = metadata.get("_worker_ip")
            ucxx_port = metadata.get("_ucxx_port")
            slot = metadata.get("_slot")
            if not (worker_ip and ucxx_port and slot is not None):
                continue
            meta_by_idx[idx] = metadata

        pending = list(meta_by_idx.keys())
        batch_results: dict = {}
        total_transfer_ms = 0.0
        total_copy_ms = 0.0

        for round_num in range(_MAX_FETCH_ROUNDS):
            if not pending:
                break

            tasks = [_read_one(idx, meta_by_idx[idx]) for idx in pending]
            failed = []

            for coro in asyncio.as_completed(tasks):
                t0 = get_trace_time()
                idx, result = await coro
                t1 = get_trace_time()
                total_transfer_ms += t1 - t0

                if result is None:
                    failed.append(idx)
                    continue

                gpu_data = _to_gpu(result)
                batch_results[idx] = gpu_data
                t2 = get_trace_time()
                total_copy_ms += t2 - t1

            if failed:
                logger.warning(
                    f"[UCXXTrainerMixin] Fetch round {round_num + 1}/{_MAX_FETCH_ROUNDS}: "
                    f"{len(failed)}/{len(pending)} episodes failed, "
                    f"{'retrying' if round_num + 1 < _MAX_FETCH_ROUNDS else 'giving up'}"
                )
            pending = failed

        if pending:
            logger.error(
                f"[UCXXTrainerMixin] {len(pending)} episodes failed after "
                f"{_MAX_FETCH_ROUNDS} rounds: indices={pending}"
            )

        return batch_results, total_transfer_ms, total_copy_ms

    def fetch_rollouts_ucxx(
        self,
        rollouts: List[Any],
        get_completion: Callable[[Any], Dict] = lambda r: r.completion
        if hasattr(r, "completion")
        else r,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Fetch rollout data via UCXX with background prefetch.

        Args:
            rollouts: List of rollout objects from cosmos-rl
            get_completion: Function to extract completion dict from rollout

        Returns:
            List of trajectory dicts with tensors on device

        Raises:
            RuntimeError: If UCXX not initialized or prefetch fails
        """
        if not self._ucxx_trainer_enabled:
            raise RuntimeError(
                "UCXX trainer not initialized. Call setup_ucxx_trainer() first."
            )

        # Phase 1: Extract metadata and identify UCXX-eligible rollouts
        items = []
        ucxx_tasks = []

        for i, rollout in enumerate(rollouts):
            item = get_completion(rollout)

            if not isinstance(item, dict):
                # Log the first few skipped items to help debug data format issues
                if i < 3:
                    attrs = (
                        [a for a in dir(item) if not a.startswith("_")][:10]
                        if item is not None
                        else []
                    )
                    logger.warning(
                        f"[UCXXTrainerMixin] Skipping rollout {i}: "
                        f"type={type(item).__name__}, sample_attrs={attrs}"
                    )
                items.append(None)
                continue

            items.append(item)

            # Check if UCXX-eligible
            if (
                item.get("_ucxx", False)
                and item.get("_ucxx_enabled")
                and self._ucxx_trainer_request_queue is not None
            ):
                ucxx_tasks.append((i, item))

        # Log Phase 1 summary
        n_skipped = sum(1 for x in items if x is None)
        n_ucxx = len(ucxx_tasks)
        n_fallback = len(items) - n_skipped - n_ucxx
        logger.debug(
            f"[UCXXTrainerMixin] Phase 1: {len(rollouts)} rollouts -> "
            f"{n_ucxx} UCXX, {n_fallback} fallback, {n_skipped} skipped"
        )

        # Phase 2: Submit to prefetch worker and wait
        ucxx_results: Dict[int, Dict[str, torch.Tensor]] = {}
        transfer_ms = 0.0
        copy_ms = 0.0
        fetch_start = get_trace_time()
        if ucxx_tasks and self._ucxx_trainer_request_queue is not None:
            batch_id = self._ucxx_trainer_batch_id
            self._ucxx_trainer_batch_id += 1

            self._ucxx_trainer_request_queue.put((batch_id, ucxx_tasks))

            try:
                result_batch_id, results, transfer_ms, copy_ms = (
                    self._ucxx_trainer_result_queue.get(
                        timeout=self._ucxx_trainer_prefetch_timeout
                    )
                )
                if result_batch_id != batch_id:
                    logger.warning(
                        f"[UCXXTrainerMixin] Batch ID mismatch: expected {batch_id}, got {result_batch_id}"
                    )
                if "_error" in results:
                    logger.warning(
                        f"[UCXXTrainerMixin] Batch {batch_id} prefetch error: "
                        f"{results['_error']}, proceeding with available results"
                    )
                    ucxx_results = {}
                else:
                    ucxx_results = results
            except queue.Empty:
                logger.error(
                    f"[UCXXTrainerMixin] Prefetch timeout after "
                    f"{self._ucxx_trainer_prefetch_timeout}s, proceeding "
                    f"without UCXX data"
                )
                ucxx_results = {}
        fetch_end = get_trace_time()

        # Phase 3: Build results
        ucxx_indices = {i for i, _ in ucxx_tasks}
        result = []
        ucxx_count = 0
        ucxx_skipped = 0
        total_bytes = 0

        for i, item in enumerate(items):
            if item is None:
                continue

            if i in ucxx_results:
                result.append(ucxx_results[i])
                ucxx_count += 1
                for val in ucxx_results[i].values():
                    if isinstance(val, torch.Tensor):
                        total_bytes += val.nelement() * val.element_size()
            elif i in ucxx_indices:
                # UCXX-eligible but fetch failed after all retries — the
                # metadata dict has no tensor data, so skip rather than
                # appending an incomplete dict that would crash training.
                ucxx_skipped += 1
            else:
                # Fallback: data came through cosmos-rl serialization
                processed = {}
                for key, value in item.items():
                    if key.startswith("_"):
                        continue
                    if isinstance(value, torch.Tensor):
                        processed[key] = value.to(self._ucxx_trainer_device)
                    elif hasattr(value, "shape"):
                        processed[key] = torch.from_numpy(value).to(
                            self._ucxx_trainer_device
                        )
                    else:
                        processed[key] = value
                result.append(processed)

        if ucxx_skipped:
            logger.warning(
                f"[UCXXTrainerMixin] {ucxx_skipped} UCXX episodes skipped "
                f"(fetch failed after all retries), batch size reduced to "
                f"{len(result)}"
            )

        if not result:
            raise RuntimeError(
                f"[UCXXTrainerMixin] All {len(rollouts)} episodes failed — "
                f"no data available for training iteration"
            )

        fetch_latency = fetch_end - fetch_start
        wait_ms = max(0.0, fetch_latency - transfer_ms - copy_ms)
        source_replicas = [m.get("_replica_id", "?") for _, m in ucxx_tasks]
        logger.debug(
            f"[Trace] thread=trainer op=ucxx_fetch "
            f"start={fetch_start:.1f} end={fetch_end:.1f} "
            f"wait_ms={wait_ms:.1f} transfer_ms={transfer_ms:.1f} copy_ms={copy_ms:.1f} "
            f"count={ucxx_count} bytes={total_bytes} "
            f"sources={','.join(str(s) for s in source_replicas)}"
        )
        logger.debug(
            f"[UCXXTrainerMixin] Fetched {ucxx_count}/{len(rollouts)} rollouts via UCXX: "
            f"{total_bytes / 1e6:.2f} MB, {fetch_latency:.1f} ms "
            f"(wait={wait_ms:.1f}, transfer={transfer_ms:.1f}, copy={copy_ms:.1f})"
        )

        # Periodic INFO summary to avoid per-iteration log noise
        self._ucxx_trainer_step_count += 1
        self._ucxx_trainer_total_ucxx += ucxx_count
        self._ucxx_trainer_total_fallback += len(rollouts) - ucxx_count - n_skipped
        self._ucxx_trainer_total_bytes += total_bytes
        self._ucxx_trainer_total_latency_ms += fetch_latency
        step = self._ucxx_trainer_step_count
        if step == 1 or step % self._UCXX_LOG_INTERVAL == 0:
            avg_ms = self._ucxx_trainer_total_latency_ms / step
            logger.info(
                f"[UCXXTrainerMixin] Iteration {step}: "
                f"{self._ucxx_trainer_total_ucxx} UCXX / "
                f"{self._ucxx_trainer_total_fallback} fallback, "
                f"{self._ucxx_trainer_total_bytes / 1e6:.1f} MB total, "
                f"avg {avg_ms:.0f} ms/iter"
            )

        return result

    def cleanup_ucxx_trainer(self) -> None:
        """Clean up UCXX trainer resources."""
        # Stop prefetch worker
        if self._ucxx_trainer_shutdown is not None:
            self._ucxx_trainer_shutdown.set()

        if self._ucxx_trainer_prefetch_thread is not None:
            self._ucxx_trainer_prefetch_thread.join(timeout=5.0)
            if self._ucxx_trainer_prefetch_thread.is_alive():
                logger.warning(
                    "[UCXXTrainerMixin] Prefetch worker did not stop cleanly"
                )
            self._ucxx_trainer_prefetch_thread = None

        # Close UCXX client
        if self._ucxx_trainer_client is not None:
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self._ucxx_trainer_client.close())
                loop.close()
            except Exception as e:
                logger.warning(f"[UCXXTrainerMixin] Failed to close UCXX client: {e}")
            self._ucxx_trainer_client = None

        self._ucxx_trainer_enabled = False
        # Final summary
        if self._ucxx_trainer_step_count > 0:
            avg_ms = self._ucxx_trainer_total_latency_ms / self._ucxx_trainer_step_count
            logger.info(
                f"[UCXXTrainerMixin] Final: {self._ucxx_trainer_step_count} iterations, "
                f"{self._ucxx_trainer_total_ucxx} UCXX / "
                f"{self._ucxx_trainer_total_fallback} fallback, "
                f"{self._ucxx_trainer_total_bytes / 1e6:.1f} MB, "
                f"avg {avg_ms:.0f} ms/iter"
            )
        logger.info("[UCXXTrainerMixin] Cleaned up")
