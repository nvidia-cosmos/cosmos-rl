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

"""Producer-side UCXX payload transport.

The trainer-side counterpart lives in
:class:`cosmos_rl.utils.payload_transport.ucxx.data_packer_mixin.UCXXDataPackerMixin`,
which subclasses :class:`PrefetchDataPackerMixin` and slots into the
DataPacker protocol's ``get_policy_input()`` rather than coupling to a
specific trainer class. The deprecated ``UCXXTrainerMixin`` is not part of the
producer surface.
"""

import fcntl
import socket
import struct
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.payload_transport.layout import TensorLayout, WIRE_VERSION
from cosmos_rl.utils.payload_transport.storage import WritablePayload
from cosmos_rl.utils.payload_transport.ucxx.ucxx_buffer import (
    UCXX_AVAILABLE,
    UCXXBuffer,
    UCXXBufferConfig,
    reset_ucxx_context,
)
from cosmos_rl.utils.payload_transport.ucxx.operation import UCXXOperation

from cosmos_rl.utils.payload_transport.pack import (
    PackStats,
    pack_trajectory_into,
)
from cosmos_rl.utils.trace import get_trace_time
from cosmos_rl.utils.trajectory import (
    EPISODE_LENGTH,
    REWARDS,
    build_trajectory_schema,
    episode_length,
)


# Failed CUDA publication may leave asynchronous kernels holding source,
# staging, or pinned-host pointers. A Python exception is not device-completion
# proof, so these owners intentionally remain reachable for process lifetime.
_FAILED_PACK_OWNERS = []


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


class UCXXPayloadProducer:
    """Own UCXX publication, serving and teardown for one rollout producer."""

    _ucxx_buffer: Optional[UCXXBuffer] = None
    _ucxx_replica_id: str = ""
    _ucxx_enabled: bool = False
    _ucxx_ip: str = ""
    _ucxx_port: int = 0
    _ucxx_max_steps: int = 100
    _ucxx_obs_dim: int = 4
    _ucxx_action_dim: int = 2
    _ucxx_packed_cpu: Optional[torch.Tensor] = None  # Pinned CPU staging buffer
    _ucxx_gpu_staging: Optional[Dict[torch.device, torch.Tensor]] = None
    _ucxx_tensor_offsets: Optional[Dict[str, int]] = None
    _ucxx_entry_data_size: int = 0

    @property
    def default_layout(self) -> TensorLayout:
        """Return the layout compiled during setup."""
        if not self._ucxx_enabled:
            raise RuntimeError("UCXX payload producer is not attached")
        return self._ucxx_layout

    def metrics(self) -> Dict[str, Any]:
        """Return producer counters without exposing the ring buffer itself."""
        if self._ucxx_buffer is None:
            return {}
        counters = self._ucxx_buffer.get_metrics()
        return {
            "writes_total": counters.writes_total,
            "reads_total": counters.reads_total,
            "drops_total": counters.drops_total,
            "ready_slots": self._ucxx_buffer._buffer.get_ready_count(),
            "max_slots": self._ucxx_buffer._buffer.max_entries,
            **dict(getattr(self, "_ucxx_payload_metrics", {})),
        }

    def _record_pack(self, stats: PackStats) -> None:
        metrics = self.__dict__.setdefault("_ucxx_payload_metrics", {})
        for name, value in (
            ("adapter_payloads", 1),
            ("adapter_fields", stats.fields),
            ("adapter_bytes", stats.bytes),
            ("adapter_copy_batches", stats.batched_calls),
            ("adapter_fallback_fields", stats.fallback_fields),
        ):
            metrics[name] = metrics.get(name, 0) + value

    def acquire(self, *, layout: Optional[TensorLayout] = None) -> WritablePayload:
        """Reserve a shared-memory slot and expose typed CPU views into it."""
        if not self._ucxx_enabled or self._ucxx_buffer is None:
            raise RuntimeError("UCXX payload producer is not attached")
        selected = layout or self._ucxx_layout
        if selected != self._ucxx_layout:
            raise ValueError(
                "UCXX producer layout is fixed when its shared buffer is created"
            )
        slot, generation, raw = self._ucxx_buffer.acquire_write_slot()
        storage = torch.from_numpy(raw)
        selected.zero_padding(storage)
        return WritablePayload(
            owner=self,
            layout=selected,
            storage=storage,
            tensors=selected.views(storage),
            opaque=(slot, generation),
            on_abandon=lambda: self._ucxx_buffer.abandon_write_slot(slot, generation),
        )

    def register_layout(self, layout: TensorLayout) -> None:
        """Validate the single layout fixed by this producer's shared ring."""
        if not self._ucxx_enabled:
            raise RuntimeError("UCXX payload producer is not attached")
        if layout != self._ucxx_layout:
            raise ValueError(
                "UCXX producer layout is fixed when its shared buffer is created"
            )

    def setup(self, *args, **kwargs) -> None:
        """Own partial setup and reject replacement of a live producer."""
        from cosmos_rl.utils.payload_transport.lifecycle import TransportClose

        previous = getattr(self, "_ucxx_close_operation", None)
        if self._ucxx_enabled or (previous is not None and not previous.completed):
            raise RuntimeError("UCXX transport is already attached or still closing")
        self._ucxx_write_lock = threading.Lock()
        self._ucxx_close_operation = TransportClose(self._close_owned)
        try:
            UCXXPayloadProducer._setup(self, *args, **kwargs)
        except BaseException:
            try:
                self.close()
            except Exception:
                logger.exception("UCXX attachment rollback failed")
            raise

    def _setup(
        self,
        replica_id: str,
        max_steps: int,
        obs_dim: int,
        action_dim: int,
        port: int = 0,
        config: Optional[UCXXBufferConfig] = None,
        schema: Optional[list] = None,
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
                "UCXX is required for UCXXPayloadProducer. "
                "Install with: pip install ucxx-cu12"
            )

        self._ucxx_replica_id = replica_id
        self._ucxx_max_steps = max_steps
        self._ucxx_obs_dim = obs_dim
        self._ucxx_action_dim = action_dim
        self._ucxx_sender_incarnation = uuid.uuid4().hex
        self._ucxx_gpu_staging = {}

        try:
            # Build schema for trajectory data (required for zero-pack protocol).
            # Shared with NCCL so both transports agree on the wire layout.
            schema = (
                list(schema)
                if schema is not None
                else build_trajectory_schema(
                    {
                        "max_steps": max_steps,
                        "obs_dim": obs_dim,
                        "action_dim": action_dim,
                    }
                )
            )

            # Create UCXX buffer with server and schema
            buffer_config = config or UCXXBufferConfig(
                max_entries=1000,
                entry_size_bytes=65536,
            )
            buffer_config.buffer_name = f"ucxx_rollout_{replica_id}"
            buffer_config.port = port  # Set port in config
            buffer_config.schema = schema  # Schema required for zero-pack protocol
            buffer_config.sender_incarnation = self._ucxx_sender_incarnation

            self._ucxx_layout = TensorLayout.compile(schema)
            buffer_config.layout = self._ucxx_layout

            self._ucxx_buffer = UCXXBuffer(buffer_config)

            # Pre-compute schema layout for coalesced writes
            self._ucxx_tensor_offsets = self._ucxx_layout.offsets()
            self._ucxx_entry_data_size = self._ucxx_layout.payload_nbytes
            self._ucxx_schema = schema
            self._ucxx_payload_metrics = {}

            # Pre-allocate pinned CPU staging buffer for bulk D2H + SHM copy.
            # Pinned memory enables DMA and avoids per-tensor cudaMemcpy overhead.
            self._ucxx_packed_cpu = torch.empty(
                self._ucxx_entry_data_size, dtype=torch.uint8, pin_memory=True
            )

            # Start UCXX server
            self._ucxx_server_start_attempted = True
            self._ucxx_buffer.start_server()
            self._ucxx_ip = self._ucxx_buffer.local_ip
            self._ucxx_port = self._ucxx_buffer.port

            self._ucxx_enabled = True
            logger.info(
                f"[UCXXPayloadProducer] Worker '{replica_id}' ready at "
                f"{self._ucxx_ip}:{self._ucxx_port} "
                f"(max_steps={max_steps}, obs_dim={obs_dim}, action_dim={action_dim}, "
                f"entry_size={self._ucxx_entry_data_size / 1e6:.1f} MB)"
            )

        except Exception as e:
            logger.error("UCXX setup failed: %s", e)
            raise

    def publish(
        self,
        trajectory: Dict[str, Any],
        *,
        layout: Optional[TensorLayout] = None,
        copy_policy: str = "batched",
    ) -> Optional[Dict[str, Any]]:
        with getattr(self, "_ucxx_write_lock", threading.Lock()):
            return self._publish(trajectory, layout=layout, copy_policy=copy_policy)

    def _publish(
        self,
        trajectory: Dict[str, Any],
        *,
        layout: Optional[TensorLayout] = None,
        copy_policy: str = "batched",
    ) -> Optional[Dict[str, Any]]:
        """Write trajectory to buffer with padding and return metadata for UCXX fetch.

        Uses a coalesced write strategy for large payloads:
        1. Pack all schema tensors into a single contiguous GPU buffer
        2. One bulk ``cudaMemcpy`` D2H into a pre-allocated pinned CPU buffer
        3. One bulk ``memoryview`` copy into the SHM slot

        This eliminates per-tensor Python loop overhead and reaches closer to
        hardware bandwidth limits (PCIe for D2H, DDR for SHM).

        CPU tensors are packed directly into a reserved shared-memory slot.
        """
        if not self._ucxx_enabled or self._ucxx_buffer is None:
            return None

        try:
            layout = layout or self._ucxx_layout
            if layout != self._ucxx_layout:
                raise ValueError(
                    "UCXX producer layout is fixed when its shared buffer is created"
                )
            ep_len = episode_length(
                trajectory, self._ucxx_schema, default=self._ucxx_max_steps
            )
            rewards = trajectory.get(REWARDS, [])
            reward_values = rewards.tolist() if hasattr(rewards, "tolist") else rewards

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

                staging = self._ucxx_gpu_staging
                if staging is None:
                    staging = self._ucxx_gpu_staging = {}
                gpu_packed = staging.get(device)
                if gpu_packed is None:
                    gpu_packed = torch.empty(
                        self._ucxx_entry_data_size,
                        dtype=torch.uint8,
                        device=device,
                    )
                    staging[device] = gpu_packed
                pinned_packed = self._ucxx_packed_cpu
                try:
                    layout.zero_padding(gpu_packed)

                    stats = pack_trajectory_into(
                        gpu_packed,
                        trajectory,
                        self._ucxx_schema,
                        self._ucxx_tensor_offsets,
                        ep_len,
                        device=device,
                        copy_policy=copy_policy,
                    )
                    self._record_pack(stats)

                    # Single bulk D2H copy into pinned staging buffer
                    pinned_packed.copy_(gpu_packed, non_blocking=False)

                    t_gpu2cpu_end = get_trace_time()

                    # Single bulk SHM write
                    slot = self._ucxx_buffer.write_raw(
                        memoryview(pinned_packed.numpy())
                    )
                    t_shm_end = get_trace_time()
                    total_bytes = self._ucxx_entry_data_size
                    slot_generation = self._ucxx_buffer.get_slot_generation(slot)
                except BaseException:
                    staging.pop(device, None)
                    self._ucxx_packed_cpu = None
                    _FAILED_PACK_OWNERS.append(
                        (self, trajectory, gpu_packed, pinned_packed)
                    )
                    raise
            else:
                # --- CPU path: batched copies straight into reserved SHM ---
                # Reserving first avoids both the materialized normalized dict
                # and the second per-field copy performed by SharedRingBuffer.
                slot, slot_generation, raw = self._ucxx_buffer.acquire_write_slot()
                try:
                    storage = torch.from_numpy(raw)
                    layout.zero_padding(storage)
                    t_gpu2cpu_end = t_gpu2cpu_start
                    stats = pack_trajectory_into(
                        storage,
                        trajectory,
                        self._ucxx_schema,
                        self._ucxx_tensor_offsets,
                        ep_len,
                        device="cpu",
                        copy_policy=copy_policy,
                    )
                    self._record_pack(stats)
                    self._ucxx_buffer.publish_write_slot(slot, slot_generation)
                    t_shm_end = get_trace_time()
                    # The server transfers the complete aligned slot. Keep the
                    # trace comparable with the GPU path and with receive-side
                    # accounting; adapter_bytes separately reports field bytes.
                    total_bytes = self._ucxx_entry_data_size
                except BaseException:
                    self._ucxx_buffer.abandon_write_slot(slot, slot_generation)
                    raise

            gpu2cpu_ms = t_gpu2cpu_end - t_gpu2cpu_start
            shm_ms = t_shm_end - t_gpu2cpu_end
            thread_name = threading.current_thread().name
            logger.debug(
                f"[Trace] thread={thread_name} op=ucxx_write "
                f"start={t_gpu2cpu_start:.1f} end={t_shm_end:.1f} "
                f"gpu2cpu_ms={gpu2cpu_ms:.1f} shm_ms={shm_ms:.1f} "
                f"bytes={total_bytes}"
            )

            return self._build_reference(
                slot,
                slot_generation,
                layout,
                reward_values,
                ep_len,
            )
        except Exception as e:
            if copy_policy == "strict_batched":
                raise
            logger.error(f"[UCXXPayloadProducer] Publish failed: {e}")
            return None

    def publish_storage(
        self,
        payload: WritablePayload,
        *,
        ready_event: Any = None,
    ) -> Dict[str, Any]:
        """Publish a directly populated shared-memory slot without repacking."""
        with getattr(self, "_ucxx_write_lock", threading.Lock()):
            if not self._ucxx_enabled or self._ucxx_buffer is None:
                raise RuntimeError("UCXX payload producer is not attached")
            slot, generation = payload._submit(self)
            readiness = None
            try:
                if ready_event is not None:
                    readiness = UCXXOperation(
                        self._ucxx_buffer.config.send_timeout,
                        "UCXX producer payload readiness",
                        owners=(self, payload, self._ucxx_buffer),
                    )
                    try:
                        readiness.deadline.wait_event(ready_event)
                        readiness.complete()
                    except BaseException as error:
                        readiness.fail(
                            f"producer readiness completion uncertain: {error}"
                        )
                ep_len = episode_length(
                    payload.tensors,
                    payload.layout.schema(),
                    default=self._ucxx_max_steps,
                )
                rewards = payload.tensors.get(REWARDS, [])
                reward_values = (
                    rewards.tolist() if hasattr(rewards, "tolist") else rewards
                )
                reference = self._build_reference(
                    slot,
                    generation,
                    payload.layout,
                    reward_values,
                    ep_len,
                )
                self._ucxx_buffer.publish_write_slot(slot, generation)
                metrics = self.__dict__.setdefault("_ucxx_payload_metrics", {})
                metrics["direct_payloads"] = metrics.get("direct_payloads", 0) + 1
            except BaseException:
                if readiness is None or readiness.failure is None:
                    self._ucxx_buffer.abandon_write_slot(slot, generation)
                raise
            return reference

    def _build_reference(
        self,
        slot: int,
        generation: int,
        layout: TensorLayout,
        rewards: Any,
        ep_len: int,
    ) -> Dict[str, Any]:
        return {
            "_ucxx": True,
            "_ucxx_enabled": True,
            "_worker_ip": self._ucxx_ip,
            "_ucxx_port": self._ucxx_port,
            "_ports": self._ucxx_buffer.ports,
            "_slot": slot,
            "_slot_generation": generation,
            "_replica_id": self._ucxx_replica_id,
            "_wire_version": WIRE_VERSION,
            "_layout_id": layout.layout_id,
            "_sender_incarnation": self._ucxx_sender_incarnation,
            REWARDS: rewards,
            EPISODE_LENGTH: ep_len,
        }

    def close(self, timeout: float = 5.0) -> None:
        """Close with a caller deadline while the operation retains ownership."""
        from cosmos_rl.utils.payload_transport.lifecycle import get_close_operation

        # Stop new writers before waiting for the current write to finish.
        self._ucxx_server_start_attempted = self._ucxx_enabled or getattr(
            self, "_ucxx_server_start_attempted", False
        )
        self._ucxx_enabled = False
        operation = getattr(self, "_ucxx_close_operation", None)
        if operation is None:
            operation = get_close_operation(
                self, "_ucxx_close_operation", self._close_owned
            )
        operation.close(timeout)

    def _close_owned(self) -> None:
        """Clean up UCXX resources.

        Order matters:

        1. ``stop_server`` retires this producer's listeners, endpoints and
           request waiters on their owning loops, then joins the threads and
           releases its global-context lease.
        2. Only the last quiescent owner resets the context and joins its native
           progress thread. Other producers/pooled clients retain their leases;
           their progress must continue. The explicit reset below also covers
           partial startup and refuses an unproven untracked-context drain.
        3. ``close()`` releases this producer's buffer once its own native users
           are quiescent, even if another owner's progress thread is still live.

        Reset also covers partial server startup. A failed server join retains
        the buffer and prevents context reset or shared-memory release.
        """
        with getattr(self, "_ucxx_write_lock", threading.Lock()):
            started_ucxx = self._ucxx_enabled or getattr(
                self, "_ucxx_server_start_attempted", False
            )
            self._ucxx_server_start_attempted = started_ucxx
            self._ucxx_enabled = False
            if self._ucxx_buffer:
                self._ucxx_buffer.stop_server()
                if started_ucxx:
                    reset_ucxx_context()
                self._ucxx_buffer.close()
                self._ucxx_buffer = None
            self._ucxx_server_start_attempted = False
            self._ucxx_packed_cpu = None
            self._ucxx_gpu_staging = None
            logger.info(
                "[UCXXPayloadProducer] Worker '%s' cleaned up",
                getattr(self, "_ucxx_replica_id", "?"),
            )
