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

"""UCXX mixins for rollout workers.

Active mixins:

1. :class:`UCXXRolloutMixin` -- rollout workers: write to SharedRingBuffer
   and serve over UCXX.

The trainer-side counterpart lives in
:class:`cosmos_rl.utils.payload_transport.ucxx.data_packer_mixin.UCXXDataPackerMixin`,
which subclasses :class:`PrefetchDataPackerMixin` and slots into the
DataPacker protocol's ``get_policy_input()`` rather than coupling to a
specific trainer class.  An earlier ``UCXXTrainerMixin`` (deprecated
and superseded by ``UCXXDataPackerMixin``) was removed in this PR.
"""

import fcntl
import socket
import struct
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.payload_transport.ucxx.ucxx_buffer import (
    UCXX_AVAILABLE,
    UCXXBuffer,
    UCXXBufferConfig,
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
