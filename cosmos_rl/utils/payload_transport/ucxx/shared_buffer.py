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

"""Shared-memory ring buffer used by the UCXX payload transport.

Tensors are copied directly to shared memory (``memcpy`` via numpy) with
no pickle in the fast path: each schema-described entry has a fixed
layout so the writer and reader simply ``memmove`` bytes.

The buffer uses a four-state slot machine
(``FREE → WRITING → READY → READING → FREE``) implemented in the entry
metadata header, so producers and consumers can coordinate without
shared-memory locks beyond a single per-process :class:`threading.Lock`
guarding the header.
"""

import os
import struct
import socket
from dataclasses import dataclass, field
from enum import IntEnum
from multiprocessing import shared_memory
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.payload_transport.ucxx.tensor_spec import TensorSpec


class SlotError(Exception):
    """Exception for buffer slot operations."""

    pass


class SlotState(IntEnum):
    """
    State machine for buffer slots.

    Transitions:
        FREE -> WRITING: Writer acquires slot
        WRITING -> READY: Writer completes write
        READY -> READING: Reader starts reading
        READING -> FREE: Reader completes and marks consumed

        Overwrite path (when buffer full):
        READY -> WRITING: Writer overwrites unconsumed slot (logs warning)
    """

    FREE = 0  # Slot is available for writing
    WRITING = 1  # Writer is actively writing to slot
    READY = 2  # Data is ready to be read
    READING = 3  # Reader is actively reading from slot


@dataclass
class BufferMetrics:
    """Metrics for buffer operations."""

    writes_total: int = 0
    reads_total: int = 0
    drops_total: int = 0  # Overwrites of unconsumed data

    @property
    def current_fill(self) -> int:
        """Approximate fill level (writes - reads - drops)."""
        return max(0, self.writes_total - self.reads_total - self.drops_total)


@dataclass
class BufferConfig:
    """Configuration for shared buffer."""

    max_entries: int = 64
    entry_size_bytes: int = 65536  # Fallback if no schema provided
    buffer_name: str = ""

    # Schema: list of TensorSpecs defining what tensors are stored
    # If empty, falls back to pickle-based serialization
    schema: List[TensorSpec] = field(default_factory=list)

    def __post_init__(self):
        if not self.buffer_name:
            self.buffer_name = f"cosmos_rl_shm_{os.getpid()}_{id(self)}"


class SharedRingBuffer:
    """
    Fast shared memory ring buffer for inter-process RL data transfer.

    Each entry represents ONE COMPLETE ROLLOUT (trajectory) with a fixed schema.
    All entries have identical size, enabling direct memcpy without serialization.

    Typical schema for RL:
        observations:   (max_steps, obs_dim)    float32
        actions:        (max_steps, action_dim) float32
        rewards:        (max_steps,)            float32
        terminated:     (max_steps,)            bool
        truncated:      (max_steps,)            bool
        episode_length: (1,)                    int64  <- actual valid steps

    Variable-length episodes are padded to max_steps; episode_length indicates
    how many steps are valid (the rest is zero-padding).

    Operations:
        Write: memcpy from source tensor to shared array (no serialization)
        Read:  return view into shared memory (true fast read)

    Memory layout:
        [Header: 24 bytes (write_idx, read_idx, entry_count)]
        [Entry 0 metadata: 16 bytes][Entry 0 tensors...]
        [Entry 1 metadata: 16 bytes][Entry 1 tensors...]
        ...

    Entry metadata: (size: u64, state: u8 (SlotState), padding)

    Slot State Machine:
        FREE -> WRITING -> READY -> READING -> FREE
        (Overwrite path: READY -> WRITING when buffer full)
    """

    HEADER_SIZE = 24  # write_idx, read_idx, entry_count (3 x u64)
    ENTRY_META_SIZE = 16  # size (u64) + state (u8) + padding (7 bytes)

    def __init__(self, config: BufferConfig, create: bool = False):
        self.config = config
        self.buffer_name = config.buffer_name
        self.max_entries = config.max_entries
        self.schema = config.schema

        # Calculate sizes
        if self.schema:
            # Schema-based: fixed layout for each entry
            self.entry_data_size = sum(spec.nbytes for spec in self.schema)
        else:
            # Fallback: variable size entries (still uses pickle)
            self.entry_data_size = config.entry_size_bytes

        self.entry_size = self.ENTRY_META_SIZE + self.entry_data_size
        self.total_size = self.HEADER_SIZE + (self.max_entries * self.entry_size)

        # Create/attach shared memory
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._lock = Lock()
        self._is_creator = create

        # Metrics (local to this process, not shared)
        self._metrics = BufferMetrics()

        self._init_shm(create)

        # Pre-compute tensor offsets within each entry
        self._tensor_offsets: Dict[str, int] = {}
        if self.schema:
            offset = 0
            for spec in self.schema:
                self._tensor_offsets[spec.name] = offset
                offset += spec.nbytes

    def _init_shm(self, create: bool) -> None:
        """Initialize shared memory segment."""
        try:
            if create:
                # Clean up any existing segment with same name
                try:
                    old_shm = shared_memory.SharedMemory(name=self.buffer_name)
                    old_shm.close()
                    old_shm.unlink()
                except FileNotFoundError:
                    pass

                self._shm = shared_memory.SharedMemory(
                    name=self.buffer_name, create=True, size=self.total_size
                )
                # Initialize header
                self._write_header(0, 0, 0)

                # Initialize all slots to FREE state
                for i in range(self.max_entries):
                    self._write_entry_meta(i, 0, SlotState.FREE)

                logger.info(
                    f"[SharedRingBuffer] Created '{self.buffer_name}' "
                    f"({self.total_size / 1024:.1f} KB, {self.max_entries} entries)"
                )
            else:
                self._shm = shared_memory.SharedMemory(name=self.buffer_name)
                logger.info(f"[SharedRingBuffer] Attached to '{self.buffer_name}'")
        except Exception as e:
            logger.error(f"[SharedRingBuffer] Failed to create/attach: {e}")
            raise

    def _write_header(self, write_idx: int, read_idx: int, entry_count: int) -> None:
        """Write header to shared memory."""
        struct.pack_into("QQQ", self._shm.buf, 0, write_idx, read_idx, entry_count)

    def _read_header(self) -> Tuple[int, int, int]:
        """Read header from shared memory."""
        return struct.unpack_from("QQQ", self._shm.buf, 0)

    def _entry_offset(self, index: int) -> int:
        """Get byte offset for entry metadata."""
        return self.HEADER_SIZE + (index * self.entry_size)

    def _entry_data_offset(self, index: int) -> int:
        """Get byte offset for entry data."""
        return self._entry_offset(index) + self.ENTRY_META_SIZE

    def _write_entry_meta(self, index: int, size: int, state: SlotState) -> None:
        """Write entry metadata with state."""
        offset = self._entry_offset(index)
        # Format: Q (u64 size) + B (u8 state) + 7 bytes padding = 16 bytes
        struct.pack_into("QB", self._shm.buf, offset, size, state)

    def _read_entry_meta(self, index: int) -> Tuple[int, SlotState]:
        """Read entry metadata."""
        offset = self._entry_offset(index)
        size, state_val = struct.unpack_from("QB", self._shm.buf, offset)
        return size, SlotState(state_val)

    def _try_transition(
        self, index: int, from_state: SlotState, to_state: SlotState
    ) -> bool:
        """
        Attempt atomic state transition for a slot.

        Args:
            index: Slot index
            from_state: Expected current state
            to_state: Desired new state

        Returns:
            True if transition succeeded, False if current state didn't match.
        """
        with self._lock:
            size, current_state = self._read_entry_meta(index)
            if current_state != from_state:
                return False
            self._write_entry_meta(index, size, to_state)
            return True

    def get_slot_state(self, index: int) -> SlotState:
        """Get the current state of a slot."""
        _, state = self._read_entry_meta(index)
        return state

    def write(self, data: Dict[str, Any], overwrite_if_full: bool = True) -> int:
        """
        Write data to buffer (fast for tensors).

        Args:
            data: Dict of tensors/arrays. Keys must match schema if schema is defined.
            overwrite_if_full: If True, overwrite oldest unconsumed entry when full.
                               If False, raise SlotError when full.

        Returns:
            Slot index where data was written.

        Raises:
            SlotError: If buffer is full and overwrite_if_full=False.
        """
        with self._lock:
            write_idx, read_idx, entry_count = self._read_header()

            # Get slot (ring buffer)
            slot = write_idx % self.max_entries

            # Check slot state and handle appropriately
            _, current_state = self._read_entry_meta(slot)

            if current_state == SlotState.READY:
                # Slot has unconsumed data
                if entry_count >= self.max_entries:
                    if not overwrite_if_full:
                        raise SlotError(f"Buffer full, slot {slot} not consumed")
                    # Overwrite mode: log warning and continue (data loss is acceptable
                    # for RL - old trajectories become stale anyway)
                    logger.warning(
                        f"[SharedRingBuffer] Overwriting unconsumed slot {slot} "
                        f"write_idx={write_idx} entry_count={entry_count}"
                    )
                    self._metrics.drops_total += 1
            elif current_state == SlotState.WRITING:
                raise SlotError(f"Slot {slot} is being written by another process")
            elif current_state == SlotState.READING:
                raise SlotError(f"Slot {slot} is being read, cannot overwrite")

            # Transition to WRITING state
            self._write_entry_meta(slot, 0, SlotState.WRITING)

            data_offset = self._entry_data_offset(slot)

            if self.schema:
                # Fast path: write tensors directly
                self._write_tensors(data, data_offset)
                size = self.entry_data_size
            else:
                # Fallback: pickle (for backward compatibility)
                import pickle

                serialized = pickle.dumps(data)
                size = len(serialized)
                if size > self.entry_data_size:
                    # Reset state on error
                    self._write_entry_meta(slot, 0, SlotState.FREE)
                    raise SlotError(
                        f"Data size {size} exceeds entry size {self.entry_data_size}"
                    )
                self._shm.buf[data_offset : data_offset + size] = serialized

            # Transition to READY state
            self._write_entry_meta(slot, size, SlotState.READY)

            # Update header
            new_count = min(entry_count + 1, self.max_entries)
            self._write_header(write_idx + 1, read_idx, new_count)

            # Update metrics
            self._metrics.writes_total += 1

            return slot

    def write_raw(self, buf: bytes, overwrite_if_full: bool = True) -> int:
        """Write a pre-packed contiguous buffer to the next slot.

        This is the fast path used by ``UCXXRolloutMixin.write_to_buffer``:
        the caller has already coalesced all tensors into a single
        byte-buffer matching the schema layout, so we skip per-tensor
        iteration and do a single ``memoryview`` copy (one ``memmove``).

        Args:
            buf: Contiguous bytes/memoryview whose length equals
                ``self.entry_data_size``.
            overwrite_if_full: Same semantics as ``write()``.

        Returns:
            Slot index where data was written.
        """
        if len(buf) != self.entry_data_size:
            raise SlotError(
                f"Raw buffer size {len(buf)} != expected {self.entry_data_size}"
            )

        with self._lock:
            write_idx, read_idx, entry_count = self._read_header()
            slot = write_idx % self.max_entries

            _, current_state = self._read_entry_meta(slot)
            if current_state == SlotState.READY:
                if entry_count >= self.max_entries:
                    if not overwrite_if_full:
                        raise SlotError(f"Buffer full, slot {slot} not consumed")
                    logger.warning(
                        f"[SharedRingBuffer] Overwriting unconsumed slot {slot} "
                        f"write_idx={write_idx} entry_count={entry_count}"
                    )
                    self._metrics.drops_total += 1
            elif current_state == SlotState.WRITING:
                raise SlotError(f"Slot {slot} is being written by another process")
            elif current_state == SlotState.READING:
                raise SlotError(f"Slot {slot} is being read, cannot overwrite")

            self._write_entry_meta(slot, 0, SlotState.WRITING)

            data_offset = self._entry_data_offset(slot)
            self._shm.buf[data_offset : data_offset + self.entry_data_size] = buf

            self._write_entry_meta(slot, self.entry_data_size, SlotState.READY)
            new_count = min(entry_count + 1, self.max_entries)
            self._write_header(write_idx + 1, read_idx, new_count)
            self._metrics.writes_total += 1
            return slot

    def _write_tensors(self, data: Dict[str, Any], base_offset: int) -> None:
        """Write tensors directly to shared memory (no serialization)."""
        for spec in self.schema:
            if spec.name not in data:
                raise SlotError(f"Missing tensor '{spec.name}' in data")

            tensor = data[spec.name]

            # Convert to numpy if needed
            if isinstance(tensor, torch.Tensor):
                arr = tensor.detach().cpu().numpy()
            elif isinstance(tensor, np.ndarray):
                arr = tensor
            else:
                # Scalar or other - convert to numpy
                arr = np.array(tensor, dtype=spec.dtype)

            # Ensure correct dtype and contiguous
            if arr.dtype != spec.dtype:
                arr = arr.astype(spec.dtype)
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)

            # Create view into shared memory and copy directly (memcpy, no tobytes())
            offset = base_offset + self._tensor_offsets[spec.name]

            # Get destination as numpy array view
            dst = np.ndarray(
                shape=spec.shape, dtype=spec.dtype, buffer=self._shm.buf, offset=offset
            )

            # Reshape source to match destination
            src = arr.reshape(spec.shape) if arr.shape != spec.shape else arr

            # Direct memcpy via numpy (TRUE fast write)
            np.copyto(dst, src)

    def is_ready(self, index: int) -> bool:
        """Check if an entry is ready to read."""
        _, state = self._read_entry_meta(index)
        return state == SlotState.READY

    def get_ready_count(self) -> int:
        """Get number of slots currently in ``READY`` state.

        Counts slot states directly rather than returning the header's
        ``entry_count``: ``entry_count`` is monotonic up to
        ``max_entries`` and saturates after the first ring lap, so it
        does **not** reflect how many slots are actually ready to be
        consumed at a given instant.  Use this method when you need
        the live ready-slot count (e.g. for diagnostics); use
        :meth:`is_full` for the full/not-full predicate.
        """
        return len(self.get_ready_indices())

    def is_full(self) -> bool:
        """Check if buffer is full."""
        _, _, entry_count = self._read_header()
        return entry_count >= self.max_entries

    def read(self, index: int) -> Dict[str, Any]:
        """
        Read data from buffer (fast for tensors).

        Transitions slot state: READY -> READING (during read) -> stays READING
        Call mark_consumed() after processing to transition to FREE.

        Args:
            index: Slot index to read from.

        Returns:
            Dict of numpy arrays (copies from shared memory).

        Raises:
            SlotError: If entry is not ready.
        """
        with self._lock:
            size, state = self._read_entry_meta(index)

            if state != SlotState.READY:
                raise SlotError(f"Entry {index} not ready (state={state.name})")

            # Transition to READING state
            self._write_entry_meta(index, size, SlotState.READING)

        data_offset = self._entry_data_offset(index)

        if self.schema:
            # Fast path: read tensors directly
            result = self._read_tensors(data_offset)
        else:
            # Fallback: pickle
            import pickle

            serialized = bytes(self._shm.buf[data_offset : data_offset + size])
            result = pickle.loads(serialized)

        # Update metrics
        self._metrics.reads_total += 1

        return result

    def _read_tensors(self, base_offset: int) -> Dict[str, np.ndarray]:
        """Read tensors directly from shared memory (fast when possible)."""
        result = {}

        for spec in self.schema:
            offset = base_offset + self._tensor_offsets[spec.name]

            # Create numpy array view into shared memory (TRUE fast read)
            # Note: This is a view, not a copy. Modifications affect shared memory.
            arr = np.ndarray(
                shape=spec.shape, dtype=spec.dtype, buffer=self._shm.buf, offset=offset
            )

            # Return a copy to be safe (caller may hold reference after we move on)
            # For true fast, caller should process immediately
            result[spec.name] = arr.copy()

        return result

    def try_read(self, index: int) -> Optional[Dict[str, Any]]:
        """
        Try to read data from buffer, return None if not ready.

        Non-blocking version of read() for consumer-faster-than-producer scenarios.
        Transitions slot state: READY -> READING if successful.
        Call mark_consumed() after processing to transition to FREE.

        Args:
            index: Slot index to read from.

        Returns:
            Dict of numpy arrays, or None if entry not ready.
        """
        with self._lock:
            size, state = self._read_entry_meta(index)

            if state != SlotState.READY:
                return None

            # Transition to READING state
            self._write_entry_meta(index, size, SlotState.READING)

        data_offset = self._entry_data_offset(index)

        if self.schema:
            result = self._read_tensors(data_offset)
        else:
            import pickle

            serialized = bytes(self._shm.buf[data_offset : data_offset + size])
            result = pickle.loads(serialized)

        # Update metrics
        self._metrics.reads_total += 1

        return result

    def get_ready_indices(self) -> List[int]:
        """
        Get list of slot indices that are ready to be consumed.

        Returns:
            List of slot indices with ready data (state == READY).
        """
        ready_indices = []
        for i in range(self.max_entries):
            _, state = self._read_entry_meta(i)
            if state == SlotState.READY:
                ready_indices.append(i)
        return ready_indices

    def read_view(self, index: int) -> Dict[str, np.ndarray]:
        """
        Read data as views into shared memory (TRUE fast).

        WARNING: Returned arrays are views. They become invalid after mark_consumed()
        or if the slot is overwritten. Process immediately!

        Transitions slot state: READY -> READING.
        Call mark_consumed() after processing to transition to FREE.

        Args:
            index: Slot index to read from.

        Returns:
            Dict of numpy array views into shared memory.

        Raises:
            SlotError: If entry is not ready.
        """
        with self._lock:
            size, state = self._read_entry_meta(index)

            if state != SlotState.READY:
                raise SlotError(f"Entry {index} not ready (state={state.name})")

            if not self.schema:
                raise SlotError("read_view() requires schema-based buffer")

            # Transition to READING state
            self._write_entry_meta(index, size, SlotState.READING)

        data_offset = self._entry_data_offset(index)
        result = {}

        for spec in self.schema:
            offset = data_offset + self._tensor_offsets[spec.name]

            # True fast: numpy view into shared memory
            result[spec.name] = np.ndarray(
                shape=spec.shape, dtype=spec.dtype, buffer=self._shm.buf, offset=offset
            )

        # Update metrics
        self._metrics.reads_total += 1

        return result

    def read_raw(self, index: int) -> np.ndarray:
        """Return the raw bytes of a slot as a contiguous uint8 array view.

        Transitions slot state: READY -> READING on first call.

        Read-side counterpart of :meth:`write_raw`; used by the UCXX
        server's :meth:`UCXXBuffer._handle_connection` for the single
        coalesced ``send(raw_buf)`` per slot.  Each slot read has a
        single intended owner: the handler that performed the
        ``READY -> READING`` transition is responsible for finalising
        with :meth:`mark_consumed` (success) or
        :meth:`release_reading` (failure).

        Re-entry from ``READING`` is **tolerated, not encouraged** --
        if a stale orphan handler is still mid-``send`` when the
        client times out and rotates to a fresh server thread, the
        new handler can re-acquire a view (the writer cannot recycle
        a READING slot, so the bytes are stable).  The defensive
        guards in :meth:`mark_consumed` and :meth:`release_reading`
        make the duplicate-finalise call from the orphan handler a
        no-op.
        """
        with self._lock:
            size, state = self._read_entry_meta(index)

            if state == SlotState.READY:
                if not self.schema:
                    raise SlotError("read_raw() requires schema-based buffer")
                self._write_entry_meta(index, size, SlotState.READING)
            elif state != SlotState.READING:
                raise SlotError(f"Entry {index} not ready (state={state.name})")

        data_offset = self._entry_data_offset(index)
        self._metrics.reads_total += 1
        return np.ndarray(
            shape=(self.entry_data_size,),
            dtype=np.uint8,
            buffer=self._shm.buf,
            offset=data_offset,
        )

    def mark_consumed(self, index: int) -> None:
        """
        Mark entry as consumed (slot can be reused).

        Transitions slot state: READING -> FREE.

        Defensive against stale callers: if the slot is not currently
        in READING state the call is a no-op.  Mirrors the early-
        return guard in :meth:`release_reading`.  Without this guard a
        stale ``mark_consumed`` (e.g. an orphan reader exiting after
        the slot has been recycled by the writer) would silently
        clobber a slot that is mid-WRITE or already filled with a new
        payload, dropping data.
        """
        with self._lock:
            _, state = self._read_entry_meta(index)

            if state != SlotState.READING:
                logger.warning(
                    f"[SharedRingBuffer] mark_consumed on slot {index} with unexpected "
                    f"state {state.name} (expected READING) -- ignoring (probable stale "
                    f"reader exit after slot recycle)"
                )
                return

            # Transition to FREE state
            self._write_entry_meta(index, 0, SlotState.FREE)
            logger.debug(
                f"[SharedRingBuffer] mark_consumed slot={index} (READING->FREE)"
            )

    def release_reading(self, index: int) -> None:
        """
        Release slot from READING back to READY (data preserved for retry).

        Called when a send fails mid-transfer. The data is still valid in
        shared memory, so transitioning back to READY allows a new handler
        to read_view() the same slot on the client's retry.

        Transitions slot state: READING -> READY.
        """
        with self._lock:
            size, state = self._read_entry_meta(index)

            if state != SlotState.READING:
                logger.warning(
                    f"[SharedRingBuffer] release_reading on slot {index} with "
                    f"unexpected state {state.name} (expected READING)"
                )
                return

            self._write_entry_meta(index, size, SlotState.READY)

    def get_metrics(self) -> BufferMetrics:
        """
        Get buffer metrics.

        Note: Metrics are local to this process and not shared across processes.
        Each process maintains its own metrics for operations it performs.

        Returns:
            BufferMetrics with writes_total, reads_total, drops_total, current_fill.
        """
        return self._metrics

    def get_slot_states(self) -> Dict[int, SlotState]:
        """
        Get states of all slots (for debugging/monitoring).

        Returns:
            Dict mapping slot index to SlotState.
        """
        states = {}
        for i in range(self.max_entries):
            _, state = self._read_entry_meta(i)
            states[i] = state
        return states

    def get_handle(self) -> Dict[str, Any]:
        """Get serializable handle for buffer discovery."""
        return {
            "type": "cpu_shm",
            "buffer_name": self.buffer_name,
            "max_entries": self.max_entries,
            "entry_size": self.entry_data_size,
            "node_id": socket.gethostname(),
            "schema": [
                {"name": s.name, "shape": s.shape, "dtype": str(s.dtype)}
                for s in self.schema
            ]
            if self.schema
            else None,
        }

    @classmethod
    def from_handle(cls, handle: Dict[str, Any]) -> "SharedRingBuffer":
        """Create buffer from handle (attach to existing)."""
        schema = []
        if handle.get("schema"):
            for s in handle["schema"]:
                schema.append(
                    TensorSpec(
                        name=s["name"],
                        shape=tuple(s["shape"]),
                        dtype=np.dtype(s["dtype"]),
                    )
                )

        config = BufferConfig(
            buffer_name=handle["buffer_name"],
            max_entries=handle["max_entries"],
            entry_size_bytes=handle["entry_size"],
            schema=schema,
        )
        return cls(config, create=False)

    def close(self) -> None:
        """Close shared memory (doesn't unlink)."""
        if self._shm:
            self._shm.close()
            self._shm = None

    def unlink(self) -> None:
        """Unlink (delete) shared memory. Only creator should call this.

        Can be called before or after close(). Safe to call multiple times.
        """
        if not self._is_creator:
            return

        # Try to unlink via _shm if still open
        if self._shm:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass
            self._shm = None
        else:
            # Already closed, unlink by name
            try:
                from multiprocessing.shared_memory import SharedMemory

                shm = SharedMemory(name=self.buffer_name, create=False)
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass

    def __del__(self):
        if self._is_creator:
            # Creator should unlink to clean up /dev/shm segment
            self.unlink()
        else:
            self.close()
