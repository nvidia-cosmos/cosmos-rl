# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-Python (ctypes) NCCL backend for ``cosmos_rl``.

A lightweight software watchdog is included.  When the duration of any NCCL
call exceeds the threshold given by the environment variable
``COSMOS_NCCL_TIMEOUT_MS`` (default: 600 000 ms) the corresponding
communicator is aborted via :pyfunc:`nccl_abort` and a ``RuntimeError`` is
raised.

"""
from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
from torch.cuda import Stream
from torch.distributed import ReduceOp

from cosmos_rl.utils.pynccl_wrapper import (
    NCCLLibrary,
    buffer_type,
    cudaStream_t,
    ncclComm_t,
    ncclDataTypeEnum,
    ncclRedOpTypeEnum,
    ncclUniqueId,
)

from cosmos_rl.utils.logging import logger

# ---------------------------------------------------------------------------
# NCCL ctypes binding instance (shared)
# ---------------------------------------------------------------------------
_nccl = NCCLLibrary()

# Mapping: comm_idx -> (ncclComm_t, my_rank, world_size)
_comm_store: Dict[int, Tuple[ncclComm_t, int, int]] = {}
_next_comm_idx: int = 0

# ---------------------------------------------------------------------------
# Per-thread watchdog context
# ---------------------------------------------------------------------------

_tls = threading.local()  # will lazily get a .stack attribute


def _push_ctx():
    """Create a new watchdog context and push onto thread-local stack."""
    ctx = {"comm_ids": [], "abort": False}
    if not hasattr(_tls, "stack"):
        _tls.stack = []  # type: ignore[attr-defined]
    _tls.stack.append(ctx)  # type: ignore[attr-defined]
    return ctx


def _pop_ctx():
    """Pop top context; return None if stack empty."""
    if not hasattr(_tls, "stack") or not _tls.stack:  # type: ignore[attr-defined]
        return None
    return _tls.stack.pop()  # type: ignore[attr-defined]


def _current_ctx():
    """Return current watchdog context or None if not inside a block."""
    if hasattr(_tls, "stack") and _tls.stack:  # type: ignore[attr-defined]
        return _tls.stack[-1]  # type: ignore[attr-defined]
    return None


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _dtype_enum(dtype: torch.dtype) -> int:
    """Map torch.dtype to NCCL enum (raises on unsupported)."""
    return ncclDataTypeEnum.from_torch(dtype)


def _redop_enum(op: ReduceOp) -> int:
    """Map torch.distributed.ReduceOp to NCCL enum."""
    return ncclRedOpTypeEnum.from_torch(op)


def _stream_ptr(stream: Optional[Stream] = None) -> cudaStream_t:
    """Return cudaStream_t pointer for given stream (defaults to current)."""
    if stream is None:
        stream = torch.cuda.current_stream()
    return cudaStream_t(stream.cuda_stream)


def _buf(ptr_tensor: Optional[torch.Tensor]) -> buffer_type:
    """Return void* pointer for a tensor (or null pointer if None)."""
    return buffer_type(0) if ptr_tensor is None else buffer_type(ptr_tensor.data_ptr())


def _check_tensor(tensor: torch.Tensor):
    """Validate that tensor is CUDA, contiguous and on current device."""
    if not tensor.is_cuda:
        raise ValueError("Tensor must be CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError("Tensor must be contiguous")
    if tensor.numel() == 0:
        raise ValueError("Tensor must have non-zero number of elements")
    if tensor.device.index != torch.cuda.current_device():
        raise ValueError("Tensor device mismatch current CUDA device")


def _get_timeout_ms(user_timeout: Optional[int] = None) -> int:
    """Resolve timeout value (environment variable overrides default)."""
    if user_timeout is not None:
        return user_timeout
    return int(os.getenv("COSMOS_NCCL_TIMEOUT_MS", "600000"))  # 10 minutes default


# ---------------------------------------------------------------------------
# Context-manager
# ---------------------------------------------------------------------------

@contextmanager
def nccl_timeout_watchdog(*, wait_stream: bool = False, timeout_ms: Optional[int] = None):
    """Context-manager that aborts all NCCL comms if block exceeds timeout_ms.

    Parameters
    ----------
    wait_stream : bool
        If True the watchdog measures kernel completion time by synchronising
        the CUDA stream before evaluating the timeout.  If False it measures
        enqueue latency only.
    timeout_ms : int | None
        Custom timeout; falls back to `COSMOS_NCCL_TIMEOUT_MS` when None.
    """
    timeout_ms = _get_timeout_ms(timeout_ms)

    # push new context
    ctx = _push_ctx()

    # Record a CPU‐side monotonic timestamp as the authoritative timeout reference.
    start_ts = time.monotonic()


    def _timeout_action():
        if wait_stream:
            evt = torch.cuda.Event()
            evt.record()

            while not evt.query():
                if (time.monotonic() - start_ts) * 1000.0 >= timeout_ms:
                    ctx["abort"] = True
                    logger.error(
                        f"[Watchdog] NCCL block exceeded {timeout_ms} ms "
                        "(kernel-completion latency on current stream). Will abort its communicators."
                    )
                    break
                # Yield the GIL briefly to avoid busy-waiting.
                time.sleep(0.001)

    timer = threading.Timer(timeout_ms / 1000.0, _timeout_action)
    timer.start()

    exc = None
    try:
        yield
    except BaseException as e:
        exc = e
        ctx["abort"] = True
        raise
    finally:
        timer.cancel()
        popped = _pop_ctx()
        if popped and popped["abort"]:
            for cid in set(popped["comm_ids"]):
                try:
                    nccl_abort(cid)
                except Exception:
                    pass
        if exc is not None:
            pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_nccl_uid() -> List[int]:
    """Generate a NCCL unique ID and return it as a list of 128 bytes."""
    uid = _nccl.ncclGetUniqueId()
    return list(uid.internal)


def create_nccl_comm(uid_chars: List[int], rank: int, world_size: int, timeout_ms: Optional[int] = None) -> int:
    """Create a communicator and return comm_idx handle (int)."""
    uid = ncclUniqueId()
    for i, byte in enumerate(uid_chars):
        uid.internal[i] = byte & 0xFF
    comm = _nccl.ncclCommInitRank(world_size, uid, rank)
    global _next_comm_idx
    comm_idx = _next_comm_idx
    _next_comm_idx += 1
    # Save communicator handle together with the caller's rank and world size.
    _comm_store[comm_idx] = (comm, rank, world_size)
    logger.info(f"[NCCL] Created communicator idx={comm_idx} rank={rank}/{world_size}")
    # register communicator with current watchdog context (if any)
    cur = _current_ctx()
    if cur is not None:
        cur["comm_ids"].append(comm_idx)
    return comm_idx


def get_nccl_comm_nranks(comm_idx: int) -> int:
    """Return world_size of communicator comm_idx."""
    _, _, ws = _comm_store[comm_idx]
    return ws


def nccl_abort(comm_idx: int):
    """Abort (destroy) communicator comm_idx."""
    comm = _comm_store.get(comm_idx, (None, None, None))[0]
    if comm is not None:
        _nccl.ncclCommDestroy(comm)
        logger.warning(f"[NCCL] Aborted communicator idx={comm_idx}")
    _comm_store.pop(comm_idx, None)


# Collective wrapper functions

def nccl_broadcast(tensor: torch.Tensor, rank: int, comm_idx: int, stream: Optional[Stream] = None, timeout_ms: Optional[int] = None):
    """Broadcast tensor from rank (root) to all peers in communicator.

    Parameters
    ----------
    tensor : torch.Tensor
        Buffer to send/receive. It must reside on the current CUDA device.
    rank : int
        Rank that owns the valid send buffer within the communicator.
    comm_idx : int
        Handle returned by :func:`create_nccl_comm`.
    stream : torch.cuda.Stream | None
        CUDA stream where the collective is launched (defaults to current).
    timeout_ms : int | None
        Reserved for future watchdog-based timeout handling.
    """
    _check_tensor(tensor)
    comm, my_rank, _ = _comm_store[comm_idx]

    # Only the root rank provides a valid send buffer. All ranks – including the
    # root – must supply a valid receive buffer so that they all obtain the
    # broadcasted data.
    sendbuf = _buf(tensor) if my_rank == rank else buffer_type()
    recvbuf = _buf(tensor)

    _nccl.ncclBroadcast(
        sendbuf,
        recvbuf,
        tensor.numel(),
        _dtype_enum(tensor.dtype),
        rank,
        comm,
        _stream_ptr(stream),
    )


def nccl_send(tensor: torch.Tensor, peer: int, comm_idx: int, stream: Optional[Stream] = None, timeout_ms: Optional[int] = None):
    """Point-to-point send."""
    _check_tensor(tensor)
    comm = _comm_store[comm_idx][0]
    _nccl.ncclSend(
        _buf(tensor),
        tensor.numel(),
        _dtype_enum(tensor.dtype),
        peer,
        comm,
        _stream_ptr(stream),
    )


def nccl_recv(tensor: torch.Tensor, peer: int, comm_idx: int, stream: Optional[Stream] = None, timeout_ms: Optional[int] = None):
    """Point-to-point receive."""
    _check_tensor(tensor)
    comm = _comm_store[comm_idx][0]
    _nccl.ncclRecv(
        _buf(tensor),
        tensor.numel(),
        _dtype_enum(tensor.dtype),
        peer,
        comm,
        _stream_ptr(stream),
    )


def nccl_allreduce(sendbuff: torch.Tensor, recvbuff: torch.Tensor, op: ReduceOp, comm_idx: int, stream: Optional[Stream] = None, timeout_ms: Optional[int] = None):
    """All-reduce collective."""
    _check_tensor(sendbuff)
    _check_tensor(recvbuff)
    comm = _comm_store[comm_idx][0]
    _nccl.ncclAllReduce(
        _buf(sendbuff),
        _buf(recvbuff),
        sendbuff.numel(),
        _dtype_enum(sendbuff.dtype),
        _redop_enum(op),
        comm,
        _stream_ptr(stream),
    )


def nccl_alltoall(sendbuff: torch.Tensor, recvbuff: torch.Tensor, comm_idx: int, stream: Optional[Stream] = None, timeout_ms: Optional[int] = None):
    """All-to-all emulation via AllGather (NCCL native AllToAll not exposed)."""
    _check_tensor(sendbuff)
    _check_tensor(recvbuff)
    comm = _comm_store[comm_idx][0]
    _nccl.ncclAllGather(
        _buf(sendbuff),
        _buf(recvbuff),
        sendbuff.numel(),
        _dtype_enum(sendbuff.dtype),
        comm,
        _stream_ptr(stream),
    )



# Compatibility helper (legacy API surface)

def get_nccl_timeout_ms() -> int:
    """Public helper that mirrors the old pynccl.get_nccl_timeout_ms API."""
    return _get_timeout_ms()


__all__ = [
    # management
    "create_nccl_uid",
    "create_nccl_comm",
    "nccl_abort",
    "get_nccl_comm_nranks",
    # collectives
    "nccl_broadcast",
    "nccl_send",
    "nccl_recv",
    "nccl_allreduce",
    "nccl_alltoall",
    # watchdog
    "nccl_timeout_watchdog",
    # compatibility helper
    "get_nccl_timeout_ms",
] 