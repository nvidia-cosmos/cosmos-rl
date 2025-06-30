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

"""Pure-Python (ctypes) NCCL backend for ``cosmos_rl``.

A lightweight software watchdog is included.  When the duration of any NCCL
call exceeds the threshold given by the environment variable
``COSMOS_NCCL_TIMEOUT_MS`` (default: 600 000 ms) the corresponding
communicator is aborted via :pyfunc:`nccl_abort` and a ``RuntimeError`` is
raised.

"""

from __future__ import annotations

import glob
import os
import threading
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Callable
import queue

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
def _find_nccl_so_file() -> str:
    """Find the libnccl.so* shared object file from the nvidia-nccl-cu* package."""

    # we assume `nvidia-nccl-cu*` python package is installed next to the torch
    # package (under site-packages directory)
    torch_dir = os.path.dirname(torch.__file__)
    nvidia_nccl_dir = os.path.join(os.path.dirname(torch_dir), "nvidia", "nccl")
    if not os.path.isdir(nvidia_nccl_dir):
        raise RuntimeError(
            f"Could not find `nvidia-nccl-cu*` package directory: {nvidia_nccl_dir}"
            "Please install the `nvidia-nccl-cu*` package."
        )
    # find the so files in nvidia-nccl directory
    so_files = glob.glob(os.path.join(nvidia_nccl_dir, "lib", "libnccl.so*"))
    # filter out the symbolic links
    so_files = [f for f in so_files if not os.path.islink(f)]
    if len(so_files) != 1:
        raise RuntimeError(
            f"Expected exactly one libnccl.so* file in {nvidia_nccl_dir}/lib, "
            f"but found {len(so_files)}: {so_files}. Please check your installation."
        )

    so_file = so_files[0]
    return so_file


_nccl = NCCLLibrary(so_file=_find_nccl_so_file())

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
# Lightweight async enqueue monitoring (Python worker)
# ---------------------------------------------------------------------------


class _Task:
    __slots__ = ("functor", "timeout_ms", "done", "timed_out")

    def __init__(self, functor: Callable[[], ncclComm_t], timeout_ms: int):
        self.functor = functor
        self.timeout_ms = timeout_ms
        self.done = threading.Event()
        self.timed_out = threading.Event()

    # better __repr__ for debug purposes
    def __repr__(self):  # pragma: no cover
        return f"<_Task id={id(self)} timeout_ms={self.timeout_ms}>"


_task_q: "queue.Queue[_Task]" = queue.Queue()
_worker_started = False
_worker_thread: Optional[threading.Thread] = None
_worker_device: Optional[int] = None  # GPU index used by worker

# A lock to guarantee that the NCCL background worker is only started once
_worker_init_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Background worker implementation 
# ---------------------------------------------------------------------------


def _worker_loop(device_idx: int):
    """Background thread that executes queued NCCL host calls on *device_idx*."""
    torch.cuda.set_device(device_idx)

    while True:
        task: _Task = _task_q.get()
        logger.debug(f"[Worker] Got task {task} | queue_size={_task_q.qsize()}")

        comm: ncclComm_t | None = None
        try:
            logger.debug(f"[Worker] Executing functor for task {task}")
            comm = task.functor()
            logger.debug(f"[Worker] Functor for task {task} returned comm={comm}")

            deadline = time.monotonic() + task.timeout_ms / 1000.0
            # Poll async error status until success or timeout.
            while time.monotonic() < deadline:
                logger.debug(f"[Worker] timing out task {task}")
                err = _nccl.ncclCommGetAsyncError(comm)
                logger.debug(f"[Worker] !!!!!!! !!!!! ncclCommGetAsyncError(comm={comm}) = {err}")
                if err == 0:  # ncclSuccess
                    logger.debug(f"[Worker] Task {task} async status success")
                    break
                if err == 7:  # ncclInProgress
                    logger.debug(f"[Worker] Task {task} async status in progress")
                    time.sleep(0.001)
                    continue

                # Immediate error – abort communicator and mark task failed.
                logger.error(
                    f"NCCL: async error detected (err={err}), task {task} failed"
                )
                _nccl.ncclCommAbort(comm)
                task.timed_out.set()
                break
            else:
                # Enqueue timeout hit – abort communicator.
                logger.error(
                    f"NCCL: non-blocking enqueue timed out for task {task}"
                )
                _nccl.ncclCommAbort(comm)
                task.timed_out.set()
        except Exception as e:
            logger.error(f"[Worker] Exception during task {task}: {e}")
            task.timed_out.set()
        finally:
            task.done.set()
            logger.debug(
                f"[Worker] Task {task} done | timed_out={task.timed_out.is_set()}"
            )


def _start_worker(device_idx: int):
    """Start the NCCL background worker thread exactly once (thread-safe)."""
    global _worker_started, _worker_thread, _worker_device

    # Double-checked locking pattern to avoid unnecessary acquisition.
    if _worker_started and _worker_thread is not None and _worker_thread.is_alive():
        return

    with _worker_init_lock:
        if _worker_started and _worker_thread is not None and _worker_thread.is_alive():
            return

        _worker_device = device_idx
        _worker_thread = threading.Thread(
            target=_worker_loop,
            args=(device_idx,),
            daemon=True,
            name="pynccl-worker",
        )
        _worker_thread.start()
        _worker_started = True


def _ensure_worker():
    """Legacy shim retained for compatibility – delegates to _start_worker."""
    _start_worker(torch.cuda.current_device())


def _submit_nccl(functor: Callable[[], ncclComm_t], timeout_ms: Optional[int]):
    """Run NCCL host call in worker thread, monitor enqueue-timeout."""
    if not _worker_started:
        raise RuntimeError(
            "NCCL worker thread not initialized; call create_nccl_comm first."
        )
    resolved_timeout = _get_timeout_ms(timeout_ms)
    task = _Task(functor, resolved_timeout)
    _task_q.put(task)
    task.done.wait()
    if task.timed_out.is_set():
        cur = _current_ctx()
        if cur is not None:
            cur["abort"] = True
        raise RuntimeError("NCCL: non-blocking enqueue timed out")


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
def nccl_timeout_watchdog(
    *, wait_stream: bool = False, timeout_ms: Optional[int] = None
):
    """Light-weight watchdog around a block of NCCL calls."""

    timeout_ms = _get_timeout_ms(timeout_ms)
    logger.debug("[Watchdog] Entered watchdog context")

    ctx = _push_ctx()

    start_ts = time.monotonic()
    cur_stream = torch.cuda.current_stream()

    def _do_abort():
        ctx["abort"] = True
        logger.error(
            f"[Watchdog] NCCL block exceeded {timeout_ms} ms. Aborting its communicators."
        )
        logger.debug(f"[Watchdog] Context comm_ids to abort: {ctx.get('comm_ids', [])}")
        for cid in set(ctx.get("comm_ids", [])):
            try:
                nccl_abort(cid)
            except Exception:
                pass

    timer: Optional[threading.Timer] = None

    if not wait_stream:
        timer = threading.Timer(timeout_ms / 1000.0, _do_abort)
        timer.daemon = True
        timer.start()

    exc: BaseException | None = None
    try:
        yield
    except BaseException as e:
        exc = e
        ctx["abort"] = True
        raise
    finally:
        if timer is not None:
            timer.cancel()
            logger.debug("[Watchdog] Timer canceled")

        timeout_hit = False

        if wait_stream and exc is None:
            logger.debug("[Watchdog] wait_stream=True: flushing current stream")
            evt = torch.cuda.Event()
            evt.record(cur_stream)

            while True:
                if evt.query():
                    logger.debug("[Watchdog] CUDA stream flushed; exiting wait loop")
                    # Stream completely flushed.
                    break

                elapsed_ms = (time.monotonic() - start_ts) * 1000.0
                if not timeout_hit and elapsed_ms >= timeout_ms:
                    timeout_hit = True
                    _do_abort()
                    break

                time.sleep(0.001)  # cooperative yield

        # Pop the context and perform a final abort cleanup if required.
        popped = _pop_ctx()
        logger.debug(f"[Watchdog] Context popped. abort={popped and popped.get('abort')}")
        if popped and popped.get("abort"):
            for cid in set(popped.get("comm_ids", [])):
                try:
                    logger.debug(f"[Watchdog] Aborting communicator {cid}")
                    nccl_abort(cid)
                except Exception:
                    pass
        if timeout_hit and exc is None:
            raise RuntimeError(
                "NCCL operation exceeded watchdog timeout and was aborted"
            )
        logger.debug("[Watchdog] Exiting watchdog context")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_nccl_uid() -> List[int]:
    """Generate a NCCL unique ID and return it as a list of 128 bytes."""
    uid = _nccl.ncclGetUniqueId()
    return list(uid.internal)


def create_nccl_comm(
    uid_chars: List[int], rank: int, world_size: int, timeout_ms: Optional[int] = None
) -> int:
    """Create a communicator and return comm_idx handle (int)."""
    # Start the NCCL background worker exactly once using the caller's CUDA device.
    _start_worker(torch.cuda.current_device())

    uid = ncclUniqueId()
    for i, byte in enumerate(uid_chars):
        uid.internal[i] = byte & 0xFF

    # Holder to fetch communicator created in worker thread
    holder: Dict[str, ncclComm_t] = {}

    def _init_functor() -> ncclComm_t:
        # blocking=0 equivalent not available in wrapper; rely on default
        comm_local = _nccl.ncclCommInitRank(world_size, uid, rank)
        holder["comm"] = comm_local
        return comm_local

    # Run init on worker thread with timeout protection
    _submit_nccl(_init_functor, timeout_ms)

    comm = holder.get("comm")
    if comm is None:
        raise RuntimeError("Failed to create NCCL communicator (worker did not return comm)")

    global _next_comm_idx
    comm_idx = _next_comm_idx
    _next_comm_idx += 1
    _comm_store[comm_idx] = (comm, rank, world_size)
    logger.info(f"[NCCL] Created communicator idx={comm_idx} rank={rank}/{world_size}")

    # Register communicator with current watchdog context (if any)
    cur = _current_ctx()
    if cur is not None:
        cur["comm_ids"].append(comm_idx)

    return comm_idx

# def create_nccl_comm(
#     uid_chars: List[int], rank: int, world_size: int, timeout_ms: Optional[int] = None
# ) -> int:
#     """Create a communicator and return comm_idx handle (int)."""
#     uid = ncclUniqueId()
#     for i, byte in enumerate(uid_chars):
#         uid.internal[i] = byte & 0xFF
#     comm = _nccl.ncclCommInitRank(world_size, uid, rank)
#     global _next_comm_idx
#     comm_idx = _next_comm_idx
#     _next_comm_idx += 1
#     # Save communicator handle together with the caller's rank and world size.
#     _comm_store[comm_idx] = (comm, rank, world_size)
#     logger.info(f"[NCCL] Created communicator idx={comm_idx} rank={rank}/{world_size}")
#     # register communicator with current watchdog context (if any)
#     cur = _current_ctx()
#     if cur is not None:
#         cur["comm_ids"].append(comm_idx)
#     return comm_idx


def get_nccl_comm_nranks(comm_idx: int) -> int:
    """Return world_size of communicator comm_idx."""
    _, _, ws = _comm_store[comm_idx]
    return ws


def nccl_abort(comm_idx: int):
    """Abort (destroy) communicator comm_idx."""
    comm = _comm_store.get(comm_idx, (None, None, None))[0]
    if comm is not None:
        try:
            _nccl.ncclCommAbort(comm)
        except Exception:
            _nccl.ncclCommDestroy(comm)
        logger.warning(f"[NCCL] Aborted communicator idx={comm_idx}")
    _comm_store.pop(comm_idx, None)


# Collective wrapper functions


def nccl_broadcast(
    tensor: torch.Tensor,
    rank: int,
    comm_idx: int,
    stream: Optional[Stream] = None,
    timeout_ms: Optional[int] = None,
):
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

    stream_ptr = _stream_ptr(stream)

    def _broadcast_call():
        _nccl.ncclBroadcast(
            sendbuf,
            recvbuf,
            tensor.numel(),
            _dtype_enum(tensor.dtype),
            rank,
            comm,
            stream_ptr,
        )
        return comm

    _submit_nccl(_broadcast_call, timeout_ms)


def nccl_send(
    tensor: torch.Tensor,
    peer: int,
    comm_idx: int,
    stream: Optional[Stream] = None,
    timeout_ms: Optional[int] = None,
):
    """Point-to-point send."""
    _check_tensor(tensor)
    comm = _comm_store[comm_idx][0]

    stream_ptr = _stream_ptr(stream)

    def _send_call():
        _nccl.ncclSend(
            _buf(tensor),
            tensor.numel(),
            _dtype_enum(tensor.dtype),
            peer,
            comm,
            stream_ptr,
        )
        return comm

    _submit_nccl(_send_call, timeout_ms)


def nccl_recv(
    tensor: torch.Tensor,
    peer: int,
    comm_idx: int,
    stream: Optional[Stream] = None,
    timeout_ms: Optional[int] = None,
):
    """Point-to-point receive."""
    _check_tensor(tensor)
    comm = _comm_store[comm_idx][0]

    stream_ptr = _stream_ptr(stream)

    def _recv_call():
        _nccl.ncclRecv(
            _buf(tensor),
            tensor.numel(),
            _dtype_enum(tensor.dtype),
            peer,
            comm,
            stream_ptr,
        )
        return comm

    _submit_nccl(_recv_call, timeout_ms)


def nccl_allreduce(
    sendbuff: torch.Tensor,
    recvbuff: torch.Tensor,
    op: ReduceOp,
    comm_idx: int,
    stream: Optional[Stream] = None,
    timeout_ms: Optional[int] = None,
):
    """All-reduce collective."""
    _check_tensor(sendbuff)
    _check_tensor(recvbuff)
    comm = _comm_store[comm_idx][0]
    stream_ptr = _stream_ptr(stream)

    def _allreduce_call():
        _nccl.ncclAllReduce(
            _buf(sendbuff),
            _buf(recvbuff),
            sendbuff.numel(),
            _dtype_enum(sendbuff.dtype),
            _redop_enum(op),
            comm,
            stream_ptr,
        )
        return comm

    _submit_nccl(_allreduce_call, timeout_ms)


def nccl_alltoall(
    sendbuff: torch.Tensor,
    recvbuff: torch.Tensor,
    comm_idx: int,
    stream: Optional[Stream] = None,
    timeout_ms: Optional[int] = None,
):
    """All-to-all emulation via AllGather (NCCL native AllToAll not exposed)."""
    _check_tensor(sendbuff)
    _check_tensor(recvbuff)
    comm = _comm_store[comm_idx][0]

    stream_ptr = _stream_ptr(stream)

    def _alltoall_call():
        _nccl.ncclAllGather(
            _buf(sendbuff),
            _buf(recvbuff),
            sendbuff.numel(),
            _dtype_enum(sendbuff.dtype),
            comm,
            stream_ptr,
        )
        return comm

    _submit_nccl(_alltoall_call, timeout_ms)


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
