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

"""Async weight synchronization for disaggregated rollout workers.

This module implements an opt-in asynchronous weight sync path where P2R
(policy-to-rollout) and R2R (rollout-to-rollout) weight transfers execute
on a dedicated background thread with its own CUDA stream.  Weights are
written into a *buffer model* (a parameter-only clone) and copied to the
live model at explicit sync points -- either before each
``rollout_generation()`` call or before each policy inference call.

Architecture
------------

::

    Main thread                    WeightSyncThread
    ───────────                    ────────────────
    rollout_generation()           ← P2R/R2R commands from controller
      └─ sync_buffer_to_live()        └─ execute on weight_sync_stream
           copy buffer → live              write into buffer_state_dict
           (on inference_stream)           record CUDA event
                                           bump _buffer_version

Enabling
--------
Set ``[rollout].async_r2r_sync`` to ``"generation"`` or ``"inference"``
in the experiment config.  Default is ``"disabled"`` (synchronous path).

- ``generation``: sync buffer to live before each ``rollout_generation()``.
- ``inference``: additionally sync before each policy forward pass.

The ``[rollout].broadcast_all_params`` toggle (default ``false``) controls
whether R2R broadcasts all model parameters or only trainable ones.  Set
to ``true`` for models with non-trainable params that must be synced
(e.g. frozen vision encoders).
"""

from __future__ import annotations

import os
import queue
import threading
import time
from enum import Enum
from typing import TYPE_CHECKING, Optional

import torch
from torch.distributed.tensor import DTensor

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.pynccl import (
    bounded_drain_or_abort,
    nccl_abort_all,
    nccl_broadcast,
    nccl_group_end,
    nccl_group_start,
)

# Bounded wait for in-flight GPU work on the WeightSyncThread stream during
# teardown.  A grouped R2R broadcast is enqueued asynchronously and pynccl only
# bounds the enqueue phase, so a broadcast whose peer departed can hang on the
# device with no watchdog.  After this deadline we abort all NCCL comms so
# ``stop()`` (and in turn ``destroy_distributed``) cannot wedge.
_WST_STREAM_DRAIN_TIMEOUT_S = float(
    os.getenv("COSMOS_WST_STREAM_DRAIN_TIMEOUT_S", "10.0")
)
_WST_QUEUE_DRAIN_TIMEOUT_S = float(
    os.getenv("COSMOS_WST_QUEUE_DRAIN_TIMEOUT_S", "120.0")
)

if TYPE_CHECKING:
    pass

try:
    import redis as _redis_lib
except ImportError:
    _redis_lib = None


class AsyncR2RSyncMode(Enum):
    """When to synchronize the async R2R broadcast with inference.

    - ``DISABLED``: R2R runs synchronously on ``inference_stream``.
    - ``GENERATION``: R2R runs on a separate CUDA stream; buffer is synced
      to live model before each ``rollout_generation()`` call.
    - ``INFERENCE``: Like GENERATION, but also syncs before each policy
      inference call inside the rollout servicer.
    """

    DISABLED = "disabled"
    GENERATION = "generation"
    INFERENCE = "inference"


_R2R_BARRIER_TIMEOUT_S = 120
_SYNC_NOOP_LOG_INTERVAL = 50


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def get_async_r2r_sync_mode(worker) -> AsyncR2RSyncMode:
    """Read ``async_r2r_sync`` from ``[rollout]`` in worker config."""
    return AsyncR2RSyncMode(worker.config.rollout.async_r2r_sync)


def get_broadcast_all_params(worker) -> bool:
    """Read ``broadcast_all_params`` from ``[rollout]`` in worker config."""
    return worker.config.rollout.broadcast_all_params


# ---------------------------------------------------------------------------
# Buffer model helpers
# ---------------------------------------------------------------------------


def create_buffer_model(worker, device=None) -> None:
    """Create a parameter-only buffer by cloning live model state_dict."""
    model = worker.rollout.get_underlying_model()
    target_device = device or next(model.parameters()).device
    buffer_sd: dict[str, torch.Tensor] = {}
    for name, param in model.state_dict().items():
        buffer_sd[name] = param.detach().clone().to(target_device)
    worker._buffer_state_dict = buffer_sd
    # Monotonic counters: _buffer_version is bumped by WeightSyncThread,
    # _buffer_synced_version by sync_buffer_to_live on the main thread.
    # CPython GIL guarantees atomic int reads/writes, so no lock is needed.
    worker._buffer_version = 0
    worker._buffer_synced_version = 0
    total_bytes = sum(t.nelement() * t.element_size() for t in buffer_sd.values())
    logger.info(
        "[WeightSync] Created buffer model: %d tensors, %.1f MB on %s",
        len(buffer_sd),
        total_bytes / (1024 * 1024),
        target_device,
    )


def _storage_extent(tensor: torch.Tensor) -> int:
    """Number of storage elements ``tensor`` spans past its storage offset."""
    if tensor.numel() == 0:
        return 0
    return 1 + sum(
        (size - 1) * stride for size, stride in zip(tensor.size(), tensor.stride())
    )


def _rebuild_over_buffer(view_tensor, sd, buffer_sd, storage_to_sd_keys):
    """Return a buffer-backed equivalent of ``view_tensor``, or ``None``.

    Resolves the base parameter by untyped-storage identity, then either
    maps the base's buffer clone directly (when ``view_tensor`` is the
    base itself under another name) or rebuilds the same view over the
    clone.
    """
    if not isinstance(view_tensor, torch.Tensor) or isinstance(view_tensor, DTensor):
        return None
    storage_key = (view_tensor.device, view_tensor.untyped_storage().data_ptr())
    view_offset = view_tensor.storage_offset()
    for sd_key in storage_to_sd_keys.get(storage_key, ()):
        live_base = sd[sd_key]
        buffer_base = buffer_sd.get(sd_key)
        if buffer_base is None or buffer_base.dtype != view_tensor.dtype:
            continue
        # The entry is the base parameter under another name: its buffer
        # clone is a standalone same-shape tensor, so map it directly (the
        # clone of a non-dense base has different strides, which is fine
        # for a whole-tensor receive target).
        if (
            view_tensor.size() == live_base.size()
            and view_tensor.stride() == live_base.stride()
            and view_offset == live_base.storage_offset()
        ):
            return buffer_base
        # Otherwise rebuild the view over the clone.  The clone's storage
        # starts at offset zero, so rebase the view's offset against the
        # base; strides only transfer when the clone preserved the base's
        # layout, and the rebuilt view must stay inside the base's extent.
        relative_offset = view_offset - live_base.storage_offset()
        if (
            relative_offset >= 0
            and live_base.stride() == buffer_base.stride()
            and buffer_base.storage_offset() == 0
            and relative_offset + _storage_extent(view_tensor)
            <= _storage_extent(live_base)
        ):
            return torch.as_strided(
                buffer_base,
                view_tensor.size(),
                view_tensor.stride(),
                relative_offset,
            )
    return None


def redirect_view_map_to_buffer(worker) -> None:
    """Replace weight_inplace_view_map entries with buffer_model tensors.

    After this call, P2R nccl_recv writes directly into the buffer
    tensors instead of the live model parameters.

    Raises ``RuntimeError`` if any entry cannot be redirected: a receive
    target left on the live model would race inference, and its received
    updates would be overwritten with stale buffer data by the next
    ``sync_buffer_to_live``.
    """
    buffer_sd = worker._buffer_state_dict
    old_map = worker.weight_inplace_view_map
    model = worker.rollout.get_underlying_model()

    # View entries (e.g. the q/k/v slices of a fused qkv parameter) do not
    # appear in the state dict under their own key, and their data_ptr sits
    # somewhere inside the base parameter's storage. Resolve them by storage
    # identity and rebuild the same view over the buffer's clone of the base
    # tensor; matching by data_ptr alone would map an offset-zero view to the
    # full base tensor and leave nonzero-offset views unredirected.
    sd = model.state_dict()
    storage_to_sd_keys: dict[tuple[torch.device, int], list[str]] = {}
    for name, tensor in sd.items():
        if not isinstance(tensor, torch.Tensor) or isinstance(tensor, DTensor):
            # DTensor storages have no accessible data pointer; DTensor
            # entries are redirected by the exact-name match below.
            continue
        storage_to_sd_keys.setdefault(
            (tensor.device, tensor.untyped_storage().data_ptr()), []
        ).append(name)

    new_map: dict[str, torch.Tensor] = {}
    view_redirected = 0
    failed: list[str] = []
    for hf_key, view_tensor in old_map.items():
        if hf_key in buffer_sd and buffer_sd[hf_key].shape == view_tensor.shape:
            new_map[hf_key] = buffer_sd[hf_key]
            continue
        buffered = _rebuild_over_buffer(view_tensor, sd, buffer_sd, storage_to_sd_keys)
        if buffered is not None:
            new_map[hf_key] = buffered
            view_redirected += 1
            continue
        failed.append(hf_key)

    if failed:
        raise RuntimeError(
            f"[WeightSync] Could not redirect {len(failed)}/{len(old_map)} view "
            f"map entries to buffer tensors (first entries: {failed[:5]}). "
            "Async weight sync requires every receive target to be "
            "buffer-backed: a live-model target would race inference and its "
            "received updates would be overwritten with stale data by the "
            'next buffer->live sync. Set rollout.async_r2r_sync="disabled" '
            "for this model."
        )

    worker.weight_inplace_view_map = new_map
    logger.info(
        "[WeightSync] Redirected all %d view map entries to buffer tensors "
        "(%d rebuilt as views over buffer base tensors)",
        len(old_map),
        view_redirected,
    )


def sync_buffer_to_live(worker) -> None:
    """Copy buffer params to live model if a new version is available.

    This is a pure data-plane operation: it copies tensors from the
    buffer model into the live model.  It does **not** trigger
    validation or shutdown — those are handled by
    ``process_wst_deferred_actions`` on the main thread.

    Non-blocking on CPU.  inference_stream.wait_event(last_event) ensures
    the GPU-side copy executes after the most recently completed write on
    the weight-sync stream.
    """
    buf_ver = getattr(worker, "_buffer_version", 0)
    synced_ver = getattr(worker, "_buffer_synced_version", 0)
    if buf_ver <= synced_ver:
        cnt = getattr(worker, "_sync_noop_cnt", 0) + 1
        worker._sync_noop_cnt = cnt
        if cnt == 1 or cnt % _SYNC_NOOP_LOG_INTERVAL == 0:
            logger.debug(
                "[WeightSync] sync_buffer_to_live: no-op (buf_ver=%d, "
                "synced_ver=%d, noop_count=%d)",
                buf_ver,
                synced_ver,
                cnt,
            )
        return

    worker._sync_noop_cnt = 0
    wst: WeightSyncThread | None = getattr(worker, "_weight_sync_thread", None)
    has_event = wst is not None and wst._last_event is not None

    inf_stream = worker.inference_stream
    if has_event:
        inf_stream.wait_event(wst._last_event)

    live_sd = getattr(worker, "_live_state_dict_cache", None)
    if live_sd is None:
        model = worker.rollout.get_underlying_model()
        live_sd = model.state_dict()
        worker._live_state_dict_cache = live_sd
    buffer_sd = worker._buffer_state_dict
    t0 = time.monotonic()
    with torch.cuda.stream(inf_stream):
        for name in live_sd:
            if name in buffer_sd:
                live_sd[name].copy_(buffer_sd[name])
    worker._buffer_synced_version = buf_ver
    elapsed_ms = (time.monotonic() - t0) * 1000
    # ``superseded`` = buffer versions transferred since the last adopt but
    # jumped over here (never adopted into the live model) -> wasted NCCL
    # transfers.  Coalescing (Phase 2) should drive this to ~0.
    superseded = max(0, buf_ver - synced_ver - 1)
    logger.info(
        "[WeightSync] Synced buffer -> live (%d params, ver %d->%d, "
        "superseded=%d, wait_event=%s, %.1f ms CPU enqueue)",
        len(live_sd),
        synced_ver,
        buf_ver,
        superseded,
        has_event,
        elapsed_ms,
    )


def process_wst_deferred_actions(worker) -> None:
    """Handle validation and shutdown flags set by the WeightSyncThread.

    Must be called on the main thread only (never from inference
    callbacks).  The WST sets lightweight flags when it completes a
    broadcast that requires validation or shutdown; this function
    reacts to those flags.
    """
    if getattr(worker, "_pending_validation_step", None) is not None:
        worker._pending_validation_step = None
        if worker.validation_flag.is_set():
            worker.do_validation()
    if getattr(worker, "_pending_shutdown", False):
        worker._pending_shutdown = False
        data = {"is_end": True, "prompt_idx": -1, "completion_token_ids": []}
        worker.redis_controller.publish_teacher_request(data, worker.replica_name)
        logger.info("[WeightSync] Published end event to reference")
        if worker.validation_flag.is_set():
            worker.do_validation()
        worker.shutdown_signal.set()
        worker.shutdown_mp_signal.set()


# ---------------------------------------------------------------------------
# WeightSyncThread
# ---------------------------------------------------------------------------


class WeightSyncThread:
    """Background thread executing P2R and R2R on a dedicated CUDA stream.

    P2R commands have higher priority (0) than R2R commands (1).
    The thread writes into ``_buffer_state_dict``; the live model is
    never touched.

    P2R is executed by calling ``worker._execute_p2r_recv(command, stream)``
    directly with the WST's own CUDA stream, avoiding any stream-swap hacks.
    """

    def __init__(self, worker):
        self._worker = worker
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._seq = 0
        self._stream = torch.cuda.Stream()
        self._stop = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._last_event: torch.cuda.Event | None = None
        self._fence_failed = False
        self._fenced_seq = -1
        self._task_failed = False
        # Backlog observability (see weight-sync coalescing plan): high-water
        # queue depth and total executed transfers.  A healthy (coalesced) run
        # keeps ``_max_qdepth`` ~1; a piled-up run shows it climbing while most
        # transfers are superseded before they are ever adopted.
        self._max_qdepth = 0
        self._executed = 0
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="weight-sync",
        )

    def start(self) -> None:
        """Start the background thread."""
        self._thread.start()
        logger.info("[WeightSyncThread] Started background thread")

    def enqueue_p2r(self, command) -> None:
        """Enqueue a P2R command with highest priority."""
        self._seq += 1
        self._idle.clear()
        self._queue.put((0, self._seq, ("p2r", command)))
        qdepth = self._queue.qsize()
        if qdepth > self._max_qdepth:
            self._max_qdepth = qdepth
        logger.info(
            "[WeightSyncThread] Enqueued P2R (step=%s, seq=%d, buf_ver=%d, "
            "qdepth=%d, max_qdepth=%d)",
            getattr(command, "weight_step", "?"),
            self._seq,
            getattr(self._worker, "_buffer_version", -1),
            qdepth,
            self._max_qdepth,
        )

    def enqueue_r2r(self, command) -> None:
        """Enqueue an R2R command with lower priority."""
        self._seq += 1
        self._idle.clear()
        self._queue.put((1, self._seq, ("r2r", command)))
        qdepth = self._queue.qsize()
        if qdepth > self._max_qdepth:
            self._max_qdepth = qdepth
        logger.info(
            "[WeightSyncThread] Enqueued R2R (step=%s, seq=%d, buf_ver=%d, "
            "qdepth=%d, max_qdepth=%d)",
            getattr(command, "weight_step", "?"),
            self._seq,
            getattr(self._worker, "_buffer_version", -1),
            qdepth,
            self._max_qdepth,
        )

    def fence(
        self,
        queue_timeout: float = _WST_QUEUE_DRAIN_TIMEOUT_S,
        stream_timeout: float = _WST_STREAM_DRAIN_TIMEOUT_S,
    ) -> bool:
        """Fence every command received before STOP.

        Queue completion only proves that the background thread has enqueued
        the CUDA work.  The stream fence is therefore ordered strictly after
        ``queue.join()``.  A queue timeout is an abnormal teardown path: abort
        NCCL, attempt the bounded stream drain, and report failure instead of
        allowing the caller to treat a warning as successful synchronization.
        """
        if getattr(self, "_fence_failed", False):
            return False

        current_seq = getattr(self, "_seq", None)
        if (
            current_seq is not None
            and getattr(self, "_fenced_seq", None) == current_seq
            and getattr(self._queue, "unfinished_tasks", 1) == 0
        ):
            return True

        done = threading.Event()

        def _join_with_timeout():
            self._queue.join()
            done.set()

        t = threading.Thread(target=_join_with_timeout, daemon=True)
        t.start()
        queue_drained = done.wait(timeout=queue_timeout)
        if not queue_drained:
            logger.error(
                "[ABNORMAL teardown] WeightSyncThread[%s] queue did not "
                "drain within %.1fs; aborting NCCL before shutdown",
                self._worker.replica_name,
                queue_timeout,
            )
            try:
                nccl_abort_all()
            except Exception:
                logger.exception(
                    "[WeightSyncThread] NCCL abort failed after queue timeout"
                )

        stream_drained = bounded_drain_or_abort(
            self._stream,
            stream_timeout,
            f"WeightSyncThread[{self._worker.replica_name}]",
        )
        result = (
            queue_drained
            and stream_drained
            and not getattr(self, "_task_failed", False)
        )
        self._fence_failed = not result
        self._fenced_seq = current_seq
        return result

    def drain(self, timeout: float = _WST_QUEUE_DRAIN_TIMEOUT_S) -> bool:
        """Compatibility wrapper for the full queue-and-stream fence."""
        return self.fence(queue_timeout=timeout)

    def stop(self) -> bool:
        """Signal the thread to stop and wait for it to finish.

        A Python thread join is not sufficient: a grouped R2R broadcast
        enqueued on ``self._stream`` runs asynchronously on the GPU, and pynccl
        only bounds the *enqueue* phase (``run_task`` stops polling after
        ``ncclSuccess``).  Fence the queue and stream before asking the thread
        to exit so no accepted task is dropped; timeout paths abort NCCL and
        report failure.
        """
        fenced = self.fence()
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            logger.error(
                "[ABNORMAL teardown] WeightSyncThread[%s] did not stop "
                "within 10s; aborting NCCL",
                self._worker.replica_name,
            )
            try:
                nccl_abort_all()
            except Exception:
                logger.exception(
                    "[WeightSyncThread] NCCL abort failed after thread timeout"
                )
            fenced = False
        # A timed-out task can leave its barrier only after _stop is set. Drain
        # once more after the thread exits so no CUDA work can appear behind
        # the earlier timeout-path drain.
        post_stop_drained = bounded_drain_or_abort(
            self._stream,
            _WST_STREAM_DRAIN_TIMEOUT_S,
            f"WeightSyncThread[{self._worker.replica_name}] post-stop",
        )
        return fenced and post_stop_drained

    def _run(self) -> None:
        torch.cuda.set_device(self._worker.device)
        logger.info(
            "[WeightSyncThread] Thread started on device %s",
            self._worker.device,
        )
        while not self._stop.is_set():
            try:
                _, seq, (cmd_type, command) = self._queue.get(timeout=0.1)
            except queue.Empty:
                self._idle.set()
                continue
            try:
                if cmd_type == "p2r":
                    self._execute_p2r(command)
                elif cmd_type == "r2r":
                    self._execute_r2r(command)
            except Exception:
                self._task_failed = True
                logger.exception(
                    "[WeightSyncThread] Error executing %s command",
                    cmd_type,
                )
            finally:
                self._queue.task_done()
                if self._queue.empty():
                    self._idle.set()

    def _execute_p2r(self, command) -> None:
        """Run the P2R receive on the WST's CUDA stream."""
        t0 = time.monotonic()
        self._worker._execute_p2r_recv(command, self._stream)

        self._last_event = torch.cuda.Event()
        self._last_event.record(self._stream)
        self._worker._buffer_version += 1
        self._executed += 1
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "[WeightSyncThread] P2R done (step=%s, ver=%s, %.1f ms, "
            "qdepth_after=%d, executed=%d)",
            command.weight_step,
            self._worker.current_weight_version,
            elapsed_ms,
            self._queue.qsize(),
            self._executed,
        )

    def _execute_r2r(self, command) -> None:
        """Redis barrier + grouped NCCL broadcast on buffer_model.

        When commands are routed directly from the background command
        thread (bypassing the main-thread handler), the WST is
        responsible for bookkeeping that would normally be done in the
        handler: ``flush_pending_sends``, ``set_weight_synced``.
        """
        worker = self._worker

        # Flush any pending async NCCL sends before reusing the communicator.
        if hasattr(worker, "data_packer") and hasattr(
            worker.data_packer, "flush_pending_sends"
        ):
            worker.data_packer.flush_pending_sends()

        weight_step = command.weight_step
        # Use the controller's authoritative recipient set for this round as the
        # barrier participant count so it stays in lockstep as replicas finish.
        expected_world_size = len(getattr(command, "dst_replica_names", None) or [])
        if expected_world_size > 1 and not r2r_barrier(
            worker, weight_step, expected_world_size=expected_world_size
        ):
            logger.info(
                "[WeightSyncThread] R2R cancelled during teardown (step=%s)",
                weight_step,
            )
            return
        t0 = time.monotonic()
        if expected_world_size <= 1:
            # BuildMesh intentionally creates no NCCL communicator for one
            # replica. P2R already populated its buffer; retain R2R's version
            # and validation bookkeeping without touching a stale communicator.
            transferred_cnt, bytes_broadcast = 0, 0
        else:
            transferred_cnt, bytes_broadcast = do_nccl_broadcast_grouped(
                worker,
                command.src_replica_name,
                self._stream,
            )
        self._last_event = torch.cuda.Event()
        self._last_event.record(self._stream)
        worker._buffer_version += 1
        self._executed += 1

        if weight_step is not None:
            worker.current_weight_version = weight_step

        if weight_step is not None and weight_step >= 0:
            cfg = worker.config
            is_initial = weight_step == 0 and cfg.validation.val_before_train
            is_periodic = weight_step > 0 and weight_step % cfg.validation.freq == 0
            is_final = weight_step == command.total_steps
            should_do_validation = cfg.validation.enable and (
                is_initial or is_periodic or is_final
            )
            if should_do_validation:
                worker.current_step = weight_step
                worker.validation_flag.set()
                worker._pending_validation_step = weight_step

        if command.replica_should_stop():
            worker._pending_shutdown = True

        # Mark weight_synced only after the broadcast has completed,
        # so the main loop does not start serving before weights are
        # actually available in the buffer.
        if not worker.state.weight_synced():
            worker.state.set_weight_synced()
            logger.info(
                "[WeightSyncThread] set_weight_synced after first R2R broadcast "
                "(step=%s)",
                weight_step,
            )

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "[WeightSyncThread] R2R done: %d params, %.1f MB, %.0f ms, step=%s, "
            "ver=%s, qdepth_after=%d, executed=%d",
            transferred_cnt,
            bytes_broadcast / (1024 * 1024),
            elapsed_ms,
            weight_step,
            worker.current_weight_version,
            self._queue.qsize(),
            self._executed,
        )


# ---------------------------------------------------------------------------
# Redis barrier for R2R
# ---------------------------------------------------------------------------


def setup_redis_barrier(worker) -> None:
    """Set up Redis client and barrier prefix for R2R coordination.

    Idempotent.  The WeightSyncThread uses these attributes in
    ``r2r_barrier`` to synchronize all rollout workers before each
    NCCL broadcast.
    """
    if hasattr(worker, "_r2r_redis"):
        return

    worker._r2r_redis = None
    worker._r2r_world_size = len(getattr(worker, "replica_name_to_rank", {}))

    if _redis_lib is not None:
        try:
            redis_host = "localhost"
            redis_port = 6379
            redis_db = 0
            redis_controller = getattr(worker, "redis_controller", None)
            if redis_controller and hasattr(redis_controller, "redis_clients"):
                clients = redis_controller.redis_clients
                if clients:
                    conn_kwargs = clients[0].connection_pool.connection_kwargs
                    redis_host = conn_kwargs.get("host", redis_host)
                    redis_port = conn_kwargs.get("port", redis_port)
                    redis_db = conn_kwargs.get("db", redis_db)
            config = getattr(worker, "config", None)
            if config and hasattr(config, "redis") and config.redis:
                redis_port = int(config.redis)
            r2r_redis = _redis_lib.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True,
            )
            r2r_redis.ping()
            worker._r2r_redis = r2r_redis
        except Exception as exc:
            logger.warning(
                "[WeightSync] Redis unavailable for R2R barrier (%s); "
                "barrier will be skipped.",
                exc,
            )

    exp_name = "default"
    try:
        exp_name = getattr(worker.config.logging, "experiment_name", "default")
    except Exception:
        pass
    job_id = os.environ.get("SLURM_JOB_ID", "test")
    worker._r2r_barrier_prefix = f"cosmos_rl:{exp_name}:{job_id}:r2r"

    logger.info(
        "[WeightSync] Redis barrier setup (redis=%s, world_size=%d, barrier_prefix=%s)",
        worker._r2r_redis is not None,
        worker._r2r_world_size,
        worker._r2r_barrier_prefix,
    )


def r2r_barrier(
    worker, weight_step: int, expected_world_size: Optional[int] = None
) -> bool:
    """Redis-based barrier so all rollout workers start R2R broadcast together.

    Uses an atomic INCR counter per weight step.  The last worker to arrive
    publishes a "go" signal; earlier workers block on pub/sub until they
    receive it (or timeout).  Silently skipped if Redis is unavailable.

    ``expected_world_size`` is the authoritative number of participants for
    *this* broadcast round, derived from the controller's ``dst_replica_names``
    in the R2R command.  Using it (instead of the cached ``_r2r_world_size``,
    which is frozen at setup) keeps the barrier in lockstep with the controller:
    when a replica reaches end-of-data and the controller drops it from the
    broadcast set, the remaining workers expect the shrunken count and the
    barrier completes immediately rather than spinning for the full timeout.
    """
    r2r_redis = getattr(worker, "_r2r_redis", None)
    if expected_world_size is not None and expected_world_size > 0:
        world_size = expected_world_size
    else:
        # Fall back to the live mesh size, then the cached value.  The cached
        # ``_r2r_world_size`` becomes stale once replicas leave the mesh.
        world_size = len(getattr(worker, "replica_name_to_rank", {})) or getattr(
            worker, "_r2r_world_size", 0
        )
    if r2r_redis is None or world_size <= 1:
        return True

    prefix = worker._r2r_barrier_prefix
    barrier_key = f"{prefix}:barrier:{weight_step}"
    go_channel = f"{prefix}:go:{weight_step}"

    try:
        count = r2r_redis.incr(barrier_key)
        r2r_redis.expire(barrier_key, 600)

        if count >= world_size:
            r2r_redis.publish(go_channel, "go")
            logger.info(
                "[R2R Barrier] Last worker arrived (count=%d/%d, step=%d), "
                "published go signal.",
                count,
                world_size,
                weight_step,
            )
            return True

        logger.info(
            "[R2R Barrier] Waiting for other workers (count=%d/%d, step=%d)...",
            count,
            world_size,
            weight_step,
        )
        t0 = time.monotonic()

        pubsub = r2r_redis.pubsub()
        pubsub.subscribe(go_channel)
        try:
            recheck = int(r2r_redis.get(barrier_key) or 0)
            if recheck >= world_size:
                elapsed_ms = (time.monotonic() - t0) * 1000
                logger.info(
                    "[R2R Barrier] Go signal already sent (recheck=%d/%d), "
                    "%.1f ms wait.",
                    recheck,
                    world_size,
                    elapsed_ms,
                )
                return True

            # Allow teardown to interrupt the wait: if the WeightSyncThread is
            # asked to stop, abort the barrier rather than blocking ``wst.stop()``
            # (and in turn ``destroy_distributed()``) for the full timeout.
            wst = getattr(worker, "_weight_sync_thread", None)
            stop_event = getattr(wst, "_stop", None)
            deadline = time.monotonic() + _R2R_BARRIER_TIMEOUT_S
            while time.monotonic() < deadline:
                if stop_event is not None and stop_event.is_set():
                    logger.info(
                        "[R2R Barrier] Stop requested while waiting (step=%d); "
                        "aborting barrier.",
                        weight_step,
                    )
                    return False
                msg = pubsub.get_message(timeout=1.0)
                if msg is not None and msg.get("type") == "message":
                    break
            else:
                logger.warning(
                    "[R2R Barrier] Timed out after %ds waiting for go signal "
                    "(step=%d). Proceeding anyway.",
                    _R2R_BARRIER_TIMEOUT_S,
                    weight_step,
                )
        finally:
            pubsub.unsubscribe(go_channel)
            pubsub.close()

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "[R2R Barrier] All workers ready (step=%d), waited %.1f ms.",
            weight_step,
            elapsed_ms,
        )
        return True
    except Exception as exc:
        logger.warning("[R2R Barrier] Redis error (%s); skipping barrier.", exc)
        return True


# ---------------------------------------------------------------------------
# NCCL broadcast helpers
# ---------------------------------------------------------------------------


def do_nccl_broadcast_grouped(worker, src_replica_name: str, stream) -> tuple:
    """Grouped NCCL broadcast of all model params using group start/end.

    Uses buffer tensors when ``_buffer_state_dict`` exists.
    Returns ``(param_count, bytes_broadcast)``.
    """
    bytes_broadcast = 0
    transferred_cnt = 0
    non_contig: list[tuple[torch.Tensor, torch.Tensor]] = []
    with torch.cuda.stream(stream):
        assert worker.rank_in_rollout_repicas >= 0
        assert len(worker.replica_name_to_rank) > 0
        comm_idx = worker.global_commnicator_idex
        src_rank = worker.replica_name_to_rank[src_replica_name]

        buffer_sd = getattr(worker, "_buffer_state_dict", None)
        if buffer_sd is not None:
            params_iter = buffer_sd.items()
        else:
            model = worker.rollout.get_underlying_model()
            params_iter = model.state_dict().items()

        with torch.inference_mode():
            nccl_group_start(comm_idx)
            for _, param in params_iter:
                if param.is_contiguous():
                    nccl_broadcast(param, src_rank, comm_idx)
                else:
                    recv_tensor = param.contiguous()
                    nccl_broadcast(recv_tensor, src_rank, comm_idx)
                    non_contig.append((param, recv_tensor))
                bytes_broadcast += param.nelement() * param.element_size()
                transferred_cnt += 1
            nccl_group_end(comm_idx)
            for param, recv_tensor in non_contig:
                param.copy_(recv_tensor)
    return transferred_cnt, bytes_broadcast


# ---------------------------------------------------------------------------
# Orchestration: ensure_wst, install_inference_sync
# ---------------------------------------------------------------------------


def ensure_wst(worker) -> WeightSyncThread:
    """Idempotent setup: buffer model, Redis barrier, view-map redirect, WST.

    Safe to call multiple times.  After this returns the WST is running
    and all P2R / R2R commands can be enqueued to it.
    """
    if not hasattr(worker, "_buffer_state_dict"):
        create_buffer_model(worker)
    if hasattr(worker, "weight_inplace_view_map") and not getattr(
        worker, "_view_map_redirected", False
    ):
        redirect_view_map_to_buffer(worker)
        worker._view_map_redirected = True
    setup_redis_barrier(worker)
    if not hasattr(worker, "_weight_sync_thread"):
        worker._weight_sync_thread = WeightSyncThread(worker)
    wst: WeightSyncThread = worker._weight_sync_thread
    if not wst._thread.is_alive():
        wst.start()
    return wst


def install_inference_sync(worker) -> None:
    """Wrap the rollout servicer's policy_fn to sync buffer before each call.

    In "inference" mode, P2R/R2R may still be in-flight on the WST
    when a callback triggers policy inference.  This wrapper ensures
    buffer params are copied to the live model before each forward pass.
    """
    rollout = worker.rollout
    servicer = getattr(rollout, "_servicer", None)
    if servicer is None:
        logger.warning(
            "[WeightSync] Cannot install inference-level sync: "
            "rollout has no _servicer attribute. Falling back to "
            "generation-level sync."
        )
        return

    original_policy_fn = servicer.policy_fn
    _inf_sync_count = [0]

    def _synced_policy_fn(observation):
        _inf_sync_count[0] += 1
        t0 = time.monotonic()
        sync_buffer_to_live(worker)
        sync_ms = (time.monotonic() - t0) * 1000
        if sync_ms > 0.5 or _inf_sync_count[0] <= 3:
            logger.info(
                "[InferenceSync] policy_fn call #%d: sync=%.2fms, "
                "buf_ver=%d, synced_ver=%d",
                _inf_sync_count[0],
                sync_ms,
                getattr(worker, "_buffer_version", -1),
                getattr(worker, "_buffer_synced_version", -1),
            )
        return original_policy_fn(observation)

    servicer.policy_fn = _synced_policy_fn
    logger.info(
        "[WeightSync] Installed inference-level buffer sync on rollout servicer"
    )
