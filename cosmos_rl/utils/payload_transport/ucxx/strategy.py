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
"""UCXX payload transport, expressed as a composable strategy.

Owns the UCXX client and the async read path; knows nothing about *when* a
fetch happens, which is the packer's job.

Moved off ``UCXXDataPackerMixin`` unchanged apart from dropping the
``_ucxx_dp_`` prefix and taking the iteration counter as an argument.  Note it
deliberately does NOT implement :meth:`before_join`: unlike ``ncclCommAbort``,
closing a UCXX client from another thread while the worker is inside its own
event loop has no such guarantee, so teardown widens the join budget and closes
afterwards instead (see :meth:`shutdown`).
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, List, Optional

import numpy as np
from cosmos_rl.utils.transport_failure import TransportUnusableError
import torch

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.payload_transport.layout import (
    LayoutCache,
    TensorLayout,
    WIRE_VERSION,
    validate_layout_id,
)
from cosmos_rl.utils.payload_transport.pack import NP_TO_TORCH
from cosmos_rl.utils.trajectory import (
    EPISODE_LENGTH,
    VARLEN_FIELDS,
    truncate_trajectory,
    validate_episode_length,
)
from cosmos_rl.utils.payload_transport.strategy import PayloadTransportStrategy
from cosmos_rl.utils.payload_transport.receive_memory import (
    ReceiveBufferPool,
    ReceivedBatch,
    ReusableViewMap,
    storage_bytes,
)
from cosmos_rl.utils.trace import get_trace_time
from cosmos_rl.utils.payload_transport.ucxx.ucxx_buffer import (
    UCXX_AVAILABLE,
    UCXXClient,
    PayloadIdentityError,
)
from cosmos_rl.utils.payload_transport.ucxx.operation import UCXXOperation


# Errors that are worth retrying at the data-packer layer.  Mirrors
# the rotation set in :data:`ucxx_buffer._PORT_ROTATABLE_ERRORS` --
# transport-class failures on a specific server thread can succeed on
# the next attempt because :class:`UCXXClient` will rotate to a
# different (worker_ip, port) endpoint internally.
#
# Deliberately excluded:
#   * ``StaleSlotError`` -- the slot has been overwritten on the
#     producer side; no amount of retrying brings it back.  The
#     :class:`PrefetchDataPackerMixin` upper layer drops the episode
#     via ``_on_resolve_failed`` on the very first attempt.

_TRANSIENT_UCXX_ERRORS = frozenset(
    {
        "UCXXCanceledError",
        "UCXXConnectionResetError",
        "UCXXCloseError",
    }
)

_MAX_FETCH_ROUNDS = 3
_MAX_PORTS_PER_REFERENCE = 64
_MAX_UINT64 = (1 << 64) - 1


_LOG_INTERVAL = 50


class _PlacedPayload(dict):
    """Tensor views plus the backing-storage retirement operation."""

    def __init__(self, values, *, owner, release):
        super().__init__(values)
        self.owner = owner
        self.release_storage = release


class UCXXTransportStrategy(PayloadTransportStrategy):
    """Resolve UCXX payload references for a single consumer."""

    def __init__(self):
        self._io_lock = threading.RLock()
        self._event_loop = None
        self._failure = None
        self._layout_cache = LayoutCache()
        self._receive_pool = ReceiveBufferPool()

    def _run_async(self, coroutine):
        # UCXX ties progress work to the event loop used at initialization.
        # Keep that loop alive across reads and endpoint teardown, and serialize
        # prefetch/synchronous fallback so it is never run on two threads.
        with self._io_lock:
            if self._failure is not None:
                coroutine.close()
                raise TransportUnusableError(self._failure)
            if self._event_loop is None:
                self._event_loop = asyncio.new_event_loop()
            return self._event_loop.run_until_complete(coroutine)

    _client: Optional[UCXXClient] = None
    _device: Optional[torch.device] = None
    _max_attempts: int = 2
    _read_timeout: float = 30.0
    _total_ucxx: int = 0
    _total_fallback: int = 0
    _total_bytes: int = 0
    _total_latency_ms: float = 0.0
    _last_bytes: int = 0
    _last_count: int = 0
    _last_transfer_ms: float = 0.0
    _last_copy_ms: float = 0.0
    _steps: int = 0

    @property
    def read_timeout(self) -> float:
        """Per-read budget; also sizes the packer's join wait during teardown."""
        return self._read_timeout

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(
        self,
        *,
        device: torch.device,
        max_attempts: int = 2,
        read_timeout: float = 30.0,
    ) -> None:
        """Create the UCXX client.

        Normally invoked by the packer composing this strategy, which is itself
        driven by :meth:`UCXXPayloadTransport.attach_data_packer` during
        ``CommMixin.init_data_packer``.  Scheduling knobs such as
        ``prefetch_timeout`` stop at the packer and never reach here.

        Args:
            device: Target GPU device for fetched tensors.
            max_attempts: Total attempts per remote slot read (initial +
                retries on transient UCX errors).  Defaults to 2.
            read_timeout: Whole-attempt timeout (seconds), including native
                connection/read completion. Device-copy completion has the same
                budget. Uncertain completion is terminal, not a retry.
        """
        if not UCXX_AVAILABLE:
            raise RuntimeError(
                "UCXX is required for UCXXTransportStrategy. "
                "Install with: pip install ucxx-cu12"
            )

        self._device = device
        self._max_attempts = max(1, max_attempts)
        self._read_timeout = read_timeout

        async def initialize_client():
            self._client = UCXXClient()

        self._run_async(initialize_client())

        logger.info(
            "[UCXXTransportStrategy] Initialised: device=%s, "
            "max_attempts=%d, read_timeout=%ss",
            device,
            self._max_attempts,
            read_timeout,
        )

    def shutdown(self) -> None:
        """Close the client and log the run summary.

        Called AFTER the packer has joined its worker -- see the module
        docstring for why UCXX closes late rather than aborting early.
        """
        with self._io_lock:
            if self._client is not None:
                self._run_async(self._client.close())
                self._client = None
            if self._event_loop is not None:
                self._event_loop.close()
                self._event_loop = None
            self._receive_pool.close()

        if self._steps > 0:
            avg_ms = self._total_latency_ms / self._steps
            logger.info(
                "[UCXXTransportStrategy] Final: %d iters, %d UCXX / %d fallback, "
                "%.1f MB, avg %.0f ms/iter",
                self._steps,
                self._total_ucxx,
                self._total_fallback,
                self._total_bytes / 1e6,
                avg_ms,
            )
        logger.info("[UCXXTransportStrategy] Shut down")

    def should_intercept(self, rollout_output: Any) -> bool:
        """UCXX wire-format predicate.

        UCXX-tagged completions are dicts produced by ``PayloadContext`` carrying
        ``{_ucxx: True,
        _ucxx_enabled: True, _worker_ip, _ucxx_port, _slot, ...}``.
        Plain trajectories and ``"nccl:<id>"`` strings fall through
        untouched (NCCL has its own intercept predicate in a sibling
        mixin).
        """
        if not isinstance(rollout_output, dict):
            return False
        if not rollout_output.get("_ucxx"):
            return False
        # ``_ucxx_enabled`` is the runtime kill-switch on the rollout
        # side; honor it on the trainer side too so a worker that
        # disabled UCXX mid-flight (e.g. fallback to Redis) is handled
        # correctly.  Treat absence as "enabled" for backward compat.
        return rollout_output.get("_ucxx_enabled", True)

    def cache_key(self, rollout_output: Any) -> str:
        return self._ref_cache_key(_validate_ucxx_reference(rollout_output))

    # NOTE: deliberately NO _filter_prefetch_tasks override.  The base default
    # already delegates to :meth:`_should_intercept`, which is the only way to
    # guarantee that what the trainer intercepts is exactly what gets
    # prefetched.  The override that used to live here re-implemented the
    # predicate as ``ro.get("_ucxx") and ro.get("_ucxx_enabled")`` -- reading a
    # MISSING ``_ucxx_enabled`` as disabled, where _should_intercept reads it as
    # enabled.  Such a ref was intercepted but never prefetched, so every one of
    # its episodes took the blocking _sync_fetch path forever: a silent,
    # permanent slow path rather than a failure.

    def fetch_batch(self, tasks: List[Any]) -> Dict[str, Any]:
        """Run the async UCXX fetch on this thread's event loop.

        Called by the base mixin's worker loop.  Returns
        ``{cache_key: gpu_data}``.
        """
        validated_tasks = [
            (idx, _validate_ucxx_reference(metadata)) for idx, metadata in tasks
        ]
        raw_results, transfer_ms, copy_ms = self._run_async(
            self._fetch_all(validated_tasks)
        )

        cache_results: Dict[str, Any] = {}
        keys_by_idx = {
            idx: self._ref_cache_key(metadata) for idx, metadata in validated_tasks
        }
        total_bytes = 0
        ucxx_count = 0
        for idx, gpu_data in raw_results.items():
            key = keys_by_idx.get(idx, str(idx))
            cache_results[key] = gpu_data
            ucxx_count += 1
            for val in gpu_data.values():
                if isinstance(val, torch.Tensor):
                    total_bytes += val.nelement() * val.element_size()

        if isinstance(raw_results, ReceivedBatch):
            raw_results.clear()
            raw_results.update(cache_results)
            cache_results = raw_results

        # Stash for _on_prefetch_complete (called from the trainer
        # thread once wait_prefetch drains the result queue).  We
        # intentionally don't accumulate counters here -- the base
        # mixin guarantees _on_prefetch_complete sees exactly the
        # results from this batch.
        self._last_bytes = total_bytes
        self._last_transfer_ms = transfer_ms
        self._last_copy_ms = copy_ms
        self._last_count = ucxx_count

        # Decoupled trace event (actual I/O timestamps from the bg thread,
        # not the trainer's wait_prefetch).
        if ucxx_count > 0:
            logger.debug(
                "[Trace] thread=ucxx_prefetch op=ucxx_fetch "
                "transfer_ms=%.1f copy_ms=%.1f count=%d bytes=%d",
                transfer_ms,
                copy_ms,
                ucxx_count,
                total_bytes,
            )

        return cache_results

    def sync_fetch(self, rollout_output: Any) -> Optional[Dict[str, torch.Tensor]]:
        """Blocking single-episode UCXX fetch (cache-miss fallback)."""
        if self._client is None:
            return None
        try:
            results, _, _ = self._run_async(self._fetch_all([(0, rollout_output)]))
            payload = results.get(0)
            if payload is None or not isinstance(results, ReceivedBatch):
                return payload
            try:
                return {name: tensor.clone() for name, tensor in payload.items()}
            finally:
                results.release()
        except TransportUnusableError:
            raise
        except Exception as e:
            logger.warning("[UCXXTransportStrategy] Sync fallback failed: %s", e)
            return None

    def on_prefetch_complete(
        self,
        batch_id: int,
        n_results: int,
        fetch_ms: float,
        step: int,
    ) -> None:
        """Accumulate UCXX-specific stats; emit periodic INFO summaries."""
        # Batch's own figures, set by fetch_batch -- not n_results, which the
        # base passes as len(self._prefetch_cache) and would over-count.
        self._total_ucxx += self._last_count
        self._total_bytes += self._last_bytes
        self._total_latency_ms += fetch_ms
        self._steps = step
        if step == 1 or step % _LOG_INTERVAL == 0:
            avg_ms = self._total_latency_ms / step
            logger.info(
                "[UCXXTransportStrategy] Iteration %d: %d UCXX, "
                "%.1f MB total, avg %.0f ms/iter",
                step,
                self._total_ucxx,
                self._total_bytes / 1e6,
                avg_ms,
            )

    def on_resolve_failed(self, rollout_output: Any, cache_key: str) -> None:
        """Bump UCXX-specific fallback counter when an episode is skipped.

        The base mixin already logs the warning; this hook just records
        the event for the periodic INFO summary.
        """
        self._total_fallback += 1

    # get_policy_input is inherited unchanged from PrefetchDataPackerMixin.

    # ------------------------------------------------------------------
    # Async fetch (UCXX-specific; unchanged from pre-refactor)
    # ------------------------------------------------------------------

    def _mark_failed(self, reason):
        self._failure = reason

    def _copy_to_device(self, result):
        client, device = self._client, torch.device(self._device)
        pinned_buf = result.get("_pinned_buf")
        try:
            layout = result.get("_layout")
            if not isinstance(layout, TensorLayout):
                raise ValueError("UCXX result carries no validated tensor layout")
            if (
                not isinstance(pinned_buf, torch.Tensor)
                or pinned_buf.dtype != torch.uint8
                or pinned_buf.ndim != 1
                or not pinned_buf.is_contiguous()
            ):
                raise ValueError("UCXX result carries no valid backing buffer")
            arrays = {
                key: value
                for key, value in result.items()
                if key not in {"_pinned_buf", "_layout"}
            }
            # Validate and read scalar metadata on the host BEFORE any
            # asynchronous copy. A malformed schema must not send us into a
            # GPU-error fallback.
            for key, value in arrays.items():
                if isinstance(value, np.ndarray):
                    if value.dtype not in NP_TO_TORCH or not value.flags.c_contiguous:
                        raise ValueError(f"Unsupported UCXX array for key {key!r}")
            ep_len_array = arrays.get(EPISODE_LENGTH)
            if ep_len_array is not None:
                fields = [arrays[key] for key in VARLEN_FIELDS if key in arrays]
                validate_episode_length(
                    ep_len_array, [value.shape[0] for value in fields]
                )
            if pinned_buf.numel() != layout.payload_nbytes:
                raise ValueError("UCXX pinned buffer and decoded schema size disagree")
        except BaseException:
            if isinstance(pinned_buf, torch.Tensor):
                self._return_pinned_if_open(pinned_buf)
            raise

        if device.type != "cuda":
            bind_views = getattr(client, "bind_pinned_views", None)
            gpu_data = (
                bind_views(pinned_buf, layout, reset_fields=VARLEN_FIELDS)
                if callable(bind_views)
                else layout.views(pinned_buf)
            )
            owner = pinned_buf

            def release_storage():
                return self._return_pinned_if_open(pinned_buf)
        else:
            try:
                storage_lease = self._receive_pool.acquire(
                    layout.layout_id, layout.payload_nbytes, device
                )
            except torch.OutOfMemoryError as error:
                # No device operation has been issued, so the host receive
                # buffer can be returned immediately without a CUDA fence.
                self._return_pinned_if_open(pinned_buf)
                raise ValueError(
                    "UCXX device-copy allocation failed before native issue"
                ) from error
            operation = UCXXOperation(
                self._read_timeout,
                "UCXX device copy",
                owners=(self, client, result, pinned_buf, storage_lease),
                on_failure=self._mark_failed,
            )
            gpu_data = {}
            try:
                with torch.cuda.device(device):
                    stream = torch.cuda.current_stream(device)
                    operation.owners.append(stream)
                    if pinned_buf is not None:
                        raw_gpu = storage_lease.tensor
                        raw_gpu.copy_(pinned_buf, non_blocking=True)
                        operation.owners.append(raw_gpu)
                        gpu_data = storage_lease.bind_views(
                            layout, reset_fields=VARLEN_FIELDS
                        )
                        operation.owners.append(gpu_data)
                    else:
                        raise ValueError("UCXX device result carries no backing buffer")
                    event = torch.cuda.Event()
                    operation.owners.append(event)
                    event.record(stream)
                    operation.deadline.wait_event(event)
                    operation.complete()
            except torch.OutOfMemoryError as error:
                # Allocation failure need not kill training if a bounded device
                # fence proves that earlier copies no longer read this storage.
                try:
                    with torch.cuda.device(device):
                        event = torch.cuda.Event()
                        operation.owners.append(event)
                        event.record(torch.cuda.current_stream(device))
                        operation.deadline.wait_event(event)
                        operation.complete()
                except BaseException as completion_error:
                    operation.fail(
                        f"OOM cleanup completion uncertain: {completion_error}"
                    )
                if pinned_buf is not None:
                    client.return_pinned(pinned_buf)
                storage_lease.release()
                raise ValueError(
                    "UCXX device-copy allocation failed after safe drain"
                ) from error
            except BaseException as error:
                # Even .to()/record() can issue native work and then raise. No
                # second copy, buffer recycle, or stream cleanup without proof.
                operation.fail(f"device-copy completion uncertain: {error}")
            if pinned_buf is not None:
                client.return_pinned(pinned_buf)
            owner = storage_lease.tensor
            release_storage = storage_lease.release

        try:
            truncate_trajectory(gpu_data)
        except BaseException:
            release_storage()
            raise
        if isinstance(gpu_data, ReusableViewMap):
            # Preserve the cached mapping itself. Copying it into a plain dict
            # would hide consumer key mutations from reset(), allowing one
            # batch's replaced metadata view to leak into the next reuse.
            return gpu_data.retain(owner=owner, release=release_storage)
        return _PlacedPayload(gpu_data, owner=owner, release=release_storage)

    def _return_pinned_if_open(self, pinned_buf):
        client = self._client
        if client is not None:
            return_pinned = getattr(client, "return_pinned_if_open", None)
            if callable(return_pinned):
                return_pinned(pinned_buf)
            elif (
                not getattr(client, "_closing", False)
                and getattr(client, "_failure", None) is None
            ):
                client.return_pinned(pinned_buf)

    async def _fetch_all(self, ucxx_tasks: list) -> tuple:
        """Fetch all episodes concurrently with multi-round retry.

        Returns ``(results_dict, transfer_ms, copy_ms)`` where
        ``results_dict`` maps task index -> GPU tensor dict.
        """
        client = self._client

        async def _read_one(idx: int, metadata: dict):
            worker_ip = metadata.get("_worker_ip")
            ucxx_port = metadata.get("_ucxx_port")
            slot = metadata.get("_slot")
            generation = metadata.get("_slot_generation")
            ports = metadata.get("_ports")

            layout = metadata["_resolved_layout"]
            sender_incarnation = metadata["_sender_incarnation"]

            max_attempts = max(1, self._max_attempts)
            read_timeout = self._read_timeout
            data = None
            retryable = True
            for attempt in range(1, max_attempts + 1):
                try:
                    data = await client.read(
                        worker_ip,
                        ucxx_port,
                        slot,
                        layout,
                        ports=ports,
                        timeout=read_timeout,
                        generation=generation,
                        sender_incarnation=sender_incarnation,
                    )
                    break
                except TransportUnusableError:
                    raise
                except PayloadIdentityError as error:
                    raise TransportUnusableError(
                        f"UCXX payload reference is not trustworthy: {error}"
                    ) from error
                except Exception as e:
                    if type(e).__name__ not in _TRANSIENT_UCXX_ERRORS:
                        # Non-transient (e.g. ``StaleSlotError``,
                        # protocol error from the server): retrying is
                        # pointless.  Mark non-retryable so Layer C
                        # skips the slot in subsequent rounds.
                        logger.error(
                            "[UCXXTransportStrategy] Non-transient error reading "
                            "%s:%s slot=%s: %s: %s",
                            worker_ip,
                            ucxx_port,
                            slot,
                            type(e).__name__,
                            e,
                        )
                        retryable = False
                        return idx, None, retryable
                    if attempt == max_attempts:
                        logger.warning(
                            "[UCXXTransportStrategy] All %d attempts failed for "
                            "%s:%s slot=%s: %s: %s",
                            max_attempts,
                            worker_ip,
                            ucxx_port,
                            slot,
                            type(e).__name__,
                            e,
                        )
                        return idx, None, retryable
                    logger.warning(
                        "[UCXXTransportStrategy] Transient error reading "
                        "%s:%s slot=%s (attempt %d/%d): %s, retrying",
                        worker_ip,
                        ucxx_port,
                        slot,
                        attempt,
                        max_attempts,
                        type(e).__name__,
                    )
            return idx, data, retryable

        meta_by_idx: dict = {}
        for idx, metadata in ucxx_tasks:
            meta_by_idx[idx] = _validate_ucxx_reference(metadata)

        for metadata in meta_by_idx.values():
            layout_id = metadata["_layout_id"]
            try:
                layout = self._layout_cache.resolve(layout_id, None)
            except KeyError:
                ports = metadata.get("_ports") or []
                port = ports[0] if ports else metadata["_ucxx_port"]
                try:
                    descriptor = await client.read_layout(
                        metadata["_worker_ip"],
                        port,
                        sender_incarnation=metadata["_sender_incarnation"],
                        timeout=self._read_timeout,
                        ports=ports,
                    )
                except PayloadIdentityError as error:
                    raise TransportUnusableError(
                        f"UCXX payload reference is not trustworthy: {error}"
                    ) from error
                try:
                    layout = self._layout_cache.resolve(layout_id, descriptor)
                except (KeyError, TypeError, ValueError) as error:
                    raise TransportUnusableError(
                        f"UCXX payload layout is not trustworthy: {error}"
                    ) from error
            metadata["_resolved_layout"] = layout

        pending = list(meta_by_idx.keys())
        batch_results: dict = {}
        placements = []
        total_transfer_ms = 0.0
        total_copy_ms = 0.0

        for round_num in range(_MAX_FETCH_ROUNDS):
            if not pending:
                break

            tasks = [
                asyncio.create_task(_read_one(idx, meta_by_idx[idx])) for idx in pending
            ]
            failed = []

            try:
                for coro in asyncio.as_completed(tasks):
                    t0 = get_trace_time()
                    idx, result, retryable = await coro
                    t1 = get_trace_time()
                    total_transfer_ms += t1 - t0

                    if result is None:
                        if retryable:
                            failed.append(idx)
                        continue

                    try:
                        gpu_data = self._copy_to_device(result)
                    except ValueError as error:
                        # Rejection before device work is recoverable. Any error
                        # after native issue is already TransportUnusableError.
                        logger.error(
                            "[UCXXTransportStrategy] Invalid payload: %s", error
                        )
                        continue
                    batch_results[idx] = gpu_data
                    placements.append(gpu_data)
                    t2 = get_trace_time()
                    total_copy_ms += t2 - t1
            except BaseException as error:
                for payload in placements:
                    try:
                        payload.release_storage()
                    except Exception:
                        logger.exception(
                            "[UCXXTransportStrategy] Failed to retire an already "
                            "completed placement while unwinding the batch"
                        )
                placements.clear()
                batch_results.clear()
                if any(not task.done() for task in tasks):
                    operation = UCXXOperation(
                        self._read_timeout,
                        "UCXX concurrent batch",
                        owners=(self, tasks, batch_results),
                        on_failure=self._mark_failed,
                    )
                    operation.fail(f"cannot abandon outstanding native reads: {error}")
                if isinstance(error, TransportUnusableError):
                    self._mark_failed(str(error))
                raise
            finally:
                # Observe completed failures without cancelling pending native
                # waiters. Terminal operations own the latter until process exit.
                for task in tasks:
                    if task.done() and not task.cancelled():
                        task.exception()

            if failed:
                logger.warning(
                    "[UCXXTransportStrategy] Fetch round %d/%d: "
                    "%d/%d episodes failed, %s",
                    round_num + 1,
                    _MAX_FETCH_ROUNDS,
                    len(failed),
                    len(pending),
                    "retrying" if round_num + 1 < _MAX_FETCH_ROUNDS else "giving up",
                )
            pending = failed

        if pending:
            logger.error(
                "[UCXXTransportStrategy] %d episodes failed after %d rounds: indices=%s",
                len(pending),
                _MAX_FETCH_ROUNDS,
                pending,
            )

        owners = tuple(payload.owner for payload in placements)

        def release_storage():
            for payload in placements:
                payload.release_storage()

        values = dict(batch_results)
        batch = ReceivedBatch(
            values,
            nbytes=storage_bytes(values.values()),
            owners=owners,
            on_release=release_storage,
        )
        return batch, total_transfer_ms, total_copy_ms

    # ------------------------------------------------------------------
    # Cache key helpers (kept as static methods for tests + the
    # cross-task lookup helper used inside _fetch_batch).
    # ------------------------------------------------------------------

    @staticmethod
    def _ref_cache_key(metadata: dict) -> str:
        return (
            f"{metadata.get('_worker_ip')}:"
            f"{metadata.get('_ucxx_port')}:"
            f"{metadata.get('_sender_incarnation')}:"
            f"{metadata.get('_slot')}:{metadata.get('_slot_generation')}:"
            f"{metadata.get('_layout_id')}"
        )


def _validate_ucxx_reference(metadata: Any) -> dict:
    """Copy and validate every identity field used by UCXX and its cache."""
    if not isinstance(metadata, dict):
        raise TransportUnusableError(
            "UCXX payload reference metadata must be a mapping"
        )
    validated = dict(metadata)
    worker_ip = validated.get("_worker_ip")
    ucxx_port = validated.get("_ucxx_port")
    slot = validated.get("_slot")
    generation = validated.get("_slot_generation")
    if not isinstance(worker_ip, str) or not worker_ip or len(worker_ip) > 255:
        raise TransportUnusableError(
            "UCXX payload reference has an invalid worker address"
        )
    if (
        not isinstance(ucxx_port, int)
        or isinstance(ucxx_port, bool)
        or not 1 <= ucxx_port <= 65535
    ):
        raise TransportUnusableError(
            "UCXX payload reference has an invalid server port"
        )
    if (
        not isinstance(slot, int)
        or isinstance(slot, bool)
        or not 0 <= slot <= _MAX_UINT64
    ):
        raise TransportUnusableError(
            "UCXX payload reference has an invalid slot identity"
        )
    if (
        not isinstance(generation, int)
        or isinstance(generation, bool)
        or not 1 <= generation <= _MAX_UINT64
    ):
        raise TransportUnusableError(
            "UCXX payload reference has an invalid slot generation"
        )
    if validated.get("_wire_version") != WIRE_VERSION:
        raise TransportUnusableError(
            "UCXX payload reference has an unsupported wire version"
        )
    sender_incarnation = validated.get("_sender_incarnation")
    if (
        not isinstance(sender_incarnation, str)
        or not sender_incarnation
        or len(sender_incarnation) > 256
    ):
        raise TransportUnusableError(
            "UCXX payload reference has an invalid sender incarnation"
        )
    try:
        validate_layout_id(validated.get("_layout_id"))
    except ValueError as error:
        raise TransportUnusableError(
            f"UCXX payload reference has an invalid tensor layout identity: {error}"
        ) from error
    ports = validated.get("_ports")
    if ports is not None and (
        not isinstance(ports, (list, tuple))
        or len(ports) > _MAX_PORTS_PER_REFERENCE
        or any(
            not isinstance(port, int)
            or isinstance(port, bool)
            or not 1 <= port <= 65535
            for port in ports
        )
    ):
        raise TransportUnusableError(
            "UCXX payload reference has an invalid server port list"
        )
    return validated


def compose_ucxx_transport(
    packer: Any,
    *,
    device: Any,
    prefetch_timeout: float = 300.0,
    max_attempts: int = 2,
    read_timeout: float = 30.0,
) -> None:
    """Attach a fresh UCXX strategy to ``packer`` and start its prefetch worker.

    The one place that wiring lives, so the mixin and the composed attach path
    cannot drift.  ``packer`` need only be a ``PrefetchDataPackerMixin``; no
    UCXX ancestry is required.
    """
    strategy = UCXXTransportStrategy()
    packer.set_transport_strategy(strategy)
    try:
        strategy.setup(
            device=device, max_attempts=max_attempts, read_timeout=read_timeout
        )
        packer._setup_prefetch(
            prefetch_timeout=prefetch_timeout,
            thread_name="UCXXDataPackerPrefetch",
        )
    except BaseException:
        try:
            packer.close_transport()
        except Exception as cleanup_error:
            logger.error("UCXX attachment rollback failed: %s", cleanup_error)
        raise
