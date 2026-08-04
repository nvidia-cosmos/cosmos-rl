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
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.trajectory import EPISODE_LENGTH, VARLEN_FIELDS
from cosmos_rl.utils.payload_transport.strategy import PayloadTransportStrategy
from cosmos_rl.utils.trace import get_trace_time
from cosmos_rl.utils.payload_transport.ucxx.ucxx_buffer import (
    UCXX_AVAILABLE,
    UCXXClient,
)


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
        "TimeoutError",
    }
)

_MAX_FETCH_ROUNDS = 3


_LOG_INTERVAL = 50


# Numpy → torch dtype map for the bulk pinned-buffer copy in
# ``_to_gpu``.  Defined at module scope so it is built once at import
# time rather than on every fetch.
_NP_TO_TORCH = {
    np.dtype("float32"): torch.float32,
    np.dtype("float64"): torch.float64,
    np.dtype("float16"): torch.float16,
    np.dtype("int64"): torch.int64,
    np.dtype("int32"): torch.int32,
    np.dtype("int16"): torch.int16,
    np.dtype("int8"): torch.int8,
    np.dtype("uint8"): torch.uint8,
    np.dtype("bool"): torch.bool,
}


class UCXXTransportStrategy(PayloadTransportStrategy):
    """Resolve UCXX payload references for a single consumer."""

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
            read_timeout: Per-await timeout (seconds) inside one
                ``UCXXClient.read`` call -- bounds a single ``send`` /
                ``recv`` operation.
        """
        if not UCXX_AVAILABLE:
            raise RuntimeError(
                "UCXX is required for UCXXTransportStrategy. "
                "Install with: pip install ucxx-cu12"
            )

        self._device = device
        self._max_attempts = max(1, max_attempts)
        self._read_timeout = read_timeout
        self._client = UCXXClient()

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
        if self._client is not None:
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self._client.close())
                loop.close()
            except Exception as e:
                logger.warning("[UCXXTransportStrategy] Failed to close client: %s", e)
            self._client = None

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

        UCXX-tagged completions are dicts produced by
        :class:`UCXXRolloutMixin` carrying ``{_ucxx: True,
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
        return self._ref_cache_key(rollout_output)

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
        loop = asyncio.new_event_loop()
        try:
            raw_results, transfer_ms, copy_ms = loop.run_until_complete(
                self._fetch_all(tasks)
            )
        finally:
            loop.close()

        cache_results: Dict[str, Any] = {}
        total_bytes = 0
        ucxx_count = 0
        for idx, gpu_data in raw_results.items():
            key = self._ref_cache_key_from_task(tasks, idx)
            cache_results[key] = gpu_data
            ucxx_count += 1
            for val in gpu_data.values():
                if isinstance(val, torch.Tensor):
                    total_bytes += val.nelement() * val.element_size()

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
        loop = asyncio.new_event_loop()
        try:
            results, _, _ = loop.run_until_complete(
                self._fetch_all([(0, rollout_output)])
            )
            return results.get(0)
        except Exception as e:
            logger.warning("[UCXXTransportStrategy] Sync fallback failed: %s", e)
            return None
        finally:
            loop.close()

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

    async def _fetch_all(self, ucxx_tasks: list) -> tuple:
        """Fetch all episodes concurrently with multi-round retry.

        Returns ``(results_dict, transfer_ms, copy_ms)`` where
        ``results_dict`` maps task index -> GPU tensor dict.
        """
        client = self._client
        device = self._device

        async def _read_one(idx: int, metadata: dict):
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
                from cosmos_rl.utils.trajectory import (
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
                        schema,
                        ports=ports,
                        timeout=read_timeout,
                    )
                    break
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

        def _to_gpu(result: dict) -> dict:
            pinned_buf = result.pop("_pinned_buf", None)
            if pinned_buf is not None:
                try:
                    raw_gpu = pinned_buf.to(device, non_blocking=True)
                    torch.cuda.current_stream().synchronize()

                    gpu_data: Dict[str, Any] = {}
                    offset = 0
                    for key, value in result.items():
                        if not hasattr(value, "shape"):
                            gpu_data[key] = value
                            continue
                        nbytes = value.nbytes
                        td = _NP_TO_TORCH.get(value.dtype)
                        if td is None:
                            raise ValueError(
                                f"Unsupported dtype {value.dtype} for key '{key}'"
                            )
                        gpu_data[key] = (
                            raw_gpu[offset : offset + nbytes]
                            .clone()
                            .view(td)
                            .reshape(value.shape)
                        )
                        offset += nbytes
                except Exception as e:
                    logger.error(
                        "[UCXXTransportStrategy] Bulk GPU copy failed (%s), "
                        "falling back to per-tensor copy",
                        e,
                    )
                    gpu_data = {}
                    for key, value in result.items():
                        if hasattr(value, "shape"):
                            gpu_data[key] = torch.from_numpy(value.copy()).to(
                                device, non_blocking=True
                            )
                        else:
                            gpu_data[key] = value
                finally:
                    client.return_pinned(pinned_buf)
            else:
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
                for key in VARLEN_FIELDS:
                    if key in gpu_data and gpu_data[key].shape[0] > ep_len:
                        gpu_data[key] = gpu_data[key][:ep_len]
            return gpu_data

        meta_by_idx: dict = {}
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
                idx, result, retryable = await coro
                t1 = get_trace_time()
                total_transfer_ms += t1 - t0

                if result is None:
                    if retryable:
                        failed.append(idx)
                    # Non-retryable failures (e.g. stale slot): drop
                    # immediately so the round-level retry doesn't
                    # waste another ~RTT per round on a slot that
                    # cannot be resurrected.
                    continue

                gpu_data = _to_gpu(result)
                batch_results[idx] = gpu_data
                t2 = get_trace_time()
                total_copy_ms += t2 - t1

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

        return batch_results, total_transfer_ms, total_copy_ms

    # ------------------------------------------------------------------
    # Cache key helpers (kept as static methods for tests + the
    # cross-task lookup helper used inside _fetch_batch).
    # ------------------------------------------------------------------

    @staticmethod
    def _ref_cache_key(metadata: dict) -> str:
        return (
            f"{metadata.get('_worker_ip')}:"
            f"{metadata.get('_ucxx_port')}:"
            f"{metadata.get('_slot')}"
        )

    @staticmethod
    def _ref_cache_key_from_task(
        ucxx_tasks: list,
        idx: int,
    ) -> str:
        for task_idx, metadata in ucxx_tasks:
            if task_idx == idx:
                return (
                    f"{metadata.get('_worker_ip')}:"
                    f"{metadata.get('_ucxx_port')}:"
                    f"{metadata.get('_slot')}"
                )
        return str(idx)


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
    strategy.setup(device=device, max_attempts=max_attempts, read_timeout=read_timeout)
    packer.set_transport_strategy(strategy)
    packer._setup_prefetch(
        prefetch_timeout=prefetch_timeout,
        thread_name="UCXXDataPackerPrefetch",
    )
