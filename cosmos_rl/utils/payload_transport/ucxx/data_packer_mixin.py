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

"""UCXXDataPackerMixin -- UCXX-specific subclass of PrefetchDataPackerMixin.

This file is intentionally **thin**: the prefetch + double-buffer +
early-train-ack scheduling is owned by
:class:`cosmos_rl.utils.payload_transport.prefetch_mixin.PrefetchDataPackerMixin`,
and this subclass plugs in the UCXX-specific bits:

* the wire-format predicate (``_should_intercept``: dict with
  ``_ucxx: True``)
* the cache key (``_cache_key``: ``"ip:port:slot"``)
* the actual zero-copy fetch (``_fetch_batch`` -> async UCXXClient.read
  with single-chunk-per-slot transfers, port rotation on transient
  transport errors, and multi-round retry)
* a sync-fallback fetch for cache misses
* a periodic INFO summary of UCXX bytes / latency

A future ``NCCLDataPackerMixin`` will subclass the same base and plug in
its own predicates / cache keys / recv path -- they will share *all* of
the scheduling code.

Usage::

    class UCXXMyDataPacker(UCXXDataPackerMixin, MyDataPacker):
        pass

MRO ensures :meth:`PrefetchDataPackerMixin.get_policy_input` (inherited)
intercepts before delegating to ``MyDataPacker``.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.payload_transport.prefetch_mixin import (
    PrefetchDataPackerMixin,
    get_trace_time,
)
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

OBSERVATIONS = "observations"
ACTIONS = "actions"
REWARDS = "rewards"
TERMINATED = "terminated"
TRUNCATED = "truncated"
EPISODE_LENGTH = "episode_length"

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


class UCXXDataPackerMixin(PrefetchDataPackerMixin):
    """UCXX subclass of :class:`PrefetchDataPackerMixin`.

    See :mod:`cosmos_rl.utils.payload_transport.prefetch_mixin` for the
    transport-agnostic scheduling layer.  Place **before** the concrete
    DataPacker in the MRO::

        class UCXXSimpleRLDataPacker(UCXXDataPackerMixin, SimpleRLDataPacker):
            pass
    """

    # UCXX-specific state.  Scheduling state (queues, thread, cache,
    # double-buffer, step counter) is owned by the base mixin and must
    # not be duplicated here.
    _ucxx_dp_client: Optional[UCXXClient] = None
    _ucxx_dp_device: Optional[torch.device] = None
    _ucxx_dp_max_attempts: int = 2
    _ucxx_dp_read_timeout: float = 5.0

    # Cumulative stats for periodic INFO summaries (UCXX-specific so the
    # base mixin's _on_prefetch_complete hook is the right place).
    _ucxx_dp_total_ucxx: int = 0
    _ucxx_dp_total_fallback: int = 0
    _ucxx_dp_total_bytes: int = 0
    _ucxx_dp_total_latency_ms: float = 0.0

    # Captured by _fetch_batch each round, consumed in _on_prefetch_complete.
    _ucxx_dp_last_bytes: int = 0

    # ------------------------------------------------------------------
    # Backward-compat aliases for downstream / test code that referenced
    # the old UCXX-prefixed internals.  Keep at least until the
    # _setup_ucxx_data_packer -> _setup_prefetch migration is fully
    # rolled out (>= two minor releases after MR5 lands).
    # ------------------------------------------------------------------

    @property
    def _ucxx_dp_enabled(self) -> bool:
        return self._prefetch_enabled

    @_ucxx_dp_enabled.setter
    def _ucxx_dp_enabled(self, value: bool) -> None:
        self._prefetch_enabled = bool(value)

    @property
    def _ucxx_dp_prefetch_cache(self) -> Dict[str, Any]:
        return self._prefetch_cache

    @_ucxx_dp_prefetch_cache.setter
    def _ucxx_dp_prefetch_cache(self, value: Dict[str, Any]) -> None:
        self._prefetch_cache = value

    @property
    def _ucxx_dp_step_count(self) -> int:
        return self._prefetch_step_count

    @property
    def _ucxx_dp_prefetch_timeout(self) -> float:
        return self._prefetch_timeout_s

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def _setup_ucxx_data_packer(
        self,
        *,
        device: torch.device,
        prefetch_timeout: float = 300.0,
        max_attempts: int = 2,
        read_timeout: float = 5.0,
    ) -> None:
        """Initialise UCXX client + start the (inherited) prefetch thread.

        NOTE: Normally invoked **automatically** by
        :meth:`UCXXPayloadTransport.attach_data_packer` during
        ``CommMixin.init_data_packer``.  Direct calls are only needed
        in tests or unusual lifecycle setups.

        The leading underscore signals "framework-internal" -- the
        attach hook is the public surface.  A backward-compatible alias
        ``setup_ucxx_data_packer`` is preserved below for in-flight
        downstream code that calls the original public name.

        Args:
            device: Target GPU device for fetched tensors.
            prefetch_timeout: Per-batch wait ceiling (seconds) for the
                prefetch worker thread's result queue.
            max_attempts: Total attempts per remote slot read (initial +
                retries on transient UCX errors).  Defaults to 2.
            read_timeout: Per-await timeout (seconds) inside one
                ``UCXXClient.read`` call -- bounds a single ``send`` /
                ``recv`` operation, distinct from ``prefetch_timeout``.
        """
        if not UCXX_AVAILABLE:
            raise RuntimeError(
                "UCXX is required for UCXXDataPackerMixin. "
                "Install with: pip install ucxx-cu12"
            )

        self._ucxx_dp_device = device
        self._ucxx_dp_max_attempts = max(1, max_attempts)
        self._ucxx_dp_read_timeout = read_timeout
        self._ucxx_dp_client = UCXXClient()

        self._setup_prefetch(
            prefetch_timeout=prefetch_timeout,
            thread_name="UCXXDataPackerPrefetch",
        )

        logger.info(
            "[UCXXDataPackerMixin] Initialised: device=%s, timeout=%ss, "
            "max_attempts=%d, read_timeout=%ss",
            device,
            prefetch_timeout,
            self._ucxx_dp_max_attempts,
            read_timeout,
        )

    def setup_ucxx_data_packer(
        self,
        device: torch.device,
        prefetch_timeout: float = 300.0,
        max_attempts: int = 2,
        read_timeout: float = 5.0,
    ) -> None:
        """DEPRECATED: use :meth:`_setup_ucxx_data_packer` (kwargs-only).

        Kept as a thin shim because some downstream forks call the
        original public name positionally.  Forwards to the new entry
        point with keyword arguments.  Remove no earlier than two minor
        releases after this PR lands.
        """
        self._setup_ucxx_data_packer(
            device=device,
            prefetch_timeout=prefetch_timeout,
            max_attempts=max_attempts,
            read_timeout=read_timeout,
        )

    def shutdown_ucxx_data_packer(self) -> None:
        """Stop background thread and release UCXX resources."""
        self.shutdown_prefetch()

        if self._ucxx_dp_client is not None:
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self._ucxx_dp_client.close())
                loop.close()
            except Exception as e:
                logger.warning("[UCXXDataPackerMixin] Failed to close client: %s", e)
            self._ucxx_dp_client = None

        if self._prefetch_step_count > 0:
            avg_ms = self._ucxx_dp_total_latency_ms / self._prefetch_step_count
            logger.info(
                "[UCXXDataPackerMixin] Final: %d iters, %d UCXX / %d fallback, "
                "%.1f MB, avg %.0f ms/iter",
                self._prefetch_step_count,
                self._ucxx_dp_total_ucxx,
                self._ucxx_dp_total_fallback,
                self._ucxx_dp_total_bytes / 1e6,
                avg_ms,
            )
        logger.info("[UCXXDataPackerMixin] Shut down")

    # ------------------------------------------------------------------
    # PrefetchDataPackerMixin hook implementations
    # ------------------------------------------------------------------

    def _should_intercept(self, rollout_output: Any) -> bool:
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

    def _cache_key(self, rollout_output: Any) -> str:
        return self._ucxx_dp_cache_key(rollout_output)

    def _filter_prefetch_tasks(self, rollouts: List[Any]) -> List[Any]:
        """UCXX requires both ``_ucxx`` AND ``_ucxx_enabled`` true.

        Slightly stricter than the base default (which would defer to
        :meth:`_should_intercept` -- equivalent here, but kept explicit
        because UCXX shipped with the dual-flag check before the base
        mixin existed and downstream observability hooks may rely on
        seeing both flags evaluated).
        """
        tasks: List[Any] = []
        for i, rollout in enumerate(rollouts):
            ro = rollout.completion if hasattr(rollout, "completion") else rollout
            if isinstance(ro, dict) and ro.get("_ucxx") and ro.get("_ucxx_enabled"):
                tasks.append((i, ro))
        return tasks

    def _fetch_batch(self, tasks: List[Any]) -> Dict[str, Any]:
        """Run the async UCXX fetch on this thread's event loop.

        Called by the base mixin's worker loop.  Returns
        ``{cache_key: gpu_data}``.
        """
        loop = asyncio.new_event_loop()
        try:
            raw_results, transfer_ms, copy_ms = loop.run_until_complete(
                self._ucxx_dp_fetch_all(tasks)
            )
        finally:
            loop.close()

        cache_results: Dict[str, Any] = {}
        total_bytes = 0
        ucxx_count = 0
        for idx, gpu_data in raw_results.items():
            key = self._ucxx_dp_cache_key_from_task(tasks, idx)
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
        self._ucxx_dp_last_bytes = total_bytes
        self._ucxx_dp_last_transfer_ms = transfer_ms
        self._ucxx_dp_last_copy_ms = copy_ms
        self._ucxx_dp_last_count = ucxx_count

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

    def _sync_fetch(self, rollout_output: Any) -> Optional[Dict[str, torch.Tensor]]:
        """Blocking single-episode UCXX fetch (cache-miss fallback)."""
        if self._ucxx_dp_client is None:
            return None
        loop = asyncio.new_event_loop()
        try:
            results, _, _ = loop.run_until_complete(
                self._ucxx_dp_fetch_all([(0, rollout_output)])
            )
            return results.get(0)
        except Exception as e:
            logger.warning("[UCXXDataPackerMixin] Sync fallback failed: %s", e)
            return None
        finally:
            loop.close()

    def _on_prefetch_complete(
        self,
        batch_id: int,
        n_results: int,
        fetch_ms: float,
    ) -> None:
        """Accumulate UCXX-specific stats; emit periodic INFO summaries."""
        self._ucxx_dp_total_ucxx += getattr(self, "_ucxx_dp_last_count", n_results)
        self._ucxx_dp_total_bytes += getattr(self, "_ucxx_dp_last_bytes", 0)
        self._ucxx_dp_total_latency_ms += fetch_ms
        step = self._prefetch_step_count
        if step == 1 or step % _LOG_INTERVAL == 0:
            avg_ms = self._ucxx_dp_total_latency_ms / step
            logger.info(
                "[UCXXDataPackerMixin] Iteration %d: %d UCXX, "
                "%.1f MB total, avg %.0f ms/iter",
                step,
                self._ucxx_dp_total_ucxx,
                self._ucxx_dp_total_bytes / 1e6,
                avg_ms,
            )

    def _on_resolve_failed(self, rollout_output: Any, cache_key: str) -> None:
        """Bump UCXX-specific fallback counter when an episode is skipped.

        The base mixin already logs the warning; this hook just records
        the event for the periodic INFO summary.
        """
        self._ucxx_dp_total_fallback += 1

    # get_policy_input is inherited unchanged from PrefetchDataPackerMixin.

    # ------------------------------------------------------------------
    # Async fetch (UCXX-specific; unchanged from pre-refactor)
    # ------------------------------------------------------------------

    async def _ucxx_dp_fetch_all(self, ucxx_tasks: list) -> tuple:
        """Fetch all episodes concurrently with multi-round retry.

        Returns ``(results_dict, transfer_ms, copy_ms)`` where
        ``results_dict`` maps task index -> GPU tensor dict.
        """
        client = self._ucxx_dp_client
        device = self._ucxx_dp_device

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

            max_attempts = max(1, self._ucxx_dp_max_attempts)
            read_timeout = self._ucxx_dp_read_timeout
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
                            "[UCXXDataPackerMixin] Non-transient error reading "
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
                            "[UCXXDataPackerMixin] All %d attempts failed for "
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
                        "[UCXXDataPackerMixin] Transient error reading "
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
                        "[UCXXDataPackerMixin] Bulk GPU copy failed (%s), "
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
                for key in (OBSERVATIONS, ACTIONS, REWARDS, TERMINATED, TRUNCATED):
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
                    "[UCXXDataPackerMixin] Fetch round %d/%d: "
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
                "[UCXXDataPackerMixin] %d episodes failed after %d rounds: indices=%s",
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
    def _ucxx_dp_cache_key(metadata: dict) -> str:
        return (
            f"{metadata.get('_worker_ip')}:"
            f"{metadata.get('_ucxx_port')}:"
            f"{metadata.get('_slot')}"
        )

    @staticmethod
    def _ucxx_dp_cache_key_from_task(
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
