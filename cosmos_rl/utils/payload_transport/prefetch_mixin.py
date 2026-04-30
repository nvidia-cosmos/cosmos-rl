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

"""Transport-agnostic prefetch / double-buffer / early-train-ack mixin.

Background
----------
Heavy-payload transports (UCXX RDMA, NCCL point-to-point, …) all share
the same scheduling shape on the trainer side:

1. The rollout completion is a *reference* (a dict tag, an ``nccl:<id>``
   string, …) that must be resolved to actual tensors by an extra fetch.
2. Resolving N references in a batch is the slow step on each iteration.
3. The trainer can hide that latency by:
   * **prefetching** the next iteration's batch in the background
     while the current iteration's compute runs (pipeline overlap), and
   * **deferring** the wait until the *following* iteration so that
     ``step_training`` returns early and the rollout worker's train-ack
     fires sooner (early-ack, double-buffer).

That scheduling state machine has nothing to do with which transport is
moving the bytes -- only the actual fetch does.  This mixin owns the
scheduling and exposes a small set of subclass hooks for the transport
to plug into.

Subclass contract
-----------------
Concrete transport packers (``UCXXDataPackerMixin``,
``NCCLDataPackerMixin`` (future), …) inherit from this mixin and
override:

* :meth:`_should_intercept(rollout_output)` -- returns ``True`` if the
  rollout completion is a transport reference this mixin should resolve
  before delegating to the underlying packer.  Default: ``False``
  (everything passes straight through).
* :meth:`_cache_key(rollout_output)` -- returns a stable string key for
  the resolved payload.
* :meth:`_filter_prefetch_tasks(rollouts)` -- returns the subset of a
  rollout batch that should be prefetched as ``[(idx, ref), ...]``.
  Default: every rollout whose completion satisfies
  ``_should_intercept``.
* :meth:`_fetch_batch(tasks)` -- runs synchronously on the background
  thread; returns ``{cache_key: payload}``.  Must be implemented.
* :meth:`_sync_fetch(rollout_output)` -- blocking single-ref fallback
  used when ``get_policy_input`` hits a cache miss (e.g. when prefetch
  hasn't happened yet).  Default: ``None`` (skip episode).
* :meth:`_on_prefetch_complete(batch_id, n_results, fetch_ms)` -- hook
  for periodic stats logging.  Default: no-op.

The base mixin owns the queues / thread / state machine; subclasses own
the wire-format and the actual byte-moving.

Composition example
-------------------
::

    class UCXXMyDataPacker(UCXXDataPackerMixin, MyDataPacker):
        pass

    class NCCLMyDataPacker(NCCLDataPackerMixin, MyDataPacker):
        pass

The MRO ensures the mixin's ``get_policy_input`` runs first, intercepts
references, and only then delegates to ``MyDataPacker`` via ``super()``.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Dict, List, Optional

from cosmos_rl.utils.logging import logger

try:
    from cosmos_rl.utils.trace import get_trace_time  # type: ignore
except ImportError:  # pragma: no cover - fallback path

    def get_trace_time() -> float:  # type: ignore[no-redef]
        return time.perf_counter() * 1000.0


__all__ = ["PrefetchDataPackerMixin"]


class PrefetchDataPackerMixin:
    """Transport-agnostic prefetch + double-buffer + early-ack scheduler.

    See module docstring for the subclass-hook contract.
    """

    # ------------------------------------------------------------------
    # Scheduling state (owned by the base; subclasses should not touch
    # these directly -- use the public API or override the hooks).
    # ------------------------------------------------------------------
    _prefetch_enabled: bool = False
    _prefetch_request_queue: Optional[queue.Queue] = None
    _prefetch_result_queue: Optional[queue.Queue] = None
    _prefetch_shutdown: Optional[threading.Event] = None
    _prefetch_thread: Optional[threading.Thread] = None
    _prefetch_batch_id: int = 0
    _prefetch_cache: Dict[str, Any] = {}
    _prefetch_timeout_s: float = 300.0
    _prefetch_step_count: int = 0

    # Double-buffer state for early-ack.  Owned here so any concrete
    # subclass gets it for free.
    _prefetch_buffer: Optional[list] = None
    _prefetch_pending: bool = False
    _prefetch_rollouts: Optional[list] = None

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def _setup_prefetch(
        self,
        *,
        prefetch_timeout: float = 300.0,
        thread_name: str = "PrefetchDataPacker",
    ) -> None:
        """Start the background fetch thread and arm the scheduling state.

        Idempotent: calling twice without a :meth:`shutdown_prefetch` in
        between leaves the existing worker running and just refreshes
        the timeout.  This makes test-driven re-init paths painless.
        """
        self._prefetch_timeout_s = prefetch_timeout
        if self._prefetch_enabled:
            return

        self._prefetch_cache = {}
        self._prefetch_request_queue = queue.Queue()
        self._prefetch_result_queue = queue.Queue()
        self._prefetch_shutdown = threading.Event()
        self._prefetch_shutdown.clear()

        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker_loop,
            name=thread_name,
            daemon=True,
        )
        self._prefetch_thread.start()
        self._prefetch_enabled = True

    def shutdown_prefetch(self) -> None:
        """Stop the background thread.  Safe to call multiple times."""
        if self._prefetch_shutdown is not None:
            self._prefetch_shutdown.set()
        if self._prefetch_thread is not None:
            self._prefetch_thread.join(timeout=5.0)
            if self._prefetch_thread.is_alive():
                logger.warning(
                    "[PrefetchDataPackerMixin] Prefetch thread did not stop cleanly"
                )
            self._prefetch_thread = None
        self._prefetch_enabled = False

    # ------------------------------------------------------------------
    # Subclass hooks (override in transport-specific mixin)
    # ------------------------------------------------------------------

    def _should_intercept(self, rollout_output: Any) -> bool:
        """Return True if ``rollout_output`` is a transport reference.

        Default: never intercept (the mixin becomes a no-op pass-through).
        """
        return False

    def _cache_key(self, rollout_output: Any) -> str:
        """Stable string key for the resolved payload of ``rollout_output``.

        Subclasses must override when ``_should_intercept`` can return True.
        """
        raise NotImplementedError(
            "Subclass must override _cache_key when _should_intercept may return True"
        )

    def _filter_prefetch_tasks(self, rollouts: List[Any]) -> List[Any]:
        """Pick the subset of a rollout batch eligible for prefetch.

        Default: every rollout whose completion satisfies
        :meth:`_should_intercept`.  Returned tuples are
        ``(idx, completion_ref)`` -- ``idx`` is opaque to the base
        layer and just propagates back to ``_fetch_batch`` so subclass
        implementations can correlate batch indices with sources.
        """
        tasks: List[Any] = []
        for i, rollout in enumerate(rollouts):
            ro = rollout.completion if hasattr(rollout, "completion") else rollout
            if self._should_intercept(ro):
                tasks.append((i, ro))
        return tasks

    def _fetch_batch(self, tasks: List[Any]) -> Dict[str, Any]:
        """Fetch a batch of references; return ``{cache_key: payload}``.

        Runs on the background prefetch thread.  Subclasses must implement.
        """
        raise NotImplementedError(
            "Subclass must implement _fetch_batch to provide the actual "
            "transport-specific fetch logic"
        )

    def _sync_fetch(self, rollout_output: Any) -> Optional[Any]:
        """Blocking single-ref fallback used on a cache miss.

        Default: return ``None`` (which causes ``get_policy_input`` to
        skip the episode).  Subclasses may override to provide a
        synchronous transport fetch for the not-yet-prefetched case.
        """
        return None

    def _on_prefetch_complete(
        self,
        batch_id: int,
        n_results: int,
        fetch_ms: float,
    ) -> None:
        """Hook called after each ``wait_prefetch`` populates the cache.

        Default: no-op.  Subclasses can use this to emit periodic INFO
        summaries, increment cumulative counters, etc.
        """
        return None

    def _on_resolve_failed(self, rollout_output: Any, cache_key: str) -> None:
        """Hook called when both cache lookup and ``_sync_fetch`` returned
        ``None`` for an intercepted reference.

        Default: no-op (the base ``get_policy_input`` already logs a
        warning).  Subclasses use this to bump fallback counters or
        emit transport-specific telemetry without having to override
        the entire dispatch.
        """
        return None

    # ------------------------------------------------------------------
    # Trainer-facing scheduling API
    # ------------------------------------------------------------------

    def start_prefetch(self, rollouts: List[Any]) -> None:
        """Submit ``rollouts`` for background fetch.  Non-blocking.

        Pair with :meth:`wait_prefetch` (or with the deferred-wait API
        below) before iterating ``get_policy_input`` over the batch.
        No-op when the prefetch thread isn't running yet.
        """
        if not self._prefetch_enabled or self._prefetch_request_queue is None:
            return
        tasks = self._filter_prefetch_tasks(rollouts)
        if not tasks:
            return
        batch_id = self._prefetch_batch_id
        self._prefetch_batch_id += 1
        self._prefetch_request_queue.put((batch_id, tasks))

    def wait_prefetch(self) -> None:
        """Block until the in-flight prefetch completes; populate cache.

        After this returns, ``get_policy_input`` resolves references
        from ``_prefetch_cache`` (O(1) dict lookup).
        """
        if not self._prefetch_enabled or self._prefetch_result_queue is None:
            return
        try:
            batch_id, results, fetch_ms = self._prefetch_result_queue.get(
                timeout=self._prefetch_timeout_s
            )
        except queue.Empty:
            logger.error(
                "[PrefetchDataPackerMixin] prefetch timeout after %ss",
                self._prefetch_timeout_s,
            )
            self._prefetch_cache = {}
            return

        if isinstance(results, dict) and "_error" in results:
            logger.warning(
                "[PrefetchDataPackerMixin] batch %d prefetch error: %s",
                batch_id,
                results["_error"],
            )
            self._prefetch_cache = {}
        else:
            self._prefetch_cache = results

        self._prefetch_step_count += 1
        try:
            self._on_prefetch_complete(batch_id, len(self._prefetch_cache), fetch_ms)
        except Exception as exc:  # pragma: no cover - hook bug shouldn't crash trainer
            logger.warning(
                "[PrefetchDataPackerMixin] _on_prefetch_complete raised %s; continuing",
                exc,
            )

    # --- Deferred-wait / early-ack -------------------------------------

    @property
    def is_cold_start(self) -> bool:
        """True when no prefetched data is buffered yet (first iteration)."""
        return self._prefetch_buffer is None and not self._prefetch_pending

    @property
    def prefetch_buffer(self) -> Optional[list]:
        """Rollouts whose payloads are already resolved in the cache."""
        return self._prefetch_buffer

    def collect_prefetch(self) -> Optional[list]:
        """Resolve any deferred prefetch from the previous iteration.

        Call at the **top** of each training iteration.  If a defer is
        pending, this blocks until the background fetch completes, then
        rotates the double-buffer.  Returns the current buffer
        (``None`` on cold start).
        """
        if self._prefetch_pending:
            collect_start = get_trace_time()
            self.wait_prefetch()
            collect_end = get_trace_time()
            logger.debug(
                "[Trace] thread=trainer op=deferred_prefetch_collect "
                "start=%.1f end=%.1f waited_ms=%.1f",
                collect_start,
                collect_end,
                collect_end - collect_start,
            )
            self._prefetch_buffer = self._prefetch_rollouts
            self._prefetch_pending = False
            self._prefetch_rollouts = None
        return self._prefetch_buffer

    def defer_prefetch(self, rollouts: list) -> None:
        """Buffer ``rollouts`` for the next iteration.

        On **cold start** the fetch was already drained via
        ``wait_prefetch`` so this just seeds the buffer.  On **steady
        state** the wait is deferred until the next ``collect_prefetch``
        so ``step_training`` can return immediately and the rollout
        worker's train-ack fires sooner.
        """
        if self._prefetch_buffer is None:
            self._prefetch_buffer = rollouts
        else:
            self._prefetch_pending = True
            self._prefetch_rollouts = rollouts

    # ------------------------------------------------------------------
    # Background prefetch thread
    # ------------------------------------------------------------------

    def _prefetch_worker_loop(self) -> None:
        """Pull tasks from the request queue, dispatch ``_fetch_batch``."""
        try:
            while not self._prefetch_shutdown.is_set():
                try:
                    batch_id, tasks = self._prefetch_request_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                fetch_start = get_trace_time()
                try:
                    results = self._fetch_batch(tasks)
                except Exception as e:
                    err = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
                    logger.error(
                        "[PrefetchDataPackerMixin] batch %d failed: %s",
                        batch_id,
                        err,
                    )
                    results = {"_error": err}
                fetch_end = get_trace_time()

                self._prefetch_result_queue.put(
                    (batch_id, results, fetch_end - fetch_start)
                )
        except Exception as e:  # pragma: no cover - worker-thread crash
            logger.error("[PrefetchDataPackerMixin] worker loop error: %s", e)
        finally:
            logger.debug("[PrefetchDataPackerMixin] worker loop stopped")

    # ------------------------------------------------------------------
    # get_policy_input dispatch
    # ------------------------------------------------------------------

    def get_policy_input(
        self,
        sample: Any = None,
        rollout_output: Any = None,
        n_ignore_prefix_tokens: int = 0,
        **kwargs,
    ) -> Any:
        """Resolve transport references, then delegate to the concrete packer.

        For inputs the subclass declines to intercept (the common case
        for plain trajectories), this is a transparent pass-through to
        ``super().get_policy_input``.
        """
        if rollout_output is not None and self._should_intercept(rollout_output):
            cache_key = self._cache_key(rollout_output)
            resolved = self._prefetch_cache.get(cache_key)
            if resolved is None:
                resolved = self._sync_fetch(rollout_output)
            if resolved is not None:
                return super().get_policy_input(
                    sample, resolved, n_ignore_prefix_tokens, **kwargs
                )
            logger.warning(
                "[PrefetchDataPackerMixin] resolve failed for %s, skipping episode",
                cache_key,
            )
            self._on_resolve_failed(rollout_output, cache_key)
            return None
        return super().get_policy_input(
            sample, rollout_output, n_ignore_prefix_tokens, **kwargs
        )
