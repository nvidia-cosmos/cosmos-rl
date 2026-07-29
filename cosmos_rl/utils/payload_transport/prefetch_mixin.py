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
from typing import Any, Callable, Dict, List, Optional

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.payload_transport.strategy import PayloadTransportStrategy
from cosmos_rl.utils.trace import get_trace_time


__all__ = ["PrefetchDataPackerMixin"]


class PrefetchDataPackerMixin:
    """Transport-agnostic prefetch + double-buffer + early-ack scheduler.

    See module docstring for the subclass-hook contract.

    Transport behaviour arrives one of two ways:

    * **Composed** -- attach a :class:`PayloadTransportStrategy` via
      :meth:`set_transport_strategy`.  The hooks below then delegate to it,
      which lets the transport be chosen from config at runtime.
    * **Subclassed** -- override the ``_``-prefixed hooks directly, the
      original contract.  Still fully supported.

    They are alternatives, not layers.  An override *replaces* the delegating
    implementation, so a subclass that overrides a hook wins for that hook even
    if a strategy is also attached; mixing the two on the same hook is
    therefore legal but almost never what you want.
    """

    # ------------------------------------------------------------------
    # Scheduling state (owned by the base; subclasses should not touch
    # these directly -- use the public API or override the hooks).
    # ------------------------------------------------------------------
    _transport_strategy: Optional[PayloadTransportStrategy] = None
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

        # A previous shutdown may have left a worker parked inside
        # ``_fetch_batch``.  That thread reads ``self._prefetch_shutdown`` and
        # ``self._prefetch_request_queue`` LIVE, so rebinding them below would
        # hand the stale worker the fresh (cleared) event and the fresh queue:
        # it would never exit, and it would race the new worker for the same
        # requests while ``wait_prefetch`` pops one result per call.  Refuse
        # rather than silently running two workers.
        stale = self._prefetch_thread
        if stale is not None and stale.is_alive():
            raise RuntimeError(
                "[PrefetchDataPackerMixin] cannot start a prefetch worker: the "
                "previous one is still running (it did not observe shutdown "
                "within the join timeout). The transport's shutdown must unblock "
                "in-flight fetches (see the before_join hook) before re-init."
            )
        self._prefetch_thread = None

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

    def shutdown_prefetch(
        self,
        *,
        join_timeout: float = 5.0,
        before_join: Optional[Callable[[], None]] = None,
    ) -> None:
        """Stop the background thread.  Safe to call multiple times.

        Ordering matters.  The worker only observes the shutdown event
        *between* batches, so a thread parked inside :meth:`_fetch_batch` is
        bounded by the transport's own (much larger) first-transfer / prefetch
        timeouts -- not by ``join_timeout``.  Joining first would therefore
        wait out a fetch that only the transport can unwedge.

        ``before_join`` runs after the event is set but BEFORE the join, so the
        transport can force its in-flight I/O to fail fast (NCCL
        ``comm_cache.abort_all``, UCXX client close).  The fetch then raises,
        the worker loop catches it, and the join completes promptly.  This
        mirrors the producer's ``cleanup_nccl``, which aborts comms *before*
        shutting down its sender pool for exactly this reason.

        When the thread does not exit within ``join_timeout`` its handle is
        deliberately RETAINED: it is still running and still reading
        ``self._prefetch_request_queue`` / ``self._prefetch_shutdown``, so
        :meth:`_setup_prefetch` must be able to see it and refuse to start a
        duplicate worker that would race it for the same queue.

        Args:
            join_timeout: Seconds to wait for the worker to exit.  Generous
                once ``before_join`` has unblocked in-flight I/O.
            before_join: Optional transport teardown invoked between setting
                the shutdown event and joining.  Exceptions are logged and
                swallowed -- teardown proceeds regardless.  Defaults to the
                attached strategy's ``before_join``, so a composed transport
                unwedges its own I/O without the caller arranging it; pass an
                explicit callable to override.
        """
        if self._prefetch_shutdown is not None:
            self._prefetch_shutdown.set()

        if before_join is None and self._transport_strategy is not None:
            before_join = self._transport_strategy.before_join

        if before_join is not None:
            try:
                before_join()
            except Exception as exc:  # pragma: no cover - teardown best-effort
                logger.warning(
                    "[PrefetchDataPackerMixin] before_join hook raised %s; "
                    "joining anyway",
                    exc,
                )

        thread = self._prefetch_thread
        if thread is not None:
            thread.join(timeout=join_timeout)
            if thread.is_alive():
                logger.warning(
                    "[PrefetchDataPackerMixin] prefetch thread still running "
                    "after %.1fs; retaining its handle so a re-init cannot "
                    "start a duplicate worker",
                    join_timeout,
                )
            else:
                self._prefetch_thread = None
        self._prefetch_enabled = False

    # ------------------------------------------------------------------
    # Subclass hooks (override in transport-specific mixin)
    # ------------------------------------------------------------------

    def set_transport_strategy(
        self, strategy: Optional[PayloadTransportStrategy]
    ) -> None:
        """Compose in a transport, replacing any previously attached one.

        Call before :meth:`_setup_prefetch`: the strategy decides what the
        worker fetches, and swapping it under a running worker would leave
        already-queued tasks being resolved by the outgoing transport.  Pass
        ``None`` to detach (the hooks fall back to their pass-through defaults).
        """
        self._transport_strategy = strategy

    def _should_intercept(self, rollout_output: Any) -> bool:
        """Return True if ``rollout_output`` is a transport reference.

        Delegates to the attached strategy; without one, never intercepts (the
        mixin becomes a no-op pass-through).
        """
        strategy = self._transport_strategy
        if strategy is not None:
            return strategy.should_intercept(rollout_output)
        return False

    def _cache_key(self, rollout_output: Any) -> str:
        """Stable string key for the resolved payload of ``rollout_output``.

        Delegates to the attached strategy.  Subclasses must override this when
        they implement ``_should_intercept`` themselves.
        """
        strategy = self._transport_strategy
        if strategy is not None:
            return strategy.cache_key(rollout_output)
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
        strategy = self._transport_strategy
        if strategy is not None:
            return strategy.filter_prefetch_tasks(rollouts)
        tasks: List[Any] = []
        for i, rollout in enumerate(rollouts):
            ro = rollout.completion if hasattr(rollout, "completion") else rollout
            if self._should_intercept(ro):
                tasks.append((i, ro))
        return tasks

    def _fetch_batch(self, tasks: List[Any]) -> Dict[str, Any]:
        """Fetch a batch of references; return ``{cache_key: payload}``.

        Runs on the background prefetch thread.  Delegates to the attached
        strategy; subclasses without one must implement it.
        """
        strategy = self._transport_strategy
        if strategy is not None:
            return strategy.fetch_batch(tasks)
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
        strategy = self._transport_strategy
        if strategy is not None:
            return strategy.sync_fetch(rollout_output)
        return None

    def _on_prefetch_complete(
        self,
        batch_id: int,
        n_results: int,
        fetch_ms: float,
    ) -> None:
        """Hook called after each ``wait_prefetch`` populates the cache.

        Default: no-op.  Subclasses can use this to emit periodic INFO
        summaries, increment cumulative counters, etc.  The strategy form also
        receives the iteration counter, which it would otherwise have to read
        off the packer.
        """
        strategy = self._transport_strategy
        if strategy is not None:
            return strategy.on_prefetch_complete(
                batch_id, n_results, fetch_ms, self._prefetch_step_count
            )
        return None

    def _on_resolve_failed(self, rollout_output: Any, cache_key: str) -> None:
        """Hook called when both cache lookup and ``_sync_fetch`` returned
        ``None`` for an intercepted reference.

        Default: no-op (the base ``get_policy_input`` already logs a
        warning).  Subclasses use this to bump fallback counters or
        emit transport-specific telemetry without having to override
        the entire dispatch.
        """
        strategy = self._transport_strategy
        if strategy is not None:
            return strategy.on_resolve_failed(rollout_output, cache_key)
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
