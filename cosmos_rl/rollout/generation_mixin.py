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

"""Structured ``rollout_generation`` template + optional per-prompt prefetch.

Why
---
Every concrete :class:`~cosmos_rl.rollout.rollout_base.RolloutBase`
subclass re-implements the same shape inside ``rollout_generation``:

1. assert engine initialized,
2. for each payload run packer-side preprocessing,
3. optionally collate the prepared samples into a batch,
4. call the backend's engine,
5. convert the engine output into ``List[RolloutResult]``,
6. wrap the whole thing in a ``try``/``except`` that returns ``[]`` on
   failure so a single bad batch doesn't kill the rollout worker.

Only step (4) is genuinely backend-specific.  Steps (2) and (5) are
trivially backend-specific (one or two lines each); the rest is
duplication.  This mixin pulls the common shape into a template method
and exposes four override hooks for the per-backend bits.

The same seam doubles as a place to overlap step (2)'s per-prompt
work with step (4)'s in-flight engine call on the previous batch.
When ``config.rollout.prefetch_rollout`` is ``True``, the rollout
worker's ``_prefetch_loop`` calls :meth:`submit_setup` as soon as it
hands a new prompt batch to the prompt queue; the mixin runs
:meth:`_prepare_sample` for each payload on a background thread keyed
by :meth:`_payload_key` so by the time the consumer hits
:meth:`rollout_generation`, the prepared samples are already done (or
in flight, in which case the template method simply waits per-future
just before :meth:`_collate_batch`).  When the flag is ``False``,
:meth:`submit_setup` is a no-op and :meth:`_prepare_sample` runs
inline; the four-hook code path is otherwise identical.

Why one mixin, not two
----------------------
"Structured hooks" and "per-prompt prefetch overlap" share the same
seam: prefetch's only job is to run :meth:`_prepare_sample` on a bg
thread.  Splitting them would force every consumer to compose two
mixins to get the natural behavior, and would expose
future-map vocabulary to backend authors who shouldn't have to think
about it.  One mixin, four hooks, prefetch is a runtime config flag.

Why opt-in (not folded into ``RolloutBase``)
--------------------------------------------
Existing backends (vLLM, TRT-LLM, VLA, WFM, NFT, ExampleHF) have
nuanced ``rollout_generation`` implementations that don't trivially
fit four hooks (vLLM has ``n_repeats``, distillation top-k, vlm vs
not; TRT-LLM has a different signature; VLA has its own simulation
loop).  Forcing all of them into the template at once is high blast
radius for low immediate benefit.  Opt-in lets the gym example adopt
now; the heavier backends migrate one PR at a time as their owners
want the hooks.

Composition
-----------
::

    @RolloutRegistry.register(rollout_type="my_backend")
    class MyBackend(RolloutGenerationMixin, RolloutBase):
        def post_init_hook(self, **kwargs):
            ...
            self.setup_generation(thread_name="MyBackendPrefetch")

        def _prepare_sample(self, payload, *, data_packer, **_):
            return data_packer.get_rollout_input(payload.prompt)

        def _generate(self, batch, *, stream, is_validation):
            return self._engine.run(batch)

        def _postprocess(self, raw, payloads, *, is_validation):
            return [RolloutResult(prompt=p.prompt, completions=[r])
                    for p, r in zip(payloads, raw)]

        def shutdown(self):
            self.shutdown_generation()
            super().shutdown()

The mixin's template ``rollout_generation`` satisfies the abstract
method declared on :class:`RolloutBase` because the mixin precedes
``RolloutBase`` in the MRO.

Threading model
---------------
The bg setup worker is a single daemon thread.  ``_prepare_sample``
must therefore be re-entrant w.r.t. the main thread that drives
``rollout_generation``.  The default implementation
(``data_packer.get_rollout_input``) is pure data shaping and is
trivially safe; backends that do stateful preparation (e.g. building
a sim env per prompt) own the synchronization.  The mixin itself
holds no shared mutable state beyond the future map and uses an
atomic dict swap so concurrent ``submit_setup`` and
:meth:`_gather_prepared_samples` are safe.

Edge cases
----------
* Prefetch disabled: ``submit_setup`` is a no-op,
  ``_gather_prepared_samples`` runs ``_prepare_sample`` inline.
  Behaviour identical to a hand-written ``rollout_generation``.
* ``_prepare_sample`` raises: the future captures the exception,
  ``_gather_prepared_samples`` re-raises in the consumer thread, the
  template's ``except`` clause hands the error to
  :meth:`_on_generation_error` (default: log + return ``[]``).
* Same payload key submitted twice: last writer wins; the previous
  future is dropped.  The controller doesn't repeat indices in normal
  flow, but cancellation paths can.
* Cold-start (a payload arrives at ``rollout_generation`` without a
  matching ``submit_setup`` call): falls back to inline preparation
  for that payload only.  This handles the very first batch of a run
  (where the prefetch loop hasn't run yet) and any controller path
  that bypasses prefetch.
* Lifecycle / programmer errors (engine not initialised, policy not
  attached, ...): assert in :meth:`_preflight_check`, not in the
  per-batch hooks.  Preflight runs outside the ``try``/``except`` so
  the exception propagates past :meth:`_on_generation_error` and
  surfaces loudly -- swallowing a "policy not set" would just produce
  silent empty-batch returns indefinitely.

Out of scope (v1)
-----------------
* Async-engine variant for ``vLLMRolloutAsync``: the mixin is sync.
  A future ``AsyncRolloutGenerationMixin`` is a sibling, not a
  subclass.
* Worker pool for concurrent ``_prepare_sample``: a single bg thread
  is enough for the colocated single-process case.  A pool would
  require an explicit knob; deferred until a backend wants it.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import Future
from typing import Any, Dict, Hashable, List, Optional, Sequence, final

from cosmos_rl.utils.logging import logger as _default_logger


# Sentinel passed to the bg worker to ask it to exit cleanly.
_SHUTDOWN = object()


# Stable contract for the per-batch trace line emitted at DEBUG level
# from ``rollout_generation``.  Downstream analyzers parse these lines,
# so the field names and order are part of the public surface.
#
# Field semantics:
#   batch_size     -- len(payloads) for this generation call
#   prefetch       -- whether the mixin's bg setup worker is enabled
#   gather_ms      -- total wall time spent in _gather_prepared_samples
#                     (= sum of wait_ms + inline_ms + bookkeeping)
#   collate_ms     -- _collate_batch wall time
#   generate_ms    -- _generate wall time (the backend engine call)
#   postprocess_ms -- _postprocess wall time
#   inline_count   -- payloads whose _prepare_sample ran on the main
#                     thread (cold-start, prefetch disabled, or future
#                     evicted/cancelled before we got to it)
#   inline_ms      -- cumulative time spent on the main thread doing
#                     inline _prepare_sample calls
#   wait_count     -- payloads whose future was in flight when we
#                     needed it (prefetch lost the race -- if this is
#                     consistently >0 the bg thread is too slow or
#                     there are too few payloads in flight)
#   wait_ms        -- cumulative time the main thread blocked on
#                     future.result().  When prefetch is fully
#                     overlapping this should trend to zero.
_TRACE_LINE_FORMAT = (
    "[mixin-trace] batch_size=%d prefetch=%s "
    "gather_ms=%.2f collate_ms=%.2f generate_ms=%.2f postprocess_ms=%.2f "
    "inline_count=%d inline_ms=%.2f wait_count=%d wait_ms=%.2f"
)


class RolloutGenerationMixin:
    """Template-method ``rollout_generation`` + opt-in per-prompt prefetch.

    See module docstring for the full design.  Subclasses override the
    four hooks (:meth:`_prepare_sample`, :meth:`_collate_batch`,
    :meth:`_generate`, :meth:`_postprocess`); ``rollout_generation``
    itself is final and dispatches them in order.

    Lifecycle: call :meth:`setup_generation` from
    :meth:`~cosmos_rl.rollout.rollout_base.RolloutBase.post_init_hook`
    (idempotent; no-op when ``config.rollout.prefetch_rollout`` is
    ``False``).  Call :meth:`shutdown_generation` from
    :meth:`~cosmos_rl.rollout.rollout_base.RolloutBase.shutdown`.
    """

    # ------------------------------------------------------------------
    # Override hooks
    # ------------------------------------------------------------------

    def _prepare_sample(
        self,
        payload: Any,
        *,
        data_packer: Any,
        data_fetcher: Any,
        is_validation: bool,
    ) -> Any:
        """Per-payload preprocessing.

        Called on the background setup worker when prefetch is on,
        inline on the calling thread when off.  Must be re-entrant
        w.r.t. anything :meth:`_generate` touches on ``self``.

        Default: ``data_packer.get_rollout_input(payload.prompt)``.
        Backends that don't have a packer (or want to handle their own
        preprocessing) override.
        """
        if data_packer is None:
            prompt = getattr(payload, "prompt", payload)
            return prompt
        prompt = getattr(payload, "prompt", payload)
        return data_packer.get_rollout_input({"prompt": prompt})

    def _collate_batch(
        self,
        samples: Sequence[Any],
        *,
        data_packer: Any,
        is_validation: bool,
    ) -> Any:
        """Combine prepared samples into the engine's batch input.

        Default: identity (returns ``samples`` unchanged).  Backends
        whose engine takes a single batch object override (e.g. an LLM
        backend that calls ``data_packer.rollout_collate_fn``).
        """
        return samples

    def _generate(
        self,
        batch: Any,
        *,
        stream: Any,
        is_validation: bool,
    ) -> Any:
        """Backend-specific engine call.  Required override.

        ``batch`` is whatever :meth:`_collate_batch` returned.  The
        return value is handed to :meth:`_postprocess` along with the
        original ``payloads`` list so backends can pair engine output
        back to prompts.
        """
        raise NotImplementedError("[RolloutGenerationMixin] _generate is required.")

    def _postprocess(
        self,
        raw: Any,
        payloads: Sequence[Any],
        *,
        is_validation: bool,
    ) -> Any:
        """Convert engine output into the final per-payload result list.

        Required override unless the engine output is already shaped
        the way the controller expects.  For the cosmos-rl
        ``RolloutBase`` contract this is ``List[RolloutResult]``;
        tensor-native backends may return ``List[Dict[str, Tensor]]``
        instead.
        """
        raise NotImplementedError("[RolloutGenerationMixin] _postprocess is required.")

    def _payload_key(self, payload: Any) -> Hashable:
        """Stable cache key for prefetch.  Default: ``prompt_idx`` or ``id()``.

        The default is correct for any payload that carries a
        ``prompt_idx`` attribute (e.g. ``RLPayload``); backends that
        receive raw dicts override.
        """
        idx = getattr(payload, "prompt_idx", None)
        if idx is not None and idx >= 0:
            return ("idx", idx)
        return ("id", id(payload))

    def _on_generation_error(
        self,
        err: BaseException,
        payloads: Sequence[Any],
    ) -> Any:
        """Called when :meth:`_generate` (or the prepare/collate hooks)
        raises.

        Default: log at ``ERROR`` level and return ``[]`` so the
        rollout worker's ``main_loop`` skips this batch and continues.
        Backends that want to retry or report partial results override.
        """
        n = len(payloads) if payloads is not None else 0
        self._gen_logger.error(
            "[RolloutGenerationMixin] generation failed for batch of %d (%s: %s); "
            "returning empty batch.",
            n,
            type(err).__name__,
            err,
        )
        return []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup_generation(
        self,
        *,
        thread_name: str = "RolloutGen",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize prefetch state.  Idempotent, safe to call multiple times.

        Reads ``self.config.rollout.prefetch_rollout`` to decide whether
        to start the background worker.  When prefetch is off, this is
        effectively a tiny constructor that initializes the per-instance
        bookkeeping without spawning a thread.

        Subclasses typically call this from ``post_init_hook``.
        """
        # Re-entrant: bail out cleanly on a second call.
        if getattr(self, "_gen_initialized", False):
            return

        self._gen_logger = logger or _default_logger
        self._setup_futures: Dict[Hashable, Future] = {}
        self._setup_futures_lock = threading.Lock()
        self._setup_shutdown = threading.Event()
        self._setup_request_queue: "queue.Queue[Any]" = queue.Queue()
        self._setup_thread: Optional[threading.Thread] = None

        cfg = getattr(self, "config", None)
        rollout_cfg = getattr(cfg, "rollout", None) if cfg is not None else None
        self._prefetch_enabled = bool(getattr(rollout_cfg, "prefetch_rollout", False))

        if self._prefetch_enabled:
            self._setup_thread = threading.Thread(
                target=self._setup_worker_loop,
                name=thread_name,
                daemon=True,
            )
            self._setup_thread.start()
            self._gen_logger.info(
                "[RolloutGenerationMixin] prefetch enabled; setup worker '%s' started",
                thread_name,
            )
        else:
            self._gen_logger.debug(
                "[RolloutGenerationMixin] prefetch disabled; running prepare inline"
            )

        self._gen_initialized = True

    def shutdown_generation(self) -> None:
        """Stop the bg worker and clear pending futures.

        Safe to call when :meth:`setup_generation` was never called or
        was called with prefetch disabled.  Joins the thread with a
        small timeout to avoid blocking shutdown forever on a stuck
        ``_prepare_sample``.
        """
        if not getattr(self, "_gen_initialized", False):
            return
        self._setup_shutdown.set()
        # Wake the worker if it's blocked on the queue.
        try:
            self._setup_request_queue.put_nowait(_SHUTDOWN)
        except Exception:  # pragma: no cover - queue.Full unreachable for unbounded
            pass
        if self._setup_thread is not None and self._setup_thread.is_alive():
            self._setup_thread.join(timeout=2.0)
        with self._setup_futures_lock:
            for fut in self._setup_futures.values():
                # Cancel best-effort; if already running on the worker,
                # ``cancel`` is a no-op and the result is discarded.
                fut.cancel()
            self._setup_futures.clear()
        self._gen_initialized = False

    # ------------------------------------------------------------------
    # Prefetch entry (called by rollout_control._prefetch_loop)
    # ------------------------------------------------------------------

    def submit_setup(self, payloads: Sequence[Any]) -> None:
        """Schedule :meth:`_prepare_sample` for each payload.

        Called from the rollout worker's ``_prefetch_loop`` thread after
        it puts a batch on the prompt queue.  No-op when prefetch is
        disabled or :meth:`setup_generation` was never called.

        Resubmissions: if a key is already pending or done, the new
        request replaces it.  This handles the rare cancellation /
        re-issue path; the controller doesn't normally repeat indices.
        """
        if not getattr(self, "_prefetch_enabled", False):
            return
        if not getattr(self, "_gen_initialized", False):
            return
        for payload in payloads:
            key = self._payload_key(payload)
            future: Future = Future()
            with self._setup_futures_lock:
                old = self._setup_futures.get(key)
                if old is not None and not old.done():
                    old.cancel()
                self._setup_futures[key] = future
            # The worker pulls (key, payload, future) and runs the hook.
            self._setup_request_queue.put((key, payload, future))

    # ------------------------------------------------------------------
    # Final template method
    # ------------------------------------------------------------------

    @final
    def rollout_generation(
        self,
        payloads: Sequence[Any],
        stream: Any = None,
        data_packer: Any = None,
        data_fetcher: Any = None,
        is_validation: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Template method: prepare → collate → generate → postprocess.

        Final by convention; backends override the four hooks instead.

        Errors during the data-shaping stages (prepare / collate /
        generate / postprocess) flow through :meth:`_on_generation_error`
        so a single bad batch doesn't kill the rollout worker.  The
        engine-initialized check is *not* wrapped — calling
        ``rollout_generation`` before ``init_engine`` is a programming
        error and should crash loudly.

        Emits a structured per-batch trace line at ``DEBUG`` level so
        downstream analyzers can prove the bg setup thread is actually
        overlapping ``_prepare_sample`` with ``_generate`` rather than
        merely creating busy work.  The line shape is intentionally
        stable; see :data:`_TRACE_LINE_FORMAT` for the contract.
        """
        if not getattr(self, "_gen_initialized", False):
            # Composing class forgot setup_generation(); fall back to
            # an inline path that's still safe (just no prefetch).
            self.setup_generation()
        # Lifecycle / programming-error checks; must propagate (not swallowed).
        self._preflight_check()
        # Gate trace emission on DEBUG so production INFO-level logs
        # stay quiet and we pay only an isEnabledFor() lookup per batch.
        trace_on = self._gen_logger.isEnabledFor(logging.DEBUG)
        try:
            t_start = time.perf_counter() if trace_on else 0.0
            samples = self._gather_prepared_samples(
                payloads,
                data_packer=data_packer,
                data_fetcher=data_fetcher,
                is_validation=is_validation,
            )
            t_gather_end = time.perf_counter() if trace_on else 0.0
            batch = self._collate_batch(
                samples, data_packer=data_packer, is_validation=is_validation
            )
            t_collate_end = time.perf_counter() if trace_on else 0.0
            raw = self._generate(batch, stream=stream, is_validation=is_validation)
            t_generate_end = time.perf_counter() if trace_on else 0.0
            result = self._postprocess(raw, payloads, is_validation=is_validation)
            t_post_end = time.perf_counter() if trace_on else 0.0

            if trace_on:
                breakdown = getattr(self, "_last_gather_timing", {})
                self._gen_logger.debug(
                    _TRACE_LINE_FORMAT,
                    len(payloads) if payloads else 0,
                    bool(self._prefetch_enabled),
                    (t_gather_end - t_start) * 1000.0,
                    (t_collate_end - t_gather_end) * 1000.0,
                    (t_generate_end - t_collate_end) * 1000.0,
                    (t_post_end - t_generate_end) * 1000.0,
                    breakdown.get("inline_count", 0),
                    breakdown.get("inline_ms_total", 0.0),
                    breakdown.get("wait_count", 0),
                    breakdown.get("wait_ms_total", 0.0),
                )
            return result
        except Exception as err:
            return self._on_generation_error(err, payloads)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _preflight_check(self) -> None:
        """Lifecycle assertions that must crash loudly, not be swallowed.

        Runs *outside* the template's ``try``/``except`` so that any
        exception raised here propagates past
        :meth:`_on_generation_error`.  The default checks that
        :meth:`init_engine` has been called.  Backends that have
        additional lifecycle invariants (e.g. ``policy is not None``,
        KV-cache allocated) override and call ``super()._preflight_check()``::

            def _preflight_check(self) -> None:
                super()._preflight_check()
                if self.policy is None:
                    raise RuntimeError(
                        "Policy not set. Call set_underlying_model() first."
                    )

        These are programming/lifecycle errors -- the worker is in an
        inconsistent state -- and must propagate.  Per-batch *data*
        errors (e.g. an unparseable prompt) belong inside the four
        hooks where they get routed through :meth:`_on_generation_error`
        and the worker keeps running.
        """
        self._assert_engine_initialized()

    def _assert_engine_initialized(self) -> None:
        """Mirror the assertion every backend writes today."""
        if not getattr(self, "_engine_initialized", False):
            raise RuntimeError(
                f"[{type(self).__name__}] engine is not initialized; "
                "call init_engine() before rollout_generation()."
            )

    def _gather_prepared_samples(
        self,
        payloads: Sequence[Any],
        *,
        data_packer: Any,
        data_fetcher: Any,
        is_validation: bool,
    ) -> List[Any]:
        """Return prepared samples in payload order.

        Prefetch path: pop the matching future per payload, await it,
        propagate any exception.  Cold-start path (no future for a
        key): run :meth:`_prepare_sample` inline for that payload.
        Inline path (prefetch disabled): run :meth:`_prepare_sample`
        for every payload on the calling thread.

        Records per-call timing on ``self._last_gather_timing`` so the
        template's trace line can attribute time to ``wait`` (future
        was still in flight when we needed it -- prefetch isn't fully
        overlapping) vs ``inline`` (no future, ran on the main thread).
        Overhead is two ``perf_counter()`` reads per payload (sub-µs);
        cheap enough to leave on unconditionally.
        """
        prefetch = getattr(self, "_prefetch_enabled", False)
        results: List[Any] = []
        inline_ms_total = 0.0
        inline_count = 0
        wait_ms_total = 0.0
        wait_count = 0
        for payload in payloads:
            future: Optional[Future] = None
            if prefetch:
                key = self._payload_key(payload)
                with self._setup_futures_lock:
                    future = self._setup_futures.pop(key, None)
            if future is not None:
                t0 = time.perf_counter()
                # Future.result re-raises whatever the hook raised.
                results.append(future.result())
                wait_ms_total += (time.perf_counter() - t0) * 1000.0
                wait_count += 1
            else:
                t0 = time.perf_counter()
                results.append(
                    self._prepare_sample(
                        payload,
                        data_packer=data_packer,
                        data_fetcher=data_fetcher,
                        is_validation=is_validation,
                    )
                )
                inline_ms_total += (time.perf_counter() - t0) * 1000.0
                inline_count += 1
        self._last_gather_timing = {
            "inline_ms_total": inline_ms_total,
            "inline_count": inline_count,
            "wait_ms_total": wait_ms_total,
            "wait_count": wait_count,
        }
        return results

    def _setup_worker_loop(self) -> None:
        """Background thread body: pull (key, payload, future) and run the hook."""
        while not self._setup_shutdown.is_set():
            try:
                item = self._setup_request_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is _SHUTDOWN:
                break
            key, payload, future = item
            if future.cancelled():
                continue
            # Use the bound method so subclasses see ``self``.
            try:
                # The setup worker doesn't have access to the
                # ``data_packer`` / ``data_fetcher`` / ``is_validation``
                # context from the eventual ``rollout_generation`` call
                # because prefetch fires before the consumer hits the
                # template method.  We resolve them from instance
                # attributes the consumer set up (e.g. via
                # ``post_init_hook``); backends that need request-scoped
                # context should pass a closure-bound payload instead.
                result = self._prepare_sample(
                    payload,
                    data_packer=getattr(self, "_prefetch_data_packer", None),
                    data_fetcher=getattr(self, "_prefetch_data_fetcher", None),
                    is_validation=False,
                )
                future.set_result(result)
            except BaseException as err:  # noqa: BLE001 - propagate to consumer
                future.set_exception(err)

    # ------------------------------------------------------------------
    # Prefetch context (set once, used by the bg worker)
    # ------------------------------------------------------------------

    def bind_prefetch_context(
        self,
        *,
        data_packer: Any = None,
        data_fetcher: Any = None,
    ) -> None:
        """Bind the data_packer / data_fetcher that the bg setup worker should
        pass to :meth:`_prepare_sample`.

        Prefetch fires from ``_prefetch_loop`` *before* the consumer
        thread reaches ``rollout_generation``, so the bg worker can't
        learn the per-call ``data_packer`` / ``data_fetcher`` arguments
        the way the inline path does.  Composing classes should call
        this once during ``post_init_engine_hook`` (or an equivalent
        late-init point) so the worker has everything it needs.

        Backends that don't use the default :meth:`_prepare_sample`
        (which reads ``data_packer``) can leave both ``None``.
        """
        if not getattr(self, "_gen_initialized", False):
            self.setup_generation()
        self._prefetch_data_packer = data_packer
        self._prefetch_data_fetcher = data_fetcher


__all__ = ["RolloutGenerationMixin"]
