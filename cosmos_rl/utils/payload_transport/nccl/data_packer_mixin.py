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

"""NCCLDataPackerMixin -- NCCL-specific subclass of PrefetchDataPackerMixin.

Like :class:`UCXXDataPackerMixin`, this file is intentionally thin: the
prefetch + double-buffer + early-train-ack scheduling is owned by
:class:`cosmos_rl.utils.payload_transport.prefetch_mixin.PrefetchDataPackerMixin`,
and this subclass plugs in the NCCL-specific bits:

* the wire-format predicate (:meth:`_should_intercept`) — accepts either
  the on-wire completion string ``"nccl:<transfer_id>"`` (the same string
  the controller's discard-cleanup dispatch keys on) *or* a richer dict
  ``{"_nccl": True, ...}`` reference; both normalize to one internal ref.
* the cache key (:meth:`_cache_key` -> ``transfer_id``).
* the actual recv path (:meth:`_fetch_all`): for the batch, run the
  per-transfer Redis rendezvous, then issue each ``nccl_recv`` STANDALONE
  (one per 2-rank pair comm — NOT wrapped in a cross-communicator
  ``ncclGroupStart/End``, which would couple independent producers into one
  completion unit whose ``ncclGroupEnd`` blocks if any producer is slow), on
  a transfer stream from the per-process pool, recording a recv-complete
  event that gates downstream training consumption.
* layered retry, all inside a single ``_fetch_batch``: per-ref
  ``max_attempts`` fresh rendezvous calls plus per-pair UID
  renegotiation.  (The base mixin drives ONE ``_fetch_batch`` per
  prefetch round and adds NO in-batch multi-round layer.  A ref the
  prefetch batch misses is retried ONCE synchronously by
  ``get_policy_input`` via ``_sync_fetch``; if that also misses, it is
  surfaced through ``_on_resolve_failed`` and the episode falls back --
  it is NOT rescheduled for a later prefetch round.)

Usage::

    class NCCLSimpleRLDataPacker(NCCLDataPackerMixin, SimpleRLDataPacker):
        pass

MRO ensures :meth:`PrefetchDataPackerMixin.get_policy_input` (inherited)
intercepts before delegating to the concrete packer.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.payload_transport.nccl.comm_cache import (
    CommCache,
    RECEIVER_LOCAL_RANK,
    SENDER_LOCAL_RANK,
)
from cosmos_rl.utils.payload_transport.nccl.context import (
    resolve_global_rank as _resolve_global_rank,
    resolve_prefix as _resolve_prefix,
)
from cosmos_rl.utils.payload_transport.nccl.protocol import (
    NCCL_COMPLETION_PREFIX,
    build_sender_request_channel,
    parse_transfer_rollout_idx,
)
from cosmos_rl.utils.payload_transport.nccl.rendezvous import (
    NcclRendezvous,
    TransferStatus,
)
from cosmos_rl.utils.trajectory import (
    EPISODE_LENGTH,
    VARLEN_FIELDS as _VARLEN_FIELDS,
    build_trajectory_schema,
    deserialize_schema,
    schema_layout,
)
from cosmos_rl.utils.payload_transport.nccl.streams import (
    bind_thread_device,
    get_transfer_stream_pool,
    record_event,
    wait_event,
)
from cosmos_rl.utils.payload_transport.prefetch_mixin import (
    PrefetchDataPackerMixin,
    get_trace_time,
)


_LOG_INTERVAL = 50

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


class NCCLDataPackerMixin(PrefetchDataPackerMixin):
    """NCCL subclass of :class:`PrefetchDataPackerMixin`.

    Place **before** the concrete DataPacker in the MRO::

        class NCCLSimpleRLDataPacker(NCCLDataPackerMixin, SimpleRLDataPacker):
            pass
    """

    # NCCL-specific state.  Scheduling state (queues, thread, cache,
    # double-buffer, step counter) is owned by the base mixin.
    _nccl_dp_device: Optional[torch.device] = None
    _nccl_dp_redis: Any = None
    _nccl_dp_config: Any = None
    _nccl_dp_rendezvous: Optional[NcclRendezvous] = None
    _nccl_dp_comm_cache: Optional[CommCache] = None
    _nccl_dp_streams: Any = None
    # Serializes the NCCL recv LAUNCH region (mirrors the producer's
    # _nccl_send_lock).  Both the prefetch worker and a trainer-thread cache-miss
    # _sync_fetch can reach _fetch_all; concurrent multi-comm recv launches on one
    # device deadlock natively.  Set in setup; lazily created in _fetch_all for
    # bare test harnesses that skip setup.
    _nccl_dp_recv_lock: Any = None
    _nccl_dp_receiver_rank: int = 0
    # Globally-unique policy-replica identity (this worker's ``replica_name``),
    # assigned by ``CommMixin._attach_payload_transport`` before setup.  Keys
    # the per-pair UID + producer comm cache so multiple policy replicas that
    # share ``receiver_rank`` never cross-wire.  See build_pair_uid_key.
    _nccl_dp_receiver_replica: Optional[str] = None
    _nccl_dp_prefix: str = ""
    _nccl_dp_max_attempts: int = 2
    _nccl_dp_recv_timeout: float = 5.0
    # Larger budget for the first (comm-creating) transfer of a pair, so a
    # slow cold-start comm build isn't cancelled + falsely quarantined.
    _nccl_dp_first_transfer_timeout: float = 30.0
    # Pairs that have completed >=1 successful transfer ("warm").  Until a pair
    # is warm we treat it patiently: long timeouts + NO abort/quarantine on
    # failure, so a healthy-but-slow comm survives the cold-start storm instead
    # of being torn down and rebuilt (churn).  Set in setup.
    _nccl_dp_warm_pairs: Any = None
    _nccl_dp_schema: Optional[list] = None

    # Cumulative stats for periodic INFO summaries.
    _nccl_dp_total_nccl: int = 0
    _nccl_dp_total_fallback: int = 0
    _nccl_dp_total_bytes: int = 0
    _nccl_dp_total_latency_ms: float = 0.0
    _nccl_dp_last_bytes: int = 0
    _nccl_dp_last_count: int = 0

    # ------------------------------------------------------------------
    # Backward-compat alias (parity with UCXXDataPackerMixin) so shared
    # test / observability code can read the prefetch cache via the
    # transport-prefixed name.
    # ------------------------------------------------------------------

    @property
    def _nccl_dp_prefetch_cache(self) -> Dict[str, Any]:
        return self._prefetch_cache

    @_nccl_dp_prefetch_cache.setter
    def _nccl_dp_prefetch_cache(self, value: Dict[str, Any]) -> None:
        self._prefetch_cache = value

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def _setup_nccl_data_packer(
        self,
        *,
        device: Any,
        redis_client: Any,
        config: Any = None,
        prefetch_timeout: float = 30.0,
        max_attempts: int = 2,
        recv_timeout: float = 5.0,
        first_transfer_timeout: float = 30.0,
    ) -> None:
        """Initialise NCCL rendezvous / comm cache + start the prefetch thread.

        Normally invoked automatically by
        :meth:`NcclPayloadTransport.attach_data_packer`.  Direct calls are
        only needed in tests or unusual lifecycle setups.

        Args:
            device: Target GPU device for fetched tensors.
            redis_client: Live Redis client for the control plane.
            config: The run :class:`Config`; supplies the experiment name
                (for the Redis key prefix) and ``custom`` schema tunables.
            prefetch_timeout: Per-batch wait ceiling (seconds).
            max_attempts: Total attempts per transfer (initial + transient
                retries).
            recv_timeout: Per-``nccl_recv`` / per-rendezvous wall-clock
                budget so a wedged sender engages retry / quarantine fast.
        """
        self._nccl_dp_device = device
        self._nccl_dp_redis = redis_client
        self._nccl_dp_config = config
        self._nccl_dp_max_attempts = max(1, max_attempts)
        self._nccl_dp_recv_timeout = recv_timeout
        self._nccl_dp_first_transfer_timeout = max(recv_timeout, first_transfer_timeout)
        self._nccl_dp_recv_lock = threading.Lock()
        self._nccl_dp_warm_pairs = set()
        self._nccl_dp_receiver_rank = _resolve_global_rank()
        # ``_attach_payload_transport`` sets ``_nccl_dp_receiver_replica`` to
        # this worker's ``replica_name`` before setup.  Fall back to a
        # rank-derived id for standalone/test setups (unique within a single
        # replica, which is all such setups have).
        if not self._nccl_dp_receiver_replica:
            self._nccl_dp_receiver_replica = f"recv{self._nccl_dp_receiver_rank}"
        self._nccl_dp_prefix = _resolve_prefix(config)
        self._nccl_dp_schema = build_trajectory_schema(_resolve_schema_dims(config))

        # The per-pair UID key must outlive a cold-start request: it is
        # published when the receiver initiates and read by the sender when it
        # finally serves (up to first_transfer_timeout later, behind the
        # comm-init storm).  If the UID's TTL expired first, the sender would
        # see a truthy uid_key, ACCEPT, but read_uid() -> None, build from an
        # empty UID, and the two comm halves would never join.  Keep the TTL
        # comfortably above the cold-start budget.
        uid_ttl_s = max(60, int(self._nccl_dp_first_transfer_timeout) + 30)
        self._nccl_dp_rendezvous = NcclRendezvous(
            redis_client, self._nccl_dp_prefix, uid_ttl_s=uid_ttl_s
        )
        self._nccl_dp_comm_cache = CommCache(
            quarantine_cooldown=max(1.0, recv_timeout * 6.0),
        )
        self._nccl_dp_streams = get_transfer_stream_pool(size=1, device=device)

        self._setup_prefetch(
            prefetch_timeout=prefetch_timeout,
            thread_name="NCCLDataPackerPrefetch",
        )

        logger.info(
            "[NCCLDataPackerMixin] Initialised: device=%s, rank=%d, timeout=%ss, "
            "max_attempts=%d, recv_timeout=%ss",
            device,
            self._nccl_dp_receiver_rank,
            prefetch_timeout,
            self._nccl_dp_max_attempts,
            recv_timeout,
        )

    def _abort_nccl_dp_comms(self) -> None:
        """Abort every cached comm so an in-flight ``nccl_recv`` returns.

        Used as ``shutdown_prefetch``'s ``before_join`` hook: the prefetch
        worker only checks the shutdown event *between* batches, so a recv
        parked on a departed peer would otherwise hold the join for the full
        first-transfer budget rather than the join timeout.
        """
        if self._nccl_dp_comm_cache is not None:
            try:
                self._nccl_dp_comm_cache.abort_all()
            except Exception as e:  # pragma: no cover - teardown best-effort
                logger.warning("[NCCLDataPackerMixin] comm abort failed: %s", e)

    def shutdown_nccl_data_packer(self) -> None:
        """Stop the prefetch thread and abort all cached communicators.

        Aborts BEFORE joining -- mirroring the producer's ``cleanup_nccl`` --
        so the worker's wedged recv fails fast instead of the join waiting on
        work only the abort can unwedge.
        """
        self.shutdown_prefetch(before_join=self._abort_nccl_dp_comms)
        if self._prefetch_step_count > 0:
            avg_ms = self._nccl_dp_total_latency_ms / self._prefetch_step_count
            logger.info(
                "[NCCLDataPackerMixin] Final: %d iters, %d NCCL / %d fallback, "
                "%.1f MB, avg %.0f ms/iter",
                self._prefetch_step_count,
                self._nccl_dp_total_nccl,
                self._nccl_dp_total_fallback,
                self._nccl_dp_total_bytes / 1e6,
                avg_ms,
            )
        logger.info("[NCCLDataPackerMixin] Shut down")

    # ------------------------------------------------------------------
    # PrefetchDataPackerMixin hook implementations
    # ------------------------------------------------------------------

    def _should_intercept(self, rollout_output: Any) -> bool:
        """NCCL wire-format predicate (string completion or dict ref)."""
        if isinstance(rollout_output, str):
            return rollout_output.startswith(NCCL_COMPLETION_PREFIX)
        if isinstance(rollout_output, dict):
            return bool(rollout_output.get("_nccl"))
        return False

    def _cache_key(self, rollout_output: Any) -> str:
        return self._nccl_dp_cache_key(rollout_output)

    def _fetch_batch(self, tasks: List[Any]) -> Dict[str, Any]:
        """Resolve a batch of NCCL references via rendezvous + standalone per-pair recvs."""
        refs: List[Tuple[Any, dict]] = []
        for idx, ro in tasks:
            ref = _parse_ref(ro, default_schema=self._nccl_dp_schema)
            if ref is not None:
                refs.append((idx, ref))

        results, total_bytes, transfer_ms = self._fetch_all(refs)

        cache_results: Dict[str, Any] = {}
        for idx, gpu_data in results.items():
            key = _cache_key_from_task(tasks, idx)
            cache_results[key] = gpu_data

        self._nccl_dp_last_bytes = total_bytes
        self._nccl_dp_last_count = len(results)
        if results:
            logger.debug(
                "[Trace] thread=nccl_prefetch op=nccl_fetch transfer_ms=%.1f "
                "count=%d bytes=%d",
                transfer_ms,
                len(results),
                total_bytes,
            )
        return cache_results

    def _sync_fetch(self, rollout_output: Any) -> Optional[Dict[str, torch.Tensor]]:
        """Blocking single-episode NCCL fetch (cache-miss fallback).

        Returns ``None`` on any failure rather than propagating: the base
        mixin calls this on the cache-miss path inside ``get_policy_input``
        without its own containment, so a raised rendezvous/recv error would
        crash the training step instead of degrading to the packer's fallback.
        Mirrors the UCXX consumer's sync-fallback contract.
        """
        try:
            # Parse inside the guard too: a malformed dict ref (e.g. a corrupt
            # ``_schema``) raises from deserialize_schema, and this path has no
            # containment above it in the base get_policy_input.
            ref = _parse_ref(rollout_output, default_schema=self._nccl_dp_schema)
            if ref is None:
                return None
            results, _, _ = self._fetch_all([(0, ref)])
        except Exception as e:
            logger.warning("[NCCLDataPackerMixin] Sync fallback failed: %s", e)
            return None
        return results.get(0)

    def _on_prefetch_complete(
        self, batch_id: int, n_results: int, fetch_ms: float
    ) -> None:
        self._nccl_dp_total_nccl += getattr(self, "_nccl_dp_last_count", n_results)
        self._nccl_dp_total_bytes += getattr(self, "_nccl_dp_last_bytes", 0)
        self._nccl_dp_total_latency_ms += fetch_ms
        step = self._prefetch_step_count
        if step == 1 or step % _LOG_INTERVAL == 0:
            avg_ms = self._nccl_dp_total_latency_ms / step
            logger.info(
                "[NCCLDataPackerMixin] Iteration %d: %d NCCL, %.1f MB total, "
                "avg %.0f ms/iter",
                step,
                self._nccl_dp_total_nccl,
                self._nccl_dp_total_bytes / 1e6,
                avg_ms,
            )

    def _on_resolve_failed(self, rollout_output: Any, cache_key: str) -> None:
        self._nccl_dp_total_fallback += 1

    # get_policy_input is inherited unchanged from PrefetchDataPackerMixin.

    # ------------------------------------------------------------------
    # NCCL recv path
    # ------------------------------------------------------------------

    def _fetch_all(self, refs: List[Tuple[Any, dict]]) -> Tuple[dict, int, float]:
        """Rendezvous + grouped ``nccl_recv`` for every ref.

        Returns ``(results_by_idx, total_bytes, transfer_ms)``.  Each ref
        is negotiated over Redis first (so we know which recvs will
        actually happen and on which comm), then each accepted recv is
        issued as a STANDALONE ``nccl_recv`` on its own 2-rank pair comm
        (deliberately NOT wrapped in a cross-communicator
        ``ncclGroupStart/End``, which would couple independent producers
        into one completion unit -- the N_POLICY>=2 wedge).  A per-ref
        ``max_attempts`` fresh-call retry wraps the rendezvous; a ref that
        still fails to resolve is dropped and re-attempted on the next
        prefetch round (there is no in-batch multi-round layer above this).
        """
        from cosmos_rl.utils import pynccl

        rv = self._nccl_dp_rendezvous
        cache = self._nccl_dp_comm_cache
        device = self._nccl_dp_device
        if rv is None or cache is None:
            return {}, 0, 0.0
        # The prefetch worker runs off the main thread; bind it to our GPU so
        # comm creation + recvs target the right device (thread-local).
        bind_thread_device(device)

        t0 = get_trace_time()
        # Phase 1: rendezvous (control plane) — sequential Redis round-trips.
        recvs: List[Tuple[Any, dict, int, torch.Tensor]] = []
        for idx, ref in refs:
            prepared = self._rendezvous_one(ref, pynccl)
            if prepared is None:
                continue
            comm_idx, recv_buf = prepared
            recvs.append((idx, ref, comm_idx, recv_buf))

        if not recvs:
            return {}, 0, get_trace_time() - t0

        # A batch is "warming" until every pair in it has transferred at least
        # once; give its recvs the long cold-start budget so a slow (storm-
        # contended) send isn't cancelled -> comm torn down -> rebuilt into the
        # same storm.
        receiver_rank = self._nccl_dp_receiver_rank
        warm = self._nccl_dp_warm_pairs
        batch_warming = any(
            _pair_key(ref, receiver_rank) not in warm for _i, ref, _c, _b in recvs
        )
        recv_timeout_ms = int(
            (
                self._nccl_dp_first_transfer_timeout
                if batch_warming
                else self._nccl_dp_recv_timeout
            )
            * 1000
        )

        # Phase 2: issue ONE STANDALONE recv per pair communicator -- NOT a
        # cross-communicator NCCL group.  Each comm is a distinct 2-rank pair
        # with a single send/recv, so grouping adds no overlap; it only turns
        # independent producers into one completion unit whose native
        # ncclGroupEnd blocks -- with NO pynccl watchdog (run_task arms its
        # deadline only AFTER the native call returns) -- if any producer has
        # not yet posted its matching send (its send lock is busy serving the
        # other policy replica).  That coupling was the N_POLICY>=2 residual
        # wedge.  Ungrouped, a slow/serialized producer only delays its own
        # recv; the others complete independently.
        results: Dict[int, dict] = {}
        total_bytes = 0
        posted: List[Tuple[Any, dict, int, torch.Tensor]] = []
        # Serialize the NCCL recv LAUNCH on this consumer's GPU, mirroring the
        # producer's _nccl_send_lock.  Both the prefetch worker (_fetch_all off the
        # prefetch thread) and a trainer-thread cache-miss _sync_fetch can reach
        # here; concurrent multi-comm recv launches on one device deadlock
        # natively -- and because run_task arms its deadline only AFTER the native
        # call returns, the recv timeout can't rescue a launch deadlock.  Only the
        # async ENQUEUE + event record run under the lock (every op is a stream
        # enqueue); the blocking synchronize() below stays lock-free so real
        # transfers still overlap across the two callers.
        recv_lock = self._nccl_dp_recv_lock
        if recv_lock is None:  # bare test harness that skipped setup
            recv_lock = self._nccl_dp_recv_lock = threading.Lock()
        with recv_lock:
            stream = self._nccl_dp_streams.acquire() if self._nccl_dp_streams else None
            for idx, ref, comm_idx, recv_buf in recvs:
                try:
                    pynccl.nccl_recv(
                        recv_buf,
                        SENDER_LOCAL_RANK,  # peer in the 2-rank comm is the sender
                        comm_idx,
                        stream=stream,
                        timeout_ms=recv_timeout_ms,
                    )
                except Exception as exc:
                    logger.warning(
                        "[NCCLDataPackerMixin] recv failed for %s: %s",
                        ref["transfer_id"],
                        exc,
                    )
                    # Isolate the failure to this pair (quarantine only if warm).
                    self._quarantine_recv_failures(
                        [(idx, ref, comm_idx, recv_buf)], cache
                    )
                    continue
                posted.append((idx, ref, comm_idx, recv_buf))

            if not posted:
                return {}, 0, get_trace_time() - t0

            # Recv-complete event gates downstream training consumption.
            done = record_event(stream)

        wait_event(None, done)  # current (compute) stream waits before reads
        if torch.cuda.is_available():
            try:
                torch.cuda.current_stream().synchronize()
            except Exception as exc:
                # A recv that ENQUEUED cleanly but whose peer never sent (dead /
                # hung producer) surfaces HERE at completion, not at the enqueue
                # try/except above.  Do NOT let it propagate: an uncaught raise
                # unwinds _fetch_all -> the prefetch worker marks the WHOLE batch
                # failed -> wait_prefetch wipes the cache -> every episode drops to
                # fallback AND the offending pair is never quarantined (quarantine
                # is only reachable from the enqueue path).  A single stream sync
                # can't attribute the failure to one recv, so conservatively
                # quarantine every posted (warm) pair and drop just this batch to
                # fallback; the dead pair(s) now cool down instead of re-storming.
                logger.warning(
                    "[NCCLDataPackerMixin] recv completion sync failed "
                    "(%d posted pairs): %s; quarantining posted pairs",
                    len(posted),
                    exc,
                )
                self._quarantine_recv_failures(posted, cache)
                return {}, 0, get_trace_time() - t0

        for idx, ref, _comm_idx, recv_buf in posted:
            gpu_data = _unpack(recv_buf, ref["schema"], device)
            results[idx] = gpu_data
            total_bytes += recv_buf.numel() * recv_buf.element_size()
            # First successful transfer -> this pair is warm (tight timeouts +
            # normal quarantine from here on).
            warm.add(_pair_key(ref, receiver_rank))

        return results, total_bytes, get_trace_time() - t0

    def _quarantine_endpoint(self, cache: Any, health_key: Any, pair: Any) -> None:
        """Quarantine a warm endpoint AND demote its pair back to 'warming'.

        Clearing the warm marker (comm-generation recovery) means the
        post-cooldown rebuild is treated as a cold start: it gets the generous
        ``first_transfer_timeout`` budget instead of the tight ``recv_timeout``,
        so the freshly-renegotiated comm isn't immediately re-quarantined by the
        very short window that caused the original failure.  Pairs with the
        producer's stale comm-half are also torn down by ``quarantine`` (abort),
        and the producer rebuilds its half on the next fresh UID.
        """
        try:
            cache.quarantine(health_key)
        except Exception as exc:  # pragma: no cover - best-effort teardown
            logger.debug(
                "[NCCLDataPackerMixin] quarantine %s raised %s; continuing",
                health_key,
                type(exc).__name__,
            )
        if pair is not None:
            self._nccl_dp_warm_pairs.discard(pair)

    def _quarantine_recv_failures(
        self, recvs: List[Tuple[Any, dict, int, torch.Tensor]], cache: Any
    ) -> None:
        """After a recv failure, quarantine only the WARM pairs.

        A warm pair (has transferred before) that fails is a genuine problem ->
        quarantine + abort its comm (``CommCache.quarantine`` prefix-matches
        the ``(sender_replica, sender_rank, receiver_rank)`` comm).  A pair that
        is still WARMING is just storm-contended -> keep its built comm and let
        it retry next round; tearing it down here is what churned N_POLICY>=2.
        """
        receiver_rank = self._nccl_dp_receiver_rank
        warm = self._nccl_dp_warm_pairs
        for _idx, ref, _comm_idx, _buf in recvs:
            pair = _pair_key(ref, receiver_rank)
            if pair not in warm:
                continue  # still warming -> keep the comm, retry
            self._quarantine_endpoint(
                cache, (ref["sender_replica"], ref["sender_rank"]), pair
            )

    def _rendezvous_one(
        self, ref: dict, pynccl: Any
    ) -> Optional[Tuple[int, torch.Tensor]]:
        """Negotiate one transfer; return ``(comm_idx, recv_buf)`` or None.

        Applies the per-ref fresh-call retry and health-aware quarantine.
        """
        rv = self._nccl_dp_rendezvous
        cache = self._nccl_dp_comm_cache
        transfer_id = ref["transfer_id"]
        sender_rank = ref["sender_rank"]
        sender_replica = ref["sender_replica"]
        receiver_rank = self._nccl_dp_receiver_rank
        receiver_replica = self._nccl_dp_receiver_replica
        # Pair key + health key are keyed on the globally-unique sender
        # replica identity so distinct rollout replicas never collide.
        pair = _pair_key(ref, receiver_rank)
        health_key = (sender_replica, sender_rank)
        # "warming" until this pair completes its first successful transfer.
        warming = pair not in self._nccl_dp_warm_pairs

        if cache.is_quarantined(health_key):
            logger.debug(
                "[NCCLDataPackerMixin] endpoint %s quarantined; skipping %s",
                health_key,
                transfer_id,
            )
            return None

        request_channel = build_sender_request_channel(self._nccl_dp_prefix, ref)

        for attempt in range(1, self._nccl_dp_max_attempts + 1):
            need_uid = pair not in cache
            # Until the pair is warm, give it the long cold-start budget so the
            # comm-init storm doesn't cancel a healthy-but-slow transfer; a warm
            # pair uses the tight steady-state budget.
            timeout = (
                self._nccl_dp_first_transfer_timeout
                if warming
                else self._nccl_dp_recv_timeout
            )
            result = rv.initiate(
                transfer_id=transfer_id,
                sender_replica=sender_replica,
                sender_rank=sender_rank,
                receiver_replica=receiver_replica,
                receiver_rank=receiver_rank,
                request_channel=request_channel,
                need_uid=need_uid,
                timeout=timeout,
                attempt=attempt,
            )
            if result.status is TransferStatus.ACCEPTED:
                try:
                    comm_idx = cache.get_or_create(
                        pair,
                        uid_chars=result.uid_chars or [],
                        local_rank=RECEIVER_LOCAL_RANK,
                    )
                except Exception as e:
                    logger.warning(
                        "[NCCLDataPackerMixin] comm build failed for %s: %s%s",
                        transfer_id,
                        e,
                        "; retry next round (warming)" if warming else "; quarantining",
                    )
                    if not warming:
                        self._quarantine_endpoint(cache, health_key, pair)
                    return None
                recv_buf = _alloc_recv_buffer(ref["schema"], self._nccl_dp_device)
                return comm_idx, recv_buf
            if result.status is TransferStatus.MISSING:
                # Producer recycled the buffer — non-retryable, drop now.
                logger.debug(
                    "[NCCLDataPackerMixin] transfer %s missing (recycled)",
                    transfer_id,
                )
                return None
            if result.status is TransferStatus.NEED_UID:
                # The sender evicted its side of the comm; our cached comm is
                # now half-open.  Drop it and retry -- the next attempt has
                # need_uid=True and mints a fresh uid so BOTH sides rebuild.
                logger.debug(
                    "[NCCLDataPackerMixin] transfer %s: sender needs a fresh "
                    "uid; dropping stale comm and renegotiating",
                    transfer_id,
                )
                cache.abort(pair)
                continue
            # CANCELLED (timeout).
            if attempt == self._nccl_dp_max_attempts:
                if not warming:
                    # A WARM pair (has transferred before) that stops
                    # rendezvousing -> the sender likely died mid-run;
                    # quarantine so we stop hammering it (and abort the comm).
                    logger.warning(
                        "[NCCLDataPackerMixin] transfer %s cancelled after %d "
                        "attempts on a warm comm; quarantining %s",
                        transfer_id,
                        self._nccl_dp_max_attempts,
                        health_key,
                    )
                    self._quarantine_endpoint(cache, health_key, pair)
                else:
                    # Still warming: the sender is warming up / storm-contended,
                    # NOT unhealthy.  KEEP any built comm and retry next round
                    # WITHOUT quarantining -- tearing it down here (and thus
                    # rebuilding into the same storm) is what churned N_POLICY>=2
                    # to 0 MB.
                    logger.debug(
                        "[NCCLDataPackerMixin] transfer %s still warming (%d "
                        "attempts); keeping comm, retry next round",
                        transfer_id,
                        self._nccl_dp_max_attempts,
                    )
        return None

    # ------------------------------------------------------------------
    # Cache-key helpers (parity with UCXX; used by tests + _fetch_batch).
    # ------------------------------------------------------------------

    @staticmethod
    def _nccl_dp_cache_key(rollout_output: Any) -> str:
        ref = _parse_ref(rollout_output)
        if ref is not None:
            return ref["transfer_id"]
        return str(rollout_output)


# ---------------------------------------------------------------------------
# Module-level helpers (ref parsing / buffer alloc / unpack)
# ---------------------------------------------------------------------------


def _parse_ref(
    rollout_output: Any, *, default_schema: Optional[list] = None
) -> Optional[dict]:
    """Normalize a completion string or dict metadata into an internal ref.

    Returns a dict with ``transfer_id``, ``sender_rank``, ``rollout_idx``,
    and ``schema`` (a list of :class:`TensorSpec`), or ``None`` if the
    input is not an NCCL reference.
    """
    if isinstance(rollout_output, str):
        if not rollout_output.startswith(NCCL_COMPLETION_PREFIX):
            return None
        transfer_id = rollout_output[len(NCCL_COMPLETION_PREFIX) :]
        rollout_idx = parse_transfer_rollout_idx(transfer_id)
        # Bare "nccl:<id>" string carries no dict metadata, so there is no
        # globally-unique replica identity -- fall back to the rollout-idx.
        # (This form is single-node / testing only; the rl-gym producer
        # returns dict metadata carrying _sender_replica.)
        return {
            "transfer_id": transfer_id,
            "rollout_idx": rollout_idx,
            "sender_rank": rollout_idx if rollout_idx >= 0 else 0,
            "sender_replica": f"rollout-{rollout_idx if rollout_idx >= 0 else 0}",
            "schema": default_schema,
        }
    if isinstance(rollout_output, dict) and rollout_output.get("_nccl"):
        transfer_id = rollout_output.get("_transfer_id", "")
        rollout_idx = rollout_output.get(
            "_rollout_idx", parse_transfer_rollout_idx(transfer_id)
        )
        schema = default_schema
        raw_schema = rollout_output.get("_schema")
        if raw_schema:
            schema = deserialize_schema(raw_schema)
        sender_rank = rollout_output.get(
            "_sender_rank", rollout_idx if rollout_idx >= 0 else 0
        )
        # Globally-unique sender identity: the producer's replica id/name.
        # Falls back to the rollout-idx form only if the producer omitted it.
        sender_replica = rollout_output.get("_sender_replica") or (
            f"rollout-{rollout_idx if rollout_idx >= 0 else 0}"
        )
        return {
            "transfer_id": transfer_id,
            "rollout_idx": rollout_idx,
            "sender_rank": sender_rank,
            "sender_replica": sender_replica,
            "schema": schema,
        }
    return None


def _pair_key(ref: dict, receiver_rank: int):
    """Globally-unique comm-cache pair key for a transfer.

    ``(sender_replica, sender_rank, receiver_rank)`` -- keyed on the
    rollout replica's globally-unique identity so two replicas sharing a
    per-replica ``sender_rank`` (e.g. both 0) map to DISTINCT communicators.
    """
    return (ref["sender_replica"], ref["sender_rank"], receiver_rank)


def _cache_key_from_task(tasks: List[Any], idx: int) -> str:
    for task_idx, ro in tasks:
        if task_idx == idx:
            return NCCLDataPackerMixin._nccl_dp_cache_key(ro)
    return str(idx)


def _alloc_recv_buffer(schema: Optional[list], device: Any) -> torch.Tensor:
    """Allocate a flat uint8 GPU buffer sized for ``schema``."""
    if schema is None:
        raise ValueError("cannot allocate NCCL recv buffer without a schema")
    _, entry_size = schema_layout(schema)
    return torch.empty(entry_size, dtype=torch.uint8, device=device)


def _unpack(recv_buf: torch.Tensor, schema: Optional[list], device: Any) -> dict:
    """Slice a flat recv buffer back into the named schema tensors."""
    if schema is None:
        return {}
    offsets, _ = schema_layout(schema)
    out: Dict[str, Any] = {}
    for spec in schema:
        td = _NP_TO_TORCH.get(np.dtype(spec.dtype))
        if td is None:
            raise ValueError(f"unsupported dtype {spec.dtype} for '{spec.name}'")
        off = offsets[spec.name]
        # Clone the byte slice BEFORE reinterpreting: a sub-tensor whose
        # storage_offset is not a multiple of the target itemsize (e.g. the
        # int64 episode_length landing at byte 300) cannot be ``view``-ed to
        # the wider dtype.  Cloning yields fresh storage at offset 0, which
        # is always aligned.  (Same reason UCXX clones before its view.)
        raw = recv_buf[off : off + spec.nbytes].clone()
        out[spec.name] = raw.view(td).reshape(spec.shape)
    _truncate_to_episode_len(out)
    return out


def _truncate_to_episode_len(data: dict) -> None:
    ep = data.get(EPISODE_LENGTH)
    if ep is None:
        return
    try:
        ep_len = int(ep.item()) if ep.numel() == 1 else int(ep[0].item())
    except Exception:
        return
    for key in _VARLEN_FIELDS:
        if key in data and data[key].shape[0] > ep_len:
            data[key] = data[key][:ep_len]


def _resolve_schema_dims(config: Any) -> dict:
    custom = getattr(config, "custom", None) or {}

    def _get(key, default):
        try:
            return int(custom.get(key, default))
        except (TypeError, ValueError):
            return default

    return {
        "max_steps": _get("nccl_max_steps", 100),
        "obs_dim": _get("nccl_obs_dim", 4),
        "action_dim": _get("nccl_action_dim", 2),
    }
