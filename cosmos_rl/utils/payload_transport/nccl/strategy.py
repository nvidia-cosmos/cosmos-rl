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

"""NCCL payload transport, expressed as a composable strategy.

Holds everything about moving a trajectory GPU->GPU -- the Redis-backed
rendezvous, the communicator cache, the transfer streams, retry and quarantine
-- and nothing about *when* a fetch happens, which is the packer's job.

This is the code the 8-GPU Slurm run and the 2-rank e2e probe validated, moved
off ``NCCLDataPackerMixin`` unchanged apart from dropping the ``_nccl_dp_``
prefix (its state no longer shares a namespace with a packer) and taking the
iteration counter as an argument instead of reading it off one.
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
    resolve_max_live_comms as _resolve_max_live_comms,
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
from cosmos_rl.utils.payload_transport.strategy import PayloadTransportStrategy
from cosmos_rl.utils.trace import get_trace_time


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


class NCCLTransportStrategy(PayloadTransportStrategy):
    """Resolve NCCL payload references for a single receiver.

    :meth:`fetch_batch` runs on the packer's prefetch thread; everything else
    runs on the caller's.  Each counter below is written from one side only,
    which is what keeps that split safe without locking.
    """

    _device: Optional[torch.device] = None
    _redis: Any = None
    _config: Any = None
    _rendezvous: Optional[NcclRendezvous] = None
    _comm_cache: Optional[CommCache] = None
    _streams: Any = None
    _recv_lock: Any = None
    _receiver_rank: int = 0
    _receiver_replica: Optional[str] = None
    _prefix: str = ""
    _max_attempts: int = 2
    _recv_timeout: float = 5.0
    _first_transfer_timeout: float = 30.0
    _warm_pairs: Any = None
    _schema: Optional[list] = None
    _total_nccl: int = 0
    _total_fallback: int = 0
    _total_bytes: int = 0
    _total_latency_ms: float = 0.0
    _last_bytes: int = 0
    _last_count: int = 0
    _steps: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(
        self,
        *,
        device: Any,
        redis_client: Any,
        config: Any = None,
        max_attempts: int = 2,
        recv_timeout: float = 5.0,
        first_transfer_timeout: float = 30.0,
        receiver_replica: Optional[str] = None,
    ) -> None:
        """Initialise the NCCL rendezvous, comm cache and transfer streams.

        Normally invoked by the packer that composes this strategy.  Direct calls are
        only needed in tests or unusual lifecycle setups.

        Args:
            device: Target GPU device for fetched tensors.
            redis_client: Live Redis client for the control plane.
            config: The run :class:`Config`; supplies the experiment name
                (for the Redis key prefix) and ``custom`` schema tunables.
            max_attempts: Total attempts per transfer (initial + transient
                retries).
            recv_timeout: Per-``nccl_recv`` / per-rendezvous wall-clock
                budget so a wedged sender engages retry / quarantine fast.
        """
        self._device = device
        self._redis = redis_client
        self._config = config
        self._max_attempts = max(1, max_attempts)
        self._recv_timeout = recv_timeout
        self._first_transfer_timeout = max(recv_timeout, first_transfer_timeout)
        self._recv_lock = threading.Lock()
        self._warm_pairs = set()
        self._receiver_rank = _resolve_global_rank()
        if receiver_replica:
            self._receiver_replica = receiver_replica
        # ``_attach_payload_transport`` sets ``_nccl_dp_receiver_replica`` to
        # this worker's ``replica_name`` before setup.  Fall back to a
        # rank-derived id for standalone/test setups (unique within a single
        # replica, which is all such setups have).
        if not self._receiver_replica:
            self._receiver_replica = f"recv{self._receiver_rank}"
        self._prefix = _resolve_prefix(config)
        self._schema = build_trajectory_schema(_resolve_schema_dims(config))

        # The per-pair UID key must outlive a cold-start request: it is
        # published when the receiver initiates and read by the sender when it
        # finally serves (up to first_transfer_timeout later, behind the
        # comm-init storm).  If the UID's TTL expired first, the sender would
        # see a truthy uid_key, ACCEPT, but read_uid() -> None, build from an
        # empty UID, and the two comm halves would never join.  Keep the TTL
        # comfortably above the cold-start budget.
        uid_ttl_s = max(60, int(self._first_transfer_timeout) + 30)
        self._rendezvous = NcclRendezvous(
            redis_client, self._prefix, uid_ttl_s=uid_ttl_s
        )
        # Cap sized from the peer fan-out (this side talks to rollout replicas),
        # never below the historical default.  A blind cap is what makes LRU
        # eviction dangerous: at the wrong scale every transfer rebuilds a comm.
        self._comm_cache = CommCache(
            max_live=_resolve_max_live_comms(config, peer_role="rollout"),
            quarantine_cooldown=max(1.0, recv_timeout * 6.0),
        )
        self._streams = get_transfer_stream_pool(size=1, device=device)

        logger.info(
            "[NCCLTransportStrategy] Initialised: device=%s, rank=%d, "
            "max_attempts=%d, recv_timeout=%ss",
            device,
            self._receiver_rank,
            self._max_attempts,
            recv_timeout,
        )

    def before_join(self) -> None:
        """Abort every cached comm so an in-flight ``nccl_recv`` returns.

        Used as ``shutdown_prefetch``'s ``before_join`` hook: the prefetch
        worker only checks the shutdown event *between* batches, so a recv
        parked on a departed peer would otherwise hold the join for the full
        first-transfer budget rather than the join timeout.
        """
        if self._comm_cache is not None:
            try:
                self._comm_cache.abort_all()
            except Exception as e:  # pragma: no cover - teardown best-effort
                logger.warning("[NCCLTransportStrategy] comm abort failed: %s", e)

    def shutdown(self) -> None:
        """Log the run summary and release the transport.

        Does NOT abort here: ``shutdown_prefetch`` already ran
        :meth:`before_join` (it defaults to the attached strategy) before the
        join, which is the ordering that lets a parked recv fail fast.
        Aborting again would just double-abort every cached comm.
        """
        if self._steps > 0:
            avg_ms = self._total_latency_ms / self._steps
            logger.info(
                "[NCCLTransportStrategy] Final: %d iters, %d NCCL / %d fallback, "
                "%.1f MB, avg %.0f ms/iter",
                self._steps,
                self._total_nccl,
                self._total_fallback,
                self._total_bytes / 1e6,
                avg_ms,
            )
        logger.info("[NCCLTransportStrategy] Shut down")

    # ------------------------------------------------------------------
    # PrefetchDataPackerMixin hook implementations
    # ------------------------------------------------------------------

    def should_intercept(self, rollout_output: Any) -> bool:
        """NCCL wire-format predicate (string completion or dict ref)."""
        if isinstance(rollout_output, str):
            return rollout_output.startswith(NCCL_COMPLETION_PREFIX)
        if isinstance(rollout_output, dict):
            return bool(rollout_output.get("_nccl"))
        return False

    def cache_key(self, rollout_output: Any) -> str:
        return self._ref_cache_key(rollout_output)

    def fetch_batch(self, tasks: List[Any]) -> Dict[str, Any]:
        """Resolve a batch of NCCL references via rendezvous + standalone per-pair recvs."""
        refs: List[Tuple[Any, dict]] = []
        for idx, ro in tasks:
            ref = _parse_ref(ro, default_schema=self._schema)
            if ref is not None:
                refs.append((idx, ref))

        results, total_bytes, transfer_ms = self._fetch_all(refs)

        cache_results: Dict[str, Any] = {}
        for idx, gpu_data in results.items():
            key = _cache_key_from_task(tasks, idx)
            cache_results[key] = gpu_data

        self._last_bytes = total_bytes
        self._last_count = len(results)
        if results:
            logger.debug(
                "[Trace] thread=nccl_prefetch op=nccl_fetch transfer_ms=%.1f "
                "count=%d bytes=%d",
                transfer_ms,
                len(results),
                total_bytes,
            )
        return cache_results

    def sync_fetch(self, rollout_output: Any) -> Optional[Dict[str, torch.Tensor]]:
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
            ref = _parse_ref(rollout_output, default_schema=self._schema)
            if ref is None:
                return None
            results, _, _ = self._fetch_all([(0, ref)])
        except Exception as e:
            logger.warning("[NCCLTransportStrategy] Sync fallback failed: %s", e)
            return None
        return results.get(0)

    def on_prefetch_complete(
        self, batch_id: int, n_results: int, fetch_ms: float, step: int
    ) -> None:
        # These are the batch's own figures, set by fetch_batch. Do NOT fall
        # back to n_results: the base passes len(self._prefetch_cache), i.e.
        # the whole double-buffered cache, which over-counts every step.
        self._total_nccl += self._last_count
        self._total_bytes += self._last_bytes
        self._total_latency_ms += fetch_ms
        self._steps = step
        if step == 1 or step % _LOG_INTERVAL == 0:
            avg_ms = self._total_latency_ms / step
            logger.info(
                "[NCCLTransportStrategy] Iteration %d: %d NCCL, %.1f MB total, "
                "avg %.0f ms/iter",
                step,
                self._total_nccl,
                self._total_bytes / 1e6,
                avg_ms,
            )

    def on_resolve_failed(self, rollout_output: Any, cache_key: str) -> None:
        self._total_fallback += 1

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

        rv = self._rendezvous
        cache = self._comm_cache
        device = self._device
        if rv is None or cache is None:
            return {}, 0, 0.0
        # The prefetch worker runs off the main thread; bind it to our GPU so
        # comm creation + recvs target the right device (thread-local).
        bind_thread_device(device)

        t0 = get_trace_time()
        # Declared OUTSIDE the try so the finally can always read it.
        recvs: List[Tuple[Any, dict, int, torch.Tensor]] = []
        # Pins taken during phase 1 must be released on EVERY exit path,
        # including the early returns and any raise below.
        try:
            # Phase 1: rendezvous (control plane) — sequential Redis round-trips.
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
            receiver_rank = self._receiver_rank
            warm = self._warm_pairs
            batch_warming = any(
                _pair_key(ref, receiver_rank) not in warm for _i, ref, _c, _b in recvs
            )
            recv_timeout_ms = int(
                (self._first_transfer_timeout if batch_warming else self._recv_timeout)
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
            recv_lock = self._recv_lock
            if recv_lock is None:  # bare test harness that skipped setup
                recv_lock = self._recv_lock = threading.Lock()
            with recv_lock:
                stream = self._streams.acquire() if self._streams else None
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
                            "[NCCLTransportStrategy] recv failed for %s: %s",
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
                        "[NCCLTransportStrategy] recv completion sync failed "
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
        finally:
            # Release the eviction pins taken by _rendezvous_one.  The comm
            # was leased from build through recv completion (the
            # synchronize above), so unpinning earlier would reopen the
            # mid-collective eviction window.
            receiver_rank = self._receiver_rank
            for _i, _ref, _c, _b in recvs:
                cache.unpin(_pair_key(_ref, receiver_rank))

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
                "[NCCLTransportStrategy] quarantine %s raised %s; continuing",
                health_key,
                type(exc).__name__,
            )
        if pair is not None:
            self._warm_pairs.discard(pair)

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
        receiver_rank = self._receiver_rank
        warm = self._warm_pairs
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
        rv = self._rendezvous
        cache = self._comm_cache
        transfer_id = ref["transfer_id"]
        sender_rank = ref["sender_rank"]
        sender_replica = ref["sender_replica"]
        receiver_rank = self._receiver_rank
        receiver_replica = self._receiver_replica
        # Pair key + health key are keyed on the globally-unique sender
        # replica identity so distinct rollout replicas never collide.
        pair = _pair_key(ref, receiver_rank)
        health_key = (sender_replica, sender_rank)
        # "warming" until this pair completes its first successful transfer.
        warming = pair not in self._warm_pairs

        if cache.is_quarantined(health_key):
            logger.debug(
                "[NCCLTransportStrategy] endpoint %s quarantined; skipping %s",
                health_key,
                transfer_id,
            )
            return None

        request_channel = build_sender_request_channel(self._prefix, ref)

        for attempt in range(1, self._max_attempts + 1):
            need_uid = pair not in cache
            # Until the pair is warm, give it the long cold-start budget so the
            # comm-init storm doesn't cancel a healthy-but-slow transfer; a warm
            # pair uses the tight steady-state budget.
            timeout = self._first_transfer_timeout if warming else self._recv_timeout
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
                    # PIN the comm: the caller holds this comm_idx until its
                    # recv completes, so an unpinned entry could be evicted +
                    # aborted mid-collective by a concurrent build for another
                    # pair.  _fetch_all releases it in a finally.
                    comm_idx = cache.get_or_create(
                        pair,
                        uid_chars=result.uid_chars or [],
                        local_rank=RECEIVER_LOCAL_RANK,
                        pin=True,
                    )
                except Exception as e:
                    logger.warning(
                        "[NCCLTransportStrategy] comm build failed for %s: %s%s",
                        transfer_id,
                        e,
                        "; retry next round (warming)" if warming else "; quarantining",
                    )
                    if not warming:
                        self._quarantine_endpoint(cache, health_key, pair)
                    return None
                try:
                    recv_buf = _alloc_recv_buffer(ref["schema"], self._device)
                except Exception:
                    # Never returned to the caller -> nothing will unpin it here.
                    cache.unpin(pair)
                    raise
                return comm_idx, recv_buf
            if result.status is TransferStatus.MISSING:
                # Producer recycled the buffer — non-retryable, drop now.
                logger.debug(
                    "[NCCLTransportStrategy] transfer %s missing (recycled)",
                    transfer_id,
                )
                return None
            if result.status is TransferStatus.NEED_UID:
                # The sender evicted its side of the comm; our cached comm is
                # now half-open.  Drop it and retry -- the next attempt has
                # need_uid=True and mints a fresh uid so BOTH sides rebuild.
                logger.debug(
                    "[NCCLTransportStrategy] transfer %s: sender needs a fresh "
                    "uid; dropping stale comm and renegotiating",
                    transfer_id,
                )
                cache.abort(pair)
                continue
            # CANCELLED (timeout).
            if attempt == self._max_attempts:
                if not warming:
                    # A WARM pair (has transferred before) that stops
                    # rendezvousing -> the sender likely died mid-run;
                    # quarantine so we stop hammering it (and abort the comm).
                    logger.warning(
                        "[NCCLTransportStrategy] transfer %s cancelled after %d "
                        "attempts on a warm comm; quarantining %s",
                        transfer_id,
                        self._max_attempts,
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
                        "[NCCLTransportStrategy] transfer %s still warming (%d "
                        "attempts); keeping comm, retry next round",
                        transfer_id,
                        self._max_attempts,
                    )
        return None

    # ------------------------------------------------------------------
    # Cache-key helpers (parity with UCXX; used by tests + _fetch_batch).
    # ------------------------------------------------------------------

    @staticmethod
    def _ref_cache_key(rollout_output: Any) -> str:
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
            return NCCLTransportStrategy._ref_cache_key(ro)
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


def compose_nccl_transport(
    packer: Any,
    *,
    device: Any,
    redis_client: Any,
    config: Any = None,
    prefetch_timeout: float = 30.0,
    max_attempts: int = 2,
    recv_timeout: float = 5.0,
    first_transfer_timeout: float = 30.0,
) -> None:
    """Attach a fresh NCCL strategy to ``packer`` and start its prefetch worker.

    The one place that wiring lives, so the mixin and the composed attach path
    cannot drift: ``NCCLDataPackerMixin._setup_nccl_data_packer`` is a call to
    this, and ``NcclPayloadTransport`` uses it for packers that compose rather
    than subclass.

    ``packer`` need only provide ``set_transport_strategy`` and
    ``_setup_prefetch`` -- i.e. be a ``PrefetchDataPackerMixin``; no NCCL
    ancestry is required.
    """
    strategy = NCCLTransportStrategy()
    strategy.setup(
        device=device,
        redis_client=redis_client,
        config=config,
        max_attempts=max_attempts,
        recv_timeout=recv_timeout,
        first_transfer_timeout=first_transfer_timeout,
        # Set by _attach_payload_transport before setup to disambiguate policy
        # replicas sharing a receiver_rank; absent on standalone/test packers.
        receiver_replica=getattr(packer, "_nccl_dp_receiver_replica", None),
    )
    packer.set_transport_strategy(strategy)
    packer._setup_prefetch(
        prefetch_timeout=prefetch_timeout,
        thread_name="NCCLDataPackerPrefetch",
    )
