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

import os
import socket
import threading
import time
from dataclasses import dataclass
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
from cosmos_rl.utils.payload_transport.nccl.header import (
    HEADER_NBYTES,
    HEADER_VERSION,
    PayloadHeaderMismatch,
    verify_header,
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
from cosmos_rl.utils.payload_transport.receive_memory import (
    ReceiveBudget,
    ReceiveMemoryError,
    ReceivedBatch,
    storage_bytes,
)
from cosmos_rl.utils.trace import get_trace_time
from cosmos_rl.utils.transport_failure import TransportUnusableError, TransportDeadline


@dataclass
class _PreparedReceive:
    comm_idx: int
    buffer: torch.Tensor
    operation: TransportDeadline
    response_key: Optional[str]

    def __iter__(self):
        return iter((self.comm_idx, self.buffer))

    def __getitem__(self, index):
        return (self.comm_idx, self.buffer)[index]


_LOG_INTERVAL = 50

# Terminal ownership, not a retry cache. Native failure/abort is not proof that
# a CUDA stream stopped touching these buffers. Retain until process exit, even
# if the caller catches the error or drops the strategy during teardown.
_FAILED_RECEIVE_OWNERS = []

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
    _receive_budget = None
    _bounded_fetch_lock = None
    _decode_account = None
    _bounded_rejections = None

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
        custom = getattr(config, "custom", None) or {}
        limit = custom.get("nccl_receive_budget_bytes", 0)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("nccl_receive_budget_bytes must be a nonnegative integer")
        self._receive_budget = (
            ReceiveBudget(
                limit, float(custom.get("nccl_receive_admission_timeout", 30.0))
            )
            if limit
            else None
        )
        self._bounded_fetch_lock = threading.Lock()
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
        # this worker's ``replica_name`` before setup.  The fallback below is
        # for standalone/test setups that never went through it.
        #
        # It MUST be globally unique, not just rank-derived.  Every policy
        # replica in a single-GPU deployment has ``receiver_rank == 0``, so a
        # bare ``recv0`` makes N replicas indistinguishable to the producer --
        # which keys its comm cache and its per-pair unique-ID on exactly this
        # string.  They then share one communicator, and a recv posted by one
        # replica takes a send meant for another: the payload arrives intact
        # but belongs to a different transfer.  Host and PID make it unique
        # without needing any cluster-wide coordination.
        if not self._receiver_replica:
            self._receiver_replica = (
                f"recv{self._receiver_rank}-{socket.gethostname()}-{os.getpid()}"
            )
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

        # ``payload_header`` is the on-wire framing version; see the producer's
        # matching line.  A rollout and a policy reporting different versions
        # cannot exchange payloads (the header check rejects every transfer).
        logger.info(
            "[NCCLTransportStrategy] Initialised: device=%s, rank=%d, "
            "max_attempts=%d, recv_timeout=%ss, payload_header=v%d",
            device,
            self._receiver_rank,
            self._max_attempts,
            recv_timeout,
            HEADER_VERSION,
        )

    def before_join(self) -> None:
        """Abort every cached comm so an in-flight ``nccl_recv`` returns.

        Used as ``shutdown_prefetch``'s ``before_join`` hook: the prefetch
        worker only checks the shutdown event *between* batches, so a recv
        parked on a departed peer would otherwise hold the join for the full
        first-transfer budget rather than the join timeout.
        """
        if self._receive_budget is not None:
            self._receive_budget.close()
        if self._comm_cache is not None:
            try:
                self._comm_cache.close()
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
        rejected = set()
        for idx, ro in tasks:
            ref = _parse_ref(ro, default_schema=self._schema)
            if ref is not None and _has_schema(ref):
                refs.append((idx, ref))
            else:
                # No rendezvous/native work was issued for this reference.
                rejected.add(idx)

        results, total_bytes, transfer_ms = self._fetch_all(refs)
        if self._receive_budget is not None and not isinstance(results, ReceivedBatch):
            # An all-rejected batch has no tensor reservation, but still owns
            # explicit outcomes until its consumer releases the batch.
            self._receive_budget.reserve(0)
            results = ReceivedBatch(results, self._receive_budget, 0)

        cache_results: Dict[str, Any] = {}
        for idx, gpu_data in results.items():
            key = _cache_key_from_task(tasks, idx)
            cache_results[key] = gpu_data

        if isinstance(results, ReceivedBatch):
            results.rejected_keys = {
                _cache_key_from_task(tasks, idx)
                for idx in results.rejected_keys | rejected
            }
            results.clear()
            results.update(cache_results)
            cache_results = results

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

        Missing/rejected data may fall back. Terminal native failure must retain
        its meaning in both synchronous and prefetched execution.
        """
        if self._receive_budget is not None:
            raise ReceiveMemoryError(
                "Bounded NCCL reception requires start_prefetch/wait_prefetch; "
                "cache-miss refetch is disabled to preserve consumer leases."
            )
        try:
            # Parse inside the guard too: a malformed dict ref (e.g. a corrupt
            # ``_schema``) raises from deserialize_schema, and this path has no
            # containment above it in the base get_policy_input.
            ref = _parse_ref(rollout_output, default_schema=self._schema)
            if ref is None or not _has_schema(ref):
                return None
            results, _, _ = self._fetch_all([(0, ref)])
        except TransportUnusableError:
            raise
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

    def receive_memory_stats(self) -> dict:
        return self._receive_budget.snapshot() if self._receive_budget else {}

    def _fetch_all(self, refs: List[Tuple[Any, dict]]) -> Tuple[dict, int, float]:
        if self._receive_budget is None or not refs:
            return self._fetch_unbounded(refs)
        # Serial bounded fetches avoid racing raw attribution and duplicate
        # reservations on the shared transfer stream. Admission precedes any
        # rendezvous, so a sender never waits while we wait for consumer release.
        with self._bounded_fetch_lock:
            return self._fetch_bounded(refs)

    def _fetch_bounded(self, refs):
        budget = self._receive_budget
        decoded_size = 0
        wire_sizes = []
        for _, ref in refs:
            _, raw_size = schema_layout(ref["schema"])
            decoded_size += sum(spec.nbytes for spec in ref["schema"])
            wire_sizes.append(HEADER_NBYTES + raw_size)
        workspace = max(wire_sizes)
        required = decoded_size + workspace
        budget.reserve(required)
        results = {}
        part = {}
        total_bytes = 0
        t0 = get_trace_time()
        failure = None
        attributed = 0
        window_storage = []

        def account_decode(tensor):
            nonlocal attributed
            window_storage.append(tensor)
            nbytes = tensor.untyped_storage().nbytes()
            attributed += nbytes
            budget.attribute(decoded=nbytes)

        self._decode_account = account_decode
        self._bounded_window_storage = window_storage
        rejected_transfers = set()
        self._bounded_rejections = rejected_transfers
        try:
            # Reserve all decoded storage plus at least one receive up front.
            # Additional workspace uses currently free bytes, never a blocking
            # reservation that could deadlock behind our own admitted batch.
            extra = budget.reserve_available(sum(wire_sizes) - workspace)
            workspace += extra
            required += extra
            cursor = 0
            while cursor < len(refs):
                if budget.closed:
                    raise ReceiveMemoryError("NCCL receive cancelled by shutdown")
                end, wire_bytes = cursor, 0
                while end < len(refs) and wire_bytes + wire_sizes[end] <= workspace:
                    wire_bytes += wire_sizes[end]
                    end += 1
                # The existing helper posts each matching receive immediately
                # after rendezvous, then observes completion for the window.
                # Never group independent communicators with ncclGroupStart.
                window = refs[cursor:end]
                ref = window[0][1]
                part, nbytes, _ = self._fetch_unbounded(window)
                if torch.cuda.is_available():
                    torch.cuda.current_stream(self._device).synchronize()
                results.update(part)
                part.clear()
                window_storage.clear()
                current = storage_bytes(results.values())
                budget.attribute(decoded=current - attributed)
                attributed = current
                total_bytes += nbytes
                budget.attribute(raw=-budget.raw)
                cursor = end
            actual = storage_bytes(results.values())
            if actual > required:
                raise ReceiveMemoryError("Decoded storage exceeded schema reservation")
        except TransportUnusableError:
            budget.close()
            _FAILED_RECEIVE_OWNERS.append(
                (budget, results, part, window_storage, self._comm_cache, self._streams)
            )
            raise
        except Exception as exc:
            # Do not carry a traceback holding raw buffers/decoded tensors into
            # the prefetch error queue. The recovery path already handles comms.
            failure = f"{type(exc).__name__}: {exc}"
            logger.error(
                "[NCCLTransportStrategy] bounded receive failed: transfer=%s "
                "schema=%s error=%s; %s",
                ref["transfer_id"],
                ref["schema"],
                failure,
                budget.snapshot(),
            )
        except BaseException:
            # Cancellation can interrupt decoding too. Do not run native
            # cleanup or discard operands without completion evidence.
            budget.close()
            _FAILED_RECEIVE_OWNERS.append(
                (budget, results, part, window_storage, self._comm_cache, self._streams)
            )
            raise
        finally:
            self._decode_account = None
            self._bounded_window_storage = None
            self._bounded_rejections = None
        if failure is not None:
            # Outstanding decode copies must finish before their charge goes away.
            if torch.cuda.is_available():
                try:
                    torch.cuda.current_stream(self._device).synchronize()
                except Exception as exc:
                    budget.close()
                    _FAILED_RECEIVE_OWNERS.append(
                        (budget, results, part, window_storage, self._comm_cache)
                    )
                    raise TransportUnusableError(
                        "CUDA completion failed; receive budget is closed and its "
                        "reservation retained. Restart the receiver."
                    ) from exc
            results.clear()
            part.clear()
            window_storage.clear()
            budget.attribute(raw=-budget.raw, decoded=-attributed)
            budget.release(required)
            raise ReceiveMemoryError(failure)
        budget.release(required - actual)
        leased = ReceivedBatch(
            results,
            budget,
            actual,
            rejected_keys={
                idx for idx, ref in refs if ref["transfer_id"] in rejected_transfers
            },
        )
        logger.info("[NCCLTransportStrategy] receive memory: %s", budget.snapshot())
        return leased, total_bytes, get_trace_time() - t0

    def _fetch_unbounded(self, refs: List[Tuple[Any, dict]]) -> Tuple[dict, int, float]:
        """Rendezvous + ``nccl_recv``, interleaved per ref.

        Returns ``(results_by_idx, total_bytes, transfer_ms)``.  Each ref is
        negotiated over Redis and then IMMEDIATELY has its matching recv
        enqueued, before the next ref is negotiated -- a producer must never
        be left holding an accepted send whose recv this consumer has not
        posted yet (see the interleaving note below).  Each recv is issued as
        a STANDALONE ``nccl_recv`` on its own 2-rank pair comm (deliberately
        NOT wrapped in a cross-communicator ``ncclGroupStart/End``, which
        would couple independent producers into one completion unit -- the
        N_POLICY>=2 wedge).  Completion is synchronized once, after the whole
        batch is enqueued, so transfers still overlap.

        A per-ref ``max_attempts`` fresh-call retry wraps the rendezvous; a
        ref that still fails to resolve is dropped and re-attempted on the
        next prefetch round (there is no in-batch multi-round layer above
        this).
        """
        from cosmos_rl.utils import pynccl

        rv = self._rendezvous
        cache = self._comm_cache
        device = self._device
        if rv is None or cache is None or not refs:
            return {}, 0, 0.0
        # The prefetch worker runs off the main thread; bind it to our GPU so
        # comm creation + recvs target the right device (thread-local).
        bind_thread_device(device)

        t0 = get_trace_time()
        # Declared OUTSIDE the try so the finally can always read it.
        recvs: List[Tuple[Any, dict, int, torch.Tensor]] = []
        native_pending = False
        retain_pins = False
        stream = None
        operations = []
        # Pins taken during phase 1 must be released on EVERY exit path,
        # including the early returns and any raise below.
        try:
            # A batch is "warming" until every pair in it has transferred at
            # least once; give its recvs the long cold-start budget so a slow
            # (storm-contended) send isn't cancelled -> comm torn down ->
            # rebuilt into the same storm.  This is decided up front, from the
            # INPUT refs, because the first recv is now enqueued before the rest
            # of the batch has rendezvoused.  Deciding it from refs rather than
            # from the resolved set is conservative in the safe direction: a ref
            # that never resolves can only hold the batch on the LONGER
            # cold-start budget, never select a too-short one.
            receiver_rank = self._receiver_rank
            warm = self._warm_pairs
            batch_warming = any(
                _pair_key(ref, receiver_rank) not in warm for _idx, ref in refs
            )
            recv_timeout_ms = int(
                (self._first_transfer_timeout if batch_warming else self._recv_timeout)
                * 1000
            )

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
            #
            # The rendezvous round-trips now run INSIDE this lock, because each recv
            # is enqueued as soon as its own rendezvous returns (see below) and the
            # lock therefore spans them.  It has to: transfer streams are handed out
            # round-robin rather than leased, so a concurrent caller can share our
            # stream -- and the completion event recorded at the end of this block
            # would then also cover ITS recvs, leaving our lock-free synchronize()
            # waiting on a peer we never negotiated with (an unbounded wait: the
            # stream sync takes no timeout).  Holding the lock across the whole
            # enqueue sequence keeps `done` covering exactly our own recvs.
            recv_lock = self._recv_lock
            if recv_lock is None:  # bare test harness that skipped setup
                recv_lock = self._recv_lock = threading.Lock()
            with recv_lock:
                stream = self._streams.acquire() if self._streams else None
                # Rendezvous (control plane) and recv enqueue (data plane) are
                # INTERLEAVED per ref: every accepted transfer gets its matching
                # recv posted BEFORE the next ref is negotiated.
                #
                # Negotiating the whole batch first (the previous structure)
                # deadlocks whenever one producer accepts more refs in this batch
                # than it has sender threads: those threads block in nccl_send
                # waiting for recvs this consumer will not post until every
                # remaining rendezvous has returned -- including the ones queued
                # behind those very sends.  That is a circular wait, not slowness,
                # and a bigger sender pool only moves the batch size that trips it.
                # Posting recv(A) before negotiating B removes the cycle outright.
                #
                # Each recv is issued STANDALONE on its own 2-rank pair comm,
                # deliberately NOT wrapped in a cross-communicator
                # ncclGroupStart/End: each comm carries a single send/recv, so
                # grouping adds no overlap and only fuses independent producers
                # into one completion unit whose native ncclGroupEnd blocks -- with
                # no pynccl watchdog, since run_task arms its deadline only after
                # the native call returns -- if any one producer has not yet posted
                # its send.  That coupling was the N_POLICY>=2 wedge.  Ungrouped, a
                # slow producer delays only its own recv.
                for idx, ref in refs:
                    prepared = self._rendezvous_one(ref, pynccl)
                    if prepared is None:
                        continue
                    comm_idx, recv_buf = prepared
                    operation = getattr(prepared, "operation", None)
                    if operation is None:
                        operation = TransportDeadline(
                            recv_timeout_ms / 1000, "NCCL receive"
                        )
                    response_key = getattr(prepared, "response_key", None)
                    operations.append((operation, response_key, None))
                    # Record the pin BEFORE attempting the recv so the finally
                    # below unpins this comm even if the enqueue raises.
                    recvs.append((idx, ref, comm_idx, recv_buf))
                    if self._receive_budget is not None:
                        self._bounded_window_storage.append(recv_buf)
                    # Mark before issue: a raising native call may already have
                    # enqueued work or partially written the destination.
                    native_pending = True
                    try:
                        pynccl.nccl_recv(
                            recv_buf,
                            SENDER_LOCAL_RANK,  # peer in the 2-rank comm is the sender
                            comm_idx,
                            stream=stream,
                            timeout_ms=operation.remaining_ms(),
                        )
                    except Exception as exc:
                        raise TransportUnusableError(
                            "NCCL receive enqueue failed; native completion unknown"
                        ) from exc
                    operations[-1] = (operation, response_key, record_event(stream))
                    posted.append((idx, ref, comm_idx, recv_buf))

                if not posted:
                    return {}, 0, get_trace_time() - t0

                # Recv-complete event gates downstream training consumption.
                done = record_event(stream)

            for operation, response_key, event in operations:
                operation.wait_event(
                    event,
                    check_peer=(
                        (lambda key=response_key: rv.check_failure(key))
                        if response_key
                        else None
                    ),
                )
            wait_event(None, done)  # current (compute) stream waits before reads

            native_pending = False

            # A pair aborted DURING this batch takes its already-posted recvs
            # down with it.  Their own enqueue succeeded, so nothing raised for
            # them -- but the communicator they were posted on is gone, and NCCL
            # will never write their buffers.  Unpacking one reads uninitialised
            # memory, which the header check then reports as a desynced stream:
            # an abort we performed ourselves, blamed on the peer.
            #
            # Detect it by identity rather than by tracking abort sites: the
            # comm is PINNED for the whole batch, so eviction cannot move it and
            # the cached comm_idx changes only if someone explicitly aborted the
            # pair (recv enqueue failure, recv-buffer alloc failure, quarantine
            # from any path).  Anything that no longer maps to the comm_idx we
            # posted on is dead.
            for entry in posted:
                if cache.get(_pair_key(entry[1], receiver_rank)) != entry[2]:
                    raise TransportUnusableError(
                        "Accepted receive communicator was invalidated: "
                        + entry[1]["transfer_id"]
                    )

            for idx, ref, _comm_idx, recv_buf in posted:
                try:
                    if self._decode_account is None:
                        gpu_data = _verify_and_unpack(recv_buf, ref, device)
                    else:
                        gpu_data = _verify_and_unpack(
                            recv_buf, ref, device, on_allocate=self._decode_account
                        )
                except PayloadHeaderMismatch as exc:
                    # Completion alone does not make an incorrectly paired
                    # stream reusable. Rebuilding one half cannot prove that
                    # queued peer work is settled; never replay as a cache miss.
                    raise TransportUnusableError(
                        f"Accepted payload identity mismatch: {exc}"
                    ) from exc
                results[idx] = gpu_data
                total_bytes += recv_buf.numel() * recv_buf.element_size()
                # First successful transfer -> this pair is warm (tight timeouts +
                # normal quarantine from here on).
                warm.add(_pair_key(ref, receiver_rank))

            decoded = record_event(None)
            for operation, _key, _event in operations:
                operation.wait_event(decoded)
            return results, total_bytes, get_trace_time() - t0
        except BaseException as exc:
            if native_pending:
                retain_pins = True
                if self._receive_budget is not None:
                    self._receive_budget.close()
                _FAILED_RECEIVE_OWNERS.append((recvs, cache, stream))
                raise TransportUnusableError(
                    "Receive interrupted before native completion; "
                    "storage, communicator pins and reservation retained. "
                    "Restart the receiver."
                ) from exc
            raise
        finally:
            # Release the eviction pins taken by _rendezvous_one.  The comm
            # was leased from build through recv completion (the
            # synchronize above), so unpinning earlier would reopen the
            # mid-collective eviction window.
            receiver_rank = self._receiver_rank
            for operation, response_key, _event in operations:
                if native_pending and response_key:
                    try:
                        rv.respond(resp_key=response_key, status=TransferStatus.FAILED)
                    except Exception:
                        logger.exception("Failed to notify payload sender")
                # Retained operands survive exception propagation. The caller
                # must take the fatal path, not reuse this transport.
                operation.close()
            if not retain_pins:
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
            if result.accepted:
                remaining = (
                    result.deadline - time.monotonic()
                    if result.deadline is not None
                    else timeout
                )
                operation = TransportDeadline(remaining, f"receive {transfer_id}")
                comm_idx = None
                try:
                    if result.response_key:
                        rv.watch_operation(result.response_key, operation)
                    # PIN the comm: the caller holds this comm_idx until its
                    # recv completes, so an unpinned entry could be evicted +
                    # aborted mid-collective by a concurrent build for another
                    # pair.  _fetch_all releases it in a finally.
                    comm_idx = cache.get_or_create(
                        pair,
                        uid_chars=result.uid_chars or [],
                        local_rank=RECEIVER_LOCAL_RANK,
                        pin=True,
                        deadline=operation,
                    )
                    recv_buf = _alloc_recv_buffer(ref["schema"], self._device)
                    if self._receive_budget is not None:
                        self._receive_budget.attribute(
                            raw=recv_buf.untyped_storage().nbytes()
                        )
                except BaseException as e:
                    # Never returned to the caller -> nothing will unpin it here.
                    if comm_idx is not None:
                        cache.unpin(pair)
                    if result.response_key:
                        try:
                            rv.respond(
                                resp_key=result.response_key,
                                status=TransferStatus.FAILED,
                            )
                        except Exception:
                            logger.exception("Failed to notify accepted sender")
                    operation.close()
                    raise TransportUnusableError(
                        f"Accepted transfer {transfer_id} could not prepare its receive"
                    ) from e
                return _PreparedReceive(
                    comm_idx, recv_buf, operation, result.response_key
                )
            if result.status is TransferStatus.MISSING:
                # Producer recycled the buffer — non-retryable, drop now.
                logger.debug(
                    "[NCCLTransportStrategy] transfer %s missing (recycled)",
                    transfer_id,
                )
                self._mark_known_rejection(ref)
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
            if result.late_accept:
                # The sender may owe an unmatched send. A local abort is not
                # proof that peer work is settled; do not retry this attempt.
                raise TransportUnusableError(
                    "Sender accepted after the receiver deadline"
                )
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

    def _mark_known_rejection(self, ref):
        if self._bounded_rejections is not None:
            self._bounded_rejections.add(ref["transfer_id"])


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
        raw_schema = rollout_output.get("_schema")
        # Dict metadata is what a producer with a PER-PAYLOAD schema emits, and
        # it always carries ``_schema``.  Do NOT fall back to the static default
        # when it is absent: that would unpack the payload at another layout's
        # offsets and hand the caller decoded garbage -- the same class of
        # failure as a mispaired buffer, minus any way to notice.  Leaving the
        # schema None makes the fetch paths drop the reference instead, so the
        # episode degrades to the Redis path.  (The bare ``nccl:<id>`` string
        # form above has no metadata channel at all and keeps the default.)
        schema = deserialize_schema(raw_schema) if raw_schema else None
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


def _has_schema(ref: dict) -> bool:
    """True if ``ref`` can be decoded; logs (and rejects) it if it cannot.

    A dict reference with no ``_schema`` reaches here only when the producer
    lost that metadata.  There is no safe way to decode the payload, so the
    fetch paths skip it and the episode falls back to the Redis transport.
    """
    if ref.get("schema") is not None:
        return True
    logger.error(
        "[NCCLTransportStrategy] NCCL reference %s carries no schema; refusing "
        "to decode its payload (episode falls back to the Redis path)",
        ref.get("transfer_id", "<unknown>"),
    )
    return False


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
    """Allocate a flat uint8 GPU buffer sized for ``schema`` plus its header.

    The producer prefixes :data:`HEADER_NBYTES` of self-describing header to
    every payload, so the wire size is header + entry size on both ends.
    """
    if schema is None:
        raise ValueError("cannot allocate NCCL recv buffer without a schema")
    _, entry_size = schema_layout(schema)
    return torch.empty(HEADER_NBYTES + entry_size, dtype=torch.uint8, device=device)


def _verify_and_unpack(
    recv_buf: torch.Tensor, ref: dict, device: Any, *, on_allocate=None
) -> dict:
    """Check the payload header, then slice the payload region by schema.

    The header is the only thing that ties the bytes in ``recv_buf`` to the
    transfer they were requested for: a 2-rank comm carries no tags, so a
    receiver that has fallen out of step with its sender would otherwise unpack
    a foreign payload with its own schema and return plausible-looking garbage.

    Raises:
        PayloadHeaderMismatch: if the buffer does not belong to ``ref``.  The
            caller must terminate the transport -- once the stream is off by one
            subsequent transfers cannot be trusted.
    """
    schema = ref.get("schema")
    if schema is None:
        return {}
    _, entry_size = schema_layout(schema)
    header = bytes(recv_buf[:HEADER_NBYTES].cpu().numpy())
    verify_header(header, transfer_id=ref["transfer_id"], payload_nbytes=entry_size)
    if on_allocate is None:
        return _unpack(recv_buf[HEADER_NBYTES:], schema, device)
    return _unpack(recv_buf[HEADER_NBYTES:], schema, device, on_allocate=on_allocate)


def _unpack(
    recv_buf: torch.Tensor, schema: Optional[list], device: Any, *, on_allocate=None
) -> dict:
    """Slice a payload region (header already stripped) into schema tensors."""
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
        if on_allocate is not None:
            on_allocate(raw)
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
    packer.set_transport_strategy(strategy)
    try:
        strategy.setup(
            device=device,
            redis_client=redis_client,
            config=config,
            max_attempts=max_attempts,
            recv_timeout=recv_timeout,
            first_transfer_timeout=first_transfer_timeout,
            receiver_replica=getattr(packer, "_nccl_dp_receiver_replica", None),
        )
        packer._setup_prefetch(
            prefetch_timeout=prefetch_timeout,
            thread_name="NCCLDataPackerPrefetch",
        )
    except BaseException:
        try:
            packer.close_transport()
        except Exception as cleanup_error:
            logger.error("NCCL attachment rollback failed: %s", cleanup_error)
        raise
