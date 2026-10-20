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
    build_layout_key,
    build_sender_request_channel,
    parse_transfer_rollout_idx,
)
from cosmos_rl.utils.payload_transport.nccl.rendezvous import (
    NcclRendezvous,
    TransferStatus,
)
from cosmos_rl.utils.payload_transport.layout import (
    LayoutCache,
    TensorLayout,
    WIRE_VERSION,
    deserialize_layout_descriptor,
    validate_layout_id,
)
from cosmos_rl.utils.trajectory import (
    VARLEN_FIELDS,
    truncate_trajectory,
    build_trajectory_schema,
)
from cosmos_rl.utils.payload_transport.nccl.streams import (
    bind_thread_device,
    get_transfer_stream_pool,
    record_event,
    wait_event,
)
from cosmos_rl.utils.payload_transport.strategy import PayloadTransportStrategy
from cosmos_rl.utils.payload_transport.receive_memory import (
    ReceiveBufferLease,
    ReceiveBufferPool,
    ReceiveBudget,
    ReceiveMemoryError,
    ReceivedBatch,
)
from cosmos_rl.utils.trace import get_trace_time
from cosmos_rl.utils.transport_failure import TransportUnusableError, TransportDeadline


@dataclass
class _PreparedReceive:
    comm_idx: int
    buffer: torch.Tensor
    lease: ReceiveBufferLease
    operation: TransportDeadline
    response_key: Optional[str]

    def __iter__(self):
        return iter((self.comm_idx, self.buffer))

    def __getitem__(self, index):
        return (self.comm_idx, self.buffer)[index]


class _DetachedReceiveLease:
    """Test/embedding adapter for a caller-supplied receive tensor."""

    def __init__(self, tensor):
        self.tensor = tensor

    def release(self):
        pass

    def quarantine(self):
        pass

    def bind_views(self, layout, *, offset=0, reset_fields=()):
        return layout.views(self.tensor.narrow(0, offset, layout.payload_nbytes))


_LOG_INTERVAL = 50

# Terminal ownership, not a retry cache. Native failure/abort is not proof that
# a CUDA stream stopped touching these buffers. Retain until process exit, even
# if the caller catches the error or drops the strategy during teardown.
_FAILED_RECEIVE_OWNERS = []


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
    _default_layout: Optional[TensorLayout] = None
    _layout_cache: Optional[LayoutCache] = None
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
    _receive_pool = None

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
        self._default_layout = TensorLayout.compile(self._schema)
        self._layout_cache = LayoutCache(
            int(custom.get("payload_layout_cache_capacity", 64))
        )
        self._layout_cache.register(self._default_layout)
        self._receive_pool = ReceiveBufferPool(
            max_cached_per_layout=int(
                custom.get("payload_receive_pool_cached_slots", 64)
            ),
            max_cached_buffers=int(custom.get("payload_receive_pool_cached_slots", 64)),
            max_cached_bytes=limit or None,
        )

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
            "max_attempts=%d, recv_timeout=%ss, payload_header=%d",
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
        if self._receive_pool is not None:
            self._receive_pool.close()
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
        try:
            ref = _parse_ref(rollout_output)
        except (TypeError, ValueError) as error:
            raise TransportUnusableError(
                f"NCCL payload reference is not trustworthy: {error}"
            ) from error
        return (
            _identity_cache_key(ref)
            if ref is not None
            else self._ref_cache_key(rollout_output)
        )

    def fetch_batch(self, tasks: List[Any]) -> Dict[str, Any]:
        """Resolve a batch of NCCL references via rendezvous + standalone per-pair recvs."""
        fetch_start = get_trace_time()
        refs: List[Tuple[Any, dict]] = []
        rejected = set()
        keys_by_idx = {}
        for idx, ro in tasks:
            try:
                ref = _parse_ref(ro)
                if ref is not None and _has_layout(ref):
                    keys_by_idx[idx] = _identity_cache_key(ref)
                    ref["layout"] = self._resolve_layout(ref)
                    refs.append((idx, ref))
                else:
                    # No rendezvous/native work was issued for this reference.
                    keys_by_idx[idx] = self._ref_cache_key(ro)
                    rejected.add(idx)
            except (KeyError, TypeError, ValueError) as error:
                raise TransportUnusableError(
                    f"NCCL payload reference is not trustworthy: {error}"
                ) from error

        results, total_bytes, transfer_ms = self._fetch_all(refs)
        received_count = len(results)
        if self._receive_budget is not None and not isinstance(results, ReceivedBatch):
            # An all-rejected batch has no tensor reservation, but still owns
            # explicit outcomes until its consumer releases the batch.
            self._receive_budget.reserve(0)
            results = ReceivedBatch(results, self._receive_budget, 0)

        cache_results: Dict[str, Any] = {}
        for idx, gpu_data in results.items():
            key = keys_by_idx.get(idx, str(idx))
            cache_results[key] = gpu_data

        if isinstance(results, ReceivedBatch):
            results.rejected_keys = {
                keys_by_idx.get(idx, str(idx))
                for idx in results.rejected_keys | rejected
            }
            results.clear()
            results.update(cache_results)
            cache_results = results

        self._last_bytes = total_bytes
        self._last_count = received_count
        self._record_fetch(received_count, total_bytes, get_trace_time() - fetch_start)
        if results:
            logger.debug(
                "[Trace] thread=nccl_prefetch op=nccl_fetch transfer_ms=%.1f "
                "count=%d bytes=%d",
                transfer_ms,
                received_count,
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
            # Parse inside the guard too: a malformed layout descriptor raises,
            # and this path has no containment above it in get_policy_input.
            ref = _parse_ref(rollout_output)
            if ref is None or not _has_layout(ref):
                return None
            ref["layout"] = self._resolve_layout(ref)
            results, _, _ = self._fetch_all([(0, ref)])
        except TransportUnusableError:
            raise
        except (KeyError, ValueError, TypeError) as e:
            raise TransportUnusableError(
                f"NCCL payload reference is not trustworthy: {e}"
            ) from e
        except Exception as e:
            logger.warning("[NCCLTransportStrategy] Sync fallback failed: %s", e)
            return None
        payload = results.get(0)
        if payload is None or not isinstance(results, ReceivedBatch):
            return payload
        try:
            return {name: tensor.clone() for name, tensor in payload.items()}
        finally:
            results.release()

    def _resolve_layout(self, ref: dict) -> TensorLayout:
        layout_id = validate_layout_id(ref.get("layout_id"))
        try:
            return self._layout_cache.resolve(layout_id, None)
        except KeyError:
            sender_replica = ref.get("sender_replica")
            sender_incarnation = ref.get("sender_incarnation")
            if not isinstance(sender_replica, str) or not sender_replica:
                raise ValueError("NCCL payload reference has no sender identity")
            if not isinstance(sender_incarnation, str) or not sender_incarnation:
                raise ValueError("NCCL payload reference has no sender incarnation")
            key = build_layout_key(
                self._prefix,
                sender_replica,
                sender_incarnation,
                layout_id,
            )
            if self._redis is None:
                raise ValueError("NCCL tensor layout registry is not initialized")
            descriptor_text = self._redis.get(key)
            if descriptor_text is None:
                raise KeyError(f"unknown tensor layout {layout_id}")
            descriptor = deserialize_layout_descriptor(descriptor_text)
            return self._layout_cache.resolve(layout_id, descriptor)

    def on_prefetch_complete(
        self, batch_id: int, n_results: int, fetch_ms: float, step: int
    ) -> None:
        """Keep the legacy hook; fetch completion already accounted for I/O."""

    def _record_fetch(self, count: int, nbytes: int, fetch_ms: float) -> None:
        # The worker owns accounting, using this fetch's local figures. Cache
        # promotion may happen later, after another fetch, or not at all.
        self._total_nccl += count
        self._total_bytes += nbytes
        self._total_latency_ms += fetch_ms
        self._steps += 1
        step = self._steps
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
        required = sum(HEADER_NBYTES + ref["layout"].payload_nbytes for _, ref in refs)
        budget.reserve(required)
        rejected_transfers = set()
        self._bounded_rejections = rejected_transfers
        try:
            result = self._fetch_unbounded(
                refs,
                budget=budget,
                reserved=required,
            )
        except TransportUnusableError:
            budget.close()
            raise
        except Exception as exc:
            logger.error(
                "[NCCLTransportStrategy] bounded receive failed: error=%s: %s; %s",
                type(exc).__name__,
                exc,
                budget.snapshot(),
            )
            budget.release(required)
            raise ReceiveMemoryError(f"{type(exc).__name__}: {exc}") from None
        finally:
            self._bounded_rejections = None
        logger.info("[NCCLTransportStrategy] receive memory: %s", budget.snapshot())
        return result

    def _fetch_unbounded(
        self,
        refs: List[Tuple[Any, dict]],
        *,
        budget=None,
        reserved=0,
    ) -> Tuple[dict, int, float]:
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
        recvs: List[Tuple[Any, dict, int, torch.Tensor, ReceiveBufferLease]] = []
        native_pending = False
        retain_pins = False
        stream = None
        operations = []
        results: Dict[int, dict] = {}
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

            total_bytes = 0
            posted: List[Tuple[Any, dict, int, torch.Tensor, ReceiveBufferLease]] = []
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
                    lease = getattr(prepared, "lease", None)
                    if lease is None:
                        lease = _DetachedReceiveLease(recv_buf)
                    operations.append((operation, response_key, None))
                    # Record the pin BEFORE attempting the recv so the finally
                    # below unpins this comm even if the enqueue raises.
                    recvs.append((idx, ref, comm_idx, recv_buf, lease))
                    # Mark before issue: a raising native call may already have
                    # enqueued work or partially written the destination.
                    native_pending = True
                    try:
                        # The allocator may reuse storage whose previous user
                        # is still queued on the allocation stream. Raw NCCL
                        # writes on a different stream must wait for that use,
                        # even though torch.empty itself launches no kernel.
                        wait_event(stream, record_event(None))
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
                    posted.append((idx, ref, comm_idx, recv_buf, lease))

                if not posted:
                    if budget is not None:
                        budget.release(reserved)
                    rejected = {
                        idx
                        for idx, ref in refs
                        if self._bounded_rejections is not None
                        and ref["transfer_id"] in self._bounded_rejections
                    }
                    return (
                        ReceivedBatch({}, budget, 0, rejected_keys=rejected),
                        0,
                        get_trace_time() - t0,
                    )

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

            for idx, ref, _comm_idx, recv_buf, _lease in posted:
                try:
                    cached_views = _lease.bind_views(
                        ref["layout"],
                        offset=HEADER_NBYTES,
                        reset_fields=VARLEN_FIELDS,
                    )
                    if self._decode_account is None:
                        gpu_data = _verify_and_unpack(
                            recv_buf, ref, device, views=cached_views
                        )
                    else:
                        gpu_data = _verify_and_unpack(
                            recv_buf,
                            ref,
                            device,
                            on_allocate=self._decode_account,
                            views=cached_views,
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
            actual = sum(
                lease.tensor.untyped_storage().nbytes() for *_, lease in posted
            )
            allocated = sum(
                lease.tensor.untyped_storage().nbytes() for *_, lease in recvs
            )
            if budget is not None:
                budget.attribute(raw=-allocated, decoded=actual)
                budget.release(reserved - actual)
            rejected = {
                idx
                for idx, ref in refs
                if self._bounded_rejections is not None
                and ref["transfer_id"] in self._bounded_rejections
            }
            leases = tuple(entry[4] for entry in posted)

            def release_leases():
                for lease in leases:
                    lease.release()

            batch = ReceivedBatch(
                results,
                budget,
                actual,
                rejected_keys=rejected,
                owners=tuple(lease.tensor for lease in leases),
                on_release=release_leases,
            )
            return batch, total_bytes, get_trace_time() - t0
        except BaseException as exc:
            if native_pending:
                retain_pins = True
                if budget is not None:
                    budget.close()
                for *_, lease in recvs:
                    lease.quarantine()
                _FAILED_RECEIVE_OWNERS.append(
                    (recvs, cache, stream, self._receive_pool)
                )
                raise TransportUnusableError(
                    "Receive interrupted before native completion; "
                    "storage, communicator pins and reservation retained. "
                    "Restart the receiver."
                ) from exc
            allocated = sum(
                lease.tensor.untyped_storage().nbytes() for *_, lease in recvs
            )
            if budget is not None and allocated:
                budget.attribute(raw=-allocated)
            for *_, lease in recvs:
                lease.release()
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
                for _i, _ref, _c, _b, _lease in recvs:
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
        self,
        recvs: List[Tuple[Any, dict, int, torch.Tensor, ReceiveBufferLease]],
        cache: Any,
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
        for _idx, ref, _comm_idx, _buf, _lease in recvs:
            pair = _pair_key(ref, receiver_rank)
            if pair not in warm:
                continue  # still warming -> keep the comm, retry
            self._quarantine_endpoint(
                cache,
                (
                    ref["sender_replica"],
                    ref.get("sender_incarnation", ""),
                    ref["sender_rank"],
                ),
                pair,
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
        health_key = (
            sender_replica,
            ref.get("sender_incarnation", ""),
            sender_rank,
        )
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
                lease = None
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
                    if self._receive_pool is None:
                        self._receive_pool = ReceiveBufferPool()
                    lease = self._receive_pool.acquire(
                        ref["layout"].layout_id,
                        HEADER_NBYTES + ref["layout"].payload_nbytes,
                        self._device,
                    )
                    recv_buf = lease.tensor
                    if self._receive_budget is not None:
                        self._receive_budget.attribute(
                            raw=recv_buf.untyped_storage().nbytes()
                        )
                except BaseException as e:
                    if lease is not None:
                        lease.release()
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
                    comm_idx,
                    recv_buf,
                    lease,
                    operation,
                    result.response_key,
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
        if isinstance(rollout_output, str) and rollout_output.startswith(
            NCCL_COMPLETION_PREFIX
        ):
            return rollout_output[len(NCCL_COMPLETION_PREFIX) :]
        if isinstance(rollout_output, dict) and rollout_output.get("_nccl"):
            return str(rollout_output.get("_transfer_id", ""))
        return str(rollout_output)

    def _mark_known_rejection(self, ref):
        if self._bounded_rejections is not None:
            self._bounded_rejections.add(ref["transfer_id"])


# ---------------------------------------------------------------------------
# Module-level helpers (ref parsing / buffer alloc / unpack)
# ---------------------------------------------------------------------------


def _parse_ref(rollout_output: Any) -> Optional[dict]:
    """Normalize a completion string or dict metadata into an internal ref.

    Returns normalized transfer and layout identity, or ``None`` if the input
    is not an NCCL reference.
    """
    if isinstance(rollout_output, str):
        if not rollout_output.startswith(NCCL_COMPLETION_PREFIX):
            return None
        raise ValueError(
            "NCCL cleanup completion strings do not carry layout or sender identity"
        )
    if isinstance(rollout_output, dict) and rollout_output.get("_nccl"):
        transfer_id = rollout_output.get("_transfer_id", "")
        if (
            not isinstance(transfer_id, str)
            or not transfer_id
            or len(transfer_id) > 1024
        ):
            raise ValueError("NCCL payload reference has an invalid transfer identity")
        rollout_idx = rollout_output.get(
            "_rollout_idx", parse_transfer_rollout_idx(transfer_id)
        )
        if (
            not isinstance(rollout_idx, int)
            or isinstance(rollout_idx, bool)
            or rollout_idx < 0
        ):
            raise ValueError("NCCL payload reference has an invalid rollout index")
        if parse_transfer_rollout_idx(transfer_id) != rollout_idx:
            raise ValueError("NCCL transfer identity and rollout index do not agree")
        if rollout_output.get("_wire_version") != WIRE_VERSION:
            raise ValueError("NCCL payload reference has an unsupported wire version")
        layout_id = validate_layout_id(rollout_output.get("_layout_id"))
        sender_rank = rollout_output.get(
            "_sender_rank", rollout_idx if rollout_idx >= 0 else 0
        )
        if (
            not isinstance(sender_rank, int)
            or isinstance(sender_rank, bool)
            or sender_rank < 0
        ):
            raise ValueError("NCCL payload reference has an invalid sender rank")
        # Globally-unique sender identity: the producer's replica id/name.
        sender_replica = rollout_output.get("_sender_replica")
        sender_incarnation = rollout_output.get("_sender_incarnation")
        if (
            not isinstance(sender_replica, str)
            or not sender_replica
            or len(sender_replica) > 256
        ):
            raise ValueError("NCCL payload reference has an invalid sender identity")
        if (
            not isinstance(sender_incarnation, str)
            or not sender_incarnation
            or len(sender_incarnation) > 256
        ):
            raise ValueError("NCCL payload reference has an invalid sender incarnation")
        return {
            "transfer_id": transfer_id,
            "rollout_idx": rollout_idx,
            "sender_rank": sender_rank,
            "sender_replica": sender_replica,
            "layout_id": layout_id,
            "sender_incarnation": sender_incarnation,
        }
    return None


def _has_layout(ref: dict) -> bool:
    """Return whether a reference carries enough identity to resolve a layout."""
    if ref.get("layout_id"):
        return True
    logger.error(
        "[NCCLTransportStrategy] NCCL reference %s carries no layout identity; refusing "
        "to decode its payload (episode falls back to the Redis path)",
        ref.get("transfer_id", "<unknown>"),
    )
    return False


def _identity_cache_key(ref: dict) -> str:
    """Bind a cached payload to every compact identity field it validated."""
    return ":".join(
        (
            ref["sender_replica"],
            ref["sender_incarnation"],
            ref["transfer_id"],
            ref["layout_id"],
        )
    )


def _pair_key(ref: dict, receiver_rank: int):
    """Globally-unique comm-cache pair key for a transfer.

    ``(sender_replica, sender_incarnation, sender_rank, receiver_rank)`` -- keyed on the
    rollout replica's globally-unique identity so two replicas sharing a
    per-replica ``sender_rank`` (e.g. both 0) map to DISTINCT communicators.
    """
    return (
        ref["sender_replica"],
        ref.get("sender_incarnation", ""),
        ref["sender_rank"],
        receiver_rank,
    )


def _alloc_recv_buffer(layout: Optional[TensorLayout], device: Any) -> torch.Tensor:
    """Allocate a flat uint8 GPU buffer sized for ``layout`` plus its header.

    The producer prefixes :data:`HEADER_NBYTES` of self-describing header to
    every payload, so the wire size is header + entry size on both ends.
    """
    if layout is None:
        raise ValueError("cannot allocate NCCL recv buffer without a layout")
    return torch.empty(
        HEADER_NBYTES + layout.payload_nbytes, dtype=torch.uint8, device=device
    )


def _verify_and_unpack(
    recv_buf: torch.Tensor,
    ref: dict,
    device: Any,
    *,
    on_allocate=None,
    views=None,
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
    layout = ref.get("layout")
    if layout is None:
        return {}
    entry_size = layout.payload_nbytes
    header = bytes(recv_buf[:HEADER_NBYTES].cpu().numpy())
    verify_header(
        header,
        transfer_id=ref["transfer_id"],
        payload_nbytes=entry_size,
        layout_id=layout.layout_id,
        sender_incarnation=ref.get("sender_incarnation", ""),
    )
    return _unpack(recv_buf[HEADER_NBYTES:], layout, device, views=views)


def _unpack(
    recv_buf: torch.Tensor,
    layout: Optional[TensorLayout],
    device: Any,
    *,
    views=None,
) -> dict:
    """Bind typed tensor views to a validated payload allocation."""
    if layout is None:
        return {}
    out = views if views is not None else layout.views(recv_buf)
    _truncate_to_episode_len(out)
    return out


def _truncate_to_episode_len(data: dict) -> None:
    truncate_trajectory(data)


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
