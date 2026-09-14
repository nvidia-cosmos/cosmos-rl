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

"""NCCLRolloutMixin — producer side of the in-tree NCCL payload transport.

Mirrors :class:`UCXXRolloutMixin` but moves bytes with NCCL P2P instead of
UCXX/SHM.  Responsibilities:

* :meth:`setup_nccl` — build the trajectory schema, the backpressured
  :class:`SendBufferRegistry`, the lazy :class:`CommCache`, the transfer
  stream pool, and start the request-serving + cleanup-subscriber threads.
* :meth:`write_to_buffer` — pack a trajectory into a fixed-schema GPU
  buffer, record a compute-stream ready-event, register the buffer, and
  return dict metadata (plus the ``nccl:<id>`` completion string).
* the serve loop — a **bounded sender-thread pool** drains a per-pair FIFO
  of accepted requests with ``nccl_send`` on a per-process transfer stream,
  so a slow peer head-of-line-blocks only its own pair, not all pairs.
* the cleanup subscriber — frees GPU buffers when the controller discards
  the rollout (``nccl_cleanup`` channel).

Per-pair send ordering
----------------------
A cached 2-rank comm is an ORDERED stream with no tags: the receiver's k-th
``nccl_recv`` takes the k-th ``nccl_send``.  The receiver posts its recvs in
the order it observes ``ACCEPTED``, and this producer accepts on a single
pub/sub listener thread, so accept order IS the order the receiver expects.
Handing each accepted send straight to a thread pool broke that -- two pool
threads racing for the launch lock can launch pair-mates in either order, and
the receiver then unpacks one payload with the other's schema.  So sends are
queued **per pair** and drained by at most one pool task per pair, in accept
order.  See :mod:`.header` for the check that catches any residual desync.

Concurrency model (from the pynccl review)
------------------------------------------
pynccl P2P runs inline on the caller thread as CUDA-stream-async enqueues;
an ``ncclSend`` only *completes* once the peer posts the matching recv.  A
single serve thread that syncs per-send would head-of-line-block every
pair, so requests are dispatched to a bounded thread pool and each send
carries a finite ``timeout_ms``; a transient failure quarantines the pair
via the comm cache rather than wedging the worker.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Set

import numpy as np
import torch

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.payload_transport.nccl.buffer_registry import (
    SendBufferEntry,
    SendBufferRegistry,
)
from cosmos_rl.utils.payload_transport.nccl.comm_cache import (
    CommCache,
    RECEIVER_LOCAL_RANK,
    SENDER_LOCAL_RANK,
)
from cosmos_rl.utils.payload_transport.nccl.header import (
    HEADER_NBYTES,
    HEADER_VERSION,
    build_header,
)
from cosmos_rl.utils.payload_transport.nccl.context import (
    resolve_custom_int as _resolve_custom_int,
    resolve_global_rank as _resolve_global_rank,
    resolve_max_live_comms as _resolve_max_live_comms,
    resolve_prefix as _resolve_prefix,
)
from cosmos_rl.utils.payload_transport.nccl.protocol import (
    NCCL_COMPLETION_PREFIX,
    build_cleanup_channel,
    build_request_channel,
    build_rollout_prefix,
)
from cosmos_rl.utils.payload_transport.nccl.rendezvous import (
    NcclRendezvous,
    TransferStatus,
    parse_request_message,
)
from cosmos_rl.utils.payload_transport.nccl.streams import (
    bind_thread_device,
    get_transfer_stream_pool,
    record_event,
    wait_event,
)
from cosmos_rl.utils.payload_transport.pack import pack_trajectory_into
from cosmos_rl.utils.trace import get_trace_time as _trace_time
from cosmos_rl.utils.trajectory import episode_length as _episode_length
from cosmos_rl.utils.trajectory import (
    EPISODE_LENGTH,
    REWARDS,
    build_trajectory_schema,
    schema_layout,
    serialize_schema,
)


def _producer_pair_key(sender_rank: int, receiver_replica: Any, receiver_rank: int):
    """Sender-side comm-cache / quarantine key for one 2-rank pair.

    Symmetric to the consumer's ``_pair_key``.  A producer serves many
    policy replicas, so ``receiver_rank`` alone is NOT unique: each policy
    replica is a separate distributed world where ``receiver_rank`` restarts
    at 0.  The globally-unique ``receiver_replica`` (the policy's
    ``replica_name``) disambiguates them so two replicas never share a comm.
    """
    return (sender_rank, receiver_replica, receiver_rank)


#: Safety margin (seconds) subtracted from a request's ``req_deadline`` before
#: this producer commits to serving it.  Replying ACCEPTED is a promise to send
#: exactly once, and the receiver posts its matching recv only if it sees that
#: reply before its own deadline.  Accepting a request that is *about* to
#: expire races the receiver's give-up: it may never post the recv, leaving our
#: send to be taken by the NEXT recv on that pair -- an off-by-one that
#: mispairs every later payload.  The margin also absorbs the small clock skew
#: between the two hosts (the deadline is wall-clock).  Cheap: the cold-start
#: budget is tens of seconds, so this only ever drops requests that were about
#: to be wasted anyway.
_ACCEPT_MARGIN_S = 0.25

#: Ceiling on the margin as a fraction of the receiver's own budget, so a
#: deployment running very short rendezvous timeouts is not left with every
#: request landing inside the margin and nothing ever served.
_ACCEPT_MARGIN_FRACTION = 0.1


def _accept_margin(req_timeout: Any) -> float:
    """How early to stop accepting, given the receiver's budget for this try."""
    try:
        budget = float(req_timeout)
    except (TypeError, ValueError):
        return _ACCEPT_MARGIN_S  # older receiver: no budget in the request
    return min(_ACCEPT_MARGIN_S, max(0.0, budget) * _ACCEPT_MARGIN_FRACTION)


@dataclass
class _PendingSend:
    """One ACCEPTED transfer awaiting its turn on a pair's ordered stream."""

    entry: SendBufferEntry
    receiver_rank: int
    uid_key: Any
    receiver_replica: Any
    transfer_id: str


class NCCLRolloutMixin:
    """Mixin for rollout workers to serve trajectory payloads over NCCL.

    Usage::

        class MyWorker(NCCLRolloutMixin, BaseWorker):
            def post_init_hook(self):
                self.setup_nccl(
                    replica_id=self.replica_name,
                    rollout_idx=self.replica_idx,
                    redis_client=self.redis_client,
                    config=self.config,
                    max_steps=100, obs_dim=4, action_dim=2,
                )

            def generate_rollout(self):
                traj = collect_trajectory()
                return self.write_to_buffer(traj) or traj

            def cleanup(self):
                self.cleanup_nccl()
    """

    _nccl_enabled: bool = False
    _nccl_registry: Optional[SendBufferRegistry] = None
    _nccl_comm_cache: Optional[CommCache] = None
    _nccl_rendezvous: Optional[NcclRendezvous] = None
    _nccl_streams: Any = None
    _nccl_redis: Any = None
    _nccl_rollout_idx: int = 0
    _nccl_sender_rank: int = 0
    _nccl_prefix: str = ""
    _nccl_schema: Optional[list] = None
    _nccl_offsets: Optional[Dict[str, int]] = None
    _nccl_entry_size: int = 0
    _nccl_device: Any = None
    _nccl_send_timeout_ms: int = 30000
    # Set by ``setup_nccl``; ``None`` on a bare harness that never started the
    # listener threads, which the drain loop below must tolerate.
    _nccl_shutdown: Any = None
    # Serializes the NCCL launch sequence across the sender-thread pool so
    # concurrent multi-comm launches on one GPU can't deadlock (set in setup).
    _nccl_send_lock: Any = None
    # Per-pair FIFO of accepted-but-unsent transfers + the set of pairs a pool
    # task is currently draining.  Guarded by ``_nccl_pair_lock``.  This is what
    # keeps each pair's sends in accept order (see the module docstring).
    _nccl_pair_lock: Any = None
    _nccl_pair_queues: Optional[Dict[Any, Deque[_PendingSend]]] = None
    _nccl_pair_draining: Optional[Set[Any]] = None

    def setup_nccl(
        self,
        *,
        replica_id: str,
        rollout_idx: int,
        redis_client: Any,
        config: Any,
        max_steps: int = 100,
        obs_dim: int = 4,
        action_dim: int = 2,
        sender_rank: Optional[int] = None,
        device: Any = None,
        num_sender_threads: Optional[int] = None,
        max_live_comms: Optional[int] = None,
        registry_capacity: int = 64,
        stream_pool_size: int = 1,
        registry_block_timeout: float = 30.0,
        send_timeout_ms: int = 30000,
    ) -> None:
        """Initialise the producer state and start the serve/cleanup threads.

        ``num_sender_threads`` and ``max_live_comms`` resolve as
        explicit-argument -> ``[custom]`` config -> derived default, so a caller
        that passes a value still wins while an operator can tune a deployment
        without touching the rollout worker.

        ``num_sender_threads`` bounds how many requests are served at once; it
        should be at least the number of policy replicas fetching concurrently,
        or requests head-of-line-block behind the pool.
        """
        if num_sender_threads is None:
            num_sender_threads = _resolve_custom_int(
                config, "nccl_num_sender_threads", 2
            )
        if max_live_comms is None:
            max_live_comms = _resolve_max_live_comms(config, peer_role="policy")
        self._nccl_replica_id = replica_id
        self._nccl_send_timeout_ms = send_timeout_ms
        self._nccl_send_lock = threading.Lock()
        self._nccl_pair_lock = threading.Lock()
        self._nccl_pair_queues = {}
        self._nccl_pair_draining = set()
        self._nccl_rollout_idx = rollout_idx
        self._nccl_redis = redis_client
        self._nccl_device = device
        self._nccl_sender_rank = (
            sender_rank if sender_rank is not None else _resolve_global_rank()
        )

        self._nccl_schema = build_trajectory_schema(
            {"max_steps": max_steps, "obs_dim": obs_dim, "action_dim": action_dim}
        )
        self._nccl_offsets, self._nccl_entry_size = schema_layout(self._nccl_schema)

        self._nccl_prefix = _resolve_prefix(config)
        self._nccl_registry = SendBufferRegistry(
            capacity=registry_capacity,
            block_timeout=registry_block_timeout,
            on_free=self._on_buffer_free,
        )
        self._nccl_comm_cache = CommCache(max_live=max_live_comms)
        self._nccl_rendezvous = NcclRendezvous(redis_client, self._nccl_prefix)
        self._nccl_streams = get_transfer_stream_pool(
            size=stream_pool_size, device=device
        )

        # Bounded sender pool + the pub/sub listener threads.
        self._nccl_executor = ThreadPoolExecutor(
            max_workers=max(1, num_sender_threads),
            thread_name_prefix="nccl-sender",
        )
        self._nccl_shutdown = threading.Event()
        self._nccl_threads: List[threading.Thread] = []
        # Request channel is routed by the globally-unique replica id so a
        # transfer request reaches exactly this producer (two rollout
        # replicas share sender_rank=0 and would otherwise collide).  The
        # cleanup channel stays keyed by the integer rollout_idx so it
        # matches the controller, which derives that index from the
        # transfer id (idempotent free tolerates any collision there).
        self._start_listener(
            channel=build_request_channel(
                build_rollout_prefix(self._nccl_prefix, replica_id)
            ),
            handler=self._dispatch_request,
            name="nccl-req-listener",
        )
        self._start_listener(
            channel=build_cleanup_channel(
                build_rollout_prefix(self._nccl_prefix, rollout_idx)
            ),
            handler=self._handle_cleanup,
            name="nccl-cleanup-listener",
        )

        self._nccl_enabled = True
        # ``payload_header`` is the on-wire framing version.  Both ends must
        # agree on it, so log it where an operator comparing a rollout and a
        # policy log can see at a glance whether they are running one build.
        logger.info(
            "[NCCLRolloutMixin] Worker '%s' ready (rollout_idx=%d, sender_rank=%d, "
            "entry_size=%.1f MB, sender_threads=%d, capacity=%d, payload_header=v%d)",
            replica_id,
            rollout_idx,
            self._nccl_sender_rank,
            self._nccl_entry_size / 1e6,
            num_sender_threads,
            registry_capacity,
            HEADER_VERSION,
        )

    # ------------------------------------------------------------------
    # Producer: pack + register
    # ------------------------------------------------------------------

    def write_to_buffer(self, trajectory: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Pack ``trajectory`` into a GPU send buffer; return dict metadata.

        The returned dict is the consumer-facing reference; its
        ``completion`` field is the ``nccl:<transfer_id>`` string the
        controller keys discard-cleanup on.  Returns ``None`` (so the
        caller falls back to the plain trajectory) if NCCL is disabled or
        packing fails.
        """
        if not self._nccl_enabled or self._nccl_registry is None:
            return None
        try:
            t0 = _trace_time()
            ep_len = _episode_length(trajectory, self._nccl_schema)
            # The transfer id is minted BEFORE packing: it is stamped into the
            # payload header so the receiver can prove the bytes it got belong
            # to the transfer it asked for.
            transfer_id = f"{self._nccl_rollout_idx}:{uuid.uuid4().hex}"
            gpu_buf, ready_event = self._pack(trajectory, ep_len, transfer_id)
            self._nccl_registry.register(
                transfer_id,
                gpu_buf,
                ready_event=ready_event,
                nbytes=HEADER_NBYTES + self._nccl_entry_size,
            )
            # Producer-side trace, mirroring UCXX's op=ucxx_write, so the
            # rl-gym log_analyzer can attribute per-rollout payload bytes.
            logger.debug(
                "[Trace] thread=%s op=nccl_write pack_ms=%.1f bytes=%d",
                threading.current_thread().name,
                _trace_time() - t0,
                self._nccl_entry_size,
            )
            rewards = trajectory.get(REWARDS)
            return {
                "_nccl": True,
                "_transfer_id": transfer_id,
                # Globally-unique sender identity so the receiver routes the
                # request + keys the comm cache per rollout replica (not just
                # per sender_rank, which is 0 for every single-GPU replica).
                "_sender_replica": self._nccl_replica_id,
                "_sender_rank": self._nccl_sender_rank,
                "_rollout_idx": self._nccl_rollout_idx,
                "_schema": serialize_schema(self._nccl_schema),
                "completion": f"{NCCL_COMPLETION_PREFIX}{transfer_id}",
                REWARDS: rewards.tolist() if hasattr(rewards, "tolist") else rewards,
                EPISODE_LENGTH: ep_len,
            }
        except Exception as e:
            # Name the rollout so a disk fallback is attributable; the packer
            # already names the offending schema field in its message.
            logger.error(
                "[NCCLRolloutMixin] write_to_buffer failed for replica=%s "
                "rollout_idx=%s; falling back to the plain trajectory: %s",
                self._nccl_replica_id,
                self._nccl_rollout_idx,
                e,
            )
            return None

    def _pack(self, trajectory: Dict[str, Any], ep_len: int, transfer_id: str):
        """Coalesce the schema tensors into one contiguous GPU uint8 buffer.

        The buffer is ``HEADER_NBYTES + entry_size`` long: a self-describing
        header (transfer id digest + payload size, see :mod:`.header`) followed
        by the schema-defined region the consumer slices.  The header costs 32
        bytes on a payload measured in hundreds of MB and is what lets the
        receiver reject a buffer that belongs to a different transfer instead
        of decoding it as garbage.

        Returns ``(gpu_buffer, ready_event)`` where ``ready_event`` is
        recorded on the compute stream once packing enqueues, so the
        transfer stream can wait on it before ``nccl_send``.
        """
        device = self._nccl_device
        entry_size = self._nccl_entry_size
        gpu_packed = torch.zeros(
            HEADER_NBYTES + entry_size, dtype=torch.uint8, device=device
        )
        header = build_header(transfer_id=transfer_id, payload_nbytes=entry_size)
        gpu_packed[:HEADER_NBYTES] = torch.frombuffer(
            bytearray(header), dtype=torch.uint8
        ).to(device)
        # A slice is a VIEW, so the packer writes straight into the payload
        # region and its schema offsets stay header-relative on both ends.
        pack_trajectory_into(
            gpu_packed[HEADER_NBYTES:],
            trajectory,
            self._nccl_schema,
            self._nccl_offsets,
            ep_len,
            device=device,
        )
        ready_event = record_event()  # on the current (compute) stream
        return gpu_packed, ready_event

    # ------------------------------------------------------------------
    # Serve loop (bounded sender pool)
    # ------------------------------------------------------------------

    def _dispatch_request(self, raw_message: Any) -> None:
        """Parse a request and serve its CONTROL plane on this listener thread.

        Only the send is handed to the bounded sender pool (see
        :meth:`_handle_request`).  Submitting the whole handler here -- the
        previous behaviour -- put the cheap Redis acknowledgement behind the
        same workers that block in ``nccl_send``, so a producer serving more
        concurrent transfers than it has sender threads could not answer the
        very requests whose recvs would unblock those sends.
        """
        msg = parse_request_message(raw_message)
        if msg is None:
            return
        self._handle_request(msg)

    def _handle_request(self, msg: Dict[str, Any]) -> None:
        """Serve one transfer request: ack on this thread, QUEUE the send.

        Runs on the pub/sub listener thread.  Everything here is bounded --
        a registry lease, a cached-comm lookup, at most one Redis UID read,
        and one Redis reply -- so the control plane answers at listener speed
        no matter how backed up the sender pool is.  The comm BUILD and the
        blocking ``nccl_send`` stay on the executor, where they belong.

        The ACCEPTED reply is written **before** the send is queued so the
        receiver can post its matching recv; a missing buffer replies MISSING
        and returns without touching NCCL.
        """
        rv = self._nccl_rendezvous
        registry = self._nccl_registry
        cache = self._nccl_comm_cache
        if rv is None or registry is None or cache is None:
            return

        transfer_id = msg.get("transfer_id", "")
        resp_key = msg.get("resp_key")
        receiver_rank = int(msg.get("receiver_rank", -1))
        # Globally-unique policy-replica identity (None for legacy single-
        # policy-replica producers; the pair key then degrades to the old
        # (sender_rank, None, receiver_rank), which is still unique for one
        # policy replica).
        receiver_replica = msg.get("receiver_replica")
        uid_key = msg.get("uid_key")

        # Bilateral cancellation: drop a request whose receiver has already
        # stopped waiting.  Serving one sends a late ACCEPTED and launches an
        # nccl_send with no matching recv, which pins the send lock + a sender
        # thread until the send watchdog fires -> starves every other consumer
        # (the N_POLICY>=4 cascade).  The dominant source of that lateness used
        # to be this handler waiting its turn in the sender-pool queue; running
        # the control plane on the listener removes that delay, but a backed-up
        # listener can still arrive late, so keep checking the receiver's
        # absolute wall-clock deadline before touching the registry or replying.
        #
        # The check carries ``_ACCEPT_MARGIN_S`` of slack so a request that is
        # merely ABOUT to expire is dropped too: accepting one races the
        # receiver's give-up, and a receiver that never posts the matching recv
        # leaves our send to be taken by its NEXT recv on this pair -- an
        # off-by-one that mispairs every payload after it.
        req_deadline = msg.get("req_deadline")
        if req_deadline is not None and time.time() >= (
            float(req_deadline) - _accept_margin(msg.get("req_timeout"))
        ):
            logger.debug(
                "[NCCLRolloutMixin] dropping expired request %s (receiver "
                "deadline passed; not sending a late unmatched ACCEPTED)",
                transfer_id,
            )
            return

        # Atomically LEASE the buffer for the duration of this send.  A bare
        # get() here would leave a use-after-free window: the send doesn't
        # record its in-flight marker until _send acquires the launch lock and
        # waits on ready_event, and in that gap capacity eviction or a discard
        # free() could pop the entry and release its GPU storage mid-send.
        entry = registry.acquire(transfer_id)
        if entry is None:
            # Buffer was recycled/evicted before the request arrived.
            if resp_key:
                rv.respond(resp_key=resp_key, status=TransferStatus.MISSING)
            return
        # From here until _send takes over lease ownership, ANY failure must
        # release the acquire() lease -- otherwise the buffer is pinned
        # un-reapable.  (_send self-balances its own lease on every exit path,
        # so the except below must NOT abandon again: a double-decrement could
        # steal a *concurrent* receiver's lease on the same shared buffer.)
        try:
            pair = _producer_pair_key(
                self._nccl_sender_rank, receiver_replica, receiver_rank
            )
            # We only need a usable UID when we must BUILD the pair comm (it is
            # not already cached on our side).  Renegotiate (ask for a fresh UID)
            # in either build case where we lack one:
            #   * receiver omitted uid_key -- it believes the comm is cached on
            #     BOTH sides, but our side evicted it (independent LRU); OR
            #   * receiver sent uid_key but the UID is unreadable (its Redis SET
            #     was lost / the key expired) -- read_uid() returns None.
            # Building from an empty UID in either case desyncs the two comm
            # builds (empty vs real) -> 600s watchdog hang, so ask the receiver
            # to re-initiate with a freshly minted + published UID instead.
            needs_build = cache.get(pair) is None
            have_uid = (
                bool(rv.read_uid(uid_key)) if (needs_build and uid_key) else False
            )
            renegotiate = needs_build and not have_uid
        except Exception as e:
            registry.abandon_inflight(entry)  # release lease: never reached _send
            logger.warning(
                "[NCCLRolloutMixin] request setup failed for %s: %s", transfer_id, e
            )
            return
        if renegotiate:
            registry.abandon_inflight(entry)  # release the lease -- not sending
            if resp_key:
                rv.respond(resp_key=resp_key, status=TransferStatus.NEED_UID)
            return
        # ACCEPTED is a promise to send exactly once, in this order: from here
        # on the receiver will post a matching recv, and the pair's stream is
        # only correct if our sends leave in the same order we accepted them.
        # This handler runs on the single pub/sub listener thread, so appending
        # to the pair's FIFO here fixes that order; a pool task drains it.
        if resp_key:
            rv.respond(resp_key=resp_key, status=TransferStatus.ACCEPTED)

        # Hand ONLY the blocking transfer to the bounded pool.  The lease taken
        # by acquire() above now outlives this function, so every outcome has to
        # balance it exactly once:
        #   * the queue/submit fails -> the send never runs and never takes
        #     ownership; _drop_pending releases it (and resyncs the pair, since
        #     we already promised a send that will not arrive);
        #   * the pending item is still queued at shutdown -> released by
        #     _abandon_queued_sends;
        #   * the send starts -> _send owns the lease and balances it on every
        #     exit path of its own finally.  Do NOT release it again here: a
        #     double-decrement could steal a concurrent receiver's lease on the
        #     same shared buffer.
        self._enqueue_send(
            pair,
            _PendingSend(
                entry=entry,
                receiver_rank=receiver_rank,
                uid_key=uid_key,
                receiver_replica=receiver_replica,
                transfer_id=transfer_id,
            ),
        )

    # ------------------------------------------------------------------
    # Per-pair send ordering
    # ------------------------------------------------------------------

    #: Guards the lazy init below.  Shared by every instance, held only while
    #: a producer first materialises its per-pair state, never during a send.
    _nccl_pair_state_init_lock = threading.Lock()

    def _ensure_pair_state(self) -> Any:
        """Return the per-pair queue lock, materialising it if setup was skipped.

        ``setup_nccl`` builds this state; bare harnesses that assemble a
        producer attribute-by-attribute do not, and the queue must not be the
        thing that breaks them.
        """
        if self._nccl_pair_lock is None:
            with NCCLRolloutMixin._nccl_pair_state_init_lock:
                if self._nccl_pair_lock is None:
                    self._nccl_pair_queues = {}
                    self._nccl_pair_draining = set()
                    # Assigned LAST: it is the sentinel the check above reads.
                    self._nccl_pair_lock = threading.Lock()
        return self._nccl_pair_lock

    def _enqueue_send(self, pair: Any, pending: _PendingSend) -> None:
        """Append an accepted send to ``pair``'s FIFO, draining it if idle.

        At most ONE pool task drains a given pair, so the pair's sends leave in
        accept order no matter how many workers the pool has.  Pool width still
        bounds how many *pairs* transfer concurrently, which is what it was
        always for (a slow peer must not head-of-line-block the others).

        Submitting each send independently -- the previous behaviour -- let two
        workers race for the launch lock and reverse two sends on one comm.
        Because a comm carries no tags, the receiver then took the wrong
        payload into the buffer it had sized and unpacked it with the other
        transfer's schema.
        """
        pair_lock = self._ensure_pair_state()
        with pair_lock:
            queue = self._nccl_pair_queues.setdefault(pair, deque())
            queue.append(pending)
            if pair in self._nccl_pair_draining:
                # A drain task is already running for this pair and will pick
                # this item up; submitting another would reorder them.
                return
            self._nccl_pair_draining.add(pair)
        try:
            self._nccl_executor.submit(self._drain_pair_sends, pair)
        except Exception as e:
            # Pool already shut down: nothing will drain this pair.  Release
            # every lease we are holding for it and tear the comm down -- we
            # promised sends that will never arrive.
            with pair_lock:
                self._nccl_pair_draining.discard(pair)
                orphans = list(self._nccl_pair_queues.pop(pair, ()))
            for orphan in orphans:
                self._drop_pending(orphan, pair, reason=f"send pool unavailable: {e}")

    def _drain_pair_sends(self, pair: Any) -> None:
        """Send every queued transfer for ``pair``, oldest first.

        Runs on one pool thread.  Exits only when the pair's queue is empty
        *and* the empty check and the de-registration happen under the same
        lock acquisition, so an item appended concurrently either lands before
        the check (this task sends it) or after the de-registration (the
        appender starts a fresh task) -- never in a gap where both sides think
        the other owns it.

        Nothing may escape this loop while the pair is still marked as being
        drained: the mark is what stops a second task from starting, so leaving
        it set would silently strand every later send for that pair.
        """
        pair_lock = self._ensure_pair_state()
        try:
            while True:
                with pair_lock:
                    queue = self._nccl_pair_queues.get(pair)
                    if not queue:
                        self._nccl_pair_queues.pop(pair, None)
                        self._nccl_pair_draining.discard(pair)
                        return
                    pending = queue.popleft()
                if self._nccl_shutdown is not None and self._nccl_shutdown.is_set():
                    self._drop_pending(pending, pair, reason="producer shutting down")
                    continue
                try:
                    self._send_and_quarantine_on_failure(
                        pending.entry,
                        pending.receiver_rank,
                        pending.uid_key,
                        pending.receiver_replica,
                        pending.transfer_id,
                    )
                except Exception as e:  # pragma: no cover - defensive
                    # _send already balanced this entry's lease, so do NOT
                    # abandon it again (a double-decrement could steal another
                    # receiver's lease on the same buffer).  Just end the pair's
                    # stream: we cannot tell whether the send launched.
                    logger.error(
                        "[NCCLRolloutMixin] send handler for %s raised %s; "
                        "aborting pair %s",
                        pending.transfer_id,
                        e,
                        pair,
                    )
                    cache = self._nccl_comm_cache
                    if cache is not None:
                        cache.abort(pair)
        except BaseException:
            # Abnormal exit only -- the normal return above has already cleared
            # the mark, and clearing it again here could release a mark a
            # freshly-started successor task now owns (two concurrent drains =
            # the reordering this whole queue exists to prevent).
            with pair_lock:
                self._nccl_pair_draining.discard(pair)
            raise

    def _drop_pending(self, pending: _PendingSend, pair: Any, *, reason: str) -> None:
        """Release an accepted send that will never be launched, and resync.

        We already replied ACCEPTED, so the receiver may have posted a recv
        that nothing will satisfy -- and, worse, the *next* send on this pair
        would be taken by that orphaned recv.  Aborting the comm ends the
        ordered stream on our side; the receiver's next request finds our half
        gone, gets NEED_UID, and both halves rebuild from a fresh unique-ID
        with empty queues.
        """
        registry = self._nccl_registry
        if registry is not None:
            registry.abandon_inflight(pending.entry)
        logger.warning(
            "[NCCLRolloutMixin] dropping accepted send %s (%s); aborting pair %s "
            "so both halves rebuild rather than run one transfer out of step",
            pending.transfer_id,
            reason,
            pair,
        )
        cache = self._nccl_comm_cache
        if cache is not None:
            cache.abort(pair)

    def _abandon_queued_sends(self) -> None:
        """Release every still-queued send (teardown).

        ``cleanup_nccl`` shuts the pool down with ``cancel_futures=True``, so a
        drain task that never started leaves its pair's queue populated and its
        leases held.  Comms are aborted separately by ``cleanup_nccl``.
        """
        with self._ensure_pair_state():
            queues = self._nccl_pair_queues or {}
            orphans = [item for queue in queues.values() for item in queue]
            if self._nccl_pair_queues is not None:
                self._nccl_pair_queues.clear()
            if self._nccl_pair_draining is not None:
                self._nccl_pair_draining.clear()
        registry = self._nccl_registry
        for pending in orphans:
            if registry is not None:
                registry.abandon_inflight(pending.entry)
        if orphans:
            logger.debug(
                "[NCCLRolloutMixin] released %d queued send(s) at teardown",
                len(orphans),
            )

    def _send_and_quarantine_on_failure(
        self,
        entry: SendBufferEntry,
        receiver_rank: int,
        uid_key: Any,
        receiver_replica: Any,
        transfer_id: str,
    ) -> None:
        """Run one queued send, quarantining its pair if it fails."""
        try:
            self._send(entry, receiver_rank, uid_key, receiver_replica)
        except Exception as e:
            # _send already released the lease in its own finally -- do NOT
            # abandon here (would double-decrement).  Just quarantine the pair.
            cache = self._nccl_comm_cache
            health_key = (self._nccl_rollout_idx, self._nccl_sender_rank)
            logger.warning(
                "[NCCLRolloutMixin] send failed for %s: %s; quarantining %s",
                transfer_id,
                e,
                health_key,
            )
            if cache is not None:
                cache.quarantine(
                    _producer_pair_key(
                        self._nccl_sender_rank, receiver_replica, receiver_rank
                    )
                )

    def _send(
        self,
        entry: SendBufferEntry,
        receiver_rank: int,
        uid_key: Any,
        receiver_replica: Any = None,
    ) -> None:
        """Build/reuse the pair comm and enqueue the standalone ``nccl_send``."""
        # The caller (_handle_request) has already LEASED this entry via
        # registry.acquire() -- balance that lease on EVERY exit path: record a
        # completion event on success (add_done_event), or drop it on any
        # failure (abandon_inflight), including an import / device-bind /
        # comm-build error before the launch block.  ``recorded`` guards against
        # a double-release.  ALL failable setup lives inside the try so no path
        # can leak the lease.
        registry = self._nccl_registry
        recorded = False
        try:
            from cosmos_rl.utils import pynccl

            rv = self._nccl_rendezvous
            cache = self._nccl_comm_cache
            # This runs on an executor thread; bind it to our GPU so comm
            # creation and the send target the right device (CUDA current
            # device is thread-local).
            bind_thread_device(self._nccl_device)
            pair = _producer_pair_key(
                self._nccl_sender_rank, receiver_replica, receiver_rank
            )
            uid = rv.read_uid(uid_key) if uid_key else None
            # LEASE the comm rather than just fetching it: the launch below holds
            # this comm_idx across the whole collective, so an unpinned entry
            # could be picked as the LRU eviction victim by a concurrent build
            # for a different pair and aborted mid-send.  The pin gates eviction
            # only -- a quarantine abort still fires, which is what unwedges a
            # stuck send.
            with cache.leased(
                pair, uid_chars=uid or [], local_rank=SENDER_LOCAL_RANK
            ) as comm_idx:
                # Serialize the actual NCCL launch on this producer's GPU.  NCCL
                # requires deterministic single-threaded host launch ordering
                # across communicators on one device; the sender pool otherwise
                # issues concurrent group/send on the shared transfer stream from
                # >1 thread, which deadlocks natively at N_POLICY>=2 (each
                # rollout serves 2 policy consumers -> 2 concurrent sends) --
                # and, because run_task only arms its deadline AFTER the native
                # call returns, a native launch deadlock bypasses the send
                # timeout entirely.  Comm build + rendezvous above stay
                # concurrent; only the launch sequence is serialized, and every
                # op under the lock is an async stream ENQUEUE (completion
                # happens on the stream after release), so the lock stays fast
                # and cannot deadlock on the peer's recv.
                with self._nccl_send_lock:
                    stream = (
                        self._nccl_streams.acquire() if self._nccl_streams else None
                    )
                    # Do not send until the trajectory tensor is actually produced.
                    wait_event(stream, entry.ready_event)
                    pynccl.nccl_group_start(comm_idx)
                    # Finite timeout so a send whose peer has departed (e.g. the
                    # policy replica tore down first at job end) aborts via
                    # pynccl's watchdog instead of wedging the sender thread
                    # forever -- which would hang the worker's exit (executor
                    # threads are non-daemon).
                    pynccl.nccl_send(
                        entry.buffer,
                        RECEIVER_LOCAL_RANK,
                        comm_idx,
                        stream=stream,
                        timeout_ms=self._nccl_send_timeout_ms,
                    )
                    pynccl.nccl_group_end(comm_idx)
                    registry_done = record_event(stream)
                    # Keep the buffer alive until this send completes.  The
                    # registry reaps it once every recorded event fires
                    # (delivery), or on cleanup / capacity pressure -- and
                    # _on_buffer_free waits before releasing so the storage is
                    # never reused while NCCL reads it.
                    registry.add_done_event(entry, registry_done)
                    recorded = True
        finally:
            if not recorded:
                # Send never recorded an event (comm-build or launch failure) ->
                # release the acquire() lease so a dropped send never pins the
                # buffer forever.
                registry.abandon_inflight(entry)

    # ------------------------------------------------------------------
    # Cleanup subscriber
    # ------------------------------------------------------------------

    def _handle_cleanup(self, raw_message: Any) -> None:
        try:
            payload = raw_message
            if isinstance(payload, (bytes, bytearray)):
                payload = payload.decode("utf-8", errors="replace")
            data = json.loads(payload) if isinstance(payload, str) else payload
        except (ValueError, TypeError):
            return
        if not isinstance(data, dict):
            return
        transfer_id = data.get("transfer_id")
        if transfer_id and self._nccl_registry is not None:
            self._nccl_registry.free(transfer_id)

    def _on_buffer_free(self, entry: SendBufferEntry) -> None:
        """Registry callback: release the GPU tensor once its sends have drained.

        A send in flight (event recorded but not yet fired) is a raw NCCL read
        on the transfer stream that PyTorch's caching allocator does NOT track,
        so dropping the last reference now could let the storage be reused while
        NCCL is still reading it -> corrupted payload.  So wait for every
        recorded send-complete event -- but **bounded**: ``Event.synchronize``
        has no timeout and would hang forever if a send never got a matching
        recv (receiver crash / discard race), so poll ``query`` up to the send
        timeout and release regardless once it elapses.  The registry invokes
        this OUTSIDE its lock, so the wait blocks nothing.

        Ordering: first wait out any still-open send LEASE (``inflight`` -- a
        send that has been accepted but not yet recorded its completion event,
        e.g. it is still blocked on the launch lock or ready_event).  Freeing
        the storage while a lease is open would drop ``entry.buffer`` before the
        send even reads it.  Because ``add_done_event`` / ``abandon_inflight``
        mutate the entry object directly, this resolves correctly even after the
        entry was evicted / freed out of the registry.

        The lease wait and the event-drain wait get SEPARATE budgets (each the
        full send timeout) rather than sharing one deadline: otherwise a lease
        that consumes most of the budget would leave almost no time for the
        recorded events to actually drain before the unconditional release.
        Each phase stays independently bounded, so a wedged sender still cannot
        hang teardown -- at the cost of freeing live storage only in the
        pathological case where a genuine send outlives 2x the send timeout.
        """
        timeout_s = max(0.0, self._nccl_send_timeout_ms / 1000.0)
        lease_deadline = time.monotonic() + timeout_s
        while entry.inflight > 0 and time.monotonic() < lease_deadline:
            time.sleep(0.001)
        # Fresh budget for the recorded events to drain.
        deadline = time.monotonic() + timeout_s
        for event in entry.done_events:
            query = getattr(event, "query", None)
            if query is None:
                continue
            try:
                while not query():
                    if time.monotonic() >= deadline:
                        logger.warning(
                            "[NCCLRolloutMixin] send event for %s still pending "
                            "after %.1fs; releasing buffer anyway",
                            entry.transfer_id,
                            self._nccl_send_timeout_ms / 1000.0,
                        )
                        break
                    time.sleep(0.001)
            except Exception:  # pragma: no cover - teardown best-effort
                pass
        entry.buffer = None

    # ------------------------------------------------------------------
    # Pub/sub listener plumbing
    # ------------------------------------------------------------------

    def _start_listener(self, *, channel: str, handler, name: str) -> None:
        thread = threading.Thread(
            target=self._listen_loop,
            args=(channel, handler),
            name=name,
            daemon=True,
        )
        thread.start()
        self._nccl_threads.append(thread)

    def _listen_loop(self, channel: str, handler) -> None:
        try:
            pubsub = self._nccl_redis.pubsub()
            pubsub.subscribe(channel)
        except Exception as e:
            logger.warning("[NCCLRolloutMixin] cannot subscribe to %s: %s", channel, e)
            return
        while not self._nccl_shutdown.is_set():
            try:
                message = pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=0.2
                )
            except Exception as e:  # pragma: no cover - transient redis error
                logger.debug(
                    "[NCCLRolloutMixin] get_message error on %s: %s", channel, e
                )
                continue
            if not message:
                continue
            data = message.get("data")
            try:
                handler(data)
            except Exception as e:  # pragma: no cover - handler isolation
                logger.warning("[NCCLRolloutMixin] handler error on %s: %s", channel, e)
        try:
            pubsub.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def cleanup_nccl(self) -> None:
        """Stop threads, abort comms, and free all buffers."""
        if not self._nccl_enabled:
            return
        # 1. Stop the pub/sub listeners so no new requests are dispatched.
        self._nccl_shutdown.set()
        for thread in getattr(self, "_nccl_threads", []):
            thread.join(timeout=5.0)
        # 2. Abort comms BEFORE joining the sender pool.  ncclCommAbort forces
        #    any in-flight nccl_send (whose peer may have departed at job end)
        #    to stop, so the executor threads unblock.  Doing this *after* a
        #    ``shutdown(wait=True)`` would deadlock -- we'd wait for the very
        #    sends that only the abort can unwedge.
        if self._nccl_comm_cache is not None:
            self._nccl_comm_cache.abort_all()
        # 3. Now shut the pool down without blocking the worker's exit path;
        #    the finite send timeout + the abort above guarantee the threads
        #    terminate promptly.
        executor = getattr(self, "_nccl_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        # 4. A cancelled drain task leaves its pair's queue populated and those
        #    entries leased; release them before clearing the registry.
        self._abandon_queued_sends()
        if self._nccl_registry is not None:
            self._nccl_registry.clear()
        self._nccl_enabled = False
        logger.info(
            "[NCCLRolloutMixin] Worker '%s' cleaned up",
            getattr(self, "_nccl_replica_id", "?"),
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _torch_dtype(np_dtype) -> torch.dtype:
    mapping = {
        np.dtype("float32"): torch.float32,
        np.dtype("float64"): torch.float64,
        np.dtype("float16"): torch.float16,
        np.dtype("int64"): torch.int64,
        np.dtype("int32"): torch.int32,
        np.dtype("bool"): torch.bool,
        np.dtype("uint8"): torch.uint8,
    }
    return mapping[np.dtype(np_dtype)]
