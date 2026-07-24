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

"""Lazy 2-rank NCCL communicator cache for payload transfer.

Rather than a global mesh or reusing weight-sync's static comm, payload
transfer builds a **2-rank communicator per ``(sender_rank,
receiver_rank)`` pair**, lazily, on first transfer between that pair, and
caches it.  This is elastic-friendly: a dead replica only forces
``nccl_abort`` on its own pair comms, never a global rebuild.

Scaling controls
----------------
Comm count is ``O(rollout_ranks × trainer_ranks)`` and each comm costs
tens of MB + a QP, so the cache enforces:

* **Consumer-driven pair set** — only pairs that actually transfer get a
  comm (the cache is populated lazily by the receiver).
* **Live-comm cap + LRU eviction** — at most ``max_live`` comms; the
  least-recently-used is aborted when the cap is exceeded.
* **Bounded concurrent init** — a semaphore caps simultaneous
  ``create_nccl_comm`` calls to avoid init storms when many pairs warm up
  at once.

Health-aware quarantine
--------------------------------
A transient NCCL error on a ``(sender_replica, sender_rank)`` endpoint
quarantines it with a cooldown: :meth:`is_quarantined` returns ``True``
until the cooldown expires, so the receiver drops/retries the episode
next round instead of wedging on a dead sender.  The cooldown map is the
shared :class:`~cosmos_rl.utils.payload_transport.rotation.HealthSkipList`
(the same helper UCXX's ``_port_skip_until`` rotation uses).

Testability
-----------
``build_fn`` / ``abort_fn`` are injectable, so the whole cache — LRU,
semaphore accounting, quarantine cooldown — is unit-testable on CPU with
fakes, without a CUDA context.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.payload_transport.rotation import HealthSkipList

__all__ = ["PairKey", "CommCache"]

# A comm key identifies one cached 2-rank communicator.  It is keyed by the
# FULL endpoint identity on both replica axes, so the two sides key it
# differently (each on the *remote* peer's identity):
#   receiver (consumer): (sender_replica, sender_rank, receiver_rank)
#   sender   (producer): (sender_rank, receiver_replica, receiver_rank)
# The tuple is therefore heterogeneous (str/int) -- typed as an opaque tuple.
# An *endpoint* is a PREFIX of the key (e.g. the receiver's failed sender
# endpoint ``(sender_replica, sender_rank)``); ``abort_endpoint`` /
# ``quarantine`` prefix-match to tear down every comm owned by that endpoint.
PairKey = Tuple[Any, ...]

# Deterministic local-rank assignment inside the 2-rank comm: the sender is
# local rank 0, the receiver is local rank 1.  Both sides must agree.
SENDER_LOCAL_RANK = 0
RECEIVER_LOCAL_RANK = 1


def _default_build_fn(uid_chars: List[int], local_rank: int) -> int:
    # Imported lazily so this module (and its tests) do not require a CUDA
    # build of pynccl just to exercise the cache bookkeeping.
    from cosmos_rl.utils.pynccl import create_nccl_comm

    return create_nccl_comm(uid_chars, local_rank, 2)


def _default_abort_fn(comm_idx: int) -> None:
    from cosmos_rl.utils.pynccl import nccl_abort

    nccl_abort(comm_idx)


class CommCache:
    """Cache of lazily-built 2-rank communicators keyed by pair.

    Args:
        max_live: Maximum simultaneously-live comms (LRU eviction beyond).
        max_concurrent_init: Max in-flight ``build_fn`` calls.
        quarantine_cooldown: Seconds an endpoint stays quarantined after a
            transient failure.
        build_fn: ``(uid_chars, local_rank) -> comm_idx``.  Defaults to
            ``pynccl.create_nccl_comm(uid, local_rank, world_size=2)``.
        abort_fn: ``(comm_idx) -> None``.  Defaults to ``pynccl.nccl_abort``.
    """

    def __init__(
        self,
        *,
        max_live: int = 128,
        max_concurrent_init: int = 4,
        quarantine_cooldown: float = 30.0,
        build_fn: Optional[Callable[[List[int], int], int]] = None,
        abort_fn: Optional[Callable[[int], None]] = None,
    ) -> None:
        self._max_live = max(1, max_live)
        self._quarantine_cooldown = max(0.0, quarantine_cooldown)
        self._build_fn = build_fn or _default_build_fn
        self._abort_fn = abort_fn or _default_abort_fn

        # LRU-ordered pair -> comm_idx (most-recently-used at the end).
        self._comms: "OrderedDict[PairKey, int]" = OrderedDict()
        # Per-pair UID *generation* the cached comm was built from
        # (``tuple(uid_chars)``, or ``None`` when built without a uid on a
        # warm-reuse path).  Used to detect when the PEER rebuilt its half with
        # a fresh UID (e.g. after a quarantine) so we abort + rebuild ours in
        # lockstep instead of split-braining (peer on the new comm, us on the
        # stale one -> 600s init/collective hang).
        self._comm_uid: Dict[PairKey, Optional[Tuple[int, ...]]] = {}
        # Per-pair build lock so two threads racing on the same pair build
        # exactly one comm (the second waits and reuses).
        self._pair_locks: Dict[PairKey, threading.Lock] = {}
        self._lock = threading.Lock()
        self._init_sem = threading.BoundedSemaphore(max(1, max_concurrent_init))

        # Health-aware quarantine (shared skip-list; see rotation.py).
        self._skiplist = HealthSkipList(cooldown=self._quarantine_cooldown)

        self._n_built = 0
        self._n_evicted = 0

    # ------------------------------------------------------------------
    # Communicator lifecycle
    # ------------------------------------------------------------------

    def _pair_lock(self, pair: PairKey) -> threading.Lock:
        with self._lock:
            lk = self._pair_locks.get(pair)
            if lk is None:
                lk = threading.Lock()
                self._pair_locks[pair] = lk
            return lk

    def get(self, pair: PairKey) -> Optional[int]:
        """Return the cached comm_idx for ``pair`` (LRU-touch), or ``None``."""
        with self._lock:
            comm_idx = self._comms.get(pair)
            if comm_idx is not None:
                self._comms.move_to_end(pair)
            return comm_idx

    def _reuse_or_drop(
        self, pair: PairKey, fp: Optional[Tuple[int, ...]]
    ) -> Optional[int]:
        """Return the cached comm for ``pair`` if it is still CURRENT.

        Current means present AND built from a UID matching ``fp`` -- or ``fp``
        is ``None`` (the caller has no UID to check, i.e. a warm reuse).  If the
        comm is present but was built from a DIFFERENT UID than ``fp``, the peer
        rebuilt its half (post-quarantine renegotiation) and ours is stale:
        abort + drop it and return ``None`` so the caller rebuilds in lockstep.
        LRU-touches on a hit.
        """
        stale_idx: Optional[int] = None
        with self._lock:
            comm_idx = self._comms.get(pair)
            if comm_idx is None:
                return None
            stored = self._comm_uid.get(pair)
            if fp is not None and stored is not None and stored != fp:
                self._comms.pop(pair, None)
                self._comm_uid.pop(pair, None)
                stale_idx = comm_idx
            else:
                self._comms.move_to_end(pair)
                return comm_idx
        # Peer rebuilt with a fresh UID -> tear down our stale half (outside the
        # lock; abort may be slow) so the rebuild below joins the peer's comm.
        logger.warning(
            "[CommCache] pair=%s UID changed (peer rebuilt); aborting stale comm "
            "idx=%s and rebuilding in lockstep",
            pair,
            stale_idx,
        )
        self._safe_abort(stale_idx)
        return None

    def get_or_create(
        self,
        pair: PairKey,
        *,
        uid_chars: List[int],
        local_rank: int,
    ) -> int:
        """Return the comm for ``pair``, building it under the init semaphore.

        Concurrent callers for the *same* pair build exactly one comm; the
        loser waits on the per-pair lock and reuses the winner's comm.
        Different pairs build concurrently up to ``max_concurrent_init``.

        A non-empty ``uid_chars`` that differs from the UID the cached comm was
        built with forces an abort + rebuild (peer-rebuilt / comm-generation
        recovery) rather than silently reusing the stale half.
        """
        fp = tuple(uid_chars) if uid_chars else None

        cached = self._reuse_or_drop(pair, fp)
        if cached is not None:
            return cached

        pair_lock = self._pair_lock(pair)
        with pair_lock:
            # Re-check under the pair lock (another thread may have built it).
            cached = self._reuse_or_drop(pair, fp)
            if cached is not None:
                return cached

            # Invariant: NEVER build from an empty unique-ID.  A 2-rank NCCL
            # comm built from an all-zero UID desyncs against a peer using the
            # real UID and wedges in ``create_nccl_comm`` for the full 600s
            # NCCL init watchdog (NOT the shorter send timeout).  This is the
            # atomic guard for the producer's UID TOCTOU: the UID validated
            # before ACCEPTED can still expire / be overwritten before we build
            # here.  Fail fast instead so the caller quarantines + the receiver
            # renegotiates a fresh UID on its recv-timeout retry.
            if not uid_chars:
                raise ValueError(
                    f"refusing to build NCCL comm for pair={pair} from an empty "
                    "unique-ID (unreadable/expired UID) -- renegotiate a fresh UID"
                )

            with self._init_sem:
                comm_idx = self._build_fn(uid_chars, local_rank)

            with self._lock:
                self._comms[pair] = comm_idx
                self._comm_uid[pair] = fp
                self._comms.move_to_end(pair)
                self._n_built += 1
                evicted = self._evict_if_needed_locked()
            # Abort evicted comms OUTSIDE the lock -- a slow/hung nccl_abort must
            # not freeze the whole cache (get/stats/other builds), matching every
            # other abort path here (_reuse_or_drop/abort/abort_endpoint/abort_all).
            for old_idx in evicted:
                self._safe_abort(old_idx)
            logger.debug(
                "[CommCache] Built comm idx=%s for pair=%s (local_rank=%d, live=%d)",
                comm_idx,
                pair,
                local_rank,
                len(self._comms),
            )
            return comm_idx

    def _evict_if_needed_locked(self) -> List[int]:
        """Pop LRU comms until within ``max_live``; return their idxs to abort.

        Caller holds ``self._lock``.  Popping/bookkeeping happens under the lock,
        but the actual ``nccl_abort`` is deferred to the caller AFTER the lock is
        released (see the sibling abort paths) so a hung abort can't wedge the
        cache for every other thread.
        """
        evicted: List[int] = []
        while len(self._comms) > self._max_live:
            old_pair, old_idx = self._comms.popitem(last=False)
            self._comm_uid.pop(old_pair, None)
            self._n_evicted += 1
            logger.debug(
                "[CommCache] Evicting LRU comm idx=%s pair=%s (cap=%d)",
                old_idx,
                old_pair,
                self._max_live,
            )
            evicted.append(old_idx)
        return evicted

    def abort(self, pair: PairKey) -> bool:
        """Abort + drop the comm for ``pair``.  Idempotent."""
        with self._lock:
            comm_idx = self._comms.pop(pair, None)
            self._comm_uid.pop(pair, None)
        if comm_idx is None:
            return False
        self._safe_abort(comm_idx)
        return True

    def abort_endpoint(self, prefix: Tuple) -> int:
        """Abort + drop every cached comm whose key starts with ``prefix``.

        A failed endpoint is named by a *prefix* of the comm key(s) it owns:
        a receiver's sender endpoint ``(sender_replica, sender_rank)`` is a
        prefix of its ``(sender_replica, sender_rank, receiver_rank)`` comm;
        a sender passes the full pair.  Prefix-matching tears the wedged
        communicator(s) down so a dead handle is never reused after a
        transient failure.  Returns the number aborted.
        """
        if not isinstance(prefix, tuple):
            return 0
        n = len(prefix)
        with self._lock:
            matched = [
                p for p in self._comms if isinstance(p, tuple) and p[:n] == prefix
            ]
            idxs = [self._comms.pop(p) for p in matched]
            for p in matched:
                self._comm_uid.pop(p, None)
        for comm_idx in idxs:
            self._safe_abort(comm_idx)
        return len(idxs)

    def abort_all(self) -> int:
        """Abort every cached comm (teardown).  Returns count aborted."""
        with self._lock:
            items = list(self._comms.items())
            self._comms.clear()
            self._comm_uid.clear()
        for _pair, comm_idx in items:
            self._safe_abort(comm_idx)
        return len(items)

    def _safe_abort(self, comm_idx: int) -> None:
        try:
            self._abort_fn(comm_idx)
        except Exception as exc:  # pragma: no cover - teardown best-effort
            logger.debug(
                "[CommCache] abort_fn raised %s for comm idx=%s; ignoring",
                type(exc).__name__,
                comm_idx,
            )

    # ------------------------------------------------------------------
    # Health-aware quarantine
    # ------------------------------------------------------------------

    def quarantine(self, health_key: Any, *, cooldown: Optional[float] = None) -> None:
        """Quarantine ``health_key`` for ``cooldown`` seconds (default cooldown).

        Also aborts every live comm owned by this endpoint -- ``health_key``
        is a *prefix* of its comm key(s) (see :meth:`abort_endpoint`) -- so a
        wedged communicator is torn down rather than reused when the cooldown
        lifts.  (Previously this only matched exact 2-tuples, which never hit
        the real 3-tuple comm keys, leaving dead comms cached.)
        """
        cd = self._quarantine_cooldown if cooldown is None else max(0.0, cooldown)
        self._skiplist.quarantine(health_key, cooldown=cd)
        logger.warning(
            "[CommCache] Quarantining endpoint %s for %.1fs after transient "
            "NCCL failure",
            health_key,
            cd,
        )
        if isinstance(health_key, tuple):
            self.abort_endpoint(health_key)

    def is_quarantined(self, health_key: Any) -> bool:
        """Return ``True`` while ``health_key`` is within its cooldown."""
        return self._skiplist.is_quarantined(health_key)

    def clear_quarantine(self, health_key: Any) -> None:
        self._skiplist.clear(health_key)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._comms)

    def __contains__(self, pair: PairKey) -> bool:
        with self._lock:
            return pair in self._comms

    @property
    def max_live(self) -> int:
        return self._max_live

    def stats(self) -> dict:
        with self._lock:
            return {
                "live": len(self._comms),
                "max_live": self._max_live,
                "built": self._n_built,
                "evicted": self._n_evicted,
                "quarantined": len(self._skiplist),
            }
