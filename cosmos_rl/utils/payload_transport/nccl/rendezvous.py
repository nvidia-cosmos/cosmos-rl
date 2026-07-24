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

"""Per-transfer NCCL rendezvous over the Redis control plane.

Why not a Lua CAS module
------------------------
The UCXX design used a ``rendezvous.py`` built on a Redis Lua
compare-and-set to elect a single winner among contending readers.  NCCL
payload transfer does **not** need that: each ``transfer_id`` has exactly
one intended receiver, so there is no multi-winner election to solve.  A
plain request / ack with **bilateral timeouts** suffices and is much
simpler to reason about (and to fake in tests).

Protocol (receiver-driven)
--------------------------
The receiver (trainer rank) drives each transfer::

    receiver                                   sender (rollout rank)
    ────────                                   ─────────────────────
    initiate(transfer_id, sender_rank):
      if comm not cached for the pair:
        uid = create_nccl_uid()
        redis.SET  pair_uid_key = uid          serve loop: on :nccl_req msg
      redis.DEL  resp_key (clear stale)   ───►   respond(...):
      redis.PUBLISH :nccl_req {req}                if buffer present:
      poll resp_key up to `timeout`:                 redis.SET resp_key=ACCEPTED
        ACCEPTED  -> build 2-rank comm  ◄──────      (both build comm, nccl_send/recv)
        MISSING   -> drop episode (recycled) ◄──     else: redis.SET resp_key=MISSING
        (no reply within timeout) -> CANCELLED

Three states: a request is implicitly ``REQUESTED``; the sender resolves
it to ``ACCEPTED`` or ``MISSING``; a receiver-side timeout yields
``CANCELLED``.  All three are terminal and idempotent — a duplicate or
late reply is ignored because the receiver clears ``resp_key`` before
each attempt and consumes (deletes) it on read.

Testability
-----------
The only Redis surface used is ``set`` / ``get`` / ``delete`` /
``publish`` (all with string values), and ``create_nccl_uid`` is
injectable, so the whole handshake — including timeout / missing / accept
paths — is unit-testable against a tiny in-memory fake Redis with no
CUDA.
"""

from __future__ import annotations

import enum
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.payload_transport.nccl.protocol import (
    build_pair_uid_key,
    build_response_key,
)

__all__ = [
    "TransferStatus",
    "RendezvousResult",
    "NcclRendezvous",
    "build_request_message",
    "parse_request_message",
]


class TransferStatus(str, enum.Enum):
    """Outcome of one per-transfer rendezvous.

    ``ACCEPTED`` / ``MISSING`` / ``CANCELLED`` are terminal; ``NEED_UID`` is
    a *retry* signal: the sender evicted its side of the comm, so the
    receiver must drop its (now half-open) cached comm and re-initiate with a
    fresh unique-ID rather than waiting forever on a comm the sender will
    never rejoin.
    """

    ACCEPTED = "accepted"
    MISSING = "missing"
    CANCELLED = "cancelled"
    NEED_UID = "need_uid"


@dataclass
class RendezvousResult:
    """Result of :meth:`NcclRendezvous.initiate`.

    Attributes:
        status: One of :class:`TransferStatus`.
        uid_chars: The NCCL unique-ID bytes to build the pair comm with,
            when the receiver had to mint one (``None`` when the comm was
            already cached and no exchange happened).
    """

    status: TransferStatus
    uid_chars: Optional[List[int]] = None

    @property
    def accepted(self) -> bool:
        return self.status is TransferStatus.ACCEPTED


def build_request_message(
    *,
    transfer_id: str,
    sender_rank: int,
    receiver_replica: Optional[str],
    receiver_rank: int,
    resp_key: str,
    uid_key: Optional[str],
    req_deadline: Optional[float] = None,
) -> str:
    """Serialize a transfer request published on the ``:nccl_req`` channel.

    ``receiver_replica`` is the requesting policy replica's globally-unique
    identity; the producer keys its comm cache by it so two policy replicas
    sharing ``receiver_rank`` do not cross-wire.

    ``req_deadline`` is the absolute wall-clock time after which the receiver
    stops waiting.  The producer drops any request it dequeues past this
    deadline instead of sending a late ACCEPTED + launching an unmatched send
    (bilateral cancellation -- prevents the executor-queue backlog from
    starving the sender pool under high policy-replica fan-out).
    """
    return json.dumps(
        {
            "transfer_id": transfer_id,
            "sender_rank": sender_rank,
            "receiver_replica": receiver_replica,
            "receiver_rank": receiver_rank,
            "resp_key": resp_key,
            "uid_key": uid_key,
            "req_deadline": req_deadline,
        }
    )


def parse_request_message(raw: Any) -> Optional[Dict[str, Any]]:
    """Parse a request message; return ``None`` if malformed."""
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    if not isinstance(raw, str):
        return None
    try:
        msg = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(msg, dict) or "transfer_id" not in msg:
        return None
    return msg


def _default_uid_fn() -> List[int]:
    from cosmos_rl.utils.pynccl import create_nccl_uid

    return create_nccl_uid()


class NcclRendezvous:
    """Receiver- and sender-side helpers for the per-transfer handshake.

    Args:
        redis_client: A connected Redis client (``decode_responses=True``
            recommended; the parser tolerates bytes either way).
        prefix: The rollout-replica Redis prefix
            (``build_rollout_prefix(build_nccl_prefix(...), idx)``).
        poll_interval: Receiver poll granularity (seconds) while waiting
            on the sender's reply.
        uid_ttl_s: Expiry on the per-pair unique-ID key so a crashed
            transfer cannot leave a stale UID around forever.
        resp_ttl_s: Expiry on the response key (defensive GC).
        uid_fn: ``() -> uid_chars``.  Defaults to
            ``pynccl.create_nccl_uid``; injectable for tests.
        clock: ``() -> float`` monotonic clock (injectable for tests).
        sleep: ``(seconds) -> None`` (injectable for tests).
    """

    def __init__(
        self,
        redis_client: Any,
        prefix: str,
        *,
        poll_interval: float = 0.01,
        uid_ttl_s: int = 60,
        resp_ttl_s: int = 60,
        uid_fn: Optional[Callable[[], List[int]]] = None,
        clock: Optional[Callable[[], float]] = None,
        sleep: Optional[Callable[[float], None]] = None,
        wall_clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self._redis = redis_client
        self._prefix = prefix
        self._poll_interval = max(1e-4, poll_interval)
        self._uid_ttl_s = uid_ttl_s
        self._resp_ttl_s = resp_ttl_s
        self._uid_fn = uid_fn or _default_uid_fn
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        # WALL-clock (not monotonic): stamped into each request as an absolute
        # deadline the producer can compare against.  Monotonic clocks are not
        # comparable across processes, so the receiver's own poll loop uses
        # ``_clock`` (monotonic) while the cross-process request deadline uses
        # ``_wall_clock``.  Assumes NTP-synced cluster clocks (exact on one node).
        self._wall_clock = wall_clock or time.time

    # ------------------------------------------------------------------
    # Receiver side
    # ------------------------------------------------------------------

    def initiate(
        self,
        *,
        transfer_id: str,
        sender_replica: str,
        sender_rank: int,
        receiver_replica: str,
        receiver_rank: int,
        request_channel: str,
        need_uid: bool,
        timeout: float,
        attempt: int = 0,
    ) -> RendezvousResult:
        """Publish a request and wait (bilaterally bounded) for the reply.

        Args:
            transfer_id: The transfer being requested.
            sender_replica: The rollout replica's globally-unique identity;
                keys the per-pair UID so distinct replicas (which may share
                ``sender_rank``) never share a UID key.
            receiver_replica: This policy replica's globally-unique identity;
                keys the per-pair UID (and, sender-side, the comm cache) so
                distinct policy replicas that share ``receiver_rank`` never
                cross-wire.
            sender_rank / receiver_rank: Ranks within their replicas; the
                local-rank assignment in the 2-rank comm.
            request_channel: The sender replica's ``:nccl_req`` channel.
            need_uid: ``True`` when the pair comm is not yet cached and a
                fresh unique-ID must be minted + published for the sender.
            timeout: Seconds to wait for the sender's reply before
                returning :attr:`TransferStatus.CANCELLED`.
            attempt: Retry generation (1-based from the caller's retry loop).
                Scopes the response key so a late reply from an abandoned
                earlier attempt cannot be consumed as this attempt's result.
        """
        resp_key = build_response_key(
            self._prefix, transfer_id, receiver_replica, receiver_rank, attempt
        )
        uid_key: Optional[str] = None
        uid_chars: Optional[List[int]] = None

        # Clear any stale reply from a prior attempt so a late duplicate
        # cannot be mistaken for this attempt's result.
        self._safe_delete(resp_key)

        if need_uid:
            uid_key = build_pair_uid_key(
                self._prefix,
                sender_replica,
                sender_rank,
                receiver_replica,
                receiver_rank,
            )
            uid_chars = self._uid_fn()
            self._safe_set(uid_key, json.dumps(uid_chars), ex=self._uid_ttl_s)

        message = build_request_message(
            transfer_id=transfer_id,
            sender_rank=sender_rank,
            receiver_replica=receiver_replica,
            receiver_rank=receiver_rank,
            resp_key=resp_key,
            uid_key=uid_key,
            req_deadline=self._wall_clock() + max(0.0, timeout),
        )
        try:
            self._redis.publish(request_channel, message)
        except Exception as exc:
            logger.warning(
                "[NcclRendezvous] publish request failed for %s: %s",
                transfer_id,
                exc,
            )
            return RendezvousResult(TransferStatus.CANCELLED, uid_chars)

        deadline = self._clock() + max(0.0, timeout)
        while True:
            reply = self._consume_reply(resp_key)
            if reply is not None:
                return RendezvousResult(reply, uid_chars)
            if self._clock() >= deadline:
                logger.debug(
                    "[NcclRendezvous] transfer %s timed out after %.3fs; cancelling",
                    transfer_id,
                    timeout,
                )
                # The published UID key is left to expire via ``uid_ttl_s``
                # (no explicit delete); a racing sender read is harmless.
                return RendezvousResult(TransferStatus.CANCELLED, uid_chars)
            self._sleep(self._poll_interval)

    def _consume_reply(self, resp_key: str) -> Optional[TransferStatus]:
        raw = self._safe_get(resp_key)
        if raw is None:
            return None
        self._safe_delete(resp_key)
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="replace")
        try:
            return TransferStatus(raw)
        except ValueError:
            logger.warning(
                "[NcclRendezvous] unrecognized reply %r; treating as missing", raw
            )
            return TransferStatus.MISSING

    # ------------------------------------------------------------------
    # Sender side
    # ------------------------------------------------------------------

    def respond(self, *, resp_key: str, status: TransferStatus) -> None:
        """Write the sender's terminal reply for a request.

        Idempotent from the receiver's perspective: the receiver clears
        ``resp_key`` before each attempt and deletes on read, so a stale
        reply from an abandoned attempt is discarded.
        """
        self._safe_set(resp_key, status.value, ex=self._resp_ttl_s)

    def read_uid(self, uid_key: Optional[str]) -> Optional[List[int]]:
        """Sender-side: read the pair unique-ID the receiver published."""
        if not uid_key:
            return None
        raw = self._safe_get(uid_key)
        if raw is None:
            return None
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="replace")
        try:
            uid = json.loads(raw)
        except (ValueError, TypeError):
            return None
        if isinstance(uid, list):
            return [int(x) for x in uid]
        return None

    # ------------------------------------------------------------------
    # Redis calls, wrapped so a transient error degrades gracefully
    # ------------------------------------------------------------------

    def _safe_get(self, key: str) -> Any:
        try:
            return self._redis.get(key)
        except Exception as exc:
            logger.debug("[NcclRendezvous] GET %s failed: %s", key, exc)
            return None

    def _safe_set(self, key: str, value: str, *, ex: Optional[int] = None) -> None:
        try:
            self._redis.set(key, value, ex=ex)
        except Exception as exc:
            logger.debug("[NcclRendezvous] SET %s failed: %s", key, exc)

    def _safe_delete(self, key: str) -> None:
        try:
            self._redis.delete(key)
        except Exception as exc:
            logger.debug("[NcclRendezvous] DEL %s failed: %s", key, exc)
