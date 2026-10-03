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

Each invocation owns a unique operation ID and persistent bounded state:
REQUESTED -> ACCEPTED -> COMPLETE/FAILED, or REQUESTED -> MISSING/NEED_UID/
CANCELLED. Redis compare-and-set seals acceptance against cancellation and
duplicate delivery. A lost publication reply is resolved against that same
operation, not blindly replayed. Accepted work carries its immutable UID and
original lifetime through queueing, initialization and device completion.

ACCEPTED is a promise, not success. Failed or ambiguous accepted work is terminal
for the worker; ordinary pre-accept rejection can still drop a missing episode.
A batched peer-outcome observer works while native calls block; independent hard
timers do not depend on Redis progress. Tests include real Redis races.
"""

from __future__ import annotations

import enum
import json
import math
import time
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.transport_failure import TransportUnusableError
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

    ``MISSING`` / ``CANCELLED`` reject before native work; ``NEED_UID`` is
    a *retry* signal: the sender evicted its side of the comm, so the
    receiver must drop its (now half-open) cached comm and re-initiate with a
    fresh unique-ID rather than waiting forever on a comm the sender will
    never rejoin.
    """

    ACCEPTED = "accepted"
    MISSING = "missing"
    CANCELLED = "cancelled"
    NEED_UID = "need_uid"
    REQUESTED = "requested"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class RendezvousResult:
    """Result of :meth:`NcclRendezvous.initiate`.

    Attributes:
        status: One of :class:`TransferStatus`.
        uid_chars: The NCCL unique-ID bytes to build the pair comm with,
            when the receiver had to mint one (``None`` when the comm was
            already cached and no exchange happened).
        late_accept: Set with ``CANCELLED`` when the sender's ``ACCEPTED``
            landed after the deadline had already passed.  The transfer is
            not replayable: the sender may owe an unmatched send, so the
            caller must terminate rather than reuse the ordered stream.
    """

    status: TransferStatus
    uid_chars: Optional[List[int]] = None
    late_accept: bool = False
    response_key: Optional[str] = None
    deadline: Optional[float] = None

    @property
    def accepted(self) -> bool:
        return self.status in (TransferStatus.ACCEPTED, TransferStatus.COMPLETE)


def build_request_message(
    *,
    transfer_id: str,
    sender_rank: int,
    receiver_replica: Optional[str],
    receiver_rank: int,
    resp_key: str,
    uid_key: Optional[str],
    req_deadline: Optional[float] = None,
    req_timeout: Optional[float] = None,
    uid_chars: Optional[List[int]] = None,
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

    ``req_timeout`` is the receiver's whole budget for this attempt.  It lets
    the producer scale its accept-margin to the budget rather than apply a flat
    one, so a deployment running very short timeouts is not left with every
    request landing inside the margin and nothing ever served.
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
            "req_timeout": req_timeout,
            "uid_chars": uid_chars,
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
        request_id_fn: Optional[Callable[[], str]] = None,
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
        self._request_id_fn = request_id_fn or (lambda: uuid.uuid4().hex)
        self._watch_lock = threading.Lock()
        self._watched = {}
        self._watcher = None

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
        request_id = self._request_id_fn()
        if request_id:
            resp_key += ":" + request_id
        uid_key: Optional[str] = None
        uid_chars: Optional[List[int]] = None
        deadline = self._clock() + max(0.0, timeout)
        # Persist state past the whole attempt. A receiver must atomically
        # cancel REQUESTED, not assume a missing/lost acknowledgement means
        # the sender did not accept. Every invocation has a fresh operation ID.
        ttl = max(self._resp_ttl_s, math.ceil(max(0.0, timeout)) + 60)
        self._redis.set(resp_key, TransferStatus.REQUESTED.value, ex=ttl)

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
            req_timeout=max(0.0, timeout),
            uid_chars=uid_chars,
        )
        try:
            self._redis.publish(request_channel, message)
        except Exception as exc:
            logger.warning(
                "[NcclRendezvous] publish request failed for %s: %s",
                transfer_id,
                exc,
            )
            # Publication may have committed despite a lost reply. Resolve the
            # same operation; do not blindly retry or start a different stream.
            pass

        while True:
            reply = self._consume_reply(resp_key)
            if reply is not None:
                if reply is TransferStatus.FAILED:
                    raise TransportUnusableError(
                        f"Accepted sender failed: {transfer_id}"
                    )
                return RendezvousResult(
                    reply, uid_chars, response_key=resp_key, deadline=deadline
                )
            if self._clock() >= deadline:
                # One last look before giving up.  A reply that landed between
                # the poll above and this check would otherwise be left in
                # Redis while the sender launches its send. An unmatched send
                # desynchronises the ordered stream; report ambiguity so the
                # caller terminates instead of mispairing the next payload.
                try:
                    cancelled = self.respond(
                        resp_key=resp_key, status=TransferStatus.CANCELLED
                    )
                except Exception as exc:
                    raise TransportUnusableError(
                        "Cannot establish whether payload request was accepted"
                    ) from exc
                late = None if cancelled else self._consume_reply(resp_key)
                logger.debug(
                    "[NcclRendezvous] transfer %s timed out after %.3fs; "
                    "cancelling (late reply: %s)",
                    transfer_id,
                    timeout,
                    late.value if late is not None else "none",
                )
                # The published UID key is left to expire via ``uid_ttl_s``
                # (no explicit delete); a racing sender read is harmless.
                return RendezvousResult(
                    TransferStatus.CANCELLED,
                    uid_chars,
                    late_accept=not cancelled
                    and late
                    not in (
                        TransferStatus.MISSING,
                        TransferStatus.NEED_UID,
                        TransferStatus.CANCELLED,
                    ),
                    response_key=resp_key,
                    deadline=deadline,
                )
            self._sleep(self._poll_interval)

    def _consume_reply(self, resp_key: str) -> Optional[TransferStatus]:
        raw = self._safe_get(resp_key)
        if raw is None:
            return None
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="replace")
        try:
            status = TransferStatus(raw)
            return None if status is TransferStatus.REQUESTED else status
        except ValueError:
            raise TransportUnusableError(f"Invalid transfer outcome: {raw!r}")

    # ------------------------------------------------------------------
    # Sender side
    # ------------------------------------------------------------------

    def respond(self, *, resp_key: str, status: TransferStatus) -> bool:
        """Atomically advance one operation, returning whether this call won."""
        expected = (
            TransferStatus.ACCEPTED
            if status in (TransferStatus.COMPLETE, TransferStatus.FAILED)
            else TransferStatus.REQUESTED
        )
        # Keep the original whole-operation TTL. Duplicate requests/replies
        # cannot accept twice or overwrite cancellation/terminal completion.
        return bool(
            self._redis.eval(
                "if redis.call('GET', KEYS[1]) == ARGV[1] then "
                "redis.call('SET', KEYS[1], ARGV[2], 'KEEPTTL'); return 1 end; return 0",
                1,
                resp_key,
                expected.value,
                status.value,
            )
        )

    def check_failure(self, response_key):
        if response_key and self._consume_reply(response_key) is TransferStatus.FAILED:
            raise TransportUnusableError("Peer reported accepted transfer failure")

    def watch_operation(self, response_key, operation):
        """Observe peer failure even while the native caller is blocked.

        One batched observer per rendezvous, not one Redis polling thread per
        payload. Its I/O can never delay the operation's independent hard timer.
        Closed operations are pruned and the daemon exits when no work remains.
        """
        with self._watch_lock:
            self._watched[response_key] = operation
            if self._watcher is None:
                self._watcher = threading.Thread(
                    target=self._watch_operations,
                    daemon=True,
                    name="payload-peer-outcomes",
                )
                self._watcher.start()

    def _watch_operations(self):
        while True:
            with self._watch_lock:
                self._watched = {
                    key: op for key, op in self._watched.items() if op.active
                }
                if not self._watched:
                    self._watcher = None
                    return
                pending = list(self._watched.items())
            try:
                outcomes = self._redis.mget([key for key, _op in pending])
            except Exception:
                outcomes = ()  # each operation still has its independent deadline
            for (_key, op), outcome in zip(pending, outcomes):
                if outcome in (
                    TransferStatus.FAILED.value,
                    TransferStatus.FAILED.value.encode(),
                ):
                    op.fail("peer reported accepted transfer failure")
            time.sleep(0.05)

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
