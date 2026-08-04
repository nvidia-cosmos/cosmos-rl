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

"""NCCL payload-transport wire protocol: Redis key/channel builders.

Split out of the former single-file ``nccl.py`` when it became a package
so the pure string-protocol helpers can be imported without pulling in
the transport backend or the pynccl-dependent rendezvous / comm-cache
modules.  Every name here is re-exported from
:mod:`cosmos_rl.utils.payload_transport.nccl` for backward compatibility.

Redis key convention
--------------------
All keys are scoped by ``{namespace}:{experiment_name}:{slurm_job_id}``
so multiple jobs sharing a Redis instance do not collide.
"""

from __future__ import annotations

import numbers
from typing import Any, List, Optional, Union

# ---------------------------------------------------------------------------
# Wire-protocol constants and key builders
# ---------------------------------------------------------------------------

NCCL_REDIS_NAMESPACE = "cosmos_rl"

NCCL_COMPLETION_PREFIX = "nccl:"


def build_nccl_prefix(*, experiment_name: str, job_id: str) -> str:
    """Root prefix for all NCCL transfer Redis keys."""
    return f"{NCCL_REDIS_NAMESPACE}:{experiment_name}:{job_id}"


def build_rollout_prefix(prefix: str, rollout_key: Union[int, str]) -> str:
    """Per-rollout-replica prefix.

    ``rollout_key`` is either the integer rollout index (controller-side
    cleanup routing, which parses it back from the transfer id) or the
    globally-unique ``sender_replica`` string (producer/consumer request
    routing).  Both are embedded verbatim.
    """
    return f"{prefix}:rollout_comm:{rollout_key}"


def build_cleanup_channel(prefix: str) -> str:
    """Pub/sub channel for cleanup messages."""
    return f"{prefix}:nccl_cleanup"


def build_request_channel(prefix: str) -> str:
    """Pub/sub channel for transfer requests."""
    return f"{prefix}:nccl_req"


def build_response_key(
    prefix: str,
    transfer_id: str,
    receiver_replica: str = "",
    receiver_rank: int = 0,
    attempt: int = 0,
) -> str:
    """Per-transfer response key the receiver polls for the sender's ack.

    The request/response handshake (``rendezvous.py``) rides the
    ``:nccl_req`` channel for the request and this short-lived Redis key
    for the sender's reply (``accepted`` / ``missing`` / ``cancelled``).

    Scoped by the receiver identity because, with TP/PP/CP policy
    parallelism, several ranks sharing a DP id request the SAME
    ``transfer_id``; an unscoped key would let one rank consume another's
    ack while the others time out.

    Also scoped by ``attempt`` (the retry generation) so a delayed reply
    from an abandoned earlier attempt cannot be mis-read as the current
    attempt's result -- each retry polls a distinct key that only the
    sender handling *that* attempt's request writes to.
    """
    return (
        f"{prefix}:nccl_resp:{transfer_id}:{receiver_replica}:{receiver_rank}:{attempt}"
    )


def build_pair_uid_key(
    prefix: str,
    sender_replica: str,
    sender_rank: int,
    receiver_replica: str,
    receiver_rank: int,
) -> str:
    """Redis key holding the NCCL unique-ID for a sender/receiver pair.

    The receiver (which drives the transfer) generates the unique-ID,
    writes it here, and requests the sender to join; both sides build the
    same deterministic key so the 2-rank communicator can be created
    without a separate rendezvous service.

    The key is fully identified by BOTH replica axes so no two physical
    endpoints ever share it:

    * ``sender_replica`` — the rollout replica's globally-unique identity
      (e.g. its ``replica_name``).  REQUIRED with >1 rollout replica:
      ``sender_rank`` alone is the rank *within* a replica (0 for
      single-GPU replicas), so two replicas would otherwise share a key.
    * ``receiver_replica`` — the policy replica's globally-unique identity.
      REQUIRED with >1 policy replica: each policy replica is a separate
      distributed world, so ``receiver_rank`` (== ``dist.get_rank()``)
      restarts at 0 per replica; two replicas would otherwise collide.
    """
    return (
        f"{prefix}:nccl_uid:{sender_replica}:{sender_rank}"
        f":{receiver_replica}:{receiver_rank}"
    )


def build_sender_request_channel(prefix: str, ref: dict) -> str:
    """`:nccl_req` channel for a specific sender, routed by its replica id.

    Routing the request by the globally-unique ``sender_replica`` ensures a
    transfer request reaches exactly ONE producer.  Keyed on ``sender_rank``
    alone, two rollout replicas would subscribe to the same channel and
    race on the shared response key.
    """
    return build_request_channel(build_rollout_prefix(prefix, ref["sender_replica"]))


def parse_transfer_rollout_idx(transfer_id: str) -> int:
    """Extract the rollout index encoded in the transfer ID prefix."""
    if ":" not in transfer_id:
        return -1
    prefix = transfer_id.split(":", maxsplit=1)[0]
    try:
        return int(prefix)
    except ValueError:
        return -1


def build_transfer_rollout_candidates(*, transfer_id: str) -> List[int]:
    """Return the canonical rollout index encoded in ``transfer_id``, if valid.

    Deliberately validates the parse ONLY -- it does not bound the index against
    a replica count.  It used to be capped by ``n_init_replicas``, which silently
    dropped every transfer belonging to a replica added mid-run (elastic
    scale-up): no cleanup was ever published for it, so the producer held its
    send buffer until capacity eviction.  The trade is asymmetric -- publishing
    to a channel nobody subscribes to is a no-op, while withholding a publish
    pins GPU memory -- so no upper bound is the correct behaviour.
    """
    parsed_prefix = parse_transfer_rollout_idx(transfer_id)
    if parsed_prefix < 0:
        return []
    return [parsed_prefix]


def _coerce_nonnegative_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Integral):
        coerced = int(value)
    elif isinstance(value, str):
        try:
            coerced = int(value)
        except ValueError:
            return None
    else:
        return None
    if coerced < 0:
        return None
    return coerced


__all__ = [
    "NCCL_COMPLETION_PREFIX",
    "NCCL_REDIS_NAMESPACE",
    "build_cleanup_channel",
    "build_nccl_prefix",
    "build_pair_uid_key",
    "build_request_channel",
    "build_response_key",
    "build_rollout_prefix",
    "build_sender_request_channel",
    "build_transfer_rollout_candidates",
    "parse_transfer_rollout_idx",
]
