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

"""NCCL payload-transport backend.

Background
----------
By default, Cosmos-RL transfers rollout completion payloads (token IDs,
log-probs, etc.) from rollout workers to the controller via Redis
streams.  For workloads with large payloads — such as VLA policies that
produce high-dimensional action tensors — the Redis path becomes a
bottleneck.

**NCCL payload transfer** is an opt-in alternative where the payload
tensors are sent directly between GPUs using NCCL point-to-point
operations.  Redis is still used as the *control plane*: rollout workers
publish transfer requests, the controller acknowledges them, and cleanup
messages are sent when stale transfers are discarded.

How it works
------------
1. A data packer that supports NCCL transfer (e.g.
   ``TensorDataPacker`` in downstream user code) exposes a
   ``redis_client`` attribute.  ``CommMixin.init_data_packer``
   injects the worker's live Redis connection into the packer after
   creation.  The packer then calls ``post_redis_injection()`` to
   complete any deferred setup (e.g. establishing NCCL communicators).

2. Rollout completions transferred via NCCL have their ``completion``
   field prefixed with ``nccl:`` followed by a transfer ID.

3. When the controller discards outdated rollouts
   (``PolicyStatusManager.filter_outdated_rollouts``), it dispatches
   to :meth:`NcclPayloadTransport.publish_cleanup_for_discarded`
   (via the registry) so the rollout worker can release the associated
   GPU buffers immediately.

Enabling
--------
Set ``[custom].payload_transfer = "nccl"`` in the experiment config.
The legacy ``[custom].nccl_payload_transfer = true`` boolean still works
as a deprecated alias.  The data packer must support this feature
(see ``BaseDataPacker`` docs in the user guide).

Redis key convention
--------------------
All keys are scoped by ``{namespace}:{experiment_name}:{slurm_job_id}``
so multiple jobs sharing a Redis instance do not collide.
"""

from __future__ import annotations

import json
import numbers
import os
import time
from typing import Any, List, Optional

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.payload_transport.registry import (
    PayloadTransport,
    PayloadTransportRegistry,
)


# ---------------------------------------------------------------------------
# Wire-protocol constants and key builders
# ---------------------------------------------------------------------------

NCCL_REDIS_NAMESPACE = "cosmos_rl"

NCCL_COMPLETION_PREFIX = "nccl:"


def build_nccl_prefix(*, experiment_name: str, job_id: str) -> str:
    """Root prefix for all NCCL transfer Redis keys."""
    return f"{NCCL_REDIS_NAMESPACE}:{experiment_name}:{job_id}"


def build_rollout_prefix(prefix: str, rollout_idx: int) -> str:
    """Per-rollout-replica prefix."""
    return f"{prefix}:rollout_comm:{rollout_idx}"


def build_cleanup_channel(prefix: str) -> str:
    """Pub/sub channel for cleanup messages."""
    return f"{prefix}:nccl_cleanup"


def build_request_channel(prefix: str) -> str:
    """Pub/sub channel for transfer requests."""
    return f"{prefix}:nccl_req"


def parse_transfer_rollout_idx(transfer_id: str) -> int:
    """Extract the rollout index encoded in the transfer ID prefix."""
    if ":" not in transfer_id:
        return -1
    prefix = transfer_id.split(":", maxsplit=1)[0]
    try:
        return int(prefix)
    except ValueError:
        return -1


def build_transfer_rollout_candidates(
    *,
    transfer_id: str,
    num_rollout_replicas: Optional[int] = None,
) -> List[int]:
    """Return the canonical rollout index encoded in ``transfer_id``, if valid."""
    normalized = _coerce_nonnegative_int(num_rollout_replicas)
    parsed_prefix = parse_transfer_rollout_idx(transfer_id)
    if parsed_prefix < 0:
        return []
    if normalized is not None and parsed_prefix >= normalized:
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


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class NcclPayloadTransport(PayloadTransport):
    """NCCL-based payload-transport backend.

    Identifies its rollouts by the ``nccl:`` completion prefix and
    publishes cleanup messages on the ``nccl_cleanup`` pub/sub channel
    when the controller discards rollouts.
    """

    name = "nccl"
    completion_prefix = NCCL_COMPLETION_PREFIX

    def publish_cleanup_for_discarded(
        self,
        *,
        transfer_ids: List[str],
        config: Any,
        redis_client: Any,
    ) -> int:
        if not transfer_ids:
            return 0
        if redis_client is None:
            return 0

        experiment_name = "default"
        try:
            experiment_name = config.logging.experiment_name
        except AttributeError:
            pass
        job_id = os.environ.get("SLURM_JOB_ID", "test")
        prefix = build_nccl_prefix(experiment_name=experiment_name, job_id=job_id)

        num_rollout_replicas: Optional[int] = None
        try:
            num_rollout_replicas = config.rollout.parallelism.n_init_replicas
        except AttributeError:
            pass

        published = 0
        max_retries = 3
        for transfer_id in transfer_ids:
            try:
                rollout_indices = build_transfer_rollout_candidates(
                    transfer_id=transfer_id,
                    num_rollout_replicas=num_rollout_replicas,
                )
                for rollout_idx in rollout_indices:
                    channel = build_cleanup_channel(
                        build_rollout_prefix(prefix, rollout_idx)
                    )
                    payload = json.dumps({"transfer_id": transfer_id})
                    for attempt in range(max_retries):
                        try:
                            redis_client.publish(channel, payload)
                            break
                        except Exception:
                            if attempt == max_retries - 1:
                                raise
                            time.sleep(0.1 * (attempt + 1))
                published += 1
            except Exception as e:
                logger.warning(
                    f"[NcclPayloadTransport] Failed to publish cleanup for "
                    f"transfer_id={transfer_id}: {e}"
                )
        return published


PayloadTransportRegistry.register_class(NcclPayloadTransport)


__all__ = [
    "NCCL_COMPLETION_PREFIX",
    "NCCL_REDIS_NAMESPACE",
    "NcclPayloadTransport",
    "build_cleanup_channel",
    "build_nccl_prefix",
    "build_request_channel",
    "build_rollout_prefix",
    "build_transfer_rollout_candidates",
    "parse_transfer_rollout_idx",
]
