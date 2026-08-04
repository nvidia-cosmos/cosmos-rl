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

"""Shared runtime-context resolution for the NCCL payload transport.

The producer (:mod:`mixins`) and consumer (:mod:`data_packer_mixin`) are the
two ends of the SAME transport and must resolve the same global rank and the
same Redis key prefix.  Keeping one copy here (rather than duplicating the
logic on each side) removes the risk of the two ends silently drifting out of
agreement -- a drift that would only surface as a hard-to-debug rendezvous
mismatch on the cluster.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from cosmos_rl.utils.payload_transport.nccl.protocol import build_nccl_prefix

__all__ = [
    "resolve_custom_int",
    "resolve_global_rank",
    "resolve_max_live_comms",
    "resolve_prefix",
]

#: Floor for the comm cache's live-comm cap.  Never size *below* the historical
#: default, so auto-derivation can only ever raise the ceiling.
DEFAULT_MAX_LIVE_COMMS = 128


def resolve_global_rank() -> int:
    """This process's rank in global rank space.

    Prefers the live ``torch.distributed`` rank; falls back to the launcher
    env (``RANK`` / ``GLOBAL_RANK`` / ``SLURM_PROCID``); defaults to 0.
    """
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    for var in ("RANK", "GLOBAL_RANK", "SLURM_PROCID"):
        val = os.environ.get(var)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                continue
    return 0


def resolve_prefix(config: Any) -> str:
    """Root Redis key prefix for this run (``{namespace}:{exp}:{job_id}``)."""
    experiment_name = "default"
    try:
        experiment_name = config.logging.experiment_name
    except AttributeError:
        pass
    job_id = os.environ.get("SLURM_JOB_ID", "test")
    return build_nccl_prefix(experiment_name=experiment_name, job_id=job_id)


def resolve_custom_int(config: Any, key: str, default: int, *, minimum: int = 1) -> int:
    """Read ``[custom].<key>`` as an int, clamped to ``minimum``.

    Returns ``default`` when the key is absent or unparseable, so a typo
    degrades to the documented default rather than crashing a launched job.
    """
    custom = getattr(config, "custom", None) or {}
    try:
        raw = custom.get(key)
    except AttributeError:
        return default
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(minimum, value)


def _replica_count(config: Any, role: str) -> Optional[int]:
    """``<role>.parallelism.n_init_replicas`` if resolvable."""
    try:
        value = int(getattr(getattr(config, role).parallelism, "n_init_replicas"))
    except (AttributeError, TypeError, ValueError):
        return None
    return value if value > 0 else None


def resolve_max_live_comms(config: Any, *, peer_role: str) -> int:
    """Live-comm cap for a :class:`CommCache`, sized from the actual fan-out.

    Every cached comm is an independent 2-rank P2P communicator, so the cap is
    resource hygiene, not a correctness bound -- and the working set is
    *topology*-bounded (pair keys carry no per-transfer component), so it equals
    the number of distinct peers.  Sizing it blindly is what makes the cap
    dangerous: at exactly the wrong scale, a cyclic access pattern over N+1 keys
    with capacity N is the pathological LRU case -- ~100% miss, every transfer
    rebuilding a comm.

    ``peer_role`` names the config section holding the peers this side talks to
    (``"policy"`` for a producer, ``"rollout"`` for a consumer).  Headroom of 4x
    absorbs elastic scale-up; the result never drops below
    :data:`DEFAULT_MAX_LIVE_COMMS`.  ``[custom].nccl_max_live_comms`` overrides.
    """
    explicit = resolve_custom_int(config, "nccl_max_live_comms", 0, minimum=0)
    if explicit > 0:
        return explicit
    peers = _replica_count(config, peer_role)
    if peers is None:
        return DEFAULT_MAX_LIVE_COMMS
    return max(DEFAULT_MAX_LIVE_COMMS, 4 * peers)
