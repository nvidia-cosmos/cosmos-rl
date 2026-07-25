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
from typing import Any

from cosmos_rl.utils.payload_transport.nccl.protocol import build_nccl_prefix

__all__ = ["resolve_global_rank", "resolve_prefix"]


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
