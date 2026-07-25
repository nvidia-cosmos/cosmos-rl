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

"""In-tree NCCL payload transport.

Formerly a single ``nccl.py`` module; promoted to a package (mirroring
``ucxx/``) once the transport grew a full producer/consumer stack.  This
``__init__`` preserves the historical import surface — every symbol that
``from cosmos_rl.utils.payload_transport.nccl import X`` resolved before
the split still resolves here — and adds the new classes.

Submodules
----------
* :mod:`.protocol` — Redis key / channel builders + transfer-id parsing
  (pure strings; no pynccl / redis dependency).
* :mod:`.transport` — :class:`NcclPayloadTransport` backend registration
  and controller-side discard cleanup.  Importing this module registers
  the ``"nccl"`` backend as a side effect.
* :mod:`.rendezvous` — per-pair NCCL unique-ID exchange + per-transfer
  three-state request/ack handshake over Redis.
* :mod:`.comm_cache` — lazy 2-rank communicator cache with health-aware
  pair quarantine, LRU eviction, and bounded concurrent init.
* :mod:`.buffer_registry` — producer-side GPU send-buffer registry with
  bounded-capacity backpressure and idempotent free-on-cleanup.
* :mod:`.streams` — per-process transfer-stream pool + CUDA event helpers.
* :mod:`.mixins` — :class:`NCCLRolloutMixin` (producer).
* :mod:`.data_packer_mixin` — :class:`NCCLDataPackerMixin` (consumer).

The heavier submodules (rendezvous / comm_cache / streams / mixins /
data_packer_mixin) pull in ``torch`` + ``pynccl``; they are imported
here so the public surface is complete, but each guards its own optional
dependencies at call time rather than import time.
"""

from cosmos_rl.utils.payload_transport.nccl.protocol import (
    NCCL_COMPLETION_PREFIX,
    NCCL_REDIS_NAMESPACE,
    build_cleanup_channel,
    build_nccl_prefix,
    build_pair_uid_key,
    build_request_channel,
    build_response_key,
    build_rollout_prefix,
    build_transfer_rollout_candidates,
    parse_transfer_rollout_idx,
)

# Importing the transport submodule registers the ``"nccl"`` backend.
from cosmos_rl.utils.payload_transport.nccl.transport import (  # noqa: F401
    NcclPayloadTransport,
)
from cosmos_rl.utils.payload_transport.nccl.buffer_registry import (
    SendBufferEntry,
    SendBufferRegistry,
)
from cosmos_rl.utils.payload_transport.nccl.comm_cache import CommCache
from cosmos_rl.utils.payload_transport.nccl.rendezvous import (
    NcclRendezvous,
    RendezvousResult,
    TransferStatus,
)
from cosmos_rl.utils.payload_transport.nccl.streams import (
    TransferStreamPool,
    get_transfer_stream_pool,
)
from cosmos_rl.utils.payload_transport.nccl.data_packer_mixin import (
    NCCLDataPackerMixin,
)
from cosmos_rl.utils.payload_transport.nccl.mixins import NCCLRolloutMixin

__all__ = [
    "NCCL_COMPLETION_PREFIX",
    "NCCL_REDIS_NAMESPACE",
    "CommCache",
    "NCCLDataPackerMixin",
    "NCCLRolloutMixin",
    "NcclPayloadTransport",
    "NcclRendezvous",
    "RendezvousResult",
    "SendBufferEntry",
    "SendBufferRegistry",
    "TransferStatus",
    "TransferStreamPool",
    "build_cleanup_channel",
    "build_nccl_prefix",
    "build_pair_uid_key",
    "build_request_channel",
    "build_response_key",
    "build_rollout_prefix",
    "build_transfer_rollout_candidates",
    "get_transfer_stream_pool",
    "parse_transfer_rollout_idx",
]
