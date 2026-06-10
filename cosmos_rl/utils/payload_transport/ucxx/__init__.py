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

"""UCXX-based payload transport.

Architecture
------------
::

    ┌─────────────┐    metadata (worker_ip, port, slot)    ┌─────────────┐
    │  Rollout    │───────────────────────────────────────►│  Policy     │
    │  Worker     │      via cosmos-rl Redis stream         │  Trainer    │
    │             │                                        │             │
    │             │◄══════════════════════════════════════►│             │
    │             │         actual trajectory data         │             │
    └─────────────┘              via UCXX                  └─────────────┘
          │                                                      │
    ┌─────┴─────┐                                          ┌─────┴─────┐
    │UCXXBuffer │                                          │UCXXClient │
    │ (server)  │                                          │ (client)  │
    └───────────┘                                          └───────────┘

Components
----------

* :class:`TensorSpec` -- fixed-shape tensor descriptor for flat schemas.
* :class:`SharedRingBuffer` -- POSIX shared-memory ring buffer with a
  :class:`SlotState` four-state machine (FREE → WRITING → READY →
  READING → FREE) for inter-process coordination.
* :class:`UCXXBuffer` -- UCXX server wrapping ``SharedRingBuffer``;
  serves data to remote trainers via the UCXX protocol.
* :class:`UCXXClient` -- UCXX client for trainers; pools endpoints to
  amortize connection setup.
* :class:`UCXXRolloutMixin` / :class:`UCXXTrainerMixin` -- mixins that
  wire UCXX into rollout workers and trainers respectively.
* :class:`UCXXPayloadTransport` -- registers the ``"ucxx"`` backend
  with :class:`~cosmos_rl.utils.payload_transport.PayloadTransportRegistry`.

Optional dependency
-------------------

UCXX itself (the Python binding ``ucxx-cu12`` for CUDA 12) is an
**optional** extra; install with::

    pip install cosmos_rl[ucxx]

When the UCXX library is not present, :data:`UCXX_AVAILABLE` is
``False`` and attempting to start a server / client raises
``RuntimeError`` rather than failing at import time.  All of the
shared-memory bits work without UCXX, so :class:`SharedRingBuffer`
remains usable for single-node profiling / testing.
"""

from cosmos_rl.utils.payload_transport.ucxx.data_packer_mixin import (
    UCXXDataPackerMixin,
)
from cosmos_rl.utils.payload_transport.ucxx.mixins import (
    UCXXRolloutMixin,
    UCXXTrainerMixin,
)
from cosmos_rl.utils.payload_transport.ucxx.shared_buffer import (
    BufferConfig,
    BufferMetrics,
    SharedRingBuffer,
    SlotError,
    SlotState,
)
from cosmos_rl.utils.payload_transport.ucxx.tensor_spec import TensorSpec
from cosmos_rl.utils.payload_transport.ucxx.transport import UCXXPayloadTransport
from cosmos_rl.utils.payload_transport.ucxx.ucxx_buffer import (
    UCXX_AVAILABLE,
    StaleSlotError,
    UCXXBuffer,
    UCXXBufferConfig,
    UCXXClient,
)


__all__ = [
    "BufferConfig",
    "BufferMetrics",
    "SharedRingBuffer",
    "SlotError",
    "SlotState",
    "StaleSlotError",
    "TensorSpec",
    "UCXX_AVAILABLE",
    "UCXXBuffer",
    "UCXXBufferConfig",
    "UCXXClient",
    "UCXXDataPackerMixin",
    "UCXXPayloadTransport",
    "UCXXRolloutMixin",
    "UCXXTrainerMixin",
]
