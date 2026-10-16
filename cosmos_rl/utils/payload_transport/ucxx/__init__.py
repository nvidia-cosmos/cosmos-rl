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

    ┌─────────────┐    metadata (worker_ip, ports, slot)   ┌─────────────┐
    │  Rollout    │───────────────────────────────────────►│  Policy     │
    │  Worker     │      via cosmos-rl Redis stream        │  Trainer    │
    │             │                                        │             │
    │             │◄══════════════════════════════════════►│             │
    │             │     trajectory bytes via UCXX          │             │
    └─────────────┘    (RDMA cross-node, shm same-node)    └─────────────┘
          │                                                      │
    ┌─────┴─────┐                                          ┌─────┴─────┐
    │UCXXBuffer │  N server threads, one UCX listener      │UCXXClient │
    │ (server)  │  per port, all sharing one SHM segment   │ (client)  │
    └───────────┘                                          └───────────┘

Slot lifecycle (single-chunk per slot, post-Commit-F)
-----------------------------------------------------
::

    writer (rollout)              reader (UCXX server handler)
    ─────────────────             ────────────────────────────
    write_raw(buf):                          recv([slot])
        FREE → WRITING                           │
        memcpy SHM                          read_raw(slot):
        WRITING → READY                          READY → READING
                                            send([0, raw_buf])
                                            on success:
                                                mark_consumed(slot)
                                                READING → FREE
                                            on uncertain send:
                                                retain slot + endpoint
                                                fail transport process

A *single* server handler owns each accepted slot until both native sends are
proven complete. Successful completion calls ``mark_consumed`` and returns the
slot to the ring. Cancellation, timeout, or an error after native work is issued
does not call ``release_reading``: Python failure is not proof that UCXX stopped
using the shared-memory bytes. The slot and endpoint remain owned, the transport
is marked unusable, and process restart is the recovery boundary.

Three retry layers
------------------

When a remote read fails, three independent retry layers attempt
recovery.  Each protects against a different failure mode:

1. ``UCXXClient.read`` -- per-call **port rotation**.  If attempt 0
   fails with a clean connection reset, close, or cancellation before
   native buffer ownership becomes uncertain, we quarantine the offending
   ``(worker_ip, port)`` for
   :data:`ucxx_buffer._PORT_QUARANTINE_SEC` and retry once on the
   next port in round-robin order.  Bounded: 2 attempts.

2. ``UCXXTransportStrategy._fetch_all`` -- per-slot **fresh-call retry**.
   Its internal read operation wraps the layer-1 call in ``max_attempts``
   (default 2) outer
   retries on transient errors only; non-retryable errors (e.g.
   :class:`StaleSlotError`) propagate immediately.

3. ``UCXXTransportStrategy._fetch_all`` -- per-batch
   **multi-round** retry.  Whole batch runs in ``asyncio.gather``;
   episodes that returned *retryable* failures are re-attempted in
   the next round (up to ``_MAX_FETCH_ROUNDS = 3``).  Non-retryable
   failures drop on the first round.

Total ceiling with the defaults is 2 × 2 × 3 = 12 attempts per slot. Each
native read is bounded by the configured ``ucxx_read_timeout`` (30 seconds by
default). A clean pre-acceptance transient failure can therefore degrade to a
missing episode after bounded retries. A timeout is not in that class: the
client cannot prove native completion, so the transport fails and retains the
operation until process exit.

Components
----------

* :class:`TensorSpec` -- fixed-shape tensor descriptor for flat schemas.
* :class:`SharedRingBuffer` -- POSIX shared-memory ring buffer with a
  :class:`SlotState` four-state machine (FREE → WRITING → READY →
  READING → FREE) for inter-process coordination.
* :class:`UCXXBuffer` -- UCXX server wrapping ``SharedRingBuffer``;
  serves data to remote trainers via the UCXX protocol.
* :class:`UCXXClient` -- UCXX client for trainers; pools endpoints
  per ``(worker_ip, port)`` and tracks a transient skip-list for
  unhealthy ports.
* :class:`~cosmos_rl.utils.payload_transport.PayloadContext` -- composition API
  that owns a UCXX producer when selected.
* :class:`UCXXDataPackerMixin` -- trainer-side mixin (subclass of
  ``PrefetchDataPackerMixin``) that resolves UCXX pointers in the
  DataPacker's ``get_policy_input()`` with prefetch + double-buffering.
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

from cosmos_rl.utils.payload_transport.ucxx.data_packer_mixin import UCXXDataPackerMixin
from cosmos_rl.utils.payload_transport.ucxx.shared_buffer import (
    BufferConfig,
    BufferMetrics,
    SharedRingBuffer,
    SlotError,
    SlotState,
)
from cosmos_rl.utils.trajectory import TensorSpec
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
]
