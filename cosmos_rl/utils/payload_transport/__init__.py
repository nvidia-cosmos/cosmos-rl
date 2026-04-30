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

"""Pluggable payload-transport backends.

By default Cosmos-RL ships rollout completion payloads (token IDs,
log-probs, reward tensors, …) from rollout workers to the controller via
**Redis streams**.  For workloads with large payloads, two opt-in
backends bypass Redis on the data plane:

* **NCCL** — point-to-point GPU sends; Redis still carries control
  metadata.  See :mod:`cosmos_rl.utils.payload_transport.nccl`.
* **UCXX** — RDMA / shared-memory zero-copy transfer (added in a
  follow-up MR).  See ``cosmos_rl.utils.payload_transport.ucxx``.

Selecting a backend
-------------------

The active backend is read from the experiment config's ``[custom]``
section using :func:`get_payload_transfer_mode`:

.. code-block:: toml

    # Preferred (string) form:
    [custom]
    payload_transfer = "nccl"           # one of "redis", "nccl", "ucxx"

    # Deprecated boolean alias (still honored):
    [custom]
    nccl_payload_transfer = true        # equivalent to payload_transfer = "nccl"

The default is ``"redis"`` when neither key is present.

Registering backends
--------------------

Each backend registers itself with :class:`PayloadTransportRegistry` at
import time:

.. code-block:: python

    from cosmos_rl.utils.payload_transport import (
        PayloadTransport, PayloadTransportRegistry,
    )

    class MyTransport(PayloadTransport):
        name = "my_transport"
        completion_prefix = "mytx:"
        ...

    PayloadTransportRegistry.register(MyTransport)

Cosmos-RL ships ``redis`` (no-op default) and ``nccl`` registered out of
the box.  ``ucxx`` registers itself when imported.
"""

from cosmos_rl.utils.payload_transport.registry import (
    DEFAULT_TRANSFER_MODE,
    LEGACY_NCCL_KEY,
    PAYLOAD_TRANSFER_KEY,
    PayloadTransport,
    PayloadTransportRegistry,
    get_payload_transfer_mode,
)

# Importing the NCCL submodule registers the backend as a side effect.
from cosmos_rl.utils.payload_transport import nccl  # noqa: F401

# Redis is the implicit no-op default — it has no side-effects to wire
# up here, but registering a sentinel keeps lookup uniform.
PayloadTransportRegistry.register_default_redis()


__all__ = [
    "DEFAULT_TRANSFER_MODE",
    "LEGACY_NCCL_KEY",
    "PAYLOAD_TRANSFER_KEY",
    "PayloadTransport",
    "PayloadTransportRegistry",
    "get_payload_transfer_mode",
]
