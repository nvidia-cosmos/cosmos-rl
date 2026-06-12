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

"""``PayloadTransport`` registration for UCXX.

UCXX intentionally diverges from NCCL on two integration points:

1. ``completion_prefix`` is ``None``.  UCXX rollouts return dict-shaped
   metadata (``{"_ucxx": True, "_worker_ip": ..., "_slot": ...}``) rather
   than a string-prefixed completion.  Setting the prefix to ``None``
   tells :meth:`PayloadTransportRegistry.handle_discarded` to skip UCXX
   cleanly when partitioning discards by prefix.

2. ``publish_cleanup_for_discarded`` inherits the base no-op (returns 0).
   Worker-side ring-buffer slots are producer-driven and auto-recycle
   when the writer overwrites an unconsumed READY slot, so the
   controller has nothing to publish.

The ``attach_data_packer`` hook drives the per-packer setup that used
to live in a manual trainer-side call, so the trainer no longer has to
remember to invoke ``_setup_ucxx_data_packer`` after constructing its
data packer.  The ``UCXXDataPackerMixin`` (MR5) provides the actual
``_setup_ucxx_data_packer`` method; on MR3b the hook is a no-op for
any packer that does not subclass that mixin.
"""

from __future__ import annotations

from typing import Any, Optional

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.payload_transport.registry import (
    PayloadTransport,
    PayloadTransportRegistry,
    RedisEndpoint,
)


class UCXXPayloadTransport(PayloadTransport):
    """UCXX backend (zero-copy RDMA / shared-memory transfer)."""

    name = "ucxx"
    # Intentionally None: UCXX uses dict-shaped completion metadata and
    # SHM ring buffers auto-recycle slots, so it does NOT participate in
    # the controller's discard-cleanup dispatch.  ``handle_discarded``
    # skips transports with completion_prefix=None, which is what we
    # want for UCXX.  See module docstring for the longer rationale.
    completion_prefix = None

    def attach_data_packer(
        self,
        packer: Any,
        *,
        config: Any,
        device: Any = None,
        redis_endpoint: Optional[RedisEndpoint] = None,
    ) -> None:
        """Wire UCXX-specific state into a UCXX-aware data packer.

        Looks for the ``_setup_ucxx_data_packer`` method on the packer
        (provided by :class:`UCXXDataPackerMixin` in MR5).  When present,
        invokes it with ``device`` plus tunables resolved from
        ``config.custom``:

        * ``ucxx_prefetch_timeout`` (float, default 30.0): per-batch wait
          ceiling for the prefetch worker thread's result queue.  Bounds
          how long the trainer is willing to block on a single prefetch
          batch before giving up.
        * ``ucxx_read_max_attempts`` (int, default 2): total attempts per
          remote slot read (initial + retries).  Retries fire only on
          transient UCX errors classified in ``_TRANSIENT_UCXX_ERRORS``;
          non-transient errors short-circuit immediately.
        * ``ucxx_read_timeout`` (float, default 5.0): per-await wall
          clock budget for each ``endpoint.send`` / ``endpoint.recv``
          inside one ``UCXXClient.read`` call (distinct from
          ``ucxx_prefetch_timeout`` above -- this bounds a single
          network operation, not a whole batch).  p99 healthy-path read
          of a ~500 MB slot is ~1 s, so 5 s gives 5x headroom.  Larger
          values just delay rotation onto a healthy port when one
          server thread wedges -- they don't help the happy path.

        No-op for packers that do not subclass the mixin -- matches the
        defensive default of the base class.
        """
        setup = getattr(packer, "_setup_ucxx_data_packer", None)
        if setup is None:
            return
        custom = getattr(config, "custom", None) or {}
        try:
            prefetch_timeout = float(custom.get("ucxx_prefetch_timeout", 30.0))
        except (TypeError, ValueError):
            prefetch_timeout = 30.0
        try:
            max_attempts = int(custom.get("ucxx_read_max_attempts", 2))
        except (TypeError, ValueError):
            max_attempts = 2
        if max_attempts < 1:
            max_attempts = 1
        try:
            read_timeout = float(custom.get("ucxx_read_timeout", 5.0))
        except (TypeError, ValueError):
            read_timeout = 5.0
        logger.debug(
            f"[UCXXPayloadTransport] Attaching UCXX data packer "
            f"(device={device}, prefetch_timeout={prefetch_timeout}, "
            f"max_attempts={max_attempts}, read_timeout={read_timeout})"
        )
        setup(
            device=device,
            prefetch_timeout=prefetch_timeout,
            max_attempts=max_attempts,
            read_timeout=read_timeout,
        )

    # publish_cleanup_for_discarded is intentionally NOT overridden:
    # the inherited default returns 0, which is correct for UCXX (SHM
    # ring slots auto-recycle on producer overwrite; the controller has
    # nothing to publish).


PayloadTransportRegistry.register_class(UCXXPayloadTransport)


__all__ = ["UCXXPayloadTransport"]
