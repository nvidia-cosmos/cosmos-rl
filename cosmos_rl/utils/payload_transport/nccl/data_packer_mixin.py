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

"""NCCL consumer packer: scheduling from the base mixin, transport from a strategy.

Kept as a mixin because that is the contract ``attach_data_packer`` dispatches
on (it duck-types ``_setup_nccl_data_packer``) and because the out-of-tree
consumer still subclasses it.  The NCCL logic itself now lives in
:class:`NCCLTransportStrategy`, which this class composes -- so the same
transport is reachable without subclassing anything (see
:meth:`PrefetchDataPackerMixin.set_transport_strategy`).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.payload_transport.nccl.strategy import (
    NCCLTransportStrategy,
    compose_nccl_transport,
)
from cosmos_rl.utils.payload_transport.prefetch_mixin import PrefetchDataPackerMixin

__all__ = ["NCCLDataPackerMixin"]


class NCCLDataPackerMixin(PrefetchDataPackerMixin):
    """Mix into a DataPacker to resolve NCCL payload references.

    Thin by design: it owns the *wiring* -- build the strategy, attach it, run
    the scheduler's lifecycle -- while every transport decision belongs to
    :class:`NCCLTransportStrategy`.
    """

    # ``_attach_payload_transport`` assigns this BEFORE calling setup, keyed on
    # ``hasattr``, to disambiguate policy replicas that share a receiver_rank.
    # It must therefore exist on the class and survive until setup consumes it.
    _nccl_dp_receiver_replica: Optional[str] = None

    # ------------------------------------------------------------------
    # Backward-compat aliases (parity with UCXXDataPackerMixin) so shared
    # test / observability code can read state via the transport-prefixed name.
    # ------------------------------------------------------------------

    @property
    def _nccl_dp_prefetch_cache(self) -> Dict[str, Any]:
        return self._prefetch_cache

    @_nccl_dp_prefetch_cache.setter
    def _nccl_dp_prefetch_cache(self, value: Dict[str, Any]) -> None:
        self._prefetch_cache = value

    @property
    def _nccl_dp_strategy(self) -> Optional[NCCLTransportStrategy]:
        """The composed transport, or None before setup."""
        return self._transport_strategy

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def _setup_nccl_data_packer(
        self,
        *,
        device: Any,
        redis_client: Any,
        config: Any = None,
        prefetch_timeout: float = 30.0,
        max_attempts: int = 2,
        recv_timeout: float = 5.0,
        first_transfer_timeout: float = 30.0,
    ) -> None:
        """Build the NCCL strategy, attach it, and start the prefetch thread.

        Normally invoked by ``NcclPayloadTransport.attach_data_packer``; direct
        calls are only needed in tests or unusual lifecycle setups.

        Args:
            device: Target GPU device for fetched tensors.
            redis_client: Live Redis client for the control plane.
            config: The run Config; supplies the experiment name (Redis key
                prefix) and ``custom`` schema tunables.
            prefetch_timeout: Per-batch wait ceiling (seconds).  Belongs to the
                scheduler, not the transport, so it stops here.
            max_attempts: Total attempts per transfer (initial + retries).
            recv_timeout: Per-recv / per-rendezvous wall-clock budget so a
                wedged sender engages retry / quarantine fast.
            first_transfer_timeout: Cold-start budget, which must absorb the
                comm-init storm.
        """
        compose_nccl_transport(
            self,
            device=device,
            redis_client=redis_client,
            config=config,
            prefetch_timeout=prefetch_timeout,
            max_attempts=max_attempts,
            recv_timeout=recv_timeout,
            first_transfer_timeout=first_transfer_timeout,
        )

    def shutdown_nccl_data_packer(self) -> None:
        """Stop the prefetch thread, then release the transport.

        ``shutdown_prefetch`` aborts the strategy's comms before joining -- it
        defaults ``before_join`` to the attached strategy -- so a worker parked
        in a recv fails fast instead of the join waiting on work only the abort
        can unwedge.
        """
        self.shutdown_prefetch()
        strategy = self._transport_strategy
        if strategy is not None:
            strategy.shutdown()
        logger.info("[NCCLDataPackerMixin] Shut down")
