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
"""UCXX consumer packer: scheduling from the base mixin, transport from a strategy.

Kept as a mixin because that is the contract ``attach_data_packer`` dispatches
on, and because downstream code subclasses it.  The UCXX logic itself now lives
in :class:`UCXXTransportStrategy`, which this class composes -- so the same
transport is reachable without subclassing anything.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.payload_transport.prefetch_mixin import PrefetchDataPackerMixin
from cosmos_rl.utils.payload_transport.ucxx.strategy import (
    UCXXTransportStrategy,
    compose_ucxx_transport,
)

__all__ = ["UCXXDataPackerMixin"]


class UCXXDataPackerMixin(PrefetchDataPackerMixin):
    """Mix into a DataPacker to resolve UCXX payload references.

    Thin by design: it owns the wiring, while every transport decision belongs
    to :class:`UCXXTransportStrategy`.
    """

    # ------------------------------------------------------------------
    # Backward-compat aliases so shared test / observability code can read
    # state via the transport-prefixed name.
    # ------------------------------------------------------------------

    @property
    def _ucxx_dp_enabled(self) -> bool:
        return self._prefetch_enabled

    @_ucxx_dp_enabled.setter
    def _ucxx_dp_enabled(self, value: bool) -> None:
        self._prefetch_enabled = bool(value)

    @property
    def _ucxx_dp_prefetch_cache(self) -> Dict[str, Any]:
        return self._prefetch_cache

    @_ucxx_dp_prefetch_cache.setter
    def _ucxx_dp_prefetch_cache(self, value: Dict[str, Any]) -> None:
        self._prefetch_cache = value

    @property
    def _ucxx_dp_step_count(self) -> int:
        return self._prefetch_step_count

    @property
    def _ucxx_dp_prefetch_timeout(self) -> float:
        return self._prefetch_timeout_s

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    @property
    def _ucxx_dp_strategy(self) -> Optional[UCXXTransportStrategy]:
        """The composed transport, or None before setup."""
        return self._transport_strategy

    @property
    def _ucxx_dp_client(self) -> Any:
        strategy = self._transport_strategy
        return None if strategy is None else strategy._client

    @property
    def _ucxx_dp_read_timeout(self) -> float:
        strategy = self._transport_strategy
        return 30.0 if strategy is None else strategy.read_timeout

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def _setup_ucxx_data_packer(
        self,
        *,
        device: Any,
        prefetch_timeout: float = 300.0,
        max_attempts: int = 2,
        read_timeout: float = 30.0,
    ) -> None:
        """Build the UCXX strategy, attach it, and start the prefetch thread.

        Normally invoked by ``UCXXPayloadTransport.attach_data_packer``; direct
        calls are only needed in tests or unusual lifecycle setups.

        Args:
            device: Target GPU device for fetched tensors.
            prefetch_timeout: Per-batch wait ceiling (seconds).  A scheduling
                knob, so it stops here rather than reaching the transport.
            max_attempts: Total attempts per remote slot read (initial +
                retries on transient UCX errors).
            read_timeout: Per-await timeout inside one ``UCXXClient.read``.
        """
        compose_ucxx_transport(
            self,
            device=device,
            prefetch_timeout=prefetch_timeout,
            max_attempts=max_attempts,
            read_timeout=read_timeout,
        )

    def setup_ucxx_data_packer(
        self,
        device: Any,
        prefetch_timeout: float = 300.0,
        max_attempts: int = 2,
        read_timeout: float = 30.0,
    ) -> None:
        """DEPRECATED: use :meth:`_setup_ucxx_data_packer` (kwargs-only).

        Kept as a thin shim because some downstream forks call the original
        public name positionally.
        """
        self._setup_ucxx_data_packer(
            device=device,
            prefetch_timeout=prefetch_timeout,
            max_attempts=max_attempts,
            read_timeout=read_timeout,
        )

    def shutdown_ucxx_data_packer(self) -> None:
        """Stop the background thread, then release UCXX resources.

        Unlike NCCL -- which aborts its comms via ``before_join`` to make the
        join immediate -- UCXX closes its client AFTER the join and instead
        widens the join budget.  ``ncclCommAbort`` is explicitly designed to be
        called while a peer thread is inside a collective; closing a UCXX client
        from a different thread (the worker runs its fetch on its own event
        loop) has no such guarantee.  Each read await is already bounded by
        ``read_timeout``, so waiting that long lets the worker unwind on its
        own.  The strategy therefore leaves ``before_join`` unimplemented.
        """
        self.shutdown_prefetch(
            join_timeout=max(5.0, float(self._ucxx_dp_read_timeout or 5.0))
        )
        strategy = self._transport_strategy
        if strategy is not None:
            strategy.shutdown()
        logger.info("[UCXXDataPackerMixin] Shut down")
