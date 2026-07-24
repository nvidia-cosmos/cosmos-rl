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

"""Health-aware rotation skip-list shared by the payload transports.

Both transports quarantine an unhealthy endpoint with a cooldown after a
transient transport-class failure and route around it until the cooldown
expires:

* UCXX rotates around an unhealthy ``(worker_ip, port)`` server thread
  (``UCXXClient._port_skip_until``).
* NCCL quarantines an unhealthy ``(sender_rank, receiver_rank)`` pair /
  ``(rollout_idx, sender_rank)`` endpoint in its 2-rank comm cache.

The mechanism is identical — a ``key -> skip_until`` timestamp map, a
lazy expiry check, and a **never-starve** fallback (if every candidate is
quarantined, return the full set rather than nothing) — so it lives here
once, in :class:`HealthSkipList`, and both transports delegate to it.

Concurrency: single-key dict ops are atomic under CPython, and the
worst-case race (reading a one-tick-stale timestamp) is harmless, so the
skip-list is intentionally lock-free — matching the original UCXX
comment.  The backing map is exposed as :attr:`skip_until` so callers that
historically poked the raw dict keep working.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

__all__ = ["HealthSkipList"]


class HealthSkipList:
    """Cooldown-based quarantine map with a never-starve healthy filter.

    Args:
        cooldown: Default seconds a key stays quarantined.
        clock: Monotonic clock (injectable for tests).
    """

    def __init__(
        self,
        cooldown: float = 30.0,
        *,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self._cooldown = max(0.0, cooldown)
        self._clock = clock or time.monotonic
        # key -> monotonic tick at which the key becomes re-eligible.
        self._until: Dict[Any, float] = {}

    @property
    def skip_until(self) -> Dict[Any, float]:
        """The backing ``key -> skip_until`` map (direct-access compat)."""
        return self._until

    def quarantine(self, key: Any, *, cooldown: Optional[float] = None) -> None:
        """Quarantine ``key`` for ``cooldown`` seconds (default cooldown).

        Re-failure during a cooldown extends the deadline to a fresh
        ``now + cooldown`` (no exponential backoff).
        """
        cd = self._cooldown if cooldown is None else max(0.0, cooldown)
        self._until[key] = self._clock() + cd

    def is_quarantined(self, key: Any) -> bool:
        """Return ``True`` while ``key`` is within its cooldown.

        Expired entries are GC'd lazily on read so the map does not grow
        unbounded.
        """
        until = self._until.get(key)
        if until is None:
            return False
        if self._clock() >= until:
            # Lazily drop the expired entry.
            self._until.pop(key, None)
            return False
        return True

    def healthy(
        self,
        candidates: List[Any],
        *,
        key_fn: Callable[[Any], Any] = lambda x: x,
    ) -> List[Any]:
        """Filter ``candidates`` to those not currently quarantined.

        Never-starve: if *every* candidate is quarantined, return the full
        list unchanged so a transient all-endpoint outage cannot wedge the
        caller (better to retry a maybe-recovered endpoint than to stall).
        """
        now = self._clock()
        healthy = [c for c in candidates if self._until.get(key_fn(c), 0.0) <= now]
        return healthy if healthy else list(candidates)

    def clear(self, key: Any) -> None:
        self._until.pop(key, None)

    def clear_all(self) -> None:
        self._until.clear()

    def __len__(self) -> int:
        return len(self._until)

    def __contains__(self, key: Any) -> bool:
        return self.is_quarantined(key)
