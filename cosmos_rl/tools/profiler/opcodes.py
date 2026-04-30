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

"""Pluggable opcode registry.

Backends register opcode handlers here so the analyzer can recognize
transport-specific events (e.g. ``ucxx_fetch``) and surface backend-
specific metrics in the summary.  This is the extension point used by
MR7's UCXX profiler support — adding new opcodes does not require
modifying the core analyzer.

Default opcodes (registered eagerly by :func:`_register_defaults`):

* ``step_training`` — trainer iteration (duration only).
* ``rollout_generation`` — rollout worker generation (duration only).
* ``redis_publish`` / ``redis_consume`` — controller payload transfer.
* ``nccl_send`` / ``nccl_recv`` — NCCL payload transfer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from cosmos_rl.tools.profiler.parser import TraceEvent


@dataclass
class OpcodeStats:
    """Aggregate statistics for one opcode."""

    op: str
    count: int = 0
    durations_ms: List[float] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return sum(self.durations_ms)

    @property
    def mean_ms(self) -> float:
        return (self.total_ms / len(self.durations_ms)) if self.durations_ms else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.durations_ms) if self.durations_ms else 0.0

    @property
    def min_ms(self) -> float:
        return min(self.durations_ms) if self.durations_ms else 0.0

    def percentile(self, p: float) -> float:
        """Linear-interpolation percentile.  ``p`` in [0, 100]."""
        if not self.durations_ms:
            return 0.0
        sorted_d = sorted(self.durations_ms)
        if len(sorted_d) == 1:
            return sorted_d[0]
        rank = (p / 100.0) * (len(sorted_d) - 1)
        lo = int(rank)
        hi = min(lo + 1, len(sorted_d) - 1)
        frac = rank - lo
        return sorted_d[lo] + frac * (sorted_d[hi] - sorted_d[lo])


# Handler signature: receives the matching events and the stats
# accumulator; returns nothing (mutates ``stats.extra`` in place when
# emitting backend-specific metrics).
OpcodeHandler = Callable[[List[TraceEvent], OpcodeStats], None]


def _default_handler(events: List[TraceEvent], stats: OpcodeStats) -> None:
    """Records duration for any event with both ``start`` and ``end``."""
    for ev in events:
        if ev.duration_ms is not None:
            stats.durations_ms.append(ev.duration_ms)


class OpcodeRegistry:
    """Singleton-style mapping ``op -> OpcodeHandler``.

    The registry is process-global so backends loaded as plugins can
    register their handlers at import time.  Use :meth:`reset` between
    tests if needed.
    """

    _registry: Dict[str, OpcodeHandler] = {}
    # Opcodes that are "known" but use the default-duration handler.
    # Tracked separately so :func:`summarize_events` can warn about
    # unknown opcodes without spamming for the common ones.
    _known: set = set()

    @classmethod
    def register(
        cls, op: str, handler: Optional[OpcodeHandler] = None
    ) -> None:
        """Register ``handler`` for opcode ``op``.

        If ``handler`` is None, registers the opcode as "known" while
        still using :func:`_default_handler` to compute durations.
        """
        cls._known.add(op)
        if handler is not None:
            cls._registry[op] = handler

    @classmethod
    def get(cls, op: str) -> OpcodeHandler:
        return cls._registry.get(op, _default_handler)

    @classmethod
    def is_known(cls, op: str) -> bool:
        return op in cls._known

    @classmethod
    def known_opcodes(cls) -> List[str]:
        return sorted(cls._known)

    @classmethod
    def reset(cls) -> None:
        cls._registry.clear()
        cls._known.clear()
        _register_defaults()


def _register_defaults() -> None:
    """Register the core (transport-agnostic) opcodes."""
    for op in (
        "step_training",
        "rollout_generation",
        "redis_publish",
        "redis_consume",
        "nccl_send",
        "nccl_recv",
        "policy_compute",
        "rollout_collate",
    ):
        OpcodeRegistry.register(op)


# Bootstrap defaults on import so callers get a usable registry without
# explicit setup.
_register_defaults()


__all__ = [
    "OpcodeHandler",
    "OpcodeRegistry",
    "OpcodeStats",
]
