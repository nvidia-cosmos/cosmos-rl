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

"""High-level analyzer entry points."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from cosmos_rl.tools.profiler.opcodes import OpcodeRegistry, OpcodeStats
from cosmos_rl.tools.profiler.parser import (
    TraceEvent,
    parse_trace_log_file,
)


@dataclass
class OpSummary:
    """Per-opcode summary, suitable for table rendering."""

    op: str
    count: int
    total_ms: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    extra: Dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_stats(cls, stats: OpcodeStats) -> "OpSummary":
        return cls(
            op=stats.op,
            count=stats.count,
            total_ms=stats.total_ms,
            mean_ms=stats.mean_ms,
            p50_ms=stats.percentile(50.0),
            p95_ms=stats.percentile(95.0),
            p99_ms=stats.percentile(99.0),
            max_ms=stats.max_ms,
            extra=dict(stats.extra),
        )


def summarize_events(events: Iterable[TraceEvent]) -> Dict[str, OpSummary]:
    """Group events by opcode, dispatch through the registry, and
    return a sorted ``op -> OpSummary`` map (longest total first)."""
    by_op: Dict[str, List[TraceEvent]] = defaultdict(list)
    for ev in events:
        by_op[ev.op].append(ev)

    out: Dict[str, OpSummary] = {}
    for op, evs in by_op.items():
        stats = OpcodeStats(op=op, count=len(evs))
        handler = OpcodeRegistry.get(op)
        handler(evs, stats)
        out[op] = OpSummary.from_stats(stats)

    # Sort by total descending so the dominant ops surface first.
    return dict(
        sorted(out.items(), key=lambda kv: kv[1].total_ms, reverse=True)
    )


def analyze(
    log_paths: Iterable[str],
    *,
    op_filter: Optional[Iterable[str]] = None,
) -> Dict[str, OpSummary]:
    """Parse one or more log files and return per-opcode summaries.

    Args:
        log_paths: Iterable of file paths.  Files are read sequentially.
        op_filter: Optional iterable of opcodes; if provided, only
            events with a matching ``op`` are summarized.
    """
    op_filter_set = set(op_filter) if op_filter else None
    all_events: List[TraceEvent] = []
    for path in log_paths:
        events = parse_trace_log_file(path)
        if op_filter_set is not None:
            events = [e for e in events if e.op in op_filter_set]
        all_events.extend(events)
    return summarize_events(all_events)


__all__ = ["OpSummary", "analyze", "summarize_events"]
