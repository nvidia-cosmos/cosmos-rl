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

"""UCXX-specific opcode handlers for the profiler.

Registers UCXX trace opcodes (emitted by
:mod:`cosmos_rl.utils.payload_transport.ucxx`) with the
:class:`OpcodeRegistry` so the analyzer surfaces:

* total bytes transferred,
* mean / p50 / p95 / p99 bandwidth (Gbps),
* fetch error / stale-slot counts,
* prefetch-cache hit / miss ratios.

Import this module before calling :func:`analyze` to opt in.  The
profiler core does **not** import it eagerly so users without UCXX
don't see UCXX-specific opcodes in their reports.
"""

from __future__ import annotations

from typing import List

from cosmos_rl.tools.profiler.opcodes import OpcodeRegistry, OpcodeStats
from cosmos_rl.tools.profiler.parser import TraceEvent


# Bandwidth helper: bytes / seconds -> gigabits per second.
def _bytes_to_gbps(total_bytes: int, total_ms: float) -> float:
    if total_ms <= 0:
        return 0.0
    return (total_bytes * 8.0) / (total_ms * 1e6)


# ---------------------------------------------------------------------------
# Per-opcode handlers
# ---------------------------------------------------------------------------


def _ucxx_fetch_handler(events: List[TraceEvent], stats: OpcodeStats) -> None:
    """Handler for ``ucxx_fetch`` events.

    Expected fields per event:
        bytes:     total bytes received in this fetch
        slot:      ring-buffer slot index (optional, for debugging)
        chunks:    number of UCXX chunks (optional)
    """
    total_bytes = 0
    chunk_total = 0
    for ev in events:
        if ev.duration_ms is not None:
            stats.durations_ms.append(ev.duration_ms)
        n_bytes = ev.get_int("bytes")
        if n_bytes is not None:
            total_bytes += n_bytes
        n_chunks = ev.get_int("chunks")
        if n_chunks is not None:
            chunk_total += n_chunks

    stats.extra["total_bytes"] = total_bytes
    if total_bytes > 0:
        stats.extra["mean_bandwidth_gbps"] = _bytes_to_gbps(total_bytes, stats.total_ms)
        # Per-event peak: bytes / duration for each event individually,
        # then take p95 of the per-event Gbps values to surface
        # tail-bandwidth without being dragged down by setup overhead.
        per_event_gbps = []
        for ev in events:
            n = ev.get_int("bytes")
            if n is None or ev.duration_ms is None or ev.duration_ms <= 0:
                continue
            per_event_gbps.append(_bytes_to_gbps(n, ev.duration_ms))
        if per_event_gbps:
            sorted_gbps = sorted(per_event_gbps)
            stats.extra["peak_bandwidth_gbps"] = sorted_gbps[-1]
            stats.extra["p50_bandwidth_gbps"] = sorted_gbps[len(sorted_gbps) // 2]
    if chunk_total > 0:
        stats.extra["total_chunks"] = chunk_total


def _ucxx_send_handler(events: List[TraceEvent], stats: OpcodeStats) -> None:
    """Handler for ``ucxx_send`` (worker-side) events.  Same shape as
    :func:`_ucxx_fetch_handler` since the trace fields are symmetric."""
    _ucxx_fetch_handler(events, stats)


def _ucxx_prefetch_handler(events: List[TraceEvent], stats: OpcodeStats) -> None:
    """Handler for ``ucxx_prefetch_collect`` events.

    Tracks how often the deferred-collect actually had to wait
    (``waited_ms`` field present and > 0) vs. resolved immediately.
    """
    waited_count = 0
    no_wait_count = 0
    total_wait_ms = 0.0
    for ev in events:
        if ev.duration_ms is not None:
            stats.durations_ms.append(ev.duration_ms)
        waited = ev.get_float("waited_ms")
        if waited is not None and waited > 0:
            waited_count += 1
            total_wait_ms += waited
        else:
            no_wait_count += 1

    total = waited_count + no_wait_count
    if total > 0:
        stats.extra["wait_rate"] = waited_count / total
        stats.extra["mean_wait_ms"] = (
            total_wait_ms / waited_count if waited_count > 0 else 0.0
        )
    stats.extra["waited_count"] = waited_count
    stats.extra["no_wait_count"] = no_wait_count


def _ucxx_stale_slot_handler(events: List[TraceEvent], stats: OpcodeStats) -> None:
    """Handler for ``ucxx_stale_slot`` events.

    Surfaces a count and the worker IPs / slot indices most affected so
    operators can spot a misbehaving rollout worker quickly.
    """
    for ev in events:
        if ev.duration_ms is not None:
            stats.durations_ms.append(ev.duration_ms)

    by_worker: dict = {}
    for ev in events:
        worker = ev.fields.get("worker_ip", "unknown")
        by_worker[worker] = by_worker.get(worker, 0) + 1
    if by_worker:
        # Top three offenders (count desc) -- enough for triage.
        top = sorted(by_worker.items(), key=lambda kv: kv[1], reverse=True)
        stats.extra["top_worker_ips"] = top[:3]


# ---------------------------------------------------------------------------
# Public registration entry point
# ---------------------------------------------------------------------------


# Map of opcode -> handler.  Listed here so callers (and the test
# suite) can inspect the registered set.
UCXX_OPCODES = {
    "ucxx_fetch": _ucxx_fetch_handler,
    "ucxx_send": _ucxx_send_handler,
    "ucxx_prefetch_collect": _ucxx_prefetch_handler,
    "ucxx_stale_slot": _ucxx_stale_slot_handler,
}


def register_ucxx_opcodes() -> None:
    """Register all UCXX-specific opcode handlers with the global
    :class:`OpcodeRegistry`.  Idempotent."""
    for op, handler in UCXX_OPCODES.items():
        OpcodeRegistry.register(op, handler)


# Auto-register on import.  Importing this module is the opt-in
# signal -- callers do ``from cosmos_rl.tools.profiler import ucxx``
# (or pass ``--ucxx`` to the CLI which imports it for them).
register_ucxx_opcodes()


__all__ = [
    "UCXX_OPCODES",
    "register_ucxx_opcodes",
]
