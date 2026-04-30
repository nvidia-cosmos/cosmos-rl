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

"""Trace timing utilities for performance analysis.

Provides a process-local monotonic time source and a per-process worker
identity that together support a structured ``[Trace] ...`` log line
convention.  The accompanying offline analyzer
(``cosmos_rl.tools.profiler``) parses these lines to produce iteration
breakdown reports.

See ``docs/profiler/trace_format.rst`` for the full log-line grammar and
field semantics.

Quick start
-----------

.. code-block:: python

    from cosmos_rl.utils.trace import (
        get_trace_time, set_worker_id, format_trace,
    )
    from cosmos_rl.utils.logging import logger

    set_worker_id("policy_0")

    start = get_trace_time()
    do_work()
    end = get_trace_time()
    logger.debug(format_trace(
        thread="trainer", op="step_training",
        start=start, end=end, iter=42,
    ))

The above emits, e.g.::

    [Trace] worker=policy_0 thread=trainer op=step_training start=12.3 end=42.7 iter=42

The ``worker=`` field is included automatically when ``set_worker_id``
has been called; otherwise it is omitted.
"""

from __future__ import annotations

import time
from typing import Any, Optional

# Process-local trace start time (set on first call to get_trace_time()).
_trace_start_time: Optional[float] = None

# Process-local worker identity for trace lines.  Typically set during
# replica initialization to a short tag (e.g. first 8 hex chars of the
# cosmos-rl ``replica_name`` UUID, or ``"<role>_<rank>"``).
_worker_id: Optional[str] = None


def set_worker_id(worker_id: str) -> None:
    """Set the per-process worker identity for trace log lines.

    Should be called once during worker/replica initialization so that
    subsequent :func:`format_trace` output includes a ``worker=...``
    field, allowing the offline analyzer to distinguish interleaved
    output from multiple replicas.

    Args:
        worker_id: A short identifier for the calling process, e.g.
            ``"policy_0"`` or the first 8 hex chars of a replica UUID.
    """
    global _worker_id
    _worker_id = worker_id


def get_worker_id() -> str:
    """Return the worker identity, or ``"?"`` if :func:`set_worker_id`
    has not been called yet."""
    return _worker_id or "?"


def get_trace_time() -> float:
    """Return milliseconds elapsed since the first trace event.

    The first call to this function in a process records the reference
    time (t = 0).  All subsequent calls return ``(now - t0) * 1000``.

    Using a process-local zero point keeps trace timestamps small and
    easy to read while still allowing the offline analyzer to align
    timelines across processes via wall-clock timestamps that the
    logging framework prefixes to each line.

    Returns:
        Milliseconds since the first call to this function in the
        current process.
    """
    global _trace_start_time
    now = time.perf_counter()
    if _trace_start_time is None:
        _trace_start_time = now
    return (now - _trace_start_time) * 1000.0


def reset_trace_time() -> None:
    """Reset the trace start time.  Useful for tests."""
    global _trace_start_time
    _trace_start_time = None


def format_trace(
    *,
    thread: str,
    op: str,
    start: Optional[float] = None,
    end: Optional[float] = None,
    **fields: Any,
) -> str:
    """Format a structured ``[Trace] ...`` log line.

    The output follows a strict ``key=value`` grammar so the offline
    analyzer can parse it without ambiguity.  See
    ``docs/profiler/trace_format.rst`` for the formal grammar.

    Args:
        thread: Logical thread / role name (``"trainer"``,
            ``"rollout"``, ``"controller"``, etc.).
        op: Operation name (e.g. ``"step_training"``,
            ``"trainer_forward"``).  The analyzer's opcode registry
            keys off this value.
        start: Optional start time in ms (typically from
            :func:`get_trace_time`).
        end: Optional end time in ms.
        **fields: Additional ``key=value`` fields appended to the line.
            Values are converted with ``repr`` for non-numeric / non-
            string types so the analyzer can round-trip them.

    Returns:
        A single-line string starting with ``[Trace]``, suitable for
        passing to ``logger.debug`` or ``logger.info``.
    """
    parts: list[str] = ["[Trace]"]
    worker = _worker_id
    if worker:
        parts.append(f"worker={worker}")
    parts.append(f"thread={thread}")
    parts.append(f"op={op}")
    if start is not None:
        parts.append(f"start={start:.1f}")
    if end is not None:
        parts.append(f"end={end:.1f}")
    for key, value in fields.items():
        parts.append(f"{key}={_format_value(value)}")
    return " ".join(parts)


def _format_value(value: Any) -> str:
    """Render a trace field value.

    Floats get one decimal place (matching ``start``/``end``
    formatting); ints, strings, and bools render via ``str``; everything
    else falls back to ``repr`` so the analyzer can detect unusual
    types and skip them gracefully.
    """
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.1f}"
    if isinstance(value, (int, str)):
        return str(value)
    return repr(value)


__all__ = [
    "format_trace",
    "get_trace_time",
    "get_worker_id",
    "reset_trace_time",
    "set_worker_id",
]
