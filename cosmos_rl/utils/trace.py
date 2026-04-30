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

The high-level :func:`trace_op` context manager wraps the common
*time-it / format / log* pattern in a single call and is the
recommended entry point for new instrumentation:

.. code-block:: python

    from cosmos_rl.utils.trace import set_worker_id, trace_op

    set_worker_id("policy_0")

    with trace_op("trainer", "step_training", iter=42):
        do_work()

    # Attach fields discovered mid-block by mutating the yielded dict:
    with trace_op("ucxx_prefetch", "ucxx_fetch") as extras:
        n_bytes = do_fetch()
        extras["bytes"] = n_bytes
        extras["count"] = 4

The above emits, e.g.::

    [Trace] worker=policy_0 thread=trainer op=step_training start=12.3 end=42.7 iter=42
    [Trace] worker=policy_0 thread=ucxx_prefetch op=ucxx_fetch start=43.0 end=45.1 bytes=1048576 count=4

The ``worker=`` field is included automatically when ``set_worker_id``
has been called; otherwise it is omitted.

Low-level building blocks (:func:`get_trace_time`, :func:`format_trace`)
remain available for call sites that cannot use a ``with`` block (e.g.
when start and end straddle async boundaries).
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

# Characters that, if present in a string field value, would break the
# `key=value` whitespace-delimited grammar.  Values containing any of
# these are emitted as JSON-quoted strings instead of raw.
_UNSAFE_CHARS = frozenset(' \t\n\r="[]')

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


def _default_logger() -> Any:
    """Return the cosmos-rl logger when available, else stdlib logger.

    Resolved lazily so importing :mod:`cosmos_rl.utils.trace` from
    contexts where the cosmos-rl logging package is not yet importable
    (early bootstrap, isolated tests) still works.
    """
    try:
        from cosmos_rl.utils.logging import logger as _logger

        return _logger
    except Exception:
        import logging

        return logging.getLogger("cosmos_rl.trace")


@contextmanager
def trace_op(
    thread: str,
    op: str,
    *,
    logger: Optional[Any] = None,
    log_level: str = "debug",
    **fields: Any,
) -> Iterator[dict]:
    """Context manager that times a block and emits a ``[Trace]`` line.

    Equivalent to::

        start = get_trace_time()
        try:
            do_work()
        finally:
            end = get_trace_time()
            logger.debug(format_trace(
                thread=thread, op=op, start=start, end=end, **fields,
            ))

    Yielded value
        A mutable ``dict`` initialised from ``**fields``.  Mutate it
        inside the ``with`` block to attach fields discovered at run
        time (e.g. ``extras["bytes"] = n``); the final
        :func:`format_trace` call uses the post-mutation contents.

    Exception handling
        If the body raises, the trace line is still emitted (with
        ``status=error`` and ``err=<exception class name>`` appended)
        and the exception then propagates unchanged.

    Args:
        thread: Logical thread / role name passed through to
            :func:`format_trace`.
        op: Operation name passed through to :func:`format_trace`.
        logger: Optional logger object.  Defaults to the cosmos-rl
            logger (``cosmos_rl.utils.logging.logger``); if that
            cannot be imported, falls back to a stdlib logger named
            ``"cosmos_rl.trace"``.
        log_level: Method name on ``logger`` used to emit the line
            (``"debug"``, ``"info"``, ``"warning"`` …).  Defaults to
            ``"debug"`` to match the existing instrumentation style.
        **fields: Initial extra ``key=value`` fields, identical in
            semantics to :func:`format_trace`'s ``**fields``.

    Example:
        Basic block timing::

            with trace_op("trainer", "step_training", iter=current_iter):
                run_iteration()

        Attaching mid-block metadata::

            with trace_op("rollout", "ucxx_fetch") as extras:
                n = do_fetch()
                extras["count"] = n
                extras["bytes"] = n * SLOT_BYTES

        Custom logger / level::

            with trace_op(
                "trainer", "trainer_forward",
                logger=my_logger, log_level="info",
            ):
                model(x)
    """
    extras: dict = dict(fields)
    start = get_trace_time()
    error_class: Optional[str] = None
    try:
        yield extras
    except BaseException as exc:
        error_class = type(exc).__name__
        raise
    finally:
        end = get_trace_time()
        if error_class is not None:
            # Don't clobber a caller-supplied status / err, but make
            # sure the failure is visible in the emitted line.
            extras.setdefault("status", "error")
            extras.setdefault("err", error_class)
        line = format_trace(thread=thread, op=op, start=start, end=end, **extras)
        emit_target = logger if logger is not None else _default_logger()
        method = getattr(emit_target, log_level, None)
        if method is None:
            # Fall back to .info if the requested level isn't defined
            # on the supplied logger; never let the trace path raise.
            method = getattr(emit_target, "info", None)
        if method is not None:
            try:
                method(line)
            except Exception:
                # Never let logging failures mask the user's code path.
                pass


def _quote_string(value: str) -> str:
    """Return ``value`` raw if it is grammar-safe, else JSON-quoted.

    Free-form string values may contain characters that would break the
    whitespace-delimited ``key=value`` grammar (notably spaces, ``=``,
    embedded quotes, and brackets).  When any such character is present
    -- or when the value is empty -- the value is emitted as a
    JSON-encoded double-quoted string with standard backslash escapes
    so the analyzer can recover the original string with ``json.loads``.

    Bare identifiers (matching ``[A-Za-z0-9_./:-]*`` content) round-trip
    unchanged for readability.
    """
    if value == "" or any(ch in _UNSAFE_CHARS for ch in value):
        return json.dumps(value)
    return value


def _format_value(value: Any) -> str:
    """Render a trace field value.

    * ``bool`` → ``0`` / ``1`` (numeric for the analyzer).
    * ``float`` → one decimal place (matching ``start``/``end``).
    * ``int`` → ``str(value)``.
    * ``str`` → raw if grammar-safe, else JSON-quoted (see
      :func:`_quote_string`).
    * Anything else → ``repr(value)``, then JSON-quoted if the repr
      itself contains grammar-unsafe characters.  This keeps unusual
      types parseable as opaque strings rather than silently corrupting
      the line.
    """
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.1f}"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return _quote_string(value)
    return _quote_string(repr(value))


__all__ = [
    "format_trace",
    "get_trace_time",
    "get_worker_id",
    "reset_trace_time",
    "set_worker_id",
    "trace_op",
]
