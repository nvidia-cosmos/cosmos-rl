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

"""Parser for ``[Trace]`` structured log lines.

The grammar is documented in ``docs/profiler/trace_format.rst`` and
mirrors the format produced by
:func:`cosmos_rl.utils.trace.format_trace`::

    [Trace] [worker=<id>] thread=<name> op=<op> [start=<ms>] [end=<ms>]
            [k1=v1 k2=v2 ...]

This parser is **trace-version-agnostic**: any extra ``key=value``
fields beyond the documented core get bundled into a ``fields`` dict so
new opcodes (e.g. UCXX-specific extensions registered in MR7) need not
modify the parser to surface their data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional


# Match a single ``key=value`` pair where the value runs up to the next
# whitespace.  Quoting is intentionally not supported in the core
# format -- callers should avoid spaces in trace values to keep parsing
# trivial across log shippers.
_KV_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<val>\S+)")

# Match the [Trace] sentinel anywhere in the line so the parser is
# agnostic to upstream timestamping / log prefixes.
TRACE_LINE_REGEX = re.compile(r"\[Trace\](?P<body>.+)$")

# Reserved field names extracted into top-level dataclass attributes.
# Anything not in this set ends up in ``TraceEvent.fields``.
_CORE_FIELDS = frozenset({"worker", "thread", "op", "start", "end"})


@dataclass
class TraceEvent:
    """Single parsed ``[Trace]`` line.

    Attributes:
        op: Opcode (required).  E.g. ``"ucxx_fetch"``, ``"step_training"``.
        thread: Logical thread / role name (required).
        worker: Optional worker identifier (e.g. ``"rollout-0"``).
        start: Wall-clock-relative start time in milliseconds, if
            present in the trace line.
        end: Wall-clock-relative end time in milliseconds, if present.
        fields: All non-core ``key=value`` pairs preserved as strings.
        raw: The original log line (without trailing newline) for
            debugging.
    """

    op: str
    thread: str
    worker: Optional[str] = None
    start: Optional[float] = None
    end: Optional[float] = None
    fields: Dict[str, str] = field(default_factory=dict)
    raw: str = ""

    @property
    def duration_ms(self) -> Optional[float]:
        if self.start is None or self.end is None:
            return None
        return self.end - self.start

    def get_float(self, key: str) -> Optional[float]:
        """Best-effort numeric extraction from ``fields``.  Returns
        None if the key is missing or unparseable."""
        v = self.fields.get(key)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def get_int(self, key: str) -> Optional[int]:
        v = self.fields.get(key)
        if v is None:
            return None
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None


def _coerce_time(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_trace_line(line: str) -> Optional[TraceEvent]:
    """Parse a single line; return None if it isn't a ``[Trace]`` line.

    The parser is permissive: lines with the sentinel but missing
    required fields (``thread`` and ``op``) are dropped silently
    rather than raising, since logs in the wild often interleave
    partial / truncated entries.
    """
    m = TRACE_LINE_REGEX.search(line.rstrip())
    if not m:
        return None

    body = m.group("body")
    pairs: Dict[str, str] = {}
    for kv in _KV_RE.finditer(body):
        pairs[kv.group("key")] = kv.group("val")

    op = pairs.get("op")
    thread = pairs.get("thread")
    if not op or not thread:
        return None

    extra = {k: v for k, v in pairs.items() if k not in _CORE_FIELDS}
    return TraceEvent(
        op=op,
        thread=thread,
        worker=pairs.get("worker"),
        start=_coerce_time(pairs["start"]) if "start" in pairs else None,
        end=_coerce_time(pairs["end"]) if "end" in pairs else None,
        fields=extra,
        raw=line.rstrip("\n"),
    )


def parse_trace_lines(lines: Iterable[str]) -> Iterator[TraceEvent]:
    """Stream-parse an iterable of log lines, yielding :class:`TraceEvent`."""
    for line in lines:
        ev = parse_trace_line(line)
        if ev is not None:
            yield ev


def parse_trace_log_file(path: str) -> List[TraceEvent]:
    """Parse one log file end-to-end.

    Uses ``errors="replace"`` so we don't choke on the occasional
    binary glyph or BOM that can sneak into rotated log files.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return list(parse_trace_lines(f))


__all__ = [
    "TRACE_LINE_REGEX",
    "TraceEvent",
    "parse_trace_line",
    "parse_trace_lines",
    "parse_trace_log_file",
]
