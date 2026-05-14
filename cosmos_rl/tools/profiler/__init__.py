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

"""``cosmos_rl.tools.profiler`` — log-based profiling tools.

This package is **transport-agnostic**: it parses the ``[Trace]``
structured log lines emitted by :mod:`cosmos_rl.utils.trace` and
groups them by opcode for high-level analysis.

Extensibility
-------------

Backends (NCCL, UCXX, future transports) register themselves with
:class:`OpcodeRegistry` so the analyzer can recognize transport-
specific opcodes (e.g. ``ucxx_fetch``) without modifying the core
parser.  See :func:`OpcodeRegistry.register` for the public extension
hook used by MR7's UCXX support.

Public surface
--------------

* :class:`TraceEvent` — parsed event dict (start, end, op, fields).
* :func:`parse_trace_lines` / :func:`parse_trace_log_file` — the parser.
* :class:`OpcodeRegistry` — pluggable opcode handler registry.
* :func:`analyze` / :func:`summarize_events` — top-level entry points.
* :mod:`.cli` — argparse-driven CLI; see ``python -m
  cosmos_rl.tools.profiler --help``.
"""

from cosmos_rl.tools.profiler.opcodes import (
    OpcodeHandler,
    OpcodeRegistry,
    OpcodeStats,
)
from cosmos_rl.tools.profiler.parser import (
    TRACE_LINE_REGEX,
    TraceEvent,
    parse_trace_line,
    parse_trace_lines,
    parse_trace_log_file,
)
from cosmos_rl.tools.profiler.summary import (
    OpSummary,
    analyze,
    summarize_events,
)


__all__ = [
    "OpSummary",
    "OpcodeHandler",
    "OpcodeRegistry",
    "OpcodeStats",
    "TRACE_LINE_REGEX",
    "TraceEvent",
    "analyze",
    "parse_trace_line",
    "parse_trace_lines",
    "parse_trace_log_file",
    "summarize_events",
]
