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

"""Argparse-driven CLI for the trace-log analyzer.

Usage::

    python -m cosmos_rl.tools.profiler analyze logs/*.log
    python -m cosmos_rl.tools.profiler analyze logs/*.log --ops step_training,nccl_send
    python -m cosmos_rl.tools.profiler analyze logs/*.log --format json

The CLI is intentionally minimal — backends can layer on plotting /
report rendering by importing :mod:`cosmos_rl.tools.profiler` and
calling :func:`analyze` themselves.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Iterable, List, Optional

from cosmos_rl.tools.profiler.opcodes import OpcodeRegistry
from cosmos_rl.tools.profiler.summary import OpSummary, analyze


def _format_table(rows: Iterable[OpSummary]) -> str:
    rows = list(rows)
    if not rows:
        return "(no [Trace] events found)\n"
    headers = ("op", "count", "total_ms", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms")
    str_rows: List[List[str]] = [list(headers)]
    for r in rows:
        str_rows.append([
            r.op,
            str(r.count),
            f"{r.total_ms:.1f}",
            f"{r.mean_ms:.2f}",
            f"{r.p50_ms:.2f}",
            f"{r.p95_ms:.2f}",
            f"{r.p99_ms:.2f}",
            f"{r.max_ms:.2f}",
        ])
    widths = [max(len(row[i]) for row in str_rows) for i in range(len(headers))]
    out_lines = []
    for row in str_rows:
        out_lines.append(
            "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        )
    return "\n".join(out_lines) + "\n"


def _format_json(rows: Iterable[OpSummary]) -> str:
    return json.dumps(
        [
            {
                "op": r.op,
                "count": r.count,
                "total_ms": r.total_ms,
                "mean_ms": r.mean_ms,
                "p50_ms": r.p50_ms,
                "p95_ms": r.p95_ms,
                "p99_ms": r.p99_ms,
                "max_ms": r.max_ms,
                "extra": r.extra,
            }
            for r in rows
        ],
        indent=2,
    ) + "\n"


def _parse_ops(arg: Optional[str]) -> Optional[List[str]]:
    if not arg:
        return None
    return [s.strip() for s in arg.split(",") if s.strip()]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cosmos_rl.tools.profiler",
        description=(
            "Analyze cosmos-rl [Trace] structured log lines. "
            "Backends register opcode handlers via OpcodeRegistry."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    analyze_p = sub.add_parser("analyze", help="Summarize trace events from log files")
    analyze_p.add_argument("logs", nargs="+", help="Path(s) to log file(s)")
    analyze_p.add_argument(
        "--ops",
        default=None,
        help="Comma-separated opcode filter (default: all opcodes)",
    )
    analyze_p.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Output format (default: table)",
    )
    analyze_p.add_argument(
        "--out",
        default="-",
        help="Output file (default: stdout)",
    )

    list_p = sub.add_parser("list-opcodes", help="List opcodes known to the registry")

    args = parser.parse_args(argv)

    if args.cmd == "list-opcodes":
        for op in OpcodeRegistry.known_opcodes():
            print(op)
        return 0

    summaries = analyze(args.logs, op_filter=_parse_ops(args.ops))
    if args.format == "json":
        out = _format_json(summaries.values())
    else:
        out = _format_table(summaries.values())
    if args.out == "-":
        sys.stdout.write(out)
    else:
        with open(args.out, "w") as f:
            f.write(out)
    return 0


if __name__ == "__main__":  # pragma: no cover - covered via __main__.py
    raise SystemExit(main())
