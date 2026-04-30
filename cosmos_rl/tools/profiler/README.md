# cosmos-rl profiler tooling

A transport-agnostic log-based profiler for cosmos-rl that parses the
`[Trace]` structured log lines emitted via
`cosmos_rl.utils.trace.format_trace()` and produces per-opcode timing
summaries.

## Quick start

```bash
# Summarize one or more log files
python -m cosmos_rl.tools.profiler analyze logs/*.log

# Filter to a subset of opcodes
python -m cosmos_rl.tools.profiler analyze logs/*.log \
    --ops step_training,rollout_generation,nccl_send

# JSON output for downstream tooling
python -m cosmos_rl.tools.profiler analyze logs/*.log --format json

# List opcodes the registry knows about (defaults + plugin-registered)
python -m cosmos_rl.tools.profiler list-opcodes
```

## Trace line grammar

The parser recognizes lines containing the `[Trace]` sentinel followed
by `key=value` pairs:

```
[Trace] [worker=<id>] thread=<name> op=<op> [start=<ms>] [end=<ms>] [k=v ...]
```

`thread` and `op` are required.  `start` / `end` are floats in
milliseconds (relative to the trace clock).  All other `key=value`
pairs are preserved verbatim in `TraceEvent.fields` so backends can
attach custom metadata (`bytes=`, `slot=`, `transport=`, etc.) without
changing the parser.

See `cosmos-rl/docs/profiler/trace_format.rst` for the canonical
grammar definition.

## Extending: opcode registry

Backends register opcode handlers at import time so new transports
(e.g. UCXX in MR7) can surface backend-specific metrics without
touching the core analyzer.

```python
from cosmos_rl.tools.profiler import OpcodeRegistry

def ucxx_fetch_handler(events, stats):
    # Default duration calculation is automatic for events with
    # start+end; this hook is for surfacing extra metrics.
    total_bytes = sum(int(e.fields.get("bytes", 0)) for e in events)
    stats.extra["total_bytes"] = total_bytes
    stats.extra["mean_bandwidth_gbps"] = (
        total_bytes * 1e3 / (stats.total_ms * 1e9)
        if stats.total_ms > 0 else 0.0
    )
    # Compute durations from start/end as usual.
    for e in events:
        if e.duration_ms is not None:
            stats.durations_ms.append(e.duration_ms)

OpcodeRegistry.register("ucxx_fetch", ucxx_fetch_handler)
```

Handlers receive the full `List[TraceEvent]` for their opcode and an
`OpcodeStats` accumulator they can populate.  Whatever they put into
`stats.extra` shows up in the JSON output and is available to
downstream tools.

## Programmatic use

```python
from cosmos_rl.tools.profiler import analyze

summaries = analyze(["logs/run-1.log", "logs/run-2.log"])
for op, summary in summaries.items():
    print(f"{op:30s} {summary.count:6d} runs  "
          f"p95={summary.p95_ms:8.2f} ms  "
          f"max={summary.max_ms:8.2f} ms")
```

## What's intentionally **not** here

Plot / report rendering is left to downstream consumers — the analyzer
emits structured data (table or JSON) so any report engine
(matplotlib, plotly, custom HTML) can consume it.  This keeps the
upstream package free of heavy plotting dependencies.
