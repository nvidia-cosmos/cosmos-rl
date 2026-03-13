# Cosmos-RL Profiler Guide

This guide explains how to enable and use runtime profiling for a running RL task.

## Prerequisites

1. Start your Cosmos-RL RL task first.
2. Make sure profiler is enabled in your TOML config.

Example:

```toml
[profiler]
enable_profiler = true

[profiler.sub_profiler_config]
active_steps = 1
warmup_steps = 1
wait_steps = 1
rank_filter = [0]
record_shape = false
profile_memory = false
with_stack = false
with_modules = false
```

Important:
- Profiling commands described below only work when profiler.enable_profiler is true.
- You can tune profiling behavior with profiler.sub_profiler_config.
- The profiler config definitions are in cosmos_rl/policy/config/__init__.py (ProfilerConfig and SubProfilerConfig).

## Step-by-Step Workflow

### 1) List running replicas

Run:

```bash
python -m cosmos_rl.cli.cli replica ls -cp 8000 -ch localhost
```

Notes:
- -cp is the controller port.
- -ch is the controller host.
- Replace 8000 and localhost with your real controller endpoint.

### 2) Pick the target policy replica

From replica ls output, identify:
- Replica name
- Replica role

Choose the policy replica you want to profile.

### 3) Enable profiling for that replica

Run:

```bash
python -m cosmos_rl.cli.cli profile set 04c8f8f4-e4e2-46ac-be3c-28ad63a7c108 -cp 8000 -ch localhost
```

Replace the replica ID, host, and port with your own values.

### 4) Confirm profiler logs on the policy replica

In policy replica logs, you should see messages like:

```text
[Profiler] init profiler for rank 0
[Profiler] start to trace for rank: 0
[Profiler] save trace for rank: 0 to file: ./outputs/.../profile_trace/04c8f8f4-e4e2-46ac-be3c-28ad63a7c108_0/0_trace.json.gz after 3 steps.
```

The save log contains the full trace file path.

### 5) Open the trace in Chrome trace viewer

The saved file is typically trace.json.gz.

You can open it with:
- Perfetto UI: https://ui.perfetto.dev
- Chrome tracing (legacy): chrome://tracing

Perfetto is recommended.

## Quick Troubleshooting

- No profiler logs:
  - Verify profiler.enable_profiler = true.
  - Check rank_filter includes your rank.
- profile set has no visible effect:
  - Re-check controller host/port and replica name.
- No trace file generated:
  - Make sure training continues for at least wait_steps + warmup_steps + active_steps.
