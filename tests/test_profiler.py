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

"""Tests for the profiler tooling (MR6).

Covers the parser grammar, the opcode registry's extension contract,
the analyzer summary aggregation (count + percentile + total ordering),
and the CLI's table / JSON / list-opcodes paths.
"""

import io
import json
import os
import tempfile
import unittest

from cosmos_rl.tools.profiler import (
    OpcodeRegistry,
    OpcodeStats,
    TraceEvent,
    analyze,
    parse_trace_line,
    parse_trace_lines,
    parse_trace_log_file,
    summarize_events,
)
from cosmos_rl.tools.profiler import cli as cli_mod


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class TestParseTraceLine(unittest.TestCase):
    def test_minimum_required_fields(self):
        ev = parse_trace_line("[Trace] thread=trainer op=step_training")
        self.assertIsNotNone(ev)
        self.assertEqual(ev.thread, "trainer")
        self.assertEqual(ev.op, "step_training")
        self.assertIsNone(ev.start)
        self.assertIsNone(ev.end)
        self.assertIsNone(ev.duration_ms)
        self.assertEqual(ev.fields, {})

    def test_full_format(self):
        line = (
            "2026-04-30 12:00:00 [Trace] worker=rollout-0 thread=fetch "
            "op=ucxx_fetch start=12.5 end=18.7 bytes=4096 slot=3"
        )
        ev = parse_trace_line(line)
        self.assertEqual(ev.worker, "rollout-0")
        self.assertEqual(ev.thread, "fetch")
        self.assertEqual(ev.op, "ucxx_fetch")
        self.assertAlmostEqual(ev.start, 12.5)
        self.assertAlmostEqual(ev.end, 18.7)
        self.assertAlmostEqual(ev.duration_ms, 6.2, places=3)
        self.assertEqual(ev.fields, {"bytes": "4096", "slot": "3"})
        self.assertEqual(ev.get_int("bytes"), 4096)
        self.assertEqual(ev.get_int("slot"), 3)

    def test_get_float_and_int_handle_missing_keys(self):
        ev = parse_trace_line("[Trace] thread=t op=x")
        self.assertIsNone(ev.get_float("nope"))
        self.assertIsNone(ev.get_int("nope"))

    def test_non_trace_line_returns_none(self):
        self.assertIsNone(parse_trace_line("ordinary log line, no sentinel"))

    def test_missing_op_or_thread_returns_none(self):
        # Sentinel present but missing required field -> permissive None.
        self.assertIsNone(parse_trace_line("[Trace] op=foo"))
        self.assertIsNone(parse_trace_line("[Trace] thread=t"))

    def test_invalid_start_end_become_none(self):
        ev = parse_trace_line("[Trace] thread=t op=x start=NaNish end=oops")
        self.assertIsNotNone(ev)
        self.assertIsNone(ev.start)
        self.assertIsNone(ev.end)

    def test_parse_lines_streams_only_matches(self):
        lines = [
            "noise line\n",
            "[Trace] thread=t op=a start=0 end=1\n",
            "more noise\n",
            "[Trace] thread=t op=b start=2 end=4\n",
        ]
        events = list(parse_trace_lines(lines))
        self.assertEqual(len(events), 2)
        self.assertEqual([e.op for e in events], ["a", "b"])

    def test_parse_log_file_round_trip(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", delete=False
        ) as f:
            f.write("[Trace] thread=t op=alpha start=0 end=10\n")
            f.write("not a trace line\n")
            f.write("[Trace] thread=t op=beta start=10 end=15\n")
            path = f.name
        try:
            events = parse_trace_log_file(path)
            self.assertEqual([e.op for e in events], ["alpha", "beta"])
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# OpcodeRegistry
# ---------------------------------------------------------------------------


class TestOpcodeRegistry(unittest.TestCase):
    def setUp(self):
        OpcodeRegistry.reset()

    def tearDown(self):
        OpcodeRegistry.reset()

    def test_default_opcodes_registered(self):
        known = OpcodeRegistry.known_opcodes()
        self.assertIn("step_training", known)
        self.assertIn("rollout_generation", known)
        self.assertIn("nccl_send", known)
        self.assertIn("redis_publish", known)

    def test_register_known_only_marks_known(self):
        OpcodeRegistry.register("custom_op")
        self.assertTrue(OpcodeRegistry.is_known("custom_op"))
        # No handler registered explicitly -> default duration handler.
        h = OpcodeRegistry.get("custom_op")
        # default_handler will compute durations from start/end.
        events = [TraceEvent(op="custom_op", thread="t", start=0.0, end=5.0)]
        stats = OpcodeStats(op="custom_op", count=1)
        h(events, stats)
        self.assertEqual(stats.durations_ms, [5.0])

    def test_register_with_handler(self):
        captured = {}

        def custom_handler(evs, stats):
            stats.extra["custom_marker"] = len(evs)

        OpcodeRegistry.register("ucxx_fetch", custom_handler)
        events = [
            TraceEvent(op="ucxx_fetch", thread="fetch", start=0.0, end=1.0),
            TraceEvent(op="ucxx_fetch", thread="fetch", start=2.0, end=4.0),
        ]
        summary = summarize_events(events)
        self.assertIn("ucxx_fetch", summary)
        self.assertEqual(summary["ucxx_fetch"].extra["custom_marker"], 2)

    def test_unknown_opcode_still_summarized_via_default(self):
        events = [TraceEvent(op="never_seen", thread="t", start=0.0, end=3.0)]
        summary = summarize_events(events)
        self.assertIn("never_seen", summary)
        self.assertEqual(summary["never_seen"].count, 1)
        self.assertAlmostEqual(summary["never_seen"].total_ms, 3.0)


# ---------------------------------------------------------------------------
# OpcodeStats percentile
# ---------------------------------------------------------------------------


class TestOpcodeStats(unittest.TestCase):
    def test_percentile_with_uniform_data(self):
        s = OpcodeStats(op="x", durations_ms=[1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(s.percentile(0), 1.0)
        self.assertAlmostEqual(s.percentile(50), 3.0)
        self.assertAlmostEqual(s.percentile(100), 5.0)

    def test_percentile_handles_empty(self):
        s = OpcodeStats(op="x")
        self.assertEqual(s.percentile(50), 0.0)
        self.assertEqual(s.mean_ms, 0.0)
        self.assertEqual(s.max_ms, 0.0)
        self.assertEqual(s.min_ms, 0.0)

    def test_percentile_single_sample(self):
        s = OpcodeStats(op="x", durations_ms=[42.0])
        self.assertAlmostEqual(s.percentile(95), 42.0)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class TestAnalyzer(unittest.TestCase):
    def setUp(self):
        # Fresh registry so prior test handlers don't leak.
        OpcodeRegistry.reset()
        self._tmp = tempfile.mkdtemp(prefix="cosmos_rl_prof_")
        self.path = os.path.join(self._tmp, "run.log")
        with open(self.path, "w") as f:
            for s, e in [(0, 10), (10, 25), (25, 30)]:
                f.write(f"[Trace] thread=trainer op=step_training start={s} end={e}\n")
            for s, e in [(0, 5), (5, 7)]:
                f.write(f"[Trace] thread=worker op=rollout_generation start={s} end={e}\n")
            f.write("not a trace line\n")

    def tearDown(self):
        OpcodeRegistry.reset()
        try:
            os.unlink(self.path)
            os.rmdir(self._tmp)
        except OSError:
            pass

    def test_analyze_groups_by_opcode(self):
        out = analyze([self.path])
        self.assertEqual(set(out), {"step_training", "rollout_generation"})

    def test_analyze_sorts_by_total_descending(self):
        out = analyze([self.path])
        ops = list(out)
        # step_training total = 10+15+5 = 30 > rollout 5+2 = 7
        self.assertEqual(ops[0], "step_training")
        self.assertEqual(ops[1], "rollout_generation")

    def test_analyze_aggregates_correctly(self):
        out = analyze([self.path])
        st = out["step_training"]
        self.assertEqual(st.count, 3)
        self.assertAlmostEqual(st.total_ms, 30.0)
        self.assertAlmostEqual(st.mean_ms, 10.0)
        self.assertAlmostEqual(st.max_ms, 15.0)

    def test_op_filter_restricts_summary(self):
        out = analyze([self.path], op_filter=["step_training"])
        self.assertEqual(set(out), {"step_training"})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli(unittest.TestCase):
    def setUp(self):
        OpcodeRegistry.reset()
        self._tmp = tempfile.mkdtemp(prefix="cosmos_rl_prof_cli_")
        self.path = os.path.join(self._tmp, "run.log")
        with open(self.path, "w") as f:
            f.write("[Trace] thread=trainer op=step_training start=0 end=10\n")
            f.write("[Trace] thread=worker op=rollout_generation start=0 end=4\n")

    def tearDown(self):
        OpcodeRegistry.reset()
        try:
            os.unlink(self.path)
            os.rmdir(self._tmp)
        except OSError:
            pass

    def _run(self, argv):
        # Capture stdout via monkeypatch.
        import sys as _sys

        old_stdout = _sys.stdout
        buf = io.StringIO()
        _sys.stdout = buf
        try:
            rc = cli_mod.main(argv)
        finally:
            _sys.stdout = old_stdout
        return rc, buf.getvalue()

    def test_analyze_table_output(self):
        rc, out = self._run(["analyze", self.path])
        self.assertEqual(rc, 0)
        self.assertIn("step_training", out)
        self.assertIn("rollout_generation", out)
        # Header should also be there.
        self.assertIn("op", out.splitlines()[0])

    def test_analyze_json_output(self):
        rc, out = self._run(["analyze", "--format", "json", self.path])
        self.assertEqual(rc, 0)
        data = json.loads(out)
        self.assertEqual(len(data), 2)
        self.assertIn(data[0]["op"], {"step_training", "rollout_generation"})

    def test_analyze_op_filter(self):
        rc, out = self._run([
            "analyze", "--ops", "step_training", self.path,
        ])
        self.assertEqual(rc, 0)
        self.assertIn("step_training", out)
        self.assertNotIn("rollout_generation", out)

    def test_list_opcodes(self):
        rc, out = self._run(["list-opcodes"])
        self.assertEqual(rc, 0)
        opcodes = out.strip().splitlines()
        self.assertIn("step_training", opcodes)
        self.assertIn("nccl_send", opcodes)


if __name__ == "__main__":
    unittest.main()
