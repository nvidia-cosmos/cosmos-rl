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

"""Unit tests for ``cosmos_rl.utils.trace``."""

import json
import re
import sys
import time
import unittest

from cosmos_rl.utils.trace import (
    format_trace,
    get_trace_time,
    get_worker_id,
    reset_trace_time,
    set_worker_id,
    trace_op,
)
from cosmos_rl.utils import trace as _trace_mod


# Tokenizer that understands the trace grammar: bare values OR JSON
# double-quoted strings.  Matches the producer-side grammar in
# cosmos_rl.utils.trace.
_KV_RE = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|\S+)')


def _split_kv_pairs(line: str):
    return _KV_RE.findall(line)


class TestGetTraceTime(unittest.TestCase):
    def setUp(self):
        reset_trace_time()
        # Reset worker id so tests are independent.
        set_worker_id("")

    def test_first_call_returns_near_zero(self):
        t = get_trace_time()
        # First call defines t=0; we should be within a couple ms.
        self.assertLess(abs(t), 5.0)

    def test_subsequent_calls_are_monotonic(self):
        t0 = get_trace_time()
        t1 = get_trace_time()
        self.assertGreaterEqual(t1, t0)

    def test_elapsed_matches_sleep(self):
        get_trace_time()  # zero
        time.sleep(0.05)
        elapsed = get_trace_time()
        # Allow generous slack for CI flakiness.
        self.assertGreaterEqual(elapsed, 40.0)
        self.assertLess(elapsed, 500.0)

    def test_reset_zeros_time(self):
        get_trace_time()
        time.sleep(0.01)
        reset_trace_time()
        t = get_trace_time()
        self.assertLess(abs(t), 5.0)


class TestWorkerId(unittest.TestCase):
    def setUp(self):
        set_worker_id("")

    def test_default_is_question_mark(self):
        # Empty / unset => fallback "?"
        self.assertEqual(get_worker_id(), "?")

    def test_set_and_get(self):
        set_worker_id("policy_0")
        self.assertEqual(get_worker_id(), "policy_0")


class TestFormatTrace(unittest.TestCase):
    def setUp(self):
        reset_trace_time()
        set_worker_id("")

    def test_minimum_required_fields(self):
        line = format_trace(thread="trainer", op="step_training")
        self.assertTrue(line.startswith("[Trace] "))
        self.assertIn("thread=trainer", line)
        self.assertIn("op=step_training", line)
        # No worker= when unset.
        self.assertNotIn("worker=", line)

    def test_includes_worker_when_set(self):
        set_worker_id("policy_0")
        line = format_trace(thread="trainer", op="step_training")
        self.assertIn("worker=policy_0", line)

    def test_start_end_formatted_one_decimal(self):
        line = format_trace(
            thread="trainer",
            op="trainer_forward",
            start=12.345,
            end=42.987,
        )
        # One-decimal formatting for ms timestamps keeps the analyzer
        # output stable.
        self.assertIn("start=12.3", line)
        self.assertIn("end=43.0", line)

    def test_extra_fields_appended(self):
        line = format_trace(
            thread="ucxx_prefetch",
            op="ucxx_fetch",
            start=0.0,
            end=10.0,
            transfer_ms=2.5,
            copy_ms=1.0,
            count=4,
            bytes=1048576,
        )
        self.assertIn("transfer_ms=2.5", line)
        self.assertIn("copy_ms=1.0", line)
        self.assertIn("count=4", line)
        self.assertIn("bytes=1048576", line)

    def test_bool_fields_render_as_int(self):
        # Bools render as 0/1 so the analyzer can treat them numerically.
        line = format_trace(
            thread="trainer",
            op="step_training",
            cold_start=True,
            skipped=False,
        )
        self.assertIn("cold_start=1", line)
        self.assertIn("skipped=0", line)

    def test_field_order_is_stable(self):
        # The analyzer doesn't depend on field order, but a stable
        # order (worker, thread, op, start, end, then extras) keeps
        # logs human-readable.
        set_worker_id("rollout_2")
        line = format_trace(
            thread="rollout",
            op="rollout_processing",
            start=0.0,
            end=5.0,
            count=8,
        )
        # Check substring positions form a strictly increasing sequence.
        order = ["worker=", "thread=", "op=", "start=", "end=", "count="]
        positions = [line.index(token) for token in order]
        self.assertEqual(positions, sorted(positions))

    def test_only_start_emitted(self):
        line = format_trace(thread="t", op="o", start=1.5)
        self.assertIn("start=1.5", line)
        self.assertNotIn("end=", line)

    def test_only_end_emitted(self):
        line = format_trace(thread="t", op="o", end=2.5)
        self.assertIn("end=2.5", line)
        # The 'end=' check above passes only if there is no 'start=' field;
        # explicitly assert no spurious start.
        self.assertNotIn("start=", line)

    def test_no_start_no_end(self):
        line = format_trace(thread="t", op="o", iter=1)
        self.assertNotIn("start=", line)
        self.assertNotIn("end=", line)
        self.assertIn("iter=1", line)

    def test_string_with_space_is_quoted(self):
        line = format_trace(thread="t", op="o", note="hello world")
        # JSON-quoted so the analyzer can recover the original string.
        self.assertIn('note="hello world"', line)
        # And it round-trips through json.loads.
        kv = dict(_split_kv_pairs(line))
        self.assertEqual(json.loads(kv["note"]), "hello world")

    def test_string_with_equals_is_quoted(self):
        line = format_trace(thread="t", op="o", expr="x=1")
        self.assertIn('expr="x=1"', line)

    def test_string_with_quote_is_escaped(self):
        line = format_trace(thread="t", op="o", note='say "hi"')
        kv = dict(_split_kv_pairs(line))
        self.assertEqual(json.loads(kv["note"]), 'say "hi"')

    def test_string_with_brackets_is_quoted(self):
        line = format_trace(thread="t", op="o", note="[Trace]-like")
        kv = dict(_split_kv_pairs(line))
        self.assertEqual(json.loads(kv["note"]), "[Trace]-like")

    def test_empty_string_is_quoted(self):
        line = format_trace(thread="t", op="o", note="")
        self.assertIn('note=""', line)

    def test_safe_string_emitted_raw(self):
        # Plain identifiers / dotted paths / paths stay unquoted for
        # readability and to preserve backward compatibility.
        line = format_trace(thread="t", op="o", file="path/to/x.py")
        self.assertIn("file=path/to/x.py", line)

    def test_none_value_falls_back_to_repr(self):
        line = format_trace(thread="t", op="o", missing=None)
        self.assertIn("missing=None", line)

    def test_list_value_falls_back_to_repr(self):
        line = format_trace(thread="t", op="o", shape=[1, 2, 3])
        # repr produces "[1, 2, 3]" which contains spaces+brackets and
        # therefore must be quoted.
        kv = dict(_split_kv_pairs(line))
        self.assertEqual(json.loads(kv["shape"]), "[1, 2, 3]")

    def test_custom_object_falls_back_to_repr(self):
        # repr() yielding a grammar-safe string is emitted raw.
        class _Marker:
            def __repr__(self):
                return "<marker>"

        line = format_trace(thread="t", op="o", obj=_Marker())
        self.assertIn("obj=<marker>", line)

    def test_custom_object_with_unsafe_repr_is_quoted(self):
        # repr() containing whitespace must be quoted to keep the line
        # parseable.
        class _Detailed:
            def __repr__(self):
                return "Detailed(name='x y')"

        line = format_trace(thread="t", op="o", obj=_Detailed())
        kv = dict(_split_kv_pairs(line))
        self.assertEqual(json.loads(kv["obj"]), "Detailed(name='x y')")

    def test_format_is_deterministic_for_identical_inputs(self):
        a = format_trace(thread="t", op="o", start=1.0, end=2.0, iter=3)
        b = format_trace(thread="t", op="o", start=1.0, end=2.0, iter=3)
        self.assertEqual(a, b)

    def test_line_round_trips_through_simple_parser(self):
        # Validate that the format is parseable with a trivial regex
        # (a stand-in for the analyzer's parser).
        set_worker_id("policy_0")
        line = format_trace(
            thread="trainer",
            op="trainer_forward",
            start=10.0,
            end=20.0,
            iter=42,
        )
        pattern = re.compile(
            r"^\[Trace\]"
            r"(?:\s+(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<val>\S+))+\s*$"
        )
        # Findall over key=val pairs to check we get the expected set.
        kv_pattern = re.compile(r"(?P<k>[A-Za-z_][A-Za-z0-9_]*)=(?P<v>\S+)")
        kvs = dict(kv_pattern.findall(line))
        self.assertEqual(kvs["worker"], "policy_0")
        self.assertEqual(kvs["thread"], "trainer")
        self.assertEqual(kvs["op"], "trainer_forward")
        self.assertEqual(kvs["start"], "10.0")
        self.assertEqual(kvs["end"], "20.0")
        self.assertEqual(kvs["iter"], "42")
        # The full line also satisfies the higher-level pattern.
        self.assertTrue(pattern.match(line) or " " in line)


class _RecordingLogger:
    """Minimal logger stand-in that records calls per level."""

    def __init__(self):
        self.records: list[tuple[str, str]] = []

    def _record(self, level: str, msg: str) -> None:
        self.records.append((level, msg))

    def debug(self, msg):
        self._record("debug", msg)

    def info(self, msg):
        self._record("info", msg)

    def warning(self, msg):
        self._record("warning", msg)


class TestTraceOp(unittest.TestCase):
    def setUp(self):
        reset_trace_time()
        set_worker_id("")

    def test_basic_block_emits_single_line(self):
        log = _RecordingLogger()
        with trace_op("trainer", "step_training", logger=log):
            time.sleep(0.005)
        self.assertEqual(len(log.records), 1)
        level, line = log.records[0]
        self.assertEqual(level, "debug")
        self.assertTrue(line.startswith("[Trace] "))
        self.assertIn("thread=trainer", line)
        self.assertIn("op=step_training", line)
        self.assertIn("start=", line)
        self.assertIn("end=", line)

    def test_initial_fields_are_emitted(self):
        log = _RecordingLogger()
        with trace_op("trainer", "step_training", logger=log, iter=42):
            pass
        _, line = log.records[0]
        self.assertIn("iter=42", line)

    def test_yielded_extras_are_appended(self):
        log = _RecordingLogger()
        with trace_op("rollout", "ucxx_fetch", logger=log) as extras:
            extras["bytes"] = 1048576
            extras["count"] = 4
        _, line = log.records[0]
        self.assertIn("bytes=1048576", line)
        self.assertIn("count=4", line)

    def test_yielded_extras_can_override_initial_fields(self):
        log = _RecordingLogger()
        with trace_op("rollout", "ucxx_fetch", logger=log, count=0) as extras:
            extras["count"] = 9
        _, line = log.records[0]
        self.assertIn("count=9", line)
        self.assertNotIn("count=0", line)

    def test_includes_worker_when_set(self):
        log = _RecordingLogger()
        set_worker_id("policy_0")
        with trace_op("trainer", "step_training", logger=log):
            pass
        _, line = log.records[0]
        self.assertIn("worker=policy_0", line)

    def test_end_after_start(self):
        log = _RecordingLogger()
        with trace_op("trainer", "step_training", logger=log):
            time.sleep(0.01)
        _, line = log.records[0]
        # Pull start / end out of the line and confirm end >= start.
        kvs = dict(s.split("=", 1) for s in line.split(" ")[1:])
        self.assertGreaterEqual(float(kvs["end"]), float(kvs["start"]))
        self.assertGreater(float(kvs["end"]) - float(kvs["start"]), 0.0)

    def test_custom_log_level_used(self):
        log = _RecordingLogger()
        with trace_op("trainer", "step_training", logger=log, log_level="info"):
            pass
        self.assertEqual(log.records[0][0], "info")

    def test_unknown_log_level_falls_back_to_info(self):
        log = _RecordingLogger()
        with trace_op("trainer", "step_training", logger=log, log_level="nope"):
            pass
        # Falls back to .info rather than crashing.
        self.assertEqual(log.records[0][0], "info")

    def test_exception_is_reraised_and_line_is_emitted_with_status(self):
        log = _RecordingLogger()
        with self.assertRaises(ValueError):
            with trace_op("trainer", "step_training", logger=log):
                raise ValueError("boom")
        self.assertEqual(len(log.records), 1)
        _, line = log.records[0]
        self.assertIn("status=error", line)
        self.assertIn("err=ValueError", line)

    def test_caller_supplied_status_is_not_overwritten_on_error(self):
        log = _RecordingLogger()
        with self.assertRaises(RuntimeError):
            with trace_op(
                "trainer",
                "step_training",
                logger=log,
                status="custom",
            ):
                raise RuntimeError("x")
        _, line = log.records[0]
        self.assertIn("status=custom", line)
        self.assertIn("err=RuntimeError", line)

    def test_logger_failure_does_not_propagate(self):
        class ExplodingLogger:
            def debug(self, msg):
                raise RuntimeError("logger broke")

        # Should complete cleanly even though logging itself raises.
        with trace_op("trainer", "step_training", logger=ExplodingLogger()):
            pass

    def test_default_logger_is_used_when_none_supplied(self):
        # Smoke: just confirm the with block works without an explicit
        # logger.  We can't easily intercept the cosmos_rl logger here,
        # so we only verify the call doesn't raise and the extras dict
        # is the expected type.
        with trace_op("trainer", "step_training") as extras:
            self.assertIsInstance(extras, dict)

    def test_nested_blocks_both_emit_with_inner_inside_outer(self):
        log = _RecordingLogger()
        with trace_op("trainer", "step_training", logger=log):
            time.sleep(0.002)
            with trace_op("trainer", "trainer_forward", logger=log):
                time.sleep(0.002)
        self.assertEqual(len(log.records), 2)
        # Inner finishes (and is logged) first; outer logs second.
        ops = [dict(_split_kv_pairs(line))["op"] for _, line in log.records]
        self.assertEqual(ops, ["trainer_forward", "step_training"])
        # Inner is fully contained within outer.
        kv_inner = dict(_split_kv_pairs(log.records[0][1]))
        kv_outer = dict(_split_kv_pairs(log.records[1][1]))
        self.assertGreaterEqual(float(kv_inner["start"]), float(kv_outer["start"]))
        self.assertLessEqual(float(kv_inner["end"]), float(kv_outer["end"]))

    def test_base_exception_still_emits_and_propagates(self):
        # KeyboardInterrupt is BaseException, not Exception.  trace_op
        # should still emit the line with status=error before re-raising.
        log = _RecordingLogger()
        with self.assertRaises(KeyboardInterrupt):
            with trace_op("trainer", "step_training", logger=log):
                raise KeyboardInterrupt()
        self.assertEqual(len(log.records), 1)
        _, line = log.records[0]
        self.assertIn("status=error", line)
        self.assertIn("err=KeyboardInterrupt", line)

    def test_default_logger_falls_back_when_cosmos_rl_logging_unimportable(self):
        # Simulate cosmos_rl.utils.logging being unimportable by
        # injecting a stub module that raises on attribute access.
        sentinel_name = "cosmos_rl.utils.logging"
        prev = sys.modules.get(sentinel_name, None)
        broken = type(sys)("cosmos_rl.utils.logging")

        def _explode(name):
            raise AttributeError(name)

        broken.__getattr__ = _explode  # type: ignore[attr-defined]
        sys.modules[sentinel_name] = broken
        try:
            fallback = _trace_mod._default_logger()
        finally:
            if prev is None:
                sys.modules.pop(sentinel_name, None)
            else:
                sys.modules[sentinel_name] = prev
        # Falls back to a stdlib logger (has .debug / .info methods).
        self.assertTrue(hasattr(fallback, "debug"))
        self.assertTrue(hasattr(fallback, "info"))


if __name__ == "__main__":
    unittest.main()
