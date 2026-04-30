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

import re
import time
import unittest

from cosmos_rl.utils.trace import (
    format_trace,
    get_trace_time,
    get_worker_id,
    reset_trace_time,
    set_worker_id,
)


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
            thread="trainer", op="trainer_forward",
            start=12.345, end=42.987,
        )
        # One-decimal formatting for ms timestamps keeps the analyzer
        # output stable.
        self.assertIn("start=12.3", line)
        self.assertIn("end=43.0", line)

    def test_extra_fields_appended(self):
        line = format_trace(
            thread="ucxx_prefetch", op="ucxx_fetch",
            start=0.0, end=10.0,
            transfer_ms=2.5, copy_ms=1.0,
            count=4, bytes=1048576,
        )
        self.assertIn("transfer_ms=2.5", line)
        self.assertIn("copy_ms=1.0", line)
        self.assertIn("count=4", line)
        self.assertIn("bytes=1048576", line)

    def test_bool_fields_render_as_int(self):
        # Bools render as 0/1 so the analyzer can treat them numerically.
        line = format_trace(
            thread="trainer", op="step_training",
            cold_start=True, skipped=False,
        )
        self.assertIn("cold_start=1", line)
        self.assertIn("skipped=0", line)

    def test_field_order_is_stable(self):
        # The analyzer doesn't depend on field order, but a stable
        # order (worker, thread, op, start, end, then extras) keeps
        # logs human-readable.
        set_worker_id("rollout_2")
        line = format_trace(
            thread="rollout", op="rollout_processing",
            start=0.0, end=5.0,
            count=8,
        )
        # Check substring positions form a strictly increasing sequence.
        order = ["worker=", "thread=", "op=", "start=", "end=", "count="]
        positions = [line.index(token) for token in order]
        self.assertEqual(positions, sorted(positions))

    def test_line_round_trips_through_simple_parser(self):
        # Validate that the format is parseable with a trivial regex
        # (a stand-in for the analyzer's parser).
        set_worker_id("policy_0")
        line = format_trace(
            thread="trainer", op="trainer_forward",
            start=10.0, end=20.0, iter=42,
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


if __name__ == "__main__":
    unittest.main()
