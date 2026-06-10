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

"""Tests for the UCXX-specific profiler extension (MR7).

Importing :mod:`cosmos_rl.tools.profiler.ucxx` registers the UCXX
opcode handlers as a side-effect; tests verify each handler surfaces
the correct metrics.
"""

import io
import os
import tempfile
import unittest

from cosmos_rl.tools.profiler import (
    OpcodeRegistry,
    TraceEvent,
    cli as cli_mod,
    summarize_events,
)
from cosmos_rl.tools.profiler.ucxx import (
    UCXX_OPCODES,
    register_ucxx_opcodes,
)


class _BaseUcxxTest(unittest.TestCase):
    def setUp(self) -> None:
        OpcodeRegistry.reset()
        register_ucxx_opcodes()

    def tearDown(self) -> None:
        OpcodeRegistry.reset()


# ---------------------------------------------------------------------------
# Registration contract
# ---------------------------------------------------------------------------


class TestRegistration(_BaseUcxxTest):
    def test_all_ucxx_opcodes_known(self):
        known = OpcodeRegistry.known_opcodes()
        for op in UCXX_OPCODES:
            self.assertIn(op, known)

    def test_register_is_idempotent(self):
        # Calling twice must not double-register or break dispatch.
        register_ucxx_opcodes()
        register_ucxx_opcodes()
        events = [
            TraceEvent(
                op="ucxx_fetch", thread="f", start=0, end=10, fields={"bytes": "1000"}
            )
        ]
        out = summarize_events(events)
        self.assertIn("ucxx_fetch", out)


# ---------------------------------------------------------------------------
# ucxx_fetch handler
# ---------------------------------------------------------------------------


class TestUcxxFetch(_BaseUcxxTest):
    def test_total_bytes_aggregated(self):
        events = [
            TraceEvent(
                op="ucxx_fetch",
                thread="f",
                start=0,
                end=10,
                fields={"bytes": "1000", "chunks": "4"},
            ),
            TraceEvent(
                op="ucxx_fetch",
                thread="f",
                start=10,
                end=20,
                fields={"bytes": "2000", "chunks": "8"},
            ),
        ]
        out = summarize_events(events)
        s = out["ucxx_fetch"]
        self.assertEqual(s.extra["total_bytes"], 3000)
        self.assertEqual(s.extra["total_chunks"], 12)
        self.assertEqual(s.count, 2)
        self.assertAlmostEqual(s.total_ms, 20.0)

    def test_bandwidth_calculation(self):
        # 1 MB transferred over 10 ms = 1e6 bytes / 10ms = 100 MB/s.
        # In Gbps: 1e6 * 8 / (10 * 1e6) = 0.8 Gbps.
        events = [
            TraceEvent(
                op="ucxx_fetch",
                thread="f",
                start=0,
                end=10,
                fields={"bytes": "1000000"},
            ),
        ]
        out = summarize_events(events)
        s = out["ucxx_fetch"]
        self.assertAlmostEqual(s.extra["mean_bandwidth_gbps"], 0.8, places=4)
        self.assertAlmostEqual(s.extra["peak_bandwidth_gbps"], 0.8, places=4)

    def test_no_bytes_field_omits_bandwidth(self):
        events = [
            TraceEvent(op="ucxx_fetch", thread="f", start=0, end=10),
        ]
        out = summarize_events(events)
        s = out["ucxx_fetch"]
        self.assertEqual(s.extra["total_bytes"], 0)
        self.assertNotIn("mean_bandwidth_gbps", s.extra)


# ---------------------------------------------------------------------------
# ucxx_send handler (delegates to the fetch handler)
# ---------------------------------------------------------------------------


class TestUcxxSend(_BaseUcxxTest):
    def test_send_uses_fetch_handler_shape(self):
        events = [
            TraceEvent(
                op="ucxx_send",
                thread="s",
                start=0,
                end=5,
                fields={"bytes": "5000"},
            ),
        ]
        out = summarize_events(events)
        self.assertIn("ucxx_send", out)
        self.assertEqual(out["ucxx_send"].extra["total_bytes"], 5000)


# ---------------------------------------------------------------------------
# ucxx_prefetch_collect handler
# ---------------------------------------------------------------------------


class TestUcxxPrefetch(_BaseUcxxTest):
    def test_wait_rate_and_mean_wait(self):
        events = [
            TraceEvent(
                op="ucxx_prefetch_collect",
                thread="t",
                start=0,
                end=1,
                fields={"waited_ms": "10.0"},
            ),
            TraceEvent(
                op="ucxx_prefetch_collect",
                thread="t",
                start=1,
                end=1.5,
                fields={"waited_ms": "20.0"},
            ),
            TraceEvent(
                op="ucxx_prefetch_collect",
                thread="t",
                start=2,
                end=2.1,
                fields={"waited_ms": "0.0"},
            ),
            TraceEvent(
                op="ucxx_prefetch_collect",
                thread="t",
                start=3,
                end=3.1,
                # waited_ms not present at all -> counts as no-wait.
            ),
        ]
        out = summarize_events(events)
        s = out["ucxx_prefetch_collect"]
        self.assertEqual(s.extra["waited_count"], 2)
        self.assertEqual(s.extra["no_wait_count"], 2)
        self.assertAlmostEqual(s.extra["wait_rate"], 0.5)
        self.assertAlmostEqual(s.extra["mean_wait_ms"], 15.0)


# ---------------------------------------------------------------------------
# ucxx_stale_slot handler
# ---------------------------------------------------------------------------


class TestUcxxStaleSlot(_BaseUcxxTest):
    def test_top_worker_ips(self):
        events = [
            TraceEvent(
                op="ucxx_stale_slot",
                thread="r",
                start=0,
                end=0.1,
                fields={"worker_ip": "10.0.0.1"},
            )
            for _ in range(5)
        ] + [
            TraceEvent(
                op="ucxx_stale_slot",
                thread="r",
                start=1,
                end=1.1,
                fields={"worker_ip": "10.0.0.2"},
            )
            for _ in range(2)
        ]
        out = summarize_events(events)
        s = out["ucxx_stale_slot"]
        self.assertEqual(s.count, 7)
        top = dict(s.extra["top_worker_ips"])
        self.assertEqual(top["10.0.0.1"], 5)
        self.assertEqual(top["10.0.0.2"], 2)


# ---------------------------------------------------------------------------
# CLI --ucxx flag
# ---------------------------------------------------------------------------


class TestCliUcxxFlag(unittest.TestCase):
    def setUp(self):
        OpcodeRegistry.reset()
        self._tmp = tempfile.mkdtemp(prefix="cosmos_rl_prof_ucxx_")
        self.path = os.path.join(self._tmp, "run.log")
        with open(self.path, "w") as f:
            f.write("[Trace] thread=fetch op=ucxx_fetch start=0 end=10 bytes=1000000\n")

    def tearDown(self):
        OpcodeRegistry.reset()
        try:
            os.unlink(self.path)
            os.rmdir(self._tmp)
        except OSError:
            pass

    def _run(self, argv):
        import sys as _sys

        old = _sys.stdout
        buf = io.StringIO()
        _sys.stdout = buf
        try:
            rc = cli_mod.main(argv)
        finally:
            _sys.stdout = old
        return rc, buf.getvalue()

    def test_list_opcodes_with_ucxx_includes_ucxx(self):
        rc, out = self._run(["list-opcodes", "--ucxx"])
        self.assertEqual(rc, 0)
        opcodes = out.strip().splitlines()
        self.assertIn("ucxx_fetch", opcodes)
        self.assertIn("ucxx_prefetch_collect", opcodes)

    def test_analyze_with_ucxx_emits_bandwidth_in_json(self):
        rc, out = self._run(["analyze", "--ucxx", "--format", "json", self.path])
        import json as _json

        self.assertEqual(rc, 0)
        data = _json.loads(out)
        rec = next(r for r in data if r["op"] == "ucxx_fetch")
        self.assertIn("total_bytes", rec["extra"])
        self.assertEqual(rec["extra"]["total_bytes"], 1000000)
        self.assertIn("mean_bandwidth_gbps", rec["extra"])


# ---------------------------------------------------------------------------
# check_ucxx — minimal smoke
# ---------------------------------------------------------------------------


class TestCheckUcxxSmoke(unittest.TestCase):
    def test_module_imports(self):
        # The diagnostic uses ctypes / subprocess only at run time, so
        # importing it must succeed even on machines without UCX.
        from cosmos_rl.tools.profiler import check_ucxx  # noqa: F401


if __name__ == "__main__":
    unittest.main()
