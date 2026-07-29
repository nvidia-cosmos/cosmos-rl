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

"""CPU-only tests for the subprocess wait/teardown helpers.

``wait_all_or_fail`` is what bounds the GPU integration suites, so it is worth
exercising against real processes rather than mocks: the failure it exists to
prevent (an unbounded wait consuming the whole suite budget) only shows up when
a process genuinely refuses to exit.
"""

import os
import subprocess
import sys
import time
import unittest

from subprocess_helpers import kill_process_group, resolve_timeout, wait_all_or_fail


def _spawn(code, new_session=False):
    """Run ``code`` in a child python, output merged into ours."""
    return subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=sys.stderr,
        stderr=sys.stderr,
        start_new_session=new_session,
    )


_SLEEP_FOREVER = "import time; time.sleep(600)"


class _Recorder:
    """Stand-in for a TestCase that records failures instead of raising."""

    def __init__(self):
        self.failures = []
        self.mismatches = []

    def fail(self, msg):
        self.failures.append(msg)
        raise AssertionError(msg)

    def assertEqual(self, actual, expected, msg=None):
        if actual != expected:
            self.mismatches.append(msg or f"{actual!r} != {expected!r}")
            raise AssertionError(msg or f"{actual!r} != {expected!r}")


class TestResolveTimeout(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop("COSMOS_TEST_TIMEOUT_SCALE", None)

    def tearDown(self):
        os.environ.pop("COSMOS_TEST_TIMEOUT_SCALE", None)
        if self._saved is not None:
            os.environ["COSMOS_TEST_TIMEOUT_SCALE"] = self._saved

    def test_unset_is_identity(self):
        self.assertEqual(resolve_timeout(600), 600)

    def test_scale_multiplies(self):
        os.environ["COSMOS_TEST_TIMEOUT_SCALE"] = "2.5"
        self.assertEqual(resolve_timeout(600), 1500)

    def test_garbage_falls_back_to_identity(self):
        # A typo in the env must not silently zero out every budget.
        for bad in ("", "abc", "0", "-3"):
            os.environ["COSMOS_TEST_TIMEOUT_SCALE"] = bad
            self.assertEqual(resolve_timeout(600), 600, f"scale={bad!r}")


class TestWaitAllOrFail(unittest.TestCase):
    def test_clean_exit_passes(self):
        procs = [_spawn("pass"), _spawn("pass")]
        wait_all_or_fail(self, procs, timeout_s=60, context="clean")
        for p in procs:
            self.assertEqual(p.returncode, 0)

    def test_nonzero_exit_fails(self):
        rec = _Recorder()
        with self.assertRaises(AssertionError):
            wait_all_or_fail(rec, [_spawn("raise SystemExit(3)")], 60, "nonzero")
        self.assertTrue(rec.mismatches, "should have reported a bad exit code")

    def test_hang_is_bounded_and_process_killed(self):
        proc = _spawn(_SLEEP_FOREVER)
        rec = _Recorder()
        started = time.monotonic()
        with self.assertRaises(AssertionError):
            wait_all_or_fail(rec, [proc], timeout_s=2, context="hang")
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 30, "wait was not bounded by timeout_s")
        self.assertTrue(any("Timed out" in m for m in rec.failures), rec.failures)
        # The helper must also reap: a surviving child would keep holding GPUs
        # (and, under torchrun, a rendezvous port) for the rest of the suite.
        self.assertIsNotNone(proc.poll(), "hung process was left running")

    def test_deadline_is_shared_not_per_process(self):
        """Three hung processes must not each get the full budget."""
        procs = [_spawn(_SLEEP_FOREVER) for _ in range(3)]
        rec = _Recorder()
        started = time.monotonic()
        with self.assertRaises(AssertionError):
            wait_all_or_fail(rec, procs, timeout_s=2, context="shared-deadline")
        elapsed = time.monotonic() - started

        # Per-process timeouts would take ~3x the budget before failing.
        self.assertLess(elapsed, 30, f"deadline was not shared (took {elapsed:.1f}s)")
        for p in procs:
            self.assertIsNotNone(p.poll(), "process left running")

    def test_later_process_still_reaped_when_earlier_one_hangs(self):
        """The finally-block must reap peers the wait never got to."""
        hung, peer = _spawn(_SLEEP_FOREVER), _spawn(_SLEEP_FOREVER)
        rec = _Recorder()
        with self.assertRaises(AssertionError):
            wait_all_or_fail(rec, [hung, peer], timeout_s=2, context="peer-reap")
        self.assertIsNotNone(hung.poll(), "hung process left running")
        self.assertIsNotNone(peer.poll(), "peer process left running")


class TestKillProcessGroup(unittest.TestCase):
    def test_does_not_kill_our_own_process_group(self):
        """A child sharing our process group must not take the runner down.

        ``Popen`` without ``start_new_session=True`` leaves the child in the
        runner's group; an unguarded ``killpg`` there would SIGKILL pytest.
        Reaching the assertion below at all is the real assertion.
        """
        child = _spawn(_SLEEP_FOREVER, new_session=False)
        self.assertEqual(os.getpgid(child.pid), os.getpgrp(), "precondition")

        kill_process_group(child)

        self.assertIsNotNone(child.poll(), "child should have been killed")

    def test_kills_a_child_that_leads_its_own_group(self):
        child = _spawn(_SLEEP_FOREVER, new_session=True)
        self.assertNotEqual(os.getpgid(child.pid), os.getpgrp(), "precondition")

        kill_process_group(child)

        self.assertIsNotNone(child.poll(), "child should have been killed")

    def test_already_exited_process_is_a_noop(self):
        child = _spawn("pass")
        child.wait()
        kill_process_group(child)
        self.assertEqual(child.returncode, 0)


if __name__ == "__main__":
    unittest.main()
