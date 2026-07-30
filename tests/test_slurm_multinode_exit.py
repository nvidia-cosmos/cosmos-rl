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

"""The multi-node sbatch template must not retry a completed run.

``COSMOS_SHUTDOWN_ON_NO_POLICY_REPLICAS`` makes the controller SIGTERM *itself*
once the last policy replica unregisters.  That is how a successful run ends,
but it reaches the batch script as 128+SIGTERM.  Read as a crash, it sent the
job through its whole retry budget re-running finished training and then
reported FAILED -- observed on a real 2-node run: four full 500-step runs, then
FAILED.

``tests/test_launcher_shutdown.py`` covers the same predicate for the
single-node CLI launcher.  This one covers the sbatch template, which is shell
and has no import surface, so the functions are extracted and driven directly.

Excusing too much would be worse than the original bug, so the negative cases
matter as much as the positive one: a SIGTERM from ``scancel``, from
pre-emption, or from the wall-clock limit must still fail loudly.
"""

import re
import signal
import subprocess
import unittest
from pathlib import Path

TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "cosmos_rl"
    / "tools"
    / "slurm"
    / "cosmos_rl_job_multi_node.sh"
)

SIGTERM_RC = 128 + signal.SIGTERM  # 143


def _extract(func_name: str) -> str:
    """Pull one shell function out of the template.

    The template is an sbatch script: sourcing it would run a job.  These
    functions are self-contained, so lifting them by text is enough to drive
    them, and it keeps the test honest about which definition it is checking.
    """
    src = TEMPLATE.read_text()
    match = re.search(
        rf"^{re.escape(func_name)}\(\) \{{.*?^\}}$",
        src,
        re.MULTILINE | re.DOTALL,
    )
    assert match, f"{func_name} not found in {TEMPLATE}"
    return match.group(0)


def _classify(code, *, flag=None, received_signal=""):
    """Run ``is_coordinated_controller_exit`` and return whether it excused."""
    script = "\n".join(
        [
            "set -u",
            f'received_signal="{received_signal}"',
            (
                f'COSMOS_SHUTDOWN_ON_NO_POLICY_REPLICAS="{flag}"'
                if flag is not None
                else "unset COSMOS_SHUTDOWN_ON_NO_POLICY_REPLICAS || true"
            ),
            _extract("is_coordinated_controller_exit"),
            f'if is_coordinated_controller_exit "{code}"; then echo YES; else echo NO; fi',
        ]
    )
    out = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 0, f"harness failed: {out.stderr}"
    return out.stdout.strip() == "YES"


class TestCoordinatedControllerExit(unittest.TestCase):
    def test_self_sigterm_with_feature_on_is_success(self):
        """The case that made every completed multi-node run report FAILED."""
        self.assertTrue(_classify(SIGTERM_RC, flag="1"))

    def test_truthiness_matches_the_python_constant(self):
        """utils/constant.py accepts 1/true/yes, case-insensitively."""
        for value in ("1", "true", "TRUE", "yes", "Yes"):
            self.assertTrue(_classify(SIGTERM_RC, flag=value), value)
        for value in ("0", "", "no", "false", "off", "2"):
            self.assertFalse(_classify(SIGTERM_RC, flag=value), value)

    def test_feature_off_or_unset_still_fails(self):
        self.assertFalse(_classify(SIGTERM_RC, flag="0"))
        self.assertFalse(_classify(SIGTERM_RC, flag=None))

    def test_other_exit_codes_are_never_excused(self):
        # 137 = SIGKILL (OOM reaper), 1 = ordinary crash, 0 handled upstream.
        for code in (1, 2, 137, 139, 255):
            self.assertFalse(_classify(code, flag="1"), code)

    def test_scancel_still_fails(self):
        """``scancel`` traps SIGTERM on the batch script itself."""
        self.assertFalse(_classify(SIGTERM_RC, flag="1", received_signal="SIGTERM"))

    def test_wall_clock_pre_timeout_still_fails(self):
        """``--signal=B:SIGUSR1@...`` fires before the time limit.

        Without the received_signal clause a job killed at its wall-clock
        limit would report success, trading one false verdict for another.
        """
        self.assertFalse(_classify(SIGTERM_RC, flag="1", received_signal="SIGUSR1"))


class TestAutoRetryGate(unittest.TestCase):
    """A success classification must actually stop the requeue."""

    @staticmethod
    def _retry_outcome(status, received_signal=""):
        script = "\n".join(
            [
                "set -u",
                f'received_signal="{received_signal}"',
                'latest_part_dir="/nonexistent"',
                'log() { echo "$@"; }',
                "scontrol() { echo SCONTROL_CALLED; }",
                _extract("handle_auto_retry"),
                f"handle_auto_retry {status}",
                'echo "rc=$?"',
            ]
        )
        out = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, timeout=60
        )
        return out.stdout

    def test_status_zero_never_requeues(self):
        out = self._retry_outcome(0)
        self.assertNotIn("SCONTROL_CALLED", out)
        self.assertIn("rc=0", out)

    def test_nonzero_without_retries_left_does_not_requeue(self):
        # remaining-retries file is absent -> treated as 0 remaining.
        out = self._retry_outcome(1)
        self.assertNotIn("SCONTROL_CALLED", out)
        self.assertIn("No retries remaining", out)


if __name__ == "__main__":
    unittest.main()
