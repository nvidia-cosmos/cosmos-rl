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

"""A coordinated controller shutdown is a success, not a job failure.

``COSMOS_SHUTDOWN_ON_NO_POLICY_REPLICAS`` makes the controller SIGTERM *itself*
once the last policy replica unregisters, so the scheduler reclaims the
allocation instead of idling to wall-clock.  The launcher used to treat that
non-zero return code as a crash, so a completed run reported FAILED to SLURM --
leaving no way to tell a finished job from a broken one without reading logs.

The predicate under test is what draws that line, so these check both
directions.  Excusing too much would be worse than the original bug: a real
SIGTERM (pre-emption, ``scancel``, an OOM reaper) must still fail loudly.
"""

import signal
import unittest
from unittest import mock

from cosmos_rl.launcher import launch_all

SIGTERM_RC = 128 + signal.SIGTERM  # 143, as relayed by the shell wrapper


class TestCoordinatedControllerExit(unittest.TestCase):
    @staticmethod
    def _predicate(enabled: bool):
        return mock.patch.object(
            launch_all, "COSMOS_SHUTDOWN_ON_NO_POLICY_REPLICAS", enabled
        )

    # -- the case that motivated this -------------------------------------

    def test_controller_sigterm_is_success_when_the_reap_is_enabled(self):
        with self._predicate(True):
            self.assertTrue(
                launch_all._is_coordinated_controller_exit(0, 0, SIGTERM_RC)
            )

    def test_negative_signal_form_is_also_accepted(self):
        """subprocess reports -SIGTERM directly when no shell wrapper relays it."""
        with self._predicate(True):
            self.assertTrue(
                launch_all._is_coordinated_controller_exit(0, 0, -signal.SIGTERM)
            )

    # -- everything that must still fail loudly ---------------------------

    def test_a_real_crash_is_still_a_failure(self):
        with self._predicate(True):
            self.assertFalse(launch_all._is_coordinated_controller_exit(0, 0, 1))

    def test_sigkill_is_still_a_failure(self):
        """SIGKILL is never the coordinated path -- it is the OOM reaper etc."""
        with self._predicate(True):
            self.assertFalse(
                launch_all._is_coordinated_controller_exit(0, 0, 128 + signal.SIGKILL)
            )

    def test_a_replica_sigterm_is_still_a_failure(self):
        """Only the controller self-terminates; a SIGTERMed replica is a problem."""
        with self._predicate(True):
            self.assertFalse(
                launch_all._is_coordinated_controller_exit(1, 0, SIGTERM_RC)
            )

    def test_sigterm_is_a_failure_when_the_reap_is_disabled(self):
        """Without the feature there is nothing to self-terminate, so this is
        an external kill -- e.g. scheduler pre-emption -- and must fail."""
        with self._predicate(False):
            self.assertFalse(
                launch_all._is_coordinated_controller_exit(0, 0, SIGTERM_RC)
            )

    def test_no_controller_means_nothing_to_excuse(self):
        with self._predicate(True):
            self.assertFalse(
                launch_all._is_coordinated_controller_exit(0, -1, SIGTERM_RC)
            )


if __name__ == "__main__":
    unittest.main()
