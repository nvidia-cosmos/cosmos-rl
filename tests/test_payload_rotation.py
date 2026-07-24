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

"""Tests for the shared health-aware rotation skip-list."""

import unittest

from cosmos_rl.utils.payload_transport.rotation import HealthSkipList


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class TestHealthSkipList(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock()
        self.sl = HealthSkipList(cooldown=10.0, clock=self.clock)

    def test_quarantine_and_expiry(self):
        self.sl.quarantine("a")
        self.assertTrue(self.sl.is_quarantined("a"))
        self.assertIn("a", self.sl)
        # Advance past the cooldown -> expires + lazily GC'd.
        self.clock.t = 10.0
        self.assertFalse(self.sl.is_quarantined("a"))
        self.assertEqual(len(self.sl), 0)

    def test_custom_cooldown(self):
        self.sl.quarantine("a", cooldown=2.0)
        self.clock.t = 1.9
        self.assertTrue(self.sl.is_quarantined("a"))
        self.clock.t = 2.0
        self.assertFalse(self.sl.is_quarantined("a"))

    def test_healthy_filters_quarantined(self):
        self.sl.quarantine(("ip", 7001))
        healthy = self.sl.healthy([7000, 7001, 7002], key_fn=lambda p: ("ip", p))
        self.assertEqual(healthy, [7000, 7002])

    def test_healthy_never_starves(self):
        for port in (7000, 7001):
            self.sl.quarantine(("ip", port))
        # All candidates quarantined -> return the full list unchanged.
        healthy = self.sl.healthy([7000, 7001], key_fn=lambda p: ("ip", p))
        self.assertEqual(healthy, [7000, 7001])

    def test_refailure_extends_deadline(self):
        self.sl.quarantine("a")  # expires at 10
        self.clock.t = 5.0
        self.sl.quarantine("a")  # re-fail -> expires at 15
        self.clock.t = 12.0
        self.assertTrue(self.sl.is_quarantined("a"))

    def test_clear(self):
        self.sl.quarantine("a")
        self.sl.clear("a")
        self.assertFalse(self.sl.is_quarantined("a"))

    def test_skip_until_backing_map_is_shared(self):
        # Direct-access compat: writing the backing map quarantines a key.
        self.sl.skip_until[("ip", 9)] = 100.0
        self.assertTrue(self.sl.is_quarantined(("ip", 9)))


if __name__ == "__main__":
    unittest.main()
