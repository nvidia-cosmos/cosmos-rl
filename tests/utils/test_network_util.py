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

import socket
import unittest

from cosmos_rl.utils.network_util import (
    bind_available_port,
    find_available_port,
    is_port_free,
)


class TestPortProbing(unittest.TestCase):
    """Port probes must reject ports occupied on any local interface."""

    def test_find_available_port_skips_port_held_on_other_interface(self):
        # Occupy a port on 127.0.0.2 only. A loopback-only probe would not
        # see it, but the services the probe guards bind 0.0.0.0 and would
        # collide with it.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
            occupied.bind(("127.0.0.2", 0))
            occupied.listen()
            occupied_port = occupied.getsockname()[1]

            chosen = find_available_port(occupied_port)

            self.assertNotEqual(chosen, occupied_port)
            self.assertGreater(chosen, occupied_port)

    def test_find_available_port_returns_free_start_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("0.0.0.0", 0))
            free_port = probe.getsockname()[1]

        self.assertEqual(find_available_port(free_port), free_port)

    def test_is_port_free_sees_port_held_on_other_interface(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
            occupied.bind(("127.0.0.2", 0))
            occupied.listen()
            occupied_port = occupied.getsockname()[1]

            self.assertFalse(is_port_free(occupied_port))

        self.assertTrue(is_port_free(occupied_port))


class TestPortOwnership(unittest.TestCase):
    """bind_available_port must reserve the port, not merely report it."""

    def test_bind_available_port_reserves_the_port(self):
        # With probe-only selection both calls would return the same port;
        # ownership means the second caller is pushed to a different one.
        first = bind_available_port(20000)
        try:
            first_port = first.getsockname()[1]
            second = bind_available_port(first_port)
            try:
                self.assertNotEqual(second.getsockname()[1], first_port)
            finally:
                second.close()
        finally:
            first.close()

    def test_bind_available_port_socket_is_listening(self):
        sock = bind_available_port(20000)
        try:
            port = sock.getsockname()[1]
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                client.settimeout(5.0)
                client.connect(("127.0.0.1", port))
        finally:
            sock.close()

    def test_bind_available_port_exact_raises_when_port_taken(self):
        holder = bind_available_port(20000)
        try:
            held_port = holder.getsockname()[1]
            with self.assertRaises(RuntimeError):
                bind_available_port(held_port, held_port + 1)
        finally:
            holder.close()

    def test_bind_available_port_exact_reserves_free_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("0.0.0.0", 0))
            free_port = probe.getsockname()[1]

        sock = bind_available_port(free_port, free_port + 1)
        try:
            self.assertEqual(sock.getsockname()[1], free_port)
        finally:
            sock.close()


if __name__ == "__main__":
    unittest.main()
