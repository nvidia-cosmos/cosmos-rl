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

import asyncio
import unittest
from typing import Optional

from cosmos_rl.dispatcher.command import (
    Command,
    CommandScope,
    CommandType,
    WeightResumeCommand,
)
from cosmos_rl.utils.command_executor import CommandExecutor


# Mock commands for testing
class MockCommand(Command):
    """Mock command for testing purposes."""

    def __init__(self, value: int, **kwargs):
        kwargs["scope"] = CommandScope.LOCAL
        kwargs["command_type"] = CommandType.DUMMY
        super().__init__(**kwargs)
        self.value = value

    @classmethod
    def from_dict(cls, dict_v):
        return cls(**dict_v)


class TestCommandExecutor(unittest.TestCase):
    """Test suite for CommandExecutor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.executor = CommandExecutor()

    def test_register_sync_handler(self):
        """Test registering a synchronous command handler."""

        def test_handler(cmd: MockCommand) -> int:
            return cmd.value * 2

        self.executor.register_command_handler(MockCommand, test_handler)
        self.assertIn(MockCommand, self.executor.command_handler_registry)
        self.assertEqual(
            self.executor.command_handler_registry[MockCommand], test_handler
        )

    def test_register_async_handler(self):
        """Test registering an asynchronous command handler."""

        async def test_handler(cmd: MockCommand) -> int:
            await asyncio.sleep(0.01)
            return cmd.value * 3

        self.executor.register_command_handler(MockCommand, test_handler)
        self.assertIn(MockCommand, self.executor.command_handler_registry)
        self.assertEqual(
            self.executor.command_handler_registry[MockCommand], test_handler
        )

    def test_execute_sync_command(self):
        """Test executing a synchronous command."""

        def test_handler(cmd: MockCommand) -> int:
            return cmd.value * 2

        self.executor.register_command_handler(MockCommand, test_handler)
        test_cmd = MockCommand(value=5)
        result = self.executor.execute_command(test_cmd)
        self.assertEqual(result, 10)

    def test_execute_command_with_no_handler(self):
        """Test executing a command with no registered handler raises ValueError."""
        test_cmd = MockCommand(value=5)
        with self.assertRaises(ValueError) as context:
            self.executor.execute_command(test_cmd)

        self.assertIn("No handler found for command type", str(context.exception))

    def test_execute_multiple_command_types(self):
        """Test executing different command types with different handlers."""

        def test_handler(cmd: MockCommand) -> int:
            return cmd.value * 2

        def weight_resume_handler(cmd: WeightResumeCommand) -> str:
            return f"Resuming weights for {cmd.replica_name}"

        self.executor.register_command_handler(MockCommand, test_handler)
        self.executor.register_command_handler(
            WeightResumeCommand, weight_resume_handler
        )

        # Test MockCommand
        test_cmd = MockCommand(value=5)
        result1 = self.executor.execute_command(test_cmd)
        self.assertEqual(result1, 10)

        # Test WeightResumeCommand
        weight_cmd = WeightResumeCommand(replica_name="test_replica")
        result2 = self.executor.execute_command(weight_cmd)
        self.assertEqual(result2, "Resuming weights for test_replica")

    def test_async_execute_command(self):
        """Test executing an async command."""

        async def test_handler(cmd: MockCommand) -> int:
            await asyncio.sleep(0.01)
            return cmd.value * 3

        self.executor.register_command_handler(MockCommand, test_handler)
        test_cmd = MockCommand(value=5)

        # Run async execution
        result = asyncio.run(self.executor.async_execute_command(test_cmd))
        self.assertEqual(result, 15)

    def test_async_execute_command_with_no_handler(self):
        """Test async executing a command with no handler raises ValueError."""
        test_cmd = MockCommand(value=5)

        with self.assertRaises(ValueError) as context:
            asyncio.run(self.executor.async_execute_command(test_cmd))

        self.assertIn("No handler found for command type", str(context.exception))

    def test_async_execute_with_mixed_handlers(self):
        """Test that async_execute_command requires async handlers."""

        async def async_handler(cmd: MockCommand) -> int:
            await asyncio.sleep(0.001)
            return cmd.value * 3

        # Register async handler and test
        self.executor.register_command_handler(MockCommand, async_handler)
        test_cmd = MockCommand(value=5)
        result = asyncio.run(self.executor.async_execute_command(test_cmd))
        self.assertEqual(result, 15)

    def test_handler_override(self):
        """Test that registering a new handler for the same command type overrides the old one."""

        def handler1(cmd: MockCommand) -> int:
            return cmd.value * 2

        def handler2(cmd: MockCommand) -> int:
            return cmd.value * 3

        # Register first handler
        self.executor.register_command_handler(MockCommand, handler1)
        result1 = self.executor.execute_command(MockCommand(value=5))
        self.assertEqual(result1, 10)

        # Override with second handler
        self.executor.register_command_handler(MockCommand, handler2)
        result2 = self.executor.execute_command(MockCommand(value=5))
        self.assertEqual(result2, 15)

    def test_handler_returns_none(self):
        """Test handler that returns None."""

        def none_handler(cmd: MockCommand) -> Optional[int]:
            return None

        self.executor.register_command_handler(MockCommand, none_handler)
        test_cmd = MockCommand(value=5)
        result = self.executor.execute_command(test_cmd)
        self.assertIsNone(result)

    def test_async_concurrent_execution(self):
        """Test executing multiple async commands concurrently."""
        results = []

        async def async_handler(cmd: MockCommand) -> int:
            await asyncio.sleep(0.01)
            result = cmd.value * 2
            results.append(result)
            return result

        self.executor.register_command_handler(MockCommand, async_handler)

        async def run_concurrent():
            tasks = [
                self.executor.async_execute_command(MockCommand(value=i))
                for i in range(1, 6)
            ]
            return await asyncio.gather(*tasks)

        gathered_results = asyncio.run(run_concurrent())
        self.assertEqual(sorted(gathered_results), [2, 4, 6, 8, 10])
        self.assertEqual(sorted(results), [2, 4, 6, 8, 10])


if __name__ == "__main__":
    unittest.main()
