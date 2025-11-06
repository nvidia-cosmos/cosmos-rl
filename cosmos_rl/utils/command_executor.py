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

from typing import Type, Callable, Any
from cosmos_rl.dispatcher.command import Command
from cosmos_rl.utils.async_utils import is_async_callable
from cosmos_rl.utils.logging import logger


class CommandExecutor:
    """
    A class to manage command handlers for policy / rollout replica.
    user should register the command handlers before using the executor.
    """

    def __init__(self):
        self.command_handler_registry = {}

    def register_command_handler(self, command_type: Type[Command], handler: Callable):
        if command_type in self.command_handler_registry:
            logger.warning(
                f"Command handler for {command_type} already registered, will override the existing handler."
            )
        self.command_handler_registry[command_type] = handler

    def get_command_handler(self, command_type: Type[Command]) -> Callable:
        handler = self.command_handler_registry.get(command_type)
        if handler is None:
            raise ValueError(f"No handler found for command type: {command_type}")
        return handler

    def execute_command(self, command: Command) -> Any:
        handler = self.get_command_handler(type(command))
        return handler(command)

    async def async_execute_command(self, command: Command) -> Any:
        handler = self.get_command_handler(type(command))

        if is_async_callable(handler):
            return await handler(command)
        else:
            return handler(command)
