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

from typing import List
from queue import Queue, Empty


class CommandDispatcher:
    """
    A simple in-memory command dispatcher for colocated mode.
    It uses Python Queue to simulate the command publish-subscribe mechanism.
    """

    def __init__(self, command_queues: List[str], is_master):
        """
        Initialize the CommandDispatcher.
        Args:
            command_queues (List[str]): List of command queue names.
            is_master (bool): Flag indicating if this instance is the master.
        """
        if is_master:
            self.command_queues = {k: Queue() for k in command_queues}
        else:
            self.command_queues = {}

    def publish_command(self, data, stream_name: str):
        """
        Publish a command to the specified command queue.
        Args:
            data : The packed command to publish.
            stream_name (str): The name of the command queue.
        """

        if stream_name in self.command_queues:
            self.command_queues[stream_name].put(data)

    def subscribe_command(self, stream_name: str) -> List[dict]:
        """
        Subscribe to commands from the specified command queue.
        Args:
            stream_name (str): The name of the command queue.
        Returns:
            list: A list of commands in msgpack format.
        """

        if stream_name not in self.command_queues:
            return []
        commands = []
        if not self.command_queues[stream_name].empty():
            try:
                command = self.command_queues[stream_name].get_nowait()
                commands.append(command)
            except Empty:
                pass
        return commands
