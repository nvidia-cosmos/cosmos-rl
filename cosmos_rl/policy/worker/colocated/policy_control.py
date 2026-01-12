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

from cosmos_rl.colocated.utils import CommandDispatcher
from cosmos_rl.policy.worker.rl_worker import RLPolicyWorker
from cosmos_rl.utils.logging import logger
import copy
from cosmos_rl.dispatcher.command import (
    Command,
    BuildMeshCommand,
    PolicyToRolloutUnicastCommand,
)
from typing import Type


class ColocatedPolicyControlWorker(RLPolicyWorker):
    """
    Colocated Policy Worker class.
    Inherits from RLPolicyWorker.
    Control the policy training control flow in colocated mode.
    """

    colocated = True
    policy_command_handler_registry = copy.deepcopy(
        RLPolicyWorker.policy_command_handler_registry
    )

    def set_command_dispatcher(self, dispatcher: CommandDispatcher):
        """
        Set the command dispatcher for the policy worker.
        Record the remote dispatcher for communication with all replicas.
        Args:
            dispatcher (CommandDispatcher): The command dispatcher.
        """
        self.redis_for_remote = self.redis_controller
        self.redis_controller = dispatcher

    def subscribe_remote_commands(self):
        """
        Subscribe and get commands from remote controller.
        """
        commands = []
        try:
            commands = self.redis_for_remote.subscribe_command(self.replica_name)
        except Exception as e:
            logger.debug(
                f"[Policy] Failed to get commands : {e} at replica {self.replica_name}, wait for next round"
            )
        commands = [Command.depack(x) for x in commands]
        return commands

    def execute_policy_to_rollout_unicast(self, command: PolicyToRolloutUnicastCommand):
        """
        No need real communication in colocated mode since they share the same model instance.
        """
        assert command.src_replica_size == self.world_size
        if not command.src_replica_name == self.replica_name:
            logger.error(
                f"[Policy] {self.replica_name} received P2R command from {command.src_replica_name}, but it is not the source replica."
            )
            return False
        return False

    def consume_command(
        self, command_type: Type[Command] = Command, no_exec=False
    ) -> bool:
        """
        Consume one command from the command dispatcher.
        """
        if self.global_rank == 0:
            commands = []
            try:
                commands = self.redis_controller.subscribe_command(self.replica_name)
            except Exception as e:
                logger.debug(
                    f"[Policy] Failed to get commands : {e} at replica {self.replica_name}, wait for next round"
                )
            for x in commands:
                command = Command.depack(x)
                if isinstance(command, BuildMeshCommand):
                    """ directly push the buildmesh command to the nccl comm, will not block main thread """
                    # broadcast the buildmesh command to all ranks
                    cmd = self.kv_store.broadcast_command(command, src=0)
                    self.is_master_replica = (
                        cmd.replica_name_to_rank[self.replica_name] == 0
                    )
                    self.inter_policy_nccl.push_cmd(cmd)
                    continue
                self.fetch_command_buffer.put_nowait(command)
        else:
            if issubclass(command_type, BuildMeshCommand):
                try:
                    bmcmd = self.kv_store.broadcast_command(None, src=0)
                    if bmcmd:
                        assert isinstance(
                            bmcmd, BuildMeshCommand
                        ), "Only buildmesh command is supported"
                        self.is_master_replica = (
                            bmcmd.replica_name_to_rank[self.replica_name] == 0
                        )
                        self.inter_policy_nccl.push_cmd(bmcmd)
                except Exception as e:
                    raise RuntimeError(f"Failed to broadcast on slave workers: {e}")
        self.broadcast_command()
        if self.command_buffer.empty():
            return False
        cmd = self.command_buffer.get_nowait()
        assert isinstance(cmd, command_type), f"Invalid command type: {type(cmd)}"
        if no_exec:
            return False
        logger.debug(f"[Policy] Executing command: {cmd}")
        abort = self.execute_command(cmd)
        return abort


# Register command handlers
ColocatedPolicyControlWorker.register_policy_command_handler(
    PolicyToRolloutUnicastCommand
)(ColocatedPolicyControlWorker.execute_policy_to_rollout_unicast)
