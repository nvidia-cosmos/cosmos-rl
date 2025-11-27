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

from cosmos_rl.policy.trainer.grpo_trainer import GRPOTrainer
from cosmos_rl.utils.logging import logger
import copy
from cosmos_rl.dispatcher.command import (
    Command,
    BuildMeshCommand,
    PolicyToRolloutUnicastCommand,
)


class ColocatedPolicyControlWorker(GRPOTrainer):
    """
    Colocated Policy Worker class.
    Inherits from GRPOTrainer.
    Control the policy training control flow in colocated mode.
    """

    colocated = True
    policy_command_handler_registry = copy.deepcopy(
        GRPOTrainer.policy_command_handler_registry
    )

    def init_redis(self):
        """
        No need to init redis in colocated mode.
        """
        pass

    def execute_policy_to_rollout_unicast(self, command: PolicyToRolloutUnicastCommand):
        """
        No need real communication in colocated mode since they share the same model instance.
        """
        assert command.src_replica_size == self.world_size
        if not command.src_replica_name == self.replica_name:
            logger.error(
                f"Policy {self.replica_name} received P2R command from {command.src_replica_name}, but it is not the source replica."
            )
            return False
        return False

    def build_global_mesh(self, command: BuildMeshCommand):
        """
        Build the global mesh for inter-policy communication.
        In colocated mode, just set the replica_name_to_rank directly and set the is_single_peer and is_comm_ready events.
        """
        assert len(command.replica_name_to_rank) == 1
        self.inter_policy_nccl.replica_name_to_rank = command.replica_name_to_rank
        assert self.replica_name in command.replica_name_to_rank
        self.inter_policy_nccl.is_single_peer.set()
        self.inter_policy_nccl.is_comm_ready.set()
        return

    def consume_command(self):
        """
        Consume one command from controller.
        """
        if self.global_rank == 0:
            commands = []
            try:
                commands = self.redis_controller.subscribe_command(self.replica_name)
            except Exception as e:
                logger.debug(
                    f"Failed to get commands : {e} at replica {self.replica_name}, wait for next round"
                )
            for x in commands:
                command = Command.depack(x)
                self.fetch_command_buffer.put_nowait(command)
        self.broadcast_command()
        if self.command_buffer.empty():
            return False
        cmd = self.command_buffer.get_nowait()
        logger.info(f"[Policy] Executing command: {cmd}")
        abort = self.execute_command(cmd)
        return abort


# Register command handlers
ColocatedPolicyControlWorker.register_policy_command_handler(
    PolicyToRolloutUnicastCommand
)(ColocatedPolicyControlWorker.execute_policy_to_rollout_unicast)
ColocatedPolicyControlWorker.register_policy_command_handler(BuildMeshCommand)(
    ColocatedPolicyControlWorker.build_global_mesh
)
