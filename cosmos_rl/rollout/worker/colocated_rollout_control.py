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

from cosmos_rl.rollout.worker.rollout_control import (
    DisaggregatedRolloutControlWorker,
)
import torch
import copy
from cosmos_rl.utils.logging import logger
from cosmos_rl.dispatcher.command import (
    Command,
    PolicyToRolloutUnicastCommand,
    RolloutToRolloutBroadcastCommand,
)
from cosmos_rl.utils import constant
from cosmos_rl.dispatcher.data.schema import RLPayload


class ColocatedRolloutControlWorker(DisaggregatedRolloutControlWorker):
    """
    Colocated Rollout Worker class.
    Inherits from DisaggregatedRolloutControlWorker.
    Control the rollout generation control flow in colocated mode.
    """

    colocated = True
    rollout_command_handler_registry = copy.deepcopy(
        DisaggregatedRolloutControlWorker.rollout_command_handler_registry
    )

    def init_redis(self):
        """
        No op for colocated rollout worker.
        """
        pass

    @torch.no_grad()
    def policy_to_rollout_unicast(self, command: PolicyToRolloutUnicastCommand):
        """
        Sync the weight from policy to rollout.
        This is Policy -> Rollout replica. Will only happen between
        a pair of policy and rollout replica.
        For colocated mode, just directly set the model from policy to rollout.
        """
        # lazy initialization of the rollout engine.
        is_for_weight_resume = command.dst_replica_name == self.replica_name
        load_format = "auto" if is_for_weight_resume else "dummy"
        self.lazy_initialize_rollout_engine(load_format)
        if command.dst_replica_name == self.replica_name:
            return
        self.rollout.set_underlying_model(self.api_client.get_policy_model())
        self.state.set_weight_synced()

    def broadcast_to_all_rollout_replica(
        self, broadcast_command: RolloutToRolloutBroadcastCommand
    ) -> None:
        """
        Broadcast the weight to all other rollout replicas.
        Will only happen between Rollout Replica 0 and all other Rollout Replicas.
        For colocated mode, only used for signaling purpose for handling weight step updates.
        """
        src_replica_name: str = broadcast_command.src_replica_name
        # dst_replica_names: List[str] = broadcast_command.dst_replica_names
        # lazy initialization of the rollout engine.
        assert self.replica_name == src_replica_name
        current_step = broadcast_command.weight_step
        if current_step is not None:
            assert (
                current_step >= self.current_weight_version
            ), f"current_step: {current_step} must be greater than or equal to self.current_weight_version: {self.current_weight_version}"
            self.current_weight_version = current_step

        if current_step is not None and current_step > 0:
            should_do_validation = self.config.validation.enable and (
                current_step % self.config.validation.freq == 0
                or current_step == broadcast_command.total_steps
            )

            if should_do_validation:
                self.current_step = current_step
                # Setting the flag, do validation in the main loop.
                self.validation_flag.set()

        if broadcast_command.replica_should_stop():
            # Do validation if the flag is set before stopping.
            if self.validation_flag.is_set():
                self.do_validation()
            self.shutdown_signal.set()
            self.shutdown_mp_signal.set()

    @torch.no_grad()
    def rollout_for_one_minor_step(self):
        """
        Rollout for one minor step given a batch of prompts.
        Returns:
            no_more_prompts (bool): Whether there is no more prompts to process.
            num_valid_results (int): Number of valid results generated in this step.
        """

        no_more_prompts = self.request_new_prompts(self.batch_size, self._prompt_queue)
        if self._prompt_queue.empty():
            return no_more_prompts, 0
        # Check if the prompt is valid for the current weight version
        first_payload: RLPayload = self._prompt_queue.queue[0][0]
        is_valid_prompt_for_current_weight_version = (
            first_payload.weight_version <= self.current_weight_version
        )
        if not is_valid_prompt_for_current_weight_version:
            # Fully Synchronized mode is enabled, we need to wait until the weight version is updated
            return no_more_prompts, 0

        _, valid_results = self.one_step_generation()
        return no_more_prompts, len(valid_results)

    def consume_command(
        self, cmd_pred=None, timeout=constant.COSMOS_ROLLOUT_CMD_WAIT_TIMEOUT
    ):
        """
        Consume one command from controller.
        """
        commands = []
        try:
            # blocking request
            commands = self.redis_controller.subscribe_command(self.replica_name)
        except Exception as e:
            logger.error(
                f"[Rollout] Failed in query commands from controller for replica {self.replica_name}\n: {str(e)}"
            )

        for instruction in commands:
            command = Command.depack(instruction)
            logger.info(f"[Rollout] Received command: {command}")
            self._command_queue.put(command)
        return super().consume_one_command(cmd_pred)


# Register command handlers
ColocatedRolloutControlWorker.register_rollout_command_handler(
    PolicyToRolloutUnicastCommand
)(ColocatedRolloutControlWorker.policy_to_rollout_unicast)
ColocatedRolloutControlWorker.register_rollout_command_handler(
    RolloutToRolloutBroadcastCommand
)(ColocatedRolloutControlWorker.broadcast_to_all_rollout_replica)
