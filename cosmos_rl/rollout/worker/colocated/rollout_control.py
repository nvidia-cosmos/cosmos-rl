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
from cosmos_rl.rollout.validation import validation_round_for_command
from cosmos_rl.utils import constant
from cosmos_rl.rollout.prompt_batch import required_weight_version
from cosmos_rl.colocated.utils import CommandDispatcher
from typing import Type
from cosmos_rl.utils import distributed as dist_util


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

    def _report_discarded_samples(
        self, count: int, weight_version=None, *, training_rejections=None
    ) -> None:
        """Skip remote capacity accounting because colocated scheduling observes its local queue."""

    def set_command_dispatcher(self, dispatcher: CommandDispatcher):
        """
        Set the command dispatcher for the policy worker.
        Args:
            dispatcher (CommandDispatcher): The command dispatcher.
        """
        self.redis_for_remote = self.redis_controller
        self.redis_controller = dispatcher

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
        if command.dst_replica_name != self.replica_name:
            return
        self.rollout.set_underlying_model(self.api_client.get_policy_model())
        if command.weight_step is not None:
            assert command.weight_step >= self.current_weight_version
            self.current_weight_version = command.weight_step
        logger.info(
            f"[Rollout] Reset model from Policy replica {command.src_replica_name} to Rollout replica {self.replica_name}."
        )
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
            assert current_step >= self.current_weight_version, (
                f"current_step: {current_step} must be greater than or equal to self.current_weight_version: {self.current_weight_version}"
            )
            self.current_weight_version = current_step

        if current_step is not None and current_step >= 0:
            is_initial_validation = (
                current_step == 0 and self.config.validation.val_before_train
            )
            is_periodic_validation = (
                current_step > 0 and current_step % self.config.validation.freq == 0
            )
            is_final_validation = current_step == broadcast_command.total_steps

            should_do_validation = self.config.validation.enable and (
                is_initial_validation
                or is_periodic_validation
                or is_final_validation
                or bool(getattr(broadcast_command, "validation_round_id", None))
            )
            if not getattr(self, "_initial_weight_broadcast_done", False):
                self._initial_weight_broadcast_done = True
                # On resume the initial step can also be a periodic boundary.
                # Controller-admitted rounds include final validation without updates.
                should_do_validation = should_do_validation and (
                    self.config.validation.val_before_train
                    or bool(getattr(broadcast_command, "validation_round_id", None))
                )

            if should_do_validation:
                self.current_step = current_step
                # Setting the flag, do validation in the main loop.
                self.validation_round_id = validation_round_for_command(
                    broadcast_command
                )
                if self.validation_round_id is not None:
                    self.validation_flag.set()

        # Do validation if the flag is set.
        if self.validation_flag.is_set():
            self.do_validation()

        if broadcast_command.replica_should_stop():
            self.shutdown_signal.set()

    @torch.no_grad()
    def rollout_for_one_minor_step(self):
        """
        Rollout for one minor step given a batch of prompts.
        Returns:
            no_more_prompts (bool): Whether there is no more prompts to process.
            num_valid_results (int): Number of valid results generated in this step.
        """

        no_more_prompts = self.request_new_prompts(
            self.batch_size,
            self._prompt_queue,
            rank_in_mesh=getattr(self, "rank_in_rollout_repicas", None),
        )
        # A final ragged scatter can leave a DP rank with no prompts. Never
        # invoke the backend on only part of its collective cohort (or enqueue
        # an empty list as a real generation batch).
        if not self._prompt_queue.empty() and not self._prompt_queue.queue[0]:
            slot = self._prompt_queue.queue[0]
            if (
                getattr(slot, "required_weight_version", self.current_weight_version)
                <= self.current_weight_version
            ):
                self._prompt_queue.get_nowait()
        has_prompts = not self._prompt_queue.empty() and bool(
            self._prompt_queue.queue[0]
        )
        valid_prompt = False
        if has_prompts:
            batch_version = required_weight_version(self._prompt_queue.queue[0])
            valid_prompt = batch_version <= self.current_weight_version
        states = dist_util.all_gather_object_cpu((valid_prompt, no_more_prompts))
        no_more_prompts = any(end for _, end in states)
        if not all(valid for valid, _ in states):
            return no_more_prompts, 0
        _, valid_results = self.one_step_generation()
        return no_more_prompts, len(valid_results)

    def subscribe_remote_commands(self):
        """
        Subscribe and get commands from remote controller.
        """
        commands = []
        try:
            commands = self.redis_for_remote.subscribe_command(self.replica_name)
        except Exception as e:
            logger.debug(
                f"[Rollout] Failed to get commands : {e} at replica {self.replica_name}, wait for next round"
            )
        commands = [Command.depack(x) for x in commands]
        return commands

    def consume_command(
        self,
        command_type: Type[Command] = Command,
        cmd_pred=None,
        timeout=constant.COSMOS_ROLLOUT_CMD_WAIT_TIMEOUT,
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
            self._command_queue.put(command)
        executed_cmd = super().consume_one_command(cmd_pred)
        logger.debug(f"[Rollout] Executing command: {executed_cmd}")
        assert executed_cmd is None or isinstance(executed_cmd, command_type), (
            f"Invalid command type: {type(executed_cmd)}"
        )
        return executed_cmd

    def prepare_shard_infos_for_weight_sync_insts(self):
        # Diffusers models do not need weight sync in colocated mode
        if not self.config.policy.is_diffusers:
            return super().prepare_shard_infos_for_weight_sync_insts()


# Register command handlers
ColocatedRolloutControlWorker.register_rollout_command_handler(
    PolicyToRolloutUnicastCommand
)(ColocatedRolloutControlWorker.policy_to_rollout_unicast)
ColocatedRolloutControlWorker.register_rollout_command_handler(
    RolloutToRolloutBroadcastCommand
)(ColocatedRolloutControlWorker.broadcast_to_all_rollout_replica)
