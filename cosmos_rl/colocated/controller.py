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

from typing import List, Optional, Callable, Any, Type, Union
from cosmos_rl.dispatcher.controller import Controller
from cosmos_rl.dispatcher.replica import Replica
from cosmos_rl.dispatcher.protocol import Role, RolloutRequest
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.wandb_logger import (
    is_wandb_available,
    init_wandb,
)
from cosmos_rl.utils.util import is_master_rank
import cosmos_rl.utils.network_util as network_util
import torch
from torch.utils.data import Dataset
from cosmos_rl.policy.config import Config
from cosmos_rl.utils.parallelism_map import ParallelizedShardMapper
from cosmos_rl.dispatcher.data.data_fetcher import ControllerDataFetcher
from cosmos_rl.policy.worker.colocated.policy_control import (
    ColocatedPolicyControlWorker,
)
from cosmos_rl.rollout.worker.colocated.rollout_control import (
    ColocatedRolloutControlWorker,
)
from cosmos_rl.colocated.utils import CommandDispatcher
from cosmos_rl.dispatcher.command import (
    WeightResumeCommand,
    PolicyToRolloutUnicastCommand,
    DataFetchCommand,
    RolloutToRolloutBroadcastCommand,
    BuildMeshCommand,
    Command,
    PolicyToPolicyUnicastCommand,
    PolicyToPolicyBroadcastCommand,
)
import numpy as np
from cosmos_rl.utils.payload import extract_rollouts
from cosmos_rl.utils.util import RollingDict
from cosmos_rl.policy.model.hf_models import HFModel
import cosmos_rl.utils.distributed as dist_util


class DummyReplica(Replica):
    """
    Dummy replica for colocated controller to manage policy and rollout workers.
    """

    def __init__(self, name: str, role: Role):
        """
        Initialize the DummyReplica.
        Args:
            name (str): The name of the replica.
            role (Role): The role of the replica (POLICY or ROLLOUT).
        """
        super().__init__(name=name, role=role, atoms=[])
        # In colocated mode, mesh_rank is always 0
        self.status.mesh_rank = 0

    @property
    def all_atoms_arrived(self) -> bool:
        return True


class ColocatedController(Controller):
    """
    Colocated controller for policy and rollout workers in the same process.
    Handles coordinations with policy and rollout including recording the step updates, issuing commands, and collecting results.
    Act as the controller in colocated mode.
    """

    def setup(
        self,
        policy: ColocatedPolicyControlWorker,
        rollout: ColocatedRolloutControlWorker,
        command_dispatcher: CommandDispatcher,
        config: Config,
        dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        custom_logger_fns: Optional[List[Callable]] = None,
        sampler: Optional[Callable] = None,
        batch_sampler: Optional[Callable] = None,
        val_sampler: Optional[Callable] = None,
        val_batch_sampler: Optional[Callable] = None,
    ):
        """
        Setup the colocated controller with policy and rollout workers.
        Args:
            policy (ColocatedPolicyControlWorker): The policy trainer instance.
            rollout (ColocatedRolloutControlWorker): The rollout worker instance.
            command_dispatcher (CommandDispatcher): The command dispatcher for issuing commands.
            config (Config): The configuration for the controller.
            dataset (Optional[Dataset]): The training dataset.
            val_dataset (Optional[Dataset]): The validation dataset.
            custom_logger_fns (Optional[List[Callable]]): Custom logger functions.
            sampler (Optional[Callable]): The training data sampler.
            batch_sampler (Optional[Callable]): The training data batch sampler.
            val_sampler (Optional[Callable]): The validation data sampler.
            val_batch_sampler (Optional[Callable]): The validation data batch sampler.
        """
        self.remote_command_manager = CommandDispatcher(
            [policy.replica_name, rollout.replica_name],
        )
        self._init_status()
        if self.config is not None:
            raise Exception(
                "[Controller] Config has been set. Please do not call setup again."
            )

        self.config = config
        self.policy = policy
        self.rollout = rollout
        task_type = config.train.train_policy.type
        self.policy_to_rollout_shard_mapper = ParallelizedShardMapper.get_instance(
            config
        )

        if "wandb" in config.logging.logger and is_wandb_available():
            init_wandb(config)
        else:
            logger.warning(
                "Wandb is not available. Please install it to use wandb logging features."
            )

        self.is_rl = task_type != "sft"
        self.weight_version_to_prompt_num = {}  # Only for on-policy.

        self.data_fetcher = ControllerDataFetcher(
            config=config,
            dataset=dataset,
            val_dataset=val_dataset,
            sampler=sampler,
            batch_sampler=batch_sampler,
            val_sampler=val_sampler,
            val_batch_sampler=val_batch_sampler,
            is_rl=self.is_rl,
        )

        ips = network_util.get_eth_ips()
        if len(ips) > 0:
            self.config.eth_ips = ";".join(ips)

        self.policy_replica = DummyReplica(
            name=self.policy.replica_name,
            role=Role.POLICY,
        )
        self.rollout_replica = DummyReplica(
            name=self.rollout.replica_name,
            role=Role.ROLLOUT,
        )
        self.command_dispatcher = command_dispatcher
        self.current_step = 0
        self.remain_samples_num = (
            len(self.data_fetcher.dataset.train_set)
            * self.config.rollout.n_generation
            * self.config.train.epoch
        )
        self.total_steps = (
            self.remain_samples_num // self.config.train.train_batch_per_replica
        )
        self.train_report_data = RollingDict(maxlen=20)
        self.filter_records = {}
        self.custom_logger_fns = (
            custom_logger_fns if custom_logger_fns is not None else []
        )

    def wait_for_remote_command(
        self,
        replica: Union[ColocatedPolicyControlWorker, ColocatedRolloutControlWorker],
        command_type: List[Type] = [],
        ignore_type: List[Type] = [],
        remove: bool = True,
    ) -> Command:
        """
        Wait for a specific command from the policy worker.
        Args:
            command_type (Type): The type of command to wait for.
            replica (Union[ColocatedPolicyControlWorker, ColocatedRolloutControlWorker]): The replica to wait for command from.
        Returns:
            Tuple[Command, bool]: The received command and a boolean indicating if it matches the expected type.
        """
        while True:
            command = self.remote_command_manager.front_command(replica.replica_name)
            if remove and command is not None:
                self.remote_command_manager.pop_command(replica.replica_name)
            if command is not None and (
                len(command_type) == 0
                or any([isinstance(command, ct) for ct in command_type])
            ):
                return command
            if any([isinstance(command, it) for it in ignore_type]):
                continue
            if command is not None:
                logger.error(
                    f"Expected command of type {command_type}, but got {type(command)} for {replica.role} replica {replica.replica_name}"
                )
                raise ValueError
            if self.policy.global_rank == 0:
                commands = replica.subscribe_remote_commands()
            else:
                commands = []
            commands = dist_util.broadcast_object_cpu(
                commands, src=0, device=torch.device("cpu")
            )
            for command in commands:
                self.remote_command_manager.publish_command(
                    command.pack(), replica.replica_name
                )

    def init_commands(self):
        """
        Initialize the system by building meshes and triggering weight resume and policy-rollout commands.
        This is called once at the beginning to prepare the initial state including model weights.
        """
        data_fetch_cmd = self.policy_consume_one_step_commands_util_data_fetch()
        self.rollout_consume_one_step_commands_util_r2r()
        assert isinstance(data_fetch_cmd, DataFetchCommand)
        self.init_data_fetch_command = data_fetch_cmd

    def policy_consume_one_step_commands_util_data_fetch(self) -> DataFetchCommand:
        """
        Consume one step commands for policy replica until DataFetchCommand is received.
        Execute necessary commands like BuildMeshCommand, WeightResumeCommand, and PolicyToPolicy commands.
        Skips other commands until DataFetchCommand is encountered.
        DataFetchCommand is returned for further processing and not executed here.
        Returns:
            DataFetchCommand: The received DataFetchCommand.
        """
        # Commands that should be executed from remote dispatcher
        should_excuted = [
            BuildMeshCommand,
            WeightResumeCommand,
            PolicyToPolicyUnicastCommand,
            PolicyToPolicyBroadcastCommand,
        ]

        # DataFetchCommand is the stopping point for each step so break when we see it
        should_stop = [DataFetchCommand]

        # PolicyToRolloutUnicastCommand is self triggered separately in colocated mode so ignore it.
        others = [PolicyToRolloutUnicastCommand]
        executed = False
        while True:
            cmd = self.wait_for_remote_command(self.policy)
            if any(isinstance(cmd, t) for t in should_excuted):
                self.command_dispatcher.publish_command(
                    cmd.pack(), self.policy.replica_name
                )
                self.policy.consume_command(type(cmd))
                executed = True
            if any(isinstance(cmd, t) for t in should_stop):
                return cmd
            if not executed:
                assert any(
                    isinstance(cmd, t) for t in others
                ), f"Unexpected command {cmd}"

    def rollout_consume_one_step_commands_util_r2r(self) -> bool:
        """
        Consume one step commands for rollout replica until RolloutToRolloutBroadcastCommand is received.
        Skips all commands and return until RolloutToRolloutBroadcastCommand is encountered.
        Determines if weight synchronization is needed based on the current step and configuration.
        If weight synchronization is needed, triggers PolicyToRolloutUnicastCommand and RolloutToRolloutBroadcastCommand.
        Returns:
            bool: Whether weight synchronization is needed.
        """
        step = self.current_step
        # All replicas have been reduced, trigger allreduce
        need_sync_weight = step % self.config.train.sync_weight_interval == 0
        # If the current step is the last step, we need to sync weight always to act as ending signal
        need_sync_weight = need_sync_weight or step == self.total_steps
        # If validation is enabled, we need to sync weight every validation step
        if self.config.validation.enable:
            need_sync_weight = need_sync_weight or (
                step % self.config.validation.freq == 0
            )
        # P->R & R->R
        if need_sync_weight:
            # Drain all commands
            # R2R command is the stopping point for each step so break when we see it
            while True:
                cmd = self.wait_for_remote_command(self.rollout)
                if isinstance(cmd, RolloutToRolloutBroadcastCommand):
                    break
            # Further trigger P2R & R2R commands locally
            PolicyToRolloutUnicastCommand.trigger(
                src_replica=self.policy_replica,
                dst_replica=self.rollout_replica,
                src_replica_size=self.policy.world_size,
                dst_replica_size=self.rollout.world_size,
                weight_step=self.current_step,
                total_steps=self.total_steps,
                redis_handler=self.command_dispatcher,
            )
            RolloutToRolloutBroadcastCommand.trigger(
                src_replica=self.rollout_replica,
                dst_replicas=[self.rollout_replica],
                weight_step=self.current_step,
                total_steps=self.total_steps,
                redis_handler=self.command_dispatcher,
            )
        return need_sync_weight

    def advance_iteration(self):
        """
        Advance the training iteration for new iteration of rollout and policy.
        """
        self.train_report_data[self.current_step] = {}
        self.current_step += 1

    def rollout_completed_for_data_fetch_n_training(self, required_rollouts: int):
        """
        Notify the controller that rollouts have been completed and trigger data fetch command.
        DataFetchCommand is triggered to fetch new data for the policy and start one training iteration.
        Args:
            required_rollouts (int): Number of rollouts that have been completed.
        """
        do_save = False

        if self.current_step == self.total_steps:
            # Always save checkpoint at the last step
            do_save = True
        elif self.config.train.ckpt.save_freq_in_epoch > 0:
            # Checkpointing based on epoch if `save_freq_in_epoch` is set
            if (
                self.remain_samples_num + required_rollouts - 1
            ) // self.samples_per_epoch != (
                self.remain_samples_num - 1
            ) // self.samples_per_epoch:
                # New epoch begins and old epoch ends
                # So check the epoch number against save_freq_in_epoch for saving checkpoint
                epoch = (
                    self.config.train.epoch
                    - (self.remain_samples_num + required_rollouts - 1)
                    // self.samples_per_epoch
                )
                do_save = epoch % self.config.train.ckpt.save_freq_in_epoch == 0
                if do_save:
                    logger.info(
                        f"[Controller] Epoch {epoch} ends, triggering checkpoint saving at step {self.current_step}"
                    )
        else:
            # Checkpointing based on step if `save_freq_in_epoch` is not set
            do_save = (
                self.current_step % self.config.train.ckpt.save_freq == 0
                and self.current_step > 0
            )

        if hasattr(self, "init_data_fetch_command"):
            data_fetch_cmd = self.init_data_fetch_command
            delattr(self, "init_data_fetch_command")
        else:
            data_fetch_cmd = self.policy_consume_one_step_commands_util_data_fetch()
            logger.debug(
                f"[Controller] DataFetchCommand details: {data_fetch_cmd} at step {self.current_step}"
            )
            if not self.config.train.resume:
                assert (
                    data_fetch_cmd.do_save
                    == (do_save and self.config.train.ckpt.enable_checkpoint)
                ), f"Expected do_save {(do_save and self.config.train.ckpt.enable_checkpoint)} but got {data_fetch_cmd.do_save}"
                assert (
                    self.current_step == data_fetch_cmd.global_step
                ), f"Expected global_step {self.current_step} but got {data_fetch_cmd.global_step}"

        self.current_step = data_fetch_cmd.global_step
        self.total_steps = data_fetch_cmd.total_steps
        self.remain_samples_num = data_fetch_cmd.remain_samples_num
        self.command_dispatcher.publish_command(
            data_fetch_cmd.pack(), self.policy.replica_name
        )

        if self.config.logging.logger and len(self.policy.data_queue.queue) > 0:
            assert (
                self.policy.global_rank == 0
            ), "Only global rank 0 collects and reports training statistics."
            rewards = []
            completion_lengths = []
            advantages = []
            filter_rewards = []
            for rollout in self.policy.data_queue.queue:
                rewards.append(rollout.reward)
                completion_length = (
                    len(rollout.completion_token_ids)
                    if self.config.train.train_policy.rollout_as_token_ids
                    else len(self.policy.trainer.tokenizer.encode(rollout.completion))
                )
                advantages.extend([rollout.advantage] * completion_length)
                filter_rewards.append(rollout.filter_reward)
                completion_lengths.append(completion_length)
            report_data = {
                "train/reward_mean": np.mean(rewards).item(),
                "train/reward_std": np.std(rewards).item(),
                "train/reward_max": np.max(rewards).item(),
                "train/reward_min": np.min(rewards).item(),
                "rollout/completion_length_mean": np.mean(completion_lengths).item(),
                "rollout/completion_length_std": np.std(completion_lengths).item(),
                "rollout/completion_length_max": np.max(completion_lengths).item(),
                "rollout/completion_length_min": np.min(completion_lengths).item(),
                "rollout/advantage_mean": np.mean(advantages).item(),
                "rollout/advantage_std": np.std(advantages).item(),
                "rollout/advantage_max": np.max(advantages).item(),
                "rollout/advantage_min": np.min(advantages).item(),
                "rollout/filter_reward_mean": np.mean(filter_rewards).item(),
                "rollout/filter_reward_std": np.std(filter_rewards).item(),
                "rollout/filter_reward_max": np.max(filter_rewards).item(),
                "rollout/filter_reward_min": np.min(filter_rewards).item(),
            }
            self.train_report_data.setdefault(self.current_step, {}).update(report_data)

    def put_rollouts(self, rollout: RolloutRequest):
        """
        Put rollouts into the policy's data queue.
        This method extracts rollouts from the rollout request and enqueues them for training.
        """
        for k, v in rollout.metrics.items():
            # Handle dynamic sampling statistics update in colocated mode
            self.train_report_data.setdefault(self.current_step, {})[k] = (
                self.train_report_data.get(self.current_step, {}).get(k, 0) + v
            )
        rollouts_list = extract_rollouts(rollout.payloads, rollout.is_end)
        # Flatten the rollouts into a single list
        rollouts = [
            rollout
            for rollouts_group in rollouts_list
            for rollout in rollouts_group  # rollouts_group: all rollouts of the same prompt.
        ]

        gathered_rollouts = dist_util.all_gather_object_cpu(
            rollouts,
            device=torch.device("cpu"),
            group=self.rollout.parallel_dims.mesh["dp"].get_group(),
        )
        rollouts = [rollout for sublist in gathered_rollouts for rollout in sublist]
        if len(rollouts) > 0:
            logger.debug(
                f"[RolloutGroup] from replica: {rollout.src_replica_name} with {len(rollout.payloads)} samples:"
                f"example: rollouts[0]\n{rollouts[0]}"
            )
        assert (
            self.rollout.parallel_dims.cp_coord[1] == 1
        ), "Colocated rollout worker only supports cp size 1."
        if self.rollout.parallel_dims.mesh["dp"].get_local_rank() == 0:
            for rollout in rollouts:
                self.policy.data_queue.put_nowait(rollout)

    def pending_policy_samples(self) -> int:
        """
        Get the number of pending samples in the policy's data queue.
        """
        return self.policy.data_queue.qsize()

    def pending_policy_samples_all_replicas(self) -> int:
        """
        Get the total number of pending samples in the policy's data queue across all replicas.
        """
        local_pending = self.pending_policy_samples()
        if self.policy.global_rank != 0:
            assert (
                self.pending_policy_samples() == 0
            ), "Only global rank 0 should have pending samples."
        return dist_util.broadcast_object_cpu(
            local_pending, src=0, device=torch.device("cpu")
        )

    def get_policy_model(self) -> Any:
        """
        Get the current policy model.
        Returns:
            torch.nn.Module: The current policy model instance.
        """
        if isinstance(self.policy.trainer.model, HFModel):
            return self.policy.trainer.model.model
        return self.policy.trainer.model

    def training_end_ack(self):
        """
        Send the training end acknowledgment to the controller.
        Args:
        """
        if is_master_rank(self.policy.parallel_dims, self.policy.global_rank):
            self.policy.api_client.post_policy_train_ack(
                self.policy.replica_name,
                self.current_step - 1,
                self.current_step - 1,
                False,
                {},
            )
