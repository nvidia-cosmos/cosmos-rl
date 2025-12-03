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

from cosmos_rl.colocated.controller import ColocatedController
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.parallelism import (
    ParallelDims,
)
from torch.utils.data import Dataset
from typing import Callable, Optional
from cosmos_rl.utils.logging import logger
from typing import List
from cosmos_rl.colocated.utils import CommandDispatcher
from cosmos_rl.policy.worker.colocated.policy_control import (
    ColocatedPolicyControlWorker,
)
from cosmos_rl.rollout.worker.colocated.rollout_control import (
    ColocatedRolloutControlWorker,
)
from cosmos_rl.dispatcher.command import (
    PolicyToRolloutUnicastCommand,
    RolloutToRolloutBroadcastCommand,
    DataFetchCommand,
)


class ColocatedRLControlWorker:
    """
    Colocated RL Control Worker class.
    Control both policy training and rollout generation control flow in colocated mode.
    Interact both policy and rollout workers with ColocatedController in this class.
    """

    def __init__(self, config: CosmosConfig, parallel_dims: ParallelDims, **kwargs):
        """
        Initialize the Colocated GRPO Control Worker.
        Args:
            config (CosmosConfig): The configuration object.
            parallel_dims (ParallelDims): The parallelism dimensions.
        """
        self.config = config
        self.policy = ColocatedPolicyControlWorker(
            config,
            parallel_dims,
            **kwargs,
        )
        # Setting up rollout parallel dims
        rollout_parallel_dims = ParallelDims.from_config(config.rollout.parallelism)
        rollout_parallel_dims.build_mesh(device_type="cuda")
        assert (
            rollout_parallel_dims.world_size == parallel_dims.world_size
        ), "Rollout and Policy parallel dims must have the same world size in colocated mode."

        self.rollout = ColocatedRolloutControlWorker(
            config,
            rollout_parallel_dims,
            **kwargs,
        )
        self.command_dispatcher = CommandDispatcher(
            [self.rollout.replica_name, self.policy.replica_name],
            is_master=self.policy.global_rank == 0,
        )
        # Link the command dispatcher to policy and rollout workers
        self.policy.set_command_dispatcher(self.command_dispatcher)
        self.rollout.set_command_dispatcher(self.command_dispatcher)

        self.setup(
            dataset=kwargs.get("dataset", None),
            val_dataset=kwargs.get("val_dataset", None),
            sampler=kwargs.get("sampler", None),
            batch_sampler=kwargs.get("batch_sampler", None),
            val_sampler=kwargs.get("val_sampler", None),
            val_batch_sampler=kwargs.get("val_batch_sampler", None),
            custom_logger_fns=kwargs.get("custom_logger_fns", []),
        )

    def setup(
        self,
        dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        custom_logger_fns: Optional[List[Callable]] = None,
        sampler: Optional[Callable] = None,
        batch_sampler: Optional[Callable] = None,
        val_sampler: Optional[Callable] = None,
        val_batch_sampler: Optional[Callable] = None,
    ):
        """
        Setup the Colocated GRPO Control Worker.
        Args:
            dataset (Optional[Dataset]): The training dataset.
            val_dataset (Optional[Dataset]): The validation dataset.
            custom_logger_fns (Optional[List[Callable]]): Custom logger functions.
            sampler (Optional[Callable]): Sampler for training dataset.
            batch_sampler (Optional[Callable]): Batch sampler for training dataset.
            val_sampler (Optional[Callable]): Sampler for validation dataset.
            val_batch_sampler (Optional[Callable]): Batch sampler for validation dataset.
        """
        self.controller = ColocatedController()
        self.controller.setup(
            policy=self.policy,
            rollout=self.rollout,
            command_dispatcher=self.command_dispatcher,
            config=self.config,
            dataset=dataset,
            val_dataset=val_dataset,
            custom_logger_fns=custom_logger_fns,
            sampler=sampler,
            batch_sampler=batch_sampler,
            val_sampler=val_sampler,
            val_batch_sampler=val_batch_sampler,
        )
        self.policy.api_client.set_controller(self.controller)
        self.rollout.api_client.set_controller(self.controller)

    def main_loop(self):
        """
        Main loop for the Colocated GRPO Control Worker.
        Control the training and rollout generation process.
        """

        # Generate the initial commands such as WeightResume, PolicyToRolloutUnicast, etc.
        self.controller.init_commands()

        # Process the initial PolicyToRolloutUnicast command
        self.rollout.consume_command(PolicyToRolloutUnicastCommand)
        # Process the initial RolloutToRolloutBroadcast command
        self.rollout.consume_command(RolloutToRolloutBroadcastCommand)
        # Process the initial PolicyToRolloutUnicast command
        self.policy.consume_command(PolicyToRolloutUnicastCommand)

        is_end = False
        while not is_end:
            self.controller.advance_iteration()
            assert (
                self.config.train.train_batch_per_replica
                % self.config.rollout.n_generation
                % self.rollout.parallel_dims.mesh["dp"].size()
                == 0
            ), f"train_batch_per_replica {self.config.train.train_batch_per_replica} must be divisible by n_generation {self.config.rollout.n_generation} and data parallel size {self.rollout.parallel_dims.mesh['dp'].size()}."
            n_prompts_per_train = (
                self.config.train.train_batch_per_replica
                // self.config.rollout.n_generation
                // self.rollout.parallel_dims.mesh["dp"].size()
            )
            while n_prompts_per_train > 0:
                logger.debug(
                    f"[Rollout] Starting minor step generation with remain {n_prompts_per_train} prompts to generate"
                )
                is_end, processed_samples = self.rollout.rollout_for_one_minor_step()
                n_prompts_per_train -= processed_samples
                if is_end:
                    break
            self.rollout.report_rollouts(block=True)
            # Handle generate rollouts if not enough like DAPO case.
            while (
                self.controller.pending_policy_samples_all_replicas()
                < self.config.train.train_batch_per_replica
            ):
                if self.controller.pending_policy_samples() > 0:
                    logger.debug(
                        f"[Rollout] Not enough rollouts generated for training in DAPO. Current pending samples: {self.controller.pending_policy_samples()}, required: {self.config.train.train_batch_per_replica}. Keep generating rollouts..."
                    )
                is_end, _ = self.rollout.rollout_for_one_minor_step()
                self.rollout.report_rollouts(block=True)
                if is_end:
                    break
            pending_samples = self.controller.pending_policy_samples_all_replicas()
            self.controller.rollout_completed_for_data_fetch_n_training(pending_samples)
            if pending_samples >= self.config.train.train_batch_per_replica:
                self.policy.consume_command(DataFetchCommand)
            else:
                logger.warning(
                    f"[Rollout] No enough prompts to generate rollouts {pending_samples} < {self.config.train.train_batch_per_replica}."
                )
                self.policy.consume_command(DataFetchCommand, no_exec=True)
                self.controller.training_end_ack()

            need_sync_weight = (
                self.controller.rollout_consume_one_step_commands_util_r2r()
            )
            if need_sync_weight:
                # Process the PolicyToRolloutUnicast command at policy side
                self.policy.consume_command(PolicyToRolloutUnicastCommand)

                # Process the PolicyToRolloutUnicast command at rollout side
                self.rollout.consume_command(PolicyToRolloutUnicastCommand)

                # Process the RolloutToRolloutBroadcast command
                self.rollout.consume_command(RolloutToRolloutBroadcastCommand)

            if self.rollout.shutdown_signal.is_set():
                is_end = True

    def execute(self):
        """
        Execute the Colocated GRPO Control Worker's main loop.
        """
        try:
            self.main_loop()
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise e
        finally:
            self.policy.destroy_worker()
            # No need to destroy rollout worker separately in colocated mode
