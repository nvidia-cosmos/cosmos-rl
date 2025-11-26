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

from typing import List, Dict, Optional, Callable, Any
from cosmos_rl.dispatcher.controller import Controller
from cosmos_rl.dispatcher.replica import Replica
from cosmos_rl.dispatcher.protocol import Role, RolloutRequest
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.wandb_logger import (
    is_wandb_available,
    init_wandb,
)
import cosmos_rl.utils.network_util as network_util
from torch.utils.data import Dataset
from cosmos_rl.policy.config import Config
from cosmos_rl.utils.parallelism_map import ParallelizedShardMapper
from cosmos_rl.dispatcher.data.data_fetcher import ControllerDataFetcher
from cosmos_rl.policy.trainer.grpo_trainer import GRPOTrainer
from cosmos_rl.rollout.worker.rollout_control_worker import (
    DisaggregatedRolloutControlWorker,
)
from cosmos_rl.dispatcher.command import (
    WeightResumeCommand,
    PolicyToRolloutUnicastCommand,
    DataFetchCommand,
    RolloutToRolloutBroadcastCommand,
    BuildMeshCommand,
)
import numpy as np
from cosmos_rl.utils.wandb_logger import (
    log_wandb,
)
from cosmos_rl.utils.payload import extract_rollouts
from cosmos_rl.utils.util import RollingDict
from cosmos_rl.policy.model.hf_models import HFModel


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
        policy: GRPOTrainer,
        rollout: DisaggregatedRolloutControlWorker,
        command_dispatcher: Any,
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
            policy (GRPOTrainer): The policy trainer instance.
            rollout (DisaggregatedRolloutControlWorker): The rollout worker instance.
            command_dispatcher (Any): The command dispatcher for issuing commands.
            config (Config): The configuration for the controller.
            dataset (Optional[Dataset]): The training dataset.
            val_dataset (Optional[Dataset]): The validation dataset.
            custom_logger_fns (Optional[List[Callable]]): Custom logger functions.
            sampler (Optional[Callable]): The training data sampler.
            batch_sampler (Optional[Callable]): The training data batch sampler.
            val_sampler (Optional[Callable]): The validation data sampler.
            val_batch_sampler (Optional[Callable]): The validation data batch sampler.
        """
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

        self.policy_step = None
        self.rollout_step = None
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

    def get_batched_prompt(self, n, validation_step=None):
        """
        Get batched prompts from data fetcher.
        Args:
            n (int): Number of prompts to fetch.
            validation_step (Optional[int]): Current validation step, if any.
        Returns:
            Tuple[List[RLPayload], bool]: A tuple of (list of prompts as RLPayload, is_end flag).
        """
        return self.data_fetcher.get_batched_prompt(n, validation_step)

    def init(self):
        """
        Initialize the system by building meshes and triggering weight resume and policy-rollout commands.
        This is called once at the beginning to prepare the initial state including model weights.
        """
        BuildMeshCommand.trigger(
            [self.policy_replica], redis_handler=self.command_dispatcher
        )
        BuildMeshCommand.trigger(
            [self.rollout_replica], redis_handler=self.command_dispatcher
        )

        if self.policy_step is None:
            self.policy_step = 0
            WeightResumeCommand.trigger(
                replica=self.policy_replica, redis_handler=self.command_dispatcher
            )

        if self.rollout_step is None:
            PolicyToRolloutUnicastCommand.trigger(
                src_replica=self.policy_replica,
                dst_replica=self.rollout_replica,
                src_replica_size=self.policy.world_size,
                dst_replica_size=self.rollout.world_size,
                weight_step=0,
                total_steps=self.total_steps,
                redis_handler=self.command_dispatcher,
            )
            RolloutToRolloutBroadcastCommand.trigger(
                src_replica=self.rollout_replica,
                dst_replicas=[self.rollout_replica],
                weight_step=0,
                total_steps=self.total_steps,
                redis_handler=self.command_dispatcher,
            )
            self.rollout_step = 0

    def rollout_completed(self, required_rollouts: int):
        """
        Notify the controller that rollouts have been completed and trigger data fetch command.
        DataFetchCommand is triggered to fetch new data for the policy and start one training iteration.
        Args:
            required_rollouts (int): Number of rollouts that have been completed.
        """
        do_save = False
        self.current_step += 1

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
        if self.config.logging.logger and len(self.policy.data_queue.queue) > 0:
            rewards = []
            completion_lengths = []
            advantages = []
            filter_rewards = []
            for rollout in self.policy.data_queue.queue:
                rewards.append(rollout.reward)
                completion_length = (
                    len(rollout.completion_token_ids)
                    if self.config.train.train_policy.rollout_as_token_ids
                    else len(self.policy.tokenizer.encode(rollout.completion))
                )
                advantages.extend([rollout.advantage] * completion_length)
                filter_rewards.append(rollout.filter_reward)
                completion_lengths.append(completion_length)
            report_data = {
                "train/reward_mean": np.mean(rewards),
                "train/reward_std": np.std(rewards),
                "train/reward_max": np.max(rewards),
                "train/reward_min": np.min(rewards),
                "rollout/completion_length_mean": np.mean(completion_lengths),
                "rollout/completion_length_std": np.std(completion_lengths),
                "rollout/completion_length_max": np.max(completion_lengths),
                "rollout/completion_length_min": np.min(completion_lengths),
                "rollout/advantage_mean": np.mean(advantages),
                "rollout/advantage_std": np.std(advantages),
                "rollout/advantage_max": np.max(advantages),
                "rollout/advantage_min": np.min(advantages),
                "rollout/filter_reward_mean": np.mean(filter_rewards),
                "rollout/filter_reward_std": np.std(filter_rewards),
                "rollout/filter_reward_max": np.max(filter_rewards),
                "rollout/filter_reward_min": np.min(filter_rewards),
            }
            self.train_report_data[self.current_step] = report_data

        DataFetchCommand.trigger(
            replica=self.policy_replica,
            items_count=required_rollouts,
            global_step=self.policy_step + 1,
            total_steps=self.total_steps,
            remain_samples_num=self.remain_samples_num,
            # Only `do_save` when checkpointing is enabled
            do_save=do_save and self.config.train.ckpt.enable_checkpoint,
            redis_handler=self.command_dispatcher,
        )

    def train_ack(
        self,
        replica_name: str,
        step: int,
        total_steps: int,
        profile_finished: bool,
        report_data: Dict[str, Any],
    ):
        """
        Handle training acknowledgment from policy replica.
        This method records the training step, triggers weight synchronization if needed,
        and logs training metrics.
        Args:
            replica_name (str): The name of the replica sending the acknowledgment.
            step (int): The current training step of the replica.
            total_steps (int): The total number of training steps.
            profile_finished (bool): Whether profiling has finished.
            report_data (Dict[str, Any]): The training metrics to report.
        """
        if replica_name == self.policy.replica_name:
            self.policy_step = step
        else:
            raise ValueError(
                f"[Controller] train_ack received from unknown replica: {replica_name}"
            )

        if not hasattr(self, "report_data_list"):
            self.report_data_list = []
        self.report_data_list.append(report_data)

        # All replicas have been reduced, trigger allreduce
        need_sync_weight = step % self.config.train.sync_weight_interval == 0
        # If the current step is the last step, we need to sync weight always to act as ending signal
        need_sync_weight = need_sync_weight or step == total_steps
        # If validation is enabled, we need to sync weight every validation step
        if self.config.validation.enable:
            need_sync_weight = need_sync_weight or (
                step % self.config.validation.freq == 0
            )

        # Sum and report data
        if self.config.logging.logger and not all(
            [not data for data in self.report_data_list]
        ):
            try:
                total_loss_avg = np.mean(
                    [data["train/loss_avg"] for data in self.report_data_list]
                )
                total_loss_max = np.max(
                    [data["train/loss_max"] for data in self.report_data_list]
                )
                total_learning_rate = self.report_data_list[0]["train/learning_rate"]
                total_iter_time_avg = np.mean(
                    [data["train/iteration_time"] for data in self.report_data_list]
                )
                # KL loss
                total_kl_loss_avg = np.mean(
                    [data.get("train/kl_loss_avg", 0) for data in self.report_data_list]
                )
                total_kl_loss_max = np.max(
                    [data.get("train/kl_loss_max", 0) for data in self.report_data_list]
                )
                total_grad_norm = np.mean(
                    [data.get("train/grad_norm", 0) for data in self.report_data_list]
                )
                total_entropy = np.mean(
                    [data.get("train/entropy", 0) for data in self.report_data_list]
                )
                total_effective_entropy = np.mean(
                    [
                        data.get("train/effective_entropy", 0)
                        for data in self.report_data_list
                    ]
                )
                train_step = self.report_data_list[0]["train_step"]
                self.report_data_list = []

                policy_report_data = {
                    "train/loss_avg": total_loss_avg,
                    "train/loss_max": total_loss_max,
                    "train/learning_rate": total_learning_rate,
                    "train/iteration_time": total_iter_time_avg,
                    "train/kl_loss_avg": total_kl_loss_avg,
                    "train/kl_loss_max": total_kl_loss_max,
                    "train/grad_norm": total_grad_norm,
                    "train/entropy": total_entropy,
                    "train/effective_entropy": total_effective_entropy,
                }

                if len(self.filter_records) > 0:
                    total_samples_for_filtering = sum(
                        v for v in self.filter_records.values()
                    )
                    for k, v in self.filter_records.items():
                        policy_report_data.update(
                            {f"rollout/{k}_ratio": v / total_samples_for_filtering}
                        )
                self.train_report_data.setdefault(train_step, {}).update(
                    policy_report_data
                )

                if "wandb" in self.config.logging.logger and is_wandb_available():
                    log_wandb(
                        data=self.train_report_data[train_step],
                        step=train_step,
                    )
                if "console" in self.config.logging.logger:
                    logger.info(
                        f"Step: {train_step}/{total_steps}, Reward Mean: {self.train_report_data[train_step]['train/reward_mean']:.4f}, Reward Std: {self.train_report_data[train_step]['train/reward_std']:.4f}, Reward Max: {self.train_report_data[train_step]['train/reward_max']:.4f}, Reward Min: {self.train_report_data[train_step]['train/reward_min']:.4f}, Completion Length Mean: {self.train_report_data[train_step]['rollout/completion_length_mean']:.2f}, Completion Length Max: {self.train_report_data[train_step]['rollout/completion_length_max']:.2f}, Average loss: {total_loss_avg:.5f}, Max loss: {total_loss_max:.5f}, Learning rate: {total_learning_rate:.5e}, Entropy: {total_entropy:.5f}, Effective Entropy: {total_effective_entropy:.5f}, Grad Norm: {total_grad_norm:.5f}, KL Loss Avg: {total_kl_loss_avg:.5f}, KL Loss Max: {total_kl_loss_max:.5f}, Iteration time: {total_iter_time_avg:.2f}s."
                    )
                    if len(self.filter_records) > 0:
                        logger.info(
                            f"Dynamic sampling rewards distribution so far: {self.filter_records}."
                        )
                self.filter_records = {}
                for logger_fn in self.custom_logger_fns:
                    try:
                        logger_fn(self.train_report_data[train_step], train_step)
                    except Exception as e:
                        logger.warning(
                            f"[Controller] Warning reporting customized training results: {e}"
                        )
            except Exception as e:
                import traceback

                logger.warning(
                    f"[Controller] Warning reporting training results: {e}\n{traceback.format_exc()}"
                )
        # P->R & R->R
        if need_sync_weight:
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

    def put_rollouts(self, rollout: RolloutRequest):
        """
        Put rollouts into the policy's data queue.
        This method extracts rollouts from the rollout request and enqueues them for training.
        """
        rollouts_list = extract_rollouts(rollout.payloads, rollout.is_end)
        # Update the statistics for dynamic sampling used for metrics collection
        # if self.config.train.train_policy.variant == "dapo":
        #     self.policy_status_manager.update_dynamic_sampling_statistics(
        #         rollout.metrics
        #     )
        # Flatten the rollouts into a single list
        rollouts = [
            rollout
            for rollouts_group in rollouts_list
            for rollout in rollouts_group  # rollouts_group: all rollouts of the same prompt.
        ]
        if len(rollouts) > 0:
            logger.debug(
                f"[RolloutGroup] from replica: {rollout.src_replica_name} with {len(rollout.payloads)} samples:"
                f"example: rollouts[0]\n{rollouts[0]}"
            )
        for rollout in rollouts:
            self.policy.data_queue.put_nowait(rollout)

    def pending_policy_samples(self) -> int:
        """
        Get the number of pending samples in the policy's data queue.
        """
        return self.policy.data_queue.qsize()

    def get_policy_model(self) -> Any:
        """
        Get the current policy model.
        Returns:
            torch.nn.Module: The current policy model instance.
        """
        if isinstance(self.policy.model, HFModel):
            logger.info("Returning underlying HF model from policy.model")
            return self.policy.model.model
        return self.policy.model
