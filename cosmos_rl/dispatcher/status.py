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

import time
import math
from queue import Queue
from strenum import StrEnum
from typing import Dict, List, Iterator, Any, Optional, Callable
from cosmos_rl.utils.constant import COSMOS_HEARTBEAT_TIMEOUT
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.util import RollingDict
from cosmos_rl.policy.config import Config
from cosmos_rl.dispatcher.replica import Replica, Atom, Rollout
from cosmos_rl.dispatcher.protocol import Role
import cosmos_rl.dispatcher.command as command
from cosmos_rl.utils.redis_stream import RedisStreamHandler
from cosmos_rl.utils.wandb_logger import (
    is_wandb_available,
    log_wandb,
)
from cosmos_rl.dispatcher.data.data_fetcher import ControllerDataFetcher
from transformers import AutoTokenizer
import numpy as np
from cosmos_rl.utils.util import aggregate_report_data


class ReplicaScalingEnum(StrEnum):
    """
    Enum for replica scaling event.
    """

    REPLICA_SCALING_UP = "replica_scaling_up"
    REPLICA_SCALING_DOWN = "replica_scaling_down"


class ReplicaScalingLog:
    event: ReplicaScalingEnum
    replica_name: str
    timestamp: int

    def __init__(
        self, event: ReplicaScalingEnum, replica_name: str, timestamp: int = None
    ):
        self.event = event
        self.replica_name = replica_name
        self.timestamp = timestamp if timestamp is not None else int(time.time())

    @staticmethod
    def up(replica: Replica):
        return ReplicaScalingLog(ReplicaScalingEnum.REPLICA_SCALING_UP, replica.name)

    @staticmethod
    def down(replica: Replica):
        return ReplicaScalingLog(ReplicaScalingEnum.REPLICA_SCALING_DOWN, replica.name)


class PolicyStatus(StrEnum):
    """
    Enum for policy status.
    There are 7 statuses:
    UNINITIALIZED: The policy is uninitialized.
    READY: The policy is ready to run.
    RUNNING: The policy is running.
    REDUCED: The policy has finished reduce.
    END: The policy has finished.
    """

    UNINITIALIZED = "uninitialized"
    READY = "ready"
    RUNNING = "running"
    REDUCED = "reduced"
    END = "end"


class PolicyStatusManager:
    """
    A class to manage the status of a policy.
    """

    policy_replicas: Dict[str, Replica]
    policy_init_done: bool = False
    replica_scaling_log: List[ReplicaScalingLog]

    # Global status
    remain_samples_num: int
    current_step: int
    total_steps: int

    # Instance status
    status: Dict[str, PolicyStatus]

    def __init__(self):
        self.policy_replicas = {}
        # number of steps that needed to interate over all the samples across all the epochs.
        self.total_steps = 0
        # current step of the policy training, this step could won't reach to total_steps because of dynmaic sampling.
        # Some samples could be filtered out due to dynamic sampling and they won't be used for policy training.
        # This step is the actual weight update step, it is also binded to the weight version.
        self.current_step = 0

        self.rollout_buffer = Queue()
        self.remain_samples_num = 0
        self.consumed_samples_num = 0
        self.samples_on_the_fly = 0

        self.status = {}

        self.train_report_data = RollingDict(maxlen=20)

        self.replica_scaling_log = []

        # Validation related
        self.val_report_data: Dict[int, List[Any]] = {}

        # Indicate whether on-policy rollout collection has completed for the current policy step
        self.on_policy_rollout_completed: bool = False

        # Record filter rewards distribution for dynamic sampling
        self.filter_records = {}

    def setup(
        self,
        config: Config,
        redis_handler: RedisStreamHandler,
        data_fetcher: ControllerDataFetcher,
        remain_samples_num: int,
        samples_per_epoch: int,
        tokenizer: Optional[AutoTokenizer] = None,
        current_step: int = 0,
        max_num_steps: Optional[int] = None,
        custom_logger_fns: Optional[List[Callable]] = None,
        hook_fns: Optional[Dict[str, Callable]] = None,
    ):
        self.redis_handler = redis_handler
        self.config = config
        self.remain_samples_num = remain_samples_num
        self.samples_per_epoch = samples_per_epoch
        self.tokenizer = tokenizer
        self.current_step = current_step
        self.max_num_steps = max_num_steps
        self.custom_logger_fns = (
            custom_logger_fns if custom_logger_fns is not None else []
        )
        self.hook_fns = hook_fns if hook_fns is not None else {}
        self.data_fetcher = data_fetcher

        self.recompute_total_steps()

    def n_atoms_per_replica(self) -> int:
        """
        Get the number of GPUs per replica.
        """
        if len(self.policy_replicas) == 0:
            return 0
        return next(iter(self.policy_replicas.values())).n_atoms_per_replica()

    def __len__(self) -> int:
        """
        Get the number of policies.
        """
        return len(self.policy_replicas)

    def __iter__(self) -> Iterator[Replica]:
        """
        Iterate over the policy replicas.
        """
        for replica in sorted(self.policy_replicas.values(), key=lambda x: x.name):
            yield replica

    def __contains__(self, replica_name: str) -> bool:
        """
        Check if the replica is in the status manager.
        """
        return replica_name in self.policy_replicas

    def __getitem__(self, replica_name: str) -> Replica:
        """
        Get the replica from the status manager.
        """
        return self.policy_replicas.get(replica_name)

    def training_finished(self) -> bool:
        """
        Check if the training is finished.
        """
        return self.current_step >= self.total_steps and self.total_steps > 0

    def maintain_life_status(self):
        """
        Maintain the life status of the rollout.
        """
        dead_replicas = set()
        now = time.time()
        for replica in self:
            if now - replica.status.heartbeat_timestamp > COSMOS_HEARTBEAT_TIMEOUT:
                logger.warning(f"[Controller] Policy {replica.name} is dead")
                dead_replicas.add(replica.name)
        for replica_name in dead_replicas:
            self.unregister(replica_name)

    def set_status(self, name: str, status: PolicyStatus):
        """
        Set the status of the policy.
        """
        if name not in self.status:
            assert (
                status == PolicyStatus.UNINITIALIZED
            ), "Policy status should be UNINITIALIZED when first created"
            self.status[name] = status
            return
        assert (
            status != PolicyStatus.UNINITIALIZED
        ), "Policy status should not be UNINITIALIZED when already created"
        self.status[name] = status

    def recompute_total_steps(
        self, explicit_num_remaining_samples: Optional[int] = None
    ):
        """
        Set the ranks of the policies.
        """
        if self.training_finished():
            # Training is finished, do not recompute total steps
            return
        # Update total_steps based on remaining samples and replicas
        num_policy_replicas = len(self.get_all_atoms_arrived_replicas())
        if num_policy_replicas == 0:
            return

        num_remaining_samples = (
            explicit_num_remaining_samples
            if explicit_num_remaining_samples is not None
            else self.remain_samples_num
        )

        steps_by_dataset = self.current_step + num_remaining_samples // (
            self.config.train.train_batch_per_replica * num_policy_replicas
        )

        # If max_num_steps is set, honour the smaller one.
        if self.config.train.max_num_steps is not None:
            self.total_steps = min(steps_by_dataset, self.config.train.max_num_steps)
        else:
            self.total_steps = steps_by_dataset

    def get_status(self, name: str) -> PolicyStatus:
        """
        Get the status of the policy.
        """
        if name not in self.status:
            raise KeyError(f"Policy {name} not found")
        return self.status[name]

    def all_with_status(self, status: List[PolicyStatus]) -> bool:
        """
        Check if all policies have the given status.
        """
        return all([x in status for x in self.status.values()])

    def all_reduced(self) -> bool:
        """
        Check if all policies are reduced.
        """
        return self.all_with_status([PolicyStatus.REDUCED])

    def all_ready(self) -> bool:
        """
        Check if all policies are ready.
        """
        return self.all_with_status([PolicyStatus.READY])

    def all_ready_or_reduced(self) -> bool:
        """
        Check if all policies are ready or reduced.
        """
        return self.all_with_status([PolicyStatus.READY, PolicyStatus.REDUCED])

    def set_ncclerror(self, replica_name: str, timestamp: int):
        """
        Set the timeout ack of the policy.
        """
        self[replica_name].status.nccl_error_timestamp = timestamp

    def clear_ncclerror(self):
        """
        Clear the timeout ack of the policy.
        """
        for replica in self:
            replica.status.nccl_error_timestamp = None

    def get_all_policy_report_ncclerror(self) -> Dict[str, int]:
        """
        Get all the timeout ack of the policies.
        """
        return {
            replica.name: replica.status.nccl_error_timestamp
            for replica in self
            if replica.status.nccl_error_timestamp is not None
        }

    def heartbeat(self, replica_name: str):
        timestamp: int = int(time.time())
        if replica_name not in self:
            logger.warning(
                f"[Controller] Replica {replica_name} not found in policy status manager."
            )
            return
        self[replica_name].status.heartbeat_timestamp = timestamp

    def shutdown(self):
        """
        Shutdown the status manager.
        """
        self.policy_init_done = False

    def unregister(self, replica_name: str):
        """
        Unregister the replica from the status manager.
        """
        assert (
            replica_name in self
        ), f"Replica {replica_name} not found in policy status manager"

        replica = self.policy_replicas.pop(replica_name)
        self.status.pop(replica_name)
        self.replica_scaling_log.append(ReplicaScalingLog.down(replica))

        if self.training_finished():
            # This policy replica is normally finished
            # Do not trigger rebuild mesh since everything is gonna be finished shortly
            logger.info(f"[Controller] Replica {replica_name} is stopping.")
            return

        valid_replicas = self.get_all_atoms_arrived_replicas()
        if replica.in_mesh and len(valid_replicas) > 0:
            self.trigger_rebuild_mesh(valid_replicas)

    def register(
        self,
        atom: Atom,
        config: Config,
        rollout_status_manager: "RolloutStatusManager",
        **kwargs,
    ):
        """
        Register the atom to the status manager.
        """
        replica = self[atom.replica_name]
        if replica is None:
            replica = Replica(atom.replica_name, Role.POLICY, [atom])
            self.policy_replicas[atom.replica_name] = replica
        else:
            replica.arrive(atom)
        atom.bind_replica(replica)
        current_policy_replica = replica

        # post register hook
        if not self.policy_init_done:
            if len(self.policy_replicas) > config.policy.parallelism.n_init_replicas:
                config.policy.parallelism.n_init_replicas = len(self.policy_replicas)
                logger.info(
                    f"[Controller] Update policy n_init_replicas to {config.policy.parallelism.n_init_replicas} replicas"
                )

        # Check if all atoms of the replica have arrived
        if replica.all_atoms_arrived:
            if replica.start_time == -1:
                replica.start_time = int(time.time())
            logger.info(
                f"[Controller] All atoms of {Role.POLICY} Replica {replica.name} has been set."
            )
            self.set_status(replica.name, PolicyStatus.UNINITIALIZED)
            # Check total valid policy replicas
            valid_replicas = []
            if not hasattr(self, "policy_atoms_in_replica"):
                self.policy_atoms_in_replica = int(math.prod(atom.group_size))

            for r in self.policy_replicas.values():
                if r.all_atoms_arrived:
                    valid_replicas.append(r)

            # Load weight for the first loaded replica policy
            if len(valid_replicas) == 1:
                assert not hasattr(
                    self, "_first_policy_replica_arrived"
                ), "Expect only one policy replica to load weight during training process"
                self._first_policy_replica_arrived = True
                # This is the first policy replica to arrive, it is responsible for weight initialization
                command.WeightResumeCommand.trigger(
                    current_policy_replica, redis_handler=self.redis_handler
                )

                # Check whether there is any valid rollout replicas
                any_valid_rollout_replica = None
                sorted_rollout_replicas = sorted(
                    rollout_status_manager.rollout_replicas.values(),
                    key=lambda x: x.start_time,
                )
                valid_rollout_replicas = []
                for r in sorted_rollout_replicas:
                    if r.all_atoms_arrived:
                        valid_rollout_replicas.append(r)
                        if any_valid_rollout_replica is None:
                            any_valid_rollout_replica = r
                if any_valid_rollout_replica:
                    command.PolicyToRolloutUnicastCommand.trigger(
                        src_replica=current_policy_replica,
                        dst_replica=any_valid_rollout_replica,
                        src_replica_size=self.policy_atoms_in_replica,
                        dst_replica_size=rollout_status_manager.rollout_atoms_in_replica,
                        weight_step=None,
                        total_steps=None,
                        redis_handler=self.redis_handler,
                    )
                    if (
                        len(valid_rollout_replicas)
                        >= config.rollout.parallelism.n_init_replicas
                    ):
                        command.RolloutToRolloutBroadcastCommand.trigger(
                            src_replica=any_valid_rollout_replica,
                            dst_replicas=valid_rollout_replicas,
                            weight_step=None,
                            total_steps=None,
                            redis_handler=self.redis_handler,
                        )
                    logger.info(
                        f"[Controller] Trigger PolicyToRolloutUnicastCommand to {any_valid_rollout_replica.name} via Policy registration"
                    )
                else:
                    logger.info(
                        "[Controller] No valid rollout replicas found, skip PolicyToRolloutUnicastCommand"
                    )
            self.post_register_hook(
                valid_replicas,
                atom.replica,
                config,
                rollout_status_manager,
            )
        return replica

    def trigger_rebuild_mesh(self, valid_replicas: List[Replica]):
        # Always tell the policy to rebuild mesh even there is only one policy replica
        sorted_valid_replicas = sorted(valid_replicas, key=lambda x: x.start_time)
        command.BuildMeshCommand.trigger(
            sorted_valid_replicas, redis_handler=self.redis_handler
        )
        self.recompute_total_steps()

    def post_register_hook(
        self,
        valid_replicas: List[Replica],
        target_replica: Replica,
        config: Config,
        rollout_status_manager: "RolloutStatusManager",
    ):
        sorted_valid_replicas = sorted(valid_replicas, key=lambda x: x.start_time)

        if config.validation.enable and config.validation.val_before_train:
            self.data_fetcher.validation_activate_dataloader(0)

        if (
            not self.policy_init_done
            and len(valid_replicas) >= config.policy.parallelism.n_init_replicas
        ):
            # This is the case when all required replicas have arrived

            self.policy_init_done = True
            # Trigger mesh building (Typically only occurs during initialization)

            # we need buildmesh, event there is only one replica. (trigger HANccl buildmesh)
            # 1. Trigger mesh building
            self.trigger_rebuild_mesh(valid_replicas)

            # 2. Trigger weight/optimizer state synchronization
            if len(valid_replicas) > 1:
                # Only broadcast when there are multiple policy replicas
                initialized_replica = None
                for replica in sorted_valid_replicas:
                    # We will select the first replica that has weights loaded in view of command
                    if (
                        replica.weights_loaded_in_view_of_command
                        and replica in valid_replicas
                    ):
                        initialized_replica = replica
                        break
                assert (
                    initialized_replica is not None
                ), "No replica was selected to load weights"
                command.PolicyToPolicyBroadcastCommand.trigger(
                    src_replica=initialized_replica,
                    dst_replicas=valid_replicas,
                    redis_handler=self.redis_handler,
                )
            # Set all policy replicas to `ready`
            for replica in valid_replicas:
                self.set_status(replica.name, PolicyStatus.READY)

            if self.config.mode == "colocated":
                # In colocated mode, we initially trigger data fetch for step 1 since the rollouts are generated locally.
                self.current_step += 1
                if self.config.validation.enable and (
                    self.current_step % self.config.validation.freq == 0
                    or self.current_step == self.total_steps
                ):
                    self.data_fetcher.validation_activate_dataloader(self.current_step)

                for replica in valid_replicas:
                    self.remain_samples_num -= self.config.train.train_batch_per_replica
                    command.DataFetchCommand.trigger(
                        replica=replica,
                        items_count=self.config.train.train_batch_per_replica,
                        global_step=self.current_step,
                        total_steps=self.total_steps,
                        # `remain_samples_num` is just for checkpointing the training progress
                        remain_samples_num=self.remain_samples_num,
                        # Only `do_save` when checkpointing is enabled
                        do_save=False,
                        redis_handler=self.redis_handler,
                    )
                    self.set_status(replica.name, PolicyStatus.RUNNING)
                    logger.info(
                        f"[Controller] Policy Replica {replica.name} is ready in colocated mode."
                    )
        elif (
            not self.policy_init_done
            and len(valid_replicas) < config.policy.parallelism.n_init_replicas
        ):
            # This is the case when replicas are in the initialization stage
            logger.info(
                f"Waiting for {config.policy.parallelism.n_init_replicas - len(valid_replicas)} more replicas to arrive"
            )
        else:
            # This is the case when the dynamic scaling is triggered
            assert (
                self.policy_init_done
            ), "Policy initialization must be done before building another mesh"

            assert (
                target_replica.status.mesh_rank == -1
            ), "Target replica should not be in the mesh"

            # This occurs when new dynamic scaling is triggered
            initialized_replica = None
            for replica in sorted_valid_replicas:
                if (
                    replica.weights_loaded_in_view_of_command
                    and replica in valid_replicas
                ):
                    # We will select the first replica that has weights loaded in view of command
                    # to broadcast weights
                    initialized_replica = replica
                    break
            assert (
                initialized_replica is not None
            ), "No replica was selected to load weights"
            self.trigger_rebuild_mesh(valid_replicas)

            command.PolicyToPolicyUnicastCommand.trigger(
                src_replica=initialized_replica,
                dst_replica=target_replica,
                redis_handler=self.redis_handler,
            )
            self.set_status(target_replica.name, PolicyStatus.READY)

    def validation_report_validation_results(
        self,
        validation_step: int,
        validation_results: List[List[Rollout]],
        rollout_status_manager: "RolloutStatusManager",
    ):
        if validation_step not in self.val_report_data:
            self.val_report_data[validation_step] = []

        self.val_report_data[validation_step].extend(validation_results)
        n_items_of_this_step = sum(
            len(x) for x in self.val_report_data[validation_step]
        )

        validation_finished = n_items_of_this_step == (
            self.data_fetcher.val_datasize or len(self.data_fetcher.val_dataloader)
        )

        if self.data_fetcher.activated_val_tqdm:
            self.data_fetcher.activated_val_tqdm.update(n_items_of_this_step)
        else:
            logger.error("[Controller] Validation tqdm is not activated")
        # Check if all rollout replicas have reported validation results
        if validation_finished and self.data_fetcher.activated_val_iter is not None:
            # Validation is finished, trigger next step training
            self.data_fetcher.clear_validation_status()

            try:
                all_rollouts_lists: List[List[Rollout]] = self.val_report_data[
                    validation_step
                ]
                if all_rollouts_lists:
                    rewards = []
                    for rollouts in all_rollouts_lists:
                        rewards.extend([r.reward for r in rollouts])
                    avg_reward = np.mean(rewards)
                    std_reward = np.std(rewards)
                    max_reward = np.max(rewards)
                    min_reward = np.min(rewards)

                    report_data = {
                        "val/reward_avg": avg_reward,
                        "val/reward_std": std_reward,
                        "val/reward_max": max_reward,
                        "val/reward_min": min_reward,
                        "val/rollout_count": len(rewards),
                        "val/step": validation_step,
                        "val/train_total_steps": self.total_steps,  # the total steps of the training when current validation step is triggered. This total_steps may change due to dynamic sampling.
                    }
                    logger.info(
                        f"[Controller] Validation finished, average reward: {avg_reward}, total rollouts: {len(rewards)}, max reward: {max_reward}, min reward: {min_reward}, std reward: {std_reward} at step {validation_step}"
                    )
                    report_data_list = [
                        rollout.report_metrics
                        if rollout.report_metrics is not None
                        else {}
                        for rollouts in all_rollouts_lists
                        for rollout in rollouts
                    ]
                    report_data = aggregate_report_data(
                        report_data_list, report_data, prefix="val/"
                    )
                    report_data_str = ", ".join(
                        [f"{k}: {v}" for k, v in report_data.items()]
                    )
                    logger.info(
                        f"[Controller] Validation report data from total {sum(len(rollouts) for rollouts in all_rollouts_lists)} rollouts: {report_data_str}"
                    )
                    if "wandb" in self.config.logging.logger and is_wandb_available():
                        log_wandb(
                            data=report_data,
                            step=validation_step,
                        )

                    # call custom logger fns
                    for custom_logger_fn in self.custom_logger_fns:
                        try:
                            custom_logger_fn(report_data, validation_step)
                        except Exception as e:
                            logger.warning(
                                f"[Controller] Error calling custom logger function: {e}"
                            )

            except Exception as e:
                logger.error(f"[Controller] Error reporting validation results: {e}")

            # The order is important, because the previous code block logs the previous step's validation results
            # while `try_trigger_data_fetch_and_training` will immediately report the next step's results
            self.try_trigger_data_fetch_and_training()

    def total_pending_rollouts(self) -> int:
        """
        Get the total pending rollouts.
        """
        return self.rollout_buffer.qsize()

    def get_all_atoms_arrived_replicas(self) -> List[Replica]:
        """
        Get all the replicas that have all atoms arrived.
        """
        return [
            replica
            for replica in self.policy_replicas.values()
            if replica.all_atoms_arrived
        ]

    def put_rollout(self, rollout: Rollout):
        """
        Dispatch the rollout to the policy replicas in a round-robin manner.
        It is that replica's responsibility to dispatch the rollout to further (DP_SHARD) atoms.
        """
        if self.config.rollout.include_stop_str_in_output:
            if self.tokenizer.eos_token is not None and rollout.completion is not None:
                if not rollout.completion.endswith(self.tokenizer.eos_token):
                    rollout.completion = rollout.completion + self.tokenizer.eos_token
                    if (
                        self.config.rollout.multi_turn_config.enable
                        and rollout.completed_conversation[-1].role == "assistant"
                    ):
                        rollout.completed_conversation[
                            -1
                        ].content += self.tokenizer.eos_token
        self.rollout_buffer.put(rollout)
        self.try_trigger_data_fetch_and_training()

    def put_rollouts(self, rollouts: List[Rollout]):
        """
        Put the rollouts to the rollout buffer.
        """
        completion_tokens_count = 0
        n_samples = 0
        if (
            self.config.train.train_policy.on_policy
            and self.on_policy_rollout_completed
        ):
            # On-policy training has already completed the required samples for this policy step
            # Filter out all these rollouts directly in remaining samples number
            self.remain_samples_num -= len(rollouts)
            return completion_tokens_count, n_samples

        for rollout in rollouts:
            if self.config.train.train_policy.rollout_as_token_ids:
                completion_tokens_count += len(rollout.completion_token_ids)
            elif not self.config.train.non_text:
                completion_tokens_count += len(
                    self.tokenizer.encode(rollout.completion)
                )
            n_samples += 1
            self.put_rollout(rollout)
            self.consumed_samples_num += 1
            if self.config.train.train_policy.on_policy:
                if self.total_pending_rollouts() == 0:
                    self.on_policy_rollout_completed = True
                    break

        return completion_tokens_count, n_samples

    def update_dynamic_sampling_statistics(self, filter_records: Dict[str, int]):
        """
        Update the dynamic sampling statistics.
        """
        for k in ["sampled", "filtered_positive", "filtered_negative"]:
            self.filter_records[k] = self.filter_records.get(k, 0) + filter_records.get(
                k, 0
            )

        # Update the remaining samples number to reflect the filtering results
        self.remain_samples_num -= filter_records.get("filtered_positive", 0)
        self.remain_samples_num -= filter_records.get("filtered_negative", 0)

    def filter_outdated_rollouts(self, rollouts: List[Rollout]) -> List[Rollout]:
        """
        Filter out the outdated rollouts based on the current step.
        """
        filtered_rollouts = []
        for idx, rollout in enumerate(rollouts):
            assert (
                rollout.weight_version <= self.current_step
            ), f"Rollout weight version {rollout.weight_version} is greater than current step {self.current_step}"
            # Estimate the step when this rollout will be used for training
            # This is estimated based on the current step, the number of pending rollouts,
            # and the number of rollouts before this rollout in the current batch.
            estimated_step = self.current_step + (
                idx + self.total_pending_rollouts()
            ) // (
                self.config.train.train_batch_per_replica
                * max(len(self.get_all_atoms_arrived_replicas()), 1)
            )
            if (
                estimated_step - rollout.weight_version
                <= self.config.train.train_policy.allowed_outdated_steps
            ):
                filtered_rollouts.append(rollout)
            else:
                logger.debug(
                    f"[Controller] Filtered out outdated rollout with version {rollout.weight_version}, current step {self.current_step}, estimated step {estimated_step}, pending rollouts {self.total_pending_rollouts()}, preceeding rollouts in this batch {idx}, allowed_outdated_steps {self.config.train.train_policy.allowed_outdated_steps}"
                )
        # Update remaining samples number
        self.remain_samples_num -= len(rollouts) - len(filtered_rollouts)
        k = "outdated"
        self.filter_records[k] = (
            self.filter_records.get(k, 0) + len(rollouts) - len(filtered_rollouts)
        )
        return filtered_rollouts

    def train_ack(
        self,
        replica_name: str,
        step: int,
        total_steps: int,
        profile_finished: bool,
        report_data: Dict[str, Any],
        rollout_status_manager: "RolloutStatusManager",
    ):
        if replica_name not in self:
            raise Exception(f"Replica {replica_name} not found")

        self.set_status(replica_name, PolicyStatus.REDUCED)

        if not hasattr(self, "report_data_list"):
            self.report_data_list = []

        self.report_data_list.append(report_data)
        if self.all_reduced():
            self.samples_on_the_fly -= self.config.train.train_batch_per_replica * len(
                self.get_all_atoms_arrived_replicas()
            )
            assert (
                self.samples_on_the_fly >= 0
            ), "samples_on_the_fly should not be negative"
            # All replicas have been reduced, trigger allreduce
            need_sync_weight = step % self.config.train.sync_weight_interval == 0
            # If the current step is the last step, we need to sync weight always to act as ending signal
            need_sync_weight = need_sync_weight or step == total_steps
            # If validation is enabled, we need to sync weight every validation step
            if self.config.validation.enable:
                need_sync_weight = need_sync_weight or (
                    step % self.config.validation.freq == 0
                )

            if profile_finished:
                # Only reset the do_profile flag if the profile is finished
                logger.debug(f"[Controller] Unset the profile mode of {replica_name}")
                self[replica_name].sub_profiler_config.do_profile = False

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
                    total_learning_rate = self.report_data_list[0][
                        "train/learning_rate"
                    ]
                    total_iter_time_avg = np.mean(
                        [data["train/iteration_time"] for data in self.report_data_list]
                    )
                    # KL loss
                    total_kl_loss_avg = np.mean(
                        [
                            data.get("train/kl_loss_avg", 0)
                            for data in self.report_data_list
                        ]
                    )
                    total_kl_loss_max = np.max(
                        [
                            data.get("train/kl_loss_max", 0)
                            for data in self.report_data_list
                        ]
                    )
                    total_grad_norm = np.mean(
                        [
                            data.get("train/grad_norm", 0)
                            for data in self.report_data_list
                        ]
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
                        "train/total_steps": total_steps,
                    }
                    policy_report_data = aggregate_report_data(
                        self.report_data_list, policy_report_data
                    )
                    if self.config.mode == "colocated":
                        for data in self.report_data_list:
                            # Handle dynamic sampling statistics update in colocated mode
                            self.update_dynamic_sampling_statistics(data)

                    if len(self.filter_records) > 0:
                        total_samples_for_filtering = sum(
                            v for v in self.filter_records.values()
                        )
                        if total_samples_for_filtering > 0:
                            for k, v in self.filter_records.items():
                                policy_report_data.update(
                                    {
                                        f"rollout/{k}_ratio": v
                                        / total_samples_for_filtering
                                    }
                                )
                    self.train_report_data.setdefault(train_step, {}).update(
                        policy_report_data
                    )
                    self.report_data_list = []

                    report_data_str = ", ".join(
                        [
                            f"{k}: {v}"
                            for k, v in self.train_report_data[train_step].items()
                        ]
                    )
                    logger.info(
                        f"[Controller] Train report data from total {self.config.train.train_batch_per_replica * len(self.get_all_atoms_arrived_replicas())} rollouts: {report_data_str}"
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
                    for custom_logger_fn in self.custom_logger_fns:
                        # We add a separate try-except block to handle the error of custom logger function.
                        # This is to avoid the error of custom logger function affecting the fundamental logging system.
                        try:
                            custom_logger_fn(
                                self.train_report_data[train_step], train_step
                            )
                        except Exception as e:
                            logger.warning(
                                f"[Controller] [Controller] Error calling custom logger function: {e}"
                            )
                except Exception as e:
                    import traceback

                    logger.warning(
                        f"[Controller] Warning reporting training results: {e}\n{traceback.format_exc()}"
                    )

            # All replicas have been reduced, trigger weight sync
            any_loaded_replica = None
            sorted_replicas = sorted(
                self.get_all_atoms_arrived_replicas(), key=lambda x: x.start_time
            )
            for replica in sorted_replicas:
                if any_loaded_replica is None:
                    any_loaded_replica = replica
                self.set_status(replica.name, PolicyStatus.READY)

            # P->R & R->R
            if need_sync_weight:
                self.trigger_weight_sync(
                    any_loaded_replica, rollout_status_manager, step, total_steps
                )
            # Trigger next step training if data is available
            self.try_trigger_data_fetch_and_training()
            if self.config.train.train_policy.on_policy:
                # Reset on-policy rollout completed flag for next step
                self.on_policy_rollout_completed = False

    def trigger_weight_sync(
        self,
        policy_replica: Replica,
        rollout_status_manager: "RolloutStatusManager",
        current_step: int,
        total_steps: int,
    ):
        any_loaded_rollout_replica = None
        valid_rollout_replicas = []
        sorted_replicas = sorted(
            rollout_status_manager.get_all_atoms_arrived_replicas(),
            key=lambda x: x.start_time,
        )
        for rollout_replica in sorted_replicas:
            if any_loaded_rollout_replica is None:
                any_loaded_rollout_replica = rollout_replica
            valid_rollout_replicas.append(rollout_replica)
        if any_loaded_rollout_replica is None:
            return
        command.PolicyToRolloutUnicastCommand.trigger(
            src_replica=policy_replica,
            dst_replica=any_loaded_rollout_replica,
            src_replica_size=self.policy_atoms_in_replica,
            dst_replica_size=rollout_status_manager.rollout_atoms_in_replica,
            weight_step=current_step,
            total_steps=total_steps,
            redis_handler=self.redis_handler,
        )

        command.RolloutToRolloutBroadcastCommand.trigger(
            src_replica=any_loaded_rollout_replica,
            dst_replicas=valid_rollout_replicas,
            weight_step=current_step,
            total_steps=total_steps,
            redis_handler=self.redis_handler,
        )

    def rollouts_enough_for_one_step(self) -> bool:
        """
        Check if the rollouts are enough.
        """
        if self.config.mode == "colocated":
            # Colocated mode always has enough rollouts since they are locally prepared.
            return True

        return self.total_pending_rollouts() >= (
            self.config.train.train_batch_per_replica
            * len(self.get_all_atoms_arrived_replicas())
        )

    def try_trigger_data_fetch_and_training(self, is_fake_last_cmd=False):
        # If the validation dataloader is activated, do not trigger data fetch and training
        if self.data_fetcher.activated_val_iter is not None:
            return

        arrived_replicas = self.get_all_atoms_arrived_replicas()
        # no replicas arrived, do nothing
        if len(arrived_replicas) == 0:
            return

        if self.training_finished():
            return

        if is_fake_last_cmd:
            required_rollouts = 0
            all_ready_or_reduced = True
            items_count = 0
            assert (
                self.current_step + 1 == self.total_steps
            ), "The last command should be fake and next step should be the last step"
        else:
            items_count = self.config.train.train_batch_per_replica
            required_rollouts = items_count * len(arrived_replicas)
            all_ready_or_reduced = (
                self.all_ready_or_reduced() and self.rollouts_enough_for_one_step()
            )

        # If the last command is fake, we need to trigger data fetch and training no matter
        # whether there are enough rollouts or whether replicas are `ready` or `reduced`.
        if all_ready_or_reduced:
            rollouts_of_this_step: List[Rollout] = []
            # Decrease the consumed rollouts number.
            self.remain_samples_num -= required_rollouts

            # From controller's perspective, the training step is already increased
            self.current_step += 1

            if self.config.validation.enable and (
                self.current_step % self.config.validation.freq == 0
                or self.current_step == self.total_steps
            ):
                self.data_fetcher.validation_activate_dataloader(self.current_step)

            # FIXME: (lms) will this dipatch style cause non-alignment with VeRL?
            # This dispatch style will cause rollouts from same prompt may be dispatched to different replicas.
            # Interleave-style data dispatch
            if not self.config.mode == "colocated":
                # Colocated mode no need real rollout dispatching since they are all local.
                for _ in range(items_count):
                    for replica in arrived_replicas:
                        rollout = self.rollout_buffer.get()
                        replica.put_rollout(rollout, self.redis_handler)
                        rollouts_of_this_step.append(rollout)

            # Decide whether to save checkpoint
            # First check if we need to save checkpoint based on epoch
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

            for replica in arrived_replicas:
                command.DataFetchCommand.trigger(
                    replica=replica,
                    items_count=items_count,
                    global_step=self.current_step,
                    total_steps=self.total_steps,
                    # `remain_samples_num` is just for checkpointing the training progress
                    remain_samples_num=self.remain_samples_num,
                    # Only `do_save` when checkpointing is enabled
                    do_save=do_save and self.config.train.ckpt.enable_checkpoint,
                    redis_handler=self.redis_handler,
                )
                self.set_status(replica.name, PolicyStatus.RUNNING)

            # Report the reward, length, etc.
            # These properties are already ready to be reported before being trained
            if self.config.logging.logger and rollouts_of_this_step:
                rewards = []
                completion_lengths = []
                advantages = []
                filter_rewards = []
                for rollout in rollouts_of_this_step:
                    rewards.append(rollout.reward)
                    completion_length = (
                        (
                            len(rollout.completion_token_ids)
                            if self.config.train.train_policy.rollout_as_token_ids
                            else len(self.tokenizer.encode(rollout.completion))
                        )
                        if not self.config.train.non_text
                        else 1
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

                report_data_list = [
                    rollout.report_metrics if rollout.report_metrics is not None else {}
                    for rollout in rollouts_of_this_step
                ]
                report_data = aggregate_report_data(
                    report_data_list, report_data, prefix="train/"
                )
                self.train_report_data[self.current_step] = report_data


class RolloutStatusManager:
    """
    A class to manage the status of rollout replicas.
    """

    rollout_replicas: Dict[str, Replica]
    rollout_init_done: bool
    replica_scaling_log: List[ReplicaScalingLog]

    def __init__(self):
        self.rollout_replicas = {}
        self.rollout_init_done = False
        self.replica_scaling_log = []

    def setup(
        self,
        config: Config,
        redis_handler: RedisStreamHandler,
    ):
        self.redis_handler = redis_handler
        self.config = config
        """
        Maintain the life status of the policy and rollout replicas.
        """
        return len(self.rollout_replicas)

    def n_atoms_per_replica(self) -> int:
        """
        Get the number of GPUs per replica.
        """
        if len(self.rollout_replicas) == 0:
            return 0
        return next(iter(self.rollout_replicas.values())).n_atoms_per_replica()

    def __len__(self) -> int:
        """
        Get the number of rollout replicas.
        """
        return len(self.rollout_replicas)

    def __iter__(self) -> Iterator[Replica]:
        """
        Iterate over the policy replicas.
        """
        for replica in sorted(self.rollout_replicas.values(), key=lambda x: x.name):
            yield replica

    def __contains__(self, replica_name: str) -> bool:
        """
        Check if the replica is in the status manager.
        """
        return replica_name in self.rollout_replicas

    def __getitem__(self, replica_name: str) -> Replica:
        """
        Get the replica from the status manager.
        """
        return self.rollout_replicas.get(replica_name)

    def maintain_life_status(self, policy_status_manager: PolicyStatusManager):
        """
        Maintain the life status of the rollout.
        """
        now = time.time()
        dead_replicas = set()
        for replica in self:
            if now - replica.status.heartbeat_timestamp > COSMOS_HEARTBEAT_TIMEOUT:
                logger.warning(f"[Controller] Rollout {replica.name} is dead")
                dead_replicas.add(replica.name)
        for replica_name in dead_replicas:
            self.unregister(replica_name, policy_status_manager=policy_status_manager)

    def heartbeat(self, replica_name: str):
        timestamp: int = int(time.time())
        if replica_name not in self:
            logger.warning(
                f"[Controller] Replica {replica_name} not found in both policy and rollout."
            )
            return
        self[replica_name].status.heartbeat_timestamp = timestamp

    ############################################################
    # utility functions
    ############################################################
    def get_all_atoms_arrived_replicas(self) -> List[Replica]:
        """
        Get all the replicas that have all atoms arrived.
        """
        return [
            replica
            for replica in self.rollout_replicas.values()
            if replica.all_atoms_arrived
        ]

    def unregister(self, replica_name: str, policy_status_manager: PolicyStatusManager):
        """
        Unregister the replica from the status manager.
        """
        assert (
            replica_name in self
        ), f"Replica {replica_name} not found in policy status manager"

        replica = self.rollout_replicas.pop(replica_name)
        self.replica_scaling_log.append(ReplicaScalingLog.down(replica))
        if policy_status_manager.training_finished():
            # This policy replica is normally finished
            # Do not trigger rebuild mesh since everything is gonna be finished shortly
            logger.info(f"[Controller] Replica {replica_name} is stopping.")
            return
        if replica.in_mesh and len(self.rollout_replicas) > 0:
            self.trigger_rebuild_mesh(self.get_all_atoms_arrived_replicas())

    def register(
        self,
        atom: Atom,
        config: Config,
        policy_status_manager: PolicyStatusManager,
        **kwargs,
    ):
        """
        Register the atom to the status manager.
        """
        replica = self[atom.replica_name]
        if replica is None:
            replica = Replica(atom.replica_name, Role.ROLLOUT, [atom])
            self.rollout_replicas[atom.replica_name] = replica
        else:
            replica.arrive(atom)
        atom.bind_replica(replica)

        # post register hook
        if not self.rollout_init_done:
            if len(self.rollout_replicas) > config.rollout.parallelism.n_init_replicas:
                config.rollout.parallelism.n_init_replicas = len(self.rollout_replicas)
                logger.info(
                    f"[Controller] Update rollout n_init_replicas to {config.rollout.parallelism.n_init_replicas} replicas"
                )

        # Check if all atoms of the replica have arrived
        if replica.all_atoms_arrived:
            if replica.start_time == -1:
                replica.start_time = int(time.time())
            logger.info(
                f"[Controller] All atoms of {Role.ROLLOUT} Replica {replica.name} has been set."
            )
            # Check total valid rollout replicas
            valid_replicas = []
            if not hasattr(self, "rollout_atoms_in_replica"):
                self.rollout_atoms_in_replica = int(math.prod(atom.group_size))
            for replica in self.rollout_replicas.values():
                if replica.all_atoms_arrived:
                    valid_replicas.append(replica)
            self.post_register_hook(
                valid_replicas,
                atom.replica,
                config,
                policy_status_manager,
            )
        return replica

    def rollout_end(self, replica_name: str):
        """
        Rollout end event.
        """
        replica = self[replica_name]
        if replica is None:
            logger.warning(
                f"[Controller] Rollout {replica_name} not found in RolloutStatusManager"
            )
            return
        replica.status.ended = True

    def all_rollouts_ended(self) -> bool:
        """
        Check if all rollouts have ended.
        """
        return all([replica.status.ended for replica in self.rollout_replicas.values()])

    def trigger_rebuild_mesh(
        self,
        valid_replicas: List[Replica],
    ):
        sorted_valid_replicas = sorted(valid_replicas, key=lambda x: x.start_time)
        command.BuildMeshCommand.trigger(
            sorted_valid_replicas, redis_handler=self.redis_handler
        )

    def post_register_hook(
        self,
        valid_replicas: List[Replica],
        target_replica: Replica,
        config: Config,
        policy_status_manager: PolicyStatusManager,
    ):
        assert target_replica in valid_replicas
        any_loaded_policy_replica = None
        sorted_valid_policy_replicas = sorted(
            [r for r in policy_status_manager], key=lambda x: x.start_time
        )
        for replica in sorted_valid_policy_replicas:
            if replica.weights_loaded_in_view_of_command:
                # We will select the first replica that has weights loaded in view of command
                # to broadcast weights
                any_loaded_policy_replica = replica
                break

        # First P->R Unicast if the policy is ready and all rollout replicas are not ready
        if (
            all(
                [
                    not replica.weights_loaded_in_view_of_command
                    for replica in valid_replicas
                ]
            )
            and any_loaded_policy_replica is not None
        ):
            command.PolicyToRolloutUnicastCommand.trigger(
                src_replica=any_loaded_policy_replica,
                dst_replica=target_replica,
                src_replica_size=policy_status_manager.policy_atoms_in_replica,
                dst_replica_size=self.rollout_atoms_in_replica,
                weight_step=None,
                total_steps=None,
                redis_handler=self.redis_handler,
            )
            logger.info(
                f"[Controller] Trigger PolicyToRolloutUnicastCommand to {target_replica.name} via Rollout registration"
            )
        else:
            logger.info(
                "[Controller] No valid policy replicas found in Rollout registration or some rollout already get weight from policy, skip PolicyToRolloutUnicastCommand"
            )

        was_already_initialized = self.rollout_init_done

        if (
            not was_already_initialized
            and len(valid_replicas) == config.rollout.parallelism.n_init_replicas
        ):
            self.rollout_init_done = True
            self.trigger_rebuild_mesh(valid_replicas)

            # ONLY ONCE PER LIFE CYCLE
            # Trigger RolloutToRolloutBroadcastCommand only once after all initial rollout replicas are loaded
            any_loaded_rollout_replica = None
            sorted_valid_replicas = sorted(valid_replicas, key=lambda x: x.start_time)
            for replica in sorted_valid_replicas:
                if (
                    replica.weights_loaded_in_view_of_command
                    and replica in valid_replicas
                ):
                    # We will select the first replica that has weights loaded in view of command
                    # to broadcast weights
                    any_loaded_rollout_replica = replica
                    break
            if any_loaded_rollout_replica is not None:
                command.RolloutToRolloutBroadcastCommand.trigger(
                    src_replica=any_loaded_rollout_replica,
                    dst_replicas=valid_replicas,
                    weight_step=None,
                    total_steps=None,
                    redis_handler=self.redis_handler,
                )
        elif not self.rollout_init_done:
            assert len(valid_replicas) < config.rollout.parallelism.n_init_replicas
            logger.info(
                f"Waiting for {config.rollout.parallelism.n_init_replicas - len(valid_replicas)} more replicas to arrive"
            )
        else:
            # Dynamic mesh building, no matter what the length of valid_replicas is,
            # we will always trigger mesh building if there are more than one rollout replicas
            self.trigger_rebuild_mesh(valid_replicas)
