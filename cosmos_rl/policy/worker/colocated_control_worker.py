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
from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.rollout.worker.rollout_control_worker import (
    DisaggregatedRolloutControlWorker,
)
from cosmos_rl.dispatcher.colocated_controller import ColocatedController
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.parallelism import (
    ParallelDims,
)
import torch
from torch.utils.data import Dataset
from typing import Union, Callable, Optional
import os
from cosmos_rl.utils.logging import logger
import cosmos_rl.utils.distributed as dist_util
import time
from queue import Queue, Empty
from cosmos_rl.dispatcher.command import (
    Command,
    BuildMeshCommand,
    PolicyToRolloutUnicastCommand,
    RolloutToRolloutBroadcastCommand,
)
from cosmos_rl.utils.parallelism_map import (
    ParallelTopoMapperGroup,
)
from typing import List
from cosmos_rl.utils import constant
from cosmos_rl.comm.base import CommMixin
from cosmos_rl.dispatcher.protocol import Role
from cosmos_rl.policy.config import Config
from cosmos_rl.dispatcher.data.schema import ConversationType, RLPayload


class ColocatedTrainer(GRPOTrainer):
    colocated = True

    def init_redis(self):
        pass

    def execute_policy_to_rollout_unicast(self, command: PolicyToRolloutUnicastCommand):
        assert command.src_replica_size == self.world_size
        if not command.src_replica_name == self.replica_name:
            logger.error(
                f"Policy {self.replica_name} received P2R command from {command.src_replica_name}, but it is not the source replica."
            )
            return False

        assert (
            self.map_w_from_policy_to_rollout is not None
        ), "No parameters to sync found."
        st = time.time()

        if self.policy_to_rollout_insts is None:
            self.policy_to_rollout_insts = []
            self.policy_to_rollout_insts = self.api_client.post_policy_shard_send_insts(
                self.global_rank
            )
        logger.debug(
            f"[Policy] Fetched {len(self.policy_to_rollout_insts)} policy_to_rollout_insts from controller."
        )
        # sort the param list by the dest_name, same as rollout
        total_bytes_sent = 0
        # There is a local-replica comm in training step
        # Here we use another comm to send weight to rollout
        # NCCL announces that multi-comm could lead to deadlocks if not synchronized
        # make sure all the send operations of all ranks are finished
        time_eclapsed = time.time() - st
        logger.info(
            f"[Policy] All {len(self.policy_to_rollout_insts)} at step {command.weight_step} send operations of finished in {time_eclapsed:.3f} seconds with {total_bytes_sent / (1024 * 1024)} MB sent."
        )
        return False

    def build_global_mesh(self, command: BuildMeshCommand):
        assert len(command.replica_name_to_rank) == 1
        self.inter_policy_nccl.replica_name_to_rank = command.replica_name_to_rank
        assert self.replica_name in command.replica_name_to_rank
        self.inter_policy_nccl.is_single_peer.set()
        self.inter_policy_nccl.is_comm_ready.set()
        return

    def consume_command(self):
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


ColocatedTrainer.register_policy_command_handler(PolicyToRolloutUnicastCommand)(
    ColocatedTrainer.execute_policy_to_rollout_unicast
)
ColocatedTrainer.register_policy_command_handler(BuildMeshCommand)(
    ColocatedTrainer.build_global_mesh
)


class ColocatedRolloutControlWorker(DisaggregatedRolloutControlWorker):
    colocated = True

    def init_redis(self):
        pass

    @torch.no_grad()
    def policy_to_rollout_unicast(self, command: PolicyToRolloutUnicastCommand):
        """
        Sync the weight from policy to rollout.
        This is Policy -> Rollout replica. Will only happen between
        a pair of policy and rollout replica.
        """
        # lazy initialization of the rollout engine.
        is_for_weight_resume = command.dst_replica_name == self.replica_name
        load_format = "auto" if is_for_weight_resume else "dummy"
        self.lazy_initialize_rollout_engine(load_format)

        if command.dst_replica_name != self.replica_name:
            return

        if not hasattr(self, "policy_to_rollout_recv_insts"):
            assert (
                not command.trainable_only
            ), "all params must be transferred at the first time P2R"
            logger.info(
                "[Rollout] Fetching policy_to_rollout_recv_insts from controller ..."
            )
            self.policy_to_rollout_recv_insts = (
                self.api_client.post_rollout_shard_recv_insts(self.global_rank)
            )
            logger.info(
                "[Rollout] Finished policy_to_rollout_recv_insts from controller."
            )
        else:
            assert (
                command.trainable_only
            ), "only trainable params should be transferred at the not first time P2R"

        self.prepare_trainable_params()
        total_recvs = 0
        total_params = 0
        for insts_group in self.policy_to_rollout_recv_insts:
            for insts_for_per_param in insts_group.param_instructions:
                total_params += 1
                total_recvs += len(insts_for_per_param.instructions)
        assert (
            total_params == len(self.recv_param_key_n_rank_list)
        ), f"Mismatch in total params and received param keys: {total_params} != {len(self.recv_param_key_n_rank_list)}"

        self.rollout.set_underlying_model(self.api_client.get_policy_model())

        with torch.cuda.stream(self.inference_stream):
            logger.info(
                f"Starting to execute {len(self.policy_to_rollout_recv_insts)}; {total_params}, {total_recvs} weight sync receives ..."
            )
            self.state.set_weight_synced()

    def broadcast_to_all_rollout_replica(
        self, broadcast_command: RolloutToRolloutBroadcastCommand
    ) -> None:
        """
        Broadcast the weight to all other rollout replicas.
        Will only happen between Rollout Replica 0 and all other Rollout Replicas.
        """
        src_replica_name: str = broadcast_command.src_replica_name
        # dst_replica_names: List[str] = broadcast_command.dst_replica_names
        # lazy initialization of the rollout engine.
        if self.replica_name != src_replica_name:
            # for replicas that needs to be broadcasted, use dummy format.
            self.lazy_initialize_rollout_engine(load_format="dummy")

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
        no_more_prompts = self.request_new_prompts(self.batch_size, self._prompt_queue)
        if self._prompt_queue.empty():
            return no_more_prompts
        # Check if the prompt is valid for the current weight version
        first_payload: RLPayload = self._prompt_queue.queue[0][0]
        is_valid_prompt_for_current_weight_version = (
            first_payload.weight_version <= self.current_weight_version
        )

        if not is_valid_prompt_for_current_weight_version:
            # Fully Synchronized mode is enabled, we need to wait until the weight version is updated
            return no_more_prompts

        payloads_list: List[RLPayload] = self._prompt_queue.get()

        rollout_results: List[RolloutResult] = self.rollout.rollout_generation(
            payloads=payloads_list,
            stream=self.inference_stream,
            data_packer=self.data_packer,
            is_validation=False,
        )

        if len(rollout_results) == 0:
            return no_more_prompts

        assert (
            len(rollout_results) == len(payloads_list)
        ), f"Error: Rollout engine returned {len(rollout_results)} for {len(payloads_list)}"

        # we need filter the result with valid completions or valid completed_conversations
        valid_result: List[RolloutResult] = []
        valid_payloads_list: List[RLPayload] = []
        if self.rollout.rollout_config.multi_turn_config.enable:
            for payload, rr in zip(payloads_list, rollout_results):
                valid_conversations: List[ConversationType] = []
                # remove those result without valid assistant message
                flag = False
                for conversation in rr.completed_conversations:
                    for msg in conversation:
                        if msg.role == "assistant" and msg.content != "":
                            flag = True
                            break
                    if flag:
                        valid_conversations.append(conversation)
                rr.completed_conversations = valid_conversations
                if len(rr.completed_conversations) > 0:
                    valid_result.append(rr)
                    valid_payloads_list.append(payload)
        else:
            # Remove empty completions
            for payload, rr in zip(payloads_list, rollout_results):
                completions = rr.completions
                skip_output = False
                total_generation_count = len(completions)
                empty_generation_count = 0
                output_texts: List[str] = []
                for j in range(total_generation_count):
                    output_text = completions[j]
                    # if output_text == "":
                    #     logger.warning(
                    #         f"[Rollout] Got empty completion for {i}th prompt {j}th generation"
                    #     )
                    #     empty_generation_count += 1
                    # else:
                    #     output_texts.append(output_text)

                    # Note: (jiaxinc)
                    # We still need to upload the output text, even if it is empty. (replace empty with eos_token)
                    # Because if fully synchronized mode is enabled, we need to make sure the expected
                    # number of global_batch_size is reached at exact time.
                    output_texts.append(
                        output_text if output_text != "" else self.eos_token
                    )
                # Skip the output if there is one or zero non-empty completions
                skip_output = (total_generation_count - empty_generation_count) <= 1
                if not skip_output:
                    rr.completions = output_texts
                    valid_result.append(rr)
                    valid_payloads_list.append(payload)

        logger.debug(f"[Rollout] generate end for rank {self.global_rank}")

        should_report = (
            self.parallel_dims.tp_coord[0] == 0
            and (self.parallel_dims.pp_coord[0] == self.parallel_dims.pp_coord[1] - 1)
            and len(valid_result) > 0
        )

        if should_report:
            # only the first tp rank in the rollout replica will post the completion to the controller.
            valid_payloads: List[RLPayload] = []

            for old_payload, result in zip(valid_payloads_list, valid_result):
                # update payload
                old_payload.completions = result.completions
                old_payload.completion_logprobs = result.completion_logprobs
                old_payload.completion_token_ids = result.completion_token_ids
                if self.rollout.rollout_config.multi_turn_config.enable:
                    old_payload.completed_conversations = result.completed_conversations
                valid_payloads.append(old_payload)

            self.reward_dispatcher.enqueue_rewards_cal(
                valid_payloads,
                False,
                self.current_weight_version,
            )

        return no_more_prompts, len(valid_payloads)

    def consume_command(
        self, cmd_pred=None, timeout=constant.COSMOS_ROLLOUT_CMD_WAIT_TIMEOUT
    ):
        """Background task to check commands from the controller"""
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


ColocatedRolloutControlWorker.register_rollout_command_handler(
    PolicyToRolloutUnicastCommand
)(ColocatedRolloutControlWorker.policy_to_rollout_unicast)
ColocatedRolloutControlWorker.register_rollout_command_handler(
    RolloutToRolloutBroadcastCommand
)(ColocatedRolloutControlWorker.broadcast_to_all_rollout_replica)


class PolicyWorkerBase(CommMixin):
    def __init__(
        self,
        config: Union[CosmosConfig],
        parallel_dims: ParallelDims,
    ) -> None:
        super().__init__()
        self.config = config
        self.parallel_dims = parallel_dims
        self.runner_init()

    def runner_init(self):
        self.role = Role.POLICY
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))  # rank in the node
        self.global_rank = int(os.environ.get("RANK", 0))  # rank in replica
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

        # Initialize the communication to controller.
        self.init_comm()
        self.init_redis()

    def set_model(self, model: torch.nn.Module):
        self.model = model

    def get_underlying_model(self):
        return self.model

    @torch.no_grad()
    def prepare_shard_infos_for_weight_sync_insts(self):
        keys_n_ranks = []
        trainable_params = self.model.trainable_params
        for name, tensor_or_callable in self.model.weight_sync_transforms:
            if isinstance(tensor_or_callable, torch.Tensor):
                keys_n_ranks.append((name, tensor_or_callable.ndim))
            else:
                assert isinstance(tensor_or_callable, Callable)
                tensor_or_callable = tensor_or_callable()
                keys_n_ranks.append((name, tensor_or_callable.ndim))
            if name not in trainable_params:
                logger.debug(f"[Policy] Not trainable for param {name}")
        local_shard_infos = ParallelTopoMapperGroup(
            self.parallel_dims,
            hf_config=self.hf_config,
            is_policy=True,
            underlying_model=self.model,
            weight_mapper=self.model.weight_mapper,
        ).prepare_local_shard_infos(keys_n_ranks, self.global_rank)
        self.all_rank_local_shard_infos = dist_util.all_gather_object_cpu(
            local_shard_infos
        )
        sorted_params_all_rank = dist_util.all_gather_object_cpu(
            sorted([x[0] for x in keys_n_ranks])
        )
        sorted_params_all_rank = [
            x
            for r, x in enumerate(sorted_params_all_rank)
            if self.parallel_dims.get_rank_in_dim("dp_cp_tp", r) == 0
        ]
        trainable_params_all_rank = dist_util.all_gather_object_cpu(trainable_params)
        self.trainable_params = set()
        for trainable_params_per_rank in trainable_params_all_rank:
            self.trainable_params.update(trainable_params_per_rank)

        if self.global_rank == 0:
            logger.info(
                f"[Policy] Parse {len(self.trainable_params)} trainable params to controller."
            )
            self.api_client.post_policy_shard_info(
                shard_infos=self.all_rank_local_shard_infos,
                param_groups=[],
                sorted_params=sorted_params_all_rank,
                trainable_params=list(self.trainable_params),
            )


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


class ColocatedGRPOControlWorker:
    def __init__(self, config: CosmosConfig, parallel_dims: ParallelDims, **kwargs):
        self.config = config
        self.parallel_dims = parallel_dims
        # GRPOTrainer.init_comm = lambda self: None
        self.policy = ColocatedTrainer(
            config,
            parallel_dims,
            **kwargs,
        )
        self.rollout: ColocatedRolloutControlWorker = ColocatedRolloutControlWorker(
            config,
            parallel_dims,
            **kwargs,
        )
        self.command_dispatcher = CommandDispatcher(
            [self.rollout.replica_name, self.policy.replica_name],
            is_master=self.policy.global_rank == 0,
        )
        self.rollout.redis_controller = self.command_dispatcher
        self.policy.redis_controller = self.command_dispatcher

    def setup(
        self,
        config: Config,
        dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        custom_logger_fns: Optional[List[Callable]] = None,
        sampler: Optional[Callable] = None,
        batch_sampler: Optional[Callable] = None,
        val_sampler: Optional[Callable] = None,
        val_batch_sampler: Optional[Callable] = None,
    ):
        self.controller = ColocatedController()
        self.controller.setup(
            policy=self.policy,
            rollout=self.rollout,
            command_dispatcher=self.command_dispatcher,
            config=config,
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
        # Generate the initial commands such as WeightResume, PolicyToRolloutUnicast, etc.
        self.controller.init()

        # Process the initial BuildMeshCommand at policy side
        self.policy.consume_command()
        # Process the initial BuildMeshCommand at rollout side
        self.rollout.consume_command()

        # Process the initial WeightResume command
        self.policy.consume_command()

        # Process the initial PolicyToRolloutUnicast command
        self.rollout.consume_command()

        # Process the initial RolloutToRolloutBroadcast command
        self.rollout.consume_command()

        # Process the initial PolicyToRolloutUnicast command
        self.policy.consume_command()

        is_end = False
        while not is_end:
            assert (
                self.config.train.train_batch_per_replica
                % self.config.rollout.n_generation
                == 0
            ), f"train_batch_per_replica {self.config.train.train_batch_per_replica} must be divisible by n_generation {self.config.rollout.n_generation}"
            n_prompts_per_train = (
                self.config.train.train_batch_per_replica
                // self.config.rollout.n_generation
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
                self.controller.pending_policy_samples()
                < self.config.train.train_batch_per_replica
            ):
                is_end, _ = self.rollout.rollout_for_one_minor_step()
                self.rollout.report_rollouts(block=True)
                if is_end:
                    break
            if self.controller.pending_policy_samples() <= 0:
                break
            self.controller.rollout_completed(self.controller.pending_policy_samples())

            # Process the PolicyToRolloutUnicast command at policy side
            self.policy.consume_command()

            # Process the PolicyToRolloutUnicast command at rollout side
            self.rollout.consume_command()

            # Process the RolloutToRolloutBroadcast command
            self.rollout.consume_command()
