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

import torch
import atexit
import time
import msgpack
import asyncio
import threading
from functools import partial
from typing import List, Optional, Union, Callable, Dict
from torch.utils.data import Dataset
from queue import Queue
import torch.distributed as dist
from queue import Empty

from cosmos_rl.dispatcher.data.packer.base import BaseDataPacker
from cosmos_rl.dispatcher.data.data_fetcher import WorkerDataFetcher
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.comm.base import CommMixin
from cosmos_rl.policy.trainer.base import TrainerRegistry
from cosmos_rl.dispatcher.data.schema import Rollout
from cosmos_rl.policy.trainer.llm_trainer.grpo_trainer import GRPOTrainer
from cosmos_rl.utils.util import is_master_rank, str2torch_dtype
from cosmos_rl.utils.distributed import HighAvailabilitylNccl, destroy_distributed
from cosmos_rl.utils.parallelism_map import (
    ParallelTopoMapperGroup,
)
from cosmos_rl.utils.pynccl import (
    create_nccl_uid,
    create_nccl_comm,
    nccl_send,
    nccl_group_start,
    nccl_group_end,
)
from cosmos_rl.dispatcher.command import (
    Command,
    BuildMeshCommand,
    PolicyToPolicyBroadcastCommand,
    PolicyToRolloutUnicastCommand,
    PolicyToPolicyUnicastCommand,
    DataFetchCommand,
    WeightResumeCommand,
)
import cosmos_rl.utils.distributed as dist_util
from cosmos_rl.utils import constant
from cosmos_rl.policy.worker.base import PolicyWorkerBase


class RLPolicyWorker(PolicyWorkerBase):
    """
    RL Policy Worker. This worker is responsible for the training of the RL.
    It interacts with the controller to fetch rollouts and commands, dispatch
    rollouts to RL trainer for step training.
    """

    config: CosmosConfig

    def __init__(self, config: CosmosConfig, parallel_dims: ParallelDims, **kwargs):
        assert isinstance(
            config, CosmosConfig
        ), "config must be a CosmosConfig object for this trainer"
        super().__init__(config, parallel_dims=parallel_dims)

        self.report_data = {}
        self.upload_thread = None

        # Model Status related
        self.model_ready = False

        # Initialize the trainer
        dataset = kwargs.get("dataset", None)
        data_packer = kwargs.get("data_packer", None)
        val_dataset = kwargs.get("val_dataset", None)
        val_data_packer = kwargs.get("val_data_packer", None)
        self.build_runner(
            dataset=dataset,
            data_packer=data_packer,
            val_dataset=val_dataset,
            val_data_packer=val_data_packer,
        )

        # Dist related

        # For mesh build
        self.inter_policy_nccl = HighAvailabilitylNccl(
            replica_name=self.replica_name,
            global_rank=self.global_rank,
            api_client=self.api_client,
        )
        self.rollouts_comm = {}
        self.kv_store = dist_util.DistKVStore(
            group=dist.distributed_c10d._get_default_group(),
            master_rank=0,
            shutdown_event=self.shutdown_signal,
        )

        # For command fetch
        self.fetch_command_buffer = Queue()
        self.command_buffer = Queue()

        # For rollouts fetch
        self.data_queue = Queue()
        self.replica_batch_for_this_step = 0

        # For Polocy to Rollout weight mapping
        self.policy_to_rollout_insts = None

        self.fetch_command_thread = None
        self.fetch_rollouts_thread = None

        atexit.register(self.handle_shutdown)

        self.p2r_nccl_uuids = {}

        # Flag for determining if the current replica is the master replica,
        # The master replica needs to:
        # - Save the checkpoint/safetensors
        self.is_master_replica = True
        self.prepare_shard_infos_for_weight_sync_insts()

        # For teacher model interaction
        self.teacher_interact_queue = Queue()
        self.teacher_interact_thread: Optional[threading.Thread] = None

    def setup(
        self,
        dataset: Optional[Union[Dataset, Callable[[CosmosConfig], Dataset]]] = None,
        val_dataset: Optional[Union[Dataset, Callable[[CosmosConfig], Dataset]]] = None,
        data_packer: Optional[BaseDataPacker] = None,
        val_data_packer: Optional[BaseDataPacker] = None,
    ):
        # setup data packer first
        self.init_data_packer(
            data_packer=data_packer,
            val_data_packer=val_data_packer,
        )
        # Set up data fetcher
        self.data_fetcher = WorkerDataFetcher(
            config=self.config,
            dataset=dataset,
            val_dataset=val_dataset,
            data_packer=self.data_packer,
            val_data_packer=self.val_data_packer,
            is_rl=True,
        )

    @torch.no_grad()
    def prepare_shard_infos_for_weight_sync_insts(self):
        keys_n_ranks = []
        trainable_params = self.trainer.model.trainable_params
        for name, tensor_or_callable in self.trainer.model.weight_sync_transforms:
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
            underlying_model=self.trainer.model,
            weight_mapper=self.trainer.model.weight_mapper,
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

    def handle_shutdown(self):
        if not hasattr(self, "_handle_shutdown_called"):
            self._handle_shutdown_called = True

            self.shutdown_signal.set()
            self.shutdown_mp_signal.set()
            self.inter_policy_nccl.shutdown()
            if self.fetch_rollouts_thread is not None:
                self.fetch_rollouts_thread.join()
                self.fetch_rollouts_thread = None

            if self.fetch_command_thread is not None:
                self.fetch_command_thread.join()
                self.fetch_command_thread = None

            if self.teacher_interact_thread is not None:
                self.teacher_interact_thread.join()
                self.teacher_interact_thread = None

            if hasattr(self, "heartbeat_thread") and self.heartbeat_thread is not None:
                self.heartbeat_thread.join()
                self.heartbeat_thread = None

            # Manually unregister from controller
            self.unregister_from_controller()

            if hasattr(self, "upload_thread") and self.upload_thread is not None:
                logger.info("[Policy] Waiting for upload thread to finish...")
                self.upload_thread.join()
                logger.info("[Policy] Upload thread finished.")
                self.upload_thread = None

            # TODO(jiaxin)
            # The background threads are daemon threads, so that they will exit when the main thread exits
            # However, the previous `.join()` may not really wait for them to stop.
            # So we need to wait for a while to ensure they have a chance to exit to prevent `exitcode:-6`

            # Another notice is that make sure the background threads detect the shutdown event in less than 15 seconds
            # Otherwise, the main thread may exit before the background threads detect the shutdown event
            time.sleep(15)

    async def fetch_rollouts(self):
        assert self.global_rank == 0, "Only rank 0 can fetch rollouts"
        while not self.shutdown_signal.is_set():
            rollouts: List[Rollout] = []
            try:
                rollouts = [
                    Rollout.model_validate(msgpack.unpackb(x))
                    for x in self.redis_controller.subscribe_rollout(self.replica_name)
                ]
            except Exception as e:
                logger.debug(
                    f"[Policy] Failed to get rollouts: {e}, wait for next round"
                )
            for rollout in rollouts:
                if rollout.teacher_result_uuid:
                    self.teacher_interact_queue.put_nowait(rollout.teacher_result_uuid)
                self.data_queue.put_nowait(rollout)

    def pre_P2R_collect_parameters(self):
        needed_tensors = []
        for insts_group in self.policy_to_rollout_insts:
            for insts_for_per_param in insts_group.param_instructions:
                dest_name = insts_for_per_param.param_name
                needed_tensors.append(dest_name)
        prepared_tensor_to_rollout = {}
        for dest_name, local_view in self.trainer.map_w_from_policy_to_rollout.items():
            if isinstance(
                local_view, Callable
            ) and self.trainer.weight_mapper.policy_pre_P2R_gather_required_for_sync(
                dest_name
            ):
                view = local_view()
                if dest_name in needed_tensors:
                    prepared_tensor_to_rollout[dest_name] = view
        return prepared_tensor_to_rollout

    @CommMixin.register_policy_command_handler(PolicyToPolicyBroadcastCommand)
    def execute_policy_to_policy_broadcast(
        self, command: PolicyToPolicyBroadcastCommand
    ):
        send = self.replica_name == command.src_replica_name
        recv = self.replica_name in command.dst_replica_names and not send
        if not send and not recv:
            return True
        st = time.time()
        # TODO(zjx): there need failure tolerance for nccl send and recv, so get nccl param from command
        send_recv_hook = partial(
            self.inter_policy_nccl.broadcast, src_replica=command.src_replica_name
        )
        len_params = self.sync_all_states(
            is_send=send,
            send_hook=send_recv_hook,
            recv_hook=send_recv_hook,
            reference_model=self.config.train.train_policy.kl_beta != 0.0,
        )
        if recv:
            self.model_ready = True
        time_eclapsed = time.time() - st
        logger.debug(
            f"[Policy] Policy2Policy Broadcast {len_params} parameters from {command.src_replica_name} (rank {self.inter_policy_nccl.get_replica_rank(command.src_replica_name)}) to {len(command.dst_replica_names)} replicas took {time_eclapsed:.3f} seconds."
        )
        return False

    @CommMixin.register_policy_command_handler(PolicyToPolicyUnicastCommand)
    def execute_policy_to_policy_unicast(self, command: PolicyToPolicyUnicastCommand):
        send = self.replica_name == command.src_replica_name
        recv = self.replica_name == command.dst_replica_name
        if not send and not recv:
            return False
        st = time.time()
        # TODO(zjx): there need failure tolerance for nccl send and recv, so get nccl param from command
        send_hook = partial(
            self.inter_policy_nccl.send, dst_replica=command.dst_replica_name
        )
        recv_hook = partial(
            self.inter_policy_nccl.recv, src_replica=command.src_replica_name
        )
        len_params = self.sync_all_states(
            is_send=send,
            send_hook=send_hook,
            recv_hook=recv_hook,
            reference_model=self.config.train.train_policy.kl_beta != 0.0,
        )
        if recv:
            self.model_ready = True
        time_eclapsed = time.time() - st
        logger.debug(
            f"[Policy] Policy2Policy Unicast {len_params} parameters from {command.src_replica_name} (rank {self.inter_policy_nccl.get_replica_rank(command.src_replica_name)}) to {command.dst_replica_name} (rank {self.inter_policy_nccl.get_replica_rank(command.dst_replica_name)}) as sender {send} took {time_eclapsed:.3f} seconds."
        )
        return False

    @CommMixin.register_policy_command_handler(PolicyToRolloutUnicastCommand)
    def execute_policy_to_rollout_unicast(self, command: PolicyToRolloutUnicastCommand):
        assert command.src_replica_size == self.world_size
        if not command.src_replica_name == self.replica_name:
            logger.error(
                f"[Policy] {self.replica_name} received P2R command from {command.src_replica_name}, but it is not the source replica."
            )
            return False

        comm_id = {}
        # Create nccl id for one policy replica to another rollout replica
        mesh_key = command.src_replica_name + "_" + command.dst_replica_name
        if mesh_key not in self.p2r_nccl_uuids:
            nccl_uuid = None
            if self.global_rank == 0:
                # Only create nccl group id in rank 0.
                nccl_uuid = create_nccl_uid()
                logger.debug(f"[Policy] mesh_key: {mesh_key}")
                self.api_client.post_nccl_comm_initiator(mesh_key, nccl_uuid)
            # broadcast the nccl group id to all ranks
            nccl_uuid = dist_util.broadcast_object_cpu(nccl_uuid)
            self.p2r_nccl_uuids[mesh_key] = nccl_uuid

        if mesh_key not in self.rollouts_comm:
            assert mesh_key in self.p2r_nccl_uuids
            nccl_uuid = self.p2r_nccl_uuids[mesh_key]
            logger.debug(
                f"[Policy] Creating nccl communicator for `P2R` with mesh_key: {mesh_key}"
            )
            comm_id = create_nccl_comm(
                nccl_uuid,
                self.global_rank,
                self.world_size + command.dst_replica_size,
            )
            logger.debug(
                f"[Policy] `P2R` nccl comm: {comm_id} for `P2R` with mesh_key: {mesh_key} is created."
            )
            self.rollouts_comm[mesh_key] = comm_id
        else:
            comm_id = self.rollouts_comm[mesh_key]
        assert (
            self.trainer.map_w_from_policy_to_rollout is not None
        ), "No parameters to sync found."
        st = time.time()

        if self.policy_to_rollout_insts is None:
            self.policy_to_rollout_insts = []
            self.policy_to_rollout_insts = self.api_client.post_policy_shard_send_insts(
                self.global_rank
            )
        # sort the param list by the dest_name, same as rollout
        total_bytes_sent = 0
        # There is a local-replica comm in training step
        # Here we use another comm to send weight to rollout
        # NCCL announces that multi-comm could lead to deadlocks if not synchronized
        with torch.cuda.stream(self.train_stream):
            with torch.no_grad():
                try:
                    if self.config.policy.lora is not None:
                        from cosmos_rl.policy.lora.plugin import (
                            merge_lora_weights_,
                            unmerge_lora_weights_,
                        )

                        # FIXME: (lms) move this to the trainer
                        merge_lora_weights_(self.trainer.model)

                    pre_P2R_collected_tensors: Dict[str, torch.Tensor] = (
                        self.pre_P2R_collect_parameters()
                    )

                    def grouped_send(grouped_send_ops):
                        nccl_group_start(comm_id)
                        for view, r_rank, dest_name in grouped_send_ops:
                            logger.debug(
                                f"[Policy] Sending tensor {dest_name} from policy rank {self.global_rank} to rollout rank {r_rank}, shape {view.shape} with dtype: {view.dtype}."
                            )
                            nccl_send(
                                view,
                                self.world_size + r_rank,
                                comm_id,
                            )
                        nccl_group_end(comm_id)
                        grouped_send_ops.clear()

                    grouped_send_ops = []
                    num_groups = 0

                    transferred_params_cnt = 0
                    skipped_params_cnt = 0
                    for insts_group in self.policy_to_rollout_insts:
                        for insts_for_per_param in insts_group.param_instructions:
                            dest_name = insts_for_per_param.param_name
                            if (
                                dest_name not in self.trainable_params
                                and command.trainable_only
                            ):
                                logger.debug(
                                    f"[Policy] Skip {dest_name} in P2R send due to non trainable."
                                )
                                skipped_params_cnt += 1
                                continue
                            transferred_params_cnt += 1

                            for inst in insts_for_per_param.instructions:
                                p_rank = inst.policy_rank
                                r_rank = inst.rollout_rank
                                tensor_split_strategys = inst.slice_strategy
                                if (
                                    dest_name
                                    not in self.trainer.map_w_from_policy_to_rollout
                                ):
                                    raise RuntimeError(
                                        f"dest_name {dest_name} not in trainer's map_w_from_policy_to_rollout"
                                    )
                                local_view = self.trainer.map_w_from_policy_to_rollout[
                                    dest_name
                                ]
                                if dest_name in pre_P2R_collected_tensors:
                                    local_view = pre_P2R_collected_tensors[dest_name]
                                elif isinstance(local_view, Callable):
                                    local_view = local_view()
                                else:
                                    pass
                                local_view = local_view.to(
                                    str2torch_dtype(self.config.train.transfer_dtype)
                                )
                                view = (
                                    local_view.cosmos_slice(tensor_split_strategys)
                                    .contiguous()
                                    .cuda()
                                )
                                assert self.global_rank == p_rank
                                logger.debug(
                                    f"[Policy] Sending {dest_name} from policy rank {self.global_rank} to rollout rank {r_rank}, {view.shape} with dtype: {view.dtype}."
                                )
                                grouped_send_ops.append((view, r_rank, dest_name))
                                total_bytes_sent += view.numel() * view.element_size()
                        num_groups += 1
                        if num_groups == constant.COSMOS_P2R_NCCL_GROUP_SIZE:
                            grouped_send(grouped_send_ops)
                            num_groups = 0

                    grouped_send(grouped_send_ops)
                finally:
                    if self.config.policy.lora is not None:
                        # Always attempt to unmerge to restore training state
                        # FIXME: (lms) move this to the trainer
                        unmerge_lora_weights_(self.trainer.model)

                if command.trainable_only:
                    if not hasattr(self, "synced_trainable_params"):
                        self.synced_trainable_params = transferred_params_cnt
                    else:
                        assert (
                            self.synced_trainable_params == transferred_params_cnt
                        ), "Trainable synced params count must match at each weight sync."

        # make sure all the send operations of all ranks are finished
        time_eclapsed = time.time() - st
        logger.debug(
            f"[Policy] All {len(self.policy_to_rollout_insts)} at step {command.weight_step} send operations of finished in {time_eclapsed:.3f} seconds with {total_bytes_sent / (1024 * 1024)} MB sent. While {skipped_params_cnt} non-trainable splitted params skipped and {transferred_params_cnt} splitted params transferred."
        )
        return False

    @CommMixin.register_policy_command_handler(WeightResumeCommand)
    def execute_weight_resume(self, command: WeightResumeCommand = None):
        return self.trainer.weight_resume()

    @CommMixin.register_policy_command_handler(DataFetchCommand)
    def execute_data_fetch(self, command: DataFetchCommand):
        if command.do_profile:
            self.profiler.start_dynamic(
                active_steps=command.active_steps,
                rank_filter=command.rank_filter,
                record_shape=command.record_shape,
                profile_memory=command.profile_memory,
                with_stack=command.with_stack,
                with_modules=command.with_modules,
            )

        assert self.replica_name == command.replica_name
        self.replica_batch_for_this_step = command.items_count

        is_fake_step = self.replica_batch_for_this_step == 0
        if not is_fake_step:
            self.trainer.update_lr_schedulers(command.total_steps)
            report_data = self.trainer.step_training(
                rollouts=self.dispatch_rollouts(),
                current_step=command.global_step,
                total_steps=command.total_steps,
                remain_samples_num=command.remain_samples_num,
                do_save_checkpoint=command.do_save,
                inter_policy_nccl=self.inter_policy_nccl,
                is_master_replica=self.is_master_replica,
            )
        else:
            report_data = {}
            logger.info(
                f"[Policy] No data to fetch for global step {command.global_step}, skip this step."
            )

        # For profiling
        self.profiler.step()

        # Train ACK
        if is_master_rank(self.parallel_dims, self.global_rank):
            self.api_client.post_policy_train_ack(
                self.replica_name,
                command.global_step,
                command.total_steps,
                self.profiler.check_finished(),
                report_data,
            )

        logger.debug(f"[Policy] Train ack sent for global step {command.global_step}.")
        return command.replica_should_stop()

    async def fetch_command(self):
        # assert self.global_rank == 0, "Only rank 0 can fetch command"
        while not self.shutdown_signal.is_set():
            # TODO(zjx): will remove separate BuildMeshCommand, and here only fetch other commands
            if self.global_rank == 0:
                # rank 0 will get command from redis
                # and broadcast the buildmesh command to all ranks
                commands = []
                try:
                    commands = self.redis_controller.subscribe_command(
                        self.replica_name
                    )
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

    def execute_command(self, command: Command):
        logger.debug(f"[Policy] Process command {command._serialize()}")

        handler = self.get_policy_command_handler(type(command))
        if handler is None:
            raise Exception(f"No such command supoorted in policy {command}")
        should_abort = handler(self, command)
        logger.debug(
            f"[Policy] Command {command._serialize()} executed with abort: {should_abort}"
        )
        return should_abort

    def broadcast_command(self):
        command = []
        if self.global_rank == 0:
            while len(self.fetch_command_buffer.queue) > 0:
                command.append(self.fetch_command_buffer.get_nowait())
        command = dist_util.broadcast_object_cpu(
            command, src=0, device=torch.device("cpu")
        )
        if len(command) > 0:
            for c in command:
                self.command_buffer.put_nowait(c)

    def dispatch_rollouts(self) -> List[Rollout]:
        def preprocess_rollouts(rollouts: List[Rollout]) -> List[Rollout]:
            """
            Processing rollouts that retrieved from the controller,
            including:
            - Getting the prompt and conversation from the local dataset if local_dataset is enabled
            - Getting the teacher result from the Redis if the teacher result uuid is not empty
            """
            assert all(
                rollout.prompt_idx >= 0 for rollout in rollouts
            ), "All rollouts from controller should have a valid prompt index"
            for i in range(len(rollouts)):
                if self.config.train.local_dataset:
                    # Populate the prompt and conversation from the local dataset
                    rollouts[i].prompt = self.data_fetcher.get_payload_by_index(
                        rollouts[i].prompt_idx
                    )
                    rollouts[i].conversation = self.data_fetcher.get_payload_by_index(
                        rollouts[i].prompt_idx,
                        attr="conversation",
                    )
            return rollouts

        rollouts = [[]]
        scattered_rollouts = [[] for _ in range(self.world_size)]
        if self.global_rank == 0:
            batch_for_this_step = (
                self.replica_batch_for_this_step
                // self.dp_world_size
                * self.dp_world_size
            )
            assert batch_for_this_step % self.dp_world_size == 0

            dp_id = 0
            for _ in range(batch_for_this_step):
                try:
                    rollout = self.data_queue.get(block=True, timeout=None)
                except Empty:
                    raise Empty(
                        "[Policy] Rollouts queue is empty, please check the dispatcher."
                    )
                for i in range(self.world_size):
                    if self.parallel_dims.get_rank_in_dim("dp", i) == dp_id:
                        scattered_rollouts[i].append(rollout)
                        # logger.info(f"[Policy] Rollout {dp_id} dispatched to rank {i}, dp world_size {self.dp_world_size}")
                dp_id += 1
                if dp_id >= self.dp_world_size:
                    dp_id = 0
        if self.world_size == 1:
            return preprocess_rollouts(scattered_rollouts[0])

        dist.scatter_object_list(
            rollouts,
            scattered_rollouts,
            src=0,
        )
        return preprocess_rollouts(rollouts[0])

    def teacher_interact_loop(self):
        """Background task to interact with teacher model for distillation"""
        assert self.global_rank == 0, "Only rank 0 can fetch rollouts"
        while not self.shutdown_signal.is_set():
            if not self.teacher_interact_queue.empty():
                teacher_result_uuid = self.teacher_interact_queue.get_nowait()
                logger.debug(
                    f"[Policy] Getting teacher result {teacher_result_uuid} from Redis"
                )
                # Interactive with teacher if the teacher result uuid is not empty
                teacher_result = self.redis_controller.get_teacher_result(
                    teacher_result_uuid
                )
                if teacher_result is None:
                    teacher_logprobs = None
                    logger.error(
                        f"[Policy] Failed to get teacher result {teacher_result_uuid} from Redis"
                    )
                else:
                    teacher_logprobs = teacher_result.get("teacher_logprobs", None)
                logger.debug(
                    f"[Policy] Teacher result: {len(teacher_logprobs) if teacher_logprobs is not None else 0} items"
                )
                if not hasattr(self.trainer, "teacher_interact_results"):
                    self.trainer.teacher_interact_results = {}
                self.trainer.teacher_interact_results[teacher_result_uuid] = (
                    teacher_logprobs
                )
            time.sleep(0.01)

    def main_loop(self):
        def fetch_command_helper(trainer: GRPOTrainer):
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            new_loop.run_until_complete(trainer.fetch_command())
            new_loop.stop()
            new_loop.close()
            return

        def fetch_rollouts_helper(trainer: GRPOTrainer):
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            new_loop.run_until_complete(trainer.fetch_rollouts())
            new_loop.stop()
            new_loop.close()
            return

        # Start the thread with daemon=True, so it will exit when the main program exits.
        # we need all ranks have fetch_command_thread, so that buildmesh command can be broadcasted to all ranks
        # TODO(zjx): we will only let rank 0 fetch and broadcast command
        self.fetch_command_thread = threading.Thread(
            target=fetch_command_helper,
            args=(self,),
            daemon=True,
            name="fetch_command_thread",
        ).start()

        if self.global_rank == 0:
            self.fetch_rollouts_thread = threading.Thread(
                target=fetch_rollouts_helper,
                args=(self,),
                daemon=True,
                name="fetch_rollouts_thread",
            ).start()
            self.teacher_interact_thread = threading.Thread(
                target=self.teacher_interact_loop,
                daemon=True,
                name="teacher_interact_thread",
            ).start()

        abort = False
        while True:
            abort_at_this_round = abort
            if abort_at_this_round:
                # Wait 30s to make sure the final potential P->R is received to finalize the Rollouts
                time.sleep(30)

            self.broadcast_command()
            while len(self.command_buffer.queue) > 0:
                cmd = self.command_buffer.get_nowait()
                abort = self.execute_command(cmd) or abort

            if abort_at_this_round:
                break
        logger.info("[Policy] Main loop finished. Shutdown background task event set.")
        self.train_stream.synchronize()
        self.handle_shutdown()

    def sync_all_states(
        self,
        is_send: bool,
        send_hook: callable,
        recv_hook: callable,
        reference_model: bool = False,
    ) -> int:
        return self.trainer.sync_all_states(
            is_send, send_hook, recv_hook, reference_model
        )

    def build_runner(
        self,
        dataset: Optional[Union[Dataset, Callable[[CosmosConfig], Dataset]]] = None,
        data_packer: Optional[BaseDataPacker] = None,
        val_dataset: Optional[Union[Dataset, Callable[[CosmosConfig], Dataset]]] = None,
        val_data_packer: Optional[BaseDataPacker] = None,
    ):
        # Initialize data packer and setup data fetcher first.
        self.setup(
            dataset=dataset,
            data_packer=data_packer,
            val_dataset=val_dataset,
            val_data_packer=val_data_packer,
        )

        self.trainer = TrainerRegistry.get_trainer_cls(
            self.config.train.train_policy.trainer_type
        )(
            self.config,
            self.parallel_dims,
            device=self.device,
            train_stream=self.train_stream,
            data_packer=self.data_packer,
            val_data_packer=self.val_data_packer,
        )

    def destroy_worker(self):
        destroy_distributed()
        logger.info("[Policy] Process group destroyed.")
