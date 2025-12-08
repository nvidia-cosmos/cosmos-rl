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
import time
import threading
from functools import partial
from typing import List, Optional, Callable, NamedTuple
import torch.distributed as dist

import tensorrt_llm
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm.executor.ipc import ZeroMqQueue as IpcQueue
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.rollout import TRTLLMRolloutWorkerBase
from cosmos_rl.rollout import State
from cosmos_rl.utils.parallelism_map import (
    ParallelTopoMapperGroup,
    WeightSyncInstructionsGroup,
)
from cosmos_rl.dispatcher.command import (
    BuildMeshCommand,
    PolicyToRolloutUnicastCommand,
    RolloutToRolloutBroadcastCommand,
    Command,
)
from cosmos_rl.utils.pynccl import (
    create_nccl_uid,
    create_nccl_comm,
    nccl_broadcast,
    nccl_recv,
    nccl_group_start,
    nccl_group_end,
)

from cosmos_rl.policy.model import ModelRegistry, WeightMapper
from cosmos_rl.utils import constant
from cosmos_rl.utils.logging import logger
import cosmos_rl.utils.distributed as dist_util
import cosmos_rl.utils.util as util
from cosmos_rl.rollout.trtllm_rollout.trtllm_common import (
    ValidationInstruction,
    ShutdownInstruction,
    RolloutWrapperInstruction,
)

trtllm_version = tensorrt_llm.__version__
logger.info(f"[Rollout] Using trtllm version: {trtllm_version}")
if trtllm_version == "1.0.0rc6":
    from cosmos_rl.rollout.trtllm_rollout.exec_loops.trtllm_1_0_0_rc6 import (
        cosmos_patched_executor_loop,
    )
else:
    raise NotImplementedError(f"Unsupported trtllm version: {trtllm_version}")

from transformers import AutoConfig

from queue import Queue

"""
1. Extend PyExecutor to support Cosmos-specific features.
2. Initialize distributed environment.
"""


class CosmosWorkerCommIpcAddrs(NamedTuple):
    replica_name_queue_addr: tuple[str, Optional[bytes]]
    weight_sync_queue_addr: tuple[str, Optional[bytes]]


class TrtLLMRolloutWorker(TRTLLMRolloutWorkerBase):
    # Trtllm will init worker twice, only for the second time, cosmos-grpo should begin to work.
    init_count = 0
    ready = False

    def __init__(self, *args, **kwargs) -> None:
        cosmos_config = kwargs.pop("cosmos_config")
        self.cosmos_ipc_queues: CosmosWorkerCommIpcAddrs = kwargs.pop(
            "cosmos_ipc_queues"
        )
        self.cosmos_replica_name_queue = IpcQueue(
            self.cosmos_ipc_queues.replica_name_queue_addr,
            is_server=False,
            name="py_executor_replica_name_queue",
        )
        self.cosmos_weight_sync_queue = IpcQueue(
            self.cosmos_ipc_queues.weight_sync_queue_addr,
            is_server=False,
            name="py_executor_weight_sync_queue",
        )

        super().__init__(*args, **kwargs)

        if TrtLLMRolloutWorker.init_count > 0:
            self.ready = True
            parallel_dims = ParallelDims.from_config(
                parallesim_config=cosmos_config.rollout.parallelism
            )
            self.parallel_dims = parallel_dims

            # build the mesh
            self.parallel_dims.build_mesh(device_type="cuda")

            self.post_init(cosmos_config, parallel_dims)

            self.cosmos_state = State()

            # CommandQueue queried from controller.
            self._command_queue: Queue[Command] = Queue()

            self.global_commnicator_idex = -1
            self.rank_in_rollout_repicas = -1

            self.policy_to_rollout_nccl_communicators = {}

            self.cosmos_batch_size = self.config.rollout.batch_size

            # For Polocy to Rollout weight mapping
            hf_config = util.retry(AutoConfig.from_pretrained)(
                self.config.policy.model_name_or_path,
                trust_remote_code=True,
            )
            self.hf_config = hf_config

            model_type = hf_config.model_type
            if not ModelRegistry.check_model_type_supported(model_type):
                logger.warning(
                    f"[Rollout] Replica can not find {model_type} in weight mapper, use {constant.COSMOS_HF_MODEL_TYPES} model type instead, with replica name: {self.replica_name}"
                )
                model_type = constant.COSMOS_HF_MODEL_TYPES
            self.weight_mapper = WeightMapper.get_weight_mapper(model_type)(hf_config)
            self.weight_mapper.setup_rollout_backend("trtllm")

            self.cosmos_model_config = hf_config
            self.enable_validation = self.config.validation.enable
            self.inference_stream = torch.cuda.current_stream()

            self._engine_initialized = False

            # If already registered, notify the main process.
            if hasattr(self, "replica_name"):
                # make sure all ranks went to here.
                dist.barrier()
                if self.global_rank == 0:
                    # only the rank 0 could notify the main process.
                    self.cosmos_replica_name_queue.put(self.replica_name)

        TrtLLMRolloutWorker.init_count += 1


class CosmosTRTLLMWorker(TrtLLMRolloutWorker, PyExecutor):
    """
    CosmosTRTLLMExecutor is a wrapper of PyExecutor to support Cosmos-specific features.
    P2R and R2R of cosmos-rl are implemented in this class.
    """

    def __init__(self, *args, **kwargs) -> None:
        # just call the init of PyExecutor
        super().__init__(*args, **kwargs)

    def prepare_shard_infos_for_weight_sync_insts(self):
        self.vllm_weight_inplace_view_map, grouped_recv_param_key_n_rank_list = (
            self.weight_mapper.cosmos_rollout_prepare_recv(self.get_underlying_model())
        )
        self.recv_param_key_n_rank_list = []
        param_groups = []
        for group in grouped_recv_param_key_n_rank_list:
            self.recv_param_key_n_rank_list.extend(group)
            if len(group) > 1:
                param_groups.append([x[0] for x in group])
        self.recv_param_key_n_rank_list = sorted(
            self.recv_param_key_n_rank_list, key=lambda x: x[0]
        )

        local_shard_infos = ParallelTopoMapperGroup(
            self.parallel_dims,
            self.cosmos_model_config,
            is_policy=False,
            underlying_model=self.get_underlying_model(),
            backend=self.backend,
            weight_mapper=self.weight_mapper,
        ).prepare_local_shard_infos(self.recv_param_key_n_rank_list, self.global_rank)

        self.all_rank_local_shard_infos = dist_util.all_gather_object_cpu(
            local_shard_infos
        )
        all_param_groups = dist_util.all_gather_object_cpu(param_groups)
        merged_groups = {}
        for r, param_groups in enumerate(all_param_groups):
            if self.parallel_dims.get_rank_in_dim("dp_cp_tp", r) != 0:
                continue
            for group in param_groups:
                group = sorted(group)
                key = tuple(group)
                if key not in merged_groups:
                    merged_groups[key] = group
        sorted_params_all_rank = dist_util.all_gather_object_cpu(
            [x[0] for x in self.recv_param_key_n_rank_list]
        )
        sorted_params_all_rank = [
            x
            for r, x in enumerate(sorted_params_all_rank)
            if self.parallel_dims.get_rank_in_dim("dp_cp_tp", r) == 0
        ]
        if self.global_rank == 0:
            self.api_client.post_rollout_shard_info(
                self.all_rank_local_shard_infos,
                list(merged_groups.values()),
                sorted_params_all_rank,
            )

    def get_underlying_model(self):
        """
        Get the underlying parallelized model in vLLM internal.
        """
        if "qwen2_5_vl" in self.hf_config.model_type:
            if not hasattr(self.model_engine.model, "visual"):
                # trick to get visual parameters also iterated.
                self.model_engine.model.visual = (
                    self.model_engine.model.mm_encoder.visual
                )
        return self.model_engine.model

    @TRTLLMRolloutWorkerBase.register_rollout_command_handler(
        BuildMeshCommand, backend="trtllm"
    )
    def build_global_mesh(self, build_mesh_command: BuildMeshCommand):
        replica_name_to_rank = build_mesh_command.replica_name_to_rank
        if self.replica_name not in replica_name_to_rank:
            raise RuntimeError(
                f"[Rollout] Replica {self.replica_name} not found in registered replicas."
            )
        self.rank_in_rollout_repicas = replica_name_to_rank[self.replica_name]

        if len(replica_name_to_rank) == 1:
            # only one rollout replica now, no need to build mesh.
            return
        # generate key for storing the NCCL group id.
        # group_0: [rank 0 in replica 0, rank 0 in replica 1, ..., rank 0 in replica n-1]
        # group_1: [rank 1 in replica 0, rank 1 in replica 1, ..., rank 1 in replica n-1]
        # ...
        # group_m-1: [rank m-1 in replica 0, rank m-1 in replica 1, ..., rank m-1 in replica n-1]
        unique_rollout_group_key = self.get_group_unique_key(replica_name_to_rank)
        logger.debug(
            f"[Rollout] Building global mesh for {self.replica_name} with unique_rollout_group_key: {unique_rollout_group_key}"
        )

        nccl_group_id = None
        if self.rank_in_rollout_repicas == 0:
            # only replica_rank == 0 have the right to generate nccl id.
            nccl_group_id = create_nccl_uid()
            self.api_client.post_nccl_comm_initiator(
                unique_rollout_group_key, nccl_group_id
            )
            logger.debug(f"[Rollout] post nccl group_id to controller: {nccl_group_id}")
        if self.rank_in_rollout_repicas != 0:
            # other replicas should query the nccl group id from controller
            # all ranks need to wait for the rollout replica 0 finished the group_id post
            # and then they can get the group_id from controller
            # all ranks not zero in replica 0 or all ranks of other replicas need to query the group_id from controller
            nccl_group_id = self.api_client.post_nccl_comm_acceptor(
                unique_rollout_group_key
            )
            if nccl_group_id is None:
                raise RuntimeError(
                    "[Rollout] Failed to query nccl group_id from controller!"
                )
            logger.debug(
                f"[Rollout] query nccl group_id from controller: {nccl_group_id} for global_rank: {self.global_rank}"
            )

        # update the cached communicator index
        logger.debug(
            f"[Rollout] Creating nccl communicator for global mesh: {unique_rollout_group_key}"
        )
        self.global_commnicator_idex = create_nccl_comm(
            nccl_group_id, self.rank_in_rollout_repicas, len(replica_name_to_rank)
        )
        # update the replcia_name to rank dict
        self.replica_name_to_rank = replica_name_to_rank

    def recv_weight_shard(
        self,
        global_rank_of_rollout: int,
        insts_group: WeightSyncInstructionsGroup,
        communicator_index: int,
        do_weight_sync_check: bool = False,
    ):
        check_inside_group = do_weight_sync_check

        total_bytes_received = 0

        all_cloned_target_tensors = []
        tensors_to_check = []

        for insts_for_per_param in insts_group.param_instructions:
            # insts_for_per_param: WeightSyncInstructionsPerParam -> inst collection for a single tensor
            insts = insts_for_per_param.instructions
            # insts: List[Tuple[int, int, Dict[int, Any]]]
            inst_dest_name = insts_for_per_param.param_name
            target_tensor = self.vllm_weight_inplace_view_map[inst_dest_name]

            if check_inside_group:
                cloned_target_tensor = target_tensor.clone()
                # clear the current view
                target_tensor.zero_()

            for inst in insts:
                p_rank = inst.policy_rank
                r_rank = inst.rollout_rank
                tensor_split_strategys = inst.slice_strategy
                assert r_rank == global_rank_of_rollout
                vllm_tensor_view = target_tensor.cosmos_slice(tensor_split_strategys)
                recv_tensor = None
                if vllm_tensor_view.is_contiguous():
                    recv_tensor = vllm_tensor_view
                else:
                    # new a temp tensor
                    recv_tensor = torch.empty_like(vllm_tensor_view).contiguous()
                logger.debug(
                    f"Recving tensor {inst_dest_name} from policy rank {p_rank} to rollout rank {r_rank}, shape {vllm_tensor_view.shape} of {target_tensor.shape}."
                )
                nccl_recv(recv_tensor, p_rank, communicator_index)
                # inplace copy
                if not vllm_tensor_view.is_contiguous():
                    all_cloned_target_tensors.append((vllm_tensor_view, recv_tensor))

                total_bytes_received += recv_tensor.numel() * recv_tensor.element_size()

            if check_inside_group:
                tensors_to_check.append(
                    (cloned_target_tensor, target_tensor, insts, inst_dest_name)
                )

        def completion_lambda(all_cloned_target_tensors, tensors_to_check):
            for view, recv_tensor in all_cloned_target_tensors:
                view.copy_(
                    recv_tensor,
                )
            all_cloned_target_tensors.clear()

            for (
                cloned_target_tensor,
                target_tensor,
                insts,
                inst_dest_name,
            ) in tensors_to_check:
                if not torch.allclose(cloned_target_tensor, target_tensor):
                    raise ValueError(
                        f"Weight sync check failed after weight sync instruction: {insts} for {inst_dest_name}."
                    )
            tensors_to_check.clear()

        return total_bytes_received, partial(
            completion_lambda, all_cloned_target_tensors, tensors_to_check
        )

    @TRTLLMRolloutWorkerBase.register_rollout_command_handler(
        PolicyToRolloutUnicastCommand, backend="trtllm"
    )
    @torch.no_grad()
    def policy_to_rollout_unicast(self, command: PolicyToRolloutUnicastCommand):
        """
        Sync the weight from policy to rollout.
        This is Policy -> Rollout replica. Will only happen between
        a pair of policy and rollout replica.
        """
        if not self._engine_initialized:
            self.prepare_shard_infos_for_weight_sync_insts()
            self._engine_initialized = True

        if command.dst_replica_name != self.replica_name:
            return

        # get the nccl_unique_id from the controller
        communicator_index = {}
        nccl_unique_id_key = command.src_replica_name + "_" + command.dst_replica_name
        if nccl_unique_id_key in self.policy_to_rollout_nccl_communicators:
            logger.debug(
                f"[Rollout] Reusing cached communicator for {nccl_unique_id_key}"
            )
            communicator_index = self.policy_to_rollout_nccl_communicators[
                nccl_unique_id_key
            ]
        else:
            logger.debug(f"[Rollout] Querying nccl group id for {nccl_unique_id_key}")
            # query the nccl group id from controller
            nccl_group_id = self.api_client.post_nccl_comm_acceptor(nccl_unique_id_key)
            if nccl_group_id is None:
                raise RuntimeError(
                    "[Rollout] Failed to query nccl group_id from controller!"
                )
            # create the communicator index
            # p_rank is the rank in policy, r_rank is the rank in rollout
            communicator_index = create_nccl_comm(
                nccl_group_id,
                self.global_rank + command.src_replica_size,
                self.world_size + command.src_replica_size,
            )
            # cache the communicator index
            self.policy_to_rollout_nccl_communicators[nccl_unique_id_key] = (
                communicator_index
            )
        if not hasattr(self, "policy_to_rollout_recv_insts"):
            self.policy_to_rollout_recv_insts = (
                self.api_client.post_rollout_shard_recv_insts(self.global_rank)
            )
            logger.info(
                "[Rollout] Finished policy_to_rollout_recv_insts from controller."
            )
        total_recvs = 0
        total_params = 0
        for insts_group in self.policy_to_rollout_recv_insts:
            for insts_for_per_param in insts_group.param_instructions:
                total_params += 1
                total_recvs += len(insts_for_per_param.instructions)

        copy_stream = torch.cuda.Stream()

        with torch.cuda.stream(self.inference_stream):
            logger.debug(
                f"[Rollout] Starting to execute {len(self.policy_to_rollout_recv_insts)}; {total_params}, {total_recvs} weight sync receives ..."
            )
            # recv the weight from policy
            st = time.time()
            total_bytes_received = 0

            pending_bytes = [0]
            pending_completions = []
            pending_groups = 0

            def flush_completions(pending_bytes, pending_completions):
                recv_ready = torch.cuda.Event()
                recv_ready.record()
                with torch.cuda.stream(copy_stream):
                    recv_ready.wait()
                    logger.debug(
                        f"Flushing {len(pending_completions)} completions, {pending_bytes[0] // 1024 // 1024}"
                    )
                    for completion in pending_completions:
                        completion()
                    pending_bytes[0] = 0
                    pending_completions.clear()

            nccl_group_start(communicator_index)

            for insts_group in self.policy_to_rollout_recv_insts:
                # insts_group: WeightSyncInstructionsGroup -> inst collection for a full weight tensor
                # handle inst group
                bytes_received, completion_fn = self.recv_weight_shard(
                    self.global_rank,
                    insts_group,
                    communicator_index,
                    command.do_weight_sync_check,
                )
                pending_bytes[0] += bytes_received
                pending_completions.append(completion_fn)
                total_bytes_received += bytes_received

                pending_groups += 1
                if pending_groups == constant.COSMOS_P2R_NCCL_GROUP_SIZE:
                    nccl_group_end(communicator_index)
                    flush_completions(pending_bytes, pending_completions)
                    nccl_group_start(communicator_index)
                    pending_groups = 0

            nccl_group_end(communicator_index)
            flush_completions(pending_bytes, pending_completions)

            with torch.cuda.stream(copy_stream):
                copy_finished = torch.cuda.Event()
                copy_finished.record()

            copy_finished.wait()
            torch.cuda.synchronize()

            time_eclapsed = time.time() - st
            logger.info(
                f"[Rollout] All {len(self.policy_to_rollout_recv_insts)} at step {command.weight_step} recv operations finished in {time_eclapsed:.3f} seconds with {total_bytes_received / (1024 * 1024)} MB received."
            )
            if not self.cosmos_state.weight_synced():
                self.cosmos_state.set_weight_synced()
                self.cosmos_weight_sync_queue.put(RolloutWrapperInstruction())

    @TRTLLMRolloutWorkerBase.register_rollout_command_handler(
        RolloutToRolloutBroadcastCommand, backend="trtllm"
    )
    def broadcast_to_all_rollout_replica(
        self, broadcast_command: RolloutToRolloutBroadcastCommand
    ) -> None:
        """
        Broadcast the weight to all other rollout replicas.
        Will only happen between Rollout Replica 0 and all other Rollout Replicas.
        """
        src_replica_name: str = broadcast_command.src_replica_name
        dst_replica_names: List[str] = broadcast_command.dst_replica_names

        if not self._engine_initialized:
            self.prepare_shard_infos_for_weight_sync_insts()
            self._engine_initialized = True
            self.cosmos_weight_sync_queue.put(RolloutWrapperInstruction())

        if len(dst_replica_names) > 1:
            # Only do broadcast if there are more than one rollout replicas.
            with torch.cuda.stream(self.inference_stream):
                assert (
                    self.rank_in_rollout_repicas >= 0
                ), "[Rollout] rank in rollout replicas should be set before broadcast."
                assert (
                    len(dst_replica_names) == len(self.replica_name_to_rank)
                ), "[Rollout] The vaild dst replicas num should match the replicas num that this worker holds."

                src_rank = self.replica_name_to_rank[src_replica_name]

                for parameter in self.get_underlying_model().state_dict().values():
                    recv_tensor = parameter
                    if not parameter.is_contiguous():
                        recv_tensor = parameter.contiguous()

                    nccl_broadcast(recv_tensor, src_rank, self.global_commnicator_idex)

                    if not parameter.is_contiguous():
                        parameter.copy_(recv_tensor)

                if not self.cosmos_state.weight_synced():
                    self.cosmos_state.set_weight_synced()
                    self.cosmos_weight_sync_queue.put(RolloutWrapperInstruction())

        current_step = broadcast_command.weight_step
        if current_step is not None and current_step > 0:
            should_do_validation = self.config.validation.enable and (
                current_step % self.config.validation.freq == 0
                or current_step == broadcast_command.total_steps
            )
            if should_do_validation:
                self.cosmos_weight_sync_queue.put(
                    ValidationInstruction(current_step, broadcast_command.total_steps)
                )

        if broadcast_command.replica_should_stop():
            # trigger the shutdown signal to main process too.
            self.cosmos_weight_sync_queue.put(ShutdownInstruction())
            self.shutdown_signal.set()
            self.shutdown_mp_signal.set()

    def query_command_from_controller(self):
        """Background task to check commands from the controller"""
        while not self.shutdown_signal.is_set():
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
                logger.debug(f"[Rollout] Received command: {command.command_type}")
                self._command_queue.put(command)

    def consume_command(self, cmd_pred: Optional[Callable[[Command], bool]] = None):
        current_command = None
        if self.global_rank == 0:
            if not self._command_queue.empty():
                if cmd_pred is None:
                    current_command = self._command_queue.get()
                else:
                    if cmd_pred(self._command_queue.queue[0]):
                        current_command = self._command_queue.get()
                    else:
                        # Do not go on if the command is not expected
                        current_command = None

        current_command = dist_util.broadcast_object_cpu(current_command)

        if current_command is not None:
            handler = self.get_rollout_command_handler(
                type(current_command), backend=self.backend
            )
            if handler is None:
                raise Exception(
                    f"No such command supoorted in rollout {current_command}"
                )
            try:
                handler(self, current_command)
                logger.debug(
                    f"[Rollout] Command executed: {current_command._serialize()} for rank: {self.global_rank}"
                )
            except Exception as e:
                raise RuntimeError(
                    f"[Rollout] Command execution failed for {current_command._serialize()}"
                ) from e

    """
    Below are the methods that modified from PyExecutor.
    """

    def _executor_loop(self):
        cosmos_patched_executor_loop(self)

    def start_worker(self):
        # Start PyExecutor worker thread. The worker thread will call `_executor_loop`
        super().start_worker()

        # Start query command thread of Cosmos-RL.
        if self.global_rank == 0 and self.ready:
            # create a thread to query command as a producer
            self.background_thread = threading.Thread(
                target=self.query_command_from_controller, daemon=True
            )
            self.background_thread.start()

    def handle_shutdown(self):
        if not hasattr(self, "_shutdown_handled"):
            self._shutdown_handled = True
            if hasattr(self, "shutdown_signal") and not self.shutdown_signal.is_set():
                self.shutdown_signal.set()
            if (
                hasattr(self, "shutdown_mp_signal")
                and not self.shutdown_mp_signal.is_set()
            ):
                self.shutdown_mp_signal.set()

        if hasattr(self, "background_thread") and self.background_thread is not None:
            self.background_thread.join()
            self.background_thread = None

        if hasattr(self, "heartbeat_thread") and self.heartbeat_thread is not None:
            self.heartbeat_thread.join()
            self.heartbeat_thread = None
        # let wrapper do this instead.
        # self.unregister_from_controller()

    def shutdown(self):
        # override pyexecutor's shutdown
        super().shutdown()
        self.handle_shutdown()
