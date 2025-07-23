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
import requests
import msgpack
from functools import partial
from typing import List, Any, Optional, Callable

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.rollout import RolloutWorkerBase
from cosmos_rl.rollout import State
from cosmos_rl.utils.parallelism_map import (
    ParallelTopoMapperGroup,
    WeightSyncInstructionsGroup,
)
from cosmos_rl.utils.util import list_to_b64, b64_to_list
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
)
from cosmos_rl.utils.api_suffix import (
    COSMOS_API_NCCL_COMM_INITIATOR_SUFFIX,
    COSMOS_API_NCCL_COMM_ACCEPTOR_SUFFIX,
    COSMOS_API_ROLLOUT_SHARD_INFOS_SUFFIX,
    COSMOS_API_ROLLOUT_SHARD_RECV_INSTS_SUFFIX,
)

from cosmos_rl.policy.model import ModelRegistry, WeightMapper
from cosmos_rl.utils import constant
from cosmos_rl.utils.logging import logger
import cosmos_rl.utils.distributed as dist_util
from cosmos_rl.utils.network_util import make_request_with_retry
import cosmos_rl.utils.util as util

from transformers import AutoConfig

from queue import Queue

"""
1. Extend PyExecutor to support Cosmos-specific features.
2. Initialize distributed environment.
"""


class CosmosTRTLLMWorker(PyExecutor, RolloutWorkerBase):
    """
    CosmosTRTLLMExecutor is a wrapper of PyExecutor to support Cosmos-specific features.
    P2R and R2R of cosmos-rl are implemented in this class.
    """

    def __init__(self, *args, **kwargs) -> None:
        # just call the init of PyExecutor
        super().__init__(*args, **kwargs)

    def set_cosmos_config(self, config: CosmosConfig):
        parallel_dims = ParallelDims.from_config(
            parallesim_config=config.rollout.parallelism
        )
        self.parallel_dims = parallel_dims

        # build the mesh
        self.parallel_dims.build_mesh(device_type="cuda")

        # init the RolloutWorkerBase
        RolloutWorkerBase.__init__(self, config, parallel_dims)
        self.post_init()

        self.cosmos_state = State()

        # CommandQueue queried from controller.
        self.cosmos_command_queue: Queue[Command] = Queue()
        self.cosmos_prompt_queue: Queue[List[List[int, str]]] = Queue()

        self.cosmos_global_commnicator_idex = -1
        self.cosmos_rank_in_rollout_repicas = -1

        self.cosmos_policy_to_rollout_nccl_communicators = {}

        self.cosmos_batch_size = self.config.rollout.batch_size

        # For Polocy to Rollout weight mapping
        hf_config = util.retry(AutoConfig.from_pretrained)(
            self.config.policy.model_name_or_path,
            trust_remote_code=True,
        )
        model_type = hf_config.model_type
        if not ModelRegistry.check_model_type_supported(model_type):
            logger.warning(
                f"[Rollout] Replica can not find {model_type} in weight mapper, use {constant.COSMOS_HF_MODEL_TYPES} model type instead, with replica name: {self.replica_name}"
            )
            model_type = constant.COSMOS_HF_MODEL_TYPES
        self.cosmos_weight_mapper = WeightMapper.get_weight_mapper(model_type)(
            hf_config
        )
        self.cosmos_model_config = hf_config

        self.inference_stream = torch.cuda.Stream()

        self.prepare_shard_infos_for_weight_sync_insts()

    def prepare_shard_infos_for_weight_sync_insts(self):
        self.vllm_weight_inplace_view_map, grouped_recv_param_key_n_rank_list = (
            self.weight_mapper.rollout_prepare_recv(self.get_underlying_model())
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
            self.model_config,
            is_policy=False,
            underlying_model=self.get_underlying_model(),
            weight_mapper=self.weight_mapper,
        )(self.recv_param_key_n_rank_list, self.global_rank)

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
            body = {
                "shard_infos": self.all_rank_local_shard_infos,
                "param_groups": list(merged_groups.values()),
                "sorted_params": sorted_params_all_rank,
            }
            data = msgpack.packb(body)
            try:
                make_request_with_retry(
                    partial(
                        requests.post,
                        data=data,
                        headers={"Content-Type": "application/msgpack"},
                    ),
                    self.get_alternative_urls(COSMOS_API_ROLLOUT_SHARD_INFOS_SUFFIX),
                    max_retries=constant.COSMOS_HTTP_RETRY_CONFIG.max_retries,
                )
            except Exception as e:
                raise RuntimeError(
                    f"[Rollout] Failed in post shard infos to controller after retries {e}."
                )

    def get_underlying_model(self):
        """
        Get the underlying parallelized model in vLLM internal.
        """
        return self.model_engine.model.model

    @RolloutWorkerBase.register_rollout_command_handler(BuildMeshCommand)
    def build_global_mesh(self, build_mesh_command: BuildMeshCommand):
        logger.info(f"[Rollout] Building global mesh for {self.replica_name}")

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
        nccl_group_id = None
        if self.rank_in_rollout_repicas == 0:
            # only replica_rank == 0 have the right to generate nccl id.
            nccl_group_id = create_nccl_uid()
            base64_nccl_group_id = list_to_b64(nccl_group_id)
            try:
                make_request_with_retry(
                    partial(
                        requests.post,
                        json={
                            "unique_pair_name": unique_rollout_group_key,
                            "handle_base64": base64_nccl_group_id,
                        },
                    ),
                    self.get_alternative_urls(COSMOS_API_NCCL_COMM_INITIATOR_SUFFIX),
                    max_retries=constant.COSMOS_HTTP_RETRY_CONFIG.max_retries,
                )
            except Exception as e:
                raise RuntimeError(
                    f"[Rollout] Failed in post nccl group_id to controller after retries {e}."
                )

        if self.rank_in_rollout_repicas != 0:
            # other replicas should query the nccl group id from controller
            # all ranks need to wait for the rollout replica 0 finished the group_id post
            # and then they can get the group_id from controller
            # all ranks not zero in replica 0 or all ranks of other replicas need to query the group_id from controller
            nccl_group_id = self.query_nccl_unique_id_from_controller(
                unique_rollout_group_key
            )
            if nccl_group_id is None:
                raise RuntimeError(
                    "[Rollout] Failed to query nccl group_id from controller!"
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

    def query_nccl_unique_id_from_controller(self, unique_id_key: str):
        # We don't have something like dist.barrier(), so just use while True loop to query it like synchronize.
        nccl_group_id = None
        # all ranks not zero in replica 0 or all ranks of other replicas need to query the group_id from controller
        try:
            r = make_request_with_retry(
                partial(
                    requests.post,
                    json={"unique_pair_name": unique_id_key},
                ),
                self.get_alternative_urls(COSMOS_API_NCCL_COMM_ACCEPTOR_SUFFIX),
                max_retries=constant.COSMOS_HTTP_LONG_WAIT_MAX_RETRY,
            )
        except Exception as e:
            raise RuntimeError(
                f"[Rollout] Failed in post nccl group_id to controller after retries {e}."
            )
        base64_nccl_group_id = r.json()["handle_base64"]
        nccl_group_id = b64_to_list(base64_nccl_group_id)
        return nccl_group_id

    def recv_weight_shard(
        self,
        global_rank_of_rollout: int,
        insts_group: WeightSyncInstructionsGroup,
        communicator_index: int,
        do_weight_sync_check: bool = False,
    ):
        check_inside_group = do_weight_sync_check
        if self.quantization_type == "fp8":
            inst_group_weight_name = (
                insts_group.param_instructions[0].param_name
            )  # take a name from the inst group to determine the full weight name
            # the full weight name that this inst group handles.
            inst_group_full_weight_name = self.weight_mapper.get_unsplited_weight_name(
                inst_group_weight_name
            )
            is_fp8_quantized_module = (
                inst_group_full_weight_name in self.vllm_quantized_weight_map
            )
            check_inside_group = do_weight_sync_check and (not is_fp8_quantized_module)

        total_bytes_received = 0

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

                nccl_recv(recv_tensor, p_rank, communicator_index)
                # inplace copy
                if not vllm_tensor_view.is_contiguous():
                    vllm_tensor_view.copy_(recv_tensor)

                total_bytes_received += recv_tensor.numel() * recv_tensor.element_size()

            if check_inside_group:
                if not torch.allclose(cloned_target_tensor, target_tensor):
                    raise ValueError(
                        f"Weight sync check failed after weight sync instruction: {insts} for {inst_dest_name}."
                    )

        # here we got one full weight tensor sync done, if it is fp8 weight, we should do the quantization and check the numerical error.
        if self.quantization_type == "fp8":
            if inst_group_full_weight_name in self.vllm_hp_weight_map:
                weight_to_quantize = self.vllm_hp_weight_map[
                    inst_group_full_weight_name
                ]  # [out_dim, in_dim]
                quantized_weight, weight_scale = self.rollout.fp8_quantization(
                    weight_to_quantize
                )
                model_param_map = self.rollout.model_param_map(self.weight_mapper)
                vllm_native_weight = model_param_map[inst_group_full_weight_name]

                # check weight sync
                if do_weight_sync_check:
                    # allclose doesn't support fp8, promote it.
                    bf16_vllm_native_weight = vllm_native_weight.to(torch.bfloat16)
                    bf16_quantized_weight = quantized_weight.to(torch.bfloat16)
                    if not torch.allclose(
                        bf16_vllm_native_weight, bf16_quantized_weight
                    ):
                        raise ValueError(
                            f"FP8 weight doesn't match after weight sync and dynamic quantization for full weight name: {inst_group_full_weight_name}."
                        )
                vllm_native_weight.copy_(quantized_weight)
                # get the scale key.
                scale_key = inst_group_full_weight_name.replace(
                    ".weight", ".weight_scale"
                )
                scale_tensor = model_param_map[scale_key]
                assert (
                    scale_tensor.shape == weight_scale.shape
                ), f"scale_tensor.shape: {scale_tensor.shape}, weight_scale.shape: {weight_scale.shape}"
                scale_tensor.copy_(weight_scale)
        else:
            # For non-fp8 weights and fp8 not enabled cases, we just do nothing
            pass

        return total_bytes_received

    @RolloutWorkerBase.register_rollout_command_handler(PolicyToRolloutUnicastCommand)
    @torch.no_grad()
    def policy_to_rollout_unicast(self, command: PolicyToRolloutUnicastCommand):
        """
        Sync the weight from policy to rollout.
        This is Policy -> Rollout replica. Will only happen between
        a pair of policy and rollout replica.
        """
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
            nccl_group_id = self.query_nccl_unique_id_from_controller(
                nccl_unique_id_key
            )
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
            self.policy_to_rollout_recv_insts = []
            try:
                insts_meta = make_request_with_retry(
                    partial(
                        requests.post,
                        json={
                            "rank": self.global_rank,
                        },
                    ),
                    self.get_alternative_urls(
                        COSMOS_API_ROLLOUT_SHARD_RECV_INSTS_SUFFIX
                    ),
                    max_retries=constant.COSMOS_HTTP_RETRY_CONFIG.max_retries,
                )
                insts = msgpack.unpackb(insts_meta.content, strict_map_key=False)
                self.policy_to_rollout_recv_insts = [
                    WeightSyncInstructionsGroup.from_dict(inst) for inst in insts
                ]
            except Exception as e:
                raise RuntimeError(
                    f"[Rollout] Failed in fetching rollout from policy insts from controller after retries {e}."
                )

        with torch.cuda.stream(self.inference_stream):
            # recv the weight from policy
            st = time.time()
            total_bytes_received = 0
            for insts_group in self.policy_to_rollout_recv_insts:
                # insts_group: WeightSyncInstructionsGroup -> inst collection for a full weight tensor
                # handle inst group
                total_bytes_received += self.recv_weight_shard(
                    self.global_rank,
                    insts_group,
                    communicator_index,
                    command.do_weight_sync_check,
                )
            time_eclapsed = time.time() - st
            logger.debug(
                f"[Rollout] All {len(self.policy_to_rollout_recv_insts)} at step {command.weight_step} recv operations finished in {time_eclapsed:.3f} seconds with {total_bytes_received / (1024 * 1024)} MB received."
            )
            self.state.set_weight_synced()

    @RolloutWorkerBase.register_rollout_command_handler(
        RolloutToRolloutBroadcastCommand
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

                if not self.state.weight_synced():
                    self.state.set_weight_synced()

        current_step = broadcast_command.weight_step
        if current_step is not None and current_step > 0:
            should_do_validation = self.config.train.enable_validation and (
                current_step % self.config.train.validation_step == 0
                or current_step == broadcast_command.total_steps
            )
            validation_queue = Queue()
            validation_results = []
            prompt_idxs: List[int] = []
            payloads: List[Any] = []
            if should_do_validation:
                # Do inline validation here
                while True:
                    is_end = self.request_new_prompts(
                        self.val_batch_size,
                        validation_queue,
                        validation_step=current_step,
                    )
                    if not validation_queue.empty():
                        prompts = validation_queue.get()
                        completions: List[List[str]] = self.rollout.rollout_generation(
                            prompt_id_and_payload_list=prompts,
                            stream=self.inference_stream,
                            data_packer=self.val_data_packer,
                            sampling_params=self.val_sampling_params,
                        )
                        if completions:
                            prompt_idxs.extend([prompt[0] for prompt in prompts])
                            payloads.extend([prompt[1] for prompt in prompts])
                            validation_results.extend(completions)

                    if is_end:
                        break

        if broadcast_command.replica_should_stop():
            self.shutdown_signal.set()

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
            handler = self.get_rollout_command_handler(type(current_command))
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
