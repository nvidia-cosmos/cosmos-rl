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
import torch
import threading
import asyncio
from queue import Queue
import atexit
import types
from functools import partial
from asyncio import timeout as asyncio_timeout

from cosmos_rl.policy.model import ModelRegistry, WeightMapper
from typing import List, Optional, Callable, Union
from transformers import AutoConfig
from cosmos_rl.rollout import RolloutWorkerBase, State
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.utils.parallelism_map import (
    WeightSyncInstructionsGroup,
    ParallelTopoMapperGroup,
)
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.logging import logger
from cosmos_rl.rollout.vllm_rollout.vllm_rollout_async import vLLMRolloutAsync
from cosmos_rl.rollout.rollout_task_scheduler import (
    RolloutTaskScheduler,
    RolloutTask,
    CompletedRollout,
)
from cosmos_rl.dispatcher.protocol import ValidationReportRequest
from cosmos_rl.dispatcher.command import (
    Command,
    BuildMeshCommand,
    PolicyToRolloutUnicastCommand,
    RolloutToRolloutBroadcastCommand,
)
import cosmos_rl.utils.util as util
import cosmos_rl.utils.pynccl as pynccl
from cosmos_rl.utils import constant
import cosmos_rl.utils.distributed as dist_utils
from cosmos_rl.dispatcher.data.schema import (
    RLPayload,
    ConversationType,
)
from torch.utils.data import Dataset
from cosmos_rl.reward.reward_calculator import RewardDispatcher
from cosmos_rl.utils.command_executor import CommandExecutor
from vllm.sampling_params import SamplingParams, RequestOutputKind

from .vllm_rollout_worker import vLLMRolloutWorker


"""
Async version of vLLMRolloutWorker using RolloutTaskScheduler.
Key differences from sync version:
- Uses RolloutTaskScheduler for async generation management
- Pauses scheduler during weight synchronization
- Uses vLLMRolloutAsync instead of vLLMRollout
"""


def filter_valid_single_turn_rollout_results(
    rollout_results: list[CompletedRollout], eos_token: str
) -> list[CompletedRollout]:
    valid_results: list[CompletedRollout] = []

    # Remove empty completions
    for cr in rollout_results:
        completions = cr.result.completions
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
            output_texts.append(output_text if output_text != "" else eos_token)
        # Skip the output if there is one or zero non-empty completions
        skip_output = (total_generation_count - empty_generation_count) <= 1
        if not skip_output:
            cr.result.completions = output_texts
            valid_results.append(cr)

        return valid_results


def filter_valid_multi_turn_rollout_results(
    rollout_results: list[CompletedRollout],
) -> list[CompletedRollout]:
    valid_results: list[CompletedRollout] = []
    for cr in rollout_results:
        valid_conversations: List[ConversationType] = []
        # remove those result without valid assistant message
        flag = False
        for conversation in cr.result.completed_conversations:
            for msg in conversation:
                if msg.role == "assistant" and msg.content != "":
                    flag = True
                    break
            if flag:
                valid_conversations.append(conversation)
        cr.result.completed_conversations = valid_conversations
        if len(cr.result.completed_conversations) > 0:
            valid_results.append(cr)

    return valid_results


class vLLMRolloutWorkerAsync(RolloutWorkerBase):
    """
    Async version of vLLMRolloutWorker with RolloutTaskScheduler.

    This worker uses coroutine-based execution:
    - Main loop runs in asyncio event loop
    - Uses RolloutTaskScheduler.run_async() for coroutine-based scheduling
    - RolloutTaskScheduler runs in the same event loop, not a separate thread
    - Pauses the scheduler during weight synchronization to avoid conflicts
    - Provides better throughput and resource utilization with async/await
    - Sync operations (API calls, reward calculation) run in thread executor
    """

    def __init__(self, config: CosmosConfig, parallel_dims: ParallelDims) -> None:
        super(vLLMRolloutWorkerAsync, self).__init__(config, parallel_dims)

        # TODO(zjx): refactor those methods to common methods in RolloutWorkerBase
        # reuse some of the vLLMRolloutWorker methods
        self.prepare_trainable_params = types.MethodType(
            vLLMRolloutWorker.prepare_trainable_params, self
        )

        self.report_rollouts = types.MethodType(vLLMRolloutWorker.report_rollouts, self)
        self.query_command_from_controller = types.MethodType(
            vLLMRolloutWorker.query_command_from_controller, self
        )
        self.query_nccl_unique_id_from_controller = types.MethodType(
            vLLMRolloutWorker.query_nccl_unique_id_from_controller, self
        )
        self.send_end_signal = types.MethodType(vLLMRolloutWorker.send_end_signal, self)

        # real init variables
        self.state = State()

        if self.config.rollout.parallelism.dp_shard_size == -1:
            self.config.rollout.parallelism.dp_shard_size = parallel_dims.dp_shard
        assert self.config.rollout.parallelism.dp_shard_size == parallel_dims.dp_shard
        assert (
            self.config.rollout.parallelism.dp_shard_size > 0
        ), "[Rollout] dp_shard_size should be greater than 0."

        # CommandQueue queried from controller.
        self._command_queue: Queue[Command] = Queue()
        self.current_weight_version = 0

        # determine the quantization type
        self.quantization_type = None
        if self.config.rollout.quantization != "none":
            self.quantization_type = self.config.rollout.quantization

        # Use async rollout engine
        self.rollout: vLLMRolloutAsync = vLLMRolloutAsync(self.config, self.tokenizer)

        # communicator index for the cached communicators in C++ binding.
        self.global_commnicator_idex = -1
        # rank in current rollout replicas.
        self.rank_in_rollout_repicas = -1

        # cache for NCCL communicators for P2R.
        self.policy_to_rollout_nccl_communicators = {}

        self.batch_size = self.config.rollout.batch_size
        if self.config.validation.enable:
            self.val_batch_size = self.config.validation.batch_size or self.batch_size
            assert (
                self.val_batch_size > 0
            ), "[Rollout] val_batch_size should be greater than 0."
        else:
            self.val_batch_size = None
        self.background_thread: threading.Thread | None = None

        # For Polocy to Rollout weight mapping
        hf_config = util.retry(AutoConfig.from_pretrained)(
            self.config.policy.model_name_or_path,
            trust_remote_code=True,
        )
        model_type = hf_config.model_type
        if self.quantization_type == "mxfp4":
            assert (
                model_type == "gpt_oss"
            ), "[Rollout] Mxfp4 quantization is only supported for GPT-OSS now."

        if not ModelRegistry.check_model_type_supported(model_type):
            logger.warning(
                f"[Rollout] Replica can not find {model_type} in weight mapper, use {constant.COSMOS_HF_MODEL_TYPES} model type instead, with replica name: {self.replica_name}"
            )
            model_type = constant.COSMOS_HF_MODEL_TYPES
        self.weight_mapper = WeightMapper.get_weight_mapper(model_type)(hf_config)
        self.model_config = hf_config

        atexit.register(self.handle_shutdown)

        self.inference_stream = torch.cuda.Stream()

        self.val_sampling_params = SamplingParams(
            n=self.config.validation.n_generation,
            logprobs=0,
            top_p=self.config.validation.top_p
            if self.config.validation.top_p is not None
            else self.config.rollout.sampling_config.top_p,
            top_k=self.config.validation.top_k
            if self.config.validation.top_k is not None
            else self.config.rollout.sampling_config.top_k,
            temperature=self.config.validation.temperature
            if self.config.validation.temperature is not None
            else self.config.rollout.sampling_config.temperature,
            repetition_penalty=self.config.validation.repetition_penalty
            if self.config.validation.repetition_penalty is not None
            else self.config.rollout.sampling_config.repetition_penalty,
            max_tokens=self.config.validation.max_response_length
            if self.config.validation.max_response_length is not None
            else self.config.rollout.max_response_length,
            stop_token_ids=self.rollout.eos_token_ids,
            include_stop_str_in_output=self.config.rollout.include_stop_str_in_output,
            detokenize=True,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        self.sampling_params = SamplingParams(
            n=self.config.rollout.n_generation,
            logprobs=0,
            top_p=self.config.rollout.sampling_config.top_p,
            top_k=self.config.rollout.sampling_config.top_k,
            temperature=self.config.rollout.sampling_config.temperature,
            repetition_penalty=self.config.rollout.sampling_config.repetition_penalty,
            max_tokens=self.config.rollout.max_response_length,
            stop_token_ids=self.rollout.eos_token_ids,
            include_stop_str_in_output=self.config.rollout.include_stop_str_in_output,
            detokenize=True,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )

        # Holding temp tensors created in `recv_tensor_creator`. Do not remove this, or
        self.misc_params = set()
        self.validation_flag = threading.Event()
        self.reward_dispatcher = RewardDispatcher()

        # Initialize RolloutTaskScheduler
        self.scheduler: Optional[RolloutTaskScheduler] = None
        self._scheduler_thread: Optional[threading.Thread] = None

        # setup the command executor
        self.command_executor = CommandExecutor()

        # TODO(zjx): below variables need remove after refactor.
        self.temp_recv_tensor_queue = Queue()

    def setup(
        self,
        dataset: Optional[Union[Dataset, Callable[[CosmosConfig], Dataset]]] = None,
        reward_fns: Optional[List[Callable]] = None,
        filter_reward_fns: Optional[List[Callable]] = None,
        val_dataset: Optional[Dataset] = None,
        val_reward_fns: Optional[List[Callable]] = None,
        num_workers: int = 8,
    ):
        self.reward_dispatcher.setup(
            config=self.config,
            dataset=dataset,
            reward_fns=reward_fns,
            filter_reward_fns=filter_reward_fns,
            val_dataset=val_dataset,
            val_reward_fns=val_reward_fns,
            data_packer=self.data_packer,
            val_data_packer=self.val_data_packer,
            num_workers=num_workers
            if self.parallel_dims.tp_coord[0] == 0
            and (self.parallel_dims.pp_coord[0] == self.parallel_dims.pp_coord[1] - 1)
            else 0,
        )

        # setup the command executor
        self.command_executor.register_command_handler(
            RolloutToRolloutBroadcastCommand, self.broadcast_to_all_rollout_replica
        )
        self.command_executor.register_command_handler(
            PolicyToRolloutUnicastCommand, self.policy_to_rollout_unicast
        )
        self.command_executor.register_command_handler(
            BuildMeshCommand,
            types.MethodType(vLLMRolloutWorker.build_global_mesh, self),
        )

    def init_scheduler(self):
        """Initialize the RolloutTaskScheduler (async version)."""
        if self.scheduler is not None:
            logger.warning("[RolloutWorkerAsync] Scheduler already initialized")
            return

        logger.info("[RolloutWorkerAsync] Initializing RolloutTaskScheduler")

        # Get max concurrent requests from config or use default
        max_concurrent_requests = (
            self.config.rollout.async_config.max_concurrent_requests
        )

        self.scheduler = RolloutTaskScheduler(
            rollout_engine=self.rollout,
            data_packer=self.data_packer,
            max_concurrent_requests=max_concurrent_requests,
            stream=self.inference_stream,
            check_interval=0.1,
        )

        logger.info(
            "[RolloutWorkerAsync] RolloutTaskScheduler initialized (will run in async mode)"
        )

    def handle_shutdown(self):
        """Handle shutdown and cleanup."""
        # Only call once
        if not hasattr(self, "_shutdown_handled"):
            self._shutdown_handled = True

            # Scheduler is stopped in main_loop_async's finally block
            logger.info("[RolloutWorkerAsync] Handling shutdown")

            self.rollout.shutdown()

            if not self.shutdown_signal.is_set():
                logger.info(
                    f"[Rollout] shutdown instruction of {self.replica_name}, setting shutdown signal"
                )
                self.shutdown_signal.set()
            if not self.shutdown_mp_signal.is_set():
                self.shutdown_mp_signal.set()
            if self.background_thread is not None:
                self.background_thread.join()
                self.background_thread = None

            if self.heartbeat_thread is not None:
                self.heartbeat_thread.join()
                self.heartbeat_thread = None
            self.unregister_from_controller()

    async def get_underlying_model(self):
        """
        Get the underlying parallelized model in vLLM internal.
        """
        return await self.rollout.get_underlying_model()

    async def prepare_shard_infos_for_weight_sync_insts(self):
        if self.quantization_type == "fp8":
            from cosmos_rl.rollout.vllm_rollout.monkey_patch_for_fp8 import (
                cache_weight_of_quantized_module,
                replace_weight_of_quantized_module,
            )
            from cosmos_rl.rollout.vllm_rollout.monkey_patch_for_fp8 import (
                post_process_view_map_for_fp8 as post_process_view_map_for_lowp,
            )
        elif self.quantization_type == "mxfp4":
            from cosmos_rl.rollout.vllm_rollout.monkey_patch_for_mxfp4 import (
                cache_weight_of_quantized_module,
                replace_weight_of_quantized_module,
            )

            from cosmos_rl.rollout.vllm_rollout.monkey_patch_for_mxfp4 import (
                post_process_view_map_for_mxfp4 as post_process_view_map_for_lowp,
            )

        if self.quantization_type is not None:
            promotion_dtype = util.str2torch_dtype(self.config.train.param_dtype)
            self.vllm_hp_weight_map, self.vllm_quantized_weight_map = (
                cache_weight_of_quantized_module(
                    await self.get_underlying_model(),
                    promotion_dtype,
                    self.weight_mapper,
                    self.parallel_dims,
                )
            )
            # replace the weight of quantized module with the high precision weight.
            # let weight in vllm_weight_inplace_view_map always in high precision for recv
            # high precision weight from policy.
            replace_weight_of_quantized_module(
                await self.get_underlying_model(),
                self.vllm_hp_weight_map,
                self.weight_mapper,
            )

        self.vllm_weight_inplace_view_map, grouped_recv_param_key_n_rank_list = (
            self.weight_mapper.cosmos_rollout_prepare_recv(
                await self.get_underlying_model()
            )
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
            underlying_model=await self.get_underlying_model(),
            weight_mapper=self.weight_mapper,
        ).prepare_local_shard_infos(self.recv_param_key_n_rank_list, self.global_rank)

        # this must be done after prepare_local_shard_infos
        if self.quantization_type is not None:
            self.vllm_weight_inplace_view_map = post_process_view_map_for_lowp(
                self.vllm_weight_inplace_view_map
            )
            # Get vllm weight back into quantized.
            replace_weight_of_quantized_module(
                await self.get_underlying_model(),
                self.vllm_quantized_weight_map,
                self.weight_mapper,
            )

        self.all_rank_local_shard_infos = dist_utils.all_gather_object_cpu(
            local_shard_infos
        )
        all_param_groups = dist_utils.all_gather_object_cpu(param_groups)
        merged_groups = {}
        for r, param_groups in enumerate(all_param_groups):
            if self.parallel_dims.get_rank_in_dim("dp_cp_tp", r) != 0:
                continue
            for group in param_groups:
                group = sorted(group)
                key = tuple(group)
                if key not in merged_groups:
                    merged_groups[key] = group
        sorted_params_all_rank = dist_utils.all_gather_object_cpu(
            [x[0] for x in self.recv_param_key_n_rank_list]
        )
        sorted_params_all_rank = [
            x
            for r, x in enumerate(sorted_params_all_rank)
            if self.parallel_dims.get_rank_in_dim("dp_cp_tp", r) == 0
        ]
        if self.global_rank == 0:
            self.api_client.post_rollout_shard_info(
                shard_infos=self.all_rank_local_shard_infos,
                param_groups=list(merged_groups.values()),
                sorted_params=sorted_params_all_rank,
            )

    async def recv_weight_shard(
        self,
        global_rank_of_rollout: int,
        insts_group: WeightSyncInstructionsGroup,
        communicator_index: int,
        trainable_only: bool,
        do_weight_sync_check: bool = False,
    ):
        target_dtype = util.str2torch_dtype(self.config.train.transfer_dtype)
        check_inside_group = do_weight_sync_check
        if self.quantization_type is not None:
            inst_group_weight_name = (
                insts_group.param_instructions[0].param_name
            )  # take a name from the inst group to determine the full weight name
            # the full weight name that this inst group handles.
            inst_group_full_weight_name = self.weight_mapper.get_unsplited_weight_name(
                inst_group_weight_name
            )
            is_lowp_quantized_module = (
                inst_group_full_weight_name in self.vllm_quantized_weight_map
            )
            check_inside_group = do_weight_sync_check and (not is_lowp_quantized_module)

        total_bytes_received = 0

        all_tensor_views_to_copy = []
        tensors_to_check = []

        def recv_tensor_creator(vllm_tensor_view: torch.Tensor):
            recv_tensor = None
            inplace = True

            # clean up completed temp recv tensor in queue if the recv tensor queue is not empty.
            while (
                not self.temp_recv_tensor_queue.empty()
                and self.temp_recv_tensor_queue.queue[0][1].query()
            ):
                # pop the completed recv tensor if its event is finished.
                self.temp_recv_tensor_queue.get()

            # In case cpu part keeps inserting too many temp tensors without sync.
            # We synchronize and clear the queue to prevent memory issues.
            if (
                not vllm_tensor_view.is_contiguous()
                or vllm_tensor_view.dtype != target_dtype
            ):
                if (
                    self.temp_recv_tensor_queue.qsize()
                    >= constant.COSMOS_RECV_TENSOR_QUEUE_SIZE
                ):
                    num_to_clear = (
                        self.temp_recv_tensor_queue.qsize()
                        - constant.COSMOS_RECV_TENSOR_QUEUE_SIZE
                        + 1
                    )
                    for _ in range(num_to_clear):
                        _, event = self.temp_recv_tensor_queue.get()
                    event.synchronize()

            if vllm_tensor_view.is_contiguous():
                recv_tensor = vllm_tensor_view
            else:
                # new a temp tensor
                recv_tensor = torch.empty_like(vllm_tensor_view).contiguous()
                inplace = False

            if vllm_tensor_view.dtype != target_dtype:
                recv_tensor = recv_tensor.to(target_dtype)
                inplace = False
            # Event for recv related operations completion tracking
            # Hold these recv_tensor, in case of buffer reusing by torch
            if not inplace:
                recv_complete_event = torch.cuda.Event()
                self.temp_recv_tensor_queue.put((recv_tensor, recv_complete_event))
            else:
                recv_complete_event = None
            return recv_tensor, recv_complete_event, inplace

        skipped_params_cnt = 0

        for insts_for_per_param in insts_group.param_instructions:
            # insts_for_per_param: WeightSyncInstructionsPerParam -> inst collection for a single tensor
            insts = insts_for_per_param.instructions
            # insts: List[Tuple[int, int, Dict[int, Any]]]
            inst_dest_name = insts_for_per_param.param_name

            if inst_dest_name not in self.trainable_params and trainable_only:
                logger.info(
                    f"[Rollout] Skip {inst_dest_name} in P2R recv due to non trainable."
                )
                skipped_params_cnt += 1
                continue

            target_tensor = self.vllm_weight_inplace_view_map[inst_dest_name]

            if check_inside_group:
                cloned_target_tensor = target_tensor.clone().cpu()
                # clear the current view
                target_tensor.zero_()

            for inst in insts:
                # Inst for different part of a tensor between policy and rollout.
                p_rank = inst.policy_rank
                r_rank = inst.rollout_rank
                tensor_split_strategys = inst.slice_strategy
                assert r_rank == global_rank_of_rollout
                vllm_tensor_view = target_tensor.cosmos_slice(tensor_split_strategys)
                recv_tensor, recv_complete_event, inplace = recv_tensor_creator(
                    vllm_tensor_view
                )
                logger.debug(
                    f"[Rollout] Recving tensor {inst_dest_name} from policy rank {p_rank} to rollout rank {r_rank}, shape {vllm_tensor_view.shape} of {target_tensor.shape} with dtype {vllm_tensor_view.dtype}."
                )
                pynccl.nccl_recv(recv_tensor, p_rank, communicator_index)

                # inplace copy
                if not inplace:
                    all_tensor_views_to_copy.append(
                        (
                            vllm_tensor_view,
                            recv_tensor,
                            recv_complete_event,
                            inst_dest_name,
                        )
                    )

                total_bytes_received += recv_tensor.numel() * recv_tensor.element_size()

            if check_inside_group:
                tensors_to_check.append(
                    (cloned_target_tensor, target_tensor, insts, inst_dest_name)
                )

        post_process_list_for_lowp = []

        if not check_inside_group and self.quantization_type is not None:
            post_process_list_for_lowp.append(inst_group_full_weight_name)

        async def completion_lambda(
            all_tensor_views_to_copy, tensors_to_check, post_process_list_for_lowp
        ):
            for (
                view,
                recv_tensor,
                recv_complete_event,
                inst_dest_name,
            ) in all_tensor_views_to_copy:
                self.weight_mapper.update_tensor_view(
                    view, recv_tensor, inst_dest_name, parallel_dims=self.parallel_dims
                )
                if recv_complete_event is not None:
                    recv_complete_event.record()
            for (
                cloned_target_tensor,
                target_tensor,
                insts,
                inst_dest_name,
            ) in tensors_to_check:
                cloned_target_tensor = cloned_target_tensor.to(target_dtype).to(
                    cloned_target_tensor.dtype
                )
                if not torch.allclose(cloned_target_tensor, target_tensor.cpu()):
                    raise ValueError(
                        f"Weight sync check failed after weight sync instruction: {insts} for {inst_dest_name}."
                    )
            tensors_to_check.clear()

            # here we got one full weight tensor sync done, if it is fp8/mxfp4 weight, we should do the quantization and check the numerical error.
            if self.quantization_type is not None:
                for inst_group_full_weight_name in post_process_list_for_lowp:
                    if self.quantization_type == "fp8":
                        if inst_group_full_weight_name in self.vllm_hp_weight_map:
                            weight_to_quantize = self.vllm_hp_weight_map[
                                inst_group_full_weight_name
                            ]  # [out_dim, in_dim]
                            quantized_weight, weight_scale = (
                                self.rollout.fp8_quantization(weight_to_quantize)
                            )
                            model_param_map = await self.rollout.model_param_map(
                                self.weight_mapper
                            )
                            vllm_native_weight = model_param_map[
                                inst_group_full_weight_name
                            ]

                            # check weight sync
                            if do_weight_sync_check:
                                # allclose doesn't support fp8, promote it.
                                bf16_vllm_native_weight = vllm_native_weight.to(
                                    torch.bfloat16
                                )
                                bf16_quantized_weight = quantized_weight.to(
                                    torch.bfloat16
                                )
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
                    elif self.quantization_type == "mxfp4":
                        # Note: For mxfp4, we don't do weight sync check for quantized weights.
                        if inst_group_full_weight_name in self.vllm_hp_weight_map:
                            if "gate_up_proj_bias" not in inst_group_full_weight_name:
                                # Weight to quantize:
                                # [local_num_experts, 2* local_intermediate_size, hidden_size] for gate_up_proj
                                # [local_num_experts, hidden_size, local_intermediate_size] for down_proj
                                weight_to_quantize = self.vllm_hp_weight_map[
                                    inst_group_full_weight_name
                                ]
                                quantized_weight, weight_scale = (
                                    self.rollout.mxfp4_quantization(weight_to_quantize)
                                )
                                # The quantized version of the weight has been removed by vLLM internally.
                                # https://github.com/zyongye/vllm/blob/6a70830065701b163e36a86fd331b41b5feac401/vllm/model_executor/layers/quantization/mxfp4.py#L328
                                # We can't get it from named_parameters.
                                vllm_native_weight = None
                                vllm_native_weight_scale = None

                                for (
                                    module_name,
                                    module,
                                ) in (
                                    await self.get_underlying_model()
                                ).named_modules():
                                    w13_weight_name = f"{module_name}.w13_weight"
                                    w2_weight_name = f"{module_name}.w2_weight"
                                    w13_compatible_weight_name = (
                                        self.weight_mapper._rollout_vllm_name_to_hf(
                                            w13_weight_name
                                        )
                                    )
                                    w2_compatible_weight_name = (
                                        self.weight_mapper._rollout_vllm_name_to_hf(
                                            w2_weight_name
                                        )
                                    )

                                    # mxfp4 weight and mxfp4 weight scale are in int8 data type.
                                    # Two fp4 are packed into one int8 memory.
                                    if (
                                        inst_group_full_weight_name
                                        == w13_compatible_weight_name
                                    ):
                                        vllm_native_weight = module.quant_method.w13_weight_triton_tensor.storage.data
                                        vllm_native_weight_scale = module.quant_method.w13_precision_config.weight_scale.storage.data
                                        break
                                    elif (
                                        inst_group_full_weight_name
                                        == w2_compatible_weight_name
                                    ):
                                        vllm_native_weight = module.quant_method.w2_weight_triton_tensor.storage.data
                                        vllm_native_weight_scale = module.quant_method.w2_precision_config.weight_scale.storage.data
                                        break

                                assert (
                                    vllm_native_weight is not None
                                ), f"Failed to find the original weight for {inst_group_full_weight_name}"
                                assert (
                                    vllm_native_weight_scale is not None
                                ), f"Failed to find the original weight scale for {inst_group_full_weight_name}"

                                with torch.inference_mode():
                                    _, dim_1, dim_2 = quantized_weight.shape

                                    # check weight sync
                                    if do_weight_sync_check:
                                        valid_native_weight = vllm_native_weight[
                                            :, :dim_1, :dim_2
                                        ]
                                        if not torch.allclose(
                                            valid_native_weight, quantized_weight
                                        ):
                                            raise ValueError(
                                                f"MXFP4 weight doesn't match after weight sync and dynamic quantization for full weight name: {inst_group_full_weight_name}."
                                            )
                                    vllm_native_weight[:, :dim_1, :dim_2].copy_(
                                        quantized_weight
                                    )
                                    # check weight sync
                                    _, dim_1, dim_2 = weight_scale.shape
                                    if do_weight_sync_check:
                                        valid_native_weight_scale = (
                                            vllm_native_weight_scale[:, :dim_1, :dim_2]
                                        )
                                        if not torch.allclose(
                                            valid_native_weight_scale, weight_scale
                                        ):
                                            raise ValueError(
                                                f"MXFP4 weight scale doesn't match after weight sync and dynamic quantization for full weight name: {inst_group_full_weight_name}."
                                            )
                                    vllm_native_weight_scale[:, :dim_1, :dim_2].copy_(
                                        weight_scale
                                    )

                            else:
                                # For w13_bias, no need to quant, just copy the weight.
                                w13_bias_hp_weight = self.vllm_hp_weight_map[
                                    inst_group_full_weight_name
                                ]
                                model_param_map = self.rollout.model_param_map(
                                    self.weight_mapper
                                )
                                vllm_native_weight = model_param_map[
                                    inst_group_full_weight_name
                                ]
                                _, dim1 = w13_bias_hp_weight.shape
                                if do_weight_sync_check:
                                    if not torch.allclose(
                                        vllm_native_weight[:, :dim1], w13_bias_hp_weight
                                    ):
                                        raise ValueError(
                                            f"gate_up_proj_bias doesn't match after weight sync for full weight name: {inst_group_full_weight_name}."
                                        )

                                vllm_native_weight[:, :dim1].copy_(w13_bias_hp_weight)
            else:
                # For non-fp8/mxfp4 weights and fp8/mxfp4 not enabled cases, we just do nothing
                pass

        return (
            total_bytes_received,
            partial(
                completion_lambda,
                all_tensor_views_to_copy,
                tensors_to_check,
                post_process_list_for_lowp,
            ),
            skipped_params_cnt,
        )

    async def lazy_initialize_rollout_engine(self, load_format):
        # lazy initialization of the vllm engine.
        if not self.rollout.is_engine_initialized():
            await self.rollout.init_engine(
                quantization=self.quantization_type,
                seed=self.config.rollout.seed,
                load_format=load_format,
            )
            # TODO(zjx): make sure pause generation while weight synchronizating.
            await self.prepare_shard_infos_for_weight_sync_insts()

    @torch.no_grad()
    async def policy_to_rollout_unicast(self, command: PolicyToRolloutUnicastCommand):
        """
        Sync the weight from policy to rollout.
        This is Policy -> Rollout replica. Will only happen between
        a pair of policy and rollout replica.
        """
        # lazy initialization of the vllm engine.
        is_for_weight_resume = command.dst_replica_name == self.replica_name
        load_format = "auto" if is_for_weight_resume else "dummy"
        await self.lazy_initialize_rollout_engine(load_format)

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
            communicator_index = pynccl.create_nccl_comm(
                nccl_group_id,
                self.global_rank + command.src_replica_size,
                self.world_size + command.src_replica_size,
            )
            # cache the communicator index
            self.policy_to_rollout_nccl_communicators[nccl_unique_id_key] = (
                communicator_index
            )

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

        copy_stream = torch.cuda.Stream()

        assert (
            total_params == len(self.recv_param_key_n_rank_list)
        ), f"Mismatch in total params and received param keys: {total_params} != {len(self.recv_param_key_n_rank_list)}"

        with torch.cuda.stream(self.inference_stream):
            logger.info(
                f"Starting to execute {len(self.policy_to_rollout_recv_insts)}; {total_params}, {total_recvs} weight sync receives ..."
            )
            # recv the weight from policy
            st = time.time()
            total_bytes_received = 0

            pending_bytes = [0]
            pending_completions = []
            pending_groups = 0

            async def flush_completions(pending_bytes, pending_completions):
                recv_ready = torch.cuda.Event()
                recv_ready.record()
                copy_stream.wait_event(recv_ready)
                with torch.cuda.stream(copy_stream):
                    logger.debug(
                        f"Flushing {len(pending_completions)} completions, {pending_bytes[0] // 1024 // 1024}"
                    )
                    for completion in pending_completions:
                        await completion()
                    pending_bytes[0] = 0
                    pending_completions.clear()

            pynccl.nccl_group_start(communicator_index)

            skipped_params_cnt = 0
            transferred_params_cnt = 0
            skipped_groups_cnt = 0
            transferred_groups_cnt = 0

            for insts_group in self.policy_to_rollout_recv_insts:
                # insts_group: WeightSyncInstructionsGroup -> inst collection for a full weight tensor
                # handle inst group
                (
                    bytes_received,
                    completion_fn,
                    skipped_cnt,
                ) = await self.recv_weight_shard(
                    self.global_rank,
                    insts_group,
                    communicator_index,
                    command.trainable_only,
                    command.do_weight_sync_check,
                )
                skipped_params_cnt += skipped_cnt
                transferred_params_cnt += (
                    len(insts_group.param_instructions) - skipped_cnt
                )
                if (
                    self.weight_mapper.get_unsplited_weight_name(
                        insts_group.param_instructions[0].param_name
                    )
                    != insts_group.param_instructions[0].param_name
                ):
                    # The params in the group of this case originally belong to the same param.
                    # The following counts related with `groups` measure the original params before split.
                    # The count related with `groups` match the count in R2R which is without split.
                    skipped_groups_cnt += 1 if skipped_cnt > 0 else 0
                    transferred_groups_cnt += 0 if skipped_cnt > 0 else 1
                else:
                    skipped_groups_cnt += skipped_cnt
                    transferred_groups_cnt += (
                        len(insts_group.param_instructions) - skipped_cnt
                    )

                pending_bytes[0] += bytes_received
                pending_completions.append(completion_fn)
                total_bytes_received += bytes_received

                pending_groups += 1
                if pending_groups == constant.COSMOS_P2R_NCCL_GROUP_SIZE:
                    pynccl.nccl_group_end(communicator_index)
                    await flush_completions(pending_bytes, pending_completions)
                    pynccl.nccl_group_start(communicator_index)
                    pending_groups = 0

            pynccl.nccl_group_end(communicator_index)
            await flush_completions(pending_bytes, pending_completions)

            with torch.cuda.stream(copy_stream):
                copy_finished = torch.cuda.Event()
                copy_finished.record()

            self.inference_stream.wait_event(copy_finished)
            self.temp_recv_tensor_queue.queue.clear()

            time_eclapsed = time.time() - st
            logger.info(
                f"[Rollout] All {len(self.policy_to_rollout_recv_insts)} at step {command.weight_step} recv operations finished in {time_eclapsed:.3f} seconds with {total_bytes_received / (1024 * 1024)} MB received. While {skipped_params_cnt} non-trainable splitted params skipped and {transferred_params_cnt} trainable splitted params transferred."
            )

            if command.trainable_only:
                if not hasattr(self, "p2r_synced_trainable_params_cnt"):
                    self.p2r_synced_trainable_params_cnt = transferred_groups_cnt
                assert (
                    self.p2r_synced_trainable_params_cnt == transferred_groups_cnt
                ), f"Count of trainable unsplitted params which have been synced in P2R {transferred_groups_cnt} must match the synced_trainable_params attribute {self.p2r_synced_trainable_params_cnt}."

            self.state.set_weight_synced()

    async def broadcast_to_all_rollout_replica(
        self, broadcast_command: RolloutToRolloutBroadcastCommand
    ) -> None:
        """
        Broadcast the weight to all other rollout replicas.
        Will only happen between Rollout Replica 0 and all other Rollout Replicas.
        """
        src_replica_name: str = broadcast_command.src_replica_name
        dst_replica_names: List[str] = broadcast_command.dst_replica_names

        # lazy initialization of the vllm engine.
        if self.replica_name != src_replica_name:
            # for replicas that needs to be broadcasted, use dummy format.
            await self.lazy_initialize_rollout_engine(load_format="dummy")

        if len(dst_replica_names) > 1:
            self.prepare_trainable_params()
            skipped_params_cnt = 0
            transferred_params_cnt = 0
            logger.info("Starting broadcasting of parameters to all replicas.")
            # Only do broadcast if there are more than one rollout replicas.
            with torch.cuda.stream(self.inference_stream):
                assert (
                    self.rank_in_rollout_repicas >= 0
                ), "[Rollout] rank in rollout replicas should be set before broadcast."
                assert (
                    len(dst_replica_names) == len(self.replica_name_to_rank)
                ), "[Rollout] The vaild dst replicas num should match the replicas num that this worker holds."

                src_rank = self.replica_name_to_rank[src_replica_name]
                with torch.inference_mode():
                    for name, parameter in self.rollout.model_param_map(
                        self.weight_mapper
                    ).items():
                        if (
                            name not in self.trainable_params
                            and broadcast_command.trainable_only
                        ):
                            logger.info(
                                f"[Rollout] Skip {name} in R2R due to non trainable."
                            )
                            skipped_params_cnt += 1
                            continue
                        transferred_params_cnt += 1

                        recv_tensor = parameter
                        if not parameter.is_contiguous():
                            recv_tensor = parameter.contiguous()

                        pynccl.nccl_broadcast(
                            recv_tensor, src_rank, self.global_commnicator_idex
                        )

                        if not parameter.is_contiguous():
                            parameter.copy_(recv_tensor)

                if not self.state.weight_synced():
                    assert not broadcast_command.trainable_only, "[Rollout] Trainable only must be set to False for the first broadcast."
                    self.state.set_weight_synced()

            logger.info(
                f"[Rollout] Finished broadcasting of parameters to all replicas. While {skipped_params_cnt} unsplitted non-trainable params skipped and {transferred_params_cnt} unsplitted params transferred."
            )
            if broadcast_command.trainable_only:
                if not hasattr(self, "r2r_synced_trainable_params_cnt"):
                    self.r2r_synced_trainable_params_cnt = transferred_params_cnt
                if hasattr(self, "p2r_synced_trainable_params_cnt"):
                    # check in R2R sender side.
                    assert (
                        self.r2r_synced_trainable_params_cnt
                        == self.p2r_synced_trainable_params_cnt + len(self.misc_params)
                    ), f"Synced params count in R2R {self.r2r_synced_trainable_params_cnt} must match the sum of count of attribute {self.p2r_synced_trainable_params_cnt} and {len(self.misc_params)}."

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
            self.shutdown_signal.set()
            self.shutdown_mp_signal.set()

    def request_new_prompts(
        self, batch_size: int, validation_step: Optional[int] = None, **kwargs
    ):
        """
        Request new prompts from the controller for both training and validation.
        """
        prompts_and_is_end = (None, False)
        if self.global_rank == 0:
            if self.scheduler.task_queue.empty():
                # blocking request
                payloads, is_end = self.api_client.get_next_prompt(
                    batch_size, validation_step=validation_step, **kwargs
                )
                prompts_and_is_end = (
                    payloads if len(payloads) > 0 else None,
                    is_end,
                )

        # Broadcast the prompts and is_end to all ranks
        prompts_and_is_end = dist_utils.broadcast_object_cpu(prompts_and_is_end)
        prompts, is_end = prompts_and_is_end
        if prompts is not None:
            sp = (
                self.sampling_params
                if validation_step is None
                else self.validation_sampling_params
            )
            tasks = [
                RolloutTask(
                    idx=prompt[0],
                    payload=RLPayload.model_validate(prompt[1]),
                    sampling_params=sp,
                )
                for prompt in prompts
            ]
            self.scheduler.put_rollout_batch(tasks)
        return is_end

    async def consume_one_command(
        self, cmd_pred: Optional[Callable[[Command], bool]] = None
    ):
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

        current_command = dist_utils.broadcast_object_cpu(current_command)

        if current_command is not None:
            try:
                await self.command_executor.async_execute_command(current_command)
                logger.debug(
                    f"[Rollout] Command executed: {current_command._serialize()} for rank: {self.global_rank}"
                )
            except Exception as e:
                raise RuntimeError(
                    f"[Rollout] Command execution failed for {current_command._serialize()}"
                ) from e
        return current_command

    async def consume_command(
        self,
        cmd_pred: Optional[Callable[[Command], bool]] = None,
        timeout=constant.COSMOS_ROLLOUT_CMD_WAIT_TIMEOUT,
    ):
        """
        Consume all pending commands for weight sync using asyncio.timeout.
        To ensure the weight update is using the up-to-date commands.
        """
        try:
            async with asyncio_timeout(float(timeout)):
                last_cmd = None
                none_cnt = 0
                while True:
                    cmd = await self.consume_one_command(cmd_pred=cmd_pred)
                    if cmd is not None:
                        last_cmd = cmd
                        none_cnt = 0
                    else:
                        none_cnt += 1

                    if none_cnt >= constant.COSMOS_ROLLOUT_CMD_WAIT_TIMES and (
                        (
                            last_cmd is not None
                            and not isinstance(last_cmd, PolicyToRolloutUnicastCommand)
                        )
                        or last_cmd is None
                    ):
                        # If continuously get None for COSMOS_ROLLOUT_CMD_WAIT_TIMES times, and the last command is not P2R command, we break.
                        # Since P2R must be followed by another R2R broadcast command, we need wait.
                        # Continuously get None for COSMOS_ROLLOUT_CMD_WAIT_TIMES times to make sure the command queue is empty at that time.
                        break

                    await asyncio.sleep(
                        float(constant.COSMOS_ROLLOUT_CMD_WAIT_INTERVAL)
                    )
        except asyncio.TimeoutError:
            # Timeout reached, return normally
            pass

    @torch.no_grad()
    async def do_validation(self):
        # submit payloads to scheduler
        prompt_idxs: List[int] = []
        validation_payloads: List[RLPayload] = []
        # Do validation here
        is_end = False
        while True:
            if not is_end:
                is_end = self.request_new_prompts(
                    self.val_batch_size,
                    validation_step=self.current_step,
                )

            # wait all tasks are completed
            while self.scheduler.completed_results() != self.val_batch_size:
                await asyncio.sleep(0.1)

            rollout_results = self.scheduler.get_all()
            prompt_idxs.extend([p.idx for p in rollout_results])
            validation_payloads.extend([p.payload for p in rollout_results])

            if is_end and self.scheduler.is_idle():
                break

        # Clear the flag to indicate validation is done.
        self.validation_flag.clear()
        should_report = self.parallel_dims.tp_coord[0] == 0 and (
            self.parallel_dims.pp_coord[0] == self.parallel_dims.pp_coord[1] - 1
        )

        if should_report:
            self.reward_dispatcher.enqueue_rewards_cal(
                validation_payloads, True, self.current_step, prompt_idxs
            )
            payloads, is_validation, current_step, empty = self.report_rollouts(
                block=True
            )
            assert (
                (is_validation and payloads is not None or payloads is None)
                and (not empty or len(validation_payloads) == 0)
            ), f"Payloads must be for validation if not empty {is_validation}, {payloads}, {empty}"
            while not empty:
                assert (
                    is_validation or payloads is None
                ), f"Payloads must be for validation if not empty {is_validation}, {payloads}, {empty}"
                if payloads is not None:
                    response = ValidationReportRequest(
                        src_replica_name=self.replica_name,
                        validation_step=current_step,
                        prompt_idxs=[],
                        payloads=payloads,
                        is_end=True,
                    )
                    self.api_client.post_validation_report(response)
                payloads, is_validation, current_step, empty = (
                    self.reward_dispatcher.dequeue_rewards_cal()
                )

    @torch.no_grad()
    async def do_generate(self):
        logger.debug(f"[Rollout] generate start for rank {self.global_rank}")

        rollout_results: List[CompletedRollout] = []
        while not self.scheduler.is_idle():
            res = self.scheduler.get_all()
            if len(res) > 0:
                rollout_results.extend(res)
            else:
                await asyncio.sleep(0.1)

        assert (
            len(rollout_results) == self.batch_size
        ), f"Error: VLLM returned {len(rollout_results)} for {self.batch_size}"

        # we need filter the result with valid completions or valid completed_conversations
        valid_results: list[CompletedRollout] = []
        if self.rollout.rollout_config.multi_turn_config.enable:
            valid_results = filter_valid_multi_turn_rollout_results(rollout_results)
        else:
            valid_results = filter_valid_single_turn_rollout_results(
                rollout_results, self.tokenizer.eos_token
            )

        logger.debug(f"[Rollout] generate end for rank {self.global_rank}")

        should_report = (
            self.parallel_dims.tp_coord[0] == 0
            and (self.parallel_dims.pp_coord[0] == self.parallel_dims.pp_coord[1] - 1)
            and len(valid_results) > 0
        )

        # wait for all tasks to complete
        await self.scheduler.wait_for_all_tasks_to_complete()

        if should_report:
            # only the first tp rank in the rollout replica will post the completion to the controller.
            valid_payloads: List[RLPayload] = []
            valid_prompt_idxs: List[int] = []

            for cr in valid_results:
                valid_prompt_idxs.append(cr.idx)

                # update payload
                cr.payload.completions = cr.result.completions
                if self.rollout.rollout_config.multi_turn_config.enable:
                    cr.payload.completed_conversations = (
                        cr.result.completed_conversations
                    )
                valid_payloads.append(cr.payload)

            self.reward_dispatcher.enqueue_rewards_cal(
                valid_payloads,
                False,
                self.current_weight_version,
                valid_prompt_idxs,
            )

        if self.state.prompt_fetch_end() and self.scheduler.task_queue.empty():
            self.state.set_prompt_consume_end()
            if self.global_rank == 0:
                self.send_end_signal()

    async def main_loop_async(self):
        """Main loop with async scheduler integration (coroutine version)."""

        await self.lazy_initialize_rollout_engine(load_format="auto")

        while not self.shutdown_signal.is_set():
            # Sync consume command
            await self.consume_command(cmd_pred=None)

            if self.validation_flag.is_set():
                # If encounter validation flag during last rollout generation or this command fetch, do validation first.
                await self.do_validation()

            # If weight is not ready, nothing else to do.
            if not self.state.weight_synced():
                continue

            # try fetching new prompts if no ending signal is set
            if not self.state.prompt_fetch_end():
                self.scheduler.pause()  # for those prmpts, we don't want to generate them immediately.
                no_more_prompts = self.request_new_prompts(self.batch_size)
                if no_more_prompts:
                    logger.info(
                        f"[Rollout] Receive prompt end, wait for {self.replica_name} to finish all rollouts generation"
                    )
                    self.state.set_prompt_fetch_end()
                    # Check if scheduler has pending or active tasks
                    if self.scheduler.task_queue.empty():
                        self.state.set_prompt_consume_end()
                        if self.global_rank == 0:
                            self.send_end_signal()

            # Report rollouts (dequeue from reward_dispatcher)
            _, is_validation, _, _ = self.report_rollouts()
            assert not is_validation, "Validation report should be handled in the broadcast command rather than main loop."

            # Check if all prompts are consumed
            if self.state.prompt_consume_end():
                assert (
                    self.scheduler.task_queue.empty() and self.state.prompt_fetch_end()
                ), "[Rollout] If prompt are all consumed, prompt queue should be empty and prompt end event should be set."
                continue
            elif self.scheduler.task_queue.empty():
                continue
            else:
                # Check if the weight version is valid for current prompts
                has_front_prompt, front_prompt_weight_version = (
                    self.scheduler.get_front_prompt_weight_version()
                )
                if not has_front_prompt:
                    continue
                is_valid_prompt_for_current_weight_version = (
                    front_prompt_weight_version <= self.current_weight_version
                )
                if not is_valid_prompt_for_current_weight_version:
                    # Fully Synchronized mode is enabled, we need to wait until the weight version is updated
                    continue

                self.scheduler.resume()  # resume the scheduler to generate the prompts
                await self.do_generate()

        logger.info(f"[Rollout] Main loop of {self.replica_name} finished")

    def work(self):
        """Main work method - runs the async event loop."""
        # Initialize scheduler after engine is ready
        self.init_scheduler()
        # we need all rank run the scheduler in the same time.
        self.scheduler.start()

        # Start the thread with daemon=True
        if self.global_rank == 0:
            # create a thread to query command as a producer
            self.background_thread = threading.Thread(
                target=self.query_command_from_controller, daemon=True
            )
            self.background_thread.start()

        # Run the async main loop
        logger.info("[RolloutWorkerAsync] Starting async event loop")
        try:
            asyncio.run(self.main_loop_async())
        except Exception as e:
            logger.error(f"[RolloutWorkerAsync] Error in async loop: {e}")
            import traceback

            traceback.print_exc()
        finally:
            logger.info("[RolloutWorkerAsync] Stopping scheduler")
            self.scheduler.stop(wait=True)

        self.inference_stream.synchronize()
        self.handle_shutdown()
