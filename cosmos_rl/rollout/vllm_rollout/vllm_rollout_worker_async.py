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
import functools
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
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.logging import logger
from cosmos_rl.rollout.vllm_rollout.vllm_rollout_async import vLLMRolloutAsync
from cosmos_rl.rollout.rollout_task_scheduler import (
    RolloutTaskScheduler,
    CompletedRollout,
)
from cosmos_rl.dispatcher.protocol import ValidationReportRequest, RolloutRequest
from cosmos_rl.dispatcher.command import (
    Command,
    PolicyToRolloutUnicastCommand,
    RolloutToRolloutBroadcastCommand,
)
import cosmos_rl.utils.util as util
from cosmos_rl.utils import constant
import cosmos_rl.utils.distributed as dist_utils
from cosmos_rl.dispatcher.data.schema import (
    RLPayload,
    ConversationType,
    IdxAndRLPayload,
)
from torch.utils.data import Dataset
from cosmos_rl.reward.reward_calculator import RewardDispatcher
from vllm import SamplingParams

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
        self.prepare_trainable_params = types.MethodType(vLLMRolloutWorker.prepare_trainable_params, self)
        self.prepare_shard_infos_for_weight_sync_insts = types.MethodType(vLLMRolloutWorker.prepare_shard_infos_for_weight_sync_insts, self)
        self.build_global_mesh = types.MethodType(vLLMRolloutWorker.build_global_mesh, self)
        self.recv_weight_shard = types.MethodType(vLLMRolloutWorker.recv_weight_shard, self)
        self.broadcast_to_all_rollout_replica = types.MethodType(vLLMRolloutWorker.broadcast_to_all_rollout_replica, self)

        self.report_rollouts = types.MethodType(vLLMRolloutWorker.report_rollouts, self)
        self.query_command_from_controller = types.MethodType(vLLMRolloutWorker.query_command_from_controller, self)
        self.query_nccl_unique_id_from_controller = types.MethodType(vLLMRolloutWorker.query_nccl_unique_id_from_controller, self)
        self.consume_one_command = types.MethodType(vLLMRolloutWorker.consume_one_command, self)
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
        )

        # Holding temp tensors created in `recv_tensor_creator`. Do not remove this, or
        self.total_temp_tensor_pool = []
        self.misc_params = set()
        self.validation_flag = threading.Event()
        self.reward_dispatcher = RewardDispatcher()

        # Initialize RolloutTaskScheduler
        self.scheduler: Optional[RolloutTaskScheduler] = None
        self._scheduler_thread: Optional[threading.Thread] = None

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

    def init_scheduler(self):
        """Initialize the RolloutTaskScheduler (async version)."""
        if self.scheduler is not None:
            logger.warning("[RolloutWorkerAsync] Scheduler already initialized")
            return

        logger.info("[RolloutWorkerAsync] Initializing RolloutTaskScheduler")

        # Get max concurrent requests from config or use default
        max_concurrent_requests = self.config.rollout.async_config.max_concurrent_requests

        self.scheduler = RolloutTaskScheduler(
            rollout_engine=self.rollout,
            data_packer=self.data_packer,
            sampling_params=self.sampling_params,
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

    def get_underlying_model(self):
        """
        Get the underlying parallelized model in vLLM internal.
        """
        return self.rollout.get_underlying_model()

    def lazy_initialize_rollout_engine(self, load_format):
        # lazy initialization of the vllm engine.
        if not self.rollout.is_engine_initialized():
            self.rollout.init_engine(
                quantization=self.quantization_type,
                seed=self.config.rollout.seed,
                load_format=load_format,
            )
            # TODO(zjx): make sure pause generation while weight synchronizating.
            self.prepare_shard_infos_for_weight_sync_insts()

    def request_new_prompts(self, batch_size: int, **kwargs):
        """
        Request new prompts from the controller for both training and validation.
        """
        prompts_and_is_end = (None, False)
        if self.global_rank == 0:
            if self.scheduler.task_queue.empty():
                # blocking request
                payloads, is_end = self.api_client.get_next_prompt(batch_size, **kwargs)
                prompts_and_is_end = (
                    payloads if len(payloads) > 0 else None,
                    is_end,
                )

        # Broadcast the prompts and is_end to all ranks
        prompts_and_is_end = dist_utils.broadcast_object_cpu(prompts_and_is_end)
        prompts, is_end = prompts_and_is_end
        if prompts is not None:
            prompts: List[IdxAndRLPayload] = [
                (prompt[0], RLPayload.model_validate(prompt[1])) for prompt in prompts
            ]
            self.scheduler.put_rollout_batch(prompts)
        return is_end

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
                    cmd = self.consume_one_command(cmd_pred=cmd_pred)
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
                    self.scheduler.task_queue.empty()
                    and self.state.prompt_fetch_end()
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
