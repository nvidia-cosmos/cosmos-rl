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
import threading
import asyncio
from queue import Queue
import atexit
import types
from asyncio import timeout as asyncio_timeout
from typing import Tuple

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
    RolloutTask,
    CompletedRollout,
)
from cosmos_rl.rollout.utils import update_payload_from_rollout_result
from cosmos_rl.dispatcher.protocol import ValidationReportRequest
from cosmos_rl.dispatcher.command import (
    Command,
    BuildMeshCommand,
    PolicyToRolloutUnicastCommand,
    RolloutToRolloutBroadcastCommand,
)
import cosmos_rl.utils.util as util
from cosmos_rl.utils import constant
import cosmos_rl.utils.distributed as dist_utils
from cosmos_rl.dispatcher.data.schema import (
    RLPayload,
    ConversationType,
)
from torch.utils.data import Dataset
from cosmos_rl.reward.reward_calculator import RewardDispatcher
from cosmos_rl.utils.command_executor import CommandExecutor
from cosmos_rl.dispatcher.data.data_fetcher import WorkerDataFetcher
from cosmos_rl.dispatcher.data.packer import DataPacker
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

    def __init__(
        self, config: CosmosConfig, parallel_dims: ParallelDims, **kwargs
    ) -> None:
        super(vLLMRolloutWorkerAsync, self).__init__(config, parallel_dims)

        # TODO(zjx): refactor those methods to common methods in RolloutWorkerBase
        # reuse some of the vLLMRolloutWorker methods
        self.prepare_trainable_params = types.MethodType(
            vLLMRolloutWorker.prepare_trainable_params, self
        )
        self.prepare_shard_infos_for_weight_sync_insts = types.MethodType(
            vLLMRolloutWorker.prepare_shard_infos_for_weight_sync_insts, self
        )
        self.recv_weight_shard = types.MethodType(
            vLLMRolloutWorker.recv_weight_shard, self
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
        self.rollout: vLLMRolloutAsync = vLLMRolloutAsync(self.config)
        self.eos_token = util.setup_tokenizer(
            self.config.policy.model_name_or_path
        ).eos_token

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
        self.current_step = 0

        self.setup(
            dataset=kwargs.get("dataset"),
            data_packer=kwargs.get("data_packer"),
            reward_fns=kwargs.get("reward_fns"),
            filter_reward_fns=kwargs.get("filter_reward_fns"),
            val_dataset=kwargs.get("val_dataset"),
            val_data_packer=kwargs.get("val_data_packer"),
            val_reward_fns=kwargs.get("val_reward_fns"),
        )

    def setup(
        self,
        dataset: Optional[Union[Dataset, Callable[[CosmosConfig], Dataset]]] = None,
        data_packer: Optional[DataPacker] = None,
        reward_fns: Optional[List[Callable]] = None,
        filter_reward_fns: Optional[List[Callable]] = None,
        val_dataset: Optional[Dataset] = None,
        val_data_packer: Optional[DataPacker] = None,
        val_reward_fns: Optional[List[Callable]] = None,
        num_workers: int = 8,
    ):
        self.init_data_packer(
            data_packer=data_packer,
            val_data_packer=val_data_packer,
        )

        self.data_fetcher = WorkerDataFetcher(
            config=self.config,
            dataset=dataset,
            val_dataset=val_dataset,
            data_packer=self.data_packer,
            val_data_packer=self.val_data_packer,
            is_rl=True,
        )

        self.reward_dispatcher.setup(
            config=self.config,
            data_fetcher=self.data_fetcher,
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
            RolloutToRolloutBroadcastCommand,
            types.MethodType(vLLMRolloutWorker.broadcast_to_all_rollout_replica, self),
        )
        self.command_executor.register_command_handler(
            PolicyToRolloutUnicastCommand,
            types.MethodType(vLLMRolloutWorker.policy_to_rollout_unicast, self),
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

    def request_new_prompts(
        self, batch_size: int, validation_step: Optional[int] = None, **kwargs
    ) -> Tuple[List[RLPayload], bool]:
        """
        Request new prompts from the controller for both training and validation.
        """
        prompts_and_is_end = (None, False)
        if self.global_rank == 0:
            if self.scheduler.task_queue.empty():
                # TODO(zjx): we haven't received enough prompts from the controller.
                # blocking request
                payloads, is_end = self.api_client.get_next_prompt(
                    batch_size, validation_step=validation_step, **kwargs
                )

                assert all(
                    payload["prompt_idx"] >= 0 for payload in payloads
                ), "All payloads should have a valid prompt index"

                is_validation = validation_step is not None
                if len(payloads) > 0:
                    if self.config.train.local_dataset:
                        for payload in payloads:
                            payload["prompt"] = self.data_fetcher.get_payload_by_index(
                                payload["prompt_idx"],
                                is_validation=is_validation,
                            )
                            payload["conversation"] = (
                                self.data_fetcher.get_payload_by_index(
                                    payload["prompt_idx"],
                                    is_validation=is_validation,
                                    attr="conversation",
                                )
                            )
                    payloads = [
                        RLPayload.model_validate(payload) for payload in payloads
                    ]
                prompts_and_is_end = (
                    payloads if len(payloads) > 0 else None,
                    is_end,
                )

        # Broadcast the prompts and is_end to all ranks
        prompts_and_is_end: Tuple[List[RLPayload], bool] = (
            dist_utils.broadcast_object_cpu(prompts_and_is_end)
        )
        return prompts_and_is_end

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

    async def do_validation(self):
        # submit payloads to scheduler
        total_prompts_count = 0
        total_validation_payload_count = 0
        # Do validation here
        is_end = False
        while True:
            if not is_end:
                (
                    fetched_prompts,
                    no_more_prompts,
                ) = await self._feed_prompts_to_scheduler(
                    self.val_batch_size, validation_step=self.current_step
                )
                total_prompts_count += fetched_prompts
                is_end |= no_more_prompts

            await asyncio.sleep(0)

            # get processed results
            rollout_results = self.scheduler.get_all()
            if rollout_results:
                validation_payloads: List[RLPayload] = []
                for rr in rollout_results:
                    pl = update_payload_from_rollout_result(
                        rr.payload,
                        rr.result,
                        self.rollout.rollout_config.multi_turn_config.enable,
                    )
                    validation_payloads.append(pl)
                total_validation_payload_count += len(validation_payloads)

                self.reward_dispatcher.enqueue_rewards_cal(
                    validation_payloads, True, self.current_step
                )

            if (
                is_end
                and self.scheduler.is_idle()
                and total_prompts_count == total_validation_payload_count
            ):
                break

        # Clear the flag to indicate validation is done.
        self.validation_flag.clear()
        should_report = self.parallel_dims.tp_coord[0] == 0 and (
            self.parallel_dims.pp_coord[0] == self.parallel_dims.pp_coord[1] - 1
        )

        if should_report:
            payloads, is_validation, current_step, empty = self.report_rollouts(
                block=True
            )
            assert (
                (is_validation and payloads is not None or payloads is None)
                and (not empty or total_validation_payload_count == 0)
            ), f"Payloads must be for validation if not empty {is_validation}, {payloads}, {empty}"
            while not empty:
                assert (
                    is_validation or payloads is None
                ), f"Payloads must be for validation if not empty {is_validation}, {payloads}, {empty}"
                if payloads is not None:
                    for i in range(len(payloads)):
                        (
                            payloads[i].completions,
                            payloads[i].completed_conversations,
                            payloads[i].completion_logprobs,
                            payloads[i].completion_token_ids,
                            _,
                        ) = self.val_data_packer.get_rollout_output(
                            payloads[i].completions,
                            payloads[i].completed_conversations,
                            payloads[i].completion_logprobs,
                            payloads[i].completion_token_ids,
                        )
                        if self.config.train.train_policy.rollout_as_token_ids:
                            payloads[i].completions = [""] * len(
                                payloads[i].completions
                            )

                    response = ValidationReportRequest(
                        src_replica_name=self.replica_name,
                        validation_step=current_step,
                        payloads=payloads,
                        is_end=True,
                    )
                    self.api_client.post_validation_report(response)
                payloads, is_validation, current_step, empty = (
                    self.reward_dispatcher.dequeue_rewards_cal()
                )

    async def _feed_prompts_to_scheduler(
        self, batch_size: int, validation_step: Optional[int] = None
    ) -> Tuple[int, bool]:
        """
        Try to fetch new prompts from the controller.
        Returns:
            fetched_prompts (int): the number of prompts fetched
            is_end (bool): whether the prompts are the end of the dataset
        """
        fetched_prompts = 0
        # try fetching new prompts
        if self.scheduler.pending_tasks() > self.scheduler.max_concurrent_requests:
            # skip fetching new prompts if the scheduler is busy
            return fetched_prompts, False

        prompts, no_more_prompts = await asyncio.to_thread(
            self.request_new_prompts,
            batch_size=batch_size,
            validation_step=validation_step,
        )

        # packing the prompts into tasks and put into the scheduler
        sp = (
            self.sampling_params
            if validation_step is None
            else self.val_sampling_params
        )
        if prompts is not None:
            fetched_prompts = len(prompts)
            tasks = [
                RolloutTask(
                    idx=prompt.prompt_idx,
                    payload=prompt,
                    sampling_params=sp,
                )
                for prompt in prompts
                # filter the prompts with valid weight version for fully synchronized mode
                if prompt.weight_version <= self.current_weight_version
            ]
            self.scheduler.put_rollout_batch(tasks)

        return fetched_prompts, no_more_prompts

    async def _check_generate_results(self) -> bool:
        """
        Check if there are enough rollout results to report.

        Returns:
            True if there are enough rollout results to report, False otherwise.
        """
        # make sure all prompts are completed before reporting.
        rollout_results: List[CompletedRollout] = self.scheduler.get_all()

        # we need filter the result with valid completions or valid completed_conversations
        valid_results: list[CompletedRollout] = []
        if self.rollout.rollout_config.multi_turn_config.enable:
            valid_results = filter_valid_multi_turn_rollout_results(rollout_results)
        else:
            valid_results = filter_valid_single_turn_rollout_results(
                rollout_results, self.eos_token
            )

        should_report = (
            self.parallel_dims.tp_coord[0] == 0
            and (self.parallel_dims.pp_coord[0] == self.parallel_dims.pp_coord[1] - 1)
            and len(valid_results) > 0
        )

        if should_report:
            # only the first tp rank in the rollout replica will post the completion to the controller.
            valid_payloads: List[RLPayload] = []

            for cr in valid_results:
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
            )

        # Report rollouts (dequeue from reward_dispatcher)
        _, is_validation, _, _ = await asyncio.to_thread(self.report_rollouts)
        assert not is_validation, "Validation report should be handled in the broadcast command rather than main loop."

    async def do_generate(self):
        """
        Do the generation process.
        """
        if not self.state.prompt_fetch_end():
            _, is_end = await self._feed_prompts_to_scheduler(
                self.batch_size, validation_step=None
            )
            if is_end:
                logger.info(
                    f"[Rollout] Receive prompt end, wait for {self.replica_name} to finish all rollouts generation"
                )
                self.state.set_prompt_fetch_end()

        # alway check the generate results to report to the controller.
        await self._check_generate_results()
        await asyncio.sleep(0)

    async def main_loop_async(self):
        """Main loop with async scheduler integration (coroutine version)."""

        self.lazy_initialize_rollout_engine(load_format="auto")

        try:
            # Start scheduler's worker loop as a coroutine
            await self.scheduler.start_async()

            while not self.shutdown_signal.is_set():
                # All operations run in the same event loop, enabling true async concurrency
                await self.consume_command(cmd_pred=None)

                # If weight is not ready, nothing else to do.
                if not self.state.weight_synced():
                    await asyncio.sleep(0)
                    continue

                # TODO(zjx):
                # 1. 在 update weight 之前，应当等待正在执行的 rollout 任务完成。
                # 2. 在 validation 时，应当确保 weight update 任务不要运行
                if self.validation_flag.is_set():
                    # If encounter validation flag during last rollout generation or this command fetch, do validation first.
                    await self.do_validation()

                # check if we should finish the rollout generation worker
                if (
                    self.state.prompt_fetch_end()
                    and self.scheduler.is_all_tasks_completed()
                    # all reward calculation tasks are reported
                    and self.reward_dispatcher.is_empty()
                ):
                    self.state.set_prompt_consume_end()

                # Check if all prompts are consumed, if so, send end signal to the controller.
                if self.state.prompt_consume_end():
                    # Send end signal to the controller
                    # Because we first report_rollouts() to the controller, so we don't need to check the reward_dispatcher queue here.
                    self.shutdown_signal.set()
                    if self.global_rank == 0:
                        self.send_end_signal()
                    continue

                # execute the fetch and generate process (async version)
                await self.do_generate()

                # This allows other coroutines to process
                await asyncio.sleep(0)
        finally:
            # Stop scheduler gracefully
            logger.info("[RolloutWorkerAsync] Stopping scheduler worker loop")
            await self.scheduler.stop_async()

        logger.info(f"[Rollout] Main loop of {self.replica_name} finished")

    def work(self):
        """Main work method - runs the async event loop."""
        # Initialize scheduler after engine is ready
        self.init_scheduler()

        # Start the thread with daemon=True
        if self.global_rank == 0:
            # create a thread to query command as a producer
            self.background_thread = threading.Thread(
                target=self.query_command_from_controller, daemon=True
            )
            self.background_thread.start()

        # Run the async main loop
        # IMPORTANT: vLLM engine, scheduler, and all async operations run in this single event loop
        logger.info("[RolloutWorkerAsync] Starting async event loop")
        try:
            asyncio.run(self.main_loop_async())
        except Exception as e:
            logger.error(f"[RolloutWorkerAsync] Error in async loop: {e}")
            import traceback

            traceback.print_exc()

        self.inference_stream.synchronize()
        self.handle_shutdown()
