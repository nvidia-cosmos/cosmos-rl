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

import threading
import asyncio
from asyncio import timeout as asyncio_timeout
from typing import Tuple

from typing import List, Optional, Callable
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.logging import logger
from cosmos_rl.rollout.worker import update_payload_from_rollout_result
from cosmos_rl.rollout.worker.rollout_control import DisaggregatedRolloutControlWorker
from cosmos_rl.rollout.worker.asynchronous.rollout_task_scheduler import (
    RolloutTaskScheduler,
    RolloutTask,
    CompletedRollout,
)
from cosmos_rl.dispatcher.protocol import ValidationReportRequest
from cosmos_rl.dispatcher.command import (
    Command,
    PolicyToRolloutUnicastCommand,
)
from cosmos_rl.utils import constant
from cosmos_rl.dispatcher.data.schema import (
    RLPayload,
    ConversationType,
)


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


class AsyncDisaggregatedRolloutControlWorker(DisaggregatedRolloutControlWorker):
    """
    Async version of DisaggregatedRolloutControlWorker.

    Inherits from DisaggregatedRolloutControlWorker.
    Control the rollout generation control flow in async mode.

    This worker uses coroutine-based execution:
    - Main loop runs in asyncio event loop
    - Uses RolloutTaskScheduler.run_async() for coroutine-based scheduling
    - RolloutTaskScheduler runs in the same event loop, not a separate thread
    - Pauses the scheduler during weight synchronization to avoid conflicts
    - Provides better throughput and resource utilization with async/await
    - Sync operations (API calls, reward calculation) run in thread executor
    """

    SUPPOPRT_BACKEND = ["vllm_async"]

    def __init__(
        self, config: CosmosConfig, parallel_dims: ParallelDims, **kwargs
    ) -> None:
        assert (
            config.rollout.backend in self.SUPPOPRT_BACKEND
        ), f"AsyncDisaggregatedRolloutControlWorker only supports {self.SUPPOPRT_BACKEND} backends, but got {config.rollout.backend}"

        # skip the init of DisaggregatedRolloutControlWorker.
        super(AsyncDisaggregatedRolloutControlWorker, self).__init__(
            config, parallel_dims
        )

        self.scheduler: Optional[RolloutTaskScheduler] = None

    def init_scheduler(self):
        """Initialize the RolloutTaskScheduler (async version)."""
        if self.scheduler is not None:
            logger.warning("[RolloutWorkerAsync] Scheduler already initialized")
            return

        logger.info("[RolloutWorkerAsync] Initializing RolloutTaskScheduler")

        self.scheduler = RolloutTaskScheduler(
            rollout_engine=self.rollout,
            data_packer=self.data_packer,
            max_concurrent_requests=self.config.rollout.async_config.max_concurrent_requests,
            stream=self.inference_stream,
            check_interval=0.1,
        )

        logger.info(
            "[RolloutWorkerAsync] RolloutTaskScheduler initialized (will run in async mode)"
        )

    def handle_shutdown(self):
        """
        override the handle_shutdown method in DisaggregatedRolloutControlWorker.
        Stop the scheduler before shutdown.

        Handle shutdown and cleanup.
        """
        # Only call once
        if not hasattr(self, "_shutdown_handled"):
            # Scheduler is stopped in main_loop_async's finally block
            logger.info("[RolloutWorkerAsync] Handling shutdown")
            self.rollout.shutdown()

        super().handle_shutdown()

    async def consume_command(
        self,
        cmd_pred: Optional[Callable[[Command], bool]] = None,
        timeout=constant.COSMOS_ROLLOUT_CMD_WAIT_TIMEOUT,
    ):
        """
        override the consume_command method in DisaggregatedRolloutControlWorker. Use asyncio.timeout to consume the command.

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
                    if self.config.train.local_dataset:
                        pl.reference_answer = self.data_fetcher.query_reference_answer(
                            pl.prompt_idx,
                            "val",
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

        no_more_prompts = self.request_new_prompts(
            batch_size=batch_size,
            prompt_queue=self._prompt_queue,
            validation_step=validation_step,
        )
        prompts = self._prompt_queue.get()

        # packing the prompts into tasks and put into the scheduler
        if prompts is not None:
            fetched_prompts = len(prompts)
            tasks = [
                RolloutTask(
                    idx=prompt.prompt_idx,
                    payload=prompt,
                    is_validation=validation_step is not None,
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
                pl = update_payload_from_rollout_result(
                    cr.payload,
                    cr.result,
                    self.rollout.rollout_config.multi_turn_config.enable,
                )
                if self.config.train.local_dataset:
                    pl.reference_answer = self.data_fetcher.query_reference_answer(
                        pl.prompt_idx
                    )
                valid_payloads.append(pl)

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

    async def main_loop(self):
        """
        override the main loop method in DisaggregatedRolloutControlWorker.

        main loop with async scheduler integration (coroutine version).
        """

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
                # 1. should wait for the ongoing rollout tasks to complete before updating weight.
                # 2. should ensure that the weight update task does not run during validation.
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
        """
        override the work method in DisaggregatedRolloutControlWorker.

        main work method - runs the async event loop.
        """
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
            asyncio.run(self.main_loop())
        except Exception as e:
            logger.error(f"[RolloutWorkerAsync] Error in async loop: {e}")
            import traceback

            traceback.print_exc()

        self.inference_stream.synchronize()
        self.handle_shutdown()
