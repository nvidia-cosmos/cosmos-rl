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
import threading
from queue import Queue
from typing import List, Tuple, Optional, Any, Callable, Union
from torch.utils.data import Dataset
import multiprocessing as mp

from cosmos_rl.utils.constant import COSMOS_REWARD_DISPATCHER_PAYLOAD_PER_TASK


from cosmos_rl.utils.logging import logger

# Note: this must be before import tensorrt_llm and calls of MPI.init
# Otherwise, environment will not be inherited by MPI.

#####
from cosmos_rl.rollout.trtllm_rollout import patch_trtllm  # noqa: F401
####

from cosmos_rl.dispatcher.protocol import RolloutRequest, ValidationReportRequest
from cosmos_rl.rollout import State, TRTLLMRolloutWorkerBase
from cosmos_rl.policy.config import Config as CosmosConfig


from cosmos_rl.rollout.trtllm_rollout.trtllm_rollout import TRTLLM_Rollout
from cosmos_rl.rollout.trtllm_rollout.trtllm_common import (
    ShutdownInstruction,
    ValidationInstruction,
)

from tensorrt_llm import SamplingParams
from tensorrt_llm.executor.ipc import ZeroMqQueue as IpcQueue
from cosmos_rl.dispatcher.api.client import APIClient
from cosmos_rl.dispatcher.data.schema import (
    RLPayload,
)
from cosmos_rl.reward.reward_calculator import RewardDispatcher


class TRTLLMRolloutWrapper(TRTLLMRolloutWorkerBase):
    """
    Rollout worker with `trtllm` as the backend. pytorch backend is used for trtllm inference.
    This worker supports MPI Session that trtllm used. TRTLLMRolloutWorker is always in a single process
    that launched by cosmos-rl, not in the mpi-process that held by trtllm.
    This worker will pull prompt from the IPCQueue that managed by `CosmosTRTLLMExecutor`.
    """

    def __init__(self, config: CosmosConfig) -> None:
        super(TRTLLMRolloutWrapper, self).__init__()
        self.post_init(config, None, init_comm=False)
        # only init some meta info.
        self.api_client = APIClient(self.role)
        self.init_meta()  # This wrapper won't handle commands, it only handle prompt fetching and end signal.

        self.state = State()

        # init the prompt queue
        self._prompt_queue: Queue[List[Tuple[int, str]]] = Queue()

        self.rollout = TRTLLM_Rollout(config, self.tokenizer)
        self.rollout.init_engine(seed=self.config.rollout.seed, load_format="auto")

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
        self.batch_size = self.config.rollout.batch_size

        if self.config.validation.enable:
            self.val_batch_size = self.config.validation.batch_size or self.batch_size
            assert (
                self.val_batch_size > 0
            ), "[Rollout] val_batch_size should be greater than 0."
        else:
            self.val_batch_size = None

        # Use IPCQueue Interactive with trtllm worker.
        self.cosmos_replica_name_queue, self.cosmos_weight_sync_queue = (
            self.get_ipc_queue()
        )

        # Note: Unlike vLLM backend, trtllm main process receive shutdown signal from trtllm worker with IPCQueue.
        self.shutdown_signal = threading.Event()
        self.shutdown_mp_signal = mp.Event()
        self.validation_event = threading.Event()

        self.life_control_thread: Optional[threading.Thread] = None

        self.reward_dispatcher = RewardDispatcher(
            payload_per_task=COSMOS_REWARD_DISPATCHER_PAYLOAD_PER_TASK
        )

        atexit.register(self.handle_shutdown)

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
            num_workers=num_workers,
        )

    def report_rollouts(self, block=False):
        while True:
            payloads, is_validation, step, empty = (
                self.reward_dispatcher.dequeue_rewards_cal()
            )
            if payloads is not None:
                if is_validation:
                    break
                response = RolloutRequest(
                    src_replica_name=self.replica_name,
                    prompt_idxs=[],
                    payloads=payloads,
                    is_end=False,
                )
                self.api_client.post_rollout_completion(response)
            elif not block or empty:
                break
        return payloads, is_validation, step, empty

    def request_new_prompts(self, batch_size: int, prompt_queue: Queue, **kwargs):
        """
        Request new prompts from the controller for both training and validation.
        """
        prompts = None
        is_end = False

        if prompt_queue.empty():
            payloads, is_end = self.api_client.get_next_prompt(batch_size, **kwargs)
            prompts = payloads if len(payloads) > 0 else None

        if prompts is not None:
            prompts = [
                (prompt[0], RLPayload.model_validate(prompt[1])) for prompt in prompts
            ]
            prompt_queue.put(prompts)
        return is_end

    def send_end_signal(self):
        """
        Send end signal to the controller.
        This is used to notify the controller that the rollout worker has finished processing all prompts.
        """
        payloads, is_validation, _, empty = self.report_rollouts(block=True)
        assert (
            not is_validation and payloads is None and empty
        ), f"Payloads must be empty and not for validation when sending end signal {is_validation}, {payloads}, {empty}"
        response = RolloutRequest(
            src_replica_name=self.replica_name,
            prompt_idxs=[],
            payloads=[],
            completions=[],
            is_end=True,
        )
        logger.info(f"[Rollout] Posting rollout end signal to controller: {response}")
        self.api_client.post_rollout_completion(response)

    @torch.no_grad()
    def main_loop(self):
        assert (
            not self.rollout.rollout_config.multi_turn_config.enable
        ), "[Rollout] multi_turn_config.enable must be False for trtllm rollout."
        while (replica_name := self.cosmos_replica_name_queue.get()) is not None:
            # Main process will be blocked here until the trtllm worker has all done the registration.
            # So the worker processes has done the registration.
            logger.info(
                f"[Rollout] Got replica name: {replica_name} from trtllm WorkerProcess"
            )
            self.replica_name = (
                replica_name  # retrieve the replica name from trtllm worker.
            )
            # Mock the result of `register_to_controller`
            self._is_registered = True
            break

        while not self.shutdown_signal.is_set():
            # 1. check if we have to do validation first
            if self.validation_event.is_set():
                # validation
                validation_queue = Queue()
                validation_results = []
                prompt_idxs: List[int] = []
                prompt_payloads: List[Any] = []
                while True:
                    is_end = self.request_new_prompts(
                        self.val_batch_size,
                        validation_queue,
                        validation_step=self.validation_step,
                    )

                    if not validation_queue.empty():
                        prompts = validation_queue.get()
                        completions: List[List[str]] = self.rollout.rollout_generation(
                            prompt_id_and_payload_list=prompts,
                            data_packer=self.val_data_packer,
                            sampling_params=self.val_sampling_params,
                        )
                        if completions:
                            prompt_idxs.extend([prompt[0] for prompt in prompts])
                            prompt_payloads.extend([prompt[1] for prompt in prompts])
                            validation_results.extend(completions)

                    if is_end:
                        break

                    validation_payloads = []
                    for old_payload, completions in zip(
                        prompt_payloads, validation_results
                    ):
                        old_payload.completions = completions
                        validation_payloads.append(old_payload)

                    self.reward_dispatcher.enqueue_rewards_cal(
                        validation_payloads, True, self.validation_step, prompt_idxs
                    )
                    payloads, is_validation, current_step, empty = self.report_rollouts(
                        block=True
                    )
                    assert (
                        (is_validation and payloads is not None or payloads is None)
                        and not empty
                    ), "Validation report should be handled in the broadcast command."
                    while not empty:
                        assert (
                            is_validation or payloads is None
                        ), "Validation report should be handled in the broadcast command."
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
                self.validation_event.clear()

            _, is_validation, _, _ = self.report_rollouts()
            assert (
                not is_validation
            ), "Validation report should be handled in the broadcast command."
            # 2. Rollout Generation
            if not self.state.prompt_fetch_end():
                # query new prompts
                no_more_prompts = self.request_new_prompts(
                    self.batch_size, self._prompt_queue
                )
                if no_more_prompts:
                    logger.info(
                        f"[Rollout] Receive prompt end, wait for {self.replica_name} to finish all rollouts generation: {self._prompt_queue.qsize()}."
                    )
                    self.state.set_prompt_fetch_end()
                    # Further make sure to set `prompt_consume_end` if no more prompts to be consumed
                    if self._prompt_queue.empty():
                        self.state.set_prompt_consume_end()
                        self.send_end_signal()

            if self.state.prompt_consume_end():
                assert (
                    self._prompt_queue.empty() and self.state.prompt_fetch_end()
                ), "[Rollout] If prompt are all consumed, prompt queue should be empty and prompt end event should be set."
                continue
            elif self._prompt_queue.empty():
                continue
            else:
                logger.debug(f"[Rollout] Rollout Generation for {self.replica_name}")
                prompts: List[Tuple[int, RLPayload]] = self._prompt_queue.get()
                logger.debug(f"[Rollout] generate start for prompts: {prompts}")

                completions: List[List[str]] = self.rollout.rollout_generation(
                    prompt_id_and_payload_list=prompts,
                    data_packer=self.data_packer,
                    sampling_params=self.sampling_params,
                )

                logger.debug(
                    f"[Rollout] completions[-1][-1] of {len(completions[-1])} completions from trtllm: {completions[-1][-1]}"
                )

                # Remove empty completions
                valid_completions: List[List[str]] = []
                prompt_indices_to_remove: List[int] = []
                if len(completions):
                    batch_size = len(prompts)
                    for i in range(batch_size):
                        completion = completions[i]
                        skip_output = False
                        total_generation_count = len(completion)
                        empty_generation_count = 0
                        output_texts = []
                        for j in range(total_generation_count):
                            output_text = completion[j]
                            if output_text == "":
                                logger.warning(
                                    f"[Rollout] Got empty completion for {i}th prompt {j}th generation"
                                )
                                empty_generation_count += 1
                            else:
                                output_texts.append(output_text)
                        # Skip the output if there is one or zero non-empty completions
                        skip_output = (
                            total_generation_count - empty_generation_count
                        ) <= 1
                        if not skip_output:
                            valid_completions.append(output_texts)
                        else:
                            prompt_indices_to_remove.append(i)
                if len(prompt_indices_to_remove):
                    prompts = [
                        prompt
                        for i, prompt in enumerate(prompts)
                        if i not in prompt_indices_to_remove
                    ]
                    assert (
                        len(prompts) == len(valid_completions)
                    ), "[Rollout] len(prompts) must be the same as len(valid_completions) after removing empty completions"

                logger.debug("[Rollout] generate end!")

                should_report = len(valid_completions) > 0

                if should_report:
                    # only the first tp rank in the rollout replica will post the completion to the controller.
                    prompt_idxs = [prompt[0] for prompt in prompts]
                    payloads = [prompt[1] for prompt in prompts]

                    valid_payloads = []
                    for old_payload, completions in zip(payloads, valid_completions):
                        old_payload.completions = completions
                        valid_payloads.append(old_payload)

                    self.reward_dispatcher.enqueue_rewards_cal(
                        valid_payloads,
                        False,
                        0,
                        prompt_idxs,
                    )

                if self.state.prompt_fetch_end() and self._prompt_queue.empty():
                    self.state.set_prompt_consume_end()
                    self.send_end_signal()

        logger.info(f"[Rollout] Main loop of {self.replica_name} finished")

    def life_control_loop(self):
        while inst := self.cosmos_weight_sync_queue.get():
            if isinstance(inst, ShutdownInstruction):
                logger.info(
                    f"[Rollout] Received shutdown instruction of {self.replica_name}, setting shutdown signal"
                )
                self.shutdown_signal.set()
                self.shutdown_mp_signal.set()
            elif isinstance(inst, ValidationInstruction):
                self.validation_event.set()
                self.validation_step = inst.validation_step
            else:
                raise ValueError(f"[Rollout] Unknown instruction: {inst}")

    def work(self):
        # Start a thread that interact with trtllm worker.
        self.life_control_thread = threading.Thread(
            target=self.life_control_loop, daemon=True
        )
        self.life_control_thread.start()

        self.main_loop()

    def get_ipc_queue(self) -> Tuple[IpcQueue, IpcQueue]:
        return (
            self.rollout.rollout_engine.cosmos_replica_name_queue,
            self.rollout.rollout_engine.cosmos_weight_sync_queue,
        )

    def handle_shutdown(self):
        if not hasattr(self, "_shutdown_handled"):
            self._shutdown_handled = True
            if not self.shutdown_signal.is_set():
                self.shutdown_signal.set()
            if not self.shutdown_mp_signal.is_set():
                self.shutdown_mp_signal.set()
            if self.life_control_thread is not None:
                # Don't wait for life_control_thread to finish
                # self.life_control_thread.join()
                # self.life_control_thread = None
                pass

            self.unregister_from_controller()
