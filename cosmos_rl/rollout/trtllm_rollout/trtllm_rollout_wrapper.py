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
import time
from queue import Queue, Empty
from typing import List, Tuple, Optional, Any, Callable, Union
from torch.utils.data import Dataset
import multiprocessing as mp

from cosmos_rl.utils.constant import (
    COSMOS_REWARD_DISPATCHER_PAYLOAD_PER_TASK,
    COSMOS_REWARD_DISPATCHER_CONCURRENCY,
)


from cosmos_rl.utils.logging import logger

# Note: this must be before import tensorrt_llm and calls of MPI.init
# Otherwise, environment will not be inherited by MPI.

#####
from cosmos_rl.rollout.trtllm_rollout import patch_trtllm  # noqa: F401
####

from cosmos_rl.dispatcher.protocol import RolloutRequest
from cosmos_rl.rollout.validation import ValidationSession
from cosmos_rl.rollout import State, TRTLLMRolloutWorkerBase
from cosmos_rl.policy.config import Config as CosmosConfig


from cosmos_rl.rollout.trtllm_rollout.trtllm_rollout import TRTLLM_Rollout
from cosmos_rl.rollout.trtllm_rollout.trtllm_common import (
    ShutdownInstruction,
    ValidationInstruction,
    RolloutWrapperInstruction,
)

from tensorrt_llm import SamplingParams
from tensorrt_llm.executor.ipc import ZeroMqQueue as IpcQueue
from cosmos_rl.dispatcher.api.client import APIClient
from cosmos_rl.dispatcher.data.schema import (
    RLPayload,
)
from cosmos_rl.reward.dispatcher import RewardDispatcher
from cosmos_rl.reward.admission import (
    apply_rollout_result_to_payload,
    consume_completion_admission_metrics,
    normalize_rollout_results,
    prepare_completion_admission_report,
    select_rollout_result_completions,
)
from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.dispatcher.data.packer.base import BaseDataPacker
from cosmos_rl.dispatcher.data.data_fetcher import WorkerDataFetcher


class TRTLLMRolloutWrapper(TRTLLMRolloutWorkerBase):
    """
    Rollout worker with `trtllm` as the backend. pytorch backend is used for trtllm inference.
    This worker supports MPI Session that trtllm used. TRTLLMRolloutWorker is always in a single process
    that launched by cosmos-rl, not in the mpi-process that held by trtllm.
    This worker will pull prompt from the IPCQueue that managed by `CosmosTRTLLMExecutor`.
    """

    def __init__(self, config: CosmosConfig, **kwargs) -> None:
        super(TRTLLMRolloutWrapper, self).__init__()
        self.post_init(config, None, init_comm=False)
        # only init some meta info.
        self.api_client = APIClient(
            self.role, controller_execution_id=config.controller_execution_id
        )

        self.state = State()

        # init the prompt queue
        self._prompt_queue: Queue[List[RLPayload]] = Queue()

        self.rollout = TRTLLM_Rollout(config)
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
            assert self.val_batch_size > 0, (
                "[Rollout] val_batch_size should be greater than 0."
            )
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
        self._lifecycle_commands = Queue()

        self.life_control_thread: Optional[threading.Thread] = None
        self.rollout_wrapper_event = threading.Event()

        self.reward_dispatcher = RewardDispatcher(
            payload_per_task=COSMOS_REWARD_DISPATCHER_PAYLOAD_PER_TASK
        )

        self.setup(
            dataset=kwargs.get("dataset"),
            data_packer=kwargs.get("data_packer"),
            reward_fns=kwargs.get("reward_fns"),
            filter_reward_fns=kwargs.get("filter_reward_fns"),
            val_dataset=kwargs.get("val_dataset"),
            val_data_packer=kwargs.get("val_data_packer"),
            val_reward_fns=kwargs.get("val_reward_fns"),
        )
        atexit.register(self.handle_shutdown)

    def setup(
        self,
        dataset: Optional[Union[Dataset, Callable[[CosmosConfig], Dataset]]] = None,
        data_packer: Optional[Union[BaseDataPacker, Callable]] = None,
        reward_fns: Optional[List[Callable]] = None,
        filter_reward_fns: Optional[List[Callable]] = None,
        val_dataset: Optional[Dataset] = None,
        val_data_packer: Optional[Union[BaseDataPacker, Callable]] = None,
        val_reward_fns: Optional[List[Callable]] = None,
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

        self.reward_dispatcher.setup(
            config=self.config,
            reward_fns=reward_fns,
            filter_reward_fns=filter_reward_fns,
            val_reward_fns=val_reward_fns,
            data_packer=self.data_packer,
            val_data_packer=self.val_data_packer,
            num_workers=COSMOS_REWARD_DISPATCHER_CONCURRENCY,
        )

    def report_rollouts(self, block=False):
        while True:
            payloads, is_validation, step, empty = (
                self.reward_dispatcher.dequeue_rewards_cal()
            )
            if payloads is not None:
                if is_validation:
                    break
                reservation_payloads = list(payloads)
                payloads, metadata = consume_completion_admission_metrics(payloads)
                metadata = prepare_completion_admission_report(metadata, step)
                for i in range(len(payloads)):
                    (
                        payloads[i].completions,
                        payloads[i].completed_conversations,
                        payloads[i].completion_logprobs,
                        payloads[i].completion_token_ids,
                        _,
                    ) = self.data_packer.get_rollout_output(
                        payloads[i].completions,
                        payloads[i].completed_conversations,
                        payloads[i].completion_logprobs,
                        payloads[i].completion_token_ids,
                    )
                    # when using local dataset, we don't need to send the prompt/conversation to the controller
                    if self.config.train.local_dataset:
                        payloads[i].prompt = None
                        payloads[i].conversation = None
                response = RolloutRequest(
                    src_replica_name=self.replica_name,
                    payloads=payloads,
                    metrics=metadata,
                    is_end=False,
                )
                from cosmos_rl.reward.reservations import attach_training_rejections

                attach_training_rejections(response, reservation_payloads)
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
            if kwargs.get("validation_step") is not None:
                payloads, is_end = self._validation_session.fetch(
                    batch_size, rank_in_mesh=kwargs.get("rank_in_mesh")
                )
            else:
                payloads, is_end = self.api_client.get_next_prompt(batch_size, **kwargs)
            if self.config.train.local_dataset:
                is_validation = kwargs.get("validation_step", None) is not None
                for payload in payloads:
                    payload["prompt"] = self.data_fetcher.get_payload_by_index(
                        payload["prompt_idx"],
                        is_validation=is_validation,
                    )
                    payload["conversation"] = self.data_fetcher.get_payload_by_index(
                        payload["prompt_idx"],
                        is_validation=is_validation,
                        attr="conversation",
                    )
            prompts = payloads if len(payloads) > 0 else None

        if prompts is not None:
            prompts = [RLPayload.model_validate(payload) for payload in prompts]
            prompt_queue.put(prompts)
        return is_end

    def send_end_signal(self):
        """
        Send end signal to the controller.
        This is used to notify the controller that the rollout worker has finished processing all prompts.
        """
        if getattr(self, "_rollout_end_acknowledged", False):
            return True

        payloads, is_validation, _, empty = self.report_rollouts(block=True)
        assert not is_validation and payloads is None and empty, (
            f"Payloads must be empty and not for validation when sending end signal {is_validation}, {payloads}, {empty}"
        )
        response = RolloutRequest(
            src_replica_name=self.replica_name,
            # This wrapper aggregates the TRT-LLM executor replica. Its inner
            # ranks keep polling commands until explicit STOP.
            stays_command_participant=True,
            payloads=[],
            is_end=True,
        )
        logger.info(f"[Rollout] Posting rollout end signal to controller: {response}")
        self._rollout_end_acknowledged = bool(
            self.api_client.post_rollout_completion(response)
        )
        return self._rollout_end_acknowledged

    def bind_report_source(self, source):
        self.api_client.adopt_rollout_reporting(source)
        self.replica_name = source["replica_name"]
        self._is_registered = True

    @torch.no_grad()
    def main_loop(self):
        assert not self.rollout.rollout_config.multi_turn_config.enable, (
            "[Rollout] multi_turn_config.enable must be False for trtllm rollout."
        )
        while (source := self.cosmos_replica_name_queue.get()) is not None:
            while not self.rollout_wrapper_event.is_set():
                # this means that inside trtllmwoker, the weight is not synced yet.
                pass
            # Main process will be blocked here until the trtllm worker has all done the registration.
            # So the worker processes has done the registration.
            self.bind_report_source(source)
            logger.info(
                f"[Rollout] Bound registered report source: {self.replica_name} from trtllm WorkerProcess"
            )
            break

        while not self.shutdown_signal.is_set():
            self.consume_lifecycle_instruction()
            if self.shutdown_signal.is_set():
                break
            # 1. check if we have to do validation first
            if self.validation_event.is_set():
                # validation
                self._validation_session = ValidationSession(
                    self.api_client,
                    self.validation_round_id,
                    self.validation_step,
                    self.replica_name,
                    0,
                )
                validation_queue = Queue()
                validation_results = []
                prompt_payloads: List[Any] = []
                while True:
                    is_end = self.request_new_prompts(
                        self.val_batch_size,
                        validation_queue,
                        validation_step=self.validation_step,
                    )

                    if not validation_queue.empty():
                        payloads_list: List[RLPayload] = validation_queue.get()
                        generated = self.rollout.rollout_generation(
                            payloads=payloads_list,
                            data_packer=self.val_data_packer,
                            data_fetcher=self.data_fetcher,
                            sampling_params=self.val_sampling_params,
                        )
                        results = normalize_rollout_results(generated)
                        if results:
                            prompt_payloads.extend(payloads_list)
                            validation_results.extend(results)

                    if is_end:
                        break

                # Complete collection before rewarding/reporting it once. A
                # final fetch may carry payloads as well as the end signal;
                # reporting inside the loop drops that final batch and replays
                # every earlier prefix on subsequent iterations.
                validation_payloads = []
                for old_payload, result in zip(prompt_payloads, validation_results):
                    apply_rollout_result_to_payload(
                        old_payload,
                        result,
                        include_completed_conversations=(
                            result.completed_conversations is not None
                        ),
                    )
                    validation_payloads.append(old_payload)

                self.reward_dispatcher.enqueue_rewards_cal(
                    validation_payloads, True, self.validation_step
                )
                payloads, is_validation, current_step, empty = self.report_rollouts(
                    block=True
                )
                assert (
                    is_validation and payloads is not None or payloads is None
                ) and (not empty or not validation_payloads), (
                    "Validation report should be handled in the broadcast command."
                )
                while not empty:
                    assert is_validation or payloads is None, (
                        "Validation report should be handled in the broadcast command."
                    )
                    if payloads is not None:
                        if current_step != self._validation_session.step:
                            raise ValueError(
                                "Validation reward belongs to another step"
                            )
                        self._validation_session.report(payloads)
                    payloads, is_validation, current_step, empty = (
                        self.reward_dispatcher.dequeue_rewards_cal()
                    )
                self._validation_session.report([], is_end=True)
                self._completed_validation_round_id = self.validation_round_id
                self._completed_validation_step = self.validation_step
                self.validation_event.clear()
                # Process an already-queued final STOP or next round before
                # starting another generation batch. Only this loop changes
                # active validation identity; the IPC thread cannot overwrite it.
                continue

            _, is_validation, _, _ = self.report_rollouts()
            assert not is_validation, (
                "Validation report should be handled in the broadcast command."
            )
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
                assert self._prompt_queue.empty() and self.state.prompt_fetch_end(), (
                    "[Rollout] If prompt are all consumed, prompt queue should be empty and prompt end event should be set."
                )
                if not self.send_end_signal():
                    time.sleep(0.1)
                continue
            elif self._prompt_queue.empty():
                continue
            else:
                logger.debug(f"[Rollout] Rollout Generation for {self.replica_name}")
                payloads: List[RLPayload] = self._prompt_queue.get()
                reserved_payloads = list(payloads)
                logger.debug(f"[Rollout] generate start for prompts: {payloads}")

                generated = self.rollout.rollout_generation(
                    payloads=payloads,
                    data_packer=self.data_packer,
                    data_fetcher=self.data_fetcher,
                    sampling_params=self.sampling_params,
                )

                rollout_results = normalize_rollout_results(generated)
                if rollout_results and rollout_results[-1].completions:
                    logger.debug(
                        "[Rollout] completions[-1][-1] of %d completions from "
                        "trtllm: %s",
                        len(rollout_results[-1].completions),
                        rollout_results[-1].completions[-1],
                    )

                # Remove empty completions while preserving every producer field
                # aligned with the same completion indices.  TRT-LLM's built-in
                # producer returns RolloutResult, while accepting the historical
                # List[List[str]] shape keeps custom backends backward compatible.
                valid_results: List[RolloutResult] = []
                prompt_indices_to_remove: List[int] = []
                if rollout_results:
                    batch_size = len(payloads)
                    if len(rollout_results) != batch_size:
                        raise ValueError(
                            "[Rollout] TRT-LLM must return one result per prompt: "
                            f"got {len(rollout_results)} results for {batch_size} prompts"
                        )
                    for i in range(batch_size):
                        result = rollout_results[i]
                        valid_indices = []
                        for j, output_text in enumerate(result.completions):
                            if output_text == "":
                                logger.warning(
                                    f"[Rollout] Got empty completion for {i}th prompt {j}th generation"
                                )
                            else:
                                valid_indices.append(j)
                        # Skip the output if there is one or zero non-empty completions
                        if len(valid_indices) > 1:
                            from cosmos_rl.reward.reservations import (
                                select_training_slots,
                            )

                            payloads[i] = select_training_slots(
                                payloads[i], valid_indices
                            )
                            valid_results.append(
                                select_rollout_result_completions(result, valid_indices)
                            )
                        else:
                            prompt_indices_to_remove.append(i)
                else:
                    prompt_indices_to_remove = list(range(len(payloads)))
                if len(prompt_indices_to_remove):
                    payloads = [
                        payload
                        for i, payload in enumerate(payloads)
                        if i not in prompt_indices_to_remove
                    ]
                    assert len(payloads) == len(valid_results), (
                        "[Rollout] len(prompts) must be the same as len(valid_results) after removing empty completions"
                    )

                from cosmos_rl.reward.reservations import discarded_training_slots

                rejected = discarded_training_slots(reserved_payloads, payloads)
                if rejected:
                    self.api_client.post_rollout_completion(
                        RolloutRequest(
                            src_replica_name=self.replica_name,
                            payloads=[],
                            training_rejections=rejected,
                            metrics={"discarded_samples": len(rejected)},
                        )
                    )

                logger.debug("[Rollout] generate end!")

                should_report = len(valid_results) > 0

                if should_report:
                    # only the first tp rank in the rollout replica will post the completion to the controller.
                    valid_payloads = []
                    for old_payload, result in zip(payloads, valid_results):
                        apply_rollout_result_to_payload(
                            old_payload,
                            result,
                            include_completed_conversations=(
                                result.completed_conversations is not None
                            ),
                        )
                        valid_payloads.append(old_payload)

                    self.reward_dispatcher.enqueue_rewards_cal(
                        valid_payloads,
                        False,
                        0,
                        bypass_reward=self.config.train.train_policy.bypass_reward,
                    )

                if self.state.prompt_fetch_end() and self._prompt_queue.empty():
                    self.state.set_prompt_consume_end()
                    self.send_end_signal()

        logger.info(f"[Rollout] Main loop of {self.replica_name} finished")

    def consume_lifecycle_instruction(self):
        """Apply validation/STOP in order, outside active generation/rewards."""
        if self.validation_event.is_set():
            return
        while True:
            try:
                inst = self._lifecycle_commands.get_nowait()
            except Empty:
                return
            if isinstance(inst, ShutdownInstruction):
                self.shutdown_signal.set()
                self.shutdown_mp_signal.set()
                return
            if not inst.validation_round_id:
                raise ValueError("Validation instruction has no round identity")
            if inst.validation_round_id == getattr(
                self, "_completed_validation_round_id", None
            ):
                if inst.validation_step != self._completed_validation_step:
                    raise ValueError("Validation retry changed its step")
                continue
            self.validation_step = inst.validation_step
            self.validation_round_id = inst.validation_round_id
            self.validation_event.set()
            return

    def life_control_loop(self):
        while inst := self.cosmos_weight_sync_queue.get():
            if isinstance(inst, (ShutdownInstruction, ValidationInstruction)):
                self._lifecycle_commands.put(inst)
            elif isinstance(inst, RolloutWrapperInstruction):
                if not self.rollout_wrapper_event.is_set():
                    self.rollout_wrapper_event.set()
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
