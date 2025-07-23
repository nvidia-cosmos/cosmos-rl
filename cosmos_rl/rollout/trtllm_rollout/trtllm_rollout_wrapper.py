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
import requests
from queue import Queue
from functools import partial
from typing import List, Tuple
from cosmos_rl.dispatcher.protocol import RolloutRequest
from cosmos_rl.rollout import State, RolloutWorkerBase
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.network_util import make_request_with_retry
from cosmos_rl.utils.api_suffix import (
    COSMOS_API_NEXT_PROMPT_SUFFIX,
    COSMOS_API_ROLLOUT_SUFFIX,
)
from cosmos_rl.utils import constant
from cosmos_rl.rollout.trtllm_rollout.trtllm_rollout import TRTLLM_Rollout
from tensorrt_llm import SamplingParams
from urllib.parse import urljoin


class TRTLLMRolloutWrapper(RolloutWorkerBase):
    """
    Rollout worker with `trtllm` as the backend. pytorch backend is used for trtllm inference.
    This worker supports MPI Session that trtllm used. TRTLLMRolloutWorker is always in a single process
    that launched by cosmos-rl, not in the mpi-process that held by trtllm.
    This worker will pull prompt from the IPCQueue that managed by `CosmosTRTLLMExecutor`.
    """

    def __init__(self, config: CosmosConfig) -> None:
        super(TRTLLMRolloutWrapper, self).__init__(
            config, None
        )  # Just set parallel_dims to None.
        self.init_meta()

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

        self.batch_size = self.config.rollout.batch_size

        # Use IPCQueue Interactive with trtllm worker.

        # only init some meta info.
        self.init_meta()  # This wrapper won't handle commands, it only handle prompt fetching and end signal.

    def get_alternative_urls(self, suffix: str):
        # Get the alternative URLs for the given suffix
        urls = []
        for remote_host in self.remote_hosts:
            urls.append(urljoin(remote_host, suffix))
        return urls

    def request_new_prompts(self, batch_size: int, prompt_queue: Queue, **kwargs):
        """
        Request new prompts from the controller for both training and validation.
        """
        prompts_and_is_end = (None, False)

        prompt_id_and_payload_list = None
        is_end = False
        url_suffix = COSMOS_API_NEXT_PROMPT_SUFFIX
        try:
            if prompt_queue.empty():
                # blocking request
                prompt_meta = make_request_with_retry(
                    partial(
                        requests.get,
                        params={
                            "n": batch_size,
                            **kwargs,
                        },
                    ),
                    self.get_alternative_urls(url_suffix),
                    max_retries=constant.COSMOS_HTTP_RETRY_CONFIG.max_retries,
                )
                prompt_meta = prompt_meta.json()
                payload = prompt_meta["prompt_id_and_payload_list"]
                if len(payload) > 0:
                    prompt_id_and_payload_list = payload
                is_end = prompt_meta.get("is_end", is_end)
            else:
                prompt_id_and_payload_list = None
        except Exception as e:
            logger.error(f"[Rollout] Failed in query prompts from controller: {str(e)}")
            prompt_id_and_payload_list = None
        prompts_and_is_end = (prompt_id_and_payload_list, is_end)
        logger.info(f"[Rollout] LMS: prompts_and_is_end: {prompts_and_is_end}")
        del prompt_id_and_payload_list, is_end

        prompts, is_end = prompts_and_is_end
        if prompts is not None:
            prompt_queue.put(prompts)
        return is_end

    def send_end_signal(self, url_suffix: str):
        """
        Send end signal to the controller.
        This is used to notify the controller that the rollout worker has finished processing all prompts.
        """
        response = RolloutRequest(
            src_replica_name=self.replica_name,
            prompt_idxs=[],
            payloads=[],
            completions=[],
            is_end=True,
        )
        try:
            logger.debug(
                f"[Rollout] Posting rollout end signal to controller: {response}"
            )
            make_request_with_retry(
                partial(
                    requests.post,
                    json=response.model_dump(),
                ),
                self.get_alternative_urls(url_suffix),
                max_retries=constant.COSMOS_HTTP_RETRY_CONFIG.max_retries,
            )
        except Exception as e:
            logger.error(
                f"[Rollout] Failed in post rollout completion to controller: {str(e)}"
            )

    def check_weight_synced(self):
        pass

    @torch.no_grad()
    def main_loop(self):
        # FIXME: (lms) Support query prompt from controller.
        # while not self.shutdown_signal.is_set():
        flag = False
        while not flag:
            logger.info("[Rollout] LMS: main_loop")
            # FIXME: (lms) query the weight synced event from trtllm IPCQueue
            # if not self.state.weight_synced():
            #     continue

            if not self.state.prompt_fetch_end():
                no_more_prompts = self.request_new_prompts(
                    self.batch_size, self._prompt_queue
                )
                if no_more_prompts:
                    logger.debug(
                        f"[Rollout] Receive prompt end, wait for {self.replica_name} to finish all rollouts generation."
                    )
                    self.state.set_prompt_fetch_end()
                    # Further make sure to set `prompt_consume_end` if no more prompts to be consumed
                    if self._prompt_queue.empty():
                        self.state.set_prompt_consume_end()
                        self.send_end_signal(COSMOS_API_ROLLOUT_SUFFIX)

                if self.state.prompt_consume_end():
                    assert (
                        self._prompt_queue.empty() and self.state.prompt_fetch_end()
                    ), "[Rollout] If prompt are all consumed, prompt queue should be empty and prompt end event should be set."
                    continue
                elif self._prompt_queue.empty():
                    continue
                else:
                    prompts: List[Tuple[int, str]] = self._prompt_queue.get()
                    logger.debug(f"[Rollout] generate start for prompts: {prompts}")

                prompts = [
                    (1, "The president of the United States is")
                ]  # FIXME: (lms) remvoe this hardcode test string
                completions: List[List[str]] = self.rollout.rollout_generation(
                    prompt_id_and_payload_list=prompts,
                    data_packer=self.data_packer,
                    sampling_params=self.sampling_params,
                )

                logger.info(f"[Rollout] LMS: completions: {completions}")

                # FIXME: (lms) remove this hardcode flag, just for testing
                flag = True

    def work(self):
        self.main_loop()
