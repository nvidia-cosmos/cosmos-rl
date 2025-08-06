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
from tensorrt_llm._torch.pyexecutor.py_executor import BatchState
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.sampler import SampleState

from cosmos_rl.utils.logging import logger


def cosmos_patched_executor_loop(self):
    torch.cuda.set_device(self.device_id)
    got_finish_signal = False
    num_dummy_request = 0
    with self._profiler() as profile_step:
        iter_start_time = time.time()
        iter_stats = None
        while not got_finish_signal or len(self.active_requests) > 0:
            # Cosmos-RL specific code start
            if self.ready:
                self.consume_command(cmd_pred=None)
                if not self.cosmos_state.weight_synced():
                    continue  # if weight is not synced, skip the generation and while-loop until weight is synced.
            # Cosmos-RL specific code end
            profile_step()

            if self.enable_iter_perf_stats:
                iter_start_time = time.time()
            new_requests = self._fetch_new_requests()
            got_finish_signal = self._merge_requests(new_requests) or got_finish_signal
            if got_finish_signal and len(self.active_requests) == 0:
                break
            if self.enable_iter_perf_stats:
                iter_stats = self._get_init_iter_stats(
                    len(new_requests), self.new_active_requests_queue_latency_ms
                )

            if self.kv_cache_transceiver:
                self._check_disagg_gen_transfer_status()

            if not got_finish_signal:
                num_dummy_request = self._get_num_dummy_request()
            if num_dummy_request > 0:
                self._merge_dummy_request(num_dummy_request)

            if self.draft_model_engine is not None:
                self._prepare_draft_requests()

            scheduled_batch, fitting_disagg_gen_init_requests, num_fitting_reqs = (
                self._schedule()
            )

            if self.kv_cache_transceiver:
                self._prepare_disagg_gen_init(fitting_disagg_gen_init_requests)
                if num_fitting_reqs == 0 and not fitting_disagg_gen_init_requests:
                    logger.warning(
                        "num_fitting_reqs=0 and fitting_disagg_gen_init_requests is empty, may not have enough kvCache"
                    )
                    self.kv_cache_transceiver.check_context_transfer_status(1)
            else:
                assert scheduled_batch.batch_size > 0, (
                    "fail to schedule any pending request, "
                    "probably run out of resource."
                )

            self.num_scheduled_requests = scheduled_batch.batch_size
            logger.debug(
                f"has {len(self.active_requests)} active_request, "
                f"scheduled {len(scheduled_batch.context_requests)} context requests and "
                f"{len(scheduled_batch.generation_requests)} generation requests"
            )

            self._pause_requests(scheduled_batch.paused_requests)

            finished_requests = []

            if scheduled_batch.batch_size > 0:
                if self.kv_cache_transceiver:
                    # For generation requests which have completed KV cache transfer
                    self._prepare_disagg_gen_transmission_complete(scheduled_batch)

                self.resource_manager.prepare_resources(scheduled_batch)
                if self.draft_model_engine is not None:
                    self._prepare_draft_tokens(scheduled_batch)

                # lms: Pytorch model forward step!!!
                batch_outputs = self._forward_step(scheduled_batch)

                sample_state = self._sample_async(scheduled_batch, batch_outputs)

                self._update_request_states(scheduled_batch)

                ctx_transmission_reqs = (
                    self._send_disagg_ctx_cache(scheduled_batch.context_requests)
                    if self.kv_cache_transceiver
                    else []
                )

                self._update_requests(sample_state)

                if self.kv_cache_transceiver:
                    # For context only req in transmission, we reset the state since decoder might have changed it
                    for req in ctx_transmission_reqs:
                        req.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

                self._handle_cancelled_requests()
                finished_requests = self._handle_responses()
                self.resource_manager.update_resources(scheduled_batch)
                if self.enable_kv_cache_events:
                    self._add_kv_cache_events()

            if self.kv_cache_transceiver and self.ctx_in_transmission_requests:
                self._terminate_ctx_finished_requests()

            self._gather_dp_requests_num()

            if self.enable_iter_perf_stats:
                iter_stats.inflight_batching_stats.num_ctx_tokens = (
                    self.model_engine.iter_states["num_ctx_tokens"]
                )
                self._process_iter_stats(
                    finished_requests,
                    self.active_requests,
                    BatchState(
                        sample_state=SampleState(scheduled_requests=scheduled_batch),
                        iter_stats=iter_stats,
                        iter_start_time=iter_start_time,
                    ),
                )

    self._executor_loop_cleanup()
