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

from cuda import cudart
from tensorrt_llm._torch.pyexecutor.py_executor import BatchState
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.sampler import SampleState
from tensorrt_llm.runtime.generation import CUASSERT


def cosmos_patched_executor_loop(self):
    torch.cuda.set_device(self.device_id)
    # ensure the context is created, otherwise, some MPI calls will fail.
    CUASSERT(cudart.cudaSetDevice(self.device_id))
    with self._profiler() as profile_step:
        sample_state = None
        iter_start_time = time.time()
        iter_stats = None
        while True:
            # Cosmos-RL specific code start
            if self.ready:
                self.consume_command(cmd_pred=None)
                if not self.cosmos_state.weight_synced():
                    continue  # if weight is not synced, skip the generation and while-loop until weight is synced.
            # Cosmos-RL specific code end

            profile_step()
            if self.enable_iter_perf_stats:
                iter_start_time = time.time()

            scheduled_batch, iter_stats = self._prepare_and_schedule_batch()
            if scheduled_batch is None:
                break

            self._pause_requests(scheduled_batch.paused_requests)

            finished_requests = []

            if scheduled_batch.batch_size > 0 or (
                self.enable_attention_dp and self.dist.tp_size > 1
            ):
                if self.kv_cache_transceiver:
                    # For generation requests which have completed KV cache transfer
                    self._prepare_disagg_gen_transmission_complete(scheduled_batch)

                    # Return the first token to the client
                    self._handle_first_token_response(scheduled_batch)

                self.resource_manager.prepare_resources(scheduled_batch)
                if self.drafter is not None and self.use_spec_decode:
                    self.drafter.prepare_draft_tokens(
                        scheduled_batch, self.resource_manager
                    )

                batch_outputs = self._forward_step(scheduled_batch)
                self._execute_guided_decoder(scheduled_batch, batch_outputs["logits"])

                sample_state = self._sample_async(scheduled_batch, batch_outputs)

                self._update_request_states(scheduled_batch)
                self._update_requests(sample_state)

                if self.kv_cache_transceiver:
                    ctx_transmission_reqs = self._send_disagg_ctx_cache(
                        scheduled_batch.context_requests
                    )
                    # For context only req in transmission, we reset the state since sampler might have changed it
                    for req in ctx_transmission_reqs:
                        req.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

                self._handle_canceled_requests()
                finished_requests = self._handle_responses()
                self.resource_manager.update_resources(scheduled_batch)
                if self.enable_kv_cache_events:
                    self._add_kv_cache_events()

            if self.kv_cache_transceiver and self.ctx_in_transmission_requests:
                self._terminate_ctx_finished_requests()

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
