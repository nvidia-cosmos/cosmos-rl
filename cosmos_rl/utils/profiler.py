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


import concurrent.futures as futures
from typing import List
import os
import torch
from functools import partial

from cosmos_rl.utils.logging import logger
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.utils.s3_utils import upload_folder_to_s3
from cosmos_rl.dispatcher.api.client import APIClient
from cosmos_rl.dispatcher.protocol import SetTracePathRequest


class CosmosProfiler:
    def __init__(
        self,
        config: CosmosConfig,
        parallel_dims: ParallelDims,
        replica_name: str,
        api_client: APIClient,
    ):
        self.config = config
        self.replica_name = replica_name
        self.enable_profile = config.profiler.enable_profiler
        self.global_rank = parallel_dims.global_rank
        self.initialized = False
        self.is_started = False
        self.is_finished = False
        self.profiled_times = 0
        self.profiler = None
        self.thread_pool = None
        self.api_client = api_client
        self.enable_nsys = config.profiler.enable_nsys
        self.nsys_armed = False
        self.nsys_started = False
        self.nsys_nvtx_range_pushed = False
        self.nsys_step_num = 0
        # We do not want the timestamp part of output dir for profiling data.
        output_dir = os.path.dirname(config.train.output_dir)
        self.output_dir = os.path.join(
            output_dir, "profile_trace", f"{self.replica_name}_{self.global_rank}"
        )
        self.thread_pool = None

        # static profiler config
        self.wait_steps = config.profiler.sub_profiler_config.wait_steps
        self.warmup_steps = config.profiler.sub_profiler_config.warmup_steps
        self.active_steps = config.profiler.sub_profiler_config.active_steps
        if len(config.profiler.sub_profiler_config.rank_filter) == 0:
            self.rank_filter = [0]
        else:
            self.rank_filter = config.profiler.sub_profiler_config.rank_filter
        self.record_shape = config.profiler.sub_profiler_config.record_shape
        self.profile_memory = config.profiler.sub_profiler_config.profile_memory
        self.with_stack = config.profiler.sub_profiler_config.with_stack
        self.with_modules = config.profiler.sub_profiler_config.with_modules

    def check(self):
        return self.profiler is not None and self.enable_profile

    def _validate_config(self):
        if len(self.rank_filter) == 0:
            self.rank_filter = [
                0,
            ]
        if self.active_steps <= 0:
            raise ValueError(
                f"[Profiler] active_steps must be positive, got {self.active_steps}"
            )

    def check_finished(self):
        finished = self.is_finished
        if self.is_finished:
            self.is_finished = False
        return finished

    def _nsys_rank_enabled(self):
        return (
            self.enable_nsys
            and self.global_rank in self.rank_filter
            and torch.cuda.is_available()
        )

    def start_nsys(self):
        if not self._nsys_rank_enabled():
            return
        self._validate_config()
        if not self.nsys_armed:
            logger.info(
                f"[Profiler][Nsight] arm cudaProfilerApi capture for rank: {self.global_rank}"
            )
            self.nsys_armed = True
            self.nsys_started = False
            self.nsys_nvtx_range_pushed = False
            self.nsys_step_num = 0
            self.is_finished = False

    def maybe_start_nsys(self):
        if not self.nsys_armed or self.nsys_started:
            return
        if self.nsys_step_num == self.wait_steps + self.warmup_steps:
            self.start_nsys_capture()

    def start_nsys_capture(self, label: str | None = None):
        if not self._nsys_rank_enabled():
            return
        self._validate_config()
        if self.nsys_started:
            return
        if not self.nsys_armed:
            self.nsys_armed = True
        capture_label = label or "cosmos.nsys.capture"
        logger.info(
            f"[Profiler][Nsight] start cudaProfilerApi capture for rank: {self.global_rank}, label: {capture_label}"
        )
        torch.cuda.cudart().cudaProfilerStart()
        try:
            torch.cuda.nvtx.range_push(
                f"{capture_label} rank={self.global_rank} replica={self.replica_name}"
            )
            self.nsys_nvtx_range_pushed = True
        except Exception:
            self.nsys_nvtx_range_pushed = False
        self.nsys_started = True

    def _step_nsys(self):
        if not self.nsys_armed:
            return
        self.nsys_step_num += 1
        if (
            self.nsys_started
            and self.nsys_step_num
            >= self.wait_steps + self.warmup_steps + self.active_steps
        ):
            self.stop_nsys(mark_finished=True)

    def stop_nsys(self, mark_finished: bool = False):
        if not self.nsys_armed:
            return
        if self.nsys_started:
            logger.info(
                f"[Profiler][Nsight] stop cudaProfilerApi capture for rank: {self.global_rank}"
            )
            if self.nsys_nvtx_range_pushed:
                try:
                    torch.cuda.nvtx.range_pop()
                except Exception:
                    pass
                self.nsys_nvtx_range_pushed = False
            torch.cuda.cudart().cudaProfilerStop()
        self.nsys_armed = False
        self.nsys_started = False
        if mark_finished:
            self.is_finished = True

    def start(self):
        # start the profiler with static config
        self.start_nsys()
        if self.enable_profile:
            if self.global_rank not in self.rank_filter:
                return
            self._validate_config()
            if not self.initialized:
                self.initialized = True
                logger.info(f"[Profiler] init profiler for rank {self.global_rank}")
                self.profiler = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    schedule=torch.profiler.schedule(
                        wait=self.wait_steps,
                        warmup=self.warmup_steps,
                        active=self.active_steps,
                        repeat=1,
                    ),
                    record_shapes=self.record_shape,
                    with_stack=self.with_stack,
                    with_modules=self.with_modules,
                    profile_memory=self.profile_memory,
                )
                self.thread_pool = futures.ThreadPoolExecutor(max_workers=4)

        if self.check():
            if not self.is_started:
                logger.info(f"[Profiler] start to trace for rank: {self.global_rank}")
                self.profiler.start()
                self.is_started = True
                self.is_finished = False

    def start_dynamic(
        self,
        active_steps: int,
        rank_filter: List[int],
        record_shape: bool,
        profile_memory: bool,
        with_stack: bool,
        with_modules: bool,
    ):
        if self.enable_profile:
            if self.global_rank not in rank_filter:
                return
            self._validate_config()
            if not self.initialized:
                self.initialized = True
                logger.info(f"[Profiler] init profiler for rank {self.global_rank}")
                self.active_steps = active_steps
                self.rank_filter = rank_filter
                self.record_shape = record_shape
                self.profile_memory = profile_memory
                self.with_stack = with_stack
                self.with_modules = with_modules
                self.profiler = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    schedule=torch.profiler.schedule(
                        wait=self.wait_steps,
                        warmup=self.warmup_steps,
                        active=self.active_steps,
                        repeat=1,
                    ),
                    record_shapes=self.record_shape,
                    with_stack=self.with_stack,
                    with_modules=self.with_modules,
                    profile_memory=self.profile_memory,
                )
                self.thread_pool = futures.ThreadPoolExecutor(max_workers=4)

        if self.enable_nsys:
            self.active_steps = active_steps
            self.rank_filter = rank_filter
            self.start_nsys()

        if self.check():
            if not self.is_started:
                logger.info(f"[Profiler] start to trace for rank: {self.global_rank}")
                self.profiler.start()
                self.is_started = True
                self.is_finished = False

    def stop(self):
        if self.check():
            if self.is_started:
                self.profiler.stop()
            self.is_started = False

    def step(self):
        self._step_nsys()
        if self.check() and self.is_started:
            logger.debug(
                f"[Profiler] Step ahead for rank: {self.global_rank}, step_num: {self.profiler.step_num}"
            )
            self.profiler.step()
            if self.should_stop():
                self.stop_and_save()

    def get_total_steps_needed(self):
        if self.profiler is not None:
            warmup_steps = self.warmup_steps
            wait_steps = self.wait_steps
            active_steps = self.active_steps
            num_steps = wait_steps + warmup_steps + active_steps
            return num_steps
        return 0

    def should_stop(self):
        if self.check():
            self.is_finished = self.profiler.step_num >= self.get_total_steps_needed()
            return self.profiler.step_num >= self.get_total_steps_needed()
        return True

    def stop_and_save(self):
        if self.check():
            self.stop()
            self.save()
            # shutdown the thread pool
            self.thread_pool.shutdown(wait=True)
            # relase the profiler
            del self.profiler
            self.profiler = None
            del self.thread_pool
            self.thread_pool = None
            self.is_started = False
            self.profiled_times += 1
            self.initialized = False
            logger.info(f"[Profiler] stop trace for rank: {self.global_rank}")

    def save(self):
        if self.check():
            os.makedirs(self.output_dir, exist_ok=True)
            trace_file_name = "trace.json.gz"
            trace_file_path = os.path.join(
                self.output_dir, f"{self.profiled_times}_{trace_file_name}"
            )
            logger.info(
                f"[Profiler] save trace for rank: {self.global_rank} to file: {trace_file_path} after {self.profiler.step_num} steps."
            )
            # save the trace asynchronously
            self.profiler.export_chrome_trace(trace_file_path)

            # report to the controller
            # only report the dir
            self.report_to_controller(os.path.dirname(trace_file_path))

    def report_to_controller(self, trace_file_dir: str):
        abs_path = os.path.abspath(trace_file_dir)
        if self.config.train.ckpt.upload_s3:
            # FIXME: (lms) Now we do not have the authority to S3 storage, test it later.
            logger.info(
                f"Uploading the trace to s3: {self.config.train.ckpt.s3_bucket}..."
            )
            self.thread_pool.submit(
                upload_folder_to_s3,
                abs_path,
                self.config.train.ckpt.s3_bucket,
                self.config.train.ckpt.s3_prefix,
            )
        else:
            # just leave the trace file in local disk.
            pass
        request = SetTracePathRequest(
            replica_name=self.replica_name,
            trace_path=abs_path,
            global_rank=self.global_rank,
        )
        report_func = partial(self.api_client.post_trace_path, request)
        self.thread_pool.submit(report_func)
