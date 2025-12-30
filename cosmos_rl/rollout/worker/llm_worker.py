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

import sys

from cosmos_rl.comm.base import WorkerBase
from cosmos_rl.policy.config import Config as RolloutConfig
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.dispatcher.api.client import APIClient
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.distributed import init_distributed, destroy_distributed
from cosmos_rl.utils.async_utils import unsafe_enable_nest_asyncio
from cosmos_rl.rollout.worker.rollout_control import (
    DisaggregatedRolloutControlWorker,
)


class LLMRolloutWorker(WorkerBase):
    def __init__(self, **kwargs):
        # For LLM, config is retrieved from controller, so we pass None to the base class.
        super().__init__(None)

        api_client = APIClient(role="ROLLOUT")
        metadata = api_client.get_controller_metadata()

        if metadata["config"] is None:
            raise RuntimeError(
                f"[Rollout] Please first go to http://{api_client.remote_ips}:{api_client.remote_port} to configure training parameters."
            )

        cosmos_rollout_config = RolloutConfig.from_dict(
            metadata["config"]
        )  # just use config as key temporarily

        task_type = cosmos_rollout_config.train.train_policy.type
        if task_type not in ["grpo"]:
            logger.info(
                "[Rollout] Task in controller is not type of Reinforcement Learning. Aborted."
            )
            sys.exit(0)

        logger.info(
            f"[Rollout] Loaded rollout configuration: {cosmos_rollout_config.rollout.model_dump()}"
        )
        self.config = cosmos_rollout_config

        self.build_runner(**kwargs)

    def execute(self):
        try:
            self.rollout_worker.work()
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise e
        finally:
            self.destroy_worker()

    def build_runner(self, **kwargs):
        rollout_backend = self.config.rollout.backend
        self.rollout_worker = None

        if rollout_backend != "trtllm":
            parallel_dims = ParallelDims.from_config(
                parallesim_config=self.config.rollout.parallelism
            )
            init_distributed()
            parallel_dims.build_mesh(device_type="cuda")
            if self.config.rollout.mode == "async":
                # In this case, we should enable nest_asyncio to allow call asyncio.run from a running event loop.
                unsafe_enable_nest_asyncio()
            self.rollout_worker = DisaggregatedRolloutControlWorker(
                self.config, parallel_dims, **kwargs
            )
        elif rollout_backend == "trtllm":
            try:
                from cosmos_rl.rollout.trtllm_rollout.trtllm_rollout_wrapper import (
                    TRTLLMRolloutWrapper,
                )
            except ImportError as e:
                logger.error(f"[Rollout] TRTLLMRolloutWrapper importing failed! {e}")
                raise e
            # if backend is trtllm, we leave distribution initialization to trtllm executor.
            self.rollout_worker = TRTLLMRolloutWrapper(self.config, **kwargs)
        else:
            raise ValueError(f"Invalid rollout backend: {rollout_backend}")

    def destroy_worker(self):
        if self.rollout_worker is not None:
            del self.rollout_worker
            self.rollout_worker = None
        destroy_distributed()
        logger.info("[Rollout] Destroy context of torch dist.")
