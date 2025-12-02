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

import os
import torch

from transformers import AutoConfig


from cosmos_rl.comm.base import WorkerBase
from cosmos_rl.comm.base import CommMixin
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils import util
from cosmos_rl.dispatcher.protocol import Role
from cosmos_rl.utils.profiler import CosmosProfiler


class PolicyWorkerBase(WorkerBase, CommMixin):
    def __init__(self, config: CosmosConfig, parallel_dims: ParallelDims, **kwargs):
        super(PolicyWorkerBase, self).__init__(config)
        self.parallel_dims = parallel_dims

        self.hf_config = util.retry(AutoConfig.from_pretrained)(
            self.config.policy.model_name_or_path,
            trust_remote_code=True,
        )

        if self.config.policy.parallelism.dp_shard_size == -1:
            self.config.policy.parallelism.dp_shard_size = parallel_dims.dp_shard
        # Parallel parameters

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.global_rank = int(os.environ.get("RANK", 0))
        self.role = Role.POLICY
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

        self.check_config()
        self.dp_rank, self.dp_world_size = 0, 1
        if self.parallel_dims.dp_enabled:
            self.dp_rank = self.parallel_dims.mesh["dp"].get_local_rank()
            self.dp_world_size = self.parallel_dims.mesh["dp"].size()

        self.train_stream = torch.cuda.current_stream()
        self.init_comm()

        # profiler is initialized after the init_comm()
        self.profiler = CosmosProfiler(
            self.config,
            parallel_dims,
            replica_name=self.replica_name,
            api_client=self.api_client,
        )

    def check_config(self):
        mini_batch = 1
        policy_type = self.config.train.train_policy.type
        train_batch_per_replica = self.config.train.train_batch_per_replica
        dp_shard_size = self.config.policy.parallelism.dp_shard_size
        error_msg = f"train_batch_per_replica({train_batch_per_replica}) of {policy_type} must be divisible by dp_shard_size({dp_shard_size})"
        mini_batch = self.config.train.train_policy.mini_batch
        if policy_type == "grpo":
            error_msg += f" * mini_batch({mini_batch})"
            assert dp_shard_size == self.parallel_dims.dp_shard
            assert dp_shard_size > 0, "dp_shard_size must be greater than 0"
            assert (
                train_batch_per_replica % (dp_shard_size * mini_batch) == 0
            ), error_msg
        else:
            # TODO(jiaxinc): Optimize this:
            #  for SFT,`train_batch_per_replica` stands for the batch_size for a DP worker,
            #  not really for a training replica
            assert (
                train_batch_per_replica % mini_batch == 0
            ), f"train_batch_per_replica({train_batch_per_replica}) of {policy_type} must be divisible by mini_batch({mini_batch})"
        logger.info("Config checked successfully")

    def execute(self):
        """
        Execute the training.
        """
        assert self.trainer is not None, "[Policy] Trainer has not been built."
        try:
            self.main_loop()
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise e
        finally:
            self.destroy_worker()
