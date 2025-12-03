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
from cosmos_rl.rollout.rollout_base import RolloutBase
import torch
from typing import Union

from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config import Config as CosmosConfig

from cosmos_rl.comm.base import CommMixin
from cosmos_rl.dispatcher.protocol import Role
from cosmos_rl.utils.logging import logger

# For rollout workers import for registration
try:
    import cosmos_rl.rollout.vllm_rollout.vllm_rollout as vllm_rollout_dummy  # noqa: F401
except ImportError as e:
    logger.debug(
        f"Failed to import VLLM Rollout. Please make sure vllm is installed. Error: {e}"
    )
    pass

try:
    import cosmos_rl.rollout.example_custom_rollout.example_custom_rollout as example_custom_rollout_dummy  # noqa: F401
except ImportError as e:
    logger.debug(
        f"Failed to import example HF Rollout. Please make sure transformers is installed. Error: {e}"
    )
    pass


class State:
    UNINITIALIZED = 0
    WEIGHT_SYNCED = 1
    PROMPT_FETCH_END = 1 << 1
    PROMPT_CONSUME_END = 1 << 2

    _state: int = UNINITIALIZED

    def __init__(self):
        self._state = self.UNINITIALIZED

    def weight_synced(self):
        return (self._state & self.WEIGHT_SYNCED) != 0

    def set_weight_synced(self):
        self._state = self._state | self.WEIGHT_SYNCED

    def prompt_fetch_end(self):
        return (self._state & self.PROMPT_FETCH_END) != 0

    def set_prompt_fetch_end(self):
        self._state = self._state | self.PROMPT_FETCH_END

    def prompt_consume_end(self):
        return (self._state & self.PROMPT_CONSUME_END) != 0

    def set_prompt_consume_end(self):
        assert (
            not self.prompt_consume_end()
        ), "Prompt consume end event should not be set twice."
        self._state = self._state | self.PROMPT_CONSUME_END


class RolloutWorkerBase(CommMixin):
    def __init__(
        self,
        config: Union[CosmosConfig],
        parallel_dims: ParallelDims,
    ) -> None:
        super().__init__()
        self.config = config
        self.parallel_dims = parallel_dims
        self.runner_init()

    def runner_init(self):
        self.role = Role.ROLLOUT
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))  # rank in the node
        self.global_rank = int(os.environ.get("RANK", 0))  # rank in replica
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

        self.backend = "vllm"

        # Initialize the communication to controller.
        self.init_comm()
        self.init_redis()

    def set_rollout(self, rollout: RolloutBase):
        self.rollout = rollout


class TRTLLMRolloutWorkerBase(CommMixin):
    """
    This class is special for TRTLLM, which will be compatible with PyExecutor in trtllm.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def post_init(
        self,
        cosmos_config: CosmosConfig,
        parallel_dims: ParallelDims,
        init_comm=True,
    ):
        self.config = cosmos_config
        self.role = Role.ROLLOUT
        self.parallel_dims = parallel_dims
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))  # rank in the node
        self.global_rank = int(os.environ.get("RANK", 0))  # rank in replica
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

        self.backend = "trtllm"

        if init_comm:
            # Initialize the communication to controller.
            self.init_comm()
            self.init_redis()
