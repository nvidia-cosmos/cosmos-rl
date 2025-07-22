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

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.rollout import RolloutWorkerBase

"""
1. Extend PyExecutor to support Cosmos-specific features.
2. Initialize distributed environment.
"""


class CosmosTRTLLMWorker(PyExecutor, RolloutWorkerBase):
    """
    CosmosTRTLLMExecutor is a wrapper of PyExecutor to support Cosmos-specific features.
    P2R and R2R of cosmos-rl are implemented in this class.
    """

    def __init__(self, *args, **kwargs) -> None:
        # just call the init of PyExecutor
        super().__init__(*args, **kwargs)

    def set_cosmos_config(self, config: CosmosConfig):
        self.cosmos_config = config
        parallel_dims = ParallelDims.from_config(
            parallesim_config=self.config.rollout.parallelism
        )
        self.parallel_dims = parallel_dims

        # build the mesh
        self.parallel_dims.build_mesh(device_type="cuda")

    def work(self):
        pass
