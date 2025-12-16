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

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.utils.distributed import init_distributed
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.dispatcher.api.client import APIClient
from cosmos_rl.reference.worker.teacher_worker import TeacherWorker


def reference_entry(**kwargs):
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    api_client = APIClient(role="REFERENCE")
    metadata = api_client.get_controller_metadata()

    if metadata["config"] is None:
        raise RuntimeError(
            f"[Policy] Please first go to http://{api_client.remote_ips}:{api_client.remote_port} to configure training parameters."
        )
    cosmos_config = CosmosConfig.from_dict(metadata["config"])

    logger.info(f"[Reference] Loaded configuration: {cosmos_config.model_dump()}")

    # Init distribution and build device mesh
    parallel_dims = ParallelDims.from_config(
        parallesim_config=cosmos_config.reference.parallelism
    )
    init_distributed()
    parallel_dims.build_mesh(device_type="cuda")

    custom_logger_fns = kwargs.get("custom_logger_fns", [])
    hook_fns = kwargs.get("hook_fns", {})

    reference_worker = TeacherWorker(
        config=cosmos_config,
        parallel_dims=parallel_dims,
        dataset=kwargs.get("dataset", None),
        data_packer=kwargs.get("data_packer", None),
        # custom logger functions and hook functions haven't been used in RLPolicyWorker yet
        custom_logger_fns=custom_logger_fns,
        hook_fns=hook_fns,
    )
    reference_worker.execute()
