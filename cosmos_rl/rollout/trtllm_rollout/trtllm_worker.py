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
import msgpack
from functools import partial
from typing import List

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.rollout import RolloutWorkerBase
from cosmos_rl.rollout import State
from cosmos_rl.utils.parallelism_map import (
    ParallelTopoMapperGroup,
)
from cosmos_rl.dispatcher.command import (
    Command,
)
from cosmos_rl.utils.api_suffix import (
    COSMOS_API_ROLLOUT_SHARD_INFOS_SUFFIX,
)
from cosmos_rl.policy.model import ModelRegistry, WeightMapper
from cosmos_rl.utils import constant
from cosmos_rl.utils.logging import logger
import cosmos_rl.utils.distributed as dist_util
from cosmos_rl.utils.network_util import make_request_with_retry
import cosmos_rl.utils.util as util

from transformers import AutoConfig

from queue import Queue

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
        parallel_dims = ParallelDims.from_config(
            parallesim_config=config.rollout.parallelism
        )
        self.parallel_dims = parallel_dims

        # build the mesh
        self.parallel_dims.build_mesh(device_type="cuda")

        # init the RolloutWorkerBase
        RolloutWorkerBase.__init__(self, config, parallel_dims)
        self.post_init()

        self.cosmos_state = State()

        # CommandQueue queried from controller.
        self.cosmos_command_queue: Queue[Command] = Queue()
        self.cosmos_prompt_queue: Queue[List[List[int, str]]] = Queue()

        self.cosmos_global_commnicator_idex = -1
        self.cosmos_rank_in_rollout_repicas = -1

        self.cosmos_policy_to_rollout_nccl_communicators = {}

        self.cosmos_batch_size = self.config.rollout.batch_size

        # For Polocy to Rollout weight mapping
        hf_config = util.retry(AutoConfig.from_pretrained)(
            self.config.policy.model_name_or_path,
            trust_remote_code=True,
        )
        model_type = hf_config.model_type
        if not ModelRegistry.check_model_type_supported(model_type):
            logger.warning(
                f"[Rollout] Replica can not find {model_type} in weight mapper, use {constant.COSMOS_HF_MODEL_TYPES} model type instead, with replica name: {self.replica_name}"
            )
            model_type = constant.COSMOS_HF_MODEL_TYPES
        self.cosmos_weight_mapper = WeightMapper.get_weight_mapper(model_type)(
            hf_config
        )
        self.cosmos_model_config = hf_config

        self.inference_stream = torch.cuda.Stream()

    def prepare_shard_infos_for_weight_sync_insts(self):
        self.vllm_weight_inplace_view_map, grouped_recv_param_key_n_rank_list = (
            self.weight_mapper.rollout_prepare_recv(self.get_underlying_model())
        )
        self.recv_param_key_n_rank_list = []
        param_groups = []
        for group in grouped_recv_param_key_n_rank_list:
            self.recv_param_key_n_rank_list.extend(group)
            if len(group) > 1:
                param_groups.append([x[0] for x in group])
        self.recv_param_key_n_rank_list = sorted(
            self.recv_param_key_n_rank_list, key=lambda x: x[0]
        )

        local_shard_infos = ParallelTopoMapperGroup(
            self.parallel_dims,
            self.model_config,
            is_policy=False,
            underlying_model=self.get_underlying_model(),
            weight_mapper=self.weight_mapper,
        ).prepare_local_shard_infos(self.recv_param_key_n_rank_list, self.global_rank)

        self.all_rank_local_shard_infos = dist_util.all_gather_object_cpu(
            local_shard_infos
        )
        all_param_groups = dist_util.all_gather_object_cpu(param_groups)
        merged_groups = {}
        for r, param_groups in enumerate(all_param_groups):
            if self.parallel_dims.get_rank_in_dim("dp_cp_tp", r) != 0:
                continue
            for group in param_groups:
                group = sorted(group)
                key = tuple(group)
                if key not in merged_groups:
                    merged_groups[key] = group
        sorted_params_all_rank = dist_util.all_gather_object_cpu(
            [x[0] for x in self.recv_param_key_n_rank_list]
        )
        sorted_params_all_rank = [
            x
            for r, x in enumerate(sorted_params_all_rank)
            if self.parallel_dims.get_rank_in_dim("dp_cp_tp", r) == 0
        ]
        if self.global_rank == 0:
            body = {
                "shard_infos": self.all_rank_local_shard_infos,
                "param_groups": list(merged_groups.values()),
                "sorted_params": sorted_params_all_rank,
            }
            data = msgpack.packb(body)
            try:
                make_request_with_retry(
                    partial(
                        requests.post,
                        data=data,
                        headers={"Content-Type": "application/msgpack"},
                    ),
                    self.get_alternative_urls(COSMOS_API_ROLLOUT_SHARD_INFOS_SUFFIX),
                    max_retries=constant.COSMOS_HTTP_RETRY_CONFIG.max_retries,
                )
            except Exception as e:
                raise RuntimeError(
                    f"[Rollout] Failed in post shard infos to controller after retries {e}."
                )

    def work(self):
        pass
