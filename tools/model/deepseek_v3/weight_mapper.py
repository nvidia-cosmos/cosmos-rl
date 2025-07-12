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
from typing import List, Tuple, Dict, Any
from cosmos_rl.policy.model.base import WeightMapper
from cosmos_rl.utils import util
from transformers import AutoConfig


class DeepseekV3MoEWeightMapper(WeightMapper):
    def __init__(self, hf_config: AutoConfig):
        super().__init__(hf_config)

    def rollout_prepare_recv(
        self,
        vllm_model: Any,
    ) -> Tuple[
        Dict[str, torch.Tensor],
        List[Tuple[str, torch.Size]],
    ]:
        raise NotImplementedError

    def policy_map_local_key_to_hf_key(self, name: str) -> str:
        name = util.clear_weight_name(name)
        if not name == "lm_head.weight":
            if not name.startswith("model."):
                name = "model." + name
        return name

    def get_policy_parallelism_strategy(self):
        raise NotImplementedError

    def get_rollout_parallelism_strategy(self):
        raise NotImplementedError
