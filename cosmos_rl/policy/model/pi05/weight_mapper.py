# Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from typing import List, Tuple
from transformers import AutoConfig

from cosmos_rl.policy.model.base import WeightMapper


class Pi05WeightMapper(WeightMapper):
    """
    Minimal weight mapper for PI05 models.

    PI05 only supports DDP and loads weights directly from HuggingFace,
    so this is a simple pass-through implementation with no weight splitting.
    """

    def __init__(self, hf_config: AutoConfig):
        super().__init__(hf_config)

    def rollout_split_local_key_n_param_to_hf_key_n_param(
        self, param_name: str, param: torch.Tensor
    ) -> List[Tuple[str, torch.Tensor]]:
        """No splitting - return param as-is."""
        return [(param_name, param)]

    def policy_decompose_param_1_to_n_for_sync(self, name):
        """No decomposition needed for DDP-only setup."""
        return []
