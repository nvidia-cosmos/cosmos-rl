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

from cosmos_rl.policy.model.base import WeightMapper


class DiffuserModelWeightMapper(WeightMapper):
    def __init__(self, diffusers_config):
        super().__init__(diffusers_config)

    def policy_map_local_key_to_hf_key(self, name: str) -> str:
        pass

    def rollout_map_local_key_to_hf_key(self, name: str) -> str:
        pass
