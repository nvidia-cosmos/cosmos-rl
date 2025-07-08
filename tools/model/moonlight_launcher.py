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

from cosmos_rl.launcher.worker_entry import main as launch_worker
from deepseek_v3 import DeepseekV3MoEModel
from deepseek_v3.weight_mapper import DeepseekV3MoEWeightMapper
from cosmos_rl.policy.model.base import ModelRegistry

if __name__ == "__main__":
    # Register the model into the registry
    ModelRegistry.register_model(
        # Model class to register
        DeepseekV3MoEModel,
        # Weight mapper for this model
        DeepseekV3MoEWeightMapper,
    )

    launch_worker()
