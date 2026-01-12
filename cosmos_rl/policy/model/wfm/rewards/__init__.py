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

from cosmos_rl.policy.config.wfm import (
    RemoteRewardConfig,
    AestheticRewardConfig,
    ImageRewardConfig,
    FakeRewardConfig,
)
from cosmos_rl.policy.model.wfm.rewards.reward_models import (
    RemoteReward,
    AestheticReward,
    ImageReward,
    FakeReward,
)
from typing import Union


def get_reward_model(
    reward_config: Union[
        AestheticRewardConfig, ImageRewardConfig, RemoteRewardConfig, FakeRewardConfig
    ],
):
    if isinstance(reward_config, AestheticRewardConfig):
        return AestheticReward(reward_config)
    elif isinstance(reward_config, ImageRewardConfig):
        return ImageReward(reward_config)
    elif isinstance(reward_config, RemoteRewardConfig):
        return RemoteReward(reward_config)
    elif isinstance(reward_config, FakeRewardConfig):
        return FakeReward(reward_config)
    else:
        raise ValueError(
            f"Unsupported reward model configuration: {type(reward_config)}"
        )
