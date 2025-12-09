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


from typing import Dict, Type
from cosmos_rl_reward.utils.logging import logger


class RewardRegistry:
    """
    A registry to map reward names to their corresponding handler classes and virtual environments.
    """

    _REWARD_REGISTRY: Dict[str, Type] = {}
    _REWARD_VENV: Dict[str, str] = {}

    @classmethod
    def register_reward(cls, reward_cls: Type):
        reward_type = reward_cls.reward_name
        RewardRegistry._REWARD_REGISTRY[reward_type] = reward_cls

    @classmethod
    def register_reward_venv(cls, reward_type: str, venv: str):
        RewardRegistry._REWARD_VENV[reward_type] = venv

    @classmethod
    def get_reward_class(cls, reward_type: str) -> Type:
        if reward_type not in RewardRegistry._REWARD_REGISTRY:
            raise ValueError(f"Reward type {reward_type} is not registered.")
        return RewardRegistry._REWARD_REGISTRY[reward_type]

    @classmethod
    def get_reward_venv(cls, reward_type: str) -> str:
        if reward_type not in RewardRegistry._REWARD_VENV:
            # Default to system python if not specified
            logger.warning(
                f"Reward venv for {reward_type} is not registered. Defaulting to system python."
            )
            return "python"
        return RewardRegistry._REWARD_VENV[reward_type]

    @classmethod
    def register(
        x,
        *,
        allow_override: bool = False,
    ):
        def decorator(cls: Type) -> Type:
            reward_type = cls.reward_name

            if (
                not allow_override
                and reward_type in RewardRegistry._REWARD_REGISTRY
                and RewardRegistry._REWARD_REGISTRY[reward_type] != cls
            ):
                raise ValueError(f"Reward {reward_type} is already registered.")
            RewardRegistry.register_reward(
                cls,
            )
            return cls

        return decorator

    @classmethod
    def check_reward_type_supported(cls, reward_type: str) -> bool:
        return reward_type in RewardRegistry._REWARD_REGISTRY

    @classmethod
    def check_reward_venv_supported(cls, reward_type: str) -> bool:
        return reward_type in RewardRegistry._REWARD_VENV

    @classmethod
    def list_registered_rewards(cls):
        return list(RewardRegistry._REWARD_REGISTRY.keys())

    @classmethod
    def list_registered_reward_venvs(cls):
        return dict(RewardRegistry._REWARD_VENV)
