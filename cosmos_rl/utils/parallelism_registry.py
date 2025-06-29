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

from typing import Dict, Callable
from enum import Enum


class ParallelismStrategyRole(Enum):
    POLICY = 1
    ROLLOUT = 1 << 1
    ALL = POLICY | ROLLOUT

    def __or__(self, other):
        return self.value | other.value

    def __and__(self, other):
        return self.value & other.value


_PARALLELISM_STRATEGY_REGISTRY: Dict[int, Dict[str, Callable]] = {
    ParallelismStrategyRole.POLICY: {},
    ParallelismStrategyRole.ROLLOUT: {},
}


def register_parallelism_strategy(
    reg_key: str,
    *,
    allow_override: bool = False,
    role: ParallelismStrategyRole = ParallelismStrategyRole.ALL,
):
    def decorator(func: Callable) -> Callable:
        if role & ParallelismStrategyRole.POLICY:
            if (
                not allow_override
                and reg_key
                in _PARALLELISM_STRATEGY_REGISTRY[ParallelismStrategyRole.POLICY]
            ):
                raise ValueError(f"Function '{reg_key}' is already registered.")
            _PARALLELISM_STRATEGY_REGISTRY[ParallelismStrategyRole.POLICY][reg_key] = (
                func
            )
        if role & ParallelismStrategyRole.ROLLOUT:
            if (
                not allow_override
                and reg_key
                in _PARALLELISM_STRATEGY_REGISTRY[ParallelismStrategyRole.ROLLOUT]
            ):
                raise ValueError(f"Function '{reg_key}' is already registered.")
            _PARALLELISM_STRATEGY_REGISTRY[ParallelismStrategyRole.ROLLOUT][reg_key] = (
                func
            )
        return func

    return decorator


def query_parallelism_strategy(
    model_name: str, role: ParallelismStrategyRole
) -> Callable:
    return _PARALLELISM_STRATEGY_REGISTRY[role][model_name]


def get_policy_parallelism_strategy(model_name: str) -> Callable:
    return query_parallelism_strategy(model_name, ParallelismStrategyRole.POLICY)


def get_rollout_parallelism_strategy(model_name: str) -> Callable:
    return query_parallelism_strategy(model_name, ParallelismStrategyRole.ROLLOUT)
