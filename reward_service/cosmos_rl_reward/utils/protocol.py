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

"""
Protocol constants for the cosmos_rl_reward package.
Defines API endpoint suffixes used for interacting with the reward service.
"""

COSMOS_RL_REWARD_REDIS_PUSH_API_SUFFIX = "/api/reward/push"
COSMOS_RL_REWARD_REDIS_PULL_API_SUFFIX = "/api/reward/pull"
COSMOS_RL_REWARD_PING_API_SUFFIX = "/api/reward/ping"
COSMOS_RL_REWARD_LATENT_ATTR_API_SUFFIX = "/api/reward/latent_attr"
COSMOS_RL_REWARD_ENQUEUE_API_SUFFIX = "/api/reward/enqueue"
