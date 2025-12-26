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

from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.rollout.schema import RolloutResult


def update_payload_from_rollout_result(
    payload: RLPayload, rollout_result: RolloutResult, is_multi_turn: bool
) -> RLPayload:
    payload.completions = rollout_result.completions
    payload.completion_logprobs = rollout_result.completion_logprobs
    payload.completion_token_ids = rollout_result.completion_token_ids
    if is_multi_turn:
        payload.completed_conversations = rollout_result.completed_conversations
    return payload
