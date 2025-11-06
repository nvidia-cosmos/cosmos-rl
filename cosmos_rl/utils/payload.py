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

from typing import List, Tuple
from cosmos_rl.dispatcher.data.schema import Rollout, RLPayload


def extract_rollouts(
    payloads: List[RLPayload], is_end: bool, prompt_idxs: List[int] = []
) -> Tuple[List[List[Rollout]], List[List[Rollout]]]:
    # Extract rollouts from payloads of request
    # Separate valid and invalid rollouts for Dynamic Sampling
    # Dynamic Sampling: Filter out the rollouts that the rewards are all the same
    valid_rollouts_list: List[List[Rollout]] = []
    invalid_rollouts_list: List[List[Rollout]] = []
    for idx, payload in enumerate(payloads):
        assert (
            len(payload.completions)
            == len(payload.completed_conversations)
            == len(payload.rewards)
            == len(payload.advantages)
            == len(payload.n_ignore_prefix_tokens)
        ), "Length of completions, completed_conversations, rewards, advantages and n_ignore_prefix_tokens must be the same"
        if payload.filter_rewards is None:
            payload.filter_rewards = [0.0] * len(payload.rewards)
        if payload.completions_token_length is None:
            payload.completions_token_length = [0] * len(payload.rewards)

        rollouts = [
            Rollout(
                prompt=payload.prompt,
                prompt_idx=payload.prompt_idx,
                conversation=payload.conversation,
                completion=completion,
                completed_conversation=completed_conversation,
                is_end=is_end,
                reward=reward,
                filter_reward=filter_reward,
                advantage=advantage,
                n_ignore_prefix_tokens=n_ignore_prefix_tokens,
                completion_token_length=completion_token_length,
            )
            for completion, completed_conversation, reward, advantage, n_ignore_prefix_tokens, filter_reward, completion_token_length in zip(
                payload.completions,
                payload.completed_conversations,
                payload.rewards,
                payload.advantages,
                payload.n_ignore_prefix_tokens,
                payload.filter_rewards,
                payload.completions_token_length,
            )
        ]
        assert all(
            rollout.prompt_idx >= 0 for rollout in rollouts
        ), "All rollouts should have a valid prompt index"
        if payload.valid:
            valid_rollouts_list.append(rollouts)
        else:
            invalid_rollouts_list.append(rollouts)
    return valid_rollouts_list, invalid_rollouts_list
