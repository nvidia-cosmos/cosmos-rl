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

from typing import List
from cosmos_rl.dispatcher.data.schema import Rollout, RLPayload


def populate_for_none_fields(payload: RLPayload):
    len_of_rewards = len(payload.rewards)
    if payload.filter_rewards is None:
        payload.filter_rewards = [0.0] * len_of_rewards
    if payload.completion_logprobs is None:
        payload.completion_logprobs = [[] for _ in range(len_of_rewards)]
    if payload.completion_token_ids is None:
        payload.completion_token_ids = [[] for _ in range(len_of_rewards)]
    if payload.report_metrics is None:
        payload.report_metrics = [{} for _ in range(len_of_rewards)]
    if payload.teacher_result_uuids is None:
        payload.teacher_result_uuids = [""] * len_of_rewards
    if payload.completions is None:
        payload.completions = [None] * len_of_rewards
    if payload.completed_conversations is None:
        payload.completed_conversations = [None] * len_of_rewards


def extract_rollouts(
    payloads: List[RLPayload],
    is_end: bool,
    is_validation: bool = False,
) -> List[List[Rollout]]:
    # Extract rollouts from payloads of request
    # The invalid rollouts have already been filtered out by the rollout worekrs if dyanmic sampling is enabled.

    rollouts_list: List[List[Rollout]] = []
    for _, payload in enumerate(payloads):
        if not is_validation:
            # if this func is called for validation, we don't need to check the length of `completions`.
            assert (
                len(payload.completions)
                == len(payload.completed_conversations)
                == len(payload.rewards)
                == len(payload.advantages)
                == len(payload.n_ignore_prefix_tokens)
            ), "Length of completions, completed_conversations, rewards, advantages and n_ignore_prefix_tokens must be the same"

        populate_for_none_fields(payload)

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
                completion_token_ids=completion_token_ids,
                completion_logprobs=completion_logprobs,
                weight_version=payload.weight_version,
                report_metrics=report_metrics,
                teacher_result_uuid=teacher_result_uuid,
                prompt_logprobs=payload.prompt_logprobs,
            )
            for completion, completed_conversation, reward, advantage, n_ignore_prefix_tokens, filter_reward, completion_token_ids, completion_logprobs, report_metrics, teacher_result_uuid in zip(
                payload.completions,
                payload.completed_conversations,
                payload.rewards,
                payload.advantages,
                payload.n_ignore_prefix_tokens,
                payload.filter_rewards,
                payload.completion_token_ids,
                payload.completion_logprobs,
                payload.report_metrics,
                payload.teacher_result_uuids,
            )
        ]
        assert all(
            rollout.prompt_idx >= 0 for rollout in rollouts
        ), "All rollouts should have a valid prompt index"

        rollouts_list.append(rollouts)
    return rollouts_list
