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
from cosmos_rl.dispatcher.algo.base import RuleBasedAlgo
from cosmos_rl.utils.logging import logger
from cosmos_rl.dispatcher.data.schema import RLPayload, Rollout


class RolloutGroup:
    """
    RolloutGroup is a data structure that contains the prompt and completions of a rollout.
    For MutliModal-LM, image/video/audio could be included in the extra_info.
    """

    def __init__(
        self,
        prompt_idx: int,
        payload: RLPayload,
        is_end: bool,
        reference_answer: str,
    ):
        self.prompt_idx: int = prompt_idx
        self.payload: RLPayload = payload
        self.is_end: bool = is_end
        self.reference_answer: str = reference_answer

    def compute_rollouts(self, algo: RuleBasedAlgo) -> List[Rollout]:
        """
        Compute rewards and advantages for the rollouts in the group.
        Args:
            algo (RuleBasedAlgo): The reward algorithm to compute rewards and advantages.
        Returns:
            List[Rollout]: List of Rollout with rewards and advantages.
        """
        assert (
            self.reference_answer is not None
        ), "[RolloutGroup] Reference answer is not provided"
        assert (
            self.payload.completions is not None and len(self.payload.completions) > 0
        ), "[RolloutGroup] Completions are not provided correctly, please check the `rollout_generation` to make sure its returned `RolloutResult.completions` has a length of the number of generated samples."
        rewards = [
            # completion can be any objects such as tensors and videos in tensor native or video modes,
            # so that reward functions can compute reward directly from tensors or videos
            algo.compute_reward(
                completion,
                self.reference_answer,
                prompt=self.payload.prompt,
            )
            for completion in self.payload.completions
        ]
        logger.debug(f"[RolloutGroup] Rewards: {rewards}")
        advantages = algo.compute_advantage([r[0] for r in rewards])
        logger.debug(f"[RolloutGroup] Advantages: {advantages}")

        if self.payload.cumulative_logprob is not None:
            # Find the best reward and cumulative logprob from the group by the cumulative logprob
            # We need calculate the most likely mode reward which is the reward of the completion
            # with the highest cumulative logprob and highest probability
            assert (
                len(self.payload.cumulative_logprob) == len(rewards)
            ), "[RolloutGroup] The length of cumulative_logprob should be the same as the length of completions"
            best_reward = None
            best_cumulative_logprob = None
            for i, reward in enumerate(rewards):
                if self.payload.cumulative_logprob[i] is None:
                    continue
                if (
                    best_cumulative_logprob is None
                    or self.payload.cumulative_logprob[i] > best_cumulative_logprob
                ):
                    best_reward = reward[0]
                    best_cumulative_logprob = self.payload.cumulative_logprob[i]
            if best_reward is not None:
                # Only assign the best reward to the first rollout in the group
                rewards[0][2]["most_likely_mode_reward_mean"] = best_reward
                rewards[0][2]["most_likely_mode_reward_count"] = 1

        # If the completed_conversations is not provided, we use None for all the rollouts
        if self.payload.completed_conversations is not None:
            completed_conversations = self.payload.completed_conversations
        else:
            completed_conversations = [[] for _ in range(len(self.payload.completions))]

        if self.payload.completion_logprobs is None:
            self.payload.completion_logprobs = [
                [] for _ in range(len(self.payload.completions))
            ]

        if self.payload.completion_token_ids is None:
            self.payload.completion_token_ids = [
                [] for _ in range(len(self.payload.completions))
            ]

        return [
            Rollout(
                prompt=self.payload.prompt,
                conversation=self.payload.conversation,
                completion=completion,
                completed_conversation=completed_conversation,
                is_end=self.is_end,
                reward=reward[0],
                advantage=advantage,
                prompt_idx=self.payload.prompt_idx,
                filter_reward=reward[1],
                completion_logprobs=logprobs,
                completion_token_ids=token_ids,
                report_metrics=reward[2],
            )
            for completion, completed_conversation, reward, advantage, logprobs, token_ids in zip(
                self.payload.completions,
                completed_conversations,
                rewards,
                advantages,
                self.payload.completion_logprobs,
                self.payload.completion_token_ids,
            )
        ]


class BatchedRolloutGroup:
    """
    Batched Wrapper of the RolloutGroup
    """

    def __init__(self):
        self.rollout_groups: List[RolloutGroup] = []

    def __len__(self):
        return len(self.rollout_groups)

    def __getitem__(self, idx: int) -> RolloutGroup:
        return self.rollout_groups[idx]

    def __setitem__(self, idx: int, rollout_group: RolloutGroup):
        self.rollout_groups[idx] = rollout_group

    def __delitem__(self, idx: int):
        del self.rollout_groups[idx]

    @classmethod
    def from_rollout_groups(
        cls, rollout_groups: List[RolloutGroup]
    ) -> "BatchedRolloutGroup":
        batched_rollout_group = cls()
        batched_rollout_group.rollout_groups = rollout_groups
        return batched_rollout_group
