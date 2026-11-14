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

from typing import Any, Dict, List, Optional
from cosmos_rl.dispatcher.algo.base import RuleBasedAlgo
from cosmos_rl.utils.logging import logger
from cosmos_rl.dispatcher.data.schema import RLPayload, Rollout
from cosmos_rl.reward.admission import (
    CompletionAdmission,
    resolve_completion_admission,
)


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
        self.completion_admission: Optional[CompletionAdmission] = None
        self.all_reward_metrics: List[Dict[str, Any]] = []
        self.all_rewards: List[float] = []
        self.all_filter_rewards: List[float] = []
        self.excluded_reward_metrics: Dict[int, Dict[str, Any]] = {}

    def compute_rollouts(
        self,
        algo: RuleBasedAlgo,
        *,
        apply_completion_admission: bool = True,
    ) -> List[Rollout]:
        """
        Compute rewards and advantages for the rollouts in the group.
        Args:
            algo (RuleBasedAlgo): The reward algorithm to compute rewards and advantages.
        Returns:
            List[Rollout]: List of Rollout with rewards and advantages.
        """
        assert self.reference_answer is not None, (
            "[RolloutGroup] Reference answer is not provided"
        )
        assert (
            self.payload.completions is not None and len(self.payload.completions) > 0
        ), (
            "[RolloutGroup] Completions are not provided correctly, please check the `rollout_generation` to make sure its returned `RolloutResult.completions` has a length of the number of generated samples."
        )

        # completion can be any objects such as tensors and videos in tensor native or video modes,
        # so that reward functions can compute reward directly from tensors or videos
        rewards = algo.compute_reward(
            self.payload.completions,
            self.reference_answer,
            prompt=self.payload.prompt,
        )
        assert (
            len(rewards[0])
            == len(rewards[1])
            == len(rewards[2])
            == len(self.payload.completions)
        ), (
            "[RolloutGroup] The length of rewards, filter_rewards, reward_metrics should be the same as the length of completions"
        )
        logger.debug(f"[RolloutGroup] Rewards: {rewards}")

        admission = resolve_completion_admission(
            self.payload,
            algo.minimum_trainable_completions,
            enabled=apply_completion_admission,
        )
        self.completion_admission = admission
        self.all_reward_metrics = rewards[2]
        self.all_rewards = rewards[0]
        self.all_filter_rewards = rewards[1]
        training_excluded_indices = (
            range(admission.original_size)
            if admission.group_excluded
            else admission.excluded_indices
        )
        self.excluded_reward_metrics = {
            idx: rewards[2][idx] for idx in training_excluded_indices
        }

        if admission.excluded_indices:
            logger.debug(
                "[CompletionAdmission] Excluding completions from prompt_idx=%s: "
                "indices=%s reasons=%s reward_metrics=%s",
                self.prompt_idx,
                admission.excluded_indices,
                admission.drop_reason_counts,
                self.excluded_reward_metrics,
            )

        if admission.group_excluded:
            logger.info(
                "[CompletionAdmission] Excluding prompt_idx=%s group from training: "
                "original_size=%d eligible_size=%d minimum=%d excluded=%d reasons=%s",
                self.prompt_idx,
                admission.original_size,
                admission.eligible_size,
                admission.minimum_trainable_completions,
                len(admission.excluded_indices),
                admission.drop_reason_counts,
            )
            return []

        eligible_rewards = [rewards[0][i] for i in admission.eligible_indices]
        advantages = algo.compute_advantage(eligible_rewards)
        assert len(advantages) == admission.eligible_size, (
            "[RolloutGroup] The length of advantages should be the same as the "
            "number of admitted completions"
        )
        logger.debug(f"[RolloutGroup] Advantages: {advantages}")

        if self.payload.cumulative_logprob is not None:
            # Find the best reward and cumulative logprob from the group by the cumulative logprob
            # We need calculate the most likely mode reward which is the reward of the completion
            # with the highest cumulative logprob and highest probability
            assert len(self.payload.cumulative_logprob) == len(rewards[0]), (
                "[RolloutGroup] The length of cumulative_logprob should be the same as the length of completions"
            )
            best_reward = None
            best_cumulative_logprob = None
            for i, reward in enumerate(rewards[0]):
                if self.payload.cumulative_logprob[i] is None:
                    continue
                if (
                    best_cumulative_logprob is None
                    or self.payload.cumulative_logprob[i] > best_cumulative_logprob
                ):
                    best_reward = reward
                    best_cumulative_logprob = self.payload.cumulative_logprob[i]
            if best_reward is not None:
                # Only assign the best reward to the first admitted rollout in the group.
                first_eligible = admission.eligible_indices[0]
                rewards[2][first_eligible]["most_likely_mode_reward_mean"] = best_reward
                rewards[2][first_eligible]["most_likely_mode_reward_count"] = 1

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

        rollouts = []
        for index, advantage in zip(admission.eligible_indices, advantages):
            rollouts.append(
                Rollout(
                    prompt=self.payload.prompt,
                    conversation=self.payload.conversation,
                    completion=self.payload.completions[index],
                    completed_conversation=completed_conversations[index],
                    is_end=self.is_end,
                    reward=rewards[0][index],
                    advantage=advantage,
                    prompt_idx=self.payload.prompt_idx,
                    filter_reward=rewards[1][index],
                    completion_logprobs=self.payload.completion_logprobs[index],
                    completion_token_ids=self.payload.completion_token_ids[index],
                    report_metrics=rewards[2][index],
                )
            )
        return rollouts


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
