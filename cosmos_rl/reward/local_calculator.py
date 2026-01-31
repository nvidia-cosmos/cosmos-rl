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

from typing import List, Dict, Optional, Callable, Tuple
from cosmos_rl.utils.logging import logger
from cosmos_rl.dispatcher.data.schema import RLPayload, Rollout
from cosmos_rl.dispatcher.algo.base import REGISTERED_ALGOs
from cosmos_rl.dispatcher.algo.reward import Reward
from cosmos_rl.dispatcher.data.packer import BaseDataPacker
from cosmos_rl.policy.config import Config
from cosmos_rl.reward.base import RolloutGroup
import cosmos_rl.utils.constant as constant
import cosmos_rl.utils.util as util


class LocalRewardCalculator:
    """
    LocalRewardCalculator is responsible for calculating the rewards for the rollouts locally.
    It adds rewards and advantages to the RLPayload.
    It supports dynamic sampling to filter out rollouts that have the same filter rewards with valid=False.
    It also supports finding shared prefix among rollouts and ignore the prefix tokens during training.
    """

    def setup(
        self,
        config: Config,
        reward_fns: Optional[List[Callable]] = None,
        filter_reward_fns: Optional[List[Callable]] = None,
        val_reward_fns: Optional[List[Callable]] = None,
        data_packer: Optional[BaseDataPacker] = None,
        val_data_packer: Optional[BaseDataPacker] = None,
    ) -> None:
        """
        Setup the LocalRewardCalculator with the given configuration and data packers.
        Args:
            config (Config): The configuration for the reward calculator.
            reward_fns (Optional[List[Callable]]): The list of reward functions for training.
            filter_reward_fns (Optional[List[Callable]]): The list of filter reward functions for dynamic sampling.
            val_reward_fns (Optional[List[Callable]]): The list of reward functions for validation.
            data_packer (Optional[BaseDataPacker]): The data packer for processing the payloads.
            val_data_packer (Optional[BaseDataPacker]): The data packer for processing the validation payloads.
        """
        if hasattr(self, "rl_algo"):
            logger.warning(
                "[LocalRewardCalculator] LocalRewardCalculator is already setup, returning directly."
            )
            return
        self.config = config
        self.tokenizer = util.setup_tokenizer(self.config.policy.model_name_or_path)

        self.rl_algo = REGISTERED_ALGOs[constant.Algo.GRPO](
            reward_fn=Reward(
                config=config,
                tokenier=self.tokenizer,
                reward_function=config.train.train_policy.reward_function,
                explicit_reward_fn=reward_fns,
                explicit_filter_reward_fn=filter_reward_fns,
                data_packer=data_packer,
            ),
            unbiased=config.train.train_policy.unbiased_advantage,
        )
        if config.validation.enable:
            if not config.validation.reward_function:
                if val_reward_fns is None:
                    val_reward_fns = reward_fns
                    if val_reward_fns is not None:
                        logger.info(
                            "[Reward] No validation reward functions provided, using the same reward functions as training."
                        )
                config.validation.reward_function = (
                    config.train.train_policy.reward_function
                )
                logger.info(
                    "[Reward] No validation reward function config specified, using the same reward function as training."
                )
            self.val_rl_algo = REGISTERED_ALGOs[constant.Algo.GRPO](
                reward_fn=Reward(
                    config=config,
                    tokenier=self.tokenizer,
                    reward_function=config.validation.reward_function,
                    explicit_reward_fn=val_reward_fns,
                    data_packer=val_data_packer,
                )
            )

    @classmethod
    def get_instance(cls) -> "LocalRewardCalculator":
        """
        Get the singleton instance of the LocalRewardCalculator.
        Returns:
            LocalRewardCalculator: The singleton instance of the LocalRewardCalculator.
        """
        if not hasattr(cls, "_instance"):
            cls._instance = cls()
        return cls._instance

    def compute_validation_rewards(
        self,
        payloads: List[RLPayload],
        step: int,
    ) -> Tuple[List[RLPayload], bool, int]:
        """
        Compute rewards and advantages for the given payloads using validation reward function.
        Args:
            payloads (List[RLPayload]): List of RLPayload to compute rewards for.
            step (int): The weight step where the payloads are generated.
        Returns:
            Tuple[List[RLPayload], bool, int]: (payloads, is_validation, step)
                payloads: List of RLPayload with rewards and advantages
                is_validation: whether the payloads are from validation set (always True)
                step: the weight step where the payloads are generated
        """

        assert all(
            payload.prompt_idx >= 0 for payload in payloads
        ), "[Reward] All payloads should have a valid prompt index"
        rollout_groups: List[RolloutGroup] = [
            RolloutGroup(
                prompt_idx=payload.prompt_idx,
                payload=payload,
                # Only report once per replica, so is_end is always True
                is_end=True,
                reference_answer=payload.reference_answer,
            )
            for payload in payloads
        ]

        rollouts_list: List[List[Rollout]] = [
            rollout_group.compute_rollouts(self.val_rl_algo)
            for rollout_group in rollout_groups
        ]
        payload_list: List[RLPayload] = []
        # Dynamic Sampling: Filter out the rollouts that the rewards are all the same
        for idx, rollouts_group in enumerate(rollouts_list):
            payload_list.append(
                RLPayload(
                    prompt=rollouts_group[0].prompt,
                    prompt_idx=rollouts_group[0].prompt_idx,
                    conversation=rollouts_group[0].conversation,
                    completions=[rollout.completion for rollout in rollouts_group],
                    completed_conversations=[
                        rollout.completed_conversation for rollout in rollouts_group
                    ],
                    reference_answer=None,
                    n_ignore_prefix_tokens=[
                        rollout.n_ignore_prefix_tokens for rollout in rollouts_group
                    ],
                    rewards=[rollout.reward for rollout in rollouts_group],
                    advantages=[rollout.advantage for rollout in rollouts_group],
                    filter_rewards=[
                        rollout.filter_reward for rollout in rollouts_group
                    ],
                    valid=True,
                    weight_version=payloads[idx].weight_version,
                    report_metrics=[
                        rollout.report_metrics
                        if rollout.report_metrics is not None
                        else {}
                        for rollout in rollouts_group
                    ],
                    cumulative_logprob=payloads[idx].cumulative_logprob,
                    teacher_result_uuids=payloads[idx].teacher_result_uuids,
                    prompt_logprobs=payloads[idx].prompt_logprobs,
                    prompt_token_ids=payloads[idx].prompt_token_ids,
                )
            )
        return payload_list, True, step

    def compute_rewards(
        self,
        payloads: List[RLPayload],
        is_validation: bool,
        step: int,
    ) -> Tuple[List[RLPayload], bool, int]:
        """
        Compute rewards and advantages for the given payloads.
        If is_validation is True, use the validation reward function and return all rollouts.
        If is_validation is False, use the training reward function and apply dynamic sampling.
        Args:
            payloads (List[RLPayload]): List of RLPayload to compute rewards for.
            is_validation (bool): Whether the payloads are from validation set.
            step (int): The weight step where the payloads are generated.
        Returns:
            Tuple[List[RLPayload], bool, int]: (payloads, is_validation, step)
                payloads: List of RLPayload with rewards and advantages
                is_validation: whether the payloads are from validation set
                step: the weight step where the payloads are generated
        """

        if is_validation:
            return self.compute_validation_rewards(payloads, step)

        assert all(
            payload.prompt_idx >= 0 for payload in payloads
        ), "[Reward] All payloads should have a valid prompt index"
        # Placeholder for advantage computation logic
        rollout_groups: List[RolloutGroup] = [
            RolloutGroup(
                prompt_idx=payload.prompt_idx,
                payload=payload,
                is_end=False,
                reference_answer=payload.reference_answer,
            )
            for payload in payloads
        ]

        rollouts_list: List[List[Rollout]] = [
            rollout_group.compute_rollouts(self.rl_algo)
            for rollout_group in rollout_groups
        ]
        payload_list: List[RLPayload] = []
        # Dynamic Sampling: Filter out the rollouts that the rewards are all the same
        for idx, rollouts_group in enumerate(rollouts_list):
            if self.config.train.non_text:
                rollout_tokens = []
            else:
                rollout_tokens = [
                    [t[0] for t in rollout.completion_token_ids]
                    if self.config.train.train_policy.rollout_as_token_ids
                    else self.tokenizer(
                        rollout.completion, add_special_tokens=False
                    ).input_ids
                    for rollout in rollouts_group
                ]
            # Only filter_reward is considered for dynamic sampling
            if len(set([rollout.filter_reward for rollout in rollouts_group])) > 1:
                # Preprocess the valid rollouts to find if shared prefix exists
                # If exists,
                #   - if the shared prefix hold different rewards, the prefix may lead to bias
                #   - else: do nothing
                # (shared_prefix) -> index of rollouts
                shared_prefix_groups: Dict[Tuple[int, ...], List[int]] = (
                    util.find_maximal_prefix_groups(
                        rollout_tokens,
                        N=self.config.train.train_policy.min_filter_prefix_tokens,
                    )
                )
                for shared_prefix, rollout_indices in shared_prefix_groups.items():
                    assert (
                        len(rollout_indices) > 1
                    ), "Shared prefix group should not be empty"
                    # Check if the shared prefix holds different rewards
                    rewards = [rollouts_group[i].reward for i in rollout_indices]
                    if len(set(rewards)) > 1:
                        n_ignore_prefix_tokens = len(shared_prefix)
                        if shared_prefix[-1] == self.tokenizer.eos_token_id:
                            shared_prefix = shared_prefix[:-1]
                        prefix_str = self.tokenizer.decode(shared_prefix)
                        for rollout_index in rollout_indices:
                            # Only do this if shared_prefix != rollout.completion
                            # Else the whole sample will be ignored, which cause training issues.
                            if prefix_str != rollouts_group[rollout_index].completion:
                                rollouts_group[
                                    rollout_index
                                ].n_ignore_prefix_tokens = n_ignore_prefix_tokens

                payload_list.append(
                    RLPayload(
                        prompt=rollouts_group[0].prompt,
                        prompt_idx=rollouts_group[0].prompt_idx,
                        conversation=rollouts_group[0].conversation,
                        completions=[rollout.completion for rollout in rollouts_group],
                        completed_conversations=[
                            rollout.completed_conversation for rollout in rollouts_group
                        ],
                        reference_answer=None,
                        n_ignore_prefix_tokens=[
                            rollout.n_ignore_prefix_tokens for rollout in rollouts_group
                        ],
                        rewards=[rollout.reward for rollout in rollouts_group],
                        filter_rewards=[
                            rollout.filter_reward for rollout in rollouts_group
                        ],
                        advantages=[rollout.advantage for rollout in rollouts_group],
                        valid=True,
                        completion_logprobs=[
                            rollout.completion_logprobs
                            if rollout.completion_logprobs is not None
                            else []
                            for rollout in rollouts_group
                        ],
                        completion_token_ids=[
                            rollout.completion_token_ids
                            if rollout.completion_token_ids is not None
                            else []
                            for rollout in rollouts_group
                        ],
                        weight_version=payloads[idx].weight_version,
                        report_metrics=[
                            rollout.report_metrics
                            if rollout.report_metrics is not None
                            else {}
                            for rollout in rollouts_group
                        ],
                        cumulative_logprob=payloads[idx].cumulative_logprob,
                        teacher_result_uuids=payloads[idx].teacher_result_uuids,
                        prompt_logprobs=payloads[idx].prompt_logprobs,
                        prompt_token_ids=payloads[idx].prompt_token_ids,
                    )
                )
            else:
                # If the rewards are all the same, we need to sample one rollout from the group
                payload_list.append(
                    RLPayload(
                        prompt=rollouts_group[0].prompt,
                        prompt_idx=rollouts_group[0].prompt_idx,
                        conversation=rollouts_group[0].conversation,
                        completions=[rollout.completion for rollout in rollouts_group],
                        completed_conversations=[
                            rollout.completed_conversation for rollout in rollouts_group
                        ],
                        reference_answer=None,
                        n_ignore_prefix_tokens=[
                            rollout.n_ignore_prefix_tokens for rollout in rollouts_group
                        ],
                        rewards=[rollout.reward for rollout in rollouts_group],
                        filter_rewards=[
                            rollout.filter_reward for rollout in rollouts_group
                        ],
                        advantages=[rollout.advantage for rollout in rollouts_group],
                        valid=False,
                        completion_logprobs=[
                            rollout.completion_logprobs
                            if rollout.completion_logprobs is not None
                            else []
                            for rollout in rollouts_group
                        ],
                        completion_token_ids=[
                            rollout.completion_token_ids
                            if rollout.completion_token_ids is not None
                            else []
                            for rollout in rollouts_group
                        ],
                        weight_version=payloads[idx].weight_version,
                        report_metrics=[
                            rollout.report_metrics
                            if rollout.report_metrics is not None
                            else {}
                            for rollout in rollouts_group
                        ],
                        cumulative_logprob=payloads[idx].cumulative_logprob,
                        teacher_result_uuids=payloads[idx].teacher_result_uuids,
                        prompt_logprobs=payloads[idx].prompt_logprobs,
                        prompt_token_ids=payloads[idx].prompt_token_ids,
                    )
                )
        return payload_list, False, step
