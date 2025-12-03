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
from cosmos_rl.dispatcher.algo.base import RuleBasedAlgo
from cosmos_rl.utils.logging import logger
from cosmos_rl.dispatcher.data.schema import RLPayload, Rollout
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from cosmos_rl.dispatcher.algo.base import REGISTERED_ALGOs
from cosmos_rl.dispatcher.algo.reward import Reward
from cosmos_rl.dispatcher.data.packer import BaseDataPacker
from cosmos_rl.policy.config import Config
import cosmos_rl.utils.constant as constant
import cosmos_rl.utils.util as util
from queue import Queue


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
            for i, completion in enumerate(self.payload.completions)
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


class RewardCalculator:
    """
    RewardCalculator is responsible for calculating the rewards for the rollouts.
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
        Setup the RewardCalculator with the given configuration and data packers.
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
                "[RewardCalculator] RewardCalculator is already setup, returning directly."
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
    def get_instance(cls) -> "RewardCalculator":
        """
        Get the singleton instance of the RewardCalculator.
        Returns:
            RewardCalculator: The singleton instance of the RewardCalculator.
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
            for _, payload in enumerate(payloads)
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
            for _, payload in enumerate(payloads)
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
                    rollout.completion_token_ids
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
                    )
                )
        return payload_list, False, step


class RewardDispatcher:
    """
    RewardDispatcher is responsible for dispatching the reward calculation tasks to the RewardCalculator.
    It uses a ProcessPoolExecutor to parallelize the reward calculation.
    It also uses a Queue to store the tasks and results.
    """

    def __init__(self, payload_per_task: int = 1):
        self.reward_calculator = RewardCalculator()
        self.task_queue = Queue()
        self.payload_per_task = payload_per_task

    def setup(
        self,
        config: Config,
        reward_fns: Optional[List[Callable]] = None,
        filter_reward_fns: Optional[List[Callable]] = None,
        val_reward_fns: Optional[List[Callable]] = None,
        data_packer: Optional[BaseDataPacker] = None,
        val_data_packer: Optional[BaseDataPacker] = None,
        num_workers: int = 2,
    ) -> None:
        """
        Setup the RewardCalculator with the given configuration and data packers.
        Args:
            config (Config): The configuration for the reward calculator.
            reward_fns (Optional[List[Callable]]): The list of reward functions for training.
            filter_reward_fns (Optional[List[Callable]]): The list of filter reward functions for dynamic sampling.
            val_reward_fns (Optional[List[Callable]]): The list of reward functions for validation.
            data_packer (Optional[BaseDataPacker]): The data packer for processing the payloads.
            val_data_packer (Optional[BaseDataPacker]): The data packer for processing the validation payloads.
            num_workers (int): The number of worker processes for parallel reward calculation.
        """

        def worker_init(
            config,
            reward_fns,
            filter_reward_fns,
            val_reward_fns,
            data_packer,
            val_data_packer,
        ):
            reward_calculator = RewardCalculator.get_instance()
            reward_calculator.setup(
                config=config,
                reward_fns=reward_fns,
                filter_reward_fns=filter_reward_fns,
                val_reward_fns=val_reward_fns,
                data_packer=data_packer,
                val_data_packer=val_data_packer,
            )

        worker_init(
            config,
            reward_fns,
            filter_reward_fns,
            val_reward_fns,
            data_packer,
            val_data_packer,
        )
        if num_workers > 0:
            # ThreadPoolExecutor is used here to avoid the overhead of ProcessPoolExecutor in non-text mode.
            # Unlike ProcessPoolExecutor, ThreadPoolExecutor can parse the tensors, videos, images directly
            executor = (
                ThreadPoolExecutor if config.train.non_text else ProcessPoolExecutor
            )
            self.executor = executor(
                max_workers=num_workers,
                initializer=worker_init,
                initargs=(
                    config,
                    reward_fns,
                    filter_reward_fns,
                    val_reward_fns,
                    data_packer,
                    val_data_packer,
                ),
            )
        else:
            self.executor = None

    @staticmethod
    def compute_rewards(payloads, is_validation, step):
        """
        Static method to compute rewards using the singleton RewardCalculator instance.
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
        reward_calculator = RewardCalculator.get_instance()
        return reward_calculator.compute_rewards(payloads, is_validation, step)

    def enqueue_rewards_cal(
        self,
        payloads: List[RLPayload],
        is_validation: bool,
        step: int,
    ) -> None:
        """
        Enqueue the reward calculation task.
        The task will be executed in
        a separate process and the result will be stored in the task queue.
        Args:
            payloads (List[RLPayload]): List of RLPayload to compute rewards for.
            is_validation (bool): Whether the payloads are from validation set.
            step (int): The weight step where the payloads are generated.
        """
        for i in range(0, len(payloads), self.payload_per_task):
            self.task_queue.put(
                self.executor.submit(
                    RewardDispatcher.compute_rewards,
                    payloads[i : i + self.payload_per_task],
                    is_validation,
                    step,
                )
            )

    def dequeue_rewards_cal(
        self,
    ) -> Tuple[Optional[List[RLPayload]], bool, int, bool]:
        """
        Dequeue the reward calculation result.
        If the task queue is empty, return None.
        If the task is not done, return None.
        If the task is done, return the result.
        If the task queue is empty and all tasks are done, return None and True.

        Returns:
            Tuple[List[RLPayload], bool, int, bool]: (payloads, is_validation, step, all_done)
                payloads: List of RLPayload with rewards and advantages
                is_validation: whether the payloads are from validation set
                step: the weight step where the payloads are generated
                all_done: whether all pending tasks are done
        """
        if not self.task_queue.empty():
            if self.task_queue.queue[0].done():
                payloads, is_validation, step = self.task_queue.get().result()
                return payloads, is_validation, step, False
            else:
                return None, False, -1, False
        else:
            return None, False, -1, True

    def is_empty(self) -> bool:
        """
        Check if the task queue is empty.
        Returns:
            True if the task queue is empty, False otherwise.
        """
        return self.task_queue.empty()
