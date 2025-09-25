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

from typing import List, Dict, Any, Optional, Callable, Tuple
from cosmos_rl.dispatcher.algo.base import RuleBasedAlgo
from cosmos_rl.utils.logging import logger
from cosmos_rl.dispatcher.data.schema import RLPayload, Rollout
from concurrent.futures import ProcessPoolExecutor
from torch.utils.data import Dataset
from cosmos_rl.dispatcher.algo.base import REGISTERED_ALGOs
from cosmos_rl.dispatcher.algo.reward import Reward
from cosmos_rl.dispatcher.data import (
    CosmosDataset,
    CosmosValidationDataset,
)
from cosmos_rl.dispatcher.data.packer import DataPacker
from cosmos_rl.policy.config import Config
import cosmos_rl.utils.constant as constant
from transformers import AutoTokenizer
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
        rewards = [
            algo.compute_reward(
                completion, self.reference_answer, prompt=self.payload.prompt
            )
            for completion in self.payload.completions
        ]
        logger.debug(f"[RolloutGroup] Rewards: {rewards}")
        advantages = algo.compute_advantage([r[0] for r in rewards])
        logger.debug(f"[RolloutGroup] Advantages: {advantages}")

        # If the completed_conversations is not provided, we use None for all the rollouts
        if self.payload.completed_conversations is not None:
            completed_conversations = self.payload.completed_conversations
        else:
            completed_conversations = [[]] * len(self.payload.completions)

        return [
            Rollout(
                prompt=self.payload.prompt,
                conversation=self.payload.conversation,
                completion=completion,
                completed_conversation=completed_conversation,
                is_end=self.is_end,
                reward=reward[0],
                advantage=advantage,
                prompt_idx=self.prompt_idx,
                filter_reward=reward[1],
            )
            for completion, completed_conversation, reward, advantage in zip(
                self.payload.completions, completed_conversations, rewards, advantages
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
        dataset: Optional[Dataset] = None,
        reward_fns: Optional[List[Callable]] = None,
        filter_reward_fns: Optional[List[Callable]] = None,
        val_dataset: Optional[Dataset] = None,
        val_reward_fns: Optional[List[Callable]] = None,
        data_packer: Optional[DataPacker] = None,
        val_data_packer: Optional[DataPacker] = None,
    ) -> None:
        """
        Setup the RewardCalculator with the given configuration and datasets.
        Args:
            config (Config): The configuration for the reward calculator.
            dataset (Optional[Dataset]): The training dataset.
            reward_fns (Optional[List[Callable]]): The list of reward functions for training.
            filter_reward_fns (Optional[List[Callable]]): The list of filter reward functions for dynamic sampling.
            val_dataset (Optional[Dataset]): The validation dataset.
            val_reward_fns (Optional[List[Callable]]): The list of reward functions for validation.
            data_packer (Optional[DataPacker]): The data packer for processing the payloads.
            val_data_packer (Optional[DataPacker]): The data packer for processing the validation payloads.
        """
        self.config = config
        self.tokenizer = util.retry(AutoTokenizer.from_pretrained)(
            self.config.policy.model_name_or_path
        )

        if config.rollout.reference_answer_in_local:
            if dataset is not None and isinstance(dataset, Callable):
                dataset = dataset(config)
            if val_dataset is not None and isinstance(val_dataset, Callable):
                val_dataset = val_dataset(config)

            if dataset is not None:
                assert isinstance(dataset, Dataset)
                self.dataset = CosmosDataset(
                    config=config, train_set=dataset, tokenizer=self.tokenizer
                )
                logger.info(
                    "[Reward] Using provided dataset for training, dataset specification in the toml config will be ignored"
                )
            else:
                self.dataset = CosmosDataset(config=config, tokenizer=self.tokenizer)
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
            if config.rollout.reference_answer_in_local:
                if val_dataset is not None:
                    assert isinstance(val_dataset, Dataset)
                    self.val_dataset = CosmosValidationDataset(
                        config=config, val_set=val_dataset, tokenizer=self.tokenizer
                    )
                    logger.info(
                        "[Reward] Using provided validation dataset for validation, dataset specification in the toml config will be ignored"
                    )
                else:
                    self.val_dataset = CosmosValidationDataset(
                        config=config, tokenizer=self.tokenizer
                    )
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

    def query_reference_answer(
        self, prompt_idx: int, dataset_type: str = "train"
    ) -> Any:
        """
        Query the reference answer from the dataset based on the prompt index.
        Args:
            prompt_idx (int): The index of the prompt in the dataset.
            dataset_type (str): The type of the dataset, either "train" or "val".
        Returns:
            Any: The reference answer corresponding to the prompt index.
        """
        if dataset_type == "train":
            return self.dataset.train_set.get_reference_answer(prompt_idx)
        elif dataset_type == "val":
            return self.val_dataset.val_set.get_reference_answer(prompt_idx)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    def compute_validation_rewards(
        self,
        payloads: List[RLPayload],
        step: int,
        prompt_idxs: List[int] = [],
    ) -> Tuple[List[RLPayload], bool, int]:
        """
        Compute rewards and advantages for the given payloads using validation reward function.
        Args:
            payloads (List[RLPayload]): List of RLPayload to compute rewards for.
            step (int): The weight step where the payloads are generated.
            prompt_idxs (List[int]): List of prompt indices corresponding to the payloads.
        Returns:
            Tuple[List[RLPayload], bool, int]: (payloads, is_validation, step)
                payloads: List of RLPayload with rewards and advantages
                is_validation: whether the payloads are from validation set (always True)
                step: the weight step where the payloads are generated
        """

        assert (
            not self.config.rollout.reference_answer_in_local
            or len(prompt_idxs) == len(payloads)
        ), "[Reward] prompt_idxs length should match payloads length when reference_answer_in_local is True"
        rollout_groups: List[RolloutGroup] = [
            RolloutGroup(
                prompt_idx=prompt_idxs[i] if len(prompt_idxs) == len(payloads) else -1,
                payload=payload,
                # Only report once per replica, so is_end is always True
                is_end=True,
                reference_answer=payload.reference_answer
                if not self.config.rollout.reference_answer_in_local
                else self.query_reference_answer(prompt_idxs[i], "val"),
            )
            for i, payload in enumerate(payloads)
        ]

        rollouts_list: List[List[Rollout]] = [
            rollout_group.compute_rollouts(self.val_rl_algo)
            for rollout_group in rollout_groups
        ]
        payload_list: List[RLPayload] = []
        # Dynamic Sampling: Filter out the rollouts that the rewards are all the same
        for rollouts_group in rollouts_list:
            payload_list.append(
                RLPayload(
                    prompt=rollouts_group[0].prompt,
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
                    valid=True,
                )
            )
        return payload_list, True, step

    def compute_rewards(
        self,
        payloads: List[RLPayload],
        is_validation: bool,
        step: int,
        prompt_idxs: List[int] = [],
    ) -> Tuple[List[RLPayload], bool, int]:
        """
        Compute rewards and advantages for the given payloads.
        If is_validation is True, use the validation reward function and return all rollouts.
        If is_validation is False, use the training reward function and apply dynamic sampling.
        Args:
            payloads (List[RLPayload]): List of RLPayload to compute rewards for.
            is_validation (bool): Whether the payloads are from validation set.
            step (int): The weight step where the payloads are generated.
            prompt_idxs (List[int]): List of prompt indices corresponding to the payloads.
        Returns:
            Tuple[List[RLPayload], bool, int]: (payloads, is_validation, step)
                payloads: List of RLPayload with rewards and advantages
                is_validation: whether the payloads are from validation set
                step: the weight step where the payloads are generated
        """

        if is_validation:
            return self.compute_validation_rewards(payloads, step, prompt_idxs)

        assert (
            not self.config.rollout.reference_answer_in_local
            or len(prompt_idxs) == len(payloads)
        ), "[Reward] prompt_idxs length should match payloads length when reference_answer_in_local is True"
        # Placeholder for advantage computation logic
        rollout_groups: List[RolloutGroup] = [
            RolloutGroup(
                prompt_idx=prompt_idxs[i] if len(prompt_idxs) == len(payloads) else -1,
                payload=payload,
                is_end=False,
                reference_answer=payload.reference_answer
                if not self.config.rollout.reference_answer_in_local
                else self.query_reference_answer(prompt_idxs[i]),
            )
            for i, payload in enumerate(payloads)
        ]

        rollouts_list: List[List[Rollout]] = [
            rollout_group.compute_rollouts(self.rl_algo)
            for rollout_group in rollout_groups
        ]
        payload_list: List[RLPayload] = []
        # Dynamic Sampling: Filter out the rollouts that the rewards are all the same
        for rollouts_group in rollouts_list:
            # Only filter_reward is considered for dynamic sampling
            if len(set([rollout.filter_reward for rollout in rollouts_group])) > 1:
                # Preprocess the valid rollouts to find if shared prefix exists
                # If exists,
                #   - if the shared prefix hold different rewards, the prefix may lead to bias
                #   - else: do nothing
                # (shared_prefix) -> index of rollouts
                shared_prefix_groups: Dict[Tuple[int, ...], List[int]] = (
                    util.find_maximal_prefix_groups(
                        [
                            self.tokenizer(
                                rollout.completion, add_special_tokens=False
                            ).input_ids
                            for rollout in rollouts_group
                        ],
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
                        valid=True,
                    )
                )
            else:
                # If the rewards are all the same, we need to sample one rollout from the group
                payload_list.append(
                    RLPayload(
                        prompt=rollouts_group[0].prompt,
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
                        valid=False,
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
        dataset: Optional[Dataset] = None,
        reward_fns: Optional[List[Callable]] = None,
        filter_reward_fns: Optional[List[Callable]] = None,
        val_dataset: Optional[Dataset] = None,
        val_reward_fns: Optional[List[Callable]] = None,
        data_packer: Optional[DataPacker] = None,
        val_data_packer: Optional[DataPacker] = None,
        num_workers: int = 8,
    ) -> None:
        """
        Setup the RewardCalculator with the given configuration and datasets.
        Args:
            config (Config): The configuration for the reward calculator.
            dataset (Optional[Dataset]): The training dataset.
            reward_fns (Optional[List[Callable]]): The list of reward functions for training.
            filter_reward_fns (Optional[List[Callable]]): The list of filter reward functions for dynamic sampling.
            val_dataset (Optional[Dataset]): The validation dataset.
            val_reward_fns (Optional[List[Callable]]): The list of reward functions for validation.
            data_packer (Optional[DataPacker]): The data packer for processing the payloads.
            val_data_packer (Optional[DataPacker]): The data packer for processing the validation payloads.
            num_workers (int): The number of worker processes for parallel reward calculation.
        """

        def worker_init(
            config,
            dataset,
            reward_fns,
            filter_reward_fns,
            val_dataset,
            val_reward_fns,
            data_packer,
            val_data_packer,
        ):
            reward_calculator = RewardCalculator.get_instance()
            reward_calculator.setup(
                config=config,
                dataset=dataset,
                reward_fns=reward_fns,
                filter_reward_fns=filter_reward_fns,
                val_dataset=val_dataset,
                val_reward_fns=val_reward_fns,
                data_packer=data_packer,
                val_data_packer=val_data_packer,
            )

        if num_workers > 0:
            self.executor = ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=worker_init,
                initargs=(
                    config,
                    dataset,
                    reward_fns,
                    filter_reward_fns,
                    val_dataset,
                    val_reward_fns,
                    data_packer,
                    val_data_packer,
                ),
            )
        else:
            self.executor = None

    @staticmethod
    def compute_rewards(payloads, is_validation, step, prompt_idxs):
        """
        Static method to compute rewards using the singleton RewardCalculator instance.
        Args:
            payloads (List[RLPayload]): List of RLPayload to compute rewards for.
            is_validation (bool): Whether the payloads are from validation set.
            step (int): The weight step where the payloads are generated.
            prompt_idxs (List[int]): List of prompt indices corresponding to the payloads.
        Returns:
            Tuple[List[RLPayload], bool, int]: (payloads, is_validation, step)
                payloads: List of RLPayload with rewards and advantages
                is_validation: whether the payloads are from validation set
                step: the weight step where the payloads are generated
        """
        reward_calculator = RewardCalculator.get_instance()
        return reward_calculator.compute_rewards(
            payloads, is_validation, step, prompt_idxs
        )

    def enqueue_rewards_cal(
        self,
        payloads: List[RLPayload],
        is_validation: bool,
        step: int,
        prompt_idxs: List[int] = [],
    ) -> None:
        """
        Enqueue the reward calculation task.
        The task will be executed in
        a separate process and the result will be stored in the task queue.
        Args:
            payloads (List[RLPayload]): List of RLPayload to compute rewards for.
            is_validation (bool): Whether the payloads are from validation set.
            step (int): The weight step where the payloads are generated.
            prompt_idxs (List[int]): List of prompt indices corresponding to the payloads.
        """
        for i in range(0, len(payloads), self.payload_per_task):
            self.task_queue.put(
                self.executor.submit(
                    RewardDispatcher.compute_rewards,
                    payloads[i : i + self.payload_per_task],
                    is_validation,
                    step,
                    prompt_idxs[i : i + self.payload_per_task],
                )
            )

    def dequeue_rewards_cal(
        self,
    ) -> Optional[Tuple[List[RLPayload], bool, int, bool]]:
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
