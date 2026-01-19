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

from queue import Queue
from typing import List, Optional, Callable, Tuple
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor

from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.dispatcher.data.packer import BaseDataPacker
from cosmos_rl.policy.config import Config
from cosmos_rl.reward.remote_calculator import RemoteRewardCalculator
from cosmos_rl.reward.local_calculator import LocalRewardCalculator


class RewardDispatcher:
    """
    RewardDispatcher is responsible for dispatching the reward calculation tasks to the RewardCalculator.
    It uses a ProcessPoolExecutor to parallelize the reward calculation.
    It also uses a Queue to store the tasks and results.
    """

    def __init__(self, payload_per_task: int = 1):
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

        self.is_remote = config.train.train_policy.use_remote_reward

        def worker_init(
            config,
            reward_fns,
            filter_reward_fns,
            val_reward_fns,
            data_packer,
            val_data_packer,
        ):
            if self.is_remote:
                self.reward_calculator = RemoteRewardCalculator.get_instance()
            else:
                self.reward_calculator = LocalRewardCalculator.get_instance()
            self.reward_calculator.setup(
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
        if not self.is_remote and num_workers > 0:
            # Use multiprocessing for local reward calculation
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
    def compute_rewards(payloads, is_validation, step, is_remote=False):
        """
        Static method to compute rewards using the singleton RewardCalculator instance.
        Args:
            payloads (List[RLPayload]): List of RLPayload to compute rewards for.
            is_validation (bool): Whether the payloads are from validation set.
            step (int): The weight step where the payloads are generated.
            is_remote (bool): Whether to use RemoteRewardCalculator or LocalRewardCalculator.
        Returns:
            Tuple[List[RLPayload], bool, int]: (payloads, is_validation, step)
                payloads: List of RLPayload with rewards and advantages
                is_validation: whether the payloads are from validation set
                step: the weight step where the payloads are generated
        """
        if is_remote:
            reward_calculator = RemoteRewardCalculator.get_instance()
        else:
            reward_calculator = LocalRewardCalculator.get_instance()
        return reward_calculator.compute_rewards(payloads, is_validation, step)

    def enqueue_rewards_cal(
        self,
        payloads: List[RLPayload],
        is_validation: bool,
        step: int,
        bypass_reward: bool = False,
    ) -> None:
        """
        Enqueue the reward calculation task.
        Args:
            payloads (List[RLPayload]): List of RLPayload to compute rewards for.
            is_validation (bool): Whether the payloads are from validation set.
            step (int): The weight step where the payloads are generated.
            bypass_reward (bool): Whether to bypass the reward calculation and set rewards to zero.
        """
        if bypass_reward:
            for i in range(0, len(payloads), self.payload_per_task):
                # Directly return the payloads with zero rewards and advantages
                for payload in payloads[i : i + self.payload_per_task]:
                    payload.rewards = [0.0 for _ in payload.completions]
                    payload.advantages = [0.0 for _ in payload.completions]
                    payload.filter_rewards = [0.0 for _ in payload.completions]
                    payload.report_metrics = [{} for _ in payload.completions]
                    if payload.completed_conversations is None:
                        payload.completed_conversations = [
                            [] for _ in range(len(payload.completions))
                        ]
                    if payload.completion_logprobs is None:
                        payload.completion_logprobs = [
                            [] for _ in range(len(payload.completions))
                        ]
                    if payload.completion_token_ids is None:
                        payload.completion_token_ids = [
                            [] for _ in range(len(payload.completions))
                        ]
                    if payload.n_ignore_prefix_tokens is None:
                        payload.n_ignore_prefix_tokens = [
                            0 for _ in payload.completions
                        ]
                self.task_queue.put(
                    (
                        payloads[i : i + self.payload_per_task],
                        is_validation,
                        step,
                        self.is_remote,
                    )
                )
        else:
            # For the remote reward, we send one payload at a time, and get the uuid back.
            # Save the uuid in the task queue to track the result later.
            if self.is_remote:
                for payload in payloads:
                    self.task_queue.put(
                        RewardDispatcher.compute_rewards(
                            [payload], is_validation, step, self.is_remote
                        )
                    )
            # For the local reward, the task will be executed in a separate process.
            # The result will be stored in the task queue.
            else:
                for i in range(0, len(payloads), self.payload_per_task):
                    self.task_queue.put(
                        self.executor.submit(
                            RewardDispatcher.compute_rewards,
                            payloads[i : i + self.payload_per_task],
                            is_validation,
                            step,
                            self.is_remote,
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
        payloads = None
        is_validation = False
        step = -1
        all_done = False

        if not self.task_queue.empty():
            if self.is_remote:
                payloads, is_validation, step = self.reward_calculator.get_results(
                    self.task_queue
                )
            elif not isinstance(self.task_queue.queue[0], Future):
                assert isinstance(self.task_queue.queue[0], tuple)
                payloads, is_validation, step = self.task_queue.get()
            elif self.task_queue.queue[0].done():
                payloads, is_validation, step = self.task_queue.get().result()
        else:
            all_done = True
        return payloads, is_validation, step, all_done

    def is_empty(self) -> bool:
        """
        Check if the task queue is empty.
        Returns:
            True if the task queue is empty, False otherwise.
        """
        return self.task_queue.empty()
