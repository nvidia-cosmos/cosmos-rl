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
import threading
from typing import List, Optional, Callable, Tuple
from concurrent.futures import (
    CancelledError,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
)

from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.dispatcher.algo.base import REGISTERED_ALGOs
from cosmos_rl.dispatcher.data.packer import BaseDataPacker
from cosmos_rl.policy.config import Config
from cosmos_rl.reward.admission import (
    resolve_completion_admission,
    select_payload_completions,
)
from cosmos_rl.reward.callbacks import (
    annotate_reward_result,
    close_reward_callbacks,
    finish_reward_reads,
    initialize_reward_callbacks,
    reward_callbacks_for,
    reward_read_scope,
)
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
        self.executor = None
        self._shutdown_lock = threading.Lock()
        self._shutdown_thread = None
        self._shutdown_error = None
        self._canceled_cleanup_error = None

    def shutdown(self, timeout_s: float = 15.0) -> None:
        """Stop owned reward workers without blocking teardown indefinitely.

        ``concurrent.futures`` has no bounded ``shutdown(wait=True)`` API. Run
        that ownership boundary in one daemon helper and wait for it with a
        finite budget. A timeout deliberately leaves ``executor`` attached so
        the caller can retain every resource that a still-running reward task
        may reference.
        """
        if timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")

        with self._shutdown_lock:
            executor = self.executor
            remote = getattr(self, "is_remote", False)
            if (
                executor is None
                and not remote
                and not hasattr(self, "reward_callbacks")
            ):
                return
            thread = self._shutdown_thread
            if thread is None:

                def shutdown_executor():
                    first_error = None
                    quiesced = False
                    calculator = getattr(self, "reward_calculator", None)
                    try:
                        if remote:
                            calculator.close()
                            quiesced = True
                        elif executor is None:
                            quiesced = True
                        else:
                            executor.shutdown(wait=True, cancel_futures=True)
                            quiesced = True
                        first_error = self._canceled_cleanup_error
                    except BaseException as exc:
                        first_error = exc
                        quiesced = remote and getattr(
                            calculator, "_readers_quiesced", False
                        )
                    if quiesced:
                        for owner in (calculator, self):
                            try:
                                close_reward_callbacks(owner)
                            except BaseException as exc:
                                if first_error is None:
                                    first_error = exc
                    self._shutdown_error = first_error

                thread = threading.Thread(
                    target=shutdown_executor,
                    daemon=True,
                    name="RewardDispatcherShutdown",
                )
                self._shutdown_thread = thread
                thread.start()

        thread.join(timeout=timeout_s)
        if thread.is_alive():
            raise TimeoutError(
                f"Reward executor did not stop within {timeout_s:g} seconds"
            )

        with self._shutdown_lock:
            error = self._shutdown_error
            if error is None:
                self.executor = None
        if error is not None:
            raise RuntimeError("Reward executor shutdown failed") from error

    def setup(
        self,
        config: Config,
        reward_fns: Optional[List[Callable]] = None,
        filter_reward_fns: Optional[List[Callable]] = None,
        val_reward_fns: Optional[List[Callable]] = None,
        data_packer: Optional[BaseDataPacker] = None,
        val_data_packer: Optional[BaseDataPacker] = None,
        num_workers: int = 2,
        hook_fns=None,
        callback_context=None,
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

        initialize_reward_callbacks(self, hook_fns, callback_context)
        self.callback_context = callback_context
        self.data_packer = data_packer
        self.val_data_packer = val_data_packer
        self.is_remote = config.train.train_policy.use_remote_reward
        self.minimum_trainable_completions = REGISTERED_ALGOs[
            config.train.train_policy.algo
        ].minimum_trainable_completions
        if self.is_remote:
            self.remote_batch_size = config.train.train_policy.remote_reward.batch_size

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
                hook_fns=hook_fns,
                callback_context=self.callback_context,
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
            with reward_read_scope(
                getattr(
                    self, "val_data_packer" if is_validation else "data_packer", None
                ),
                payloads,
                is_validation=is_validation,
                callbacks=reward_callbacks_for(self, is_validation),
            ):
                results = self._bypass_rewards(payloads, is_validation, step)
            for result in results:
                self.task_queue.put(result)
        else:
            # For the remote reward, we batch payloads by total completions count.
            # remote_batch_size refers to the max number of completions per request,
            # so the number of payloads per batch varies depending on each payload's
            # completion count.
            if self.is_remote:
                batch = []
                batch_completions = 0
                for payload in payloads:
                    n = len(payload.completions)
                    if batch and batch_completions + n > self.remote_batch_size:
                        self.task_queue.put(
                            RewardDispatcher.compute_rewards(
                                batch, is_validation, step, self.is_remote
                            )
                        )
                        batch = []
                        batch_completions = 0
                    batch.append(payload)
                    batch_completions += n
                if batch:
                    self.task_queue.put(
                        RewardDispatcher.compute_rewards(
                            batch, is_validation, step, self.is_remote
                        )
                    )
            # For the local reward, the task will be executed in a separate process.
            # The result will be stored in the task queue.
            else:
                for i in range(0, len(payloads), self.payload_per_task):
                    batch = payloads[i : i + self.payload_per_task]
                    packer = getattr(
                        self,
                        "val_data_packer" if is_validation else "data_packer",
                        None,
                    )
                    try:
                        future = self.executor.submit(
                            RewardDispatcher.compute_rewards,
                            batch,
                            is_validation,
                            step,
                            self.is_remote,
                        )
                    except BaseException:
                        with reward_read_scope(
                            packer,
                            payloads[i:],
                            is_validation=is_validation,
                            callbacks=reward_callbacks_for(self, is_validation),
                        ):
                            raise

                    def finish_canceled(future, owned=list(batch), packer=packer):
                        original_payloads = tuple(owned)
                        # Future callbacks otherwise retain the original large
                        # inputs even after their reward result is available.
                        owned.clear()
                        if future.cancelled():
                            cleanup_error = finish_reward_reads(
                                packer,
                                original_payloads,
                                is_validation=is_validation,
                                callbacks=reward_callbacks_for(self, is_validation),
                                error=CancelledError(
                                    "Reward task canceled before evaluation"
                                ),
                            )
                            if cleanup_error is not None:
                                with self._shutdown_lock:
                                    if self._canceled_cleanup_error is None:
                                        self._canceled_cleanup_error = cleanup_error

                    future.add_done_callback(finish_canceled)
                    self.task_queue.put(future)

    def _bypass_rewards(self, payloads, is_validation, step):
        results = []
        for i in range(0, len(payloads), self.payload_per_task):
            # Directly return the payloads with zero rewards and advantages
            for payload_idx in range(i, min(i + self.payload_per_task, len(payloads))):
                payload = payloads[payload_idx]
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
                    payload.n_ignore_prefix_tokens = [0 for _ in payload.completions]
                admission = resolve_completion_admission(
                    payload,
                    self.minimum_trainable_completions,
                    enabled=not is_validation,
                )
                annotate_reward_result(
                    getattr(
                        self,
                        "val_data_packer" if is_validation else "data_packer",
                        None,
                    ),
                    payload,
                    admission,
                    payload.rewards,
                    payload.filter_rewards,
                    [] if admission.group_excluded else [0.0] * admission.eligible_size,
                    payload.report_metrics,
                    is_validation=is_validation,
                    callbacks=reward_callbacks_for(self, is_validation),
                )
                payloads[payload_idx] = select_payload_completions(payload, admission)
                if admission.group_excluded:
                    payloads[payload_idx].valid = False
            results.append(
                (
                    payloads[i : i + self.payload_per_task],
                    is_validation,
                    step,
                )
            )
        return results

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
