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

import asyncio
import torch
import threading
from typing import List, Optional, Set
from queue import Queue
from dataclasses import dataclass
from contextlib import contextmanager
import time

from vllm import SamplingParams
from cosmos_rl.dispatcher.data import RLPayload, IdxAndRLPayload
from cosmos_rl.dispatcher.data.packer import DataPacker
from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.rollout.rollout_base import RolloutBase
from cosmos_rl.utils.logging import logger


@dataclass
class CompletedRollout:
    """Represents a completed rollout generation."""

    # the index of the payload in the dataset
    idx: int
    payload: RLPayload
    result: RolloutResult


class RolloutTaskScheduler:
    """
    Schedules and manages asynchronous rollout task execution using RolloutBase interface.

    This scheduler implements a producer-consumer pattern:
    - Internally manages task_queue and complete_queue
    - Accepts payloads via put_rollout() method
    - Runs a background async loop that monitors task_queue in a separate thread
    - Controls concurrent generation based on max_concurrent_requests
    - Calls rollout engine's rollout_generation() for each payload
    - Provides get() method to retrieve completed results

    Thread Safety:
    - Uses threading.Event() for _running and _paused flags to ensure thread-safe state management
    - Prevents memory visibility issues across threads
    - All state changes are atomic and immediately visible to worker thread

    Key Features:
    - Producer-consumer pattern with Queue for thread-safe communication
    - Automatic task concurrency control (max_concurrent_requests)
    - Support for pause/resume during critical operations (e.g., weight sync)
    - Context manager for temporary pause (with scheduler.paused())
    - Graceful shutdown with active task completion

    Usage Example:
    ```python
    # Initialize the scheduler
    scheduler = RolloutTaskScheduler(
        rollout_engine=rollout_engine,
        data_packer=data_packer,
        sampling_params=sampling_params,
        max_concurrent_requests=10
    )

    # Start the background worker thread
    scheduler.start()

    # Put payloads into scheduler
    scheduler.put_rollout(payload1)
    scheduler.put_rollout(payload2)

    # Get completed results (non-blocking)
    completed = scheduler.get(block=False)
    if completed:
        print(f"Prompt: {completed.result.prompt}")
        print(f"Completions: {completed.result.completions}")

    # Pause during critical operations (e.g., weight synchronization)
    with scheduler.paused(wait_for_active_tasks=True):
        # All active tasks completed, safe to update weights
        sync_weights()
        # Scheduler automatically resumes after this block

    # Or manually control pause/resume
    scheduler.pause()
    sync_weights()
    scheduler.resume()

    # Check scheduler status
    stats = scheduler.get_stats()
    print(f"Active tasks: {stats['active_tasks']}")
    print(f"Pending tasks: {stats['pending_tasks']}")
    print(f"Completed results: {stats['completed_results']}")

    # Stop the scheduler when done
    scheduler.stop()
    ```

    Architecture:
    ```
    Main Thread                    Worker Thread (separate event loop)
    ┌─────────────┐               ┌──────────────────────────────┐
    │             │               │  asyncio event loop          │
    │ put_rollout │──► task_queue │  ┌─────────────────────┐    │
    │             │               │  │ _worker_loop()      │    │
    │             │               │  │  ├─ create_task()   │    │
    │ get()       │◄── complete_  │  │  ├─ max_concurrent  │    │
    │             │    queue      │  │  └─ _generate_...() │    │
    │             │               │  └─────────────────────┘    │
    │ pause()     │──► _paused    │                              │
    │ resume()    │    (Event)    │  ┌─────────┐  ┌─────────┐  │
    │             │               │  │ Task 1  │  │ Task 2  │  │
    │ stop()      │──► _running   │  └─────────┘  └─────────┘  │
    │             │    (Event)    │       ↓            ↓        │
    └─────────────┘               │   rollout_generation()      │
                                  └──────────────────────────────┘
    ```
    """

    def __init__(
        self,
        rollout_engine: RolloutBase,
        data_packer: DataPacker,
        sampling_params: SamplingParams,
        max_concurrent_requests: int = 10,
        stream: Optional[torch.cuda.Stream] = None,
        check_interval: float = 0.1,
    ):
        """
        Initialize the RolloutTaskScheduler.

        Args:
            rollout_engine: The rollout engine implementing RolloutBase interface
            data_packer: Data packer for processing payloads
            sampling_params: Sampling parameters for generation
            max_concurrent_requests: Maximum number of concurrent generation requests
            stream: CUDA stream for generation (optional)
            check_interval: Interval (in seconds) to check task_queue when empty
        """
        self.rollout_engine = rollout_engine
        self.data_packer = data_packer
        self.sampling_params = sampling_params
        self.max_concurrent_requests = max_concurrent_requests
        self.stream = stream
        self.check_interval = check_interval

        # Create internal queues
        self.task_queue = Queue()
        self.complete_queue = Queue()

        # Track running state
        self._running = threading.Event()
        self._paused = threading.Event()
        self._worker_thread = None
        self._loop = None

        # Track active tasks
        self.active_tasks: Set[asyncio.Task] = set()
        self.total_processed = 0
        self.total_submitted = 0

        logger.info(
            f"[RolloutTaskScheduler] Initialized with max_concurrent_requests={max_concurrent_requests}"
        )

    def put_rollout(self, payload: IdxAndRLPayload):
        """
        Put a single payload into the task queue for processing.

        Args:
            payload: The RLPayload to process
        """
        self.task_queue.put(payload)
        self.total_submitted += 1
        logger.debug(
            f"[RolloutTaskScheduler] Added payload to task queue "
            f"(total submitted: {self.total_submitted})"
        )

    def put_rollout_batch(self, payloads: List[IdxAndRLPayload]):
        """
        Put multiple payloads into the task queue for processing.

        Args:
            payloads: List of RLPayloads to process
        """
        for payload in payloads:
            self.task_queue.put(payload)
        self.total_submitted += len(payloads)
        logger.debug(
            f"[RolloutTaskScheduler] Added {len(payloads)} payloads to task queue "
            f"(total submitted: {self.total_submitted})"
        )

    async def _generate_single(
        self, payload: IdxAndRLPayload
    ) -> Optional[CompletedRollout]:
        """
        Generate completion for a single payload asynchronously.

        Args:
            payload: The RLPayload to generate from

        Returns:
            CompletedRollout object containing the payload and result
        """
        try:
            idx, rawPayload = payload
            # Call rollout engine's async generation method
            results = await self.rollout_engine.rollout_generation(
                payloads=[rawPayload],
                stream=self.stream,
                data_packer=self.data_packer,
                sampling_params=self.sampling_params,
            )

            if results and len(results) > 0:
                result = results[0]
                completed = CompletedRollout(idx=idx, payload=rawPayload, result=result)

                # Put the completed result into the queue
                self.complete_queue.put(completed)

                self.total_processed += 1
                logger.debug(
                    f"[RolloutTaskScheduler] Completed generation for payload "
                    f"({self.total_processed} total processed)"
                )

                return completed
            else:
                logger.warning(
                    "[RolloutTaskScheduler] Generation returned empty results"
                )
                return None

        except Exception as e:
            logger.error(f"[RolloutTaskScheduler] Error during generation: {str(e)}")
            import traceback

            traceback.print_exc()
            return None

    async def _worker_loop(self):
        """
        Main worker loop that monitors task_queue and manages generation tasks.

        This loop:
        1. Checks task_queue for new payloads
        2. Launches new generation tasks if under max_concurrent_requests
        3. Monitors running tasks and removes completed ones
        """
        logger.info("[RolloutTaskScheduler] Worker loop started")

        while self._running.is_set():
            # Check and clean up completed tasks
            completed_tasks = {task for task in self.active_tasks if task.done()}
            for task in completed_tasks:
                self.active_tasks.remove(task)
                # Retrieve any exceptions
                try:
                    await task
                except Exception as e:
                    logger.error(
                        f"[RolloutTaskScheduler] Task failed with exception: {e}"
                    )

            # Try to start new tasks if we have capacity and not paused
            if not self._paused.is_set():
                while (
                    len(self.active_tasks) < self.max_concurrent_requests
                    and not self.task_queue.empty()
                ):
                    try:
                        # Get payload from task queue (non-blocking)
                        payload = self.task_queue.get_nowait()

                        # Create and start a new generation task
                        task = asyncio.create_task(self._generate_single(payload))
                        self.active_tasks.add(task)

                        logger.debug(
                            f"[RolloutTaskScheduler] Started new task "
                            f"(active: {len(self.active_tasks)}/{self.max_concurrent_requests})"
                        )

                    except Exception:
                        # Queue is empty or other error
                        break

            # Sleep briefly before next iteration
            await asyncio.sleep(self.check_interval)

        # Wait for remaining tasks to complete before shutting down
        if self.active_tasks:
            logger.info(
                f"[RolloutTaskScheduler] Waiting for {len(self.active_tasks)} tasks to complete..."
            )
            await asyncio.gather(*self.active_tasks, return_exceptions=True)

        logger.info("[RolloutTaskScheduler] Worker loop stopped")

    def _run_event_loop(self):
        """
        Run the asyncio event loop in a separate thread.

        This method is called by the worker thread and:
        1. Creates a new event loop for the worker thread
        2. Sets it as the thread's current event loop
        3. Runs the _worker_loop() coroutine to completion
        4. Properly closes the event loop on exit

        Note: Each thread must have its own event loop.
        """
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._worker_loop())
        finally:
            self._loop.close()

    def start(self):
        """
        Start the background worker thread with its own event loop.

        This creates a new thread that runs an independent asyncio event loop.
        The worker thread monitors the task_queue and manages concurrent task execution.

        Thread Safety:
            Uses threading.Event() for _running flag to ensure thread-safe state management.
        """
        if self._running.is_set():
            logger.warning("[RolloutTaskScheduler] Scheduler is already running")
            return

        self._running.set()
        self._worker_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._worker_thread.start()

        logger.info("[RolloutTaskScheduler] Background worker started")

    def stop(self, wait: bool = True):
        """
        Stop the background worker thread gracefully.

        This method:
        1. Clears the _running event flag (thread-safe)
        2. Signals the worker thread to stop
        3. Optionally waits for the thread to complete (default: True)
        4. The worker thread will finish processing active tasks before exiting

        Args:
            wait: Whether to wait for the worker thread to finish (default: True).
                  If True, blocks until thread completes (with 10s timeout).
                  If False, returns immediately without waiting.

        Thread Safety:
            Uses threading.Event.clear() for atomic state change visible to worker thread.
        """
        if not self._running.is_set():
            logger.warning("[RolloutTaskScheduler] Scheduler is not running")
            return

        logger.info("[RolloutTaskScheduler] Stopping background worker...")
        self._running.clear()

        if wait and self._worker_thread:
            self._worker_thread.join(timeout=10)

        logger.info("[RolloutTaskScheduler] Background worker stopped")

    def pause(self):
        """
        Pause the scheduler from processing new tasks from task_queue.

        Currently running tasks will continue to completion, but no new tasks
        will be started until resume() is called.
        """
        if not self._running.is_set():
            logger.warning(
                "[RolloutTaskScheduler] Cannot pause: scheduler is not running"
            )
            return

        if self._paused.is_set():
            logger.warning("[RolloutTaskScheduler] Scheduler is already paused")
            return

        self._paused.set()
        logger.debug(
            "[RolloutTaskScheduler] Scheduler paused (active tasks will continue)"
        )

    def resume(self):
        """
        Resume the scheduler to continue processing tasks from task_queue.
        """
        if not self._running.is_set():
            logger.warning(
                "[RolloutTaskScheduler] Cannot resume: scheduler is not running"
            )
            return

        if not self._paused.is_set():
            logger.warning("[RolloutTaskScheduler] Scheduler is not paused")
            return

        self._paused.clear()
        logger.debug("[RolloutTaskScheduler] Scheduler resumed")

    def is_paused(self) -> bool:
        """
        Check if the scheduler is currently paused.

        Returns:
            True if paused, False otherwise
        """
        return self._paused.is_set()

    @contextmanager
    def paused(
        self, wait_for_active_tasks: bool = True, timeout: Optional[float] = None
    ):
        """
        Context manager to temporarily pause the scheduler.

        Automatically resumes the scheduler when exiting the context,
        even if an exception occurs.

        Args:
            wait_for_active_tasks: If True, wait for active tasks to complete before yielding
            timeout: Maximum time to wait for active tasks (None means wait forever)

        Usage:
            ```python
            # Pause scheduler during weight sync
            with scheduler.paused():
                # Scheduler is paused here
                sync_weights()
                # Scheduler will automatically resume, even if exception occurs
            ```

        Raises:
            RuntimeError: If scheduler is not running
            TimeoutError: If waiting for active tasks times out
        """
        if not self._running.is_set():
            raise RuntimeError(
                "[RolloutTaskScheduler] Cannot pause: scheduler is not running"
            )

        was_already_paused = self._paused.is_set()

        try:
            # Pause if not already paused
            if not was_already_paused:
                self._paused.set()
                logger.info("[RolloutTaskScheduler] Scheduler paused (context manager)")

                # Wait for active tasks to complete if requested
                if wait_for_active_tasks:
                    start_time = time.time()
                    while len(self.active_tasks) > 0:
                        if timeout is not None and (time.time() - start_time) > timeout:
                            raise TimeoutError(
                                f"[RolloutTaskScheduler] Timeout waiting for {len(self.active_tasks)} active tasks"
                            )
                        time.sleep(0.1)
                        logger.debug(
                            f"[RolloutTaskScheduler] Waiting for {len(self.active_tasks)} active tasks to complete"
                        )
                    logger.info("[RolloutTaskScheduler] All active tasks completed")

            # Yield control to the context block
            yield self

        finally:
            # Resume only if we paused it (not if it was already paused)
            if not was_already_paused and self._paused.is_set():
                self._paused.clear()
                logger.info(
                    "[RolloutTaskScheduler] Scheduler resumed (context manager)"
                )

    def get(
        self, block: bool = True, timeout: Optional[float] = None
    ) -> Optional[CompletedRollout]:
        """
        Get a completed rollout from the completion queue.

        Args:
            block: If True, block until a result is available (default: True)
            timeout: Timeout in seconds when blocking (None means wait forever)

        Returns:
            CompletedRollout object if available, None if queue is empty (when block=False)
            or timeout occurs

        Raises:
            queue.Empty: If block=True and timeout occurs (can be caught if needed)
        """
        try:
            if block:
                return self.complete_queue.get(block=True, timeout=timeout)
            else:
                return self.complete_queue.get_nowait()
        except Exception:
            return None

    def get_all(self) -> List[CompletedRollout]:
        """
        Get all available completed rollouts from the completion queue (non-blocking).

        Returns:
            List of CompletedRollout objects (empty list if none available)
        """
        results = []
        while not self.complete_queue.empty():
            try:
                item = self.complete_queue.get_nowait()
                results.append(item)
            except Exception:
                break
        return results

    async def draining_activate_tasks(self, timeout: Optional[float] = None):
        """
        Draining all active tasks from the active tasks set, and block processing task from task queue.

        Which is useful when we want to update the weights during training.

        Args:
            timeout: Maximum time in seconds to wait for active tasks (None means wait forever)

        Returns:
            True if all active tasks are drained, False if timeout occurs
        """
        start_time = time.time()
        while len(self.active_tasks) > 0:
            if timeout is not None and (time.time() - start_time) > timeout:
                raise TimeoutError(
                    f"[RolloutTaskScheduler] Timeout waiting for {len(self.active_tasks)} active tasks"
                )
            await asyncio.sleep(0.1)

    async def wait_for_all_tasks_to_complete(self):
        """
        Wait for all tasks to complete.
        """
        while len(self.active_tasks) > 0:
            await asyncio.sleep(0.1)

    def is_idle(self) -> bool:
        """
        Check if the scheduler is idle.
        """
        return (
            self.is_running()
            and len(self.active_tasks) == 0
            and self.task_queue.empty()
            and self.complete_queue.empty()
        )

    def has_results(self) -> bool:
        """
        Check if there are any completed results available.

        Returns:
            True if results are available, False otherwise
        """
        return not self.complete_queue.empty()

    def pending_tasks(self) -> int:
        """
        Get the number of tasks waiting in the task queue.

        Returns:
            Number of pending tasks
        """
        return self.task_queue.qsize()

    def completed_results(self) -> int:
        """
        Get the number of completed results available for retrieval.

        Returns:
            Number of completed results in the queue
        """
        return self.complete_queue.qsize()

    def is_running(self) -> bool:
        """
        Check if the scheduler is currently running.

        Returns:
            True if running, False otherwise
        """
        return self._running.is_set()

    def get_stats(self) -> dict:
        """
        Get statistics about the rollout task scheduler.

        Returns:
            Dictionary with statistics
        """
        return {
            "running": self._running.is_set(),
            "paused": self._paused.is_set(),
            "total_submitted": self.total_submitted,
            "total_processed": self.total_processed,
            "active_tasks": len(self.active_tasks),
            "pending_tasks": self.task_queue.qsize(),
            "completed_results": self.complete_queue.qsize(),
            "max_concurrent_requests": self.max_concurrent_requests,
        }

    # TODO(zjx): below are some HACKs in this class, be careful when calling them.
    def get_front_prompt_weight_version(self) -> (bool, int):
        """
        Get the weight version of the front prompt in the task queue.
        """

        if self.task_queue.empty():
            return (False, 0)

        payload: RLPayload = self.task_queue.queue[0][1]
        return (True, payload.weight_version)
