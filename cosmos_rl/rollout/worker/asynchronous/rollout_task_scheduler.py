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
from typing import List, Optional, Set, Callable
from queue import Empty, Queue
from dataclasses import dataclass
from contextlib import contextmanager
from functools import partial
import time

from cosmos_rl.dispatcher.data import RLPayload
from cosmos_rl.dispatcher.data.packer import DataPacker
from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.rollout.rollout_base import RolloutBase
from cosmos_rl.utils.logging import logger


@dataclass
class RolloutTask:
    """Represents a rollout task to be executed."""

    idx: int
    payload: RLPayload
    is_validation: bool = False


@dataclass
class CompletedRollout:
    """Represents a completed rollout generation."""

    # the index of the payload in the dataset
    idx: int
    payload: RLPayload
    result: RolloutResult
    is_validation: bool = False


class RolloutTaskScheduler:
    """
    Schedules and manages asynchronous rollout task execution using RolloutBase interface.

    This scheduler implements a producer-consumer pattern:
    - Internally manages task_queue and complete_queue
    - Accepts payloads via put_rollout() method
    - Runs a background async loop that monitors task_queue (in separate thread or async mode)
    - Controls concurrent generation based on max_concurrent_requests
    - Calls rollout engine's rollout_generation() for each payload
    - Provides get() method to retrieve completed results

    Thread Safety:
    - A condition protects admission, active-task registration and pause ownership.
    - A draining pause waits for scheduler tasks, not backend device completion.

    Key Features:
    - Producer-consumer pattern with Queue for thread-safe communication
    - Automatic task concurrency control (max_concurrent_requests)
    - Support for pause/resume during critical operations (e.g., weight sync)
    - Context manager for temporary pause (with scheduler.paused())
    - Graceful shutdown with active task completion
    - Dual operation modes: thread-based (start/stop) or async-based (start_async/stop_async)

    Usage Example (Thread Mode):
    ```python
    # Initialize the scheduler
    scheduler = RolloutTaskScheduler(
        rollout_engine=rollout_engine,
        data_packer=data_packer,
        max_concurrent_requests=10
    )

    # Start the background worker thread
    scheduler.start()

    # Create and put rollout tasks into scheduler
    task1 = RolloutTask(idx=0, payload=payload1, is_validation=False)
    task2 = RolloutTask(idx=1, payload=payload2, is_validation=False)
    scheduler.put_rollout(task1)
    scheduler.put_rollout(task2)

    # Get completed results (non-blocking)
    completed = scheduler.get(block=False)
    if completed:
        print(f"Index: {completed.idx}")
        print(f"Prompt: {completed.result.prompt}")
        print(f"Completions: {completed.result.completions}")

    # Pause during critical operations (e.g., weight synchronization)
    with scheduler.paused(wait_for_active_tasks=True):
        # All scheduler tasks completed; also fence backend device work before writes.
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

    Usage Example (Async Mode):
    ```python
    # Initialize the scheduler
    scheduler = RolloutTaskScheduler(
        rollout_engine=rollout_engine,
        data_packer=data_packer,
        max_concurrent_requests=10
    )

    # Start the scheduler in async mode
    await scheduler.start_async()

    # Submit tasks (same as thread mode)
    scheduler.put_rollout(task1)
    scheduler.put_rollout(task2)

    # Wait for all tasks to complete
    await scheduler.wait_all_tasks_completed()

    # Stop the scheduler
    await scheduler.stop_async()
    ```

    Architecture:

    Thread Mode (start/stop):
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

    Async Mode (start_async/stop_async):
    ```
    Async Context (same event loop)
    ┌────────────────────────────────────────────┐
    │ put_rollout ──► task_queue                 │
    │                     │                      │
    │ await start_async() │                      │
    │         │           ▼                      │
    │         │      _worker_loop()              │
    │         │       ├─ create_task()           │
    │         │       ├─ max_concurrent          │
    │         │       └─ _generate_single()      │
    │         │                                  │
    │ get() ◄── complete_queue                   │
    │                                            │
    │ await stop_async()                         │
    │         │                                  │
    │         └─► _worker_task (awaited)         │
    └────────────────────────────────────────────┘
    ```

    Note: The scheduler supports either thread mode OR async mode, not both simultaneously.
    The mode is determined by which start method is called first (start() or start_async()).
    """

    def __init__(
        self,
        rollout_engine: RolloutBase,
        data_packer: DataPacker,
        max_concurrent_requests: int = 10,
        stream: Optional[torch.cuda.Stream] = None,
        check_interval: float = 0.1,
        val_data_packer: Optional[DataPacker] = None,
    ):
        """
        Initialize the RolloutTaskScheduler.

        Args:
            rollout_engine: The rollout engine implementing RolloutBase interface
            data_packer: Data packer for processing payloads
            max_concurrent_requests: Maximum number of concurrent generation requests
            stream: CUDA stream for generation (optional)
            check_interval: Interval (in seconds) to check task_queue when empty
            val_data_packer: Validation packer; defaults to the training packer
        """
        self.rollout_engine = rollout_engine
        self.data_packer = data_packer
        self.val_data_packer = (
            data_packer if val_data_packer is None else val_data_packer
        )
        self.max_concurrent_requests = max_concurrent_requests
        self.stream = stream
        self.check_interval = check_interval

        # Create internal queues
        self.task_queue = Queue()
        self.complete_queue = Queue()

        # Track running state
        self._running = threading.Event()
        self._paused = threading.Event()
        self._admission = threading.Condition()
        self._manual_pause = False
        self._pause_owners = 0
        self._worker_thread = (
            None  # Thread object when running in thread mode (start())
        )
        self._worker_task = (
            None  # asyncio.Task when running in async mode (start_async())
        )
        # share engine thread's event loop to the worker thread.
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Track active tasks
        self._active_tasks: Set[asyncio.Task] = set()
        self.total_processed = 0
        self.total_submitted = 0

        logger.info(
            f"[RolloutTaskScheduler] Initialized with max_concurrent_requests={max_concurrent_requests}"
        )

    def put_rollout(self, task: RolloutTask):
        """
        Put a single rollout task into the task queue for processing.

        Args:
            task: The RolloutTask containing payload index, RLPayload
        """
        self.task_queue.put(task)
        self.total_submitted += 1
        logger.debug(
            f"[RolloutTaskScheduler] Added payload to task queue "
            f"(total submitted: {self.total_submitted})"
        )

    def put_rollout_batch(self, tasks: List[RolloutTask]):
        """
        Put multiple rollout tasks into the task queue for processing.

        Args:
            tasks: List of RolloutTask objects, each containing payload index, RLPayload
        """
        for task in tasks:
            self.task_queue.put(task)
        self.total_submitted += len(tasks)
        logger.debug(
            f"[RolloutTaskScheduler] Added {len(tasks)} tasks to task queue "
            f"(total submitted: {self.total_submitted})"
        )

    async def _generate_single(self, task: RolloutTask) -> Optional[CompletedRollout]:
        """
        Generate completion for a single task asynchronously.

        Args:
            task: The RolloutTask containing payload

        Returns:
            CompletedRollout object containing the payload and result
        """
        try:
            # Call rollout engine's async generation method
            results = await self.rollout_engine.rollout_generation(
                payloads=[task.payload],
                stream=self.stream,
                data_packer=self.val_data_packer
                if task.is_validation
                else self.data_packer,
                data_fetcher=None,  # data should already be loaded in the task
                is_validation=task.is_validation,
            )

            if results and len(results) != 1:
                raise ValueError("Expected exactly one result for one rollout task")
            if results:
                # because we only put one payload into the rollout engine, so the results is a list with one element
                result = results[0]
            else:
                logger.warning(
                    "[RolloutTaskScheduler] Generation returned empty results"
                )
                result = RolloutResult(completions=[])

        except asyncio.CancelledError:
            # Cancellation is a terminal outcome for this accepted prompt too.
            # Return normally so the done callback cannot publish it twice.
            result = RolloutResult(completions=[])
        except Exception as e:
            logger.error(f"[RolloutTaskScheduler] Error during generation: {str(e)}")
            import traceback

            traceback.print_exc()
            result = RolloutResult(completions=[])

        # Publication failures are not generation failures: never publish a
        # second terminal result while handling an error from this step.
        return self._publish_terminal(task, result)

    def _publish_terminal(self, task, result):
        completed = CompletedRollout(
            task.idx, task.payload, result, is_validation=task.is_validation
        )
        self.complete_queue.put(completed)
        self.total_processed += 1
        return completed

    def _task_finished(self, rollout_task, task):
        try:
            if task.cancelled():
                # A task cancelled before its first coroutine instruction never
                # enters _generate_single's exception handler.
                self._publish_terminal(rollout_task, RolloutResult(completions=[]))
            elif task.exception() is not None:
                logger.error("[RolloutTaskScheduler] Task failed: %s", task.exception())
        finally:
            with self._admission:
                self._active_tasks.discard(task)
                self.task_queue.task_done()
                self._admission.notify_all()

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
            while True:
                # A pause must cover the entire dequeue/registration boundary.
                # Recheck between admissions so a busy queue cannot bypass it.
                with self._admission:
                    if (
                        self._paused.is_set()
                        or not self._running.is_set()
                        or len(self._active_tasks) >= self.max_concurrent_requests
                    ):
                        break
                    try:
                        rollout_task = self.task_queue.get_nowait()
                    except Empty:
                        break
                    task = asyncio.create_task(self._generate_single(rollout_task))
                    self._active_tasks.add(task)
                    task.add_done_callback(partial(self._task_finished, rollout_task))

            # Sleep briefly before next iteration
            await asyncio.sleep(self.check_interval)

        # Wait for remaining tasks to complete before shutting down
        if self._active_tasks:
            logger.info(
                f"[RolloutTaskScheduler] Waiting for {len(self._active_tasks)} tasks to complete..."
            )
            await asyncio.gather(*self._active_tasks, return_exceptions=True)

        # Stopping admission must not silently lose tasks already accepted but
        # not launched. Their caller still owns the terminal reporting step.
        while True:
            try:
                pending = self.task_queue.get_nowait()
            except Empty:
                break
            self._publish_terminal(pending, RolloutResult(completions=[]))
            self.task_queue.task_done()

        logger.info("[RolloutTaskScheduler] Worker loop stopped")

    def _try_shutdown_rollout_engine(self):
        """
        Try to shutdown the rollout engine to release the resources.
        """

        if not self.rollout_engine.is_engine_initialized():
            logger.warning(
                "[RolloutTaskScheduler] Rollout engine is not initialized, skipping shutdown"
            )
            return

        # rollout engine in async mode usually has a child thread to run the generation, to avoid blocking the main thread exit, we need to call the shutdown method to release the resources.
        self.rollout_engine.shutdown()

    def _run_event_loop(self, init_engine_hook: Callable):
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
            # we must initialize the rollout engine in the worker thread to avoid binding event_loop to the main thread's event_loop.
            assert not self.rollout_engine.is_engine_initialized(), (
                "Rollout engine should not be initialized before starting the scheduler worker loop"
            )
            init_engine_hook(self.rollout_engine)

            # mark the scheduler as running after the rollout engine is initialized.
            self._running.set()
            self._loop.run_until_complete(self._worker_loop())
        finally:
            with self._admission:
                self._running.clear()
                self._admission.notify_all()

            # Cancel all remaining tasks before closing the loop to prevent "Task was destroyed but it is pending" errors
            try:
                pending = asyncio.all_tasks(self._loop)
                if pending:
                    logger.debug(
                        f"[RolloutTaskScheduler] Cancelling {len(pending)} pending tasks before closing event loop"
                    )
                    for task in pending:
                        task.cancel()
                    # Wait for all tasks to be cancelled
                    self._loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            except Exception as e:
                logger.warning(
                    f"[RolloutTaskScheduler] Error while cancelling pending tasks: {e}"
                )
            finally:
                self._loop.close()

    def start(self, init_engine_hook: Callable, wait_initialized: bool = False):
        """
        Start the background worker thread with its own event loop.

        This creates a new thread that runs an independent asyncio event loop.
        The worker thread monitors the task_queue and manages concurrent task execution.

        Note: Cannot be used together with start_async() - only one mode can be active at a time.

        Args:
            init_engine_hook: A hook function to initialize the rollout engine. The function should take the rollout engine as an argument and initialize it.
            wait_initialized: Whether to wait for the rollout engine to be initialized before returning.

        Thread Safety:
            Uses threading.Event() for _running flag to ensure thread-safe state management.

        Raises:
            AssertionError: If start_async() was already called (async mode is active)
        """
        assert self._worker_task is None, (
            "Not support to start the scheduler worker loop both in thread and async mode"
        )

        if self._running.is_set():
            logger.warning("[RolloutTaskScheduler] Scheduler is already running")
            return

        self._worker_thread = threading.Thread(
            target=self._run_event_loop,
            daemon=True,
            name="RolloutTaskSchedulerWorker",
            args=(init_engine_hook,),
        )
        self._worker_thread.start()

        if wait_initialized:
            self._running.wait()

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
        with self._admission:
            self._running.clear()
            self._admission.notify_all()

        if wait and self._worker_thread:
            self._worker_thread.join(timeout=10)

        self._try_shutdown_rollout_engine()
        logger.info("[RolloutTaskScheduler] Background worker stopped")

    async def start_async(self, init_engine_hook: Callable):
        """
        Start the scheduler worker loop asynchronously in the current event loop.

        Unlike start() which creates a separate worker thread with its own event loop,
        this method runs the worker loop as an async task in the current event loop.
        This is useful when the scheduler needs to run in the same event loop as the
        calling code (e.g., in async applications or notebooks).

        Note: Cannot be used together with start() - only one mode can be active at a time.

        Thread Safety:
            Uses threading.Event() for _running flag to ensure thread-safe state management.

        Raises:
            AssertionError: If start() was already called (thread mode is active)
        """
        assert self._worker_thread is None, (
            "Not support to start the scheduler worker loop both in thread and async mode"
        )

        if self._running.is_set():
            logger.warning("[RolloutTaskScheduler] Scheduler is already running")
            return

        init_engine_hook(self.rollout_engine)

        # mark the scheduler as running after the rollout engine is initialized.
        self._loop = asyncio.get_event_loop()
        self._running.set()
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("[RolloutTaskScheduler] Background worker started")

    async def stop_async(self, wait: bool = True):
        """
        Stop the scheduler asynchronously.

        This method stops the async worker loop started by start_async().
        It signals the worker to stop and optionally waits for it to complete.

        Args:
            wait: Whether to wait for the worker task to finish (default: True).
                  If True, awaits the worker task until it completes.
                  If False, returns immediately without waiting.

        Thread Safety:
            Uses threading.Event.clear() for atomic state change visible to worker loop.
        """
        if not self._running.is_set():
            logger.warning("[RolloutTaskScheduler] Scheduler is not running")
            return

        logger.info("[RolloutTaskScheduler] Stopping background worker...")

        with self._admission:
            self._running.clear()
            self._admission.notify_all()
        if wait and self._worker_task:
            await self._worker_task

        self._try_shutdown_rollout_engine()
        logger.info("[RolloutTaskScheduler] Background worker stopped")

    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        """
        Get the event loop for the rollout engine.

        If the rollout engine runnning in a seperated thread, it will use the different event loop from main thread.
        Any asyncio operations invoked in main thread will be blocked because of the async task never run in rollout engine thread.
        In this case, we must submit the async task to the rollout engine thread's event loop.

        For example
        ```python
        # main thread
        _f = asyncio.run_coroutine_threadsafe(asyncio.sleep(1), scheduler.get_event_loop())
        _f.result()
        ```
        """
        if not self._running.is_set():
            raise RuntimeError("[RolloutTaskScheduler] Scheduler is not running")

        if self._loop is None:
            raise RuntimeError("[RolloutTaskScheduler] Event loop is not initialized")
        return self._loop

    def pause(self):
        """
        Pause the scheduler from processing new tasks from task_queue.

        Currently running tasks will continue to completion, but no new tasks
        will be started until resume() is called.
        """
        with self._admission:
            if not self._running.is_set():
                logger.warning(
                    "[RolloutTaskScheduler] Cannot pause: scheduler is not running"
                )
                return
            self._manual_pause = True
            self._update_pause_state()

    def resume(self):
        """
        Release the manual pause. Active pause contexts retain their ownership.
        """
        with self._admission:
            self._manual_pause = False
            self._update_pause_state()

    def _update_pause_state(self):
        """Caller holds _admission; only the final owner resumes admission."""
        if self._manual_pause or self._pause_owners:
            self._paused.set()
        else:
            self._paused.clear()

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

        Releases this context's pause when exiting, even after an exception.
        Other contexts and manual pauses remain effective. Nested draining
        contexts still wait for active work. This is an admission barrier, not
        writer mutual exclusion or a backend device fence.

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
            RuntimeError: If stopped, or a draining wait would block its own loop
            TimeoutError: If waiting for active tasks times out
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._admission:
            if not self._running.is_set():
                raise RuntimeError(
                    "[RolloutTaskScheduler] Cannot pause: scheduler is not running"
                )
            self._pause_owners += 1
            self._update_pause_state()
        try:
            if wait_for_active_tasks:
                with self._admission:
                    try:
                        current_loop = asyncio.get_running_loop()
                    except RuntimeError:
                        current_loop = None
                    while self._active_tasks:
                        if current_loop is not None and current_loop is self._loop:
                            raise RuntimeError(
                                "Cannot drain scheduler tasks from their own event loop"
                            )
                        if not self._running.is_set():
                            raise RuntimeError("Scheduler stopped while pausing")
                        remaining = (
                            None if deadline is None else deadline - time.monotonic()
                        )
                        if remaining is not None and remaining <= 0:
                            raise TimeoutError(
                                f"Timeout waiting for {len(self._active_tasks)} active tasks"
                            )
                        self._admission.wait(timeout=remaining)
                    if not self._running.is_set():
                        raise RuntimeError("Scheduler stopped while pausing")
            yield self
        finally:
            with self._admission:
                self._pause_owners -= 1
                self._update_pause_state()

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
        Drain all active tasks from the active tasks set and block processing new tasks from task queue.

        This is useful when we want to update the weights during training.

        Args:
            timeout: Maximum time in seconds to wait for active tasks (None means wait forever)

        Raises:
            TimeoutError: If timeout occurs before all active tasks are drained
        """
        start_time = time.time()
        while len(self._active_tasks) > 0:
            if timeout is not None and (time.time() - start_time) > timeout:
                raise TimeoutError(
                    f"[RolloutTaskScheduler] Timeout waiting for {len(self._active_tasks)} active tasks"
                )
            await asyncio.sleep(0.1)

    def is_idle(self) -> bool:
        """
        Check if the scheduler is idle (running with no tasks and no results).

        Returns:
            True if scheduler is running, has no active tasks, no pending tasks, and no completed results
        """
        return (
            self.is_running()
            and self.is_all_tasks_completed()
            and self.complete_queue.empty()
        )

    def is_busy(self) -> bool:
        """
        Check if the scheduler is busy (running with active tasks more than the max concurrent requests).
        """
        return (
            self.is_running()
            and len(self._active_tasks) >= self.max_concurrent_requests
        )

    def is_all_tasks_completed(self) -> bool:
        """
        Check if all tasks are completed (no pending or active tasks).

        Returns:
            True if task queue is empty and there are no active tasks
        """
        # Queue ownership spans dequeue -> active registration -> terminal
        # publication. Inspecting the queue and active set separately can miss
        # a task in the gap and falsely declare the phase drained.
        with self.task_queue.mutex:
            return self.task_queue.unfinished_tasks == 0

    async def wait_all_tasks_completed(self):
        """
        Wait asynchronously for all pending and active tasks to complete.

        This method blocks until the task queue is empty and all active tasks finish.
        """
        while not self.is_all_tasks_completed():
            await asyncio.sleep(0.1)

    def has_results(self) -> bool:
        """
        Check if there are any completed results available.

        Returns:
            True if results are available, False otherwise
        """
        return not self.complete_queue.empty()

    def active_tasks(self) -> int:
        """
        Get the number of active tasks.

        Returns:
            Number of active tasks
        """
        return len(self._active_tasks)

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
            "active_tasks": self.active_tasks(),
            "pending_tasks": self.pending_tasks(),
            "completed_results": self.completed_results(),
            "max_concurrent_requests": self.max_concurrent_requests,
        }
