# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Completion-driven reward dequeue and the real blocking report loops."""

from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
import threading
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest

from cosmos_rl.reward.dispatcher import RewardDispatcher
from cosmos_rl.rollout.worker.colocated.rollout_control import (
    ColocatedRolloutControlWorker,
)
from cosmos_rl.rollout.worker.rollout_control import DisaggregatedRolloutControlWorker
from test_trt_validation_delivery import lifecycle_method


PENDING = (None, False, -1, False)
EMPTY = (None, False, -1, True)


class ObservedFuture(Future):
    def __init__(self):
        super().__init__()
        self.waiting = threading.Event()
        self.second_wait = threading.Event()
        self.wait_count = 0

    def exception(self, timeout=None):
        self.wait_count += 1
        self.waiting.set()
        if self.wait_count >= 2:
            self.second_wait.set()
        return super().exception(timeout=timeout)


def dispatcher():
    owner = RewardDispatcher()
    owner.is_remote = False
    return owner


@pytest.mark.parametrize("block", [False, True])
@pytest.mark.parametrize("kind", ["empty", "bypass", "completed"])
def test_immediate_paths_never_wait(block, kind):
    owner = dispatcher()
    result = ([object()], True, 7)
    if kind == "bypass":
        owner.task_queue.put(result)
    elif kind == "completed":
        future = ObservedFuture()
        future.set_result(result)
        owner.task_queue.put(future)
    expected = EMPTY if kind == "empty" else (*result, False)
    assert owner.dequeue_rewards_cal(block=block) == expected
    assert owner.is_empty()
    if kind == "completed":
        assert future.wait_count == 0


def test_default_nonblocking_call_does_not_wait_for_head():
    owner, future = dispatcher(), ObservedFuture()
    owner.task_queue.put(future)
    assert owner.dequeue_rewards_cal() == PENDING
    assert not owner.is_empty()
    assert future.wait_count == 0


def test_wait_releases_queue_and_preserves_fifo_when_later_task_finishes_first():
    owner, first, second = dispatcher(), ObservedFuture(), Future()
    a, b = ([object()], False, 1), ([object()], True, 2)
    owner.task_queue.put(first)
    second.set_result(b)
    with ThreadPoolExecutor(2) as threads:
        waiting = threads.submit(owner.dequeue_rewards_cal, block=True)
        try:
            assert first.waiting.wait(2)
            # A producer and another nonblocking consumer must both be able to
            # progress while the blocking caller waits for the FIFO head.
            threads.submit(owner.task_queue.put, second).result(timeout=2)
            assert (
                threads.submit(owner.dequeue_rewards_cal).result(timeout=2) == PENDING
            )
            assert not waiting.done()
        finally:
            first.set_result(a)
        assert waiting.result(timeout=2) == (*a, False)
    assert first.wait_count == 1
    assert owner.dequeue_rewards_cal(block=True) == (*b, False)
    assert owner.dequeue_rewards_cal(block=True) == EMPTY


@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("outcome", ["error", "timeout_error", "cancel"])
def test_task_errors_and_cancellation_are_dequeued_and_propagated_once(
    pending, outcome
):
    owner, future = dispatcher(), ObservedFuture()
    owner.task_queue.put(future)
    error = (
        TimeoutError("reward task failed")
        if outcome == "timeout_error"
        else ValueError("reward task failed")
    )

    def finish():
        if outcome == "cancel":
            assert future.cancel()
        else:
            future.set_exception(error)

    if not pending:
        finish()
    with ThreadPoolExecutor(1) as threads:
        waiting = threads.submit(owner.dequeue_rewards_cal, block=True)
        if pending:
            try:
                assert future.waiting.wait(2)
            finally:
                finish()
        with pytest.raises(
            CancelledError if outcome == "cancel" else type(error)
        ) as caught:
            waiting.result(timeout=2)
    if outcome != "cancel":
        assert caught.value is error
    assert owner.dequeue_rewards_cal() == EMPTY


def test_shutdown_check_interval_is_not_a_reward_deadline():
    owner, future = dispatcher(), ObservedFuture()
    owner.task_queue.put(future)
    with ThreadPoolExecutor(1) as threads:
        waiting = threads.submit(owner.dequeue_rewards_cal, block=True)
        try:
            assert future.second_wait.wait(3)
            assert not waiting.done()
        finally:
            future.set_result(([], True, 7))
        assert waiting.result(timeout=2) == ([], True, 7, False)


def test_owner_stop_interrupts_wait_without_consuming_or_canceling_work():
    owner, future, stop = dispatcher(), ObservedFuture(), threading.Event()
    owner.task_queue.put(future)
    with ThreadPoolExecutor(1) as threads:
        waiting = threads.submit(owner.dequeue_rewards_cal, block=True, stop_event=stop)
        try:
            assert future.waiting.wait(2)
            stop.set()
            with pytest.raises(CancelledError, match="shutdown"):
                waiting.result(timeout=2)
            assert not future.done()
            assert not owner.is_empty()
        finally:
            future.set_result(([], False, 7))
    assert owner.dequeue_rewards_cal() == ([], False, 7, False)


def test_dispatcher_shutdown_interrupts_wait_and_retains_running_task_resources():
    owner, release = dispatcher(), threading.Event()
    owner.executor = ThreadPoolExecutor(1)
    entered = threading.Event()

    def calculate():
        entered.set()
        assert release.wait(10)
        return [], False, 7

    running = owner.executor.submit(calculate)
    head = ObservedFuture()
    running.add_done_callback(lambda task: head.set_result(task.result()))
    owner.task_queue.put(head)
    with ThreadPoolExecutor(1) as threads:
        waiting = threads.submit(owner.dequeue_rewards_cal, block=True)
        try:
            assert entered.wait(2) and head.waiting.wait(2)
            with pytest.raises(TimeoutError, match="Reward executor"):
                owner.shutdown(timeout_s=0.01)
            with pytest.raises(CancelledError, match="shutdown"):
                waiting.result(timeout=2)
            assert owner.executor is not None
            assert not owner.is_empty() and not running.done()
        finally:
            release.set()
            owner.shutdown(timeout_s=5)
    assert owner.dequeue_rewards_cal() == ([], False, 7, False)


def test_concurrent_waiters_claim_each_task_once():
    owner, first, second = dispatcher(), ObservedFuture(), Future()
    a, b = ([object()], False, 1), ([object()], False, 2)
    owner.task_queue.put(first)
    owner.task_queue.put(second)
    with ThreadPoolExecutor(2) as threads:
        calls = [
            threads.submit(owner.dequeue_rewards_cal, block=True) for _ in range(2)
        ]
        try:
            assert first.second_wait.wait(2)
        finally:
            first.set_result(a)
            second.set_result(b)
        results = [call.result(timeout=2) for call in calls]
    assert sorted(result[2] for result in results) == [1, 2]
    assert owner.dequeue_rewards_cal() == EMPTY


@pytest.mark.parametrize("block", [False, True])
def test_remote_reward_contract_is_unchanged(block):
    owner = dispatcher()
    owner.is_remote = True
    owner.task_queue.put("remote-request")
    owner.reward_calculator = SimpleNamespace(
        get_results=Mock(return_value=(None, False, -1))
    )
    assert owner.dequeue_rewards_cal(block=block) == PENDING
    owner.reward_calculator.get_results.assert_called_once_with(owner.task_queue)
    assert not owner.is_empty()


@pytest.fixture(params=["disaggregated", "colocated", "trt"])
def reporter(request):
    if request.param == "trt":
        worker = SimpleNamespace()
        worker.report_rollouts = MethodType(lifecycle_method("report_rollouts"), worker)
    else:
        cls = (
            ColocatedRolloutControlWorker
            if request.param == "colocated"
            else DisaggregatedRolloutControlWorker
        )
        worker = object.__new__(cls)
    worker.reward_dispatcher, worker.shutdown_signal = dispatcher(), threading.Event()
    return worker


def test_native_blocking_report_waits_for_completion(reporter):
    owner, future = reporter.reward_dispatcher, ObservedFuture()
    owner.task_queue.put(future)
    with ThreadPoolExecutor(1) as threads:
        waiting = threads.submit(reporter.report_rollouts, block=True)
        try:
            assert future.waiting.wait(1), (
                "native report busy-polls instead of waiting on completion"
            )
            assert not waiting.done()
        finally:
            future.set_result(([], True, 7))
        assert waiting.result(timeout=2) == ([], True, 7, False)
    assert future.wait_count == 1 and owner.is_empty()


def test_native_nonblocking_report_remains_immediate(reporter):
    owner, future = reporter.reward_dispatcher, ObservedFuture()
    owner.task_queue.put(future)
    assert reporter.report_rollouts() == PENDING
    assert future.wait_count == 0 and not owner.is_empty()


def test_native_report_observes_owner_shutdown(reporter):
    owner, future = reporter.reward_dispatcher, ObservedFuture()
    owner.task_queue.put(future)
    with ThreadPoolExecutor(1) as threads:
        waiting = threads.submit(reporter.report_rollouts, block=True)
        try:
            assert future.waiting.wait(2)
            reporter.shutdown_signal.set()
            with pytest.raises(CancelledError, match="shutdown"):
                waiting.result(timeout=2)
            assert not future.done() and not owner.is_empty()
        finally:
            future.set_result(([], True, 7))


@pytest.mark.parametrize("error_type", [ValueError, TimeoutError, CancelledError])
def test_native_report_propagates_task_failure_once(reporter, error_type):
    owner, future = reporter.reward_dispatcher, ObservedFuture()
    owner.task_queue.put(future)
    error = error_type("reward failed")
    with ThreadPoolExecutor(1) as threads:
        waiting = threads.submit(reporter.report_rollouts, block=True)
        try:
            assert future.waiting.wait(2)
        finally:
            future.set_exception(error)
        with pytest.raises(error_type) as caught:
            waiting.result(timeout=2)
    assert caught.value is error and owner.is_empty()


def test_native_enqueue_cancellation_settles_reward_reader_once():
    from test_reward_callbacks import Packer, payload

    owner, future, packer = dispatcher(), ObservedFuture(), Packer()
    owner.data_packer = packer
    owner.executor = Mock(submit=Mock(return_value=future))
    owner.enqueue_rewards_cal([payload()], False, 7)
    with ThreadPoolExecutor(1) as threads:
        waiting = threads.submit(owner.dequeue_rewards_cal, block=True)
        try:
            assert future.waiting.wait(2)
        finally:
            assert future.cancel()
        with pytest.raises(CancelledError):
            waiting.result(timeout=2)
    assert owner.dequeue_rewards_cal() == EMPTY
    assert len(packer.notifications) == 1
    assert isinstance(packer.notifications[0][2], CancelledError)
