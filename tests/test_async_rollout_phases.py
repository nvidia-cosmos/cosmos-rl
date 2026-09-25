# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Async phase/terminal contracts without requiring an external LLM engine."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import os
from queue import Queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch

from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.reward.dispatcher import RewardDispatcher
from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.rollout.worker.asynchronous.rollout_task_scheduler import (
    CompletedRollout,
    RolloutTask,
    RolloutTaskScheduler,
)
from cosmos_rl.rollout.worker.rollout_control import DisaggregatedRolloutControlWorker


@pytest.mark.parametrize("identified", [False, True])
@pytest.mark.parametrize("is_validation", [False, True])
@pytest.mark.parametrize(
    "outcome", ["healthy", "empty", "failed", "extra", "cancelled"]
)
def test_every_started_task_has_one_phase_identified_terminal_result(
    identified, is_validation, outcome
):
    error = {"failed": RuntimeError("injected"), "cancelled": asyncio.CancelledError()}
    engine = SimpleNamespace(
        rollout_generation=AsyncMock(
            return_value=[RolloutResult(completions=["ok"])]
            * {"healthy": 1, "extra": 2}.get(outcome, 0),
            side_effect=error.get(outcome),
        )
    )
    train_packer, val_packer = object(), object()
    scheduler = RolloutTaskScheduler(engine, train_packer, val_data_packer=val_packer)
    payload = RLPayload(
        prompt_idx=0, prompt="p", completion_sequences=[0] if identified else None
    )
    completed = asyncio.run(
        scheduler._generate_single(RolloutTask(0, payload, is_validation))
    )
    assert completed is scheduler.get(block=False)
    assert completed.is_validation is is_validation
    assert completed.payload is payload
    assert completed.result.completions == (["ok"] if outcome == "healthy" else [])
    assert scheduler.total_processed == 1 and scheduler.get(block=False) is None
    assert engine.rollout_generation.call_args.kwargs["is_validation"] is is_validation
    assert engine.rollout_generation.call_args.kwargs["data_packer"] is (
        val_packer if is_validation else train_packer
    )


@pytest.mark.parametrize("is_validation", [False, True])
def test_cancellation_before_first_coroutine_instruction_reports_once(is_validation):
    async def run():
        scheduler = RolloutTaskScheduler(
            SimpleNamespace(rollout_generation=AsyncMock()), object()
        )
        accepted = RolloutTask(0, RLPayload(prompt_idx=0), is_validation)
        scheduler.put_rollout(accepted)
        task = asyncio.create_task(
            scheduler._generate_single(scheduler.task_queue.get_nowait())
        )
        scheduler._active_tasks.add(task)
        task.add_done_callback(lambda task: scheduler._task_finished(accepted, task))
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        completed = scheduler.get_all()
        assert len(completed) == 1 and completed[0].is_validation is is_validation
        assert completed[0].result.completions == []
        assert scheduler.is_all_tasks_completed() and not scheduler._active_tasks
        scheduler.rollout_engine.rollout_generation.assert_not_called()

    asyncio.run(run())


@pytest.mark.parametrize("healthy", [False, True])
def test_terminal_publication_error_is_not_retried_as_a_generation_failure(healthy):
    engine = SimpleNamespace(
        rollout_generation=AsyncMock(
            return_value=[RolloutResult(completions=["ok"])] if healthy else []
        )
    )
    scheduler = RolloutTaskScheduler(engine, object())
    scheduler._publish_terminal = Mock(side_effect=RuntimeError("publication failed"))
    with pytest.raises(RuntimeError, match="publication failed"):
        asyncio.run(scheduler._generate_single(RolloutTask(0, RLPayload(prompt_idx=0))))
    scheduler._publish_terminal.assert_called_once()


def test_dequeued_but_not_registered_task_is_not_a_drained_phase():
    scheduler = RolloutTaskScheduler(SimpleNamespace(), object())
    scheduler._running.set()
    scheduler.put_rollout(RolloutTask(0, RLPayload(prompt_idx=0)))
    scheduler.task_queue.get_nowait()
    assert not scheduler.is_all_tasks_completed()
    assert not scheduler.is_idle()


def test_stopping_scheduler_settles_queued_unlaunched_tasks():
    async def run():
        engine = SimpleNamespace(
            rollout_generation=AsyncMock(), is_engine_initialized=lambda: False
        )
        scheduler = RolloutTaskScheduler(engine, object())
        await scheduler.start_async(lambda _: None)
        for index in range(5):
            scheduler.put_rollout(
                RolloutTask(index, RLPayload(prompt_idx=index), bool(index % 2))
            )
        await scheduler.stop_async()
        results = scheduler.get_all()
        assert [result.idx for result in results] == list(range(5))
        assert [result.is_validation for result in results] == [
            False,
            True,
            False,
            True,
            False,
        ]
        assert all(result.result.completions == [] for result in results)
        assert scheduler.total_processed == 5 and scheduler.is_all_tasks_completed()
        engine.rollout_generation.assert_not_called()

    asyncio.run(run())


@contextmanager
def running_scheduler(engine, train_packer, val_packer):
    engine.is_engine_initialized = lambda: False
    scheduler = RolloutTaskScheduler(
        engine,
        train_packer,
        val_data_packer=val_packer,
        max_concurrent_requests=2,
        check_interval=0.001,
    )
    scheduler.start(lambda _: None, wait_initialized=True)
    try:
        yield scheduler
    finally:
        scheduler.stop()
        assert not scheduler._worker_thread.is_alive()


def phase_worker(scheduler, train_packer):
    worker = object.__new__(DisaggregatedRolloutControlWorker)
    worker.scheduler = scheduler
    worker._is_async_rollout = True
    worker._prompt_queue = Queue()
    worker.rank_in_rollout_repicas = 0
    worker.current_step = worker.current_weight_version = 7
    worker.val_batch_size = 2
    worker.should_report = True
    worker.replica_name = "rollout-test"
    worker.is_diffusers = False
    worker.config = SimpleNamespace(
        validation=SimpleNamespace(n_generation=1),
        rollout=SimpleNamespace(
            n_generation=1, multi_turn_config=SimpleNamespace(enable=False)
        ),
        train=SimpleNamespace(
            non_text=True,
            local_dataset=False,
            train_policy=SimpleNamespace(
                variant="grpo", rollout_as_token_ids=False, bypass_reward=False
            ),
        ),
    )
    worker.data_packer = train_packer
    worker.enqueue_teacher_calculation = lambda payloads: payloads
    worker._report_discarded_samples = Mock()
    worker.shutdown_signal = threading.Event()
    worker.validation_flag = threading.Event()
    worker.validation_flag.set()
    worker.api_client = SimpleNamespace(
        post_rollout_completion=Mock(), post_validation_report=Mock()
    )
    worker.reward_dispatcher = RewardDispatcher()
    worker.reward_dispatcher.is_remote = False
    worker.reward_dispatcher.executor = None
    return worker


@pytest.mark.parametrize("train_outcome", ["healthy", "empty", "failed"])
def test_validation_drains_training_generation_and_rewards_with_same_prompt_indices(
    train_outcome,
):
    train_packer = SimpleNamespace(get_rollout_output=lambda *values: (*values, None))
    val_packer = object()
    device = "cuda" if os.environ.get("COSMOS_REQUIRE_CUDA") == "1" else "cpu"
    if device == "cuda":
        assert torch.cuda.is_available(), "GPU gate must not silently run on CPU"
    observed = []
    train_rewards_done = threading.Event()

    async def generate(*, payloads, data_packer, is_validation, **kwargs):
        payload = payloads[0]
        assert data_packer is (val_packer if is_validation else train_packer)
        assert payload.prompt == ("val" if is_validation else "train")
        if is_validation:
            assert train_rewards_done.is_set()
        await asyncio.sleep(0.01)
        if not is_validation and payload.prompt_idx == 0:
            if train_outcome == "failed":
                raise RuntimeError("injected training generation failure")
            if train_outcome == "empty":
                return []
        # Real device work inside the async scheduler thread, not a backend
        # quiescence/IPC test. Distinct phases intentionally reuse prompt IDs.
        value = torch.full((4, 4), 2.0 if is_validation else 1.0, device=device)
        result = (value @ value).sum().item()
        assert result == (256.0 if is_validation else 64.0)
        observed.append((is_validation, payload.prompt_idx))
        return [RolloutResult(completions=[str(result)])]

    engine = SimpleNamespace(rollout_generation=generate)
    with running_scheduler(engine, train_packer, val_packer) as scheduler:
        worker = phase_worker(scheduler, train_packer)
        with ThreadPoolExecutor(max_workers=2) as rewards:

            def calculate(payloads, is_validation, step, **kwargs):
                if not is_validation:
                    time.sleep(0.02)
                    train_rewards_done.set()
                for payload in payloads:
                    payload.rewards = [1.0]
                return payloads, is_validation, step

            def enqueue(payloads, is_validation, step, **kwargs):
                worker.reward_dispatcher.task_queue.put(
                    rewards.submit(calculate, payloads, is_validation, step)
                )

            worker.reward_dispatcher.enqueue_rewards_cal = enqueue
            submitted = set()

            def fetch(count, queue, validation_step, **kwargs):
                assert validation_step == 7
                assert scheduler.is_idle()
                assert worker.reward_dispatcher.is_empty()
                expected = 2 if train_outcome == "healthy" else 1
                reports = worker.api_client.post_rollout_completion.call_args_list
                assert sum(len(call.args[0].payloads) for call in reports) == expected
                assert not submitted
                submitted.add("validation")
                queue.put([RLPayload(prompt_idx=i, prompt="val") for i in range(2)])
                return True

            worker.request_new_prompts = fetch
            scheduler.put_rollout_batch(
                [
                    RolloutTask(i, RLPayload(prompt_idx=i, prompt="train"))
                    for i in range(2)
                ]
            )
            worker.do_validation()
            assert scheduler.is_idle() and scheduler.total_processed == 4
            assert observed.count((True, 0)) == observed.count((True, 1)) == 1
            reports = worker.api_client.post_validation_report.call_args_list
            assert sum(len(call.args[0].payloads) for call in reports) == 2
            assert all(
                p.prompt == "val" for call in reports for p in call.args[0].payloads
            )
            assert not worker.validation_flag.is_set()
            assert worker.reward_dispatcher.is_empty()
            assert sum(
                call.args[0] for call in worker._report_discarded_samples.call_args_list
            ) == int(train_outcome != "healthy")


@pytest.mark.parametrize("outcome", ["empty", "failed", "partial", "surplus"])
def test_incomplete_validation_fails_explicitly_without_fabricated_rewards(outcome):
    engine = SimpleNamespace(
        rollout_generation=AsyncMock(
            return_value=[
                RolloutResult(completions=["ok"] * (1 if outcome == "partial" else 3))
            ]
            if outcome in ("partial", "surplus")
            else [],
            side_effect=RuntimeError("injected") if outcome == "failed" else None,
        )
    )
    packer = object()
    with running_scheduler(engine, packer, packer) as scheduler:
        worker = phase_worker(scheduler, packer)
        worker.config.validation.n_generation = 2
        worker.reward_dispatcher.enqueue_rewards_cal = Mock()

        def fetch(count, queue, **kwargs):
            queue.put([RLPayload(prompt_idx=0, prompt="val")])
            return True

        worker.request_new_prompts = fetch
        with pytest.raises(RuntimeError, match="Incomplete validation generation"):
            worker.do_validation()
        worker.reward_dispatcher.enqueue_rewards_cal.assert_not_called()
        worker.api_client.post_validation_report.assert_not_called()


def test_validation_result_cannot_be_consumed_as_training():
    worker = object.__new__(DisaggregatedRolloutControlWorker)
    worker.scheduler = SimpleNamespace(
        get_all=lambda: [
            CompletedRollout(
                0, RLPayload(prompt_idx=0), RolloutResult(completions=["val"]), True
            )
        ]
    )
    worker._filter_valid_rollout_results_and_report = Mock()
    with pytest.raises(RuntimeError, match="Validation result crossed"):
        worker._stream_generation_collect_results()
    worker._filter_valid_rollout_results_and_report.assert_not_called()
