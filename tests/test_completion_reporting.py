# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import pickle
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock
import subprocess
import sys

import pytest

from cosmos_rl.dispatcher.controller import Controller
from cosmos_rl.dispatcher.data.admission_state import CompletionAdmissionState
from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.reward.admission import (
    resolve_completion_admission,
    select_payload_completions,
)
from cosmos_rl.reward.identity import CompletionReporter
from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.rollout.worker.rollout_control import DisaggregatedRolloutControlWorker
from cosmos_rl.rollout.worker.asynchronous.rollout_task_scheduler import (
    RolloutTaskScheduler,
    RolloutTask,
)


def worker():
    w = object.__new__(DisaggregatedRolloutControlWorker)
    w.config = SimpleNamespace(
        rollout=SimpleNamespace(n_generation=3, async_r2r_sync="disabled")
    )
    w.current_weight_version = 7
    w.completion_reporter = CompletionReporter("source", 0)
    w.api_client = SimpleNamespace(post_rollout_completion=Mock(return_value=True))
    return w


@pytest.mark.parametrize("outcome", ["success", "empty", "exception"])
def test_reservations_exist_before_generation_and_failures_keep_them(outcome):
    w = worker()
    payloads = [RLPayload(prompt_idx=4), RLPayload(prompt_idx=4)]

    def generate(**kwargs):
        assert [p.completion_sequences for p in payloads] == [[0, 1, 2], [3, 4, 5]]
        assert kwargs["current_weight_version"] == 7
        if outcome == "exception":
            raise RuntimeError("generation fault")
        return (
            []
            if outcome == "empty"
            else [RolloutResult(completions=["a", "b", "c"])] * 2
        )

    w.rollout = SimpleNamespace(rollout_generation=generate)
    if outcome == "exception":
        with pytest.raises(RuntimeError, match="generation fault"):
            w._call_rollout_generation(payloads=payloads, is_validation=False)
    else:
        w._call_rollout_generation(payloads=payloads, is_validation=False)
    if outcome != "success":
        report = w.api_client.post_rollout_completion.call_args.args[0]
        assert [f.identity.sequence for f in report.completion_failures] == list(
            range(6)
        )
        assert all(f.identity.weight_version == 7 for f in report.completion_failures)
    else:
        w.api_client.post_rollout_completion.assert_not_called()


def test_selection_preserves_ids_versions_rejected_refs_and_pickle_boundary():
    reporter = CompletionReporter("source", 1)
    source = RLPayload(
        completions=["a", "bad", "c"],
        completion_trainable=[True, False, True],
        completion_drop_reasons=[None, "quality", None],
    )
    reporter.reserve([source], 3, 7)
    source = pickle.loads(pickle.dumps(source))
    selected = select_payload_completions(
        source, resolve_completion_admission(source, 2)
    )
    packer = SimpleNamespace(get_rollout_output=lambda *values: (*values, None))
    report = reporter.report([selected], packer)
    assert [i.sequence for i in report.completion_identities] == [0, 2]
    failure = report.completion_failures[0]
    assert failure.identity.sequence == 1
    assert failure.identity.weight_version == 7
    assert failure.payload.completion == "bad"
    assert "completion_sequences" not in report.model_dump()["payloads"][0]
    assert (
        report.model_dump() == report.model_dump()
    )  # serialization never allocates IDs


def test_insufficient_group_reports_every_reservation():
    source = RLPayload(
        completions=["a", "b", "c"], completion_trainable=[True, False, False]
    )
    reporter = CompletionReporter("source", 0)
    reporter.reserve([source], 3, 2)
    selected = select_payload_completions(
        source, resolve_completion_admission(source, 2)
    )
    report = reporter.report(
        [selected], SimpleNamespace(get_rollout_output=lambda *values: (*values, None))
    )
    assert report.payloads == []
    assert report.completion_identities == []
    assert [f.identity.sequence for f in report.completion_failures] == [0, 1, 2]
    assert all(f.reason == "insufficient_group" for f in report.completion_failures)


def test_exhausted_report_delivery_is_not_silent():
    w = worker()
    w.api_client.post_rollout_completion.return_value = False
    with pytest.raises(RuntimeError, match="delivery failed"):
        w._post_identified_report(
            w.completion_reporter.generation_failure([], "generation_error")
        )


def test_validation_does_not_allocate_training_identities():
    w = worker()
    w.rollout = SimpleNamespace(rollout_generation=lambda **kwargs: [])
    payload = RLPayload()
    w._call_rollout_generation(payloads=[payload], is_validation=True)
    assert payload.completion_sequences is None
    w.api_client.post_rollout_completion.assert_not_called()


def test_identification_does_not_force_a_quality_mask_on_legacy_groups():
    w = worker()
    w.should_report = True
    w.config.rollout.multi_turn_config = SimpleNamespace(enable=False)
    w.config.train = SimpleNamespace(
        non_text=True,
        local_dataset=False,
        train_policy=SimpleNamespace(bypass_reward=False),
    )
    w.enqueue_teacher_calculation = lambda payloads: payloads
    w.reward_dispatcher = SimpleNamespace(enqueue_rewards_cal=Mock())
    payload = RLPayload(completion_sequences=[0], weight_version=7)
    w._enqueue_masked_results([RolloutResult(completions=["one"])], [payload])
    assert payload.completion_trainable is None
    assert payload.completion_sequences == [0]
    assert w.reward_dispatcher.enqueue_rewards_cal.call_args.args[2] == 7


def test_non_reporting_rank_never_enqueues_masked_results():
    w = worker()
    w.should_report = False
    w.reward_dispatcher = SimpleNamespace(enqueue_rewards_cal=Mock())
    w._enqueue_masked_results(
        [RolloutResult(completions=["a"], completion_trainable=[False])], [RLPayload()]
    )
    w.reward_dispatcher.enqueue_rewards_cal.assert_not_called()
    w.api_client.post_rollout_completion.assert_not_called()


@pytest.mark.parametrize("raises", [False, True])
def test_async_generation_failure_returns_reserved_payload(raises):
    scheduler = object.__new__(RolloutTaskScheduler)
    scheduler.rollout_engine = SimpleNamespace(
        rollout_generation=AsyncMock(
            side_effect=RuntimeError("fault") if raises else None, return_value=[]
        )
    )
    scheduler.stream = scheduler.data_packer = None
    scheduler.complete_queue = Queue()
    payload = RLPayload(completion_sequences=[12, 13], weight_version=9)
    result = asyncio.run(scheduler._generate_single(RolloutTask(1, payload)))
    assert result is scheduler.complete_queue.get_nowait()
    assert result.payload.completion_sequences == [12, 13]
    assert result.result.completions == []
    call = scheduler.rollout_engine.rollout_generation.call_args
    assert call.kwargs["payloads"][0].weight_version == 9
    assert (
        "current_weight_version" not in call.kwargs
    )  # async engine API has no such parameter


@pytest.mark.parametrize("stage", ["settle", "filter", "buffer"])
def test_partial_settlement_failure_exits_process_nonzero(stage):
    program = """
import asyncio
from types import SimpleNamespace
from cosmos_rl.dispatcher.controller import Controller
def fail(*args):
    raise RuntimeError("injected settlement failure")
async def buffer(*args):
    if STAGE == "buffer": fail()
c = object.__new__(Controller)
c.life_cycle_lock = asyncio.Lock()
c.completion_admission = SimpleNamespace(prepare=lambda *args: None, settle=fail if STAGE == "settle" else lambda *args: [], failed=False)
c.policy_status_manager = SimpleNamespace(filter_outdated_rollouts=fail if STAGE == "filter" else lambda x: x)
c.put_rollouts = buffer
asyncio.run(c.put_application_rollouts(None, []))
raise SystemExit(0)
"""
    result = subprocess.run(
        [sys.executable, "-c", f"STAGE={stage!r}\n" + program],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 86, result.stderr
    assert "terminating controller" in result.stderr


def test_prepare_error_remains_atomic_and_nonfatal():
    c = object.__new__(Controller)
    c.life_cycle_lock = asyncio.Lock()
    c.completion_admission = CompletionAdmissionState()
    c.completion_admission.prepare = Mock(side_effect=ValueError("bad report"))
    with pytest.raises(ValueError, match="bad report"):
        asyncio.run(c.put_application_rollouts(None, []))
    assert not c.completion_admission.failed
