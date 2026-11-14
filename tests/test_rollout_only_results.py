# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Generation/scoring/result ownership: no GRPO preparation or result loss."""

import asyncio
import importlib.util
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import pytest
import torch

from cosmos_rl.dispatcher import run_web_panel as web
from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.dispatcher.protocol import RolloutRequest
from cosmos_rl.reward import local_calculator, remote_calculator
from cosmos_rl.reward.dispatcher import RewardDispatcher
from cosmos_rl.reward.local_calculator import LocalRewardCalculator
from cosmos_rl.reward.remote_calculator import RemoteRewardCalculator
from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.rollout.worker.rollout_control import DisaggregatedRolloutControlWorker
from cosmos_rl.utils.application_callbacks import (
    CallbackContext,
    CallbackRegistry,
    CallbackRegistration,
)
import test_rollout_only as fixtures
import test_rollout_resource_ownership as ownership
from cosmos_rl.rollout.worker.asynchronous.rollout_task_scheduler import (
    CompletedRollout,
)


def forbidden(*args, **kwargs):
    raise AssertionError("Training preparation must not execute")


@pytest.mark.parametrize("remote", [False, True])
def test_unscored_setup_does_not_construct_any_scorer_or_tokenizer(monkeypatch, remote):
    config = fixtures.configuration()
    data = config.model_dump()
    data["train"]["train_policy"].update(
        bypass_reward=True, use_remote_reward=remote, reward_function={}
    )
    data["validation"]["use_remote_reward"] = remote
    config = type(config).model_validate(data)
    monkeypatch.setattr(
        LocalRewardCalculator, "_instance", LocalRewardCalculator(), raising=False
    )
    monkeypatch.setattr(local_calculator.util, "setup_tokenizer", forbidden)
    monkeypatch.setattr(local_calculator, "Reward", forbidden)
    monkeypatch.setattr(RemoteRewardCalculator, "get_instance", forbidden)
    dispatch = RewardDispatcher()
    dispatch.setup(config, num_workers=2)
    try:
        payload = RLPayload(
            prompt_idx=0,
            completions=[""],
            completion_trainable=[False],
            completion_drop_reasons=["quality"],
        )
        dispatch.enqueue_rewards_cal([payload], False, 0, bypass_reward=True)
        result, _, _, _ = dispatch.dequeue_rewards_cal()
        assert result == [payload]
        assert payload.completions == [""]
        assert payload.completion_trainable == [False]
        assert payload.rewards == [0.0]  # internal wire placeholder
        assert dispatch.executor is None
    finally:
        dispatch.shutdown()


@pytest.mark.parametrize("mask", [[False], [True], [False, False, False, False]])
def test_local_scoring_preserves_every_completion_without_reference_or_advantages(
    monkeypatch, mask
):
    config = fixtures.configuration()
    seen = []

    def score(completion, reference, **kwargs):
        assert reference is None
        seen.append(completion)
        return len(completion), {"length": len(completion)}

    monkeypatch.setattr(local_calculator.util, "setup_tokenizer", lambda _: None)
    monkeypatch.setattr(local_calculator.util, "find_maximal_prefix_groups", forbidden)
    monkeypatch.setattr(local_calculator, "REGISTERED_ALGOs", {})
    calculator = LocalRewardCalculator()
    calculator.setup(config, reward_fns=[score])
    payload = RLPayload(
        prompt_idx=0,
        prompt="question",
        reference_answer=None,
        completions=["x" * index for index in range(len(mask))],
        completion_trainable=mask,
        completion_drop_reasons=["quality"] * len(mask),
        training_work_id="work",
        training_completion_slots=list(range(len(mask))),
    )
    outputs, validation, step = calculator.compute_rewards([payload], False, 0)
    assert outputs[0] is payload
    assert not validation and step == 0
    assert seen == payload.completions
    assert payload.completion_trainable == mask
    assert payload.training_completion_slots == list(range(len(mask)))
    assert payload.rewards == list(range(len(mask)))
    assert payload.advantages == [0.0] * len(mask)
    assert payload.n_ignore_prefix_tokens == [0] * len(mask)


@pytest.mark.parametrize("size", [1, 4])
def test_remote_scoring_preserves_low_quality_outputs_and_native_slots(
    monkeypatch, size
):
    config = fixtures.configuration()
    monkeypatch.setattr(remote_calculator, "REGISTERED_ALGOs", {})
    monkeypatch.setattr(
        remote_calculator, "Wan2pt1VAEInterface", lambda **_: None, raising=False
    )
    calculator = RemoteRewardCalculator()
    calculator.setup(config)
    enqueued = []
    calculator.enqueue_request = lambda array, data: (
        enqueued.append((array, data)) or ("request", "replica")
    )
    payload = RLPayload(
        prompt_idx=0,
        prompt={"prompt": "question"},
        reference_answer=None,
        completions=torch.ones(size, 3, 2, 2),
        completion_trainable=[False] * size,
        completion_drop_reasons=["quality"] * size,
        extra_info={"modality": "image"},
        training_work_id="work",
        training_completion_slots=list(range(size)),
    )
    original = payload.completions
    request = calculator.compute_rewards([payload], False, 0)
    calculator.fetch_reward = lambda *_: torch.arange(size, dtype=torch.float32)
    pending = Queue()
    pending.put(request)
    result, validation, step = calculator.get_results(pending)
    assert len(enqueued) == 1
    assert result[0] is payload and payload.completions is original
    assert payload.completion_trainable == [False] * size
    assert payload.training_completion_slots == list(range(size))
    assert payload.rewards == list(range(size))
    assert payload.advantages == [0.0] * size
    assert not validation and step == 0


def pipeline(monkeypatch, *, count=1, bypass=True, sink=None, report_failure=False):
    order, events = [], []
    ctrl = fixtures.controller()
    ctrl.config.rollout.n_generation = count
    ctrl.config.train.local_dataset = False
    ctrl.config.train.train_policy.bypass_reward = bypass
    ctrl.data_fetcher.get_batched_prompt.return_value = (
        [RLPayload(prompt_idx=0, prompt="question")],
        False,
    )
    replica, fetch = fixtures.fetch_source(ctrl, monkeypatch)
    source = RLPayload.model_validate(fetch(1, 0)["payloads_list"][0])
    ctrl.config.controller_execution_id = "execution"
    ctrl.policy_status_manager._publish_payload_transport_cleanup = (
        lambda *_: order.append("cleanup")
    )
    worker = object.__new__(DisaggregatedRolloutControlWorker)
    worker.config, worker.should_report = ctrl.config, True
    worker.replica_name, worker.global_rank, worker.current_weight_version = (
        "source",
        0,
        0,
    )

    def consume(event):
        order.append("sink")
        events.append(event)
        if sink is not None:
            sink(event)

    worker.application_callbacks = CallbackRegistry(
        {"rollout_result": consume}, CallbackContext("rollout", "source", 0)
    )
    monkeypatch.setattr(
        LocalRewardCalculator, "_instance", LocalRewardCalculator(), raising=False
    )
    monkeypatch.setattr(local_calculator.util, "setup_tokenizer", lambda _: None)
    worker.reward_dispatcher = RewardDispatcher()
    worker.reward_dispatcher.setup(
        ctrl.config,
        reward_fns=[lambda value, reference, **_: float(len(value))],
        num_workers=0,
    )

    def pack(*values):
        order.append("pack")
        assert events  # Consumer completed while original completion data was valid.
        return (*values, None)

    worker.data_packer = SimpleNamespace(get_rollout_output=pack)
    sequence = 0

    def report(request):
        nonlocal sequence
        order.append("report")
        if report_failure:
            raise RuntimeError("native report failed")
        request.report_session_id, request.report_sequence = "session", sequence
        request.src_global_rank, request.controller_execution_id = 0, "execution"
        sequence += 1
        wire = request.model_dump()
        # A lost HTTP response replays this exact request, after sink invocation.
        result = asyncio.run(web.put_rollout_group(RolloutRequest.model_validate(wire)))
        retry = asyncio.run(web.put_rollout_group(RolloutRequest.model_validate(wire)))
        assert result == retry and not hasattr(result, "status_code")
        return True

    worker.api_client = SimpleNamespace(post_rollout_completion=report)
    return worker, source, ctrl, order, events


@pytest.mark.parametrize("bypass", [False, True])
def test_result_sink_precedes_packing_and_native_cleanup_with_retry(
    monkeypatch, bypass
):
    worker, source, ctrl, order, events = pipeline(monkeypatch, bypass=bypass)
    result = RolloutResult(
        completions=[""],
        completion_trainable=[False],
        completion_drop_reasons=["quality"],
    )
    worker._filter_valid_rollout_results_and_report([result], [source])
    worker.report_rollouts()
    assert order == ["sink", "pack", "report", "cleanup"]
    assert len(events) == 1
    event = events[0]
    assert (event.execution_id, event.work_id, event.prompt_idx) == (
        "execution",
        source.training_work_id,
        0,
    )
    assert event.model_name_or_path == ctrl.config.policy.model_name_or_path
    assert event.model_revision is None and event.weight_version == 0
    outcome = event.outcomes[0]
    assert outcome.slot == 0 and outcome.status == "generated"
    assert (
        outcome.completion == ""
        and outcome.quality_trainable is False
        and outcome.quality_reason == "quality"
    )
    assert outcome.score == (None if bypass else 0.0)
    assert ctrl.rollout_only_state.outstanding_completions == 0
    assert ctrl.policy_status_manager.samples_on_the_fly == 0
    assert ctrl.policy_status_manager.rollout_buffer.empty()


def test_partial_and_empty_generation_emit_all_native_slot_outcomes(monkeypatch):
    worker, source, ctrl, order, events = pipeline(monkeypatch, count=3)
    worker._filter_valid_rollout_results_and_report(
        [RolloutResult(completions=["one"])], [source]
    )
    worker.report_rollouts()
    assert [item.status for item in events[0].outcomes] == [
        "generated",
        "generation_failed",
        "generation_failed",
    ]
    assert [item.slot for item in events[0].outcomes] == [0, 1, 2]
    assert ctrl.rollout_only_state.outstanding_completions == 0
    worker, source, ctrl, order, events = pipeline(monkeypatch, count=2)
    worker._filter_valid_rollout_results_and_report(
        [RolloutResult(completions=[])], [source]
    )
    assert [item.status for item in events[0].outcomes] == ["generation_failed"] * 2
    assert order == ["sink", "report"]
    assert ctrl.rollout_only_state.outstanding_completions == 0


def test_required_consumer_failure_prevents_packing_and_native_settlement(monkeypatch):
    def fail(event):
        raise RuntimeError("sink failed")

    worker, source, ctrl, order, _ = pipeline(monkeypatch, sink=fail)
    worker._filter_valid_rollout_results_and_report(
        [RolloutResult(completions=["x"])], [source]
    )
    with pytest.raises(RuntimeError, match="sink failed"):
        worker.report_rollouts()
    assert order == ["sink"]
    assert ctrl.rollout_only_state.outstanding_completions == 1


def test_failed_native_delivery_does_not_reinvoke_consumer(monkeypatch):
    worker, source, _, order, events = pipeline(monkeypatch, report_failure=True)
    worker._filter_valid_rollout_results_and_report(
        [RolloutResult(completions=["x"])], [source]
    )
    with pytest.raises(RuntimeError, match="native report failed"):
        worker.report_rollouts()
    worker.report_rollouts()
    assert order == ["sink", "pack", "report"] and len(events) == 1


def test_generation_error_attempts_every_failure_consumer_and_preserves_first_error(
    monkeypatch,
):
    worker, source, _, _, events = pipeline(
        monkeypatch,
        count=2,
        sink=lambda _: (_ for _ in ()).throw(RuntimeError("sink failed")),
    )
    worker.rollout = SimpleNamespace(
        rollout_generation=lambda **_: (_ for _ in ()).throw(
            ValueError("generation failed")
        )
    )
    other = source.model_copy(update={"training_work_id": "other-work"})
    with pytest.raises(ValueError, match="generation failed"):
        worker._call_rollout_generation(payloads=[source, other])
    assert len(events) == 2
    assert all(
        item.status == "generation_failed"
        for event in events
        for item in event.outcomes
    )


def test_non_reporting_rank_never_invokes_result_consumer(monkeypatch):
    worker, source, _, _, events = pipeline(monkeypatch)
    worker.should_report = False
    worker._filter_valid_rollout_results_and_report(
        [RolloutResult(completions=[])], [source]
    )
    assert not events


def test_result_consumer_rejects_best_effort_queue():
    registry = CallbackRegistry(
        {"rollout_result": CallbackRegistration(lambda _: None, policy="best_effort")},
        CallbackContext("rollout"),
    )
    with pytest.raises(ValueError, match="required"):
        registry.validate("rollout_result", borrowed=True, required=True)


def test_jsonl_example_copies_real_values_and_rejects_opaque_data(tmp_path):
    path = Path(__file__).resolve().parents[1] / "examples/rollout_only_jsonl.py"
    spec = importlib.util.spec_from_file_location("rollout_jsonl_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "results.jsonl"
    sink = module.JSONLResults(output)
    sink(
        {
            "tensor": torch.tensor([1.0, 2.0]),
            "reference": {"uri": "artifact://completion"},
        }
    )
    sink.close()
    import json

    recorded = json.loads(output.read_text())
    assert recorded["tensor"]["values"] == [1.0, 2.0]
    assert recorded["reference"]["uri"] == "artifact://completion"
    with pytest.raises(TypeError, match="artifact reference"):
        module.json_value(object())


def test_scoring_errors_are_strict_in_local_and_remote_paths(monkeypatch):
    config = fixtures.configuration()
    monkeypatch.setattr(local_calculator.util, "setup_tokenizer", lambda _: None)
    calculator = LocalRewardCalculator()
    calculator.setup(
        config,
        reward_fns=[
            lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("score failed"))
        ],
    )
    with pytest.raises(ValueError, match="score failed"):
        calculator.compute_rewards(
            [RLPayload(prompt_idx=0, completions=["x"])], False, 0
        )
    remote = RemoteRewardCalculator()
    remote.rollout_only = True
    remote.uuid2stage = {"request": "training"}
    remote.fetch_reward = lambda *_: (_ for _ in ()).throw(
        ValueError("remote score failed")
    )
    pending = Queue()
    pending.put("request")
    with pytest.raises(ValueError, match="remote score failed"):
        remote.get_results(pending)


@pytest.mark.parametrize("close_failure", [False, True])
def test_required_sink_failure_closes_owned_resources_and_keeps_first_error(
    monkeypatch, close_failure
):
    trace = []

    class Sink:
        def __call__(self, event):
            trace.append("sink")
            raise ValueError("original sink error")

        def close(self):
            trace.append("sink close")
            if close_failure:
                raise RuntimeError("secondary close error")

    worker, source, _, _, _ = pipeline(monkeypatch)
    callbacks = CallbackRegistry({"rollout_result": Sink()}, CallbackContext("rollout"))
    worker.application_callbacks = callbacks
    worker._filter_valid_rollout_results_and_report(
        [RolloutResult(completions=["x"])], [source]
    )
    owner = ownership.control(trace)
    owner.application_callbacks = callbacks
    owner.main_loop = worker.report_rollouts
    with pytest.raises(ValueError, match="original sink error"):
        owner.work()
    assert (
        "engine" in trace and "train packer" in trace and "validation packer" in trace
    )
    assert trace.index("sink") < trace.index("engine") < trace.index("sink close")
    assert trace[-1] == "unregister"


def test_async_terminal_generation_failure_uses_same_required_consumer(monkeypatch):
    worker, source, ctrl, order, events = pipeline(monkeypatch)
    worker.scheduler = SimpleNamespace(
        get_all=lambda: [
            CompletedRollout(0, source, RolloutResult(completions=[]), False)
        ]
    )
    worker._stream_generation_collect_results()
    assert len(events) == 1 and events[0].outcomes[0].status == "generation_failed"
    assert order == ["sink", "report"]
    assert ctrl.rollout_only_state.outstanding_completions == 0


@pytest.mark.parametrize("scores", [[1.0, 2.0], [float("nan")]])
def test_remote_malformed_score_cardinality_or_values_is_strict(monkeypatch, scores):
    calculator = RemoteRewardCalculator()
    calculator.rollout_only = True
    calculator.uuid2stage = {"request": "training"}
    calculator.uuid2step = {"request": 0}
    calculator.uuid2payload = {"request": [RLPayload(prompt_idx=0, completions=["x"])]}
    calculator.uuid2replica = {"request": "replica"}
    calculator.uuid2completions_per_payload = {"request": [1]}
    calculator.fetch_reward = lambda *_: torch.tensor(scores)
    queue = Queue()
    queue.put("request")
    with pytest.raises(ValueError, match="every generated|finite"):
        calculator.get_results(queue)


@pytest.mark.parametrize("score", [float("nan"), float("inf")])
def test_local_nonfinite_scores_are_strict(monkeypatch, score):
    config = fixtures.configuration()
    monkeypatch.setattr(local_calculator.util, "setup_tokenizer", lambda _: None)
    calculator = LocalRewardCalculator()
    calculator.setup(config, reward_fns=[lambda *args, **kwargs: score])
    with pytest.raises(ValueError, match="finite"):
        calculator.compute_rewards(
            [RLPayload(prompt_idx=0, completions=["x"])], False, 0
        )
