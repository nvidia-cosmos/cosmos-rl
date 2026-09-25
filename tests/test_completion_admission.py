# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from queue import Queue
import pickle
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from cosmos_rl.dispatcher.algo.grpo import GRPO
from cosmos_rl.dispatcher.algo.base import REGISTERED_ALGOs
from cosmos_rl.dispatcher.data.schema import ChatMessage, RLPayload
from cosmos_rl.dispatcher.status import PolicyStatusManager
from cosmos_rl.reward.admission import (
    COMPLETION_ADMISSION_REPORT_ID_KEY,
    COMPLETION_ADMISSION_WEIGHT_VERSION_KEY,
    apply_rollout_result_to_payload,
    consume_completion_admission_metrics,
    normalize_rollout_results,
    prepare_completion_admission_report,
    resolve_completion_admission,
    select_payload_completions,
)
from cosmos_rl.reward.base import RolloutGroup
from cosmos_rl.reward.dispatcher import RewardDispatcher
from cosmos_rl.reward.local_calculator import LocalRewardCalculator
from cosmos_rl.reward.remote_calculator import RemoteRewardCalculator
from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.rollout.worker.rollout_control import DisaggregatedRolloutControlWorker
from cosmos_rl.utils.payload import extract_rollouts


class _TestAlgo(GRPO):
    def __init__(self, rewards):
        super().__init__(reward_fn=None)
        self.rewards = rewards
        self.reward_inputs = None

    def compute_reward(self, to_be_evaluated, reference, prompt=None, **kwargs):
        self.reward_inputs = list(to_be_evaluated)
        return (
            list(self.rewards),
            list(self.rewards),
            [{"reward_metric": reward} for reward in self.rewards],
        )


class _IdentityAdvantageAlgo(_TestAlgo):
    minimum_trainable_completions = 1

    def __init__(self, reward_fn=None, **kwargs):
        del reward_fn, kwargs
        super().__init__([])
        self.advantage_inputs = None

    def compute_advantage(self, rewards):
        self.advantage_inputs = list(rewards)
        return [reward + 10.0 for reward in rewards]


def _payload(mask=None, reasons=None):
    return RLPayload(
        prompt="prompt",
        prompt_idx=7,
        reference_answer="answer",
        completions=["c0", "c1", "c2", "c3"],
        completed_conversations=[[], [], [], []],
        completion_logprobs=[[[0.0]], [[1.0]], [[2.0]], [[3.0]]],
        completion_token_ids=[[[0]], [[1]], [[2]], [[3]]],
        cumulative_logprob=[0.0, 1.0, 2.0, 3.0],
        teacher_result_uuids=["u0", "u1", "u2", "u3"],
        extra_info={"aligned": [10, 11, 12, 13], "group": "metadata"},
        completion_trainable=mask,
        completion_drop_reasons=reasons,
    )


@pytest.mark.parametrize("invalid_index", [0, 1, 3])
def test_rollout_group_excludes_before_advantage(invalid_index):
    rewards = [1.0, 7.0, 3.0, 5.0]
    mask = [True] * 4
    mask[invalid_index] = False
    reasons = [None] * 4
    reasons[invalid_index] = "missing_log_probs"
    payload = _payload(mask, reasons)
    algo = _TestAlgo(rewards)
    group = RolloutGroup(7, payload, False, "answer")

    rollouts = group.compute_rollouts(algo)

    eligible_indices = [i for i in range(4) if i != invalid_index]
    eligible_rewards = np.asarray([rewards[i] for i in eligible_indices])
    expected = (eligible_rewards - eligible_rewards.mean()) / (
        eligible_rewards.std() + 1e-6
    )
    assert algo.reward_inputs == payload.completions
    assert [rollout.completion for rollout in rollouts] == [
        payload.completions[i] for i in eligible_indices
    ]
    assert [rollout.advantage for rollout in rollouts] == pytest.approx(expected)
    assert group.excluded_reward_metrics == {
        invalid_index: {"reward_metric": rewards[invalid_index]}
    }
    assert group.completion_admission.drop_reason_counts == {"missing_log_probs": 1}


@pytest.mark.parametrize(
    "mask",
    [
        [False, False, False, False],
        [False, True, False, False],
    ],
)
def test_rollout_group_skips_insufficient_explicit_group(mask):
    algo = _TestAlgo([1.0, 2.0, 3.0, 4.0])
    group = RolloutGroup(7, _payload(mask), False, "answer")

    assert group.compute_rollouts(algo) == []
    assert algo.reward_inputs == ["c0", "c1", "c2", "c3"]
    assert group.completion_admission.group_excluded


def test_absent_mask_preserves_single_completion_group():
    payload = RLPayload(
        prompt_idx=1,
        completions=["only"],
        reference_answer="answer",
    )
    group = RolloutGroup(1, payload, False, "answer")

    rollouts = group.compute_rollouts(_TestAlgo([2.0]))

    assert len(rollouts) == 1
    assert rollouts[0].completion == "only"
    assert rollouts[0].advantage == 0.0
    assert rollouts[0].report_metrics == {"reward_metric": 2.0}


def test_validation_does_not_apply_training_admission():
    payload = _payload([False, True, False, False])
    group = RolloutGroup(7, payload, True, "answer")

    rollouts = group.compute_rollouts(
        _TestAlgo([1.0, 2.0, 3.0, 4.0]),
        apply_completion_admission=False,
    )

    assert [rollout.completion for rollout in rollouts] == payload.completions


def test_validation_ignores_misaligned_training_admission_metadata():
    payload = _payload([False], ["stale", "metadata"])
    group = RolloutGroup(7, payload, True, "answer")

    rollouts = group.compute_rollouts(
        _TestAlgo([1.0, 2.0, 3.0, 4.0]),
        apply_completion_admission=False,
    )

    assert [rollout.completion for rollout in rollouts] == payload.completions


def test_admission_rejects_misaligned_fields():
    payload = _payload([True, False], [None, "bad"])

    with pytest.raises(ValueError, match="completion_trainable"):
        resolve_completion_admission(payload, 2)


def test_select_payload_keeps_completion_fields_aligned():
    payload = _payload(
        [False, True, False, True],
        ["bad-0", None, "bad-2", None],
    )
    admission = resolve_completion_admission(payload, 2)

    selected = select_payload_completions(payload, admission)

    assert selected.completions == ["c1", "c3"]
    assert selected.completion_logprobs == [[[1.0]], [[3.0]]]
    assert selected.completion_token_ids == [[[1]], [[3]]]
    assert selected.cumulative_logprob == [1.0, 3.0]
    assert selected.teacher_result_uuids == ["u1", "u3"]
    assert selected.extra_info == {
        "aligned": [11, 13],
        "group": "metadata",
    }
    assert selected.completion_trainable is None
    assert selected.completion_drop_reasons is None


def test_select_payload_supports_tensor_native_completions():
    completions = torch.arange(8).reshape(4, 2)
    payload = RLPayload(
        completions=completions,
        completion_trainable=[False, True, False, True],
    )

    selected = select_payload_completions(
        payload, resolve_completion_admission(payload, 2)
    )

    assert torch.equal(selected.completions, completions[[1, 3]])


def test_insufficient_payload_is_metadata_only_and_accounted():
    payload = _payload([False, True, False, False])
    admission = resolve_completion_admission(payload, 2)
    selected = select_payload_completions(payload, admission)

    kept, metrics = consume_completion_admission_metrics([selected])

    assert kept == []
    assert metrics["rollout/completion_admission_original_count"] == 4
    assert metrics["rollout/completion_admission_eligible_count"] == 1
    assert metrics["rollout/completion_admission_excluded_count"] == 3
    assert metrics["rollout/completion_admission_insufficient_group_count"] == 1
    assert metrics["rollout/completion_admission_training_discarded_count"] == 4


def test_partial_admission_keeps_payload_and_accounts_reason():
    payload = _payload([True, False, True, True])
    selected = select_payload_completions(
        payload, resolve_completion_admission(payload, 2)
    )

    kept, metrics = consume_completion_admission_metrics([selected])

    assert kept == [selected]
    assert metrics["rollout/completion_admission_excluded_count"] == 1
    assert metrics["rollout/completion_admission_training_discarded_count"] == 1
    assert metrics["rollout/completion_admission_reason_unspecified_count"] == 1


def test_unknown_drop_reasons_use_bounded_other_metric():
    payload = _payload(
        [False, False, False, True],
        ["request-specific-1", "request-specific-2", "missing_log_probs", None],
    )

    admission = resolve_completion_admission(payload, 1)

    assert admission.drop_reason_counts == {
        "other": 2,
        "missing_log_probs": 1,
    }
    assert all("request_specific" not in key for key in admission.metrics())


def test_internal_admission_metrics_survive_reward_worker_pickle_only():
    payload = _payload([True, False, True, True])
    selected = select_payload_completions(
        payload, resolve_completion_admission(payload, 2)
    )

    restored = pickle.loads(pickle.dumps(selected))

    assert (
        restored.completion_admission_metrics == selected.completion_admission_metrics
    )
    assert "completion_admission_metrics" not in selected.model_dump()


def test_controller_accumulates_admission_metrics():
    manager = object.__new__(PolicyStatusManager)
    manager.completion_admission_records = {}
    manager._applied_completion_admission_report_ids = {}
    metrics = {
        "rollout/completion_admission_excluded_count": 2,
        "rollout/completion_admission_reason_missing_log_probs_count": 2,
        "rollout/completion_admission_excluded_reward_score_sum": -3.5,
        "rollout/completion_admission_excluded_reward_score_count": 2,
        "unrelated": 99,
    }

    assert manager.update_completion_admission_statistics(
        metrics,
        source_replica="rollout-0",
        report_id="report-0",
        training_step=3,
    )
    assert manager.update_completion_admission_statistics(
        metrics,
        source_replica="rollout-0",
        report_id="report-1",
        training_step=3,
    )

    assert manager.completion_admission_records == {
        3: {
            "rollout/completion_admission_excluded_count": 4,
            "rollout/completion_admission_reason_missing_log_probs_count": 4,
            "rollout/completion_admission_excluded_reward_score_sum": -7.0,
            "rollout/completion_admission_excluded_reward_score_count": 4,
        }
    }

    taken = manager._take_completion_admission_statistics(3)
    assert taken
    assert manager.completion_admission_records == {}


def test_controller_keeps_admission_metrics_separate_by_training_step_and_report():
    manager = object.__new__(PolicyStatusManager)
    manager.completion_admission_records = {}
    manager._applied_completion_admission_report_ids = {}
    metrics = {"rollout/completion_admission_excluded_count": 1}

    assert manager.update_completion_admission_statistics(
        metrics,
        source_replica="rollout-0",
        report_id="version-4",
        training_step=4,
    )
    assert manager.update_completion_admission_statistics(
        metrics,
        source_replica="rollout-0",
        report_id="version-5",
        training_step=5,
    )
    assert not manager.update_completion_admission_statistics(
        metrics,
        source_replica="rollout-0",
        report_id="version-4",
        training_step=4,
    )

    assert manager._take_completion_admission_statistics(4) == metrics
    assert manager.completion_admission_records == {5: metrics}


def test_controller_targets_admission_metrics_to_next_unfilled_training_step():
    manager = PolicyStatusManager()
    manager.current_step = 10
    manager.config = SimpleNamespace(
        train=SimpleNamespace(
            train_batch_per_replica=4,
            train_policy=SimpleNamespace(data_dispatch_as_rank_in_mesh=False),
        )
    )

    assert manager.next_rollout_training_step() == 11
    for _ in range(4):
        manager.rollout_buffer.put(object())
    assert manager.next_rollout_training_step() == 12


def test_local_reward_payload_remains_aligned_after_admission():
    calculator = LocalRewardCalculator()
    calculator.rl_algo = _TestAlgo([1.0, 100.0, 3.0, 5.0])
    calculator.config = SimpleNamespace(
        train=SimpleNamespace(
            non_text=True,
            train_policy=SimpleNamespace(min_filter_prefix_tokens=None),
        )
    )
    payload = _payload([True, False, True, True])

    result, is_validation, step = calculator.compute_rewards([payload], False, 4)

    assert not is_validation
    assert step == 4
    assert result[0].completions == ["c0", "c2", "c3"]
    assert result[0].completion_logprobs == [[[0.0]], [[2.0]], [[3.0]]]
    assert result[0].cumulative_logprob == [0.0, 2.0, 3.0]
    assert result[0].teacher_result_uuids == ["u0", "u2", "u3"]
    assert result[0].extra_info["aligned"] == [10, 12, 13]
    admitted_payloads, metrics = consume_completion_admission_metrics(result)
    assert (
        metrics["rollout/completion_admission_excluded_reward_reward_metric_sum"]
        == 100.0
    )
    assert (
        metrics["rollout/completion_admission_excluded_reward_reward_metric_count"] == 1
    )
    trainer_rollouts = extract_rollouts(admitted_payloads, is_end=False)
    assert [rollout.completion for rollout in trainer_rollouts[0]] == [
        "c0",
        "c2",
        "c3",
    ]


def test_fully_rejected_group_preserves_excluded_reward_telemetry():
    calculator = LocalRewardCalculator()
    calculator.rl_algo = _TestAlgo([1.0, 2.0, 3.0, 4.0])
    calculator.config = SimpleNamespace(
        train=SimpleNamespace(
            non_text=True,
            train_policy=SimpleNamespace(min_filter_prefix_tokens=None),
        )
    )
    payload = _payload([False, True, False, False])

    result, _, _ = calculator.compute_rewards([payload], False, 4)
    kept, metrics = consume_completion_admission_metrics(result)

    assert kept == []
    assert (
        metrics["rollout/completion_admission_excluded_reward_reward_metric_sum"]
        == 10.0
    )
    assert (
        metrics["rollout/completion_admission_excluded_reward_reward_metric_count"] == 4
    )


def test_remote_reward_filters_before_normalization(monkeypatch):
    payload = _payload(
        [True, False, True, True],
        [None, "missing_log_probs", None, None],
    )
    calculator = RemoteRewardCalculator()
    calculator.minimum_trainable_completions = 2
    calculator.rl_algo = _TestAlgo([])
    calculator.uuid2payload = {"request": [payload]}
    calculator.uuid2replica = {"request": None}
    calculator.uuid2stage = {"request": "training"}
    calculator.uuid2step = {"request": 9}
    calculator.uuid2completions_per_payload = {"request": [4]}
    all_rewards = torch.tensor([1.0, 100.0, 3.0, 5.0])
    monkeypatch.setattr(calculator, "fetch_reward", lambda *_: all_rewards)
    requests = Queue()
    requests.put("request")

    result, is_validation, step = calculator.get_results(requests)

    eligible_rewards = np.asarray([1.0, 3.0, 5.0])
    expected = (eligible_rewards - eligible_rewards.mean()) / (
        eligible_rewards.std() + 1e-6
    )
    assert not is_validation
    assert step == 9
    assert result[0].completions == ["c0", "c2", "c3"]
    assert result[0].rewards == [1.0, 3.0, 5.0]
    assert result[0].advantages == pytest.approx(expected.tolist())
    assert result[0].completion_logprobs == [[[0.0]], [[2.0]], [[3.0]]]


def test_remote_reward_uses_registered_algorithm_advantage(monkeypatch):
    payload = _payload([True, False, True, True])
    calculator = RemoteRewardCalculator()
    algo_name = "test_identity_advantage"
    monkeypatch.setitem(REGISTERED_ALGOs, algo_name, _IdentityAdvantageAlgo)
    monkeypatch.setattr(
        "cosmos_rl.reward.remote_calculator.Wan2pt1VAEInterface",
        lambda **_: None,
        raising=False,
    )
    config = SimpleNamespace(
        train=SimpleNamespace(
            train_policy=SimpleNamespace(
                algo=algo_name,
                unbiased_advantage=False,
                remote_reward=SimpleNamespace(),
            )
        ),
        validation=SimpleNamespace(remote_reward=SimpleNamespace()),
        policy=SimpleNamespace(
            diffusers=SimpleNamespace(tokenizer=SimpleNamespace(model_dump=lambda: {}))
        ),
    )
    calculator.setup(config)
    calculator.uuid2payload = {"request": [payload]}
    calculator.uuid2replica = {"request": None}
    calculator.uuid2stage = {"request": "training"}
    calculator.uuid2step = {"request": 9}
    calculator.uuid2completions_per_payload = {"request": [4]}
    monkeypatch.setattr(
        calculator, "fetch_reward", lambda *_: torch.tensor([1.0, 100.0, 3.0, 5.0])
    )
    requests = Queue()
    requests.put("request")

    result, _, _ = calculator.get_results(requests)

    assert calculator.rl_algo.advantage_inputs == [1.0, 3.0, 5.0]
    assert result[0].advantages == [11.0, 13.0, 15.0]


@pytest.mark.parametrize("mask", [None, [True, False, True, True]])
def test_remote_rewards_preserve_reservations_with_and_without_quality_masks(
    mask, monkeypatch
):
    payload = _payload(mask)
    payload.completion_sequences = [4, 5, 6, 7]
    payload.weight_version = 9
    calculator = RemoteRewardCalculator()
    calculator.minimum_trainable_completions = 2
    calculator.rl_algo = _TestAlgo([])
    calculator.uuid2payload = {"request": [payload]}
    calculator.uuid2replica = {"request": None}
    calculator.uuid2stage = {"request": "training"}
    calculator.uuid2step = {"request": 9}
    calculator.uuid2completions_per_payload = {"request": [4]}
    monkeypatch.setattr(
        calculator, "fetch_reward", lambda *_: torch.tensor([1.0, 100.0, 3.0, 5.0])
    )
    requests = Queue()
    requests.put("request")
    result, _, _ = calculator.get_results(requests)
    assert result[0].weight_version == 9
    assert result[0].completion_sequences == ([4, 6, 7] if mask else [4, 5, 6, 7])
    assert [r["sequence"] for r in result[0].completion_rejections] == (
        [5] if mask else []
    )


def test_remote_validation_ignores_misaligned_admission_metadata(monkeypatch):
    payload = _payload([False], ["stale"])
    payload.validation_work_id = 17
    calculator = RemoteRewardCalculator()
    calculator.rl_algo = _IdentityAdvantageAlgo()
    calculator.minimum_trainable_completions = 1
    calculator.uuid2payload = {"request": [payload]}
    calculator.uuid2replica = {"request": None}
    calculator.uuid2stage = {"request": "validation"}
    calculator.uuid2step = {"request": 9}
    calculator.uuid2completions_per_payload = {"request": [4]}
    monkeypatch.setattr(
        calculator, "fetch_reward", lambda *_: torch.tensor([1.0, 2.0, 3.0, 4.0])
    )
    requests = Queue()
    requests.put("request")

    result, is_validation, _ = calculator.get_results(requests)

    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    expected = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
    assert is_validation
    assert result[0].validation_work_id == 17
    assert result[0].completions == payload.completions
    assert result[0].advantages == pytest.approx(expected.tolist())
    assert calculator.rl_algo.advantage_inputs is None


def test_local_validation_preserves_issued_work_identity():
    calculator = LocalRewardCalculator()
    calculator.val_rl_algo = _TestAlgo([1.0, 2.0, 3.0, 4.0])
    payload = _payload()
    payload.validation_work_id = 17
    result, is_validation, step = calculator.compute_validation_rewards([payload], 9)
    assert is_validation and step == 9
    assert result[0].validation_work_id == 17
    assert result[0].rewards == [1.0, 2.0, 3.0, 4.0]


def test_bypass_reward_applies_admission():
    dispatcher = RewardDispatcher(payload_per_task=1)
    dispatcher.is_remote = False
    dispatcher.minimum_trainable_completions = 2
    payload = _payload([False, True, False, True])

    dispatcher.enqueue_rewards_cal([payload], False, 3, bypass_reward=True)
    result, is_validation, step, all_done = dispatcher.dequeue_rewards_cal()

    assert not is_validation
    assert step == 3
    assert not all_done
    assert result[0].completions == ["c1", "c3"]
    assert result[0].rewards == [0.0, 0.0]
    assert result[0].advantages == [0.0, 0.0]


def test_rollout_worker_propagates_producer_admission_fields():
    worker = object.__new__(DisaggregatedRolloutControlWorker)
    worker.config = SimpleNamespace(
        train=SimpleNamespace(
            non_text=False,
            local_dataset=False,
            train_policy=SimpleNamespace(bypass_reward=False),
        ),
        rollout=SimpleNamespace(
            n_generation=2,
            multi_turn_config=SimpleNamespace(enable=False),
        ),
    )
    worker.should_report = True
    worker.eos_token = "<eos>"
    worker.current_weight_version = 5
    worker._report_discarded_samples = lambda _: None
    worker.enqueue_teacher_calculation = lambda payloads: payloads
    enqueued = []
    worker.reward_dispatcher = SimpleNamespace(
        enqueue_rewards_cal=lambda payloads, *args, **kwargs: enqueued.extend(payloads)
    )
    result = RolloutResult(
        completions=["good", "bad"],
        completion_trainable=[True, False],
        completion_drop_reasons=[None, "missing_log_probs"],
    )

    worker._filter_valid_rollout_results_and_report([result], [RLPayload(prompt_idx=3)])

    assert enqueued[0].completion_trainable == [True, False]
    assert enqueued[0].completion_drop_reasons == [None, "missing_log_probs"]


def test_multiturn_filter_keeps_every_completion_field_aligned():
    worker = object.__new__(DisaggregatedRolloutControlWorker)
    worker.config = SimpleNamespace(
        train=SimpleNamespace(
            non_text=False,
            local_dataset=False,
            train_policy=SimpleNamespace(bypass_reward=False),
        ),
        rollout=SimpleNamespace(
            n_generation=2,
            multi_turn_config=SimpleNamespace(enable=True),
        ),
    )
    worker.should_report = True
    worker.current_weight_version = 5
    worker._report_discarded_samples = MagicMock()
    worker.enqueue_teacher_calculation = lambda payloads: payloads
    enqueued = []
    worker.reward_dispatcher = SimpleNamespace(
        enqueue_rewards_cal=lambda payloads, *args, **kwargs: enqueued.extend(payloads)
    )
    invalid_conversation = [ChatMessage(role="assistant", content="")]
    valid_conversation = [ChatMessage(role="assistant", content="answer")]
    result = RolloutResult(
        completions=["invalid", "valid"],
        completed_conversations=[invalid_conversation, valid_conversation],
        completion_trainable=[False, True],
        completion_drop_reasons=["invalid_completion", None],
        completion_logprobs=[[[0.0]], [[1.0]]],
        completion_token_ids=[[[10]], [[20]]],
        cumulative_logprob=[-2.0, -1.0],
        extra_info={"aligned": ["bad", "good"], "group": "metadata"},
    )

    valid_payloads, valid_results = worker._filter_valid_rollout_results_and_report(
        [result], [RLPayload(prompt_idx=3)]
    )

    assert len(valid_payloads) == 1
    assert valid_payloads[0].prompt_idx == 3
    # Preserve group slots up to reward processing. Selection (and minimum
    # size validation) must happen once, before advantage computation.
    assert valid_results[0].completions == ["invalid", "valid"]
    selected = select_payload_completions(
        enqueued[0], resolve_completion_admission(enqueued[0], 1)
    )
    assert selected.completions == ["valid"]
    assert selected.completed_conversations == [valid_conversation]
    assert selected.completion_logprobs == [[[1.0]]]
    assert selected.completion_token_ids == [[[20]]]
    assert selected.cumulative_logprob == [-1.0]
    assert selected.extra_info == {
        "aligned": ["good"],
        "group": "metadata",
    }
    assert enqueued[0].completion_trainable == [False, True]
    assert resolve_completion_admission(enqueued[0], 2).group_excluded
    worker._report_discarded_samples.assert_not_called()


def test_rollout_reporter_accounts_for_excluded_completion():
    payload = _payload([True, False, True, True])
    payload = select_payload_completions(
        payload, resolve_completion_admission(payload, 2)
    )
    results = iter(
        [
            ([payload], False, 2, False),
            (None, False, -1, True),
        ]
    )
    worker = object.__new__(DisaggregatedRolloutControlWorker)
    worker.reward_dispatcher = SimpleNamespace(
        dequeue_rewards_cal=lambda: next(results)
    )
    worker.config = SimpleNamespace(
        train=SimpleNamespace(
            local_dataset=False,
            train_policy=SimpleNamespace(
                variant="grpo",
                rollout_as_token_ids=False,
            ),
        )
    )
    worker.data_packer = SimpleNamespace(
        get_rollout_output=lambda *values: (*values, None)
    )
    worker.replica_name = "rollout-0"
    worker.api_client = SimpleNamespace(post_rollout_completion=MagicMock())

    worker.report_rollouts()

    request = worker.api_client.post_rollout_completion.call_args.args[0]
    assert request.payloads[0].completions == ["c0", "c2", "c3"]
    assert request.metrics["rollout/completion_admission_excluded_count"] == 1
    assert request.metrics["discarded_samples"] == 1
    assert request.metrics["discarded_weight_version"] == 2
    assert request.metrics[COMPLETION_ADMISSION_WEIGHT_VERSION_KEY] == 2
    assert request.metrics[COMPLETION_ADMISSION_REPORT_ID_KEY]
    assert (
        request.metrics["discard_report_id"]
        == request.metrics[COMPLETION_ADMISSION_REPORT_ID_KEY]
    )


def test_prepare_admission_report_is_stable_for_http_retry():
    report = prepare_completion_admission_report(
        {
            "rollout/completion_admission_excluded_count": 1,
            "rollout/completion_admission_training_discarded_count": 1,
        },
        weight_version=7,
        report_id="stable-report",
    )

    assert report[COMPLETION_ADMISSION_REPORT_ID_KEY] == "stable-report"
    assert report[COMPLETION_ADMISSION_WEIGHT_VERSION_KEY] == 7
    assert report["discard_report_id"] == "stable-report"
    assert report["discarded_weight_version"] == 7


def test_rollout_result_admission_propagates_for_all_backend_wrappers():
    payload = RLPayload(prompt_idx=3)
    result = RolloutResult(
        completions=["keep", "drop"],
        completion_trainable=[True, False],
        completion_drop_reasons=[None, "missing_log_probs"],
        completion_logprobs=[[[1.0]], [[2.0]]],
        completion_token_ids=[[[10]], [[20]]],
        cumulative_logprob=[-1.0, -2.0],
        extra_info={"source": "custom_trtllm"},
    )

    apply_rollout_result_to_payload(
        payload, result, include_completed_conversations=True
    )

    assert payload.completion_trainable == [True, False]
    assert payload.completion_drop_reasons == [None, "missing_log_probs"]
    assert payload.completion_logprobs == [[[1.0]], [[2.0]]]
    assert payload.completion_token_ids == [[[10]], [[20]]]
    assert payload.extra_info == {"source": "custom_trtllm"}


def test_rollout_result_normalization_preserves_masks_and_legacy_outputs():
    explicit = RolloutResult(
        completions=["keep", "drop"],
        completion_trainable=[True, False],
        completion_drop_reasons=[None, "missing_log_probs"],
    )

    normalized = normalize_rollout_results([explicit, ["legacy-0", "legacy-1"]])

    assert normalized[0] is explicit
    assert normalized[0].completion_trainable == [True, False]
    assert normalized[1].completions == ["legacy-0", "legacy-1"]
    assert normalized[1].completion_trainable is None
