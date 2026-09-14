# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from queue import Queue
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from cosmos_rl.dispatcher.controller import Controller
from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.dispatcher.protocol import RolloutRequest
from cosmos_rl.dispatcher.status import PolicyStatusManager
from cosmos_rl.reward.admission import prepare_completion_admission_report
from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.rollout.worker.colocated.rollout_control import (
    ColocatedRolloutControlWorker,
)
from cosmos_rl.rollout.worker.rollout_control import (
    DisaggregatedRolloutControlWorker,
)


def _rollout_worker(
    *,
    n_generation: int = 2,
    should_report: bool = True,
    worker_type=DisaggregatedRolloutControlWorker,
):
    worker = object.__new__(worker_type)
    worker.config = SimpleNamespace(
        train=SimpleNamespace(
            non_text=True,
            local_dataset=False,
            train_policy=SimpleNamespace(bypass_reward=False),
        ),
        rollout=SimpleNamespace(
            n_generation=n_generation,
            multi_turn_config=SimpleNamespace(enable=False),
        ),
    )
    worker.should_report = should_report
    worker.replica_name = "rollout-0"
    worker.global_rank = 3
    worker.current_weight_version = 0
    worker.api_client = SimpleNamespace(
        post_rollout_completion=MagicMock(return_value=True)
    )
    worker.reward_dispatcher = SimpleNamespace(enqueue_rewards_cal=MagicMock())
    worker.enqueue_teacher_calculation = lambda payloads: payloads
    return worker


def test_empty_non_text_result_reports_reserved_samples():
    worker = _rollout_worker(n_generation=2)
    payload = SimpleNamespace(prompt_idx=7)

    valid_payloads, valid_results = worker._filter_valid_rollout_results_and_report(
        [RolloutResult(completions=[])],
        [payload],
    )

    assert valid_payloads == []
    assert valid_results == []
    worker.reward_dispatcher.enqueue_rewards_cal.assert_not_called()
    request = worker.api_client.post_rollout_completion.call_args.args[0]
    assert request.payloads == []
    assert request.src_replica_name == "rollout-0"
    assert request.src_global_rank == 3
    assert request.metrics["discarded_samples"] == 2
    assert request.metrics["discard_report_id"]
    assert request.metrics["discarded_weight_version"] == 0


def test_empty_outer_result_reports_every_consumed_prompt():
    worker = _rollout_worker(n_generation=4)
    worker._prompt_queue = Queue()
    worker._prompt_queue.put(
        [SimpleNamespace(prompt_idx=0), SimpleNamespace(prompt_idx=1)]
    )
    worker._call_rollout_generation = MagicMock(return_value=[])
    worker.inference_stream = None
    worker.data_packer = None
    worker.data_fetcher = None

    assert worker.one_step_generation() is False

    request = worker.api_client.post_rollout_completion.call_args.args[0]
    assert request.metrics["discarded_samples"] == 8


def test_non_reporting_rank_does_not_report_discard():
    worker = _rollout_worker(should_report=False)

    worker._filter_valid_rollout_results_and_report(
        [RolloutResult(completions=[])],
        [SimpleNamespace(prompt_idx=0)],
    )

    worker.api_client.post_rollout_completion.assert_not_called()


def test_colocated_worker_does_not_report_discarded_samples():
    worker = _rollout_worker(worker_type=ColocatedRolloutControlWorker)

    valid_payloads, valid_results = worker._filter_valid_rollout_results_and_report(
        [RolloutResult(completions=[])],
        [SimpleNamespace(prompt_idx=0)],
    )

    assert valid_payloads == []
    assert valid_results == []
    worker.api_client.post_rollout_completion.assert_not_called()


def test_discard_settlement_is_idempotent_per_replica_and_report():
    manager = PolicyStatusManager()
    manager.samples_on_the_fly = 10

    assert manager.settle_discarded_samples("rollout-0", "report-1", 3) == 3
    assert manager.samples_on_the_fly == 7
    assert manager.settle_discarded_samples("rollout-0", "report-1", 3) == 0
    assert manager.samples_on_the_fly == 7
    assert manager.settle_discarded_samples("rollout-0", "report-2", 2) == 2
    assert manager.samples_on_the_fly == 5
    assert manager.filter_records["rollout_failed"] == 5

    manager.forget_discard_reports("rollout-0")
    assert "rollout-0" not in manager._applied_discard_report_ids


def test_discard_settlement_requires_report_id():
    manager = PolicyStatusManager()
    manager.samples_on_the_fly = 5

    assert manager.settle_discarded_samples("rollout-0", None, 2) == 0
    assert manager.samples_on_the_fly == 5
    assert manager.filter_records == {}


def test_on_policy_discard_reopens_prompt_slot_for_same_weight_version():
    controller = object.__new__(Controller)
    controller.config = SimpleNamespace(
        mode="disaggregated",
        rollout=SimpleNamespace(n_generation=4),
        train=SimpleNamespace(
            train_policy=SimpleNamespace(on_policy=True, variant="grpo")
        ),
    )
    controller.weight_version_to_prompt_num = {5: 2}
    controller.weight_version_to_replacement_prompt_num = {}
    replacement = RLPayload(prompt_idx=9)

    assert controller.register_discarded_samples_for_refill(5, 1) == 1
    controller._assign_prompt_weight_versions(
        [replacement], starting_weight_version=5, prompt_quota=2
    )

    assert replacement.weight_version == 5
    assert controller.weight_version_to_prompt_num == {5: 3}
    assert controller.weight_version_to_replacement_prompt_num == {5: 0}
    assert controller.weight_version_to_replacement_prompt_issued == {5: 1}


def test_partial_discard_reports_share_one_replacement_prompt():
    controller = object.__new__(Controller)
    controller.config = SimpleNamespace(
        mode="disaggregated",
        rollout=SimpleNamespace(n_generation=4),
        train=SimpleNamespace(
            train_policy=SimpleNamespace(on_policy=True, variant="grpo")
        ),
    )
    controller.weight_version_to_prompt_num = {5: 2}
    controller.weight_version_to_replacement_prompt_num = {}

    assert controller.register_discarded_samples_for_refill(5, 1) == 1
    assert controller.register_discarded_samples_for_refill(5, 1) == 0
    assert controller.weight_version_to_replacement_prompt_num == {5: 1}


def test_non_on_policy_discard_does_not_change_prompt_version_capacity():
    controller = object.__new__(Controller)
    controller.config = SimpleNamespace(
        mode="disaggregated",
        rollout=SimpleNamespace(n_generation=4),
        train=SimpleNamespace(
            train_policy=SimpleNamespace(on_policy=False, variant="grpo")
        ),
    )

    assert controller.register_discarded_samples_for_refill(5, 1) == 0
    assert not hasattr(controller, "weight_version_to_replacement_prompt_num")


def test_prompt_without_refill_credit_advances_to_next_weight_version():
    controller = object.__new__(Controller)
    controller.weight_version_to_prompt_num = {5: 2}
    controller.weight_version_to_replacement_prompt_num = {}
    payload = RLPayload(prompt_idx=9)

    controller._assign_prompt_weight_versions(
        [payload], starting_weight_version=5, prompt_quota=2
    )

    assert payload.weight_version == 6


def test_on_policy_prompt_fetch_uses_same_weight_after_partial_admission():
    controller = object.__new__(Controller)
    policy_status = MagicMock()
    policy_status.__len__.return_value = 1
    policy_status.current_step = 5
    policy_status.total_pending_rollouts.return_value = 7
    policy_status.samples_on_the_fly = 7
    policy_status.replica_scaling_log = []
    policy_status.training_finished.return_value = False
    controller.policy_status_manager = policy_status
    controller.rollout_status_manager = SimpleNamespace(replica_scaling_log=[])
    controller.data_fetcher = SimpleNamespace(
        get_batched_prompt=MagicMock(return_value=([RLPayload(prompt_idx=9)], False))
    )
    controller.config = SimpleNamespace(
        mode="disaggregated",
        validation=SimpleNamespace(enable=False),
        rollout=SimpleNamespace(n_generation=4),
        train=SimpleNamespace(
            train_batch_per_replica=8,
            train_policy=SimpleNamespace(
                type="grpo",
                variant="grpo",
                on_policy=True,
                allowed_outdated_steps=0,
                outdated_rollout_fetch_batch_size=0,
                max_inflight_steps=None,
                max_retry_for_on_policy=0,
            ),
        ),
    )
    controller.weight_version_to_prompt_num = {5: 2}
    controller.weight_version_to_replacement_prompt_num = {}
    controller._soft_throttle_engaged_since = None
    controller._soft_throttle_last_log_ts = 0.0
    controller.register_discarded_samples_for_refill(5, 1)

    payloads, is_end = asyncio.run(controller._get_batched_prompt_impl(1))

    assert not is_end
    assert [payload.weight_version for payload in payloads] == [5]
    assert policy_status.samples_on_the_fly == 11


def test_http_discard_report_settles_before_normal_admission():
    from cosmos_rl.dispatcher import run_web_panel

    policy_status = SimpleNamespace(
        _parse_non_negative_count=PolicyStatusManager._parse_non_negative_count,
        settle_discarded_samples=MagicMock(return_value=4),
        rollout_admission_closed=lambda: False,
        filter_outdated_rollouts=lambda rollouts: rollouts,
    )
    fake_controller = SimpleNamespace(
        policy_status_manager=policy_status,
        config=SimpleNamespace(
            train=SimpleNamespace(train_policy=SimpleNamespace(variant="grpo"))
        ),
        put_rollouts=AsyncMock(),
        register_discarded_samples_for_refill=MagicMock(),
    )
    request = RolloutRequest(
        src_replica_name="rollout-0",
        payloads=[],
        metrics={
            "discarded_samples": 4,
            "discard_report_id": "report-1",
        },
    )

    with patch.object(run_web_panel, "controller", fake_controller):
        response = asyncio.run(run_web_panel.put_rollout_group(request))

    assert response == {"message": "Rollout put"}
    policy_status.settle_discarded_samples.assert_called_once_with(
        source_replica="rollout-0",
        report_id="report-1",
        count=4,
        weight_version=None,
    )
    fake_controller.register_discarded_samples_for_refill.assert_called_once_with(
        None, 4, "report-1"
    )
    fake_controller.put_rollouts.assert_awaited_once_with([])


def test_http_admission_retry_is_idempotent_for_metrics_settlement_and_refill():
    from cosmos_rl.dispatcher import run_web_panel

    policy_status = PolicyStatusManager()
    policy_status.samples_on_the_fly = 5
    policy_status.config = SimpleNamespace(
        train=SimpleNamespace(
            train_batch_per_replica=4,
            train_policy=SimpleNamespace(data_dispatch_as_rank_in_mesh=False),
        )
    )
    policy_status.rollout_admission_closed = lambda: False
    policy_status.filter_outdated_rollouts = lambda rollouts: rollouts
    fake_controller = SimpleNamespace(
        policy_status_manager=policy_status,
        config=SimpleNamespace(
            train=SimpleNamespace(train_policy=SimpleNamespace(variant="grpo"))
        ),
        put_rollouts=AsyncMock(),
        register_discarded_samples_for_refill=MagicMock(),
    )
    metrics = prepare_completion_admission_report(
        {
            "rollout/completion_admission_excluded_count": 1,
            "rollout/completion_admission_training_discarded_count": 1,
        },
        weight_version=3,
        report_id="stable-admission-report",
    )
    request = RolloutRequest(
        src_replica_name="rollout-0",
        payloads=[],
        metrics=metrics,
    )

    with patch.object(run_web_panel, "controller", fake_controller):
        asyncio.run(run_web_panel.put_rollout_group(request))
        asyncio.run(run_web_panel.put_rollout_group(request))

    assert policy_status.completion_admission_records == {
        1: {
            "rollout/completion_admission_excluded_count": 1,
            "rollout/completion_admission_training_discarded_count": 1,
        }
    }
    assert policy_status.samples_on_the_fly == 4
    fake_controller.register_discarded_samples_for_refill.assert_called_once_with(
        3, 1, "stable-admission-report"
    )


def test_http_admission_discard_refills_strict_on_policy_step_at_same_weight():
    from cosmos_rl.dispatcher import run_web_panel

    config = SimpleNamespace(
        mode="disaggregated",
        validation=SimpleNamespace(enable=False),
        rollout=SimpleNamespace(n_generation=4),
        train=SimpleNamespace(
            train_batch_per_replica=8,
            train_policy=SimpleNamespace(
                type="grpo",
                variant="grpo",
                on_policy=True,
                data_dispatch_as_rank_in_mesh=False,
                allowed_outdated_steps=0,
                outdated_rollout_fetch_batch_size=0,
                max_inflight_steps=None,
                max_retry_for_on_policy=0,
            ),
        ),
    )
    policy_status = PolicyStatusManager()
    policy_status.config = config
    policy_status.current_step = 5
    policy_status.total_steps = 100
    policy_status.samples_on_the_fly = 8
    policy_status.remain_samples_num = 100
    policy_status.policy_replicas = {
        "policy-0": SimpleNamespace(all_atoms_arrived=True)
    }
    for _ in range(7):
        policy_status.rollout_buffer.put(object())

    controller = object.__new__(Controller)
    controller.config = config
    controller.policy_status_manager = policy_status
    controller.rollout_status_manager = SimpleNamespace(replica_scaling_log=[])
    controller.data_fetcher = SimpleNamespace(
        get_batched_prompt=MagicMock(return_value=([RLPayload(prompt_idx=9)], False))
    )
    controller.weight_version_to_prompt_num = {5: 2}
    controller.weight_version_to_replacement_prompt_num = {}
    controller._soft_throttle_engaged_since = None
    controller._soft_throttle_last_log_ts = 0.0
    controller.put_rollouts = AsyncMock()

    request = RolloutRequest(
        src_replica_name="rollout-0",
        payloads=[],
        metrics=prepare_completion_admission_report(
            {
                "rollout/completion_admission_excluded_count": 1,
                "rollout/completion_admission_training_discarded_count": 1,
            },
            weight_version=5,
            report_id="strict-on-policy-discard",
        ),
    )

    with patch.object(run_web_panel, "controller", controller):
        asyncio.run(run_web_panel.put_rollout_group(request))
    replacement_payloads, is_end = asyncio.run(controller._get_batched_prompt_impl(1))

    assert not is_end
    assert policy_status.samples_on_the_fly == 11
    assert replacement_payloads[0].weight_version == 5
