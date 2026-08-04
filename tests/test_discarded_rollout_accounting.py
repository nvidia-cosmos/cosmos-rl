# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from queue import Queue
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from cosmos_rl.dispatcher.protocol import RolloutRequest
from cosmos_rl.dispatcher.status import PolicyStatusManager
from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.rollout.worker.rollout_control import (
    DisaggregatedRolloutControlWorker,
)


def _rollout_worker(*, n_generation: int = 2, should_report: bool = True):
    worker = object.__new__(DisaggregatedRolloutControlWorker)
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


def test_http_discard_report_settles_before_normal_admission():
    from cosmos_rl.dispatcher import run_web_panel

    policy_status = SimpleNamespace(
        _parse_non_negative_count=PolicyStatusManager._parse_non_negative_count,
        settle_discarded_samples=MagicMock(),
        rollout_admission_closed=lambda: False,
        filter_outdated_rollouts=lambda rollouts: rollouts,
    )
    fake_controller = SimpleNamespace(
        policy_status_manager=policy_status,
        config=SimpleNamespace(
            train=SimpleNamespace(train_policy=SimpleNamespace(variant="grpo"))
        ),
        put_rollouts=AsyncMock(),
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
    )
    fake_controller.put_rollouts.assert_awaited_once_with([])
