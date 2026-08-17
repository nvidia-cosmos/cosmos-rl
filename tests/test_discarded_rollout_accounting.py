# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from queue import Queue
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cosmos_rl.dispatcher.controller import Controller
from cosmos_rl.dispatcher.data.schema import Rollout
from cosmos_rl.dispatcher.protocol import RolloutRequest
from cosmos_rl.dispatcher.status import PolicyStatusManager
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
    assert request.metrics["discarded_prompt_slots"] == 1
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
    assert request.metrics["discarded_prompt_slots"] == 2


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
    manager.weight_version_to_prompt_num = {0: 5}

    assert (
        manager.settle_discarded_samples("rollout-0", "report-1", 3, prompt_slots=1)
        == 3
    )
    assert manager.samples_on_the_fly == 7
    assert manager.weight_version_to_prompt_num == {0: 4}
    assert (
        manager.settle_discarded_samples("rollout-0", "report-1", 3, prompt_slots=1)
        == 0
    )
    assert manager.samples_on_the_fly == 7
    assert manager.weight_version_to_prompt_num == {0: 4}
    assert (
        manager.settle_discarded_samples("rollout-0", "report-2", 2, prompt_slots=1)
        == 2
    )
    assert manager.samples_on_the_fly == 5
    assert manager.weight_version_to_prompt_num == {0: 3}
    assert manager.filter_records["rollout_failed"] == 5

    manager.forget_discard_reports("rollout-0")
    assert "rollout-0" not in manager._applied_discard_report_ids


def test_discard_settlement_requires_report_id():
    manager = PolicyStatusManager()
    manager.samples_on_the_fly = 5

    assert manager.settle_discarded_samples("rollout-0", None, 2) == 0
    assert manager.samples_on_the_fly == 5
    assert manager.filter_records == {}


def test_discard_settlement_infers_slots_for_older_workers():
    manager = PolicyStatusManager()
    manager.config = SimpleNamespace(rollout=SimpleNamespace(n_generation=2))
    manager.samples_on_the_fly = 4
    manager.weight_version_to_prompt_num = {0: 2}

    assert manager.settle_discarded_samples("rollout-0", "report-1", 4) == 4
    assert manager.samples_on_the_fly == 0
    assert manager.weight_version_to_prompt_num == {}


def test_prompt_slot_release_retires_the_active_leading_edge():
    manager = PolicyStatusManager()
    manager.current_step = 3
    manager.weight_version_to_prompt_num = {1: 2, 3: 1, 4: 2}

    assert manager.release_prompt_slots(2, "test") == 2
    assert manager.weight_version_to_prompt_num == {3: 1}
    assert manager.release_prompt_slots(2, "test") == 1
    assert manager.weight_version_to_prompt_num == {}


def test_http_discard_report_settles_before_normal_admission():
    from cosmos_rl.dispatcher import run_web_panel

    policy_status = SimpleNamespace(
        _parse_non_negative_count=PolicyStatusManager._parse_non_negative_count,
        settle_discarded_samples=MagicMock(),
        rollout_admission_closed=lambda: False,
        filter_outdated_rollouts=lambda rollouts, *, prompt_groups: rollouts,
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
            "discarded_prompt_slots": 2,
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
        prompt_slots=2,
    )
    fake_controller.put_rollouts.assert_awaited_once_with([])


def _prompt_dispatch_controller(*, variant="grpo", max_retry=0):
    config = SimpleNamespace(
        mode="disaggregated",
        train=SimpleNamespace(
            train_batch_per_replica=4,
            train_policy=SimpleNamespace(
                type="grpo",
                variant=variant,
                allowed_outdated_steps=2,
                outdated_rollout_fetch_batch_size=0,
                max_inflight_steps=None,
                max_retry_for_on_policy=max_retry,
                data_dispatch_as_rank_in_mesh=False,
            ),
        ),
        rollout=SimpleNamespace(n_generation=2),
        validation=SimpleNamespace(enable=False),
    )
    status = PolicyStatusManager()
    status.config = config
    status.policy_replicas = {"policy-0": SimpleNamespace()}

    class DataFetcher:
        def __init__(self):
            self.next_prompt_idx = 0

        def get_batched_prompt(
            self,
            count,
            validation_step,
            rank_in_mesh,
            weight_version=None,
        ):
            payloads = [
                SimpleNamespace(
                    prompt_idx=self.next_prompt_idx + offset,
                    extra_info=None,
                )
                for offset in range(count)
            ]
            self.next_prompt_idx += count
            return payloads, False

    controller = object.__new__(Controller)
    controller.config = config
    controller.policy_status_manager = status
    controller.rollout_status_manager = SimpleNamespace(replica_scaling_log=[])
    controller.data_fetcher = DataFetcher()
    controller.weight_version_to_prompt_num = status.weight_version_to_prompt_num
    controller.weight_version_to_prompt_attempt_num = {}
    controller._soft_throttle_engaged_since = None
    controller._soft_throttle_last_log_ts = 0.0
    return controller, status


def test_dead_prompts_do_not_ratchet_dispatch_versions():
    controller, status = _prompt_dispatch_controller()
    dispatched_versions = []

    for report_index in range(8):
        payloads, is_end = asyncio.run(
            controller._get_batched_prompt_impl(2, None, None)
        )
        assert not is_end
        dispatched_versions.extend(payload.weight_version for payload in payloads)
        status.settle_discarded_samples(
            "rollout-0",
            f"report-{report_index}",
            count=4,
            prompt_slots=2,
        )

    assert max(dispatched_versions) <= 2
    assert set(dispatched_versions) == {0}
    assert status.samples_on_the_fly == 0
    assert status.weight_version_to_prompt_num == {}


def test_outdated_filter_releases_only_fully_discarded_prompt_groups():
    manager = PolicyStatusManager()
    manager.config = SimpleNamespace(
        train=SimpleNamespace(
            train_batch_per_replica=2,
            sync_weight_interval=1,
            train_policy=SimpleNamespace(
                allowed_outdated_steps=1,
                data_dispatch_as_rank_in_mesh=False,
            ),
        ),
        rollout=SimpleNamespace(n_generation=3),
    )
    manager.current_step = 1
    manager.samples_on_the_fly = 6
    manager.remain_samples_num = 100
    manager.weight_version_to_prompt_num = {1: 2}
    manager._publish_payload_transport_cleanup = MagicMock()
    prompt_groups = [
        [
            Rollout(prompt_idx=0, weight_version=0, completion=f"p0-{index}")
            for index in range(3)
        ],
        [
            Rollout(prompt_idx=1, weight_version=0, completion=f"p1-{index}")
            for index in range(3)
        ],
    ]
    rollouts = [rollout for group in prompt_groups for rollout in group]

    accepted = manager.filter_outdated_rollouts(
        rollouts,
        prompt_groups=prompt_groups,
    )

    assert accepted == prompt_groups[0][:2]
    assert manager.samples_on_the_fly == 2
    assert manager.weight_version_to_prompt_num == {1: 1}


def test_dapo_slot_release_preserves_cumulative_retry_limit():
    controller, status = _prompt_dispatch_controller(variant="dapo", max_retry=1)
    payloads, _ = asyncio.run(controller._get_batched_prompt_impl(2))
    assert len(payloads) == 2
    assert status.weight_version_to_prompt_num == {0: 2}
    assert controller.weight_version_to_prompt_attempt_num == {0: 2}

    status.update_dynamic_sampling_statistics(
        {
            "filtered_positive": 4,
            "filtered_prompt_slots": 2,
        }
    )
    assert status.weight_version_to_prompt_num == {}
    assert controller.weight_version_to_prompt_attempt_num == {0: 2}

    with pytest.raises(RuntimeError, match="After 1 retries"):
        asyncio.run(controller._get_batched_prompt_impl(2))


def test_dapo_reports_filtered_prompt_slots():
    worker = _rollout_worker(n_generation=2)
    valid = SimpleNamespace(valid=True, completions=["a", "b"])
    filtered = SimpleNamespace(
        valid=False,
        completions=["c", "d"],
        filter_rewards=[1.0, 1.0],
    )

    payloads, metrics = worker.dynamic_sampling([valid, filtered])

    assert payloads == [valid]
    assert metrics == {
        "sampled": 2,
        "filtered_positive": 2,
        "filtered_prompt_slots": 1,
    }

    manager = PolicyStatusManager()
    manager.config = SimpleNamespace(rollout=SimpleNamespace(n_generation=2))
    manager.samples_on_the_fly = 2
    manager.remain_samples_num = 2
    manager.weight_version_to_prompt_num = {0: 1}
    manager.update_dynamic_sampling_statistics(metrics)
    assert manager.samples_on_the_fly == 0
    assert manager.weight_version_to_prompt_num == {}


def test_colocated_dapo_does_not_report_prompt_slots():
    worker = _rollout_worker(
        n_generation=2,
        worker_type=ColocatedRolloutControlWorker,
    )
    filtered = SimpleNamespace(
        valid=False,
        completions=["a", "b"],
        filter_rewards=[-1.0, -1.0],
    )

    _, metrics = worker.dynamic_sampling([filtered])

    assert metrics == {"filtered_negative": 2}
