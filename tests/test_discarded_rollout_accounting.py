# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from queue import Queue
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from cosmos_rl.dispatcher.controller import Controller
from cosmos_rl.dispatcher.data.schema import RLPayload, Rollout
from cosmos_rl.dispatcher.protocol import RolloutRequest
from cosmos_rl.dispatcher.status import PolicyStatusManager
from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.rollout.worker.asynchronous.rollout_task_scheduler import (
    CompletedRollout,
    RolloutTask,
    RolloutTaskScheduler,
)
from cosmos_rl.rollout.worker.colocated.rollout_control import (
    ColocatedRolloutControlWorker,
)
from cosmos_rl.rollout.worker.rollout_control import (
    DisaggregatedRolloutControlWorker,
)
from cosmos_rl.utils.payload import extract_rollouts


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
    payload = SimpleNamespace(prompt_idx=7, prompt_dispatch_id="dispatch-7")

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
    assert request.metrics["discarded_prompt_dispatch_ids"] == ["dispatch-7"]
    assert request.metrics["discard_report_id"]


def test_empty_outer_result_reports_every_consumed_prompt():
    worker = _rollout_worker(n_generation=4)
    worker._prompt_queue = Queue()
    worker._prompt_queue.put(
        [
            SimpleNamespace(prompt_idx=0, prompt_dispatch_id="dispatch-0"),
            SimpleNamespace(prompt_idx=1, prompt_dispatch_id="dispatch-1"),
        ]
    )
    worker._call_rollout_generation = MagicMock(return_value=[])
    worker.inference_stream = None
    worker.data_packer = None
    worker.data_fetcher = None

    assert worker.one_step_generation() is False

    request = worker.api_client.post_rollout_completion.call_args.args[0]
    assert request.metrics["discarded_samples"] == 8
    assert request.metrics["discarded_prompt_dispatch_ids"] == [
        "dispatch-0",
        "dispatch-1",
    ]


def test_generation_exception_reports_every_consumed_prompt():
    worker = _rollout_worker(n_generation=2)
    worker._prompt_queue = Queue()
    worker._prompt_queue.put(
        [
            SimpleNamespace(prompt_idx=0, prompt_dispatch_id="dispatch-0"),
            SimpleNamespace(prompt_idx=1, prompt_dispatch_id="dispatch-1"),
        ]
    )
    worker._call_rollout_generation = MagicMock(
        side_effect=RuntimeError("generation failed")
    )
    worker.inference_stream = None
    worker.data_packer = None
    worker.data_fetcher = None

    assert worker.one_step_generation() is False

    request = worker.api_client.post_rollout_completion.call_args.args[0]
    assert request.metrics["discarded_samples"] == 4
    assert request.metrics["discarded_prompt_dispatch_ids"] == [
        "dispatch-0",
        "dispatch-1",
    ]


def test_short_outer_result_reports_missing_prompt_dispatch():
    worker = _rollout_worker(n_generation=2)
    payloads = [
        SimpleNamespace(prompt_idx=0, prompt_dispatch_id="dispatch-0"),
        SimpleNamespace(prompt_idx=1, prompt_dispatch_id="dispatch-1"),
    ]
    worker._prompt_queue = Queue()
    worker._prompt_queue.put(payloads)
    worker._call_rollout_generation = MagicMock(
        return_value=[RolloutResult(completions=["a", "b"])]
    )
    worker.inference_stream = None
    worker.data_packer = None
    worker.data_fetcher = None

    valid_payloads, valid_results = worker.one_step_generation()

    assert valid_payloads == [payloads[0]]
    assert len(valid_results) == 1
    request = worker.api_client.post_rollout_completion.call_args.args[0]
    assert request.metrics["discarded_samples"] == 2
    assert request.metrics["discarded_prompt_dispatch_ids"] == ["dispatch-1"]


def test_async_scheduler_publishes_empty_generation_as_terminal_failure():
    scheduler = RolloutTaskScheduler(
        rollout_engine=SimpleNamespace(
            rollout_generation=AsyncMock(return_value=[]),
        ),
        data_packer=None,
    )
    task = RolloutTask(
        idx=7,
        payload=RLPayload(
            prompt_idx=7,
            prompt_dispatch_id="dispatch-7",
        ),
    )

    completed = asyncio.run(scheduler._generate_single(task))

    assert completed.result is None
    assert scheduler.total_processed == 1
    assert scheduler.get_all() == [completed]


def test_async_collection_reports_failed_prompt_dispatch():
    worker = _rollout_worker(n_generation=2)
    payload = RLPayload(
        prompt_idx=7,
        prompt_dispatch_id="dispatch-7",
    )
    worker.scheduler = SimpleNamespace(
        get_all=lambda: [
            CompletedRollout(
                idx=7,
                payload=payload,
                result=None,
            )
        ]
    )

    worker._stream_generation_collect_results()

    request = worker.api_client.post_rollout_completion.call_args.args[0]
    assert request.metrics["discarded_samples"] == 2
    assert request.metrics["discarded_prompt_dispatch_ids"] == ["dispatch-7"]


def test_async_validation_generation_failure_is_not_silently_omitted():
    worker = object.__new__(DisaggregatedRolloutControlWorker)
    worker._is_async_rollout = True
    worker.val_batch_size = 1
    worker.current_step = 0
    worker._stream_generation_feed_prompts = MagicMock(return_value=(1, False))
    worker.scheduler = SimpleNamespace(
        is_idle=lambda: False,
        get_all=lambda: [
            CompletedRollout(
                idx=7,
                payload=RLPayload(prompt_idx=7),
                result=None,
            )
        ],
    )

    with pytest.raises(RuntimeError, match="failed for 1 validation prompts"):
        worker.do_validation()


def test_partial_result_reports_only_missing_samples():
    worker = _rollout_worker(n_generation=4)
    payload = SimpleNamespace(prompt_idx=7, prompt_dispatch_id="dispatch-7")

    valid_payloads, valid_results = worker._filter_valid_rollout_results_and_report(
        [RolloutResult(completions=["a", "b", "c"])],
        [payload],
    )

    assert valid_payloads == [payload]
    assert len(valid_results) == 1
    request = worker.api_client.post_rollout_completion.call_args.args[0]
    assert request.metrics["discarded_samples"] == 1
    assert request.metrics["discarded_prompt_dispatch_ids"] == []


def test_oversized_prompt_group_settles_reserved_dispatch():
    worker = _rollout_worker(n_generation=2)
    payload = SimpleNamespace(prompt_idx=7, prompt_dispatch_id="dispatch-7")

    valid_payloads, valid_results = worker._filter_valid_rollout_results_and_report(
        [RolloutResult(completions=["a", "b", "unexpected"])],
        [payload],
    )

    assert valid_payloads == []
    assert valid_results == []
    worker.reward_dispatcher.enqueue_rewards_cal.assert_not_called()
    request = worker.api_client.post_rollout_completion.call_args.args[0]
    assert request.metrics["discarded_samples"] == 2
    assert request.metrics["discarded_prompt_dispatch_ids"] == ["dispatch-7"]


def test_oversized_tensor_group_settles_without_boolean_conversion():
    worker = _rollout_worker(n_generation=2)
    payload = SimpleNamespace(prompt_idx=7, prompt_dispatch_id="dispatch-7")

    valid_payloads, valid_results = worker._filter_valid_rollout_results_and_report(
        [RolloutResult(completions=torch.ones(3, 1))],
        [payload],
    )

    assert valid_payloads == []
    assert valid_results == []
    request = worker.api_client.post_rollout_completion.call_args.args[0]
    assert request.metrics["discarded_samples"] == 2
    assert request.metrics["discarded_prompt_dispatch_ids"] == ["dispatch-7"]


def test_mixed_results_report_reserved_minus_emitted_samples():
    worker = _rollout_worker(n_generation=4)
    payloads = [
        SimpleNamespace(
            prompt_idx=index,
            prompt_dispatch_id=f"dispatch-{index}",
        )
        for index in range(3)
    ]

    valid_payloads, valid_results = worker._filter_valid_rollout_results_and_report(
        [
            RolloutResult(completions=["a", "b", "c", "d"]),
            RolloutResult(completions=["e", "f", "g"]),
            RolloutResult(completions=[]),
        ],
        payloads,
    )

    assert valid_payloads == payloads[:2]
    assert len(valid_results) == 2
    request = worker.api_client.post_rollout_completion.call_args.args[0]
    assert request.metrics["discarded_samples"] == 5
    assert request.metrics["discarded_prompt_dispatch_ids"] == ["dispatch-2"]


def test_partial_result_settlement_balances_reserved_capacity():
    worker = _rollout_worker(n_generation=4)
    manager = PolicyStatusManager()
    manager.config = SimpleNamespace(rollout=SimpleNamespace(n_generation=4))

    for report_index in range(5):
        manager.samples_on_the_fly += 4
        worker._filter_valid_rollout_results_and_report(
            [RolloutResult(completions=["a", "b", "c"])],
            [
                SimpleNamespace(
                    prompt_idx=report_index,
                    prompt_dispatch_id=f"dispatch-{report_index}",
                )
            ],
        )
        request = worker.api_client.post_rollout_completion.call_args.args[0]
        manager.settle_discarded_samples(
            "rollout-0",
            request.metrics["discard_report_id"],
            request.metrics["discarded_samples"],
            prompt_dispatch_ids=request.metrics["discarded_prompt_dispatch_ids"],
        )
        manager.samples_on_the_fly -= 3
        assert manager.samples_on_the_fly == 0


def test_non_reporting_rank_does_not_report_discard():
    worker = _rollout_worker(should_report=False)

    worker._filter_valid_rollout_results_and_report(
        [RolloutResult(completions=[])],
        [SimpleNamespace(prompt_idx=0, prompt_dispatch_id="dispatch-0")],
    )

    worker.api_client.post_rollout_completion.assert_not_called()


def test_colocated_worker_does_not_report_discarded_samples():
    worker = _rollout_worker(worker_type=ColocatedRolloutControlWorker)

    valid_payloads, valid_results = worker._filter_valid_rollout_results_and_report(
        [RolloutResult(completions=[])],
        [SimpleNamespace(prompt_idx=0, prompt_dispatch_id="dispatch-0")],
    )

    assert valid_payloads == []
    assert valid_results == []
    worker.api_client.post_rollout_completion.assert_not_called()


def test_discard_settlement_is_idempotent_per_replica_and_report():
    manager = PolicyStatusManager()
    manager.samples_on_the_fly = 10
    dispatch_ids = [manager.register_prompt_dispatch(0) for _ in range(3)]

    assert (
        manager.settle_discarded_samples(
            "rollout-0",
            "report-1",
            3,
            prompt_dispatch_ids=[dispatch_ids[0]],
        )
        == 3
    )
    assert manager.samples_on_the_fly == 7
    assert manager.weight_version_to_prompt_num == {0: 2}
    assert (
        manager.settle_discarded_samples(
            "rollout-0",
            "report-1",
            3,
            prompt_dispatch_ids=[dispatch_ids[0]],
        )
        == 0
    )
    assert manager.samples_on_the_fly == 7
    assert manager.weight_version_to_prompt_num == {0: 2}
    assert (
        manager.settle_discarded_samples(
            "rollout-0",
            "report-2",
            2,
            prompt_dispatch_ids=[dispatch_ids[1]],
        )
        == 2
    )
    assert manager.samples_on_the_fly == 5
    assert manager.weight_version_to_prompt_num == {0: 1}
    assert manager.filter_records["rollout_failed"] == 5

    manager.forget_discard_reports("rollout-0")
    assert "rollout-0" not in manager._applied_discard_report_ids


def test_discard_settlement_requires_report_id():
    manager = PolicyStatusManager()
    manager.samples_on_the_fly = 5

    assert manager.settle_discarded_samples("rollout-0", None, 2) == 0
    assert manager.samples_on_the_fly == 5
    assert manager.filter_records == {}


def test_discard_settlement_does_not_guess_slots_for_older_workers():
    manager = PolicyStatusManager()
    manager.samples_on_the_fly = 4
    manager.register_prompt_dispatch(0)
    manager.register_prompt_dispatch(0)

    assert manager.settle_discarded_samples("rollout-0", "report-1", 4) == 4
    assert manager.samples_on_the_fly == 0
    assert manager.weight_version_to_prompt_num == {0: 2}


def test_prompt_slot_release_uses_the_original_dispatch_version():
    manager = PolicyStatusManager()
    dispatch_0 = manager.register_prompt_dispatch(0)
    dispatch_1 = manager.register_prompt_dispatch(1)
    manager.register_prompt_dispatch(1)

    assert (
        manager.resolve_prompt_dispatches(
            [dispatch_0],
            "test",
            release_slots=True,
        )
        == 1
    )
    assert manager.weight_version_to_prompt_num == {1: 2}
    assert (
        manager.resolve_prompt_dispatches(
            [dispatch_1],
            "test",
            release_slots=True,
        )
        == 1
    )
    assert manager.weight_version_to_prompt_num == {1: 1}


def test_prompt_dispatch_resolution_is_idempotent_across_retried_requests():
    manager = PolicyStatusManager()
    failed_id = manager.register_prompt_dispatch(0)
    manager.register_prompt_dispatch(0)

    assert (
        manager.resolve_prompt_dispatches(
            [failed_id],
            "first_request",
            release_slots=True,
        )
        == 1
    )
    assert (
        manager.resolve_prompt_dispatches(
            [failed_id],
            "retried_request",
            release_slots=True,
        )
        == 0
    )
    assert manager.weight_version_to_prompt_num == {0: 1}


def test_prompt_dispatch_state_is_pruned_after_step_advancement():
    manager = PolicyStatusManager()
    obsolete_id = manager.register_prompt_dispatch(0)
    active_id = manager.register_prompt_dispatch(1)

    manager.current_step = 1
    manager.prune_prompt_dispatches()

    assert manager.weight_version_to_prompt_num == {1: 1}
    assert obsolete_id not in manager._prompt_dispatch_versions
    assert manager._prompt_dispatch_versions == {active_id: 1}


def test_prompt_dispatch_identity_survives_rollout_extraction():
    payload = RLPayload(
        prompt_idx=7,
        prompt_dispatch_id="dispatch-7",
        completions=["completion"],
        rewards=[1.0],
        advantages=[0.0],
    )

    rollouts = extract_rollouts([payload], is_end=False)

    assert rollouts[0][0].prompt_dispatch_id == "dispatch-7"


def test_http_discard_report_settles_before_normal_admission():
    from cosmos_rl.dispatcher import run_web_panel

    policy_status = SimpleNamespace(
        _parse_non_negative_count=PolicyStatusManager._parse_non_negative_count,
        parse_prompt_dispatch_ids=PolicyStatusManager.parse_prompt_dispatch_ids,
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
            "discarded_prompt_dispatch_ids": ["dispatch-0", "dispatch-1"],
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
        prompt_dispatch_ids=["dispatch-0", "dispatch-1"],
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
            prompt_dispatch_ids=[payload.prompt_dispatch_id for payload in payloads],
        )

    assert max(dispatched_versions) <= 2
    assert set(dispatched_versions) == {0}
    assert status.samples_on_the_fly == 0
    assert status.weight_version_to_prompt_num == {}


def test_failed_prompt_reopens_its_exact_version_in_spanning_batch():
    controller, status = _prompt_dispatch_controller()
    payloads, _ = asyncio.run(controller._get_batched_prompt_impl(3))
    assert [payload.weight_version for payload in payloads] == [0, 0, 1]

    status.settle_discarded_samples(
        "rollout-0",
        "report-0",
        count=2,
        prompt_dispatch_ids=[payloads[0].prompt_dispatch_id],
    )
    assert status.weight_version_to_prompt_num == {0: 1, 1: 1}

    replacement, _ = asyncio.run(controller._get_batched_prompt_impl(1))
    assert replacement[0].weight_version == 0
    assert status.weight_version_to_prompt_num == {0: 2, 1: 1}


def test_scaling_bypass_does_not_reserve_a_tracked_prompt_slot():
    controller, status = _prompt_dispatch_controller()
    controller.rollout_status_manager.replica_scaling_log = [object()]

    payloads, _ = asyncio.run(controller._get_batched_prompt_impl(1))

    assert payloads[0].weight_version == 0
    assert getattr(payloads[0], "prompt_dispatch_id", None) is None
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
    dispatch_ids = [manager.register_prompt_dispatch(1) for _ in range(2)]
    manager._publish_payload_transport_cleanup = MagicMock()
    prompt_groups = [
        [
            Rollout(
                prompt_idx=0,
                prompt_dispatch_id=dispatch_ids[0],
                weight_version=0,
                completion=f"p0-{index}",
            )
            for index in range(3)
        ],
        [
            Rollout(
                prompt_idx=1,
                prompt_dispatch_id=dispatch_ids[1],
                weight_version=0,
                completion=f"p1-{index}",
            )
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
    assert manager._prompt_dispatch_versions == {}
    assert (
        manager.resolve_prompt_dispatches(
            [dispatch_ids[0]],
            "late_duplicate",
            release_slots=True,
        )
        == 0
    )
    assert manager.weight_version_to_prompt_num == {1: 1}


def test_terminal_cleanup_releases_reported_prompt_dispatches():
    manager = PolicyStatusManager()
    manager.samples_on_the_fly = 3
    completed_id = manager.register_prompt_dispatch(0)
    filtered_id = manager.register_prompt_dispatch(0)
    manager._publish_payload_transport_cleanup = MagicMock()

    settled = manager.cleanup_terminal_rollouts(
        [
            Rollout(
                prompt_idx=0,
                prompt_dispatch_id=completed_id,
                weight_version=0,
                completion="completion",
            )
        ],
        {
            "sampled": 1,
            "filtered_positive": 2,
            "filtered_prompt_dispatch_ids": [filtered_id],
        },
        is_dapo=True,
    )

    assert settled == 3
    assert manager.samples_on_the_fly == 0
    assert manager.weight_version_to_prompt_num == {}
    assert manager._prompt_dispatch_versions == {}


def test_dapo_slot_release_preserves_cumulative_retry_limit():
    controller, status = _prompt_dispatch_controller(variant="dapo", max_retry=1)
    payloads, _ = asyncio.run(controller._get_batched_prompt_impl(2))
    assert len(payloads) == 2
    assert status.weight_version_to_prompt_num == {0: 2}
    assert controller.weight_version_to_prompt_attempt_num == {0: 2}

    status.update_dynamic_sampling_statistics(
        {
            "filtered_positive": 4,
            "filtered_prompt_dispatch_ids": [
                payload.prompt_dispatch_id for payload in payloads
            ],
        }
    )
    assert status.weight_version_to_prompt_num == {}
    assert controller.weight_version_to_prompt_attempt_num == {0: 2}
    next_prompt_idx = controller.data_fetcher.next_prompt_idx
    dispatch_state = dict(status._prompt_dispatch_versions)
    samples_on_the_fly = status.samples_on_the_fly

    with pytest.raises(RuntimeError, match="After 1 retries"):
        asyncio.run(controller._get_batched_prompt_impl(2))
    assert controller.data_fetcher.next_prompt_idx == next_prompt_idx
    assert controller.weight_version_to_prompt_attempt_num == {0: 2}
    assert status._prompt_dispatch_versions == dispatch_state
    assert status.weight_version_to_prompt_num == {}
    assert status.samples_on_the_fly == samples_on_the_fly


def test_dapo_reports_filtered_prompt_dispatch_ids():
    worker = _rollout_worker(n_generation=2)
    manager = PolicyStatusManager()
    manager.samples_on_the_fly = 2
    manager.remain_samples_num = 2
    dispatch_id = manager.register_prompt_dispatch(0)
    valid = SimpleNamespace(
        valid=True,
        completions=["a", "b"],
        prompt_dispatch_id="dispatch-valid",
    )
    filtered = SimpleNamespace(
        valid=False,
        completions=["c", "d"],
        filter_rewards=[1.0, 1.0],
        prompt_dispatch_id=dispatch_id,
    )

    payloads, metrics = worker.dynamic_sampling([valid, filtered])

    assert payloads == [valid]
    assert metrics == {
        "sampled": 2,
        "filtered_positive": 2,
        "filtered_prompt_dispatch_ids": [dispatch_id],
    }

    manager.update_dynamic_sampling_statistics(metrics)
    assert manager.samples_on_the_fly == 0
    assert manager.weight_version_to_prompt_num == {}


def test_colocated_dapo_does_not_report_prompt_dispatch_ids():
    worker = _rollout_worker(
        n_generation=2,
        worker_type=ColocatedRolloutControlWorker,
    )
    filtered = SimpleNamespace(
        valid=False,
        completions=["a", "b"],
        filter_rewards=[-1.0, -1.0],
        prompt_dispatch_id="dispatch-filtered",
    )

    _, metrics = worker.dynamic_sampling([filtered])

    assert metrics == {"filtered_negative": 2}
