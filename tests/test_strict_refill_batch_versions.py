# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""A shared-queue strict fetch must reserve only its permitted version."""

import asyncio
from queue import Queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.rollout import State
from cosmos_rl.rollout.worker.rollout_control import DisaggregatedRolloutControlWorker
from test_zero_staleness_refill_deadlock import _config, _controller


def controller(*, issued=12, pending=22, staleness=0, on_policy=False):
    config = _config(on_policy=on_policy, allowed_outdated_steps=staleness)
    config.rollout.n_generation = 2
    config.train.train_policy.max_inflight_steps = staleness + 1
    config.train.train_policy.data_dispatch_as_rank_in_mesh = False
    item = _controller(
        config, current_step=4, samples_on_the_fly=pending, pending_rollouts=pending
    )
    item.weight_version_to_prompt_num = {4: issued}
    item.weight_version_to_replacement_prompt_num = {4: 1}
    item.weight_version_to_replacement_prompt_issued = {}
    item.weight_version_to_discarded_sample_num = {4: 2}
    item.data_fetcher.get_batched_prompt = Mock(
        side_effect=lambda n, *args, **kwargs: (
            [RLPayload(prompt_idx=i) for i in range(n)],
            False,
        )
    )
    return item


def fetch(item, count):
    return asyncio.run(item._get_batched_prompt_impl(count))


@pytest.mark.parametrize("size", [1, 2, 4])
@pytest.mark.parametrize("on_policy", [False, True])
def test_refill_is_capped_before_fetching_and_reserving(size, on_policy):
    item = controller(on_policy=on_policy)
    payloads, is_end = fetch(item, size)
    assert not is_end
    assert [p.weight_version for p in payloads] == [4]
    assert item.data_fetcher.get_batched_prompt.call_args.args[0] == 1
    assert item.policy_status_manager.samples_on_the_fly == 24
    assert item.weight_version_to_prompt_num == {4: 13}
    assert item.weight_version_to_replacement_prompt_issued == {4: 1}


def test_early_replacement_preserves_all_unissued_ordinary_capacity():
    item = controller(issued=2, pending=2)
    payloads, _ = fetch(item, 4)
    assert [p.weight_version for p in payloads] == [4] * 4
    assert item.weight_version_to_replacement_prompt_issued == {4: 1}
    payloads, _ = fetch(item, 20)
    assert [p.weight_version for p in payloads] == [4] * 7
    assert item.policy_status_manager.samples_on_the_fly == 24
    assert item.weight_version_to_prompt_num == {4: 13}


def test_repeated_rejections_then_step_advance_leave_full_next_quota():
    item = controller()
    for attempt in range(3):
        payloads, _ = fetch(item, 4)
        assert [p.weight_version for p in payloads] == [4]
        assert item.policy_status_manager.samples_on_the_fly == 24
        if attempt < 2:
            item.policy_status_manager.samples_on_the_fly -= 2
            assert item.register_discarded_samples_for_refill(4, 2) == 1
    assert item.weight_version_to_prompt_num == {4: 15}
    item.policy_status_manager.current_step = 5
    item.policy_status_manager.samples_on_the_fly = 0
    item.policy_status_manager.total_pending_rollouts.return_value = 0
    payloads, _ = fetch(item, 20)
    assert [p.weight_version for p in payloads] == [5] * 12
    assert item.policy_status_manager.samples_on_the_fly == 24
    assert item.weight_version_to_prompt_num[5] == 12
    assert 4 not in item.weight_version_to_replacement_prompt_issued


def test_spent_capacity_does_not_fetch_or_signal_dataset_exhaustion():
    item = controller()
    item.weight_version_to_replacement_prompt_num.clear()
    assert fetch(item, 4) == ([], False)
    item.data_fetcher.get_batched_prompt.assert_not_called()
    assert item.policy_status_manager.samples_on_the_fly == 22


@pytest.mark.parametrize("validation_enabled", [False, True])
def test_spent_capacity_preserves_training_completion_signal(validation_enabled):
    item = controller()
    item.config.validation.enable = validation_enabled
    item.config.train.train_policy.max_inflight_steps = None
    item.weight_version_to_replacement_prompt_num.clear()
    item.policy_status_manager.training_finished.return_value = True
    assert fetch(item, 4) == ([], not validation_enabled)
    item.data_fetcher.get_batched_prompt.assert_not_called()


def test_positive_staleness_retains_existing_cross_version_fetch_behavior():
    item = controller(staleness=1)
    payloads, _ = fetch(item, 2)
    assert [p.weight_version for p in payloads] == [4, 5]
    assert item.policy_status_manager.samples_on_the_fly == 26


def test_initial_strict_batch_stops_at_ordinary_quota():
    item = controller(issued=0, pending=0)
    item.weight_version_to_replacement_prompt_num.clear()
    payloads, _ = fetch(item, 20)
    assert [p.weight_version for p in payloads] == [4] * 12
    assert item.policy_status_manager.samples_on_the_fly == 24


@pytest.mark.parametrize("path", ["validation", "dapo", "colocated"])
def test_other_fetch_modes_preserve_requested_batch_size(path):
    item = controller()
    kwargs = {}
    if path == "validation":
        kwargs["validation_step"] = 4
    elif path == "dapo":
        item.config.train.train_policy.variant = "dapo"
    else:
        item.config.mode = "colocated"
        item.policy_status_manager.training_dispatches = {}
    payloads, _ = asyncio.run(item._get_batched_prompt_impl(4, **kwargs))
    assert len(payloads) == 4
    assert item.data_fetcher.get_batched_prompt.call_args.args[0] == 4


@pytest.mark.parametrize(
    "scaling_manager", ["policy_status_manager", "rollout_status_manager"]
)
def test_scaling_retains_existing_unversioned_fetch_behavior(scaling_manager):
    item = controller()
    getattr(item, scaling_manager).replica_scaling_log.append(object())
    payloads, _ = fetch(item, 4)
    assert len(payloads) == 4
    assert item.data_fetcher.get_batched_prompt.call_args.args[0] == 4


def test_worker_generates_only_prompts_valid_for_its_loaded_weights():
    item = controller()
    payloads, _ = fetch(item, 2)
    worker = object.__new__(DisaggregatedRolloutControlWorker)
    worker.replica_name = "rollout-0"
    worker.global_rank = 0
    worker.api_client = SimpleNamespace(training_fetch_failed=False)
    worker.config = SimpleNamespace(
        rollout=SimpleNamespace(async_r2r_sync="disabled", prefetch_rollout=True),
        train=SimpleNamespace(train_policy=SimpleNamespace(allowed_outdated_steps=0)),
    )
    worker.state = State()
    worker.state.set_weight_synced()
    worker.current_weight_version = 4
    worker.validation_flag = threading.Event()
    worker.shutdown_signal = threading.Event()
    worker.parallel_dims = SimpleNamespace(world_size=1)
    worker._is_async_rollout = False
    worker._prompt_queue = Queue()
    worker._prompt_queue.put(payloads)
    worker.consume_command = Mock(side_effect=[None, AssertionError("worker stalled")])
    worker.report_rollouts = Mock(return_value=(None, False, None, True))
    worker._maybe_emit_mainloop_summary = Mock()
    generated = []

    def generate():
        batch = worker._prompt_queue.get_nowait()
        assert all(p.weight_version <= worker.current_weight_version for p in batch)
        generated.extend(batch)
        worker.shutdown_signal.set()

    worker.one_step_generation = generate
    worker._main_loop_impl()
    assert [p.weight_version for p in generated] == [4]
    assert worker._mainloop_branch_counts["gen_succeeded"] == 1
