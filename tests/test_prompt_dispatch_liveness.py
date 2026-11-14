# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Terminal losses must not exhaust the usable prompt-version window."""

import asyncio
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from cosmos_rl.dispatcher.controller import Controller
from cosmos_rl.dispatcher.status import PolicyStatusManager
from test_dispatch_fetch_bounds import batched_fetcher, fetcher
from test_zero_staleness_refill_deadlock import _config


def controller(*, lag=1, group=1, quota=1, replicas=1):
    config = _config(on_policy=False, allowed_outdated_steps=lag)
    config.rollout.n_generation = group
    config.train.train_batch_per_replica = quota * group
    config.train.train_policy.data_dispatch_as_rank_in_mesh = False
    config.train.train_policy.max_inflight_steps = lag + 1
    status = PolicyStatusManager()
    status.config = config
    status.total_steps = 100
    status.policy_replicas = {
        str(rank): SimpleNamespace(all_atoms_arrived=True) for rank in range(replicas)
    }
    item = object.__new__(Controller)
    item.config = config
    item.policy_status_manager = status
    item.rollout_status_manager = SimpleNamespace(replica_scaling_log=[])
    item.data_fetcher = fetcher(quota=quota * group)
    item.data_fetcher.config.rollout.n_generation = group
    item.data_fetcher.config.train.train_policy.data_dispatch_as_rank_in_mesh = False
    item.weight_version_to_prompt_num = {}
    item._soft_throttle_engaged_since = None
    status.set_discard_refill_hook(item.register_discarded_samples_for_refill)
    return item


def fetch(item, count):
    payloads, end = asyncio.run(item._get_batched_prompt_impl(count))
    assert not end
    return payloads


def discard(item, payload, *, count=None, report=None):
    return item.policy_status_manager.settle_discarded_samples(
        "source",
        report or f"drop-{payload.index}",
        item.config.rollout.n_generation if count is None else count,
        weight_version=payload.weight_version,
    )


@pytest.mark.parametrize("lag", [1, 4])
@pytest.mark.parametrize("burst", [1, 8])
@pytest.mark.parametrize("group", [1, 8])
def test_all_usable_groups_can_be_lost_repeatedly_without_a_weight_update(
    lag, burst, group
):
    item = controller(lag=lag, group=group)
    for _ in range(3):
        issued = []
        while payloads := fetch(item, burst):
            issued.extend(payloads)
        assert [p.weight_version for p in issued] == list(range(lag + 1))
        assert item.policy_status_manager.samples_on_the_fly == (lag + 1) * group
        assert len(item.data_fetcher.pulled) == sum(
            item.weight_version_to_prompt_num.values()
        )
        for payload in issued:
            assert discard(item, payload) == group
        assert item.policy_status_manager.samples_on_the_fly == 0
        assert item.policy_status_manager.current_step == 0


def test_large_fetch_is_bounded_before_advancing_dataset():
    item = controller()
    assert [p.weight_version for p in fetch(item, 8)] == [0, 1]
    assert item.data_fetcher.pulled == [0, 1]
    assert fetch(item, 8) == []
    assert item.data_fetcher.pulled == [0, 1]


@pytest.mark.parametrize("lag", [0, 1, 4])
@pytest.mark.parametrize("group", [1, 2, 8])
def test_terminal_losses_with_dataloader_batches_larger_than_window(lag, group):
    item = controller(lag=lag, group=group)
    item.data_fetcher = batched_fetcher()
    item.data_fetcher.config.rollout.n_generation = group
    issued_indices = []
    for _ in range(3):
        issued = fetch(item, 8)
        assert [p.weight_version for p in issued] == list(range(lag + 1))
        assert item.policy_status_manager.samples_on_the_fly == (lag + 1) * group
        issued_indices.extend(p.prompt_idx for p in issued)
        assert issued_indices == list(range(len(issued_indices)))
        assert all(
            p.training_work_id is None for _, p in item.data_fetcher.fetched_data_buffer
        )
        before = list(item.data_fetcher.pulled)
        assert fetch(item, 8) == []
        assert item.data_fetcher.pulled == before
        for payload in issued:
            assert discard(item, payload, report=f"drop-{payload.prompt_idx}") == group
        assert item.policy_status_manager.samples_on_the_fly == 0
        assert item.policy_status_manager.current_step == 0


def test_window_diagnostics_include_capacity_and_admitted_work(caplog):
    caplog.set_level("INFO", logger="cosmos")
    item = controller()
    fetch(item, 8)
    message = next(
        r.getMessage()
        for r in caplog.records
        if "Prompt window limited" in r.getMessage()
    )
    assert "current_step=0 latest_reported_weight_version=-1" in message
    assert "starting_version=0 ceiling=1" in message
    assert "pending_admitted=0 dispatched_steps=[] outstanding=0" in message
    assert "normal_slots={0: 1, 1: 1} refill_slots={0: 0, 1: 0}" in message


def test_replayed_terminal_receipt_neither_settles_nor_refills_twice():
    item = controller()
    first = fetch(item, 1)[0]
    assert discard(item, first, report="receipt") == 1
    assert discard(item, first, report="receipt") == 0
    assert item.register_discarded_samples_for_refill(0, 1, "receipt") == 0
    assert item.policy_status_manager.samples_on_the_fly == 0
    assert item.weight_version_to_replacement_prompt_num == {0: 1}


def test_partial_losses_and_early_replacement_preserve_remaining_normal_quota():
    item = controller(lag=0, group=4, quota=4)
    first = fetch(item, 1)[0]
    assert discard(item, first, count=1, report="first-slot") == 1
    replacement = fetch(item, 1)[0]
    assert replacement.weight_version == 0
    assert discard(item, first, count=3, report="remaining-slots") == 3
    assert item.weight_version_to_replacement_prompt_num == {0: 0}
    assert [p.weight_version for p in fetch(item, 8)] == [0, 0, 0]
    assert item.weight_version_to_prompt_num == {0: 5}
    assert item.weight_version_to_replacement_prompt_issued == {0: 1}
    assert fetch(item, 1) == []


def test_pending_admitted_batch_preserves_native_starting_version():
    item = controller(lag=1, quota=2)
    item.policy_status_manager.rollout_buffer.put(object())
    item.policy_status_manager.rollout_buffer.put(object())
    item.policy_status_manager.samples_on_the_fly = 2
    item.weight_version_to_prompt_num = {0: 2}
    assert [p.weight_version for p in fetch(item, 8)] == [1, 1]
    assert item.data_fetcher.pulled == [0, 1]


def test_completed_versions_are_not_reissued_after_the_controller_advances():
    item = controller()
    first = fetch(item, 1)[0]
    discard(item, first)
    item.policy_status_manager.current_step = 1
    assert [p.weight_version for p in fetch(item, 8)] == [1, 2]
    assert 0 not in item.weight_version_to_replacement_prompt_num


@pytest.mark.parametrize("step,staged,lag", [(1, 0, 1), (3, 2, 1), (4, 2, 4)])
def test_batch_fits_weights_available_between_sync_intervals(step, staged, lag):
    from cosmos_rl.rollout.prompt_batch import required_weight_version

    item = controller(lag=lag)
    item.config.train.sync_weight_interval = 2
    status = item.policy_status_manager
    status.current_step = step
    status._weight_last_staged_step = staged
    status._weight_max_adopted_version = staged
    payloads = fetch(item, 8)
    assert [p.weight_version for p in payloads] == list(range(step, staged + lag + 1))
    assert required_weight_version(payloads) <= staged + lag
    assert status.total_pending_rollouts() == 0
    assert not status.dispatched_rollouts_by_step
    # Losing every group must still permit a batch usable with these weights.
    for payload in payloads:
        discard(item, payload)
    replacements = fetch(item, 8)
    assert [p.weight_version for p in replacements] == [
        p.weight_version for p in payloads
    ]
    assert required_weight_version(replacements) <= staged + lag


def test_inflight_weight_transfer_can_unblock_a_batch_without_more_training():
    item = controller(lag=1)
    status = item.policy_status_manager
    status.current_step = 3
    status._weight_last_staged_step = 3
    status._weight_max_adopted_version = 2
    assert [p.weight_version for p in fetch(item, 8)] == [3, 4]


def test_admitted_update_can_advance_a_spent_reachable_window():
    item = controller(lag=1)
    status = item.policy_status_manager
    status.current_step = 3
    status._weight_last_staged_step = status._weight_max_adopted_version = 2
    status.rollout_buffer.put(object())
    status.samples_on_the_fly = 1
    assert fetch(item, 8) == []
    assert item.data_fetcher.pulled == []
    # That already-admitted batch can finish step 4 and stage its weights.
    status.rollout_buffer.get_nowait()
    status.current_step = status._weight_last_staged_step = 4
    status.samples_on_the_fly = 0
    assert [p.weight_version for p in fetch(item, 8)] == [4, 5]


def test_retry_budget_also_checks_replacements_at_a_later_version():
    item = controller()
    item.config.train.train_policy.max_retry_for_on_policy = 1
    first, second = fetch(item, 8)
    assert first.weight_version == 0 and second.weight_version == 1
    discard(item, second)
    with pytest.raises(RuntimeError, match="weight version 1"):
        fetch(item, 1)


@pytest.mark.parametrize("budget", [0, -1])
def test_nonpositive_retry_budget_keeps_refill_unbounded(budget):
    item = controller()
    item.config.train.train_policy.max_retry_for_on_policy = budget
    for _ in range(5):
        payload = fetch(item, 1)[0]
        assert payload.weight_version == 0
        discard(item, payload)


@pytest.mark.parametrize("replicas", [2, 3])
def test_separate_producers_share_the_global_usable_quota(replicas):
    item = controller(replicas=replicas)
    issued = [fetch(item, 1)[0] for _ in range(2 * replicas)]
    assert [p.weight_version for p in issued] == [0] * replicas + [1] * replicas
    assert fetch(item, 8) == []
    discard(item, issued[0])
    assert [p.weight_version for p in fetch(item, 8)] == [0]
    assert fetch(item, 1) == []


def test_empty_window_still_signals_training_completion():
    item = controller(lag=0)
    item.weight_version_to_prompt_num = {0: 1}
    item.policy_status_manager.training_finished = lambda: True
    assert asyncio.run(item._get_batched_prompt_impl(8)) == ([], True)
    assert item.data_fetcher.pulled == []


@pytest.mark.parametrize("overflow", [0, 8])
def test_whole_fetch_respects_remaining_inflight_capacity(overflow):
    item = controller(lag=4, group=4, quota=3)
    policy = item.config.train.train_policy
    policy.max_inflight_steps = 5
    policy.outdated_rollout_fetch_batch_size = overflow
    item.policy_status_manager.samples_on_the_fly = 5 * 12 - 5
    # Only two whole groups fit the remaining five-completion capacity.
    assert len(fetch(item, 8)) == 2
    assert fetch(item, 8) == []


def test_nondivisible_training_batch_can_fetch_its_last_whole_group():
    item = controller(lag=0, group=2)
    item.config.train.train_batch_per_replica = 3
    assert [p.weight_version for p in fetch(item, 8)] == [0, 0]
    assert item.policy_status_manager.samples_on_the_fly == 4


@pytest.mark.parametrize("mode", ["validation", "sft", "dapo", "colocated", "scaling"])
def test_other_dispatch_modes_keep_their_existing_fetch_path(mode):
    from cosmos_rl.dispatcher.data.schema import RLPayload

    item = controller()
    item.data_fetcher.get_batched_prompt = Mock(return_value=([RLPayload()], False))
    kwargs = {}
    if mode == "validation":
        kwargs["validation_step"] = 0
    elif mode == "sft":
        item.config.train.train_policy.type = "sft"
    elif mode == "dapo":
        item.config.train.train_policy.variant = "dapo"
    elif mode == "colocated":
        item.config.mode = "colocated"
    else:
        item.rollout_status_manager.replica_scaling_log = ["scaling"]
    asyncio.run(item._get_batched_prompt_impl(8, **kwargs))
    assert item.data_fetcher.get_batched_prompt.call_args.args[0] == 8


def rejection(payloads, *, sequence=0):
    from cosmos_rl.dispatcher.data.schema import TrainingCompletionIdentity
    from cosmos_rl.dispatcher.protocol import RolloutRequest

    return RolloutRequest(
        src_replica_name="source",
        src_global_rank=0,
        report_session_id="session",
        report_sequence=sequence,
        payloads=[],
        training_rejections=[
            TrainingCompletionIdentity(work_id=p.training_work_id, slot=slot)
            for p in payloads
            for slot in p.training_completion_slots
        ],
        metrics={
            "discarded_samples": sum(len(p.training_completion_slots) for p in payloads)
        },
    )


def test_native_http_settlement_replacement_admission_and_training_ack(monkeypatch):
    from cosmos_rl.dispatcher.data.schema import RLPayload
    from test_training_fetch_receipts import setup, fetch as http_fetch
    from test_rollout_report_receipts import post
    from test_producer_reservations import completed_request

    item, _, identity = setup(monkeypatch)
    status = item.policy_status_manager
    policy = item.config.train.train_policy
    policy.on_policy, policy.allowed_outdated_steps = False, 1
    policy.max_inflight_steps = 2
    status.set_discard_refill_hook(item.register_discarded_samples_for_refill)
    original = [
        RLPayload.model_validate(p) for p in http_fetch(8, identity)["payloads_list"]
    ]
    assert [p.weight_version for p in original] == [0, 1]
    dropped = rejection(original)
    assert post(dropped) == post(dropped) == {"message": "Rollout put"}
    assert status.samples_on_the_fly == 0
    assert item.weight_version_to_replacement_prompt_num == {0: 1, 1: 1}
    identity["fetch_sequence"] = 1
    (replacement,) = http_fetch(1, identity)["payloads_list"]
    replacement = RLPayload.model_validate(replacement)
    assert replacement.weight_version == 0
    accepted = completed_request(replacement, count=2)
    accepted.report_sequence = 1
    assert post(accepted) == {"message": "Rollout put"}
    assert status.rollout_buffer.qsize() == 2
    status.try_trigger_data_fetch_and_training = (
        PolicyStatusManager.try_trigger_data_fetch_and_training.__get__(status)
    )
    status.should_weight_sync_after_train_ack = Mock(return_value=False)
    status.try_trigger_data_fetch_and_training()
    assert status.current_step == 1
    assert status.training_dispatches[1].rollout_count == 2
    status.train_ack("policy-0", 1, 2, False, {}, item.rollout_status_manager)
    assert status.training_dispatches[1].settled
    assert status.samples_on_the_fly == 0


def test_retry_exhaustion_over_http_exits_controller_and_launcher_workers(tmp_path):
    """Exercise HTTP, the real lifespan monitor, and launcher containment."""
    from cosmos_rl.dispatcher.api.client import APIClient
    from cosmos_rl.dispatcher.data.schema import RLPayload
    from cosmos_rl.dispatcher.protocol import Role
    from cosmos_rl.launcher.launch_all import _monitor_processes
    from cosmos_rl.utils.api_suffix import COSMOS_API_NEXT_PROMPT_SUFFIX

    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    log = tmp_path / "controller.log"
    children = []
    try:
        with log.open("w") as output:
            server = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).with_name("prompt_dispatch_budget_canary.py")),
                    str(port),
                ],
                stdout=output,
                stderr=subprocess.STDOUT,
            )
            children.append(server)
            worker = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(120)"]
            )
            children.append(worker)
            url = f"http://127.0.0.1:{port}"
            params = dict(
                n=8,
                src_replica_name="source",
                src_global_rank=0,
                fetch_session_id="session",
                fetch_sequence=0,
            )
            deadline = time.monotonic() + 45
            while True:
                assert server.poll() is None, log.read_text()
                try:
                    reply = requests.get(
                        url + COSMOS_API_NEXT_PROMPT_SUFFIX, params=params, timeout=1
                    )
                    reply.raise_for_status()
                    break
                except requests.ConnectionError:
                    assert time.monotonic() < deadline, log.read_text()
                    time.sleep(0.05)
            original = [
                RLPayload.model_validate(p) for p in reply.json()["payloads_list"]
            ]
            assert [p.weight_version for p in original] == [0, 1]
            client = APIClient(Role.ROLLOUT, remote_ips=["127.0.0.1"], remote_port=port)
            client._registered_replica_name, client._registered_global_rank = (
                "source",
                0,
            )
            client._report_session_id, client._fetch_sequence = "session", 1
            assert client.post_rollout_completion(rejection(original))
            with pytest.raises(RuntimeError, match="bounded HTTP"):
                client.get_next_prompt(8)
            assert client.training_fetch_failed
            assert server.wait(timeout=10) == 86, log.read_text()
            with pytest.raises(SystemExit) as failed:
                _monitor_processes(children, controller=server)
            assert failed.value.code == 1
            assert worker.returncode == -signal.SIGKILL
        assert "After 1 retries, samples for weight version 0" in log.read_text()
        assert "Controller monitor failed; terminating execution" in log.read_text()
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)


@pytest.mark.parametrize(
    "mode,variant,policy_type",
    [
        ("colocated", "grpo", "grpo"),
        ("disaggregated", "dapo", "grpo"),
        ("disaggregated", "grpo", "sft"),
    ],
)
def test_non_quota_modes_do_not_acquire_global_refill_credits(
    mode, variant, policy_type
):
    item = controller()
    item.config.mode = mode
    item.config.train.train_policy.variant = variant
    item.config.train.train_policy.type = policy_type
    assert item.register_discarded_samples_for_refill(0, 1, "drop") == 0
    assert not getattr(item, "weight_version_to_replacement_prompt_num", {})
