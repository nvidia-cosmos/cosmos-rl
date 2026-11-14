# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Rollout-only controller/worker lifecycle contracts without model downloads."""

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import toml

from cosmos_rl.dispatcher import run_web_panel as web
from cosmos_rl.dispatcher.controller import Controller
from cosmos_rl.dispatcher.data.schema import RLPayload, TrainingCompletionIdentity
from cosmos_rl.dispatcher.protocol import Role
from cosmos_rl.dispatcher.replica import Atom, Replica
from cosmos_rl.policy.config import Config
from cosmos_rl.launcher import launch_all
from cosmos_rl.rollout import State
import test_rollout_report_receipts as receipts
import test_rollout_resource_ownership as ownership


def configuration():
    return Config.model_validate(
        {
            "train": {"train_policy": {"type": "grpo"}},
            "policy": {"parallelism": {"n_init_replicas": 0}},
            "rollout": {"rollout_only": True, "max_inflight_prompts": 2},
        }
    )


def controller():
    instance = object.__new__(Controller)
    instance._init_dist()
    instance._init_status()
    instance.config = configuration()
    instance.policy_status_manager.config = instance.config
    instance.data_fetcher = SimpleNamespace(get_batched_prompt=Mock())
    return instance


@pytest.mark.parametrize(
    "section,changes",
    [
        ("policy", {"parallelism": {"n_init_replicas": 1}}),
        ("rollout", {"parallelism": {"n_init_replicas": 0}}),
        ("rollout", {"max_inflight_prompts": 0}),
        ("rollout", {"max_prompts": 0}),
        ("rollout", {"backend": "trtllm"}),
        ("rollout", {"completion_admission": True}),
        ("rollout", {"async_r2r_sync": "generation"}),
        ("validation", {"enable": True}),
        ("distillation", {"enable": True}),
        ("train", {"resume": True}),
        ("train", {"train_policy": {"type": "sft"}}),
        ("train", {"train_policy": {"type": "grpo", "variant": "dapo"}}),
        (
            "train",
            {"train_policy": {"type": "grpo", "data_dispatch_as_rank_in_mesh": True}},
        ),
    ],
)
def test_rejects_unsupported_configuration(section, changes):
    data = configuration().model_dump()
    data[section].update(changes)
    with pytest.raises(ValueError):
        Config.model_validate(data)


def fetch_source(ctrl, monkeypatch):
    atom = Atom(
        0,
        "127.0.0.1",
        "host",
        None,
        [0] * 4,
        [1] * 4,
        "source",
        report_session_id="session",
    )
    replica = Replica("source", Role.ROLLOUT, [atom])
    ctrl.rollout_status_manager.rollout_replicas = {"source": replica}
    monkeypatch.setattr(web, "controller", ctrl)

    def fetch(n, sequence):
        return asyncio.run(
            web.get_batched_prompt(
                n,
                src_replica_name="source",
                src_global_rank=0,
                fetch_session_id="session",
                fetch_sequence=sequence,
            )
        )

    return replica, fetch


def test_retirement_releases_only_rollout_owned_capacity(monkeypatch):
    ctrl = controller()
    ctrl.data_fetcher.get_batched_prompt.return_value = (
        [RLPayload(prompt_idx=0)],
        False,
    )
    replica, fetch = fetch_source(ctrl, monkeypatch)
    fetch(1, 0)
    assert ctrl.rollout_only_state.outstanding_completions == 16
    ctrl.rollout_status_manager.release_producer_reservations(
        replica, ctrl.policy_status_manager
    )
    ctrl.rollout_status_manager.release_producer_reservations(
        replica, ctrl.policy_status_manager
    )
    assert ctrl.rollout_only_state.outstanding_completions == 0
    assert ctrl.policy_status_manager.samples_on_the_fly == 0
    assert ctrl.policy_status_manager.remain_samples_num == 0


def test_example_config_is_valid():
    root = Path(__file__).resolve().parents[1]
    config = Config.model_validate(toml.load(root / "configs/rollout-only.toml"))
    assert config.rollout.rollout_only
    assert config.policy.parallelism.n_init_replicas == 0


def test_global_limits_do_not_overissue_or_drop_dataloader_remainder(monkeypatch):
    ctrl = controller()
    ctrl.config.rollout.max_prompts = 3
    ctrl.data_fetcher.get_batched_prompt.return_value = (
        [RLPayload(prompt_idx=i) for i in range(6)],
        False,
    )
    replica, fetch = fetch_source(ctrl, monkeypatch)
    first = fetch(20, 0)
    assert [p["prompt_idx"] for p in first["payloads_list"]] == [0, 1]
    assert not first["is_end"]
    assert ctrl.policy_status_manager.samples_on_the_fly == 0
    assert ctrl.rollout_only_state.outstanding_completions == 32
    assert fetch(20, 0) == first  # lost response retry cannot reserve twice
    # Another fetch sees a throttle, rather than end-of-data.
    assert fetch(20, 1) == {"payloads_list": [], "is_end": False}
    # Partially settling two separate prompt groups must not free a prompt.
    works = [p["training_work_id"] for p in first["payloads_list"]]
    ledger = replica.producer_reservations
    ledger.commit(ledger.prepare([(work, [0]) for work in works]))
    ctrl.rollout_only_state.settle(2)
    assert fetch(20, 2) == {"payloads_list": [], "is_end": False}
    ledger.commit(ledger.prepare([(works[0], list(range(1, 16)))]))
    ctrl.rollout_only_state.settle(15)
    second = fetch(20, 3)
    assert [p["prompt_idx"] for p in second["payloads_list"]] == [2]
    assert second["is_end"]
    assert all(
        p["weight_version"] == 0
        for p in first["payloads_list"] + second["payloads_list"]
    )
    assert fetch(20, 4) == {"payloads_list": [], "is_end": True}
    assert ctrl.policy_status_manager.samples_on_the_fly == 0
    assert ctrl.policy_status_manager.remain_samples_num == 0
    ctrl.data_fetcher.get_batched_prompt.assert_called_once_with(2)


def test_exhaustion_includes_last_batch_and_empty_input():
    ctrl = controller()
    ctrl.data_fetcher.get_batched_prompt.return_value = (
        [RLPayload(prompt_idx=0)],
        True,
    )
    payloads, ended = asyncio.run(ctrl._get_batched_prompt_impl(1))
    assert len(payloads) == 1 and ended
    assert asyncio.run(ctrl._get_batched_prompt_impl(1)) == ([], True)
    ctrl = controller()
    ctrl.data_fetcher.dataset = SimpleNamespace(train_set=[])
    assert asyncio.run(ctrl._get_batched_prompt_impl(1)) == ([], True)
    ctrl.data_fetcher.get_batched_prompt.assert_not_called()


def test_stop_closes_new_admission_but_does_not_change_model_version():
    ctrl = controller()
    ctrl.rollout_status_manager.rollout_init_done = True
    assert asyncio.run(ctrl.request_stop("evaluation complete"))
    assert not asyncio.run(ctrl.request_stop("second reason"))
    assert ctrl.policy_status_manager.stop_reason == "evaluation complete"
    assert asyncio.run(ctrl._get_batched_prompt_impl(1)) == ([], True)
    assert ctrl.policy_status_manager.current_step == 0


def test_training_without_live_policies_preserves_existing_end_signal():
    ctrl = controller()
    ctrl.config.rollout.rollout_only = False
    ctrl.config.policy.parallelism.n_init_replicas = 1
    assert asyncio.run(ctrl._get_batched_prompt_impl(1)) == ([], True)
    ctrl.data_fetcher.get_batched_prompt.assert_not_called()


def test_rejects_policy_registration_in_rollout_only():
    ctrl = controller()
    with pytest.raises(ValueError, match="only rollout"):
        asyncio.run(ctrl.register(SimpleNamespace(), Role.POLICY))
    assert len(ctrl.policy_status_manager) == 0


def test_report_retry_settles_and_cleans_without_trainer_buffer(monkeypatch):
    ctrl, atom, request = receipts.setup(monkeypatch)
    ctrl.config.rollout.rollout_only = True
    ctrl.policy_status_manager.samples_on_the_fly = 0
    ctrl.rollout_only_state = ctrl.policy_status_manager.rollout_only_state
    ctrl.rollout_only_state.outstanding_completions = 2
    ctrl.policy_status_manager.policy_replicas.clear()
    cleanup = Mock()
    ctrl.policy_status_manager._publish_payload_transport_cleanup = cleanup
    rejected = ctrl.rollout_status_manager["source"].producer_reservations.issue(0, 1)
    request.training_rejections = [TrainingCompletionIdentity(work_id=rejected, slot=0)]
    request.metrics = {"discarded_samples": 1}
    expected = {"message": "Rollout-only report processed"}
    assert receipts.post(request) == receipts.post(request) == expected
    assert ctrl.policy_status_manager.samples_on_the_fly == 0
    assert ctrl.rollout_only_state.outstanding_completions == 0
    assert ctrl.policy_status_manager.rollout_buffer.empty()
    assert ctrl.stat_n_samples == 1
    assert ctrl.policy_status_manager.filter_records == {}
    assert atom.rollout_report_receipt.sequence == 0
    cleanup.assert_called_once()
    ctrl.policy_status_manager.try_trigger_data_fetch_and_training.assert_not_called()


@pytest.mark.parametrize("failure", [None, "init", "post", "generation"])
def test_model_initialization_and_owned_cleanup(failure):
    events = []
    worker = ownership.control(events)
    worker.config = configuration()
    worker.state = State()
    worker._is_async_rollout = False
    worker.quantization_type = None
    worker.consume_command = Mock()
    worker.report_rollouts = Mock()
    worker.validation_flag = threading.Event()
    worker.prepare_shard_infos_for_weight_sync_insts = Mock()

    def initialize(**kwargs):
        assert kwargs["load_format"] == "auto"
        events.append("init")
        if failure == "init":
            raise ownership.WorkFailure("init")
        worker.rollout._engine_initialized = True

    def post(*args):
        events.append("post")
        if failure == "post":
            raise ownership.WorkFailure("post")

    worker.rollout.init_engine = initialize
    worker.rollout.post_init_engine_hook = post

    def generate():
        assert worker.state.weight_synced()
        # A repeated lazy entry cannot recreate the model or repeat its hook.
        worker.lazy_initialize_rollout_engine(load_format="auto")
        events.append("generation")
        if failure == "generation":
            raise ownership.WorkFailure("generation")

    worker.main_loop = generate
    if failure:
        with pytest.raises(ownership.WorkFailure, match=failure):
            worker.work()
    else:
        worker.work()
    assert events.count("init") == 1
    assert events.count("post") == (failure != "init")
    assert events.count("engine") == 1
    assert events[-1] == "unregister"
    assert events.index("producer transport") < events.index("engine")
    worker.prepare_shard_infos_for_weight_sync_insts.assert_not_called()


@pytest.mark.parametrize("ended", [False, True])
def test_monitor_stops_only_after_every_rollout_reporter_drains(monkeypatch, ended):
    ctrl = controller()
    policies, workers = ctrl.policy_status_manager, ctrl.rollout_status_manager
    policies.maintain_life_status = Mock()
    policies.redis_handler = SimpleNamespace(_publication_failure=None)
    workers.maintain_life_status = Mock()
    workers.rollout_init_done = True
    workers.redis_handler = Mock()
    workers.rollout_replicas = {
        "source": SimpleNamespace(status=SimpleNamespace(ended=ended))
    }
    ctrl.close_exhausted_rank_input = Mock()
    ctrl.teacher_result_manager = set()
    stopped = Mock()
    monkeypatch.setattr(web, "controller", ctrl)
    monkeypatch.setattr(web, "_policy_replicas_were_registered", False)
    monkeypatch.setattr(web, "COSMOS_ROLLOUT_SCAN_INTERVAL", 0.001)
    monkeypatch.setattr(web.StopCommand, "trigger", stopped)
    monkeypatch.setattr(web, "_observe_controller_monitor", lambda task: None)

    async def run():
        async with web.lifespan(web.app):
            await asyncio.sleep(0.02)

    asyncio.run(run())
    assert stopped.call_count == int(ended)


def test_controller_finalizes_after_rollout_only_unregister(monkeypatch):
    ctrl = controller()
    ctrl.is_rl = True
    server = SimpleNamespace(should_exit=False)
    monkeypatch.setattr(web, "controller", ctrl)
    monkeypatch.setattr(web, "server", server)
    assert not ctrl.policy_status_manager.training_finished()
    assert web._maybe_finalize("clean unregister of source")
    assert server.should_exit


@pytest.mark.parametrize("controller_first", [False, True])
def test_launcher_does_not_mask_worker_failure_after_controller_exit(
    monkeypatch, controller_first
):
    worker = Mock(pid=101)
    worker.poll.return_value = 1  # e.g. init_engine/post hook failed after registration
    controller = Mock(pid=102)
    controller.poll.return_value = 0  # last unregister already finalized HTTP server
    peer = Mock(pid=103)
    peer.poll.return_value = None
    victims = {}

    def process(pid):
        victim = Mock()
        victim.children.return_value = []
        victims[pid] = victim
        return victim

    monkeypatch.setattr("psutil.Process", process)
    processes = (
        [controller, worker, peer] if controller_first else [worker, controller, peer]
    )
    with pytest.raises(SystemExit) as failure:
        launch_all._monitor_processes(processes, controller, fail_on_worker_error=True)
    assert failure.value.code == 1
    victims[peer.pid].kill.assert_called_once()
