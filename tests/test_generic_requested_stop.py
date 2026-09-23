# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from cosmos_rl.dispatcher.step_boundary import StepBoundary
from cosmos_rl.dispatcher.status import PolicyStatusManager, PolicyStatus
from cosmos_rl.dispatcher.controller import Controller
from cosmos_rl.policy.config import Config
from cosmos_rl.policy.worker import stop


def test_stop_at_shared_boundary_not_in_middle_of_issued_update():
    async def run():
        barrier = StepBoundary()
        reason = None
        parties = {"p0", "p1"}
        first = await asyncio.gather(
            *(barrier.arrive(p, 0, parties, lambda: reason) for p in parties)
        )
        assert all(not reply["stop"] for reply in first)
        reason = "budget"
        assert not (await barrier.arrive("p0", 0, parties, lambda: reason))["stop"]
        early = asyncio.create_task(barrier.arrive("p0", 1, parties, lambda: reason))
        await asyncio.sleep(0)
        assert not early.done()
        late = await barrier.arrive("p1", 1, parties, lambda: reason)
        assert late == await early == {"stop": True, "step": 1, "reason": "budget"}
        assert not barrier.complete("p0", 1)
        assert not barrier.complete("p0", 1)
        assert barrier.complete("p1", 1)
        with pytest.raises(ValueError):
            await barrier.arrive("p0", 2, parties, lambda: reason)

    asyncio.run(run())


def test_stop_while_one_replica_waits_prevents_next_update_for_every_replica():
    async def run():
        barrier = StepBoundary()
        reason = None
        first = asyncio.create_task(barrier.arrive("a", 0, {"a", "b"}, lambda: reason))
        await asyncio.sleep(0)
        reason = "quality"
        second = await barrier.arrive("b", 0, {"a", "b"}, lambda: reason)
        assert second["stop"] and (await first)["stop"]
        assert barrier.stopped_step == 0

    asyncio.run(run())


def test_disagreement_unblocks_waiters_with_error():
    async def run():
        barrier = StepBoundary()
        replies = await asyncio.gather(
            barrier.arrive("a", 2, {"a", "b"}, lambda: None),
            barrier.arrive("b", 3, {"a", "b"}, lambda: None),
            return_exceptions=True,
        )
        assert all(isinstance(reply, ValueError) for reply in replies)

    asyncio.run(run())


def test_natural_completion_racing_stop_uses_real_completed_step():
    async def run():
        barrier = StepBoundary()
        await barrier.arrive("a", 9, {"a"}, lambda: None)
        assert barrier.complete("a", 10)
        with pytest.raises(ValueError):
            barrier.complete("a", 11)

    asyncio.run(run())


def test_boundary_endpoint_requires_every_replica_final_ack(monkeypatch):
    from cosmos_rl.dispatcher import run_web_panel as panel
    from cosmos_rl.dispatcher.protocol import StepBoundaryRequest

    manager = SimpleNamespace(
        step_boundary=StepBoundary(),
        stop_reason="budget",
        _stop_policy_recipients={"a", "b"},
        terminal_complete=False,
        current_step=0,
        get_all_atoms_arrived_replicas=lambda: [
            SimpleNamespace(name=p) for p in ("a", "b")
        ],
    )
    config = Config()
    config.train.train_policy.type = "sft"
    monkeypatch.setattr(
        panel,
        "controller",
        SimpleNamespace(config=config, policy_status_manager=manager),
    )

    async def run():
        replies = await asyncio.gather(
            *(
                panel.training_boundary(
                    StepBoundaryRequest(replica_name=p, completed_step=3)
                )
                for p in ("a", "b")
            )
        )
        assert all(reply["stop"] for reply in replies)
        for p in ("a", "b"):
            result = await panel.training_boundary(
                StepBoundaryRequest(
                    replica_name=p, completed_step=3, checkpoint_complete=True
                )
            )
            assert result["complete"] == (p == "b")
            assert manager.terminal_complete == (p == "b")
        assert manager.current_step == 3
        invalid = await panel.training_boundary(
            StepBoundaryRequest(
                replica_name="a", completed_step=4, checkpoint_complete=True
            )
        )
        assert invalid.status_code == 409

    asyncio.run(run())


@pytest.mark.parametrize(
    "policy,mode",
    [("sft", "disaggregated"), ("grpo", "colocated"), ("grpo", "colocated_separated")],
)
def test_request_is_not_restricted_to_disaggregated_grpo(policy, mode):
    manager = PolicyStatusManager()
    manager.config = Config()
    manager.config.train.train_policy.type = policy
    manager.config.mode = mode
    manager.policy_init_done = True
    manager.total_steps = 100
    manager.data_fetcher = SimpleNamespace(activated_val_iter=None)
    manager.get_all_atoms_arrived_replicas = Mock(
        return_value=[SimpleNamespace(name="p0")]
    )
    manager.status = {"p0": PolicyStatus.RUNNING}
    manager.cleanup_buffered_rollouts = Mock()
    manager.trigger_training_complete = Mock()
    controller = Controller.__new__(Controller)
    controller.config = manager.config
    controller.policy_status_manager = manager
    controller.life_cycle_lock = asyncio.Lock()
    controller.rollout_status_manager = Mock()
    controller.rollout_status_manager.get_all_atoms_arrived_replicas.return_value = [
        object()
    ]
    assert asyncio.run(controller.request_stop("budget"))
    assert manager.stop_reason == "budget"
    manager.trigger_training_complete.assert_not_called()
    if policy == "sft":
        controller.rollout_status_manager.get_all_atoms_arrived_replicas.assert_not_called()


def worker():
    return SimpleNamespace(
        global_rank=0,
        replica_name="p0",
        train_step=4,
        api_client=Mock(),
        trainer=SimpleNamespace(ckpt_manager=Mock()),
    )


def test_worker_broadcasts_decision_and_does_not_ack_failed_checkpoint(monkeypatch):
    w = worker()
    w.api_client.training_boundary.return_value = {
        "stop": True,
        "reason": "budget",
        "step": 4,
    }
    monkeypatch.setattr(
        stop.dist_util, "broadcast_object_cpu", lambda value, **_: value
    )
    monkeypatch.setattr(
        stop.dist_util, "all_reduce_tensor_object_cpu", lambda value, **_: value
    )
    assert stop.training_boundary(w, 4)
    assert w.requested_stop_reason == "budget"
    w.api_client.reset_mock()
    with pytest.raises(OSError, match="save failed"):
        stop.final_checkpoint(w, Mock(side_effect=OSError("save failed")))
    w.api_client.training_boundary.assert_not_called()
    stop.final_checkpoint(w, Mock())
    w.trainer.ckpt_manager.finalize.assert_called_once()
    w.api_client.training_boundary.assert_called_once_with(
        "p0", 4, checkpoint_complete=True
    )


def test_peer_save_failure_prevents_success_ack(monkeypatch):
    w = worker()
    monkeypatch.setattr(
        stop.dist_util,
        "all_reduce_tensor_object_cpu",
        lambda value, **_: torch.tensor([0]),
    )
    with pytest.raises(RuntimeError, match="another rank"):
        stop.final_checkpoint(w, Mock())
    w.api_client.training_boundary.assert_not_called()


def test_native_sft_checkpoint_propagates_local_failure_before_peer_barrier(
    monkeypatch,
):
    from cosmos_rl.policy.trainer.llm_trainer.sft_trainer import SFTTrainer

    trainer = SimpleNamespace(
        config=SimpleNamespace(
            train=SimpleNamespace(ckpt=SimpleNamespace(enable_checkpoint=True))
        ),
        _checkpointing=Mock(side_effect=OSError("disk full")),
    )
    monkeypatch.setattr(
        stop.dist_util, "all_reduce_tensor_object_cpu", lambda value, **_: value
    )
    with pytest.raises(OSError, match="disk full"):
        SFTTrainer.checkpointing(trainer, 100, 0, 10, is_last_step=True)


def test_colocated_terminal_command_prevents_another_generation_iteration():
    from cosmos_rl.colocated.controller import ColocatedController
    from cosmos_rl.dispatcher.command import TrainingCompleteCommand

    controller = object.__new__(ColocatedController)
    command = TrainingCompleteCommand(
        replica_name="p0",
        global_step=3,
        total_steps=3,
        remain_samples_num=8,
        final_step=2,
        checkpoint_total_steps=100,
    )
    controller.policy_consume_one_step_commands_util_data_fetch = Mock(
        return_value=command
    )
    controller.command_dispatcher = Mock()
    controller.policy = Mock(replica_name="p0")
    controller.policy.consume_command.return_value = True
    controller.rollout = Mock()
    assert not controller.prepare_iteration()
    assert not controller.prepare_iteration()
    controller.policy.consume_command.assert_called_once_with(TrainingCompleteCommand)
    assert controller.requested_stop_complete


def test_colocated_local_stop_delegates_to_authoritative_controller():
    from cosmos_rl.colocated.controller import ColocatedController

    controller = object.__new__(ColocatedController)
    controller.policy = Mock()
    controller.policy.api_client.request_stop.return_value = True
    assert asyncio.run(controller.request_stop("budget"))
    controller.policy.api_client.request_stop.assert_called_once_with("budget")


@pytest.mark.parametrize("stop_step", [0, 1])
def test_real_single_sft_loop_stops_at_actual_step(monkeypatch, stop_step):
    from cosmos_rl.policy.worker.sft_worker import SFTPolicyWorker

    w = worker()
    w.train_step = 0
    w.start_epoch = 0
    w.epoch = 1
    w.enable_dp_load_balancing = False
    w.parallel_dims = SimpleNamespace(pp_enabled=False)
    w.profiler = Mock()
    w.config = SimpleNamespace(
        profiler=SimpleNamespace(enable_nsys=False), logging=SimpleNamespace(logger=[])
    )
    w.total_steps = 100
    w._save_freq = 10
    w.train_sampler = None
    w.train_batch_sampler = None
    w.train_data_loader = []
    w.get_batch_from_dataloader = lambda _: [[0], [1], [2]]
    w.validate = Mock(return_value=None)
    w.signal_handler = None
    w.trainer.step_training = Mock(return_value={})
    w.trainer.checkpointing = Mock()
    w.custom_logger_fns = []

    def boundary(name, step, *, checkpoint_complete=False):
        if checkpoint_complete:
            return {"complete": True}
        return {
            "stop": step >= stop_step,
            "reason": "budget" if step >= stop_step else None,
            "step": step,
        }

    w.api_client.training_boundary.side_effect = boundary
    monkeypatch.setattr(
        stop.dist_util, "broadcast_object_cpu", lambda value, **_: value
    )
    monkeypatch.setattr(
        stop.dist_util, "all_reduce_tensor_object_cpu", lambda value, **_: value
    )
    monkeypatch.setattr(torch.cuda, "Event", Mock(return_value=Mock()))
    monkeypatch.setattr(
        "cosmos_rl.policy.worker.sft_worker.util.is_master_rank", lambda *_: True
    )
    SFTPolicyWorker.main_loop(w)
    assert w.trainer.step_training.call_count == stop_step
    assert w.train_step == stop_step
    assert w.trainer.checkpointing.call_args.kwargs["train_step"] == stop_step
    assert w.trainer.checkpointing.call_args.kwargs["total_steps"] == 100
    assert w.trainer.checkpointing.call_args.kwargs["is_last_step"]
    assert not any(
        call.kwargs.get("is_last_step") for call in w.validate.call_args_list
    )


@pytest.mark.parametrize("queued_sync", [False, True])
def test_colocated_wait_for_rollout_observes_policy_terminal_command(
    monkeypatch, queued_sync
):
    from cosmos_rl.colocated.controller import ColocatedController
    from cosmos_rl.colocated.utils import CommandDispatcher
    from cosmos_rl.dispatcher.command import TrainingCompleteCommand

    controller = object.__new__(ColocatedController)
    command = TrainingCompleteCommand(
        replica_name="p", global_step=2, total_steps=2, remain_samples_num=8
    )
    controller.policy = Mock(replica_name="p", global_rank=0)
    controller.policy.subscribe_remote_commands.return_value = [command]
    controller.rollout = Mock(replica_name="r")
    controller.remote_command_manager = CommandDispatcher(["p", "r"])
    if queued_sync:
        from cosmos_rl.dispatcher.command import PolicyToRolloutUnicastCommand

        controller.remote_command_manager.publish_command(
            PolicyToRolloutUnicastCommand("p", "r", 1, 1).pack(), "p"
        )
    monkeypatch.setattr(
        stop.dist_util, "broadcast_object_cpu", lambda value, **_: value
    )
    assert isinstance(
        controller.wait_for_remote_command(controller.rollout), TrainingCompleteCommand
    )
    controller.rollout.subscribe_remote_commands.assert_not_called()


@pytest.mark.parametrize("stop_step", [0, 1])
@pytest.mark.parametrize("master", [True, False])
def test_real_multi_sft_loop_stops_before_next_prompt(monkeypatch, stop_step, master):
    from queue import Queue
    from cosmos_rl.policy.worker import multi_replica_sft_worker as module

    w = worker()
    w.train_step = None
    w.loaded_train_step = None
    w.weight_sync_done = True
    w.command_buffer = Queue()
    w.data_queue = Queue()
    w.broadcast_command = Mock()
    w.parallel_dims = SimpleNamespace(pp_enabled=False)
    w.profiler = Mock()
    w.config = SimpleNamespace(
        profiler=SimpleNamespace(enable_nsys=False),
        train=SimpleNamespace(
            train_batch_per_replica=1,
            ckpt=SimpleNamespace(save_freq_in_epoch=0, save_freq=10),
        ),
    )
    w.dp_world_size = 1
    w.total_steps = 100
    w._save_freq = 10
    w.do_save = False
    w.is_master_replica = master
    w.validate = Mock(return_value=None)
    w.trainer.step_training = Mock(return_value={})
    w.trainer.checkpointing = Mock()
    w.inter_policy_nccl = Mock()
    w.train_stream = Mock()
    w.handle_shutdown = Mock()

    def prompts(**kwargs):
        w.train_step = w.train_step or 0
        w.data_queue.put("sample")
        return False, False

    w.request_new_prompts = Mock(side_effect=prompts)

    def boundary(name, step, *, checkpoint_complete=False):
        if checkpoint_complete:
            return {"complete": True}
        return {
            "stop": step >= stop_step,
            "reason": "budget" if step >= stop_step else None,
        }

    w.api_client.training_boundary.side_effect = boundary
    monkeypatch.setattr(module.threading, "Thread", Mock())
    monkeypatch.setattr(torch.cuda, "Event", Mock(return_value=Mock()))
    monkeypatch.setattr(module.util, "is_master_rank", lambda *_: True)
    monkeypatch.setattr(
        stop.dist_util, "broadcast_object_cpu", lambda value, **_: value
    )
    monkeypatch.setattr(
        stop.dist_util, "all_reduce_tensor_object_cpu", lambda value, **_: value
    )
    module.MultiReplicaSFTPolicyWorker.main_loop(w)
    assert w.request_new_prompts.call_count == stop_step
    assert w.trainer.step_training.call_count == stop_step
    assert w.train_step == stop_step
    assert w.trainer.checkpointing.call_count == int(master)
    w.api_client.training_boundary.assert_called_with(
        "p0", stop_step, checkpoint_complete=True
    )
    w.handle_shutdown.assert_called_once()
