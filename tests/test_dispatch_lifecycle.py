# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from cosmos_rl.dispatcher import run_web_panel
from cosmos_rl.dispatcher.protocol import MESH_NAMES, Role
from cosmos_rl.dispatcher.replica import Atom, Replica
from cosmos_rl.dispatcher.status import (
    PolicyStatus,
    PolicyStatusManager,
    RolloutStatusManager,
)
from test_dispatch_accounting import dispatched_manager


def atom(name="replica", *, host="localhost"):
    return Atom(
        global_rank=0,
        host_ip="127.0.0.1",
        host_name=host,
        trace_path="",
        ranks=[0] * len(MESH_NAMES),
        group_size=[1] * len(MESH_NAMES),
        replica_name=name,
    )


@pytest.mark.parametrize("role", [Role.POLICY, Role.ROLLOUT])
def test_lost_registration_reply_does_not_repeat_lifecycle_commands(role):
    manager = PolicyStatusManager() if role == Role.POLICY else RolloutStatusManager()
    replica = Replica("replica", role, [atom()])
    if role == Role.POLICY:
        manager.policy_replicas[replica.name] = replica
        manager.status[replica.name] = PolicyStatus.RUNNING
    else:
        manager.rollout_replicas[replica.name] = replica
    manager.post_register_hook = Mock()
    retry = atom()
    assert manager.register(retry, Mock(), Mock()) is replica
    assert retry.replica is replica
    manager.post_register_hook.assert_not_called()
    assert len(replica.atoms) == 1
    if role == Role.POLICY:
        assert manager.status[replica.name] == PolicyStatus.RUNNING
    with pytest.raises(ValueError, match="changed identity"):
        manager.register(atom(host="different-process-host"), Mock(), Mock())
    assert len(replica.atoms) == 1


def test_settlement_rejects_underflow_before_metrics_or_dedup_mutation():
    manager = PolicyStatusManager()
    manager.samples_on_the_fly = 2
    hook = Mock()
    manager.set_discard_refill_hook(hook)
    with pytest.raises(ValueError, match="reservations"):
        manager.settle_discarded_samples("replica", "report", 3, 5)
    assert manager.samples_on_the_fly == 2 and manager.filter_records == {}
    assert not manager._applied_discard_report_ids["replica"]
    hook.assert_not_called()
    # A rejected report has not spent its identity.
    assert manager.settle_discarded_samples("replica", "report", 2, 5) == 2


def test_training_ack_underflow_is_rejected_before_status_or_dedup_mutation():
    manager, _ = dispatched_manager()
    manager.samples_on_the_fly = 1
    with pytest.raises(ValueError, match="reservations"):
        manager.train_ack("policy-0", 1, 2, False, {}, Mock())
    assert manager.samples_on_the_fly == 1
    assert manager.status["policy-0"] == PolicyStatus.RUNNING
    assert manager.training_dispatches[1].report_digests == {}


def test_scaling_exemption_expires_at_completed_step():
    manager, _ = dispatched_manager()
    manager.replica_scaling_log = [object()]
    rollouts = SimpleNamespace(replica_scaling_log=[object()])
    manager.train_ack("policy-0", 1, 2, False, {}, rollouts)
    assert manager.replica_scaling_log == rollouts.replica_scaling_log == []


def test_last_policy_loss_is_explicit_terminal_not_rebootstrap():
    manager, replica = dispatched_manager()
    replica.in_mesh = True
    manager.get_all_atoms_arrived_replicas = lambda: list(
        manager.policy_replicas.values()
    )
    with pytest.raises(RuntimeError, match="scale-to-zero"):
        manager.unregister(replica.name)
    assert manager.terminal_error is not None
    with pytest.raises(RuntimeError, match="scale-to-zero"):
        manager.register(atom(), Mock(), Mock())


def test_monitor_failure_is_observed_and_terminates_execution(monkeypatch):
    exit_process = Mock()
    monkeypatch.setattr(run_web_panel.os, "_exit", exit_process)

    async def scenario():
        future = asyncio.get_running_loop().create_future()
        future.set_exception(RuntimeError("injected monitor failure"))
        run_web_panel._observe_controller_monitor(future)
        exit_process.assert_called_once_with(86)
        future = asyncio.get_running_loop().create_future()
        future.cancel()
        run_web_panel._observe_controller_monitor(future)
        exit_process.assert_called_once_with(86)

    asyncio.run(scenario())


def test_monitor_mutates_on_request_event_loop_and_shutdown_cancels_it(monkeypatch):
    owners = []

    async def scenario():
        observed = asyncio.Event()

        def reap():
            owners.append((threading.get_ident(), asyncio.get_running_loop()))
            observed.set()

        # Use a concrete length-bearing wrapper, not MagicMock's default zero.
        class Policies:
            terminal_error = None
            redis_handler = SimpleNamespace(_publication_failure=None)
            maintain_life_status = staticmethod(reap)
            training_finished = staticmethod(lambda: False)

            def __len__(self):
                return 1

        controller = SimpleNamespace(
            policy_status_manager=Policies(),
            life_cycle_lock=asyncio.Lock(),
            rollout_status_manager=Mock(),
            config=SimpleNamespace(validation=SimpleNamespace(enable=False)),
        )
        monkeypatch.setattr(run_web_panel, "controller", controller)
        monkeypatch.setattr(run_web_panel, "_maybe_finalize", lambda reason: False)
        monkeypatch.setattr(
            run_web_panel, "should_broadcast_stop", lambda **kwargs: False
        )
        monkeypatch.setattr(
            run_web_panel, "COSMOS_SHUTDOWN_ON_NO_POLICY_REPLICAS", False
        )
        async with run_web_panel.lifespan(run_web_panel.app):
            await asyncio.wait_for(observed.wait(), timeout=2)
        assert owners == [(threading.get_ident(), asyncio.get_running_loop())]

    asyncio.run(scenario())


def test_initial_validation_is_not_restarted_by_late_join():
    manager = PolicyStatusManager()
    manager.policy_init_done = False
    manager.config = SimpleNamespace(mode="disaggregated")
    manager.data_fetcher = SimpleNamespace(validation_activate_dataloader=Mock())
    manager.trigger_rebuild_mesh = Mock()
    manager.redis_handler = Mock()
    manager.status = {"a": PolicyStatus.UNINITIALIZED, "b": PolicyStatus.UNINITIALIZED}
    config = SimpleNamespace(
        validation=SimpleNamespace(enable=True, val_before_train=True),
        policy=SimpleNamespace(parallelism=SimpleNamespace(n_init_replicas=1)),
    )
    first = SimpleNamespace(
        name="a", start_time=0, weights_loaded_in_view_of_command=True
    )
    late = SimpleNamespace(
        name="b",
        start_time=1,
        weights_loaded_in_view_of_command=False,
        status=SimpleNamespace(mesh_rank=-1),
    )
    manager.post_register_hook([first], first, config, Mock())
    with patch("cosmos_rl.dispatcher.command.PolicyToPolicyUnicastCommand.trigger"):
        manager.post_register_hook([first, late], late, config, Mock())
    manager.data_fetcher.validation_activate_dataloader.assert_called_once_with(0)
