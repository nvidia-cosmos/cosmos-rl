"""Initial weight binding must retain the resumed policy's version."""

from types import SimpleNamespace
from threading import Event
from unittest.mock import Mock, patch


def test_rollout_registration_carries_resume_step():
    from cosmos_rl.dispatcher.status import RolloutStatusManager

    policy = SimpleNamespace(start_time=0, weights_loaded_in_view_of_command=True)

    class Policies:
        current_step = 8
        total_steps = 16
        policy_atoms_in_replica = 8

        def __iter__(self):
            return iter([policy])

    replica = SimpleNamespace(name="rollout", weights_loaded_in_view_of_command=False)
    manager = SimpleNamespace(
        rollout_atoms_in_replica=8,
        redis_handler=Mock(),
        rollout_init_done=True,
        trigger_rebuild_mesh=Mock(),
    )
    with patch(
        "cosmos_rl.dispatcher.status.command.PolicyToRolloutUnicastCommand.trigger"
    ) as emit:
        RolloutStatusManager.post_register_hook(
            manager, [replica], replica, SimpleNamespace(), Policies()
        )
    assert emit.call_args.kwargs["weight_step"] == 8
    assert emit.call_args.kwargs["total_steps"] == 16


def test_colocated_binding_sets_version_before_ready():
    from cosmos_rl.rollout.worker.colocated.rollout_control import (
        ColocatedRolloutControlWorker,
    )

    for target, step, expected in [
        ("rollout", 8, 8),
        ("other", 8, 0),
        ("rollout", None, 0),
    ]:
        worker = SimpleNamespace(
            replica_name="rollout",
            current_weight_version=0,
            lazy_initialize_rollout_engine=Mock(),
            rollout=Mock(),
            api_client=Mock(),
            state=Mock(),
        )
        worker.state.set_weight_synced.side_effect = lambda: (
            assert_version(worker.current_weight_version, expected)
        )
        command = SimpleNamespace(
            dst_replica_name=target, src_replica_name="policy", weight_step=step
        )
        ColocatedRolloutControlWorker.policy_to_rollout_unicast(worker, command)
        assert worker.current_weight_version == expected
        assert worker.state.set_weight_synced.call_count == int(target == "rollout")


def assert_version(actual, expected):
    assert actual == expected


def test_resume_broadcast_honors_initial_validation_gate():
    from cosmos_rl.rollout.worker.colocated.rollout_control import (
        ColocatedRolloutControlWorker,
    )

    for before_train in (False, True):
        flag = Event()
        worker = SimpleNamespace(
            replica_name="rollout",
            current_weight_version=8,
            config=SimpleNamespace(
                validation=SimpleNamespace(
                    enable=True, val_before_train=before_train, freq=2
                )
            ),
            validation_flag=flag,
            do_validation=Mock(side_effect=flag.clear),
        )
        command = SimpleNamespace(
            src_replica_name="rollout",
            weight_step=8,
            total_steps=16,
            replica_should_stop=lambda: False,
        )
        ColocatedRolloutControlWorker.broadcast_to_all_rollout_replica(worker, command)
        assert worker.do_validation.call_count == int(before_train)
        command.weight_step = 10
        ColocatedRolloutControlWorker.broadcast_to_all_rollout_replica(worker, command)
        assert worker.do_validation.call_count == int(before_train) + 1
