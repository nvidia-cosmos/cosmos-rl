"""Strict colocated updates cannot consume surplus samples from older weights."""

from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

from cosmos_rl.colocated.controller import ColocatedController
from cosmos_rl.dispatcher.command import DataFetchCommand, TrainingCompleteCommand


def controller(on_policy, step, versions):
    queue = Queue()
    for index, version in enumerate(versions):
        queue.put(SimpleNamespace(weight_version=version, index=index))
    return SimpleNamespace(
        config=SimpleNamespace(
            train=SimpleNamespace(train_policy=SimpleNamespace(on_policy=on_policy))
        ),
        current_step=step,
        rollout=SimpleNamespace(current_weight_version=step),
        policy=SimpleNamespace(data_queue=queue),
        train_report_data={},
    )


def test_strict_queue_discards_old_samples_and_preserves_current_fifo():
    obj = controller(True, 8, [7, 8, 0, 8])
    ColocatedController.advance_iteration(obj)
    assert obj.current_step == 9
    assert [obj.policy.data_queue.get_nowait().index for _ in range(2)] == [1, 3]
    assert obj.policy.data_queue.empty()


def test_async_queue_preserves_surplus_and_empty_ranks_advance():
    obj = controller(False, 8, [7, 0])
    ColocatedController.advance_iteration(obj)
    assert [obj.policy.data_queue.get_nowait().weight_version for _ in range(2)] == [
        7,
        0,
    ]
    empty = controller(True, 8, [])
    ColocatedController.advance_iteration(empty)
    assert empty.current_step == 9


def test_strict_queue_rejects_unsynchronized_worker():
    obj = controller(True, 8, [8])
    obj.rollout.current_weight_version = 0
    try:
        ColocatedController.advance_iteration(obj)
    except AssertionError as error:
        assert "not synchronized" in str(error)
    else:
        raise AssertionError("Accepted unsynchronized resumed weights")
    assert obj.policy.data_queue.qsize() == 1


def test_resume_initializes_local_version_before_sync():
    cmd = object.__new__(DataFetchCommand)
    cmd.global_step = 9
    cmd.total_steps = 16
    obj = SimpleNamespace(
        current_step=0,
        total_steps=100,
        policy_consume_one_step_commands_util_data_fetch=Mock(return_value=cmd),
    )
    observed = []
    obj.rollout_consume_one_step_commands_util_r2r = lambda: observed.append(
        (obj.current_step, obj.total_steps)
    )
    ColocatedController.init_commands(obj)
    assert observed == [(8, 16)]
    assert obj.init_data_fetch_command is cmd


def test_initial_stop_does_not_read_training_step_or_sync_weights():
    cmd = object.__new__(TrainingCompleteCommand)
    obj = SimpleNamespace(
        policy_consume_one_step_commands_util_data_fetch=Mock(return_value=cmd),
        finish_requested_stop=Mock(),
        rollout_consume_one_step_commands_util_r2r=Mock(),
    )
    assert ColocatedController.init_commands(obj) is False
    obj.finish_requested_stop.assert_called_once_with(cmd)
    obj.rollout_consume_one_step_commands_util_r2r.assert_not_called()
