"""Regression coverage for synchronous simulator engine cleanup."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def control_worker(**kwargs):
    from cosmos_rl.rollout.worker.rollout_control import (
        DisaggregatedRolloutControlWorker,
    )

    worker = object.__new__(DisaggregatedRolloutControlWorker)
    worker.__dict__.update(kwargs)
    worker._cleanup_payload_server = lambda: None
    worker.close_payload_transports = lambda: None
    return worker


def test_vla_shutdown_is_safe_before_initialization_and_idempotent():
    from cosmos_rl.rollout.vla_rollout import OpenVLARollout

    rollout = object.__new__(OpenVLARollout)
    rollout.shutdown()
    manager = Mock()
    rollout.env_manager = manager
    rollout.shutdown()
    rollout.shutdown()
    manager.stop_simulator.assert_called_once_with(preserve_state=False)


def test_sync_worker_shuts_down_engine_before_unregister():
    from cosmos_rl.rollout.worker.rollout_control import (
        DisaggregatedRolloutControlWorker,
    )

    events = []
    engine = Mock()
    engine.shutdown.side_effect = lambda: events.append("engine")
    worker = control_worker(
        shutdown_signal=threading.Event(),
        shutdown_mp_signal=threading.Event(),
        replica_name="test",
        background_thread=None,
        teacher_interact_thread=None,
        scheduler=None,
        heartbeat_thread=None,
        rollout=engine,
        unregister_from_controller=lambda: events.append("unregister"),
    )
    DisaggregatedRolloutControlWorker.handle_shutdown(worker)
    DisaggregatedRolloutControlWorker.handle_shutdown(worker)
    assert events == ["engine", "unregister"]


def test_async_scheduler_quiesces_before_owner_shuts_down_engine():
    from cosmos_rl.rollout.worker.rollout_control import (
        DisaggregatedRolloutControlWorker,
    )

    scheduler, engine = Mock(), Mock()
    events = []
    scheduler.stop.side_effect = lambda **_: events.append("scheduler")
    engine.shutdown.side_effect = lambda: events.append("engine")
    worker = control_worker(
        shutdown_signal=threading.Event(),
        shutdown_mp_signal=threading.Event(),
        replica_name="test",
        background_thread=None,
        teacher_interact_thread=None,
        scheduler=scheduler,
        heartbeat_thread=None,
        rollout=engine,
        unregister_from_controller=Mock(),
    )
    DisaggregatedRolloutControlWorker.handle_shutdown(worker)
    scheduler.stop.assert_called_once_with(wait=True, shutdown_engine=False)
    engine.shutdown.assert_called_once_with()
    assert events == ["scheduler", "engine"]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_engine_shutdown_error_still_joins_heartbeat_and_unregisters(asynchronous):
    from cosmos_rl.rollout.worker.rollout_control import (
        DisaggregatedRolloutControlWorker,
    )

    events = []
    failure = RuntimeError("injected engine shutdown failure")
    engine, scheduler, heartbeat = Mock(), Mock(), Mock()

    def fail_shutdown(*args, **kwargs):
        events.append("shutdown")
        raise failure

    engine.shutdown.side_effect = fail_shutdown
    scheduler.stop.side_effect = fail_shutdown
    heartbeat.join.side_effect = lambda **kwargs: events.append("heartbeat")
    heartbeat.is_alive.return_value = False
    worker = control_worker(
        shutdown_signal=threading.Event(),
        shutdown_mp_signal=threading.Event(),
        replica_name="test",
        background_thread=None,
        teacher_interact_thread=None,
        scheduler=scheduler if asynchronous else None,
        heartbeat_thread=heartbeat,
        rollout=engine,
        unregister_from_controller=lambda: events.append("unregister"),
    )
    with pytest.raises(RuntimeError) as caught:
        DisaggregatedRolloutControlWorker.handle_shutdown(worker)
    assert caught.value.__cause__ is failure
    if asynchronous:
        engine.shutdown.assert_not_called()
    assert events == ["shutdown", "unregister", "heartbeat"]
    heartbeat.join.assert_called_once_with(timeout=15.0)
    with pytest.raises(RuntimeError) as repeated:
        DisaggregatedRolloutControlWorker.handle_shutdown(worker)
    assert repeated.value is caught.value
    assert events == ["shutdown", "unregister", "heartbeat"]


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("unregister_error", [False, True])
def test_cleanup_liveness_and_finally_on_unregister_failure(terminal, unregister_error):
    from cosmos_rl.rollout.worker.rollout_control import (
        DisaggregatedRolloutControlWorker,
    )

    stopped = threading.Event()
    if terminal:
        stopped.set()
    events = []

    def cleanup():
        events.append("cleanup")
        assert stopped.is_set() is terminal

    def unregister():
        events.append("unregister")
        assert stopped.is_set() is terminal
        if unregister_error:
            raise RuntimeError("unregister failed")

    heartbeat = Mock()
    heartbeat.is_alive.return_value = False
    heartbeat.join.side_effect = lambda **_: events.append("joined")
    worker = control_worker(
        shutdown_signal=threading.Event(),
        shutdown_mp_signal=stopped,
        _heartbeat_shutdown_deadline=SimpleNamespace(value=0),
        replica_name="test",
        background_thread=None,
        teacher_interact_thread=None,
        scheduler=None,
        heartbeat_thread=heartbeat,
        rollout=SimpleNamespace(shutdown=cleanup),
        unregister_from_controller=unregister,
    )
    if unregister_error:
        with pytest.raises(RuntimeError, match="unregister") as caught:
            DisaggregatedRolloutControlWorker.handle_shutdown(worker)
        assert str(caught.value.__cause__) == "unregister failed"
    else:
        DisaggregatedRolloutControlWorker.handle_shutdown(worker)
    assert stopped.is_set()
    assert worker._heartbeat_shutdown_deadline.value > 0
    assert events == ["cleanup", "unregister", "joined"]
    if unregister_error:
        with pytest.raises(RuntimeError) as repeated:
            DisaggregatedRolloutControlWorker.handle_shutdown(worker)
        assert repeated.value is caught.value
    else:
        DisaggregatedRolloutControlWorker.handle_shutdown(worker)
    assert events == ["cleanup", "unregister", "joined"]
    heartbeat.join.assert_called_once_with(timeout=15.0)


def test_failed_stop_fence_does_not_keep_advertising_liveness():
    from cosmos_rl.rollout.worker.rollout_control import (
        DisaggregatedRolloutControlWorker,
    )

    worker = SimpleNamespace(
        _weight_sync_thread=SimpleNamespace(fence=lambda: False),
        replica_name="test",
        shutdown_signal=threading.Event(),
        shutdown_mp_signal=threading.Event(),
    )
    DisaggregatedRolloutControlWorker.handle_stop(worker, SimpleNamespace())
    assert worker.shutdown_signal.is_set() and worker.shutdown_mp_signal.is_set()


def test_grace_begins_before_payload_cleanup_and_is_not_renewed():
    from cosmos_rl.rollout.worker.rollout_control import (
        DisaggregatedRolloutControlWorker,
        _begin_shutdown_heartbeat_grace,
    )

    deadline = SimpleNamespace(value=0)

    def cleanup():
        assert deadline.value > 0

    worker = control_worker(
        shutdown_signal=threading.Event(),
        shutdown_mp_signal=threading.Event(),
        unregister_from_controller=Mock(),
        _heartbeat_shutdown_deadline=deadline,
    )
    worker._cleanup_payload_server = cleanup
    DisaggregatedRolloutControlWorker.handle_shutdown(worker)
    original = deadline.value
    DisaggregatedRolloutControlWorker.handle_shutdown(worker)
    _begin_shutdown_heartbeat_grace(worker)
    assert deadline.value == original
