# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cosmos_rl.utils.payload_transport.lifecycle import TransportClose


def test_close_is_idempotent():
    cleanup = Mock()
    operation = TransportClose(cleanup)
    operation.close()
    operation.close()
    assert operation.completed
    cleanup.assert_called_once()


def test_timeout_retains_operation_and_does_not_free_live_resources():
    release = threading.Event()
    freed = Mock()

    def cleanup():
        release.wait()
        freed()

    operation = TransportClose(cleanup)
    try:
        with pytest.raises(TimeoutError):
            operation.close(timeout=0.01)
        assert not operation.completed
        freed.assert_not_called()
    finally:
        release.set()
        operation.close()
    freed.assert_called_once()


def test_failure_is_not_retried_or_reported_as_closed():
    error = ValueError("backend failure")
    cleanup = Mock(side_effect=error)
    operation = TransportClose(cleanup)
    for _ in range(2):
        with pytest.raises(RuntimeError) as raised:
            operation.close()
        assert raised.value.__cause__ is error
    assert not operation.completed
    cleanup.assert_called_once()


def test_concurrent_close_has_one_owner():
    cleanup = Mock()
    operation = TransportClose(cleanup)
    callers = [threading.Thread(target=operation.close) for _ in range(8)]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join()
    cleanup.assert_called_once()


def test_close_thread_start_failure_remains_failed(monkeypatch):
    error = RuntimeError("cannot start thread")
    monkeypatch.setattr(threading.Thread, "start", Mock(side_effect=error))
    cleanup = Mock()
    operation = TransportClose(cleanup)
    for _ in range(2):
        with pytest.raises(RuntimeError) as raised:
            operation.close(timeout=0)
        assert raised.value.__cause__ is error
    assert not operation.completed
    cleanup.assert_not_called()


@pytest.mark.parametrize("timeout", [-1, float("inf"), float("nan")])
def test_invalid_close_deadline_does_not_start_cleanup(timeout):
    cleanup = Mock()
    operation = TransportClose(cleanup)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        operation.close(timeout)
    cleanup.assert_not_called()
    operation.close()
    cleanup.assert_called_once()


def test_ucxx_pending_server_retains_listeners_and_buffer():
    from cosmos_rl.utils.payload_transport.ucxx.ucxx_buffer import UCXXBuffer

    buffer = UCXXBuffer.__new__(UCXXBuffer)
    buffer._buffer = Mock()
    buffer._shutdown_flag = threading.Event()
    thread = Mock()
    thread.is_alive.return_value = True
    listener = Mock()
    buffer._server_threads = [thread]
    buffer._listeners = [listener]
    buffer._ports = [1234]
    with pytest.raises(TimeoutError, match="remain active"):
        buffer.stop_server(timeout=0)
    assert buffer._server_threads == [thread]
    assert buffer._listeners == [listener]
    listener.close.assert_not_called()
    buffer._buffer.close.assert_not_called()
    thread.is_alive.return_value = False
    buffer.stop_server()
    listener.close.assert_called_once()


def test_ucxx_failed_join_prevents_context_reset_and_buffer_release(monkeypatch):
    from cosmos_rl.utils.payload_transport.ucxx import mixins

    reset = Mock()
    monkeypatch.setattr(mixins, "reset_ucxx_context", reset)
    producer = mixins.UCXXRolloutMixin()
    producer._ucxx_enabled = True
    buffer = producer._ucxx_buffer = Mock()
    buffer.stop_server.side_effect = TimeoutError("active reader")
    with pytest.raises(TimeoutError):
        producer.cleanup_ucxx()
    assert not producer._ucxx_enabled
    assert producer._ucxx_buffer is buffer
    reset.assert_not_called()
    buffer.close.assert_not_called()


def test_ucxx_partial_start_resets_context_before_freeing_buffer(monkeypatch):
    from cosmos_rl.utils.payload_transport.ucxx import mixins

    order = Mock()
    monkeypatch.setattr(mixins, "reset_ucxx_context", order.reset)
    producer = mixins.UCXXRolloutMixin()
    producer._ucxx_server_start_attempted = True
    producer._ucxx_buffer = order.buffer
    producer.cleanup_ucxx()
    assert [call[0] for call in order.mock_calls] == [
        "buffer.stop_server",
        "reset",
        "buffer.close",
    ]


def test_producer_partial_setup_rolls_back_and_preserves_original_error():
    from cosmos_rl.utils.payload_transport.nccl.mixins import NCCLRolloutMixin

    producer = NCCLRolloutMixin()
    producer._nccl_registry = Mock()
    producer._nccl_comm_cache = Mock()
    failure = ValueError("after registry acquisition")
    producer._setup_nccl = Mock(side_effect=failure)
    with pytest.raises(ValueError) as raised:
        producer.setup_nccl()
    assert raised.value is failure
    producer._nccl_registry.clear.assert_called_once()
    producer._nccl_comm_cache.abort_all.assert_called_once()
    producer.cleanup_nccl()
    producer._nccl_registry.clear.assert_called_once()


def test_producer_does_not_clear_registry_until_executor_has_joined():
    from cosmos_rl.utils.payload_transport.nccl.mixins import NCCLRolloutMixin

    producer = NCCLRolloutMixin()
    release = threading.Event()
    producer._nccl_registry = Mock()
    producer._nccl_executor = Mock()
    producer._nccl_executor.shutdown.side_effect = lambda **kwargs: release.wait()
    try:
        with pytest.raises(TimeoutError):
            producer.cleanup_nccl(timeout=0.01)
        producer._nccl_registry.clear.assert_not_called()
        with pytest.raises(RuntimeError, match="still closing"):
            producer.setup_nccl()
    finally:
        release.set()
        producer.cleanup_nccl()
    producer._nccl_executor.shutdown.assert_called_once_with(
        wait=True, cancel_futures=True
    )
    producer._nccl_registry.clear.assert_called_once()


def test_subscription_failure_is_observed_and_pubsub_closed():
    from cosmos_rl.utils.payload_transport.nccl.mixins import NCCLRolloutMixin

    producer = NCCLRolloutMixin()
    producer._nccl_redis = Mock()
    pubsub = producer._nccl_redis.pubsub.return_value
    pubsub.subscribe.side_effect = ValueError("subscribe failed")
    with pytest.raises(ValueError, match="subscribe failed"):
        producer._start_listener(channel="test", handler=Mock(), name="test")
    pubsub.close.assert_called_once()


def test_prefetch_close_retains_strategy_until_worker_has_exited():
    from cosmos_rl.utils.payload_transport.prefetch_mixin import PrefetchDataPackerMixin

    packer = PrefetchDataPackerMixin()
    release = threading.Event()
    strategy = Mock()
    packer.set_transport_strategy(strategy)
    packer._prefetch_thread = threading.Thread(target=release.wait)
    packer._prefetch_thread.start()
    try:
        with pytest.raises(TimeoutError):
            packer.close_transport(timeout=0.01)
        strategy.shutdown.assert_not_called()
        with pytest.raises(RuntimeError, match="still closing"):
            packer.set_transport_strategy(Mock())
    finally:
        release.set()
        packer.close_transport()
    strategy.before_join.assert_called_once()
    strategy.shutdown.assert_called_once()
    replacement = Mock()
    packer.set_transport_strategy(replacement)
    assert packer._transport_strategy is replacement
    packer.close_transport()


@pytest.mark.parametrize("stage", ["setup", "prefetch"])
def test_composed_nccl_attachment_rolls_back_each_stage(stage, monkeypatch):
    from cosmos_rl.utils.payload_transport.prefetch_mixin import PrefetchDataPackerMixin
    from cosmos_rl.utils.payload_transport.nccl import strategy as nccl

    packer = PrefetchDataPackerMixin()
    strategy = Mock()
    monkeypatch.setattr(nccl, "NCCLTransportStrategy", lambda: strategy)
    failure = ValueError(stage)
    if stage == "setup":
        strategy.setup.side_effect = failure
    else:
        packer._setup_prefetch = Mock(side_effect=failure)
    with pytest.raises(ValueError) as raised:
        nccl.compose_nccl_transport(packer, device=None, redis_client=Mock())
    assert raised.value is failure
    strategy.before_join.assert_called_once()
    strategy.shutdown.assert_called_once()
    assert packer._transport_strategy is None


def test_worker_rolls_back_first_and_partial_second_packer(monkeypatch):
    from cosmos_rl.comm.base import CommMixin
    from cosmos_rl.utils.payload_transport.registry import PayloadTransportRegistry

    worker = object.__new__(CommMixin)
    worker.config = SimpleNamespace(custom={"payload_transfer": "redis"})
    worker.role = "policy"
    worker.data_packer = SimpleNamespace()
    worker.val_data_packer = SimpleNamespace()
    worker._build_redis_endpoint = Mock(return_value=None)
    worker._opportunistic_inject_redis = Mock()
    transport = Mock()
    failure = ValueError("second packer failed")
    transport.attach_data_packer.side_effect = [None, failure]
    monkeypatch.setattr(
        PayloadTransportRegistry, "get_optional", lambda mode: transport
    )
    with pytest.raises(ValueError) as raised:
        worker._attach_payload_transport()
    assert raised.value is failure
    assert [call.args[0] for call in transport.close_data_packer.call_args_list] == [
        worker.val_data_packer,
        worker.data_packer,
    ]
    assert not worker._payload_transport_attachments


def test_worker_keeps_failed_close_owned_for_retry():
    from cosmos_rl.comm.base import CommMixin

    worker = object.__new__(CommMixin)
    transport = Mock()
    packer = object()
    worker._payload_transport_attachments = [(transport, packer)]
    transport.close_data_packer.side_effect = [TimeoutError("busy"), None]
    with pytest.raises(RuntimeError, match="shutdown failed"):
        worker.close_payload_transports()
    assert len(worker._payload_transport_attachments) == 1
    worker.close_payload_transports()
    assert not worker._payload_transport_attachments


def test_timed_out_gpu_event_keeps_buffer_owned():
    from cosmos_rl.utils.payload_transport.nccl.mixins import NCCLRolloutMixin

    producer = NCCLRolloutMixin()
    producer._nccl_send_timeout_ms = 0
    buffer = object()
    event = Mock()
    event.query.return_value = False
    entry = SimpleNamespace(
        transfer_id="pending",
        inflight=0,
        done_events=[event],
        ready_event=None,
        buffer=buffer,
    )
    producer._on_buffer_free(entry)
    assert entry.buffer is buffer
    assert producer._nccl_retained_entries["pending"] is entry
    event.query.return_value = True
    producer._on_buffer_free(entry)
    assert entry.buffer is None
    assert not producer._nccl_retained_entries


@pytest.mark.parametrize(
    "stage",
    [
        "SendBufferRegistry",
        "CommCache",
        "NcclRendezvous",
        "get_transfer_stream_pool",
        "ThreadPoolExecutor",
        "first_listener",
        "second_listener",
    ],
)
def test_producer_rollback_at_each_acquisition_stage(stage, monkeypatch):
    from cosmos_rl.utils.payload_transport.nccl import mixins

    factories = {}
    for name in (
        "SendBufferRegistry",
        "CommCache",
        "NcclRendezvous",
        "get_transfer_stream_pool",
        "ThreadPoolExecutor",
    ):
        factories[name] = Mock()
        monkeypatch.setattr(mixins, name, factories[name])
    monkeypatch.setattr(mixins, "_resolve_prefix", lambda config: "test")
    monkeypatch.setattr(mixins, "_resolve_max_live_comms", lambda *a, **k: 4)
    producer = mixins.NCCLRolloutMixin()
    producer._start_listener = Mock()
    failure = ValueError(stage)
    if stage == "first_listener":
        producer._start_listener.side_effect = failure
    elif stage == "second_listener":
        producer._start_listener.side_effect = [None, failure]
    else:
        factories[stage].side_effect = failure
    with pytest.raises(ValueError) as raised:
        producer.setup_nccl(
            replica_id="test",
            rollout_idx=0,
            redis_client=Mock(),
            config=SimpleNamespace(custom={}),
            sender_rank=0,
        )
    assert raised.value is failure
    assert producer._nccl_close_operation.completed
    if producer._nccl_registry is not None:
        producer._nccl_registry.clear.assert_called_once()
    if getattr(producer, "_nccl_executor", None) is not None:
        producer._nccl_executor.shutdown.assert_called_once_with(
            wait=True, cancel_futures=True
        )
