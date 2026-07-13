# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import threading
import time
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cosmos_rl.colocated.api_client import ColocatedAPIClient
from cosmos_rl.dispatcher.api.client import APIClient
from cosmos_rl.dispatcher.command import (
    BuildMeshCommand,
    Command,
    RolloutToRolloutBroadcastCommand,
    StopCommand,
)
from cosmos_rl.dispatcher.protocol import RolloutRequest
from cosmos_rl.rollout.worker.rollout_control import DisaggregatedRolloutControlWorker
from cosmos_rl.rollout.worker import weight_sync
from cosmos_rl.rollout.worker.weight_sync import WeightSyncThread


def test_rollout_end_request_supports_ranked_and_legacy_reporters():
    ranked = RolloutRequest(
        src_replica_name="rollout-0",
        src_global_rank=7,
        payloads=[],
        is_end=True,
    )
    legacy = RolloutRequest(
        src_replica_name="rollout-0",
        payloads=[],
        is_end=True,
    )

    assert ranked.src_global_rank == 7
    assert legacy.src_global_rank is None


def test_http_rollout_end_reports_controller_acknowledgement():
    client = object.__new__(APIClient)
    client.max_retries = 1
    client.get_alternative_urls = lambda _suffix: ["http://controller/rollout"]
    request = RolloutRequest(
        src_replica_name="rollout-0",
        src_global_rank=7,
        payloads=[],
        is_end=True,
    )

    with patch(
        "cosmos_rl.dispatcher.api.client.make_request_with_retry"
    ) as make_request:
        acknowledged = client.post_rollout_completion(request)

    assert acknowledged is True
    assert make_request.call_count == 1


def test_http_rollout_end_reports_failed_delivery():
    client = object.__new__(APIClient)
    client.max_retries = 1
    client.get_alternative_urls = lambda _suffix: ["http://controller/rollout"]
    request = RolloutRequest(
        src_replica_name="rollout-0",
        src_global_rank=7,
        payloads=[],
        is_end=True,
    )

    with patch(
        "cosmos_rl.dispatcher.api.client.make_request_with_retry",
        side_effect=RuntimeError("controller unavailable"),
    ):
        acknowledged = client.post_rollout_completion(request)

    assert acknowledged is False


def test_colocated_rollout_end_reports_synchronous_acknowledgement():
    client = object.__new__(ColocatedAPIClient)
    client.controller = MagicMock()
    request = RolloutRequest(
        src_replica_name="rollout-0",
        src_global_rank=7,
        payloads=[],
        is_end=True,
    )

    acknowledged = client.post_rollout_completion(request)

    assert acknowledged is True
    client.controller.put_rollouts.assert_called_once_with(request)


def _rollout_end_worker(
    *,
    acknowledged: bool | list[bool],
    mode: str = "disaggregated",
    world_size: int = 1,
):
    calls = []
    requests = []
    acknowledgements = iter(acknowledged) if isinstance(acknowledged, list) else None

    def report_rollouts(*, block):
        calls.append(("flush", block))
        return None, False, None, True

    def post_rollout_completion(request):
        calls.append(("post", request.src_global_rank))
        requests.append(request)
        if acknowledgements is not None:
            return next(acknowledgements)
        return acknowledged

    worker = SimpleNamespace(
        api_client=SimpleNamespace(
            post_rollout_completion=post_rollout_completion,
        ),
        config=SimpleNamespace(mode=mode),
        global_rank=7,
        parallel_dims=SimpleNamespace(world_size=world_size),
        replica_name="rollout-0",
        report_rollouts=report_rollouts,
        should_report=True,
        shutdown_signal=threading.Event(),
        _rollout_end_acknowledged=False,
    )
    return worker, calls, requests


def test_ranked_rollout_end_flushes_once_and_waits_for_stop_after_ack():
    worker, calls, requests = _rollout_end_worker(acknowledged=True)

    first = DisaggregatedRolloutControlWorker.send_end_signal(worker)
    second = DisaggregatedRolloutControlWorker.send_end_signal(worker)

    assert first is True
    assert second is True
    assert calls == [("flush", True), ("post", 7)]
    assert requests[0].src_replica_name == "rollout-0"
    assert requests[0].src_global_rank == 7
    assert requests[0].is_end is True
    assert not worker.shutdown_signal.is_set()


def test_failed_disaggregated_end_preserves_single_process_escape():
    worker, _, _ = _rollout_end_worker(acknowledged=False)

    acknowledged = DisaggregatedRolloutControlWorker.send_end_signal(worker)

    assert acknowledged is False
    assert worker.shutdown_signal.is_set()


def test_failed_multirank_end_retries_until_acknowledged():
    worker, calls, requests = _rollout_end_worker(
        acknowledged=[False, True],
        world_size=2,
    )

    first = DisaggregatedRolloutControlWorker.send_end_signal(worker)
    assert first is False
    assert worker._rollout_end_acknowledged is False
    assert not worker.shutdown_signal.is_set()

    second = DisaggregatedRolloutControlWorker.send_end_signal(worker)
    third = DisaggregatedRolloutControlWorker.send_end_signal(worker)

    assert second is True
    assert third is True
    assert worker._rollout_end_acknowledged is True
    assert calls == [
        ("flush", True),
        ("post", 7),
        ("flush", True),
        ("post", 7),
    ]
    assert len(requests) == 2
    assert not worker.shutdown_signal.is_set()


def test_colocated_end_preserves_local_exit_after_ack():
    worker, _, _ = _rollout_end_worker(acknowledged=True, mode="colocated")

    acknowledged = DisaggregatedRolloutControlWorker.send_end_signal(worker)

    assert acknowledged is True
    assert worker.shutdown_signal.is_set()


def test_drained_nonreporting_rank_accepts_one_member_mesh_rebuild():
    worker = SimpleNamespace(
        state=SimpleNamespace(prompt_consume_end=lambda: True),
        parallel_dims=SimpleNamespace(world_size=2),
        replica_name="rollout-0",
    )
    command = SimpleNamespace(replica_name_to_rank={"rollout-0": 0})

    DisaggregatedRolloutControlWorker.build_global_mesh(worker, command)

    assert worker.rank_in_rollout_repicas == 0
    assert worker.replica_name_to_rank == {"rollout-0": 0}


def test_mesh_rebuild_fences_weight_sync_and_releases_command_router():
    mesh_ready = threading.Event()
    wst = SimpleNamespace(fence=MagicMock(return_value=True))
    worker = SimpleNamespace(
        state=SimpleNamespace(prompt_consume_end=lambda: False),
        parallel_dims=SimpleNamespace(world_size=1),
        replica_name="rollout-0",
        _weight_sync_thread=wst,
        _mesh_rebuild_ready=mesh_ready,
    )
    command = BuildMeshCommand({"rollout-0": 0})

    DisaggregatedRolloutControlWorker.build_global_mesh(worker, command)

    wst.fence.assert_called_once_with()
    assert mesh_ready.is_set()


def test_command_router_does_not_overtake_mesh_rebuild():
    mesh = BuildMeshCommand({"rollout-0": 0})
    r2r = RolloutToRolloutBroadcastCommand(
        "rollout-0",
        ["rollout-0"],
        weight_step=1,
        total_steps=2,
        trainable_only=False,
    )
    shutdown_signal = threading.Event()
    wst = SimpleNamespace(
        enqueue_p2r=MagicMock(),
        enqueue_r2r=MagicMock(side_effect=lambda _command: shutdown_signal.set()),
    )
    command_queue = Queue()
    worker = SimpleNamespace(
        replica_name="rollout-0",
        shutdown_signal=shutdown_signal,
        redis_controller=SimpleNamespace(
            subscribe_command=MagicMock(return_value=[b"mesh", b"r2r"])
        ),
        _command_queue=command_queue,
        _weight_sync_thread=wst,
    )

    with patch.object(Command, "depack", side_effect=[mesh, r2r]):
        router = threading.Thread(
            target=DisaggregatedRolloutControlWorker.query_command_from_controller,
            args=(worker,),
        )
        router.start()
        assert command_queue.get(timeout=1) is mesh
        wst.enqueue_r2r.assert_not_called()
        worker._mesh_rebuild_ready.set()
        router.join(timeout=1)

    assert not router.is_alive()
    wst.enqueue_r2r.assert_called_once_with(r2r)


class _JoinQueue:
    def __init__(self, join):
        self.join = join


def _weight_sync_thread(queue):
    wst = object.__new__(WeightSyncThread)
    wst._queue = queue
    wst._stream = object()
    wst._worker = SimpleNamespace(replica_name="rollout-0")
    return wst


def test_weight_sync_fence_waits_for_queue_before_cuda_stream():
    order = []

    def join_queue():
        order.append("queue")

    def drain_stream(_stream, _timeout, _context):
        order.append("stream")
        return True

    wst = _weight_sync_thread(_JoinQueue(join_queue))
    with patch.object(
        weight_sync,
        "bounded_drain_or_abort",
        side_effect=drain_stream,
    ):
        fenced = wst.fence(queue_timeout=0.1, stream_timeout=0.1)

    assert fenced is True
    assert order == ["queue", "stream"]


def test_weight_sync_queue_timeout_aborts_and_reports_failure():
    order = []

    def slow_join_queue():
        order.append("queue")
        time.sleep(0.1)

    def abort_nccl():
        order.append("abort")

    def drain_stream(_stream, _timeout, _context):
        order.append("stream")
        return True

    wst = _weight_sync_thread(_JoinQueue(slow_join_queue))
    with (
        patch.object(weight_sync, "nccl_abort_all", side_effect=abort_nccl),
        patch.object(
            weight_sync,
            "bounded_drain_or_abort",
            side_effect=drain_stream,
        ),
    ):
        fenced = wst.fence(queue_timeout=0.01, stream_timeout=0.1)

    assert fenced is False
    assert order == ["queue", "abort", "stream"]


def test_weight_sync_task_failure_makes_fence_fail():
    wst = _weight_sync_thread(_JoinQueue(lambda: None))
    wst._task_failed = True
    with patch.object(
        weight_sync,
        "bounded_drain_or_abort",
        return_value=True,
    ):
        assert wst.fence(queue_timeout=0.1, stream_timeout=0.1) is False


def test_weight_sync_stop_always_performs_post_stop_stream_drain():
    wst = object.__new__(WeightSyncThread)
    wst.fence = MagicMock(return_value=False)
    wst._stop = threading.Event()
    wst._thread = SimpleNamespace(is_alive=lambda: False)
    wst._stream = object()
    wst._worker = SimpleNamespace(replica_name="rollout-0")

    with patch.object(
        weight_sync,
        "bounded_drain_or_abort",
        return_value=True,
    ) as drain:
        assert wst.stop() is False

    drain.assert_called_once()
    assert wst._stop.is_set()


def test_cancelled_r2r_barrier_never_enqueues_nccl_work():
    wst = object.__new__(WeightSyncThread)
    wst._worker = SimpleNamespace(replica_name="rollout-0")
    command = SimpleNamespace(
        weight_step=3,
        dst_replica_names=["rollout-0", "rollout-1"],
    )

    with (
        patch.object(weight_sync, "r2r_barrier", return_value=False),
        patch.object(weight_sync, "do_nccl_broadcast_grouped") as broadcast,
    ):
        wst._execute_r2r(command)

    broadcast.assert_not_called()


def test_one_member_async_r2r_updates_version_without_nccl():
    state = SimpleNamespace(
        weight_synced=lambda: False,
        set_weight_synced=MagicMock(),
    )
    worker = SimpleNamespace(
        replica_name="rollout-0",
        _buffer_version=0,
        current_weight_version=0,
        state=state,
        config=SimpleNamespace(
            validation=SimpleNamespace(
                enable=False,
                val_before_train=False,
                freq=1,
            )
        ),
    )
    wst = object.__new__(WeightSyncThread)
    wst._worker = worker
    wst._stream = object()
    wst._queue = SimpleNamespace(qsize=lambda: 0)
    wst._executed = 0
    command = SimpleNamespace(
        weight_step=3,
        total_steps=5,
        src_replica_name="rollout-0",
        dst_replica_names=["rollout-0"],
        replica_should_stop=lambda: False,
    )

    with (
        patch.object(weight_sync, "do_nccl_broadcast_grouped") as broadcast,
        patch.object(weight_sync.torch.cuda, "Event") as event,
    ):
        wst._execute_r2r(command)

    broadcast.assert_not_called()
    event.return_value.record.assert_called_once_with(wst._stream)
    assert worker.current_weight_version == 3
    state.set_weight_synced.assert_called_once_with()


def test_stop_fences_weight_sync_before_setting_shutdown():
    shutdown_signal = threading.Event()
    shutdown_mp_signal = threading.Event()

    def fence():
        assert not shutdown_signal.is_set()
        assert not shutdown_mp_signal.is_set()
        return True

    wst = SimpleNamespace(fence=MagicMock(side_effect=fence))
    worker = SimpleNamespace(
        _weight_sync_thread=wst,
        replica_name="rollout-0",
        shutdown_signal=shutdown_signal,
        shutdown_mp_signal=shutdown_mp_signal,
    )

    DisaggregatedRolloutControlWorker.handle_stop(
        worker,
        StopCommand("rollout-0"),
    )

    wst.fence.assert_called_once_with()
    assert shutdown_signal.is_set()
    assert shutdown_mp_signal.is_set()
