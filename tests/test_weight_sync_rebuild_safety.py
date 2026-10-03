# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""A communicator abort is not proof that old weight-sync work has stopped."""

from queue import Queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from cosmos_rl.dispatcher.command import BuildMeshCommand
from cosmos_rl.rollout.worker.rollout_control import DisaggregatedRolloutControlWorker
from cosmos_rl.rollout.worker import weight_sync


def make_thread():
    thread = object.__new__(weight_sync.WeightSyncThread)
    thread._worker = SimpleNamespace(replica_name="rollout-0")
    thread._queue = Queue()
    thread._stream = object()
    thread._seq = 7
    thread._fenced_seq = 6
    thread._fence_failed = True
    thread._task_failed = True
    return thread


@pytest.mark.parametrize("undrained", ["queue", "stream", "new-task-error"])
def test_reset_cannot_erase_unfinished_or_newly_failed_work(undrained):
    thread = make_thread()
    old_task_active = threading.Event()
    release = threading.Event()

    def old_task():
        thread._queue.get()
        old_task_active.set()
        release.wait(5)
        thread._queue.task_done()

    runner = None
    if undrained == "queue":
        thread._queue.put("accepted-old-operation")
        runner = threading.Thread(target=old_task)
        runner.start()
        assert old_task_active.wait(1)

    real_fence = weight_sync.WeightSyncThread.fence

    def short_fence(self):
        return real_fence(self, queue_timeout=0.02, stream_timeout=0.02)

    def device_drain(*args):
        if undrained == "new-task-error":
            thread._task_failed = True
        return undrained != "stream"

    try:
        with (
            patch.object(weight_sync.WeightSyncThread, "fence", short_fence),
            patch.object(
                weight_sync, "bounded_drain_or_abort", side_effect=device_drain
            ),
            patch.object(weight_sync, "nccl_abort_all") as abort,
        ):
            with pytest.raises(RuntimeError, match="quiesce"):
                thread.reset_for_rebuild()
            assert thread._fence_failed
            if undrained == "queue":
                abort.assert_called_once_with()
                assert runner.is_alive(), "abort must not be treated as a thread join"
                assert thread._queue.unfinished_tasks == 1
            # A later caller must still see the failure, without another reset.
            assert not thread.fence()
    finally:
        release.set()
        if runner is not None:
            runner.join(1)
            assert not runner.is_alive()


def test_completed_prior_failure_can_still_rebuild():
    thread = make_thread()
    with (
        patch.object(weight_sync, "bounded_drain_or_abort", return_value=True) as drain,
        patch.object(weight_sync, "nccl_abort_all") as abort,
    ):
        assert thread.reset_for_rebuild()
        drain.assert_called_once()
        abort.assert_not_called()
        assert not thread._fence_failed
        assert not thread._task_failed
        assert thread._fenced_seq == thread._seq


def test_drain_exception_stays_latched_and_preserves_cause():
    thread = make_thread()
    failure = RuntimeError("native drain failed")
    with patch.object(weight_sync, "bounded_drain_or_abort", side_effect=failure):
        with pytest.raises(RuntimeError) as raised:
            thread.reset_for_rebuild()
    assert raised.value is failure
    assert thread._fence_failed
    assert not thread.fence()


def test_mesh_handler_does_not_publish_new_mesh_after_unsafe_reset():
    worker = SimpleNamespace(
        state=SimpleNamespace(prompt_consume_end=lambda: False),
        replica_name="rollout-0",
        _weight_sync_thread=make_thread(),
        _mesh_rebuild_ready=threading.Event(),
        global_commnicator_idex=17,
        replica_name_to_rank={"rollout-0": 0, "old-peer": 1},
        get_group_unique_key=lambda _: "new-mesh",
        api_client=Mock(),
    )
    with (
        patch.object(weight_sync, "bounded_drain_or_abort", return_value=False),
        patch("cosmos_rl.rollout.worker.rollout_control.create_nccl_comm") as create,
        patch("cosmos_rl.rollout.worker.rollout_control.create_nccl_uid") as uid,
    ):
        with pytest.raises(RuntimeError, match="quiesce"):
            DisaggregatedRolloutControlWorker.build_global_mesh(
                worker,
                BuildMeshCommand({"rollout-0": 0, "new-peer": 1}),
            )
        create.assert_not_called()
        uid.assert_not_called()
        assert worker.global_commnicator_idex == 17
        assert worker.replica_name_to_rank == {"rollout-0": 0, "old-peer": 1}
        assert not worker._mesh_rebuild_ready.is_set()
