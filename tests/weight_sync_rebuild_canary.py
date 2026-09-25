# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Actual WST queue/CUDA drain boundaries; not native NCCL abort recovery.

One GPU. A controlled receive callback delays a real device write, either before
enqueue (host ownership) or on the CUDA stream. Actual WST enqueue, task handling,
bounded drain/abort and mesh dispatch are exercised without a backend model.
"""

import argparse
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import cosmos_rl
from cosmos_rl.dispatcher.command import BuildMeshCommand
from cosmos_rl.rollout.worker.rollout_control import DisaggregatedRolloutControlWorker
from cosmos_rl.rollout.worker.weight_sync import WeightSyncThread


def exercise(phase):
    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    destination = torch.zeros(4096, device="cuda")
    torch.cuda.synchronize()

    def receive(command, stream):
        with torch.cuda.stream(stream):
            if phase == "host":
                entered.set()
                assert release.wait(10)
            elif phase == "device":
                torch.cuda._sleep(2_000_000_000)
            destination.fill_(command.weight_step)
        completed.set()

    worker = SimpleNamespace(
        device=torch.device("cuda", 0),
        replica_name="rollout-0",
        _execute_p2r_recv=receive,
        _buffer_version=0,
        # Model the ownership state used by the independent adoption path too;
        # never replace its production write context with a no-op in this probe.
        _buffer_lock=threading.Lock(),
        _buffer_write_failed=False,
        _buffer_writing=False,
        _buffer_adopt_event=None,
        _buffer_weight_version=0,
        current_weight_version=1,
        state=SimpleNamespace(prompt_consume_end=lambda: False),
        _mesh_rebuild_ready=threading.Event(),
        global_commnicator_idex=17,
        inference_stream=torch.cuda.current_stream(),
        replica_name_to_rank={"rollout-0": 0, "old-peer": 1},
        get_group_unique_key=lambda _: "new-mesh",
        api_client=Mock(),
    )
    thread = WeightSyncThread(worker)
    worker._weight_sync_thread = thread
    thread.start()
    # The canary changes only timeout values, not drain/abort implementations.
    real_fence = thread.fence
    thread.fence = lambda: real_fence(queue_timeout=0.02, stream_timeout=0.02)
    try:
        thread.enqueue_p2r(SimpleNamespace(weight_step=3))
        if phase == "host":
            assert entered.wait(5)
        else:
            assert completed.wait(5)
        if phase == "healthy":
            thread._queue.join()
            thread._stream.synchronize()
            thread._task_failed = True
            thread._fence_failed = True
        else:
            assert not thread.fence(), "fault injection did not overlap the drain"
        with (
            patch(
                "cosmos_rl.rollout.worker.rollout_control.create_nccl_comm",
                return_value=23,
            ) as create,
            patch(
                "cosmos_rl.rollout.worker.rollout_control.create_nccl_uid",
                return_value=[1, 2, 3],
            ),
        ):
            try:
                DisaggregatedRolloutControlWorker.build_global_mesh(
                    worker, BuildMeshCommand({"rollout-0": 0, "new-peer": 1})
                )
            except RuntimeError as exc:
                assert phase != "healthy" and "quiesce" in str(exc), exc
            else:
                assert phase == "healthy", "unsafe rebuild was permitted"
            if phase == "healthy":
                create.assert_called_once()
                assert worker._mesh_rebuild_ready.is_set()
            else:
                create.assert_not_called()
                assert thread._fence_failed
                assert not worker._mesh_rebuild_ready.is_set()
                assert worker.global_commnicator_idex == 17
                if phase == "host":
                    assert not completed.is_set()
                else:
                    assert not thread._stream.query(), (
                        "delayed CUDA work already finished"
                    )
        release.set()
        assert completed.wait(5)
        thread._queue.join()
        thread._stream.synchronize()
        torch.testing.assert_close(destination, torch.full_like(destination, 3))
        if phase != "healthy":
            assert thread._fence_failed, "late completion silently cleared the failure"
        print(f"WST_REBUILD_PASS phase={phase} parity=True", flush=True)
    finally:
        release.set()
        thread._stop.set()
        thread._thread.join(10)
        assert not thread._thread.is_alive()
        thread._stream.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-package", required=True)
    args = parser.parse_args()
    assert Path(cosmos_rl.__file__).resolve().parent == Path(args.expected_package)
    torch.cuda.set_device(0)
    for phase in ("healthy", "host", "device"):
        exercise(phase)


if __name__ == "__main__":
    main()
