# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Two-rank native mesh retirement and delayed-stream payload parity.

Use torchrun on one or two nodes. UID delivery uses Gloo in place of the
controller HTTP endpoint; the worker handler, NCCL handles, broadcast and
retirement are real. This does not simulate native driver failure or recovery.
"""

import argparse
from datetime import timedelta
import os
from pathlib import Path
import threading
from types import SimpleNamespace

import torch
import torch.distributed as dist

import cosmos_rl
from cosmos_rl.dispatcher.command import BuildMeshCommand
from cosmos_rl.rollout.worker.rollout_control import DisaggregatedRolloutControlWorker
from cosmos_rl.utils import pynccl


def broadcast_payload(rank, value, device):
    # Both ranks must issue identical byte counts; an integer sentinel must not
    # infer int64 on the receiver while the source's version value infers float32.
    return torch.full(
        (1 << 20,), value if rank == 0 else -1, dtype=torch.float32, device=device
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-package", required=True)
    args = parser.parse_args()
    assert Path(cosmos_rl.__file__).resolve().parent == Path(args.expected_package)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    assert dist.get_world_size() == 2
    rank = dist.get_rank()
    assert pynccl._COMM_REGISTRY.all_indices() == []

    def publish(key, uid):
        assert rank == 0
        dist.broadcast_object_list([uid], src=0)

    def receive(key):
        assert rank == 1
        data = [None]
        dist.broadcast_object_list(data, src=0)
        return data[0]

    worker = SimpleNamespace(
        state=SimpleNamespace(prompt_consume_end=lambda: False),
        replica_name=f"rollout-{rank}",
        _weight_sync_thread=None,
        _mesh_rebuild_ready=threading.Event(),
        global_commnicator_idex=-1,
        inference_stream=torch.cuda.Stream(),
        get_group_unique_key=lambda mapping: "_".join(sorted(mapping)),
        query_nccl_unique_id_from_controller=receive,
        api_client=SimpleNamespace(post_nccl_comm_initiator=publish),
    )
    held = None
    delayed = None
    previous_value = None
    overlap_seen = False
    try:
        for step, phase in enumerate(
            ("initial", "replace", "unused", "rebuild", "single")
        ):
            old = worker.global_commnicator_idex
            mapping = {"rollout-0": 0, "rollout-1": 1}
            if phase == "single":
                mapping = {"rollout-0": 0}
                if rank == 1:
                    # Model the departing peer's explicit cleanup; only the
                    # survivor receives the singleton rebuild command.
                    worker.inference_stream.synchronize()
                    pynccl.nccl_abort(old)
                    worker.global_commnicator_idex = -1
            dist.barrier()
            if delayed is not None:
                overlap_seen |= not delayed.query()
            if phase != "single" or rank == 0:
                worker._mesh_rebuild_ready.clear()
                DisaggregatedRolloutControlWorker.build_global_mesh(
                    worker, BuildMeshCommand(mapping, mesh_is_used=phase != "unused")
                )
                assert worker._mesh_rebuild_ready.is_set()
            if old >= 0:
                assert not pynccl.nccl_comm_is_registered(old)
            if held is not None:
                assert delayed.query(), "retirement returned ahead of the last reader"
                torch.testing.assert_close(held, torch.full_like(held, previous_value))
            active = phase not in ("unused", "single")
            assert len(pynccl._COMM_REGISTRY.all_indices()) == int(active)
            held = delayed = None
            if active:
                assert pynccl.nccl_comm_is_registered(worker.global_commnicator_idex)
                previous_value = step + 1.0
                with torch.cuda.stream(worker.inference_stream):
                    held = broadcast_payload(rank, previous_value, "cuda")
                    pynccl.nccl_broadcast(held, 0, worker.global_commnicator_idex)
                    torch.cuda._sleep(1_000_000_000)
                    delayed = torch.cuda.Event()
                    delayed.record()
            else:
                assert worker.global_commnicator_idex == -1
            print(
                f"MESH_RETIREMENT_PASS rank={rank} phase={phase} handles={int(active)}",
                flush=True,
            )
        assert overlap_seen, "no delayed device work overlapped mesh retirement"
        print(
            f"MESH_RETIREMENT_COMPLETE rank={rank} parity=True delayed=True", flush=True
        )
    finally:
        worker.inference_stream.synchronize()
        pynccl.nccl_abort_all()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
