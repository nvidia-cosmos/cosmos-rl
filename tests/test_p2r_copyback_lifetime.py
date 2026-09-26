# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""R-4: actual receive/copy-back closure versus allocation-stream reuse."""

import os
from contextlib import nullcontext
from queue import Queue
from types import SimpleNamespace
import weakref

import pytest
import torch

from cosmos_rl.rollout.worker.rollout_control import DisaggregatedRolloutControlWorker
from cosmos_rl.utils.parallelism_map import (
    WeightSyncInstruction,
    WeightSyncInstructionsGroup,
    WeightSyncInstructionsPerParam,
)


@pytest.mark.parametrize("retain_for_control", [False, True])
def test_receive_temporary_survives_delayed_copyback_on_another_stream(
    retain_for_control,
):
    if not torch.cuda.is_available():
        assert os.environ.get("COSMOS_REQUIRE_CUDA") != "1", "GPU gate requires CUDA"
        pytest.skip("requires CUDA streams and allocator")
    exercise_copyback(retain_for_control)


def exercise_copyback(
    retain_for_control=False,
    *,
    native_recv=None,
    rollout_rank=0,
    receive_scope=nullcontext,
):
    device = torch.device("cuda", torch.cuda.current_device())
    worker = object.__new__(DisaggregatedRolloutControlWorker)
    worker.device = device
    worker.config = SimpleNamespace(train=SimpleNamespace(transfer_dtype="float16"))
    worker.quantization_type = None
    worker.temp_recv_tensor_queue = Queue()
    worker.get_underlying_model = lambda: None
    worker.trainable_params = {"a", "b"}
    worker.parallel_dims = None
    size = 1 << 20
    worker.weight_inplace_view_map = {
        name: torch.zeros(size, dtype=torch.float32, device=device)
        for name in ("a", "b")
    }
    worker.weight_mapper = SimpleNamespace(
        update_tensor_view=lambda view, received, name, **kwargs: view.copy_(received)
    )
    received = []
    retained = []

    def recv(mesh_key, tensor, rank):
        received.append((weakref.ref(tensor), tensor.data_ptr()))
        if retain_for_control:
            retained.append(tensor)
        if native_recv is None:
            tensor.fill_(len(received))
        else:
            native_recv(tensor)

    worker.p2r_collective_manager = SimpleNamespace(recv=recv)
    group = WeightSyncInstructionsGroup(
        [
            WeightSyncInstructionsPerParam(
                name, [WeightSyncInstruction(0, rollout_rank, {})]
            )
            for name in ("a", "b")
        ]
    )
    producer, copier = torch.cuda.Stream(), torch.cuda.Stream()
    torch.cuda.synchronize()
    try:
        with torch.cuda.stream(producer):
            with receive_scope():
                _, complete, _ = worker.recv_weight_shard(
                    rollout_rank, group, "pair", False
                )
            ready = producer.record_event()
            copier.wait_event(ready)
            with torch.cuda.stream(copier):
                torch.cuda._sleep(200_000_000)
                complete()
                copied = copier.record_event()
            del complete
            # The outer loop may release queue ownership after issuing its
            # final wait_event, before the device copy has actually completed.
            # Exercise allocator protection independently of Python owners.
            worker.temp_recv_tensor_queue.queue.clear()
            # A later receive allocates on this stream before the outer P2R
            # loop's final wait_event(copy_finished). No host access to data.
            churn = []
            for _ in range(16):
                tensor = torch.empty(size, dtype=torch.float16, device=device)
                tensor.fill_(9)
                churn.append(tensor)
            reused = any(t.data_ptr() == received[0][1] for t in churn)
            producer.wait_event(copied)
        torch.cuda.synchronize()
        first = worker.weight_inplace_view_map["a"]
        assert torch.all(first == 1), (
            f"copy-back read reused temporary: reused={reused}, "
            f"wrong_values={(first != 1).sum().item()}, max={first.max().item()}, "
            f"retained={received[0][0]() is not None}"
        )
        assert torch.all(worker.weight_inplace_view_map["b"] == 2)
    finally:
        torch.cuda.synchronize()
