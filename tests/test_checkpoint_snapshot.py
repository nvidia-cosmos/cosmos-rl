# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Snapshot ownership, including a deliberately paused asynchronous writer."""

import copy
import os
import threading
from collections import OrderedDict, namedtuple
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from cosmos_rl.policy.config import Config
from cosmos_rl.utils.checkpoint import CheckpointMananger


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        if os.environ.get("COSMOS_REQUIRE_CUDA") == "1":
            pytest.fail("GPU checkpoint validation requires CUDA; refusing a skip")
        pytest.skip("CUDA is not available")
    return request.param


def test_snapshot_owns_nested_state_and_preserves_container_types(device):
    Pair = namedtuple("Pair", ["tensor", "array"])
    tensor = torch.arange(3.0, device=device, requires_grad=True) * 2
    array = np.arange(3)
    state = OrderedDict(
        direct=tensor,
        nested=[{"pair": Pair(tensor, array)}, (tensor,)],
        meta=torch.empty(2, device="meta"),
        mutable={"values": {1, 2}, "bytes": bytearray(b"abc")},
    )
    state._metadata = {"": {"version": [1]}}
    manager = CheckpointMananger.__new__(CheckpointMananger)
    snapshot = manager.offload_state_dict_cpu(state)
    assert isinstance(snapshot, OrderedDict)
    assert isinstance(snapshot["nested"][0]["pair"], Pair)
    assert isinstance(snapshot["nested"][1], tuple)
    assert "meta" not in snapshot
    with torch.no_grad():
        tensor.add_(100)
    array[:] = -1
    state["mutable"]["values"].add(3)
    state["mutable"]["bytes"][0] = 0
    state["nested"].append("new")
    state._metadata[""]["version"].append(2)
    for saved in (
        snapshot["direct"],
        snapshot["nested"][0]["pair"].tensor,
        snapshot["nested"][1][0],
    ):
        assert saved.device.type == "cpu"
        assert not saved.requires_grad
        torch.testing.assert_close(saved, torch.arange(3.0) * 2)
    np.testing.assert_array_equal(snapshot["nested"][0]["pair"].array, np.arange(3))
    assert len(snapshot["nested"]) == 2
    assert snapshot["mutable"] == {"values": {1, 2}, "bytes": bytearray(b"abc")}
    assert snapshot._metadata == {"": {"version": [1]}}


def test_async_save_freezes_state_before_writer_runs(tmp_path, device):
    config = Config.from_dict(
        {
            "train": {
                "output_dir": str(tmp_path / "run"),
                "timestamp": "snapshot-test",
                "ckpt": {
                    "enable_checkpoint": True,
                    "save_mode": "async",
                    "upload_s3": False,
                },
            }
        }
    )
    manager = CheckpointMananger(config)
    model = torch.nn.Linear(2, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    samples = torch.ones(2, 2, device=device)

    def update():
        optimizer.zero_grad()
        model(samples).square().mean().backward()
        optimizer.step()
        scheduler.step()

    update()
    expected_model = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_scheduler = copy.deepcopy(scheduler.state_dict())
    # Adam's step tensor is CPU even when its parameters are CUDA. Include
    # application-owned nested sampling state and NumPy RNG-like storage too.
    sampling_state = {"cursor": [torch.tensor(3), np.arange(3)], "order": (1, 2)}
    started = threading.Event()
    release = threading.Event()
    original_save = torch.save

    def paused_save(state, path):
        started.set()
        assert release.wait(30), "test failed to release the checkpoint writer"
        original_save(state, path)

    try:
        with patch("cosmos_rl.utils.checkpoint.torch.save", side_effect=paused_save):
            path = Path(
                manager.save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    step=1,
                    total_steps=3,
                    sampler_state=sampling_state,
                )
            )
            assert started.wait(10)
            assert not (path / ".rank_0_complete").exists()
            update()
            sampling_state["cursor"][0].add_(7)
            sampling_state["cursor"][1][:] = -1
            sampling_state["order"] = (8, 9)
            release.set()
            manager._wait_for_pending_async_saves()

        assert (path / ".rank_0_complete").exists()
        saved_model = torch.load(path / "model_rank_0.pth", weights_only=False)
        saved_optimizer = torch.load(path / "optimizer_rank_0.pth", weights_only=False)
        saved_scheduler = torch.load(path / "scheduler_rank_0.pth", weights_only=False)
        saved_extra = torch.load(path / "extra_info_rank_0.pth", weights_only=False)
        torch.testing.assert_close(saved_model, expected_model, check_device=False)
        torch.testing.assert_close(
            saved_optimizer, expected_optimizer, check_device=False
        )
        assert saved_scheduler == expected_scheduler
        assert saved_extra["step"] == 1
        assert saved_extra["sampler_state"]["cursor"][0].item() == 3
        np.testing.assert_array_equal(
            saved_extra["sampler_state"]["cursor"][1], np.arange(3)
        )
        assert saved_extra["sampler_state"]["order"] == (1, 2)
    finally:
        release.set()
        manager.finalize()


def test_snapshot_failure_submits_no_background_writes(tmp_path):
    config = Config.from_dict(
        {
            "train": {
                "output_dir": str(tmp_path / "run"),
                "ckpt": {"enable_checkpoint": True, "save_mode": "async"},
            }
        }
    )
    manager = CheckpointMananger(config)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    try:
        with patch.object(manager.executor, "submit") as submit:
            with pytest.raises(TypeError, match="generator"):
                manager.save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    step=1,
                    total_steps=3,
                    sampler_state=(item for item in ()),
                )
            submit.assert_not_called()
        assert not list(Path(manager.ckpt_output_dir).rglob(".rank_0_complete"))
    finally:
        manager.finalize()
