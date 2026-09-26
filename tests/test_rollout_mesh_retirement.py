# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Retire old mesh handles only after their last reader has drained."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from cosmos_rl.dispatcher.command import BuildMeshCommand
from cosmos_rl.rollout.worker import rollout_control
from cosmos_rl.utils import pynccl
from rollout_mesh_retirement_canary import broadcast_payload


@pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64])
def test_native_canary_peers_have_identical_payload_layout(default_dtype):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        source = broadcast_payload(0, 1.0, "cpu")
        receiver = broadcast_payload(1, 1.0, "cpu")
    finally:
        torch.set_default_dtype(previous)
    assert source.dtype == receiver.dtype == torch.float32
    assert source.numel() * source.element_size() == 4 * 1024 * 1024
    assert receiver.shape == source.shape
    torch.testing.assert_close(source, torch.ones_like(source))
    torch.testing.assert_close(receiver, -torch.ones_like(receiver))


@pytest.fixture
def mesh():
    registry = pynccl._CommunicatorRegistry()
    native = Mock()
    old_comm = object()
    old = registry.register(old_comm, 0, 2)
    worker = SimpleNamespace(
        state=SimpleNamespace(prompt_consume_end=lambda: False),
        replica_name="rollout-0",
        _weight_sync_thread=None,
        _mesh_rebuild_ready=threading.Event(),
        global_commnicator_idex=old,
        inference_stream=object(),
        replica_name_to_rank={"rollout-0": 0, "old-peer": 1},
        get_group_unique_key=lambda _: "new-mesh",
        api_client=Mock(),
    )
    with (
        patch.object(pynccl, "_COMM_REGISTRY", registry),
        patch.object(pynccl, "_nccl", native),
    ):
        yield worker, registry, native, old, old_comm


@pytest.mark.parametrize("target", ["replace", "single", "unused", "init-failure"])
def test_old_handle_is_retired_for_every_valid_rebuild_target(mesh, target):
    worker, registry, native, old, old_comm = mesh
    mapping = (
        {"rollout-0": 0} if target == "single" else {"rollout-0": 0, "new-peer": 1}
    )

    def create(*args, **kwargs):
        assert not registry.contains(old), (
            "new communicator created before old retirement"
        )
        if target == "init-failure":
            raise TimeoutError("missing new peer")
        return registry.register(object(), 0, 2)

    def drain(*args):
        assert registry.contains(old), "old handle was aborted before stream completion"
        native.ncclCommAbort.assert_not_called()
        return True

    with (
        patch.object(
            rollout_control, "bounded_drain_or_abort", side_effect=drain
        ) as drained,
        patch.object(rollout_control, "create_nccl_comm", side_effect=create),
        patch.object(rollout_control, "create_nccl_uid", return_value=[1, 2, 3]),
    ):
        rollout_control.DisaggregatedRolloutControlWorker.build_global_mesh(
            worker, BuildMeshCommand(mapping, mesh_is_used=target != "unused")
        )
    assert not registry.contains(old)
    native.ncclCommAbort.assert_called_once_with(old_comm)
    drained.assert_called_once()
    assert worker._mesh_rebuild_ready.is_set()
    if target == "replace":
        assert registry.contains(worker.global_commnicator_idex)
        assert len(registry.all_indices()) == 1
    else:
        assert worker.global_commnicator_idex == -1
        assert registry.all_indices() == []


def test_failed_drain_does_not_replace_or_publish_mesh(mesh):
    worker, registry, native, old, _ = mesh
    with (
        patch.object(rollout_control, "bounded_drain_or_abort", return_value=False),
        patch.object(rollout_control, "create_nccl_comm") as create,
        patch.object(rollout_control, "create_nccl_uid") as uid,
    ):
        with pytest.raises(RuntimeError, match="drain"):
            rollout_control.DisaggregatedRolloutControlWorker.build_global_mesh(
                worker, BuildMeshCommand({"rollout-0": 0, "new-peer": 1})
            )
    create.assert_not_called()
    uid.assert_not_called()
    assert not worker._mesh_rebuild_ready.is_set()
    assert worker.replica_name_to_rank == {"rollout-0": 0, "old-peer": 1}
    # The mocked drain did not actually abort; it cannot certify destruction.
    assert registry.contains(old)
    native.ncclCommAbort.assert_not_called()


def test_invalid_membership_does_not_retire_existing_mesh(mesh):
    worker, registry, native, old, _ = mesh
    with pytest.raises(RuntimeError, match="not found"):
        rollout_control.DisaggregatedRolloutControlWorker.build_global_mesh(
            worker, BuildMeshCommand({"someone-else": 0})
        )
    assert registry.contains(old)
    native.ncclCommAbort.assert_not_called()


@pytest.mark.parametrize("failure_phase", ["create", "record"])
def test_cuda_event_failure_cannot_certify_a_drain(failure_phase):
    event = Mock()
    if failure_phase == "record":
        event.record.side_effect = RuntimeError("CUDA event recording failed")
    with patch.object(
        pynccl.torch.cuda,
        "Event",
        side_effect=RuntimeError("CUDA event creation failed")
        if failure_phase == "create"
        else None,
        return_value=event,
    ):
        assert (
            pynccl.bounded_drain_or_abort(object(), 0.1, "audit-event-error") is False
        )
    event.query.assert_not_called()
