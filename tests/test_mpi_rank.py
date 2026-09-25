# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Global Torch ranks must not alias a reassigned local MPI communicator."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import cosmos_rl


class Comm:
    def __init__(self, rank, size, host_rank=None, host_size=None):
        self.rank, self.size = rank, size
        self.host_rank = rank if host_rank is None else host_rank
        self.host_size = size if host_size is None else host_size

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def Split_type(self, **kwargs):
        return Comm(self.host_rank, self.host_size)


def load_helper(monkeypatch, world, local_size):
    mpi = ModuleType("mpi4py")
    mpi.MPI = SimpleNamespace(COMM_WORLD=world)
    trt = ModuleType("tensorrt_llm")
    trt.__path__ = []
    utils = ModuleType("tensorrt_llm._utils")
    utils.mpi_broadcast = lambda value, root: value
    monkeypatch.setitem(sys.modules, "mpi4py", mpi)
    monkeypatch.setitem(sys.modules, "tensorrt_llm", trt)
    monkeypatch.setitem(sys.modules, "tensorrt_llm._utils", utils)
    path = Path(cosmos_rl.__file__).resolve().parent / "utils/mpi_distributed.py"
    spec = importlib.util.spec_from_file_location("audit_mpi_initializer", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    initialized = [False]
    init = Mock(side_effect=lambda *a, **kw: initialized.__setitem__(0, True))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: initialized[0])
    monkeypatch.setattr(torch.distributed, "init_process_group", init)
    device = Mock()
    monkeypatch.setattr(torch.cuda, "set_device", device)
    monkeypatch.setenv("COSMOS_WORLD_SIZE", str(world.size))
    monkeypatch.setenv("COSMOS_LOCAL_WORLD_SIZE", str(local_size))
    monkeypatch.setenv("COSMOS_RDZV_ENDPOINT", "127.0.0.1:23456")
    return helper, init, device


@pytest.mark.parametrize("rank", [0, 1])
def test_current_single_node_world_uses_unique_global_rank(monkeypatch, rank):
    helper, init, device = load_helper(monkeypatch, Comm(rank, 2), 2)
    helper.init_distributed_with_MPI()
    assert init.call_args.kwargs["rank"] == rank
    assert init.call_args.kwargs["world_size"] == 2
    device.assert_called_once_with(rank)


def test_unmodified_multi_node_world_is_rejected_before_native_init(monkeypatch):
    helper, init, device = load_helper(monkeypatch, Comm(2, 4, 0, 2), 2)
    with pytest.raises(AssertionError, match="COSMOS_LOCAL_WORLD_SIZE"):
        helper.init_distributed_with_MPI()
    init.assert_not_called()
    device.assert_not_called()


@pytest.mark.parametrize("rank", [2, 3])
def test_reassigned_local_communicator_must_not_redefine_global_torch_rank(
    monkeypatch, rank
):
    local_rank = rank - 2
    helper, init, device = load_helper(monkeypatch, Comm(rank, 4, local_rank, 2), 2)
    helper.set_mpi_comm(Comm(local_rank, 2))
    helper.init_distributed_with_MPI()
    device.assert_called_once_with(local_rank)
    assert init.call_args.kwargs["world_size"] == 4
    assert init.call_args.kwargs["rank"] == rank


def test_initialized_group_does_not_reinitialize_or_change_device(monkeypatch):
    helper, init, device = load_helper(monkeypatch, Comm(0, 2), 2)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    helper.init_distributed_with_MPI()
    init.assert_not_called()
    device.assert_not_called()
