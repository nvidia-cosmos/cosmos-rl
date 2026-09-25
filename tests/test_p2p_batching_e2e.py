# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-GPU correctness coverage for HA P2P batching."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

from cosmos_rl.policy.trainer.llm_trainer import llm_trainer as llm_trainer_module
from cosmos_rl.utils.distributed import HighAvailabilitylNccl
from cosmos_rl.utils.pynccl import create_nccl_comm, create_nccl_uid, nccl_abort


def _communicator(rank: int, comm_idx: int) -> HighAvailabilitylNccl:
    communicator = HighAvailabilitylNccl.__new__(HighAvailabilitylNccl)
    communicator.replica_name = f"policy-{rank}"
    communicator.global_rank = rank
    communicator.replica_name_to_rank = {"policy-0": 0, "policy-1": 1}
    communicator.comm_idx = comm_idx
    communicator.max_retry = 1
    communicator.default_timeout_ms = 30_000
    communicator.is_single_peer = threading.Event()
    communicator.is_comm_ready = threading.Event()
    communicator.is_comm_ready.set()
    communicator.build_mesh_lock = threading.RLock()
    communicator.api_client = SimpleNamespace(post_nccl_comm_error=lambda *_: None)
    return communicator


def _payloads(device_rank: int, *, source: bool) -> list[torch.Tensor]:
    device = torch.device(f"cuda:{device_rank}")
    if source:
        return [
            torch.arange(17, dtype=torch.float32, device=device),
            torch.arange(9, dtype=torch.int16, device=device),
            torch.tensor(True, dtype=torch.bool, device=device),
            torch.arange(33, dtype=torch.uint8, device=device),
        ]
    return [
        torch.zeros(17, dtype=torch.float32, device=device),
        torch.zeros(9, dtype=torch.int16, device=device),
        torch.tensor(False, dtype=torch.bool, device=device),
        torch.zeros(33, dtype=torch.uint8, device=device),
    ]


class _StateModel:
    def __init__(self, tensors: list[torch.Tensor]) -> None:
        self.tensors = {
            name: tensor
            for name, tensor in zip(
                ("a_f32", "b_i16", "c_bool", "d_u8"), tensors, strict=True
            )
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.tensors

    def named_buffers(self):
        return ()


class _OffloadedStateModel:
    def __init__(self, tensors: list[torch.Tensor]) -> None:
        self.tensors = {
            name: tensor
            for name, tensor in zip(("a_vector", "b_scalar"), tensors, strict=True)
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.tensors

    def named_buffers(self):
        return ()


class _EmptyStateManager:
    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, _state: dict) -> None:
        pass


class _EmptyCheckpointManager:
    def get_rng_state(self) -> dict:
        return {}

    def set_rng_state(self, _state: dict) -> None:
        pass


class _SyncOnlyTrainer(llm_trainer_module.LLMTrainer):
    def build_lr_schedulers(self) -> None:
        pass

    def step_training(self) -> None:
        pass


class _TrainerHook:
    supports_packing = True

    def __init__(self, single, batch) -> None:
        self.single = single
        self.batch = batch

    def __call__(self, tensor: torch.Tensor) -> None:
        self.single(tensor)


def _trainer(tensors: list[torch.Tensor], rank: int, model=None):
    trainer = _SyncOnlyTrainer.__new__(_SyncOnlyTrainer)
    trainer.parallel_dims = SimpleNamespace(pp_enabled=False)
    trainer.model = model if model is not None else _StateModel(tensors)
    trainer.reference_state_dict = {}
    trainer.optimizers = _EmptyStateManager()
    trainer.lr_schedulers = None
    trainer.ckpt_manager = _EmptyCheckpointManager()
    trainer.device = torch.device(f"cuda:{rank}")
    return trainer


def _check_payloads(tensors: list[torch.Tensor], rank: int) -> None:
    expected = _payloads(rank, source=True)
    for actual, source in zip(tensors, expected, strict=True):
        torch.testing.assert_close(actual.cpu(), source.cpu())


def _offloaded_payloads(*, source: bool) -> list[torch.Tensor]:
    if source:
        return [
            torch.arange(9, dtype=torch.float32),
            torch.tensor(17.5, dtype=torch.float32),
        ]
    return [torch.zeros(9, dtype=torch.float32), torch.tensor(0.0)]


def _check_offloaded_payloads(tensors: list[torch.Tensor]) -> None:
    expected = _offloaded_payloads(source=True)
    for actual, source in zip(tensors, expected, strict=True):
        assert actual.device.type == "cpu"
        torch.testing.assert_close(actual, source)


def _run_p2p_batching(rank: int, uid: list[int]) -> None:
    torch.cuda.set_device(rank)
    comm_idx = create_nccl_comm(uid, rank, 2)
    communicator = _communicator(rank, comm_idx)
    tensors = _payloads(rank, source=rank == 0)
    try:
        if rank == 0:
            communicator.send_batch(tensors, "policy-1")
        else:
            communicator.recv_batch(tensors, "policy-0")
            _check_payloads(tensors, rank)

        # Reuse the communicator for a second ordered collective sequence and
        # verify that broadcast batching has identical movement semantics.
        tensors = _payloads(rank, source=rank == 0)
        communicator.broadcast_batch(tensors, "policy-0")
        _check_payloads(tensors, rank)

        # Exercise LLMTrainer's production hybrid path: the 68-byte first
        # tensor remains direct while the remaining mixed-dtype tensors pack
        # into one 52-byte payload.
        llm_trainer_module._P2P_SYNC_BUCKET_SIZE_BYTES = 64
        llm_trainer_module._P2P_SYNC_PACK_TENSORS = True
        tensors = _payloads(rank, source=rank == 0)
        trainer = _trainer(tensors, rank)
        send_hook = _TrainerHook(
            lambda tensor: communicator.send(tensor, "policy-1"),
            lambda batch: communicator.send_batch(batch, "policy-1"),
        )
        recv_hook = _TrainerHook(
            lambda tensor: communicator.recv(tensor, "policy-0"),
            lambda batch: communicator.recv_batch(batch, "policy-0"),
        )
        assert trainer.sync_all_states(rank == 0, send_hook, recv_hook) == 4
        if rank == 1:
            _check_payloads(tensors, rank)

        tensors = _payloads(rank, source=rank == 0)
        trainer = _trainer(tensors, rank)
        broadcast_hook = _TrainerHook(
            lambda tensor: communicator.broadcast(tensor, "policy-0"),
            lambda batch: communicator.broadcast_batch(batch, "policy-0"),
        )
        assert trainer.sync_all_states(rank == 0, broadcast_hook, broadcast_hook) == 4
        _check_payloads(tensors, rank)

        # CPU/offloaded state takes a temporary CUDA path for NCCL, then must
        # unpack and copy back into the original CPU tensors.
        offloaded = _offloaded_payloads(source=rank == 0)
        trainer = _trainer(
            offloaded,
            rank,
            model=_OffloadedStateModel(offloaded),
        )
        assert trainer.sync_all_states(rank == 0, send_hook, recv_hook) == 2
        if rank == 1:
            _check_offloaded_payloads(offloaded)

        offloaded = _offloaded_payloads(source=rank == 0)
        trainer = _trainer(
            offloaded,
            rank,
            model=_OffloadedStateModel(offloaded),
        )
        assert trainer.sync_all_states(rank == 0, broadcast_hook, broadcast_hook) == 2
        _check_offloaded_payloads(offloaded)
    finally:
        nccl_abort(comm_idx)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_real_nccl_p2p_batches_preserve_mixed_dtype_payloads() -> None:
    mp.spawn(
        _run_p2p_batching,
        args=(create_nccl_uid(),),
        nprocs=2,
        join=True,
    )
