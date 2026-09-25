# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for byte-bounded policy-to-policy NCCL batching."""

from __future__ import annotations

import threading
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from cosmos_rl.policy.trainer.llm_trainer import llm_trainer as llm_trainer_module
from cosmos_rl.utils import distributed as dist_utils


def _communicator(*, max_retry: int = 3) -> dist_utils.HighAvailabilitylNccl:
    communicator = dist_utils.HighAvailabilitylNccl.__new__(
        dist_utils.HighAvailabilitylNccl
    )
    communicator.replica_name = "policy-0"
    communicator.global_rank = 0
    communicator.replica_name_to_rank = {"policy-0": 0, "policy-1": 1}
    communicator.comm_idx = 7
    communicator.max_retry = max_retry
    communicator.default_timeout_ms = 10
    communicator.is_single_peer = threading.Event()
    communicator.is_comm_ready = threading.Event()
    communicator.is_comm_ready.set()
    communicator.build_mesh_lock = threading.RLock()
    communicator.api_client = SimpleNamespace(post_nccl_comm_error=lambda *_: None)
    communicator.wait_comm_ready = lambda timeout=0: None
    return communicator


@pytest.mark.parametrize(
    ("method_name", "raw_name", "peer_name"),
    [
        ("broadcast_batch", "nccl_broadcast", "policy-0"),
        ("send_batch", "nccl_send", "policy-1"),
        ("recv_batch", "nccl_recv", "policy-1"),
    ],
)
def test_p2p_batch_uses_one_ha_window_and_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    raw_name: str,
    peer_name: str,
) -> None:
    communicator = _communicator()
    tensors = [torch.tensor([value]) for value in range(4)]
    raw_calls: list[torch.Tensor] = []
    watchdog_calls = 0

    def record_raw_call(tensor: torch.Tensor, **_kwargs) -> None:
        raw_calls.append(tensor)

    @contextmanager
    def record_watchdog(**_kwargs):
        nonlocal watchdog_calls
        watchdog_calls += 1
        yield

    monkeypatch.setattr(dist_utils, raw_name, record_raw_call)
    monkeypatch.setattr(dist_utils, "nccl_timeout_watchdog", record_watchdog)

    getattr(communicator, method_name)((tensor for tensor in tensors), peer_name)

    assert raw_calls == tensors
    assert watchdog_calls == 1


def test_p2p_batch_failure_does_not_replay_already_posted_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    communicator = _communicator(max_retry=2)
    tensors = [torch.tensor([value]) for value in range(3)]
    raw_calls: list[torch.Tensor] = []
    reports: list[Exception] = []
    failed_once = False

    def fail_once_on_second_tensor(tensor: torch.Tensor, **_kwargs) -> None:
        nonlocal failed_once
        raw_calls.append(tensor)
        if tensor is tensors[1] and not failed_once:
            failed_once = True
            raise OSError("synthetic mid-batch failure")

    communicator.api_client = SimpleNamespace(
        post_nccl_comm_error=lambda _name, error: reports.append(error)
    )
    monkeypatch.setattr(dist_utils, "nccl_send", fail_once_on_second_tensor)
    monkeypatch.setattr(
        dist_utils,
        "nccl_timeout_watchdog",
        lambda **_kwargs: nullcontext(),
    )

    with pytest.raises(dist_utils.CollectiveOperationError, match="not replayed"):
        communicator.send_batch(tensors, "policy-1")

    assert raw_calls == [tensors[0], tensors[1]]
    assert len(reports) == 1


def test_empty_p2p_batch_is_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    communicator = _communicator()
    monkeypatch.setattr(
        dist_utils,
        "nccl_broadcast",
        lambda **_kwargs: pytest.fail("empty batch issued a raw NCCL call"),
    )

    communicator.broadcast_batch([], "policy-0")


class _StateManager:
    def __init__(self, state: dict) -> None:
        self.state = state

    def state_dict(self) -> dict:
        return self.state

    def load_state_dict(self, state: dict) -> None:
        self.state = state


class _CheckpointManager:
    def __init__(self, state: dict) -> None:
        self.state = state

    def get_rng_state(self) -> dict:
        return self.state

    def set_rng_state(self, state: dict) -> None:
        self.state = state


class _ModelState(torch.nn.Module):
    def __init__(self, fill: int) -> None:
        super().__init__()
        self.register_buffer("a", torch.full((2,), fill, dtype=torch.float32))
        self.register_buffer("b", torch.full((4,), fill, dtype=torch.int16))
        self.register_buffer("c", torch.full((4,), fill, dtype=torch.uint8))
        self.register_buffer("d", torch.full((4,), fill, dtype=torch.float32))


class _SyncOnlyTrainer(llm_trainer_module.LLMTrainer):
    def build_lr_schedulers(self) -> None:
        pass

    def step_training(self) -> None:
        pass


class _InMemoryBatchTransport:
    def __init__(self) -> None:
        self.payloads: list[list[torch.Tensor]] = []
        self.sent_batch_sizes: list[int] = []
        self.received_batch_sizes: list[int] = []

    def reject_single(self, _tensor: torch.Tensor) -> None:
        pytest.fail("state sync fell back to a single-tensor hook")

    def send_batch(self, tensors) -> None:
        tensors = list(tensors)
        self.sent_batch_sizes.append(len(tensors))
        self.payloads.append([tensor.clone() for tensor in tensors])

    def recv_batch(self, tensors) -> None:
        tensors = list(tensors)
        self.received_batch_sizes.append(len(tensors))
        payload = self.payloads.pop(0)
        assert len(payload) == len(tensors)
        for destination, source in zip(tensors, payload, strict=True):
            destination.copy_(source)


class _InMemoryPackedTransport:
    def __init__(self) -> None:
        self.payloads: list[tuple[str, list[torch.Tensor] | torch.Tensor]] = []
        self.sent_call_kinds: list[str] = []
        self.received_call_kinds: list[str] = []

    def send(self, tensor: torch.Tensor) -> None:
        self.sent_call_kinds.append("single")
        self.payloads.append(("single", tensor.clone()))

    def recv(self, tensor: torch.Tensor) -> None:
        self.received_call_kinds.append("single")
        kind, payload = self.payloads.pop(0)
        assert kind == "single"
        assert isinstance(payload, torch.Tensor)
        tensor.copy_(payload)

    def send_batch(self, tensors) -> None:
        tensors = list(tensors)
        self.sent_call_kinds.append("batch")
        self.payloads.append(("batch", [tensor.clone() for tensor in tensors]))

    def recv_batch(self, tensors) -> None:
        tensors = list(tensors)
        self.received_call_kinds.append("batch")
        kind, payload = self.payloads.pop(0)
        assert kind == "batch"
        assert isinstance(payload, list)
        assert len(payload) == len(tensors)
        for destination, source in zip(tensors, payload, strict=True):
            destination.copy_(source)


def _state_sync_trainer(fill: int):
    trainer = _SyncOnlyTrainer.__new__(_SyncOnlyTrainer)
    trainer.parallel_dims = SimpleNamespace(pp_enabled=False)
    trainer.model = _ModelState(fill)
    trainer.reference_state_dict = {}
    trainer.optimizers = _StateManager(
        {"momentum": torch.full((4,), fill, dtype=torch.float32)}
    )
    trainer.lr_schedulers = _StateManager({"step": fill})
    trainer.ckpt_manager = _CheckpointManager(
        {"cpu_rng": torch.full((4,), fill, dtype=torch.uint8)}
    )
    trainer.device = torch.device("cpu")
    return trainer


def test_state_sync_uses_deterministic_byte_buckets_and_preserves_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Sorted model tensor sizes are 8, 8, 4, and 16 bytes. A 12-byte bound
    # therefore creates [8], [8, 4], and one oversized [16] bucket.
    monkeypatch.setattr(llm_trainer_module, "_P2P_SYNC_BUCKET_SIZE_BYTES", 12)
    source = _state_sync_trainer(fill=7)
    destination = _state_sync_trainer(fill=0)
    transport = _InMemoryBatchTransport()

    class Hook:
        def __init__(self, single, batch) -> None:
            self.single = single
            self.batch = batch

        def __call__(self, tensor) -> None:
            self.single(tensor)

    send_hook = Hook(transport.reject_single, transport.send_batch)
    recv_hook = Hook(transport.reject_single, transport.recv_batch)

    sent = source.sync_all_states(True, send_hook, recv_hook)
    received = destination.sync_all_states(False, send_hook, recv_hook)

    assert sent == received == 7
    assert transport.sent_batch_sizes == [1, 2, 1, 1, 1, 1]
    assert transport.received_batch_sizes == [1, 2, 1, 1, 1, 1]
    assert transport.payloads == []
    for name, source_tensor in source.model.state_dict().items():
        torch.testing.assert_close(destination.model.state_dict()[name], source_tensor)
    torch.testing.assert_close(
        destination.optimizers.state["momentum"],
        source.optimizers.state["momentum"],
    )
    assert destination.lr_schedulers.state == source.lr_schedulers.state
    torch.testing.assert_close(
        destination.ckpt_manager.state["cpu_rng"],
        source.ckpt_manager.state["cpu_rng"],
    )


def test_state_sync_packs_multi_tensor_buckets_and_preserves_mixed_dtypes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_trainer_module, "_P2P_SYNC_BUCKET_SIZE_BYTES", 12)
    monkeypatch.setattr(llm_trainer_module, "_P2P_SYNC_PACK_TENSORS", True)
    source = _state_sync_trainer(fill=7)
    destination = _state_sync_trainer(fill=0)
    transport = _InMemoryPackedTransport()

    class Hook:
        supports_packing = True

        def __init__(self, single, batch) -> None:
            self.single = single
            self.batch = batch

        def __call__(self, tensor) -> None:
            self.single(tensor)

    send_hook = Hook(transport.send, transport.send_batch)
    recv_hook = Hook(transport.recv, transport.recv_batch)

    sent = source.sync_all_states(True, send_hook, recv_hook)
    received = destination.sync_all_states(False, send_hook, recv_hook)

    assert sent == received == 7
    assert transport.sent_call_kinds == [
        "batch",
        "single",
        "batch",
        "batch",
        "batch",
        "batch",
    ]
    assert transport.received_call_kinds == transport.sent_call_kinds
    assert transport.payloads == []
    assert source._p2p_sync_packed_buffer.dtype == torch.uint8
    assert source._p2p_sync_packed_buffer.numel() == 12
    for name, source_tensor in source.model.state_dict().items():
        torch.testing.assert_close(destination.model.state_dict()[name], source_tensor)
    torch.testing.assert_close(
        destination.optimizers.state["momentum"],
        source.optimizers.state["momentum"],
    )
    assert destination.lr_schedulers.state == source.lr_schedulers.state
    torch.testing.assert_close(
        destination.ckpt_manager.state["cpu_rng"],
        source.ckpt_manager.state["cpu_rng"],
    )


def test_state_sync_packing_supports_pipeline_local_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_trainer_module, "_P2P_SYNC_BUCKET_SIZE_BYTES", 12)
    monkeypatch.setattr(llm_trainer_module, "_P2P_SYNC_PACK_TENSORS", True)
    source = _state_sync_trainer(fill=7)
    destination = _state_sync_trainer(fill=0)
    source.parallel_dims.pp_enabled = True
    destination.parallel_dims.pp_enabled = True
    source.model_parts = [source.model]
    destination.model_parts = [destination.model]
    transport = _InMemoryPackedTransport()

    class Hook:
        supports_packing = True

        def __init__(self, single, batch) -> None:
            self.single = single
            self.batch = batch

        def __call__(self, tensor) -> None:
            self.single(tensor)

    send_hook = Hook(transport.send, transport.send_batch)
    recv_hook = Hook(transport.recv, transport.recv_batch)

    assert source.sync_all_states(True, send_hook, recv_hook) == 7
    assert destination.sync_all_states(False, send_hook, recv_hook) == 7
    assert transport.payloads == []
    for name, source_tensor in source.model.state_dict().items():
        torch.testing.assert_close(destination.model.state_dict()[name], source_tensor)


def test_state_sync_packing_supports_scalar_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_trainer_module, "_P2P_SYNC_BUCKET_SIZE_BYTES", 12)
    monkeypatch.setattr(llm_trainer_module, "_P2P_SYNC_PACK_TENSORS", None)

    class ScalarState:
        def __init__(self, fill: int) -> None:
            self.state = {
                "a_vector": torch.full((4,), fill, dtype=torch.int16),
                "b_scalar": torch.tensor(fill, dtype=torch.float32),
            }

        def state_dict(self):
            return self.state

        def named_buffers(self):
            return ()

    def trainer(fill: int):
        instance = _state_sync_trainer(fill)
        instance.model = ScalarState(fill)
        instance.optimizers = _StateManager({})
        instance.lr_schedulers = None
        instance.ckpt_manager = _CheckpointManager({})
        return instance

    source = trainer(7)
    destination = trainer(0)
    source.config = SimpleNamespace(train=SimpleNamespace(p2p_sync_pack_tensors=True))
    destination.config = SimpleNamespace(
        train=SimpleNamespace(p2p_sync_pack_tensors=True)
    )
    transport = _InMemoryPackedTransport()

    class Hook:
        supports_packing = True

        def __init__(self, single, batch) -> None:
            self.single = single
            self.batch = batch

        def __call__(self, tensor) -> None:
            self.single(tensor)

    send_hook = Hook(transport.send, transport.send_batch)
    recv_hook = Hook(transport.recv, transport.recv_batch)

    assert source.sync_all_states(True, send_hook, recv_hook) == 2
    assert destination.sync_all_states(False, send_hook, recv_hook) == 2
    assert transport.sent_call_kinds == ["single"]
    assert transport.received_call_kinds == ["single"]
    for name, source_tensor in source.model.state_dict().items():
        torch.testing.assert_close(destination.model.state_dict()[name], source_tensor)


def test_state_sync_keeps_plain_single_tensor_hook_compatibility() -> None:
    source = _state_sync_trainer(fill=7)
    tensors: list[torch.Tensor] = []

    sent = source.sync_all_states(True, tensors.append, tensors.append)

    assert sent == len(tensors) == 7


def test_p2p_packing_config_is_opt_in() -> None:
    from cosmos_rl.policy.config import TrainingConfig

    assert TrainingConfig().p2p_sync_pack_tensors is False
    assert TrainingConfig(p2p_sync_pack_tensors=True).p2p_sync_pack_tensors is True
