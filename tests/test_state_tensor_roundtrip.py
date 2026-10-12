# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Optimizer/scheduler/RNG state keeps array ownership and dtype on transfer."""

import os

import numpy as np
import pytest
import torch

from cosmos_rl.policy.trainer.base import (
    extract_from_cuda_tensor,
    wrap_to_cuda_tensor,
)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        if os.environ.get("COSMOS_REQUIRE_CUDA") == "1":
            pytest.fail("Required CUDA state-transfer coverage is unavailable")
        pytest.skip("CUDA is unavailable")
    return torch.device(request.param)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.uint32])
@pytest.mark.parametrize("shape", [(), (0,), (2, 3)])
def test_array_receive_preserves_live_owner_and_dtype(device, dtype, shape):
    sent = np.full(shape, 7, dtype=dtype)
    target = np.zeros(shape, dtype=dtype)
    alias = target.view()
    wire = wrap_to_cuda_tensor(device, "state", sent)
    restored = extract_from_cuda_tensor(device, "state", target, wire)
    assert restored is target
    assert restored.dtype == dtype
    np.testing.assert_array_equal(restored, sent)
    np.testing.assert_array_equal(alias, sent)


def test_noncontiguous_array_receive_updates_its_owner(device):
    owner = np.zeros((3, 4), dtype=np.float32)
    target = owner[:, ::2]
    expected = np.arange(6, dtype=np.float32).reshape(3, 2)
    result = extract_from_cuda_tensor(
        device, "state", target, wrap_to_cuda_tensor(device, "state", expected)
    )
    assert result is target
    np.testing.assert_array_equal(owner[:, ::2], expected)
    np.testing.assert_array_equal(owner[:, 1::2], 0)


@pytest.mark.parametrize("dtype", [np.float32, np.uint32])
def test_tuple_arrays_keep_original_dtype(device, dtype):
    sent = ("array-state", np.array([1, 2, 3], dtype=dtype), (4, 5), 6)
    target = ("array-state", np.zeros(3, dtype=dtype), (0, 0), 0)
    result = extract_from_cuda_tensor(
        device, "state", target, wrap_to_cuda_tensor(device, "state", sent)
    )
    assert isinstance(result, tuple)
    assert result[1].dtype == dtype
    np.testing.assert_array_equal(result[1], sent[1])
    assert result[2:] == sent[2:]


def test_numpy_rng_transfer_keeps_state_and_next_draws(device):
    sender = np.random.RandomState(173)
    receiver = np.random.RandomState(91)
    sender.normal(size=7)
    state = sender.get_state()
    restored = extract_from_cuda_tensor(
        device,
        "numpy_rng",
        receiver.get_state(),
        wrap_to_cuda_tensor(device, "numpy_rng", state),
    )
    assert restored[1].dtype == state[1].dtype == np.uint32
    receiver.set_state(restored)
    np.testing.assert_array_equal(receiver.normal(size=16), sender.normal(size=16))


def test_invalid_array_shape_fails_before_mutation(device):
    target = np.full(3, 9, dtype=np.float32)
    with pytest.raises(ValueError, match="same shape"):
        extract_from_cuda_tensor(device, "state", target, torch.ones(2, device=device))
    np.testing.assert_array_equal(target, 9)


@pytest.mark.parametrize("packed", [False, True])
def test_actual_trainer_sync_roundtrip(device, packed, monkeypatch):
    from cosmos_rl.policy.trainer.llm_trainer import llm_trainer as module
    from test_p2p_batching import _state_sync_trainer

    monkeypatch.setattr(module, "_P2P_SYNC_BUCKET_SIZE_BYTES", 64)
    monkeypatch.setattr(module, "_P2P_SYNC_PACK_TENSORS", packed)
    sender = _state_sync_trainer(7)
    receiver = _state_sync_trainer(0)
    for trainer, seed, fill in ((sender, 173, 7), (receiver, 91, 0)):
        trainer.device = device
        trainer.optimizers.state["array"] = np.full((2, 3), fill, dtype=np.float32)
        trainer.lr_schedulers.state["array"] = np.full((), fill, dtype=np.float64)
        rng = np.random.RandomState(seed)
        rng.normal(size=7)
        trainer.ckpt_manager.state["numpy"] = rng.get_state()
    alias = receiver.optimizers.state["array"]
    queue = []

    class Hook:
        supports_packing = True

        def __init__(self, sending):
            self.sending = sending

        def __call__(self, tensor):
            if self.sending:
                queue.append(tensor.clone())
            else:
                tensor.copy_(queue.pop(0))

        def batch(self, tensors):
            for tensor in tensors:
                self(tensor)

    send, receive = Hook(True), Hook(False)
    assert sender.sync_all_states(True, send, receive) == receiver.sync_all_states(
        False, send, receive
    )
    assert not queue
    assert receiver.optimizers.state["array"] is alias
    np.testing.assert_array_equal(alias, sender.optimizers.state["array"])
    np.testing.assert_array_equal(receiver.lr_schedulers.state["array"], 7)
    state = receiver.ckpt_manager.state["numpy"]
    assert state[1].dtype == np.uint32
    np.testing.assert_array_equal(state[1], sender.ckpt_manager.state["numpy"][1])
