# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The same logical payload must produce the same bytes on every producer."""

import uuid
from unittest.mock import patch

import numpy as np
import pytest
import torch

from cosmos_rl.utils.payload_transport.pack import NP_TO_TORCH, pack_trajectory_into
from cosmos_rl.utils.payload_transport.nccl.strategy import _unpack
from cosmos_rl.utils.payload_transport.ucxx.shared_buffer import (
    BufferConfig,
    SharedRingBuffer,
    SlotState,
)
from cosmos_rl.utils.tensor_packing import (
    pack_tensors_into_buffer,
    unpack_tensors_from_buffer,
)
from cosmos_rl.utils.trajectory import (
    TensorSpec,
    build_trajectory_schema,
    schema_layout,
    truncate_trajectory,
)
from test_ucxx_rollout_mixin import _make_producer


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def packed(data, schema, length=0, device="cpu"):
    offsets, size = schema_layout(schema)
    # Verify optional fields are zeroed even with reused/nonzero storage.
    buffer = torch.full((size,), 255, dtype=torch.uint8, device=device)
    pack_trajectory_into(buffer, data, schema, offsets, length, device)
    return buffer


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(), (1,), (0,), (2, 0, 3), (2, 3)])
@pytest.mark.parametrize("dtype", list(NP_TO_TORCH))
def test_all_wire_dtypes_and_shapes(dtype, shape, device):
    schema = [TensorSpec(shape, dtype, "value")]
    value = torch.ones(shape, dtype=NP_TO_TORCH[dtype], device=device)
    restored = _unpack(packed({"value": value}, schema, device=device), schema, device)
    torch.testing.assert_close(restored["value"], value)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "kind", ["bf16", "reversed", "grad", "singleton", "transpose", "expand"]
)
def test_layout_and_dtype_conversion(kind, device):
    values = {
        "bf16": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
        "reversed": np.arange(4, dtype=np.float32)[::-1],
        "grad": torch.tensor([1.0, 2.0], requires_grad=True),
        "singleton": torch.arange(8.0).reshape(1, 8)[:, 2],
        "transpose": torch.arange(6.0).reshape(2, 3).t(),
        "expand": torch.ones(1).expand(3),
    }
    value = values[kind]
    expected = (
        value.detach().float()
        if isinstance(value, torch.Tensor)
        else torch.tensor(value.copy())
    )
    schema = [TensorSpec(tuple(value.shape), np.float32, "value")]
    restored = _unpack(packed({"value": value}, schema, device=device), schema, device)
    torch.testing.assert_close(restored["value"].cpu(), expected)


@pytest.mark.parametrize("device", DEVICES)
def test_weight_singleton_strides_both_directions(device):
    source = torch.arange(8.0, device=device).reshape(1, 8)[:, 2]
    backing = torch.full((1, 8), -1.0, device=device)
    destination = backing[:, 5]
    buf = torch.empty(8, dtype=torch.uint8, device=device)
    pack_tensors_into_buffer([source, torch.tensor(7.0, device=device)], buf)
    scalar = torch.tensor(0.0, device=device)
    unpack_tensors_from_buffer(buf, [destination, scalar])
    assert destination.item() == 2 and scalar.item() == 7
    assert torch.all(backing[:, :5] == -1) and torch.all(backing[:, 6:] == -1)


@pytest.mark.parametrize("length", [-1, 5, 1.5, True])
def test_invalid_lengths_rejected_by_pack_and_decode(length):
    schema = build_trajectory_schema(dict(max_steps=4, obs_dim=3, action_dim=1))
    with pytest.raises(ValueError, match="episode_length"):
        packed({}, schema, length)
    with pytest.raises(ValueError, match="episode_length"):
        truncate_trajectory(
            {"episode_length": torch.tensor(length), "observations": torch.ones(4, 3)}
        )


@pytest.mark.parametrize("shape", [(2, 1), (2,), (5, 3), (1, 3)])
def test_padding_never_broadcasts_or_invents_valid_rows(shape):
    schema = build_trajectory_schema(dict(max_steps=4, obs_dim=3, action_dim=1))
    with pytest.raises(ValueError, match="observations"):
        packed({"observations": torch.ones(shape)}, schema, 2)


@pytest.mark.parametrize(
    "dtype,shape", [(np.float64, (1,)), (np.int32, (1,)), (np.int64, (2,))]
)
def test_episode_schema_rejected(dtype, shape):
    with pytest.raises(ValueError, match="episode_length"):
        schema_layout([TensorSpec(shape, dtype, "episode_length")])


@pytest.fixture
def ring():
    schema = build_trajectory_schema(dict(max_steps=6, obs_dim=3, action_dim=2))
    buffer = SharedRingBuffer(
        BufferConfig(
            max_entries=1,
            schema=schema,
            buffer_name="packer_contract_" + uuid.uuid4().hex,
        ),
        create=True,
    )
    try:
        yield buffer
    finally:
        buffer.close()
        buffer.unlink()


@pytest.mark.parametrize("device", DEVICES)
def test_producer_bytes_match_shared_packer(ring, device):
    producer = _make_producer(buffer=ring)
    ring.ports = [7000]
    # Missing optional fields, BF16 and scalar custom fields in a real SHM ring.
    data = {
        "observations": torch.ones(2, 3, dtype=torch.bfloat16, device=device),
        "episode_length": 2,
    }
    metadata = producer.write_to_buffer(data)
    assert metadata is not None
    raw = ring.read_raw(metadata["_slot"])
    expected = packed(data, ring.schema, 2).cpu().numpy()
    np.testing.assert_array_equal(raw, expected)


def test_invalid_overwrite_preserves_ready_payload(ring):
    data = {"observations": torch.ones(2, 3), "episode_length": 2}
    ring.write(data)
    before = bytes(ring._shm.buf)
    with pytest.raises(ValueError, match="observations"):
        ring.write(dict(data, observations=torch.ones(2, 1)))
    assert bytes(ring._shm.buf) == before


def test_partial_copy_failure_releases_slot_and_allows_retry(ring):
    data = {"observations": torch.ones(2, 3), "episode_length": 2}
    ring.write(data)
    original = ring._write_tensors

    def fail_after_copy(*args):
        original(*args)
        raise RuntimeError("injected copy failure")

    with patch.object(ring, "_write_tensors", side_effect=fail_after_copy):
        with pytest.raises(RuntimeError, match="injected"):
            ring.write(data)
    assert ring.get_slot_state(0) == SlotState.FREE
    assert ring.write(data) == 0
    assert ring.get_slot_state(0) == SlotState.READY


def test_bad_then_valid_producer_write(ring):
    producer = _make_producer(buffer=ring)
    ring.ports = [7000]
    assert producer.write_to_buffer({"observations": torch.ones(2, 1)}) is None
    assert ring.get_slot_state(0) == SlotState.FREE
    assert producer.write_to_buffer({"observations": torch.ones(2, 3)}) is not None


@pytest.mark.parametrize(
    "rewards", [[1.0, 2.0], np.array([1.0, 2.0]), torch.tensor([1.0, 2.0])]
)
def test_cpu_producer_accepts_list_inputs_before_publication(ring, rewards):
    producer = _make_producer(buffer=ring)
    ring.ports = [7000]
    metadata = producer.write_to_buffer(
        {"observations": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "rewards": rewards}
    )
    assert metadata is not None and metadata["rewards"] == [1.0, 2.0]
    assert ring.get_slot_state(0) == SlotState.READY


@pytest.mark.parametrize("device", DEVICES)
def test_custom_scalars_keep_nccl_reference(device):
    from test_pack_canonical_stride import TestProducerKeepsNcclPath

    producer = TestProducerKeepsNcclPath()._producer()
    producer._nccl_device = torch.device(device)
    producer._nccl_schema[-1] = TensorSpec((), np.int64, "sampled_mode")
    producer._nccl_offsets, producer._nccl_entry_size = schema_layout(
        producer._nccl_schema
    )
    metadata = producer.write_to_buffer(
        {
            "observations": torch.zeros(2, 2, device=device),
            "sampled_mode": torch.tensor(3, device=device),
        }
    )
    assert metadata is not None and metadata["_nccl"]
    assert metadata["_transfer_id"] in producer._nccl_registry


def test_misaligned_fields_scalar_and_empty_episode():
    schema = [TensorSpec((1,), np.bool_, "flag"), TensorSpec((), np.float64, "scalar")]
    restored = _unpack(packed({"flag": [True], "scalar": 2.5}, schema), schema, "cpu")
    assert restored["scalar"].shape == () and restored["scalar"].item() == 2.5
    schema = build_trajectory_schema(dict(max_steps=4, obs_dim=3, action_dim=1))
    restored = _unpack(
        packed({"observations": torch.empty(0, 3)}, schema, 0), schema, "cpu"
    )
    assert restored["observations"].shape == (0, 3)


def test_raw_copy_failure_releases_slot(ring):
    from types import SimpleNamespace

    class BrokenBuffer:
        def __getitem__(self, key):
            return shm.buf[key]

        def __setitem__(self, key, value):
            if key.start == ring._entry_data_offset(0):
                raise RuntimeError("injected raw copy failure")
            shm.buf[key] = value

    shm = ring._shm
    # Header/meta access uses struct.pack_into (buffer protocol); inject only
    # the payload assignment through the SHM facade, leaving metadata helpers real.
    read_header, write_header = ring._read_header, ring._write_header
    read_meta, write_meta = ring._read_entry_meta, ring._write_entry_meta

    def on_real_shm(fn, *args):
        with patch.object(ring, "_shm", shm):
            return fn(*args)

    with (
        patch.object(ring, "_shm", SimpleNamespace(buf=BrokenBuffer())),
        patch.object(
            ring, "_read_header", side_effect=lambda: on_real_shm(read_header)
        ),
        patch.object(
            ring, "_write_header", side_effect=lambda *a: on_real_shm(write_header, *a)
        ),
        patch.object(
            ring, "_read_entry_meta", side_effect=lambda *a: on_real_shm(read_meta, *a)
        ),
        patch.object(
            ring,
            "_write_entry_meta",
            side_effect=lambda *a: on_real_shm(write_meta, *a),
        ),
    ):
        with pytest.raises(RuntimeError, match="injected raw"):
            ring.write_raw(bytes(ring.entry_data_size))
    assert ring.get_slot_state(0) == SlotState.FREE
    assert ring.write_raw(bytes(ring.entry_data_size)) == 0
