# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from cosmos_rl.dispatcher.command import RolloutToRolloutBroadcastCommand
from cosmos_rl.rollout.worker.rollout_control import (
    DisaggregatedRolloutControlWorker,
)
from cosmos_rl.rollout.worker import weight_sync
from cosmos_rl.utils.tensor_packing import (
    iter_tensor_byte_buckets,
    pack_tensors_into_buffer,
    unpack_tensors_from_buffer,
)


def test_byte_buckets_preserve_order_and_isolate_oversized_tensors() -> None:
    tensors = [
        torch.empty(2, dtype=torch.float32),
        torch.empty(3, dtype=torch.float32),
        torch.empty(5, dtype=torch.float32),
        torch.empty(1, dtype=torch.float32),
    ]

    buckets = list(iter_tensor_byte_buckets(tensors, 16))

    assert buckets == [[tensors[0]], [tensors[1]], [tensors[2]], [tensors[3]]]


def test_pack_and_unpack_mixed_dtypes_and_scalar() -> None:
    source = [
        torch.tensor([1.5, -2.0], dtype=torch.float32),
        torch.tensor([3, 4, 5], dtype=torch.int16),
        torch.tensor(7, dtype=torch.int64),
    ]
    destination = [torch.zeros_like(tensor) for tensor in source]
    buffer = torch.empty(64, dtype=torch.uint8)

    payload = pack_tensors_into_buffer(source, buffer)
    unpack_tensors_from_buffer(payload, destination)

    assert len(payload) == sum(t.numel() * t.element_size() for t in source)
    for actual, expected in zip(destination, source):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class _StateModel:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.tensors


class _WeightState:
    def __init__(self, synced: bool) -> None:
        self.synced = synced

    def weight_synced(self) -> bool:
        return self.synced

    def set_weight_synced(self) -> None:
        self.synced = True


def _r2r_worker(rank: int, tensors: dict[str, torch.Tensor], *, pack: bool):
    return SimpleNamespace(
        rank_in_rollout_repicas=rank,
        replica_name_to_rank={"rollout-0": 0},
        global_commnicator_idex=1,
        config=SimpleNamespace(
            rollout=SimpleNamespace(
                r2r_sync_pack_tensors=pack,
                r2r_sync_bucket_size_bytes=64,
            )
        ),
        rollout=SimpleNamespace(
            get_underlying_model=lambda: _StateModel(tensors),
        ),
    )


def test_production_r2r_packing_round_trips_and_reduces_collectives() -> None:
    source_tensors = {
        "float": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "bf16": torch.arange(8, dtype=torch.bfloat16),
        "noncontiguous": torch.arange(6, dtype=torch.float32).reshape(2, 3).t(),
        "scalar": torch.tensor(17, dtype=torch.int64),
    }
    destination_tensors = {
        name: torch.zeros_like(tensor) for name, tensor in source_tensors.items()
    }
    source = _r2r_worker(0, source_tensors, pack=True)
    destination = _r2r_worker(1, destination_tensors, pack=True)
    payloads: list[torch.Tensor] = []
    receiving = False
    receive_index = 0

    def fake_broadcast(tensor, _src_rank, _comm_idx):
        nonlocal receive_index
        if not receiving:
            payloads.append(tensor.clone())
        else:
            tensor.copy_(payloads[receive_index])
            receive_index += 1

    with (
        patch.object(torch.cuda, "stream", side_effect=lambda _stream: nullcontext()),
        patch.object(weight_sync, "nccl_broadcast", side_effect=fake_broadcast),
        patch.object(weight_sync, "nccl_group", side_effect=lambda _: nullcontext()),
    ):
        weight_sync.do_nccl_broadcast_grouped(source, "rollout-0", None)
        receiving = True
        weight_sync.do_nccl_broadcast_grouped(destination, "rollout-0", None)

    assert len(payloads) == 1
    assert len(payloads) < len(source_tensors)
    for name, expected in source_tensors.items():
        torch.testing.assert_close(destination_tensors[name], expected, rtol=0, atol=0)


def test_r2r_packing_is_disabled_by_default() -> None:
    tensors = {"a": torch.ones(2), "b": torch.ones(2)}
    worker = _r2r_worker(0, tensors, pack=False)

    with (
        patch.object(torch.cuda, "stream", side_effect=lambda _stream: nullcontext()),
        patch.object(weight_sync, "nccl_broadcast") as broadcast,
        patch.object(
            weight_sync, "nccl_group", side_effect=lambda _: nullcontext()
        ) as group,
    ):
        weight_sync.do_nccl_broadcast_grouped(worker, "rollout-0", None)

    assert broadcast.call_count == len(tensors)
    group.assert_called_once_with(1)
    assert not hasattr(worker, "_r2r_sync_packed_buffer")


def test_selected_r2r_default_off_preserves_per_tensor_broadcasts() -> None:
    tensors = {"a": torch.ones(2), "b": torch.ones(2)}
    worker = _r2r_worker(0, tensors, pack=False)

    with (
        patch.object(torch.cuda, "stream", side_effect=lambda _stream: nullcontext()),
        patch.object(weight_sync, "nccl_broadcast") as broadcast,
        patch.object(
            weight_sync, "nccl_group", side_effect=lambda _: nullcontext()
        ) as group,
    ):
        weight_sync.do_nccl_broadcast_tensors(
            worker,
            list(tensors.values()),
            src_rank=0,
            comm_idx=1,
            stream=None,
            group_unpacked=False,
        )

    assert broadcast.call_count == len(tensors)
    group.assert_not_called()
    assert not hasattr(worker, "_r2r_sync_packed_buffer")


class _FakeDTensor:
    def __init__(self, local: torch.Tensor) -> None:
        self.local = local

    def to_local(self) -> torch.Tensor:
        return self.local


def test_r2r_broadcasts_local_fsdp_shards() -> None:
    local_shards = [torch.ones(2), torch.ones(3)]
    fsdp_state = [_FakeDTensor(tensor) for tensor in local_shards]

    for pack in (False, True):
        worker = _r2r_worker(0, {}, pack=pack)
        with (
            patch.object(weight_sync, "DTensor", _FakeDTensor),
            patch.object(
                torch.cuda, "stream", side_effect=lambda _stream: nullcontext()
            ),
            patch.object(weight_sync, "nccl_broadcast") as broadcast,
        ):
            count, nbytes = weight_sync.do_nccl_broadcast_tensors(
                worker,
                fsdp_state,
                src_rank=0,
                comm_idx=1,
                stream=None,
                group_unpacked=False,
            )

        assert count == len(local_shards)
        assert nbytes == sum(tensor.nbytes for tensor in local_shards)
        if pack:
            assert broadcast.call_count == 1
            assert broadcast.call_args.args[0].numel() == nbytes
        else:
            assert all(
                call.args[0] is shard
                for call, shard in zip(broadcast.call_args_list, local_shards)
            )


def test_default_sync_r2r_route_packs_only_trainable_tensors() -> None:
    tensors = {
        "trainable_a": torch.tensor([1.0, 2.0]),
        "frozen": torch.tensor([3.0, 4.0]),
        "trainable_b": torch.tensor([5.0, 6.0]),
    }
    worker = SimpleNamespace(
        replica_name="rollout-0",
        state=_WeightState(synced=True),
        config=SimpleNamespace(
            rollout=SimpleNamespace(
                async_r2r_sync="disabled",
                broadcast_all_params=False,
                r2r_sync_pack_tensors=True,
                r2r_sync_bucket_size_bytes=64,
            )
        ),
        rank_in_rollout_repicas=0,
        global_commnicator_idex=1,
        replica_name_to_rank={"rollout-0": 0, "rollout-1": 1},
        rollout=SimpleNamespace(model_param_map=lambda _mapper: tensors),
        weight_mapper=object(),
        trainable_params={"trainable_a", "trainable_b"},
        prepare_trainable_params=lambda: None,
        inference_stream=None,
        non_trainable_params_received=True,
        current_weight_version=-1,
    )
    command = RolloutToRolloutBroadcastCommand(
        src_replica_name="rollout-0",
        dst_replica_names=["rollout-0", "rollout-1"],
        weight_step=None,
        total_steps=None,
        trainable_only=True,
    )

    with (
        patch.object(torch.cuda, "stream", side_effect=lambda _stream: nullcontext()),
        patch.object(weight_sync, "nccl_broadcast") as broadcast,
        patch.object(
            weight_sync, "nccl_group", side_effect=lambda _: nullcontext()
        ) as group,
    ):
        DisaggregatedRolloutControlWorker.broadcast_to_all_rollout_replica(
            worker, command
        )

    assert broadcast.call_count == 1
    payload = broadcast.call_args.args[0]
    assert payload.dtype == torch.uint8
    assert payload.numel() == 2 * tensors["trainable_a"].nbytes
    group.assert_not_called()
    assert worker.r2r_synced_trainable_params_cnt == 2
