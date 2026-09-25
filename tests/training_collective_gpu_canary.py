# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Native HA training-cohort canary and portable failure injection.

torchrun --standalone --nproc-per-node=4 tests/training_collective_gpu_canary.py healthy
torchrun --standalone --nproc-per-node=2 tests/training_collective_gpu_canary.py partial
torchrun --standalone --nproc-per-node=2 tests/training_collective_gpu_canary.py missing-peer
"""

import argparse
import os
import time
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor

from cosmos_rl.policy.trainer import batching
from cosmos_rl.utils import distributed as dist_utils
from cosmos_rl.utils.pynccl import create_nccl_comm, create_nccl_uid, nccl_abort
from test_objective_cohort import _reference
from test_objective_weighting import _Trainer
from test_p2p_batching_e2e import _communicator
from test_training_collective_contract import _unweighted_reference


class Trainer(_Trainer):
    def __init__(self, weighting, fixed, prefetch, device, dp_group, comm):
        super().__init__(weighting, fixed, prefetch)
        self.device = device
        self.model = DistributedDataParallel(
            self.model.to(device=device, dtype=torch.float32),
            device_ids=[device.index],
            process_group=dp_group,
        )
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.1
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.9)
        self.comm = comm
        self.forwards = 0

    def step_expanded_training(self, batch, **kwargs):
        for _ in range(batch.mu_iterations):
            windows = (
                [
                    (window.slots, window)
                    for window in batch.objective_windows
                    if window.global_count
                ]
                if batch.objective_windows
                else [((slot,), None) for slot in range(len(batch.minibatches))]
            )
            for slots, window in windows:
                self.optimizer.zero_grad()
                offset = 0
                for slot in slots:
                    values = torch.tensor(
                        batch.minibatches[slot], device=self.device
                    ).reshape(-1, 1)
                    losses = (self.model(values).flatten() - 1).square()
                    if window is not None:
                        loss = window.loss(losses, start=offset)
                    else:
                        scale = (
                            batch.mean_gradient_scale(slot, 4)
                            if batch.global_sample_counts is not None
                            else 1
                        )
                        loss = losses.sum() * scale
                    loss.backward()
                    offset += len(values)
                    self.forwards += 1
                dist_utils.gradient_reduce_across_dp_replicas_(
                    self.model.parameters(), self.comm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.updates += 1
        return {}


def _native_comm(rank, group, source):
    uid = [create_nccl_uid() if rank == source else None]
    dist.broadcast_object_list(uid, src=source, group=group)
    idx = create_nccl_comm(uid[0], dist.get_rank(group), 2, timeout_ms=30_000)
    comm = _communicator(dist.get_rank(group), idx)
    comm.global_rank = rank
    comm.max_retry = 3
    return comm


def healthy(rank, device):
    assert dist.get_world_size() == 4
    local_gloo = [dist.new_group(ranks, backend="gloo") for ranks in ([0, 1], [2, 3])]
    local_nccl = [dist.new_group(ranks, backend="nccl") for ranks in ([0, 1], [2, 3])]
    cohorts = [dist.new_group(ranks, backend="gloo") for ranks in ([0, 2], [1, 3])]
    dp_group = local_nccl[rank // 2]
    comm = _native_comm(rank, cohorts[rank % 2], rank % 2)
    cases = 0
    try:

        def local_gather(value):
            values = [None, None]
            dist.all_gather_object(values, value, group=local_gloo[rank // 2])
            return values

        with patch.object(batching, "_gather", local_gather):
            for weighting in (None, "sample", "episode"):
                for fixed in (None, 4):
                    for prefetch in (False, True):
                        for case in (
                            "uneven",
                            "empty_replica",
                            "all_empty",
                            "recoverable",
                        ):
                            data = [
                                [(1.0, "a")],
                                [],
                                [(2.0, "a"), (3.0, "a"), (4.0, "a")],
                                [(5.0, "b")],
                            ]
                            if case in ("empty_replica", "recoverable"):
                                data[0] = data[1] = []
                            if case == "all_empty":
                                data = [[], [], [], []]
                            trainer = Trainer(
                                weighting, fixed, prefetch, device, dp_group, comm
                            )
                            if case == "recoverable" and rank < 2:

                                def unavailable(_):
                                    raise batching.RecoverablePreparationError(
                                        "injected unavailable data"
                                    )

                                trainer.prepare_training_batch = unavailable
                            try:
                                local = data[rank]
                                assert (
                                    batching.prefetch_training_batch(trainer, local)
                                    == prefetch
                                )
                                batching.run_training_step(
                                    trainer,
                                    rollouts=local,
                                    current_step=1,
                                    inter_policy_nccl=comm,
                                )
                                expected, optimizer, scheduler = (
                                    _reference(data, weighting)
                                    if weighting
                                    else _unweighted_reference(data, fixed)
                                )
                                weight = next(trainer.model.parameters())
                                torch.testing.assert_close(
                                    weight.detach().cpu().double(),
                                    expected,
                                    atol=2e-6,
                                    rtol=2e-6,
                                )
                                assert (
                                    trainer.scheduler.state_dict()
                                    == scheduler.state_dict()
                                )
                                if trainer.updates:
                                    torch.testing.assert_close(
                                        trainer.optimizer.state[weight][
                                            "momentum_buffer"
                                        ]
                                        .cpu()
                                        .double(),
                                        optimizer.state[expected]["momentum_buffer"],
                                        atol=2e-6,
                                        rtol=2e-6,
                                    )
                                observations = [None] * 4
                                dist.all_gather_object(
                                    observations, (trainer.forwards, trainer.updates)
                                )
                                assert all(
                                    value == observations[0] for value in observations
                                )
                                cases += 1
                            finally:
                                trainer.data_packer.shutdown_prefetch()

        # Same parameter order, opposite local unused sets, mixed dtypes.
        parameters = [
            torch.nn.Parameter(torch.tensor([1.0], device=device)),
            torch.nn.Parameter(torch.tensor([2.0], device=device, dtype=torch.float64)),
            torch.nn.Parameter(torch.tensor([3.0], device=device)),
        ]
        parameters[rank // 2].grad = torch.full_like(
            parameters[rank // 2], 3.0 if rank < 2 else 5.0
        )
        dist_utils.gradient_reduce_across_dp_replicas_(parameters, comm)
        assert parameters[0].grad.item() == 1.5
        assert parameters[1].grad.item() == 2.5
        assert parameters[2].grad is None

        # Replicated DTensor placeholders retain their layout and receive the
        # remote gradient. This is not a full FSDP model-training claim.
        mesh = DeviceMesh.from_group(dp_group, "cuda", mesh_dim_names=("dp",))
        parameter = torch.nn.Parameter(
            distribute_tensor(torch.ones(2, device=device), mesh, [Replicate()])
        )
        if rank < 2:
            parameter.grad = DTensor.from_local(
                torch.full((2,), 3.0, device=device), mesh, [Replicate()]
            )
        dist_utils.gradient_reduce_across_dp_replicas_([parameter], comm)
        assert isinstance(parameter.grad, DTensor)
        assert parameter.grad.placements == parameter.placements
        torch.testing.assert_close(
            parameter.grad.to_local(), torch.full((2,), 1.5, device=device)
        )
        torch.cuda.synchronize()
        print(
            f"TRAINING_COHORT_PASS rank={rank} cases={cases} conditional_gradients=True dtensor=True",
            flush=True,
        )
        assert cases == 48
    finally:
        nccl_abort(comm.comm_idx)


def fault(rank, device, mode):
    assert dist.get_world_size() == 2
    comm = _native_comm(rank, dist.group.WORLD, 0)
    warm = torch.ones(1, device=device)
    comm.allreduce(warm, warm, dist.ReduceOp.SUM)
    assert warm.item() == 2
    weight = torch.nn.Parameter(torch.tensor([2.0], device=device))
    weight.grad = torch.ones_like(weight)
    optimizer = torch.optim.SGD([weight], lr=0.1, momentum=0.9)
    raw = dist_utils.nccl_allreduce
    calls = 0

    def injected(**kwargs):
        nonlocal calls
        calls += 1
        print(f"FAULT_REACHED rank={rank} mode={mode} native_call={calls}", flush=True)
        if mode == "missing-peer" and rank == 1:
            # The other rank reaches a warm native operation with no peer call.
            time.sleep(4)
        raw(**kwargs)
        if mode == "partial":
            torch.cuda.current_stream().synchronize()
            raise OSError("injected lost completion after a real device write")

    start = time.monotonic()
    try:
        with patch.object(dist_utils, "nccl_allreduce", injected):
            try:
                comm.allreduce(
                    weight.grad, weight.grad, dist.ReduceOp.SUM, timeout_ms=1500
                )
                optimizer.step()
            except dist_utils.CollectiveOperationError:
                assert calls == 1
                assert not comm.is_ready()
                assert not optimizer.state
                assert weight.item() == 2
                assert dist_utils._FAILED_COLLECTIVE_BUFFERS
                assert time.monotonic() - start < 20
                print(
                    f"TERMINAL_FAULT_PASS rank={rank} mode={mode} replay=False optimizer_commit=False",
                    flush=True,
                )
            else:
                raise AssertionError("injection did not fail the operation")
    finally:
        nccl_abort(comm.comm_idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("healthy", "partial", "missing-peer"))
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    assert torch.cuda.is_available()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        if args.mode == "healthy":
            healthy(rank, device)
        else:
            fault(rank, device, args.mode)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
