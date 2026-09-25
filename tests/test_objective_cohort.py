# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Two policy replicas x two DP ranks, with real independent collectives."""

from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from cosmos_rl.policy.trainer import batching
from test_objective_weighting import _Trainer


class CohortComm:
    def __init__(self, group):
        self.group = group
        self.replica_name_to_rank = {"replica0": 0, "replica1": 1}
        self.calls = 0

    def wait_comm_ready(self):
        pass

    def world_size(self):
        return 2

    def allreduce(self, send, recv, op):
        assert send is recv
        self.calls += 1
        dist.all_reduce(recv, op=op, group=self.group)


class CohortTrainer(_Trainer):
    def __init__(self, weighting, fixed, prefetch, dp_group, cohort_group):
        super().__init__(weighting, fixed, prefetch)
        self.device = torch.device("cpu")
        self.model = DistributedDataParallel(self.model, process_group=dp_group)
        self.cohort_group = cohort_group
        self.windows = []
        self.forwards = 0

    def step_expanded_training(self, batch, **kwargs):
        self.windows = [
            (window.slots, window.global_count) for window in batch.objective_windows
        ]
        for _ in range(batch.mu_iterations):
            for window in batch.objective_windows:
                if not window.global_count:
                    continue
                assert window.gradient_divisor == 4
                self.optimizer.zero_grad()
                offset = 0
                for slot in window.slots:
                    values = torch.tensor(
                        batch.minibatches[slot], dtype=torch.float64
                    ).reshape(-1, 1)
                    loss = (self.model(values).flatten() - 1).square()
                    window.loss(loss, start=offset).backward()
                    offset += len(values)
                    self.forwards += 1
                for parameter in self.model.parameters():
                    dist.all_reduce(parameter.grad, group=self.cohort_group)
                    parameter.grad.div_(2)
                self.optimizer.step()
                self.scheduler.step()
                self.updates += 1
        return {}


def _reference(data, weighting):
    parameter = torch.nn.Parameter(torch.tensor([[0.5]], dtype=torch.float64))
    optimizer = torch.optim.SGD([parameter], lr=0.1, momentum=0.9, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    groups = []
    for local in data:
        episodes = {}
        for value, identity in local:
            episodes.setdefault(identity, []).append(value)
        groups.extend(
            list(episodes.values())
            if weighting == "episode"
            else [[value] for values in episodes.values() for value in values]
        )
    for _ in range(2 if groups else 0):
        optimizer.zero_grad()
        torch.stack(
            [
                (
                    (
                        parameter.flatten()[0]
                        * torch.tensor(values, dtype=torch.float64)
                        - 1
                    ).square()
                ).mean()
                for values in groups
            ]
        ).mean().backward()
        optimizer.step()
        scheduler.step()
    return parameter, optimizer, scheduler


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=60),
    )
    try:
        groups = [dist.new_group(ranks) for ranks in ([0, 1], [2, 3], [0, 2], [1, 3])]
        local_group, cohort_group = groups[rank // 2], groups[2 + rank % 2]

        def local_gather(value):
            values = [None, None]
            dist.all_gather_object(values, value, group=local_group)
            return values

        with patch.object(batching, "_gather", local_gather):
            for weighting in ("sample", "episode"):
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
                            trainer = CohortTrainer(
                                weighting, fixed, prefetch, local_group, cohort_group
                            )
                            comm = CohortComm(cohort_group)
                            if case == "recoverable" and rank < 2:

                                def unavailable(_):
                                    raise batching.RecoverablePreparationError(
                                        "unavailable replica data"
                                    )

                                trainer.prepare_training_batch = unavailable
                            try:
                                local = data[rank]
                                assert (
                                    batching.prefetch_training_batch(trainer, local)
                                    == prefetch
                                )
                                report = batching.run_training_step(
                                    trainer,
                                    rollouts=local,
                                    current_step=1,
                                    inter_policy_nccl=comm,
                                )
                                expected, optimizer, scheduler = _reference(
                                    data, weighting
                                )
                                torch.testing.assert_close(
                                    next(trainer.model.parameters()),
                                    expected,
                                    rtol=1e-12,
                                    atol=1e-12,
                                )
                                assert (
                                    trainer.scheduler.state_dict()
                                    == scheduler.state_dict()
                                )
                                assert trainer.updates == (
                                    0 if case == "all_empty" else 2
                                )
                                assert report["batching/skipped_update"] == int(
                                    case == "all_empty"
                                )
                                if trainer.updates:
                                    torch.testing.assert_close(
                                        trainer.optimizer.state[
                                            next(trainer.model.parameters())
                                        ]["momentum_buffer"],
                                        optimizer.state[expected]["momentum_buffer"],
                                        rtol=1e-12,
                                        atol=1e-12,
                                    )
                                    assert (
                                        trainer.forwards > 0
                                    )  # Even on the empty replica.
                                observations = [None] * 4
                                dist.all_gather_object(
                                    observations, (trainer.windows, trainer.forwards)
                                )
                                assert all(
                                    value == observations[0] for value in observations
                                )
                                # One startup signature + two per-update metadata collectives,
                                # with no count payload needed when dynamic width is zero.
                                assert comm.calls == (
                                    2 if case == "all_empty" and fixed is None else 3
                                )
                                previous_calls = comm.calls
                                previous_forwards = trainer.forwards
                                previous_scheduler = trainer.scheduler.last_epoch
                                batching.run_training_step(
                                    trainer,
                                    rollouts=[],
                                    current_step=1,
                                    inter_policy_nccl=comm,
                                )
                                torch.testing.assert_close(
                                    next(trainer.model.parameters()), expected
                                )
                                assert trainer.forwards == previous_forwards
                                assert (
                                    trainer.scheduler.last_epoch == previous_scheduler
                                )
                                assert comm.calls == previous_calls + (
                                    1 if fixed is None else 2
                                )
                            finally:
                                trainer.data_packer.shutdown_prefetch()
            for failure in (
                "configuration",
                "step",
                "preparation",
                "membership",
                "missing_comm",
            ):
                trainer = CohortTrainer(
                    "sample", None, False, local_group, cohort_group
                )
                comm = CohortComm(cohort_group)
                if failure == "configuration" and rank >= 2:
                    trainer.config.train.train_policy.mu_iterations = 3
                if failure == "preparation" and rank >= 2:

                    def broken(_):
                        raise ValueError("broken preparation")

                    trainer.prepare_training_batch = broken
                if failure in ("membership", "missing_comm"):
                    batching.run_training_step(
                        trainer, rollouts=[], current_step=0, inter_policy_nccl=comm
                    )
                    comm.replica_name_to_rank = {"replica0": 0, "replacement": 1}
                with pytest.raises(ValueError):
                    batching.run_training_step(
                        trainer,
                        rollouts=[(1.0, "a")],
                        current_step=(rank // 2 if failure == "step" else 1),
                        inter_policy_nccl=None if failure == "missing_comm" else comm,
                    )
                assert trainer.forwards == 0
    finally:
        dist.destroy_process_group()


def test_cross_replica_objectives_and_collective_schedule(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "cohort"),), nprocs=4, join=True)


def test_multi_replica_cannot_silently_omit_gradient_communicator():
    from types import SimpleNamespace

    trainer = _Trainer("sample", None, False)
    trainer.config.policy = SimpleNamespace(
        parallelism=SimpleNamespace(n_init_replicas=2)
    )
    with pytest.raises(ValueError, match="require inter_policy_nccl"):
        batching.run_training_step(trainer, rollouts=[(1.0, "a")])
