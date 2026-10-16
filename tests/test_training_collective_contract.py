# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Real CPU collectives for default schedules and conditional gradient layouts."""

from datetime import timedelta
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cosmos_rl.policy.trainer import batching
from cosmos_rl.utils import distributed as dist_utils
from test_objective_cohort import CohortComm, CohortTrainer
from test_objective_weighting import _Trainer


def test_unweighted_dynamic_includes_remote_plan_before_training():
    trainer = _Trainer(None, None, False)
    trainer.device = torch.device("cpu")
    trainer.step_expanded_training = Mock(return_value={})

    class RemotePlan:
        replica_name_to_rank = {"local": 0, "remote": 1}

        def wait_comm_ready(self):
            pass

        def world_size(self):
            return 2

        def allreduce(self, send, recv, op):
            if recv.numel() == 14:
                assert op == dist.ReduceOp.MAX  # compatible startup signature
            elif recv.numel() == 4:
                assert op == dist.ReduceOp.MAX
                recv[0] = 2  # The remote replica has two minibatches.
            else:
                assert recv.numel() == 2 and op == dist.ReduceOp.SUM
                recv.add_(torch.tensor([2, 1]))

    batching.run_training_step(
        trainer,
        rollouts=[(1.0, "a")],
        current_step=1,
        inter_policy_nccl=RemotePlan(),
    )
    batch = trainer.step_expanded_training.call_args.args[0]
    assert batch.minibatches == ((1.0,), ())
    assert batch.global_sample_counts == (3, 1)


@pytest.mark.parametrize("fixed", [None, 4])
def test_unweighted_multi_replica_requires_a_gradient_cohort(fixed):
    from types import SimpleNamespace

    trainer = _Trainer(None, fixed, False)
    trainer.config.policy = SimpleNamespace(
        parallelism=SimpleNamespace(n_init_replicas=2)
    )
    trainer.step_expanded_training = Mock(return_value={})
    with pytest.raises(ValueError, match="require inter_policy_nccl"):
        batching.run_training_step(trainer, rollouts=[])
    trainer.step_expanded_training.assert_not_called()


class UnweightedTrainer(CohortTrainer):
    def step_expanded_training(self, batch, **kwargs):
        assert not batch.objective_windows
        self.windows = (len(batch.minibatches), batch.mu_iterations)
        for _ in range(batch.mu_iterations):
            for slot, samples in enumerate(batch.minibatches):
                self.optimizer.zero_grad()
                values = torch.tensor(samples, dtype=torch.float64).reshape(-1, 1)
                losses = (self.model(values).flatten() - 1).square()
                # The fixed schedule leaves normalization to the trainer. This
                # test's fixed contract averages rank-local SUMs over four ranks.
                scale = (
                    batch.mean_gradient_scale(slot, 4)
                    if batch.global_sample_counts is not None
                    else 1
                )
                (losses.sum() * scale).backward()
                for parameter in self.model.parameters():
                    dist.all_reduce(parameter.grad, group=self.cohort_group)
                    parameter.grad.div_(2)
                self.optimizer.step()
                self.scheduler.step()
                self.updates += 1
                self.forwards += 1
        return {}


def _unweighted_reference(data, fixed):
    parameter = torch.nn.Parameter(torch.tensor([[0.5]], dtype=torch.float64))
    optimizer = torch.optim.SGD([parameter], lr=0.1, momentum=0.9, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    width = fixed or max((len(local) + 1) // 2 for local in data)
    slots = [
        [value for local in data for value, _ in local[2 * slot : 2 * slot + 2]]
        for slot in range(width)
    ]
    for _ in range(2):
        for samples in slots:
            if not samples and fixed is None:
                continue
            optimizer.zero_grad()
            values = torch.tensor(samples, dtype=torch.float64)
            loss = (parameter.flatten()[0] * values - 1).square().sum()
            (loss / (4 if fixed else len(samples))).backward()
            optimizer.step()
            scheduler.step()
    return parameter, optimizer, scheduler


def _schedule_worker(rank, rendezvous):
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
            for fixed in (None, 4):
                for prefetch in (False, True):
                    for case in ("uneven", "empty_replica", "all_empty", "recoverable"):
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
                        trainer = UnweightedTrainer(
                            None, fixed, prefetch, local_group, cohort_group
                        )
                        comm = CohortComm(cohort_group)
                        if case == "recoverable" and rank < 2:

                            def unavailable(_):
                                raise batching.RecoverablePreparationError(
                                    "missing episode"
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
                            expected, optimizer, scheduler = _unweighted_reference(
                                data, fixed
                            )
                            weight = next(trainer.model.parameters())
                            torch.testing.assert_close(
                                weight, expected, atol=1e-12, rtol=1e-12
                            )
                            assert (
                                trainer.scheduler.state_dict() == scheduler.state_dict()
                            )
                            if trainer.updates:
                                torch.testing.assert_close(
                                    trainer.optimizer.state[weight]["momentum_buffer"],
                                    optimizer.state[expected]["momentum_buffer"],
                                    atol=1e-12,
                                    rtol=1e-12,
                                )
                            observations = [None] * 4
                            dist.all_gather_object(
                                observations, (trainer.windows, trainer.updates)
                            )
                            assert all(
                                value == observations[0] for value in observations
                            )
                            # Fixed unweighted scheduling adds one upfront
                            # signature, with no per-update metadata exchange.
                            assert comm.calls == (
                                1 if fixed else (2 if case == "all_empty" else 3)
                            )
                            if fixed:
                                batching.run_training_step(
                                    trainer,
                                    rollouts=[],
                                    current_step=2,
                                    inter_policy_nccl=comm,
                                )
                                assert comm.calls == 1
                        finally:
                            trainer.data_packer.shutdown_prefetch()
    finally:
        dist.destroy_process_group()


def test_unweighted_full_cohort_fixed_and_dynamic_schedules(tmp_path):
    mp.spawn(_schedule_worker, args=(str(tmp_path / "schedule"),), nprocs=4, join=True)


class GradientComm:
    def wait_comm_ready(self):
        pass

    def allreduce(self, send, recv, op, timeout_ms=None):
        assert send is recv
        dist.all_reduce(recv, op=dist.ReduceOp.SUM)
        if op == dist.ReduceOp.AVG:
            recv.div_(dist.get_world_size())
        else:
            assert op == dist.ReduceOp.SUM


def _gradient_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        with patch.object(torch.Tensor, "cuda", lambda tensor: tensor):
            for mixed_dtype in (False, True):
                parameters = [
                    torch.nn.Parameter(torch.tensor([1.0])),
                    torch.nn.Parameter(
                        torch.tensor(
                            [2.0], dtype=torch.float64 if mixed_dtype else torch.float32
                        )
                    ),
                    torch.nn.Parameter(torch.tensor([3.0])),
                    torch.nn.Parameter(torch.tensor([4.0]), requires_grad=False),
                ]
                optimizer = torch.optim.SGD(
                    parameters, lr=0.1, momentum=0.9, weight_decay=0.1
                )
                # A globally unused parameter with OLD optimizer state must
                # still skip momentum and weight decay, not become a zero grad.
                optimizer.state[parameters[2]]["momentum_buffer"] = torch.tensor([7.0])
                parameters[rank].grad = torch.full_like(
                    parameters[rank], 3.0 if rank == 0 else 5.0
                )
                dist_utils.gradient_reduce_across_dp_replicas_(
                    parameters, GradientComm()
                )
                assert parameters[0].grad.item() == 1.5
                assert parameters[1].grad.item() == 2.5
                assert parameters[2].grad is parameters[3].grad is None
                optimizer.step()
                assert parameters[2].item() == 3.0
                assert optimizer.state[parameters[2]]["momentum_buffer"].item() == 7.0
                before = [p.detach().clone() for p in parameters]
                optimizer.zero_grad(set_to_none=True)
                dist_utils.gradient_reduce_across_dp_replicas_(
                    parameters, GradientComm()
                )
                optimizer.step()
                for parameter, value in zip(parameters, before):
                    assert parameter.grad is None
                    torch.testing.assert_close(parameter, value, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_conditional_parameter_layout_and_globally_unused_optimizer_state(tmp_path):
    mp.spawn(_gradient_worker, args=(str(tmp_path / "gradients"),), nprocs=2, join=True)
