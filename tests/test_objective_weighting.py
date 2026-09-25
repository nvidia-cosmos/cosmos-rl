# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from cosmos_rl.policy.trainer.objectives import (
    ObjectiveWindow,
    masked_sample_means,
    vla_objective,
)
from cosmos_rl.policy.trainer.batching import (
    ExpandedSampleBatching,
    ExpandedTrainingBatch,
    run_training_step,
    prefetch_training_batch,
    RecoverablePreparationError,
)
from cosmos_rl.utils.payload_transport.prefetch_mixin import PrefetchDataPackerMixin


@pytest.mark.parametrize("weighting,expected", [("sample", 3.0), ("episode", 4.0)])
def test_explicit_objectives(weighting, expected):
    plan = ObjectiveWindow.prepare(["short", "long", "long", "long"], weighting)
    losses = torch.tensor([6.0, 2.0, 2.0, 2.0], requires_grad=True)
    whole = plan.loss(losses, global_count=plan.count)
    split = sum(
        plan.loss(losses[i : i + 1], start=i, global_count=plan.count) for i in range(4)
    )
    torch.testing.assert_close(whole, split)
    assert whole.item() == pytest.approx(expected)
    whole.backward()
    assert losses.grad.sum().item() == pytest.approx(1.0)


def test_masked_samples_and_empty_graph():
    x = torch.tensor([[2.0, 4.0], [9.0, 10.0]], requires_grad=True)
    means = masked_sample_means(x, torch.tensor([[True, True], [False, False]]))
    torch.testing.assert_close(means, torch.tensor([3.0]))
    empty = masked_sample_means(x, torch.zeros_like(x, dtype=torch.bool))
    ObjectiveWindow.prepare([], "episode").loss(empty, global_count=0).backward()
    torch.testing.assert_close(x.grad, torch.zeros_like(x))


@pytest.mark.parametrize("divisor", [1, 2, 6])
def test_explicit_gradient_reduction_compensation(divisor):
    weight = torch.tensor(2.0, requires_grad=True)
    objective = ObjectiveWindow.prepare([0, 0, 1], "episode")
    objective.loss(
        weight * torch.tensor([1.0, 3.0, 8.0]), global_count=4, gradient_divisor=divisor
    ).backward()
    assert weight.grad.item() / divisor == pytest.approx((2.0 + 8.0) / 4.0)


@pytest.mark.parametrize("weighting", ["sample", "episode"])
def test_accumulation_has_multiple_windows_and_partial_tail(weighting):
    trainer = _Trainer(weighting, None, False)
    trainer.batching_contract = ExpandedSampleBatching(
        partial_tail="include", objective_weighting=weighting, accumulation_steps=2
    )
    data = [(1.0, "a"), (2.0, "a"), (3.0, "b"), (4.0, "b"), (5.0, "c")]
    observed = []
    trainer.step_expanded_training = (
        lambda batch, **kwargs: observed.append(batch) or {}
    )
    run_training_step(trainer, rollouts=data)
    windows = observed[0].objective_windows
    assert [window.slots for window in windows] == [(0, 1), (2,)]
    assert [window.global_count for window in windows] == (
        [4, 1] if weighting == "sample" else [2, 1]
    )


def test_episode_cannot_cross_optimizer_updates():
    trainer = _Trainer("episode", None, False)
    trainer.batching_contract = ExpandedSampleBatching(
        partial_tail="include", objective_weighting="episode", accumulation_steps=1
    )
    with pytest.raises(ValueError, match="one optimizer"):
        run_training_step(trainer, rollouts=[(1.0, "a"), (2.0, "a"), (3.0, "a")])


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("fixed", [None, 4])
def test_recoverable_preparation_skips_weighted_optimizer(prefetch, fixed):
    trainer = _Trainer("episode", fixed, prefetch)

    def unavailable(_):
        raise RecoverablePreparationError("unavailable episode")

    trainer.prepare_training_batch = unavailable
    try:
        data = [(1.0, "a")]
        assert prefetch_training_batch(trainer, data) == prefetch
        report = run_training_step(trainer, rollouts=data)
        assert report["batching/skipped_update"] == 1
        assert report["batching/preparation_failed"] == 1
        assert trainer.updates == 0
        assert not trainer.optimizer.state
    finally:
        trainer.data_packer.shutdown_prefetch()


class _Packer(PrefetchDataPackerMixin):
    def _should_intercept(self, value):
        return False


class _Trainer:
    def __init__(self, weighting, fixed, prefetch, distributed=False):
        self.batching_contract = ExpandedSampleBatching(
            partial_tail="include",
            fixed_minibatches=fixed,
            objective_weighting=weighting,
            accumulation_steps=4,
        )
        self.config = SimpleNamespace(
            train=SimpleNamespace(
                train_policy=SimpleNamespace(mini_batch=2, mu_iterations=2)
            )
        )
        self.model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
        self.model.weight.data.fill_(0.5)
        if distributed:
            self.model = DistributedDataParallel(self.model)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.1
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.9)
        self.data_packer = _Packer()
        if prefetch:
            self.data_packer._setup_prefetch(prefetch_timeout=10)
        self.updates = 0

    def prepare_training_batch(self, rollouts):
        return ExpandedTrainingBatch(
            tuple(
                tuple(value for value, _ in rollouts[i : i + 2])
                for i in range(0, len(rollouts), 2)
            ),
            episode_ids=tuple(
                tuple(identity for _, identity in rollouts[i : i + 2])
                for i in range(0, len(rollouts), 2)
            ),
        )

    def step_expanded_training(self, batch, **kwargs):
        for _ in range(batch.mu_iterations):
            for window in batch.objective_windows:
                if not window.global_count:
                    continue
                self.optimizer.zero_grad()
                offset = 0
                for slot in window.slots:
                    x = torch.tensor(
                        batch.minibatches[slot], dtype=torch.float64
                    ).reshape(-1, 1)
                    losses = (self.model(x).flatten() - 1).square()
                    window.loss(
                        losses,
                        start=offset,
                        gradient_divisor=dist.get_world_size()
                        if dist.is_initialized()
                        else 1,
                    ).backward()
                    offset += len(x)
                self.optimizer.step()
                self.scheduler.step()
                self.updates += 1
        return {}


def _distributed(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=40),
    )
    try:
        for dp in (True, False):
            for weighting in ("sample", "episode"):
                trainer = SimpleNamespace(
                    config=SimpleNamespace(
                        vla=SimpleNamespace(objective_weighting=weighting)
                    ),
                    device=torch.device("cpu"),
                    parallel_dims=SimpleNamespace(
                        dp_enabled=dp,
                        mesh={
                            "dp": SimpleNamespace(get_group=lambda: dist.group.WORLD)
                        },
                    ),
                )
                comm = SimpleNamespace(
                    wait_comm_ready=lambda: None,
                    world_size=lambda: 1 if dp else 2,
                    allreduce=lambda send, recv, op: None
                    if dp
                    else dist.all_reduce(recv, op=op),
                )
                masks = (
                    [torch.ones(3, 1), torch.ones(1, 1)]
                    if rank == 0
                    else [torch.zeros(1, 1), torch.zeros(1, 1)]
                )
                objective, count, divisor, chunks = vla_objective(
                    trainer, ({"logprob_masks": mask} for mask in masks), comm
                )
                assert count == (4 if weighting == "sample" else 2)
                assert divisor == 2
                assert chunks == (3 if dp or rank == 0 else 1)
                assert objective.count == (count if rank == 0 else 0)
        for weighting in ("sample", "episode"):
            for fixed in (None, 4):
                for prefetch in (False, True):
                    for case in ("uneven", "empty_rank", "all_empty", "filtered"):
                        data = [
                            [(1.0, "a"), (2.0, "a"), (3.0, "a"), (5.0, "b")],
                            [(4.0, "a")],
                        ]
                        if case == "empty_rank":
                            data[1] = []
                        elif case == "all_empty":
                            data = [[], []]
                        elif case == "filtered":
                            data[0][1] = (float("nan"), "a")
                        trainer = _Trainer(weighting, fixed, prefetch, distributed=True)
                        try:
                            local = data[rank]
                            assert prefetch_training_batch(trainer, local) == prefetch
                            run_training_step(trainer, rollouts=local)
                            reference = torch.nn.Parameter(
                                torch.tensor([[0.5]], dtype=torch.float64)
                            )
                            optimizer = torch.optim.SGD(
                                [reference], lr=0.1, momentum=0.9, weight_decay=0.1
                            )
                            scheduler = torch.optim.lr_scheduler.StepLR(
                                optimizer, 1, gamma=0.9
                            )
                            # Independent explicit mean of scalar samples or complete episode means.
                            groups = []
                            for rank_data in data:
                                episodes = {}
                                for value, identity in rank_data:
                                    if value == value:
                                        episodes.setdefault(identity, []).append(value)
                                groups.extend(
                                    list(episodes.values())
                                    if weighting == "episode"
                                    else [
                                        [x]
                                        for values in episodes.values()
                                        for x in values
                                    ]
                                )
                            for _ in range(2 if groups else 0):
                                optimizer.zero_grad()
                                loss = torch.stack(
                                    [
                                        (
                                            (
                                                reference.flatten()[0]
                                                * torch.tensor(g, dtype=torch.float64)
                                                - 1
                                            )
                                            ** 2
                                        ).mean()
                                        for g in groups
                                    ]
                                ).mean()
                                loss.backward()
                                optimizer.step()
                                scheduler.step()
                            torch.testing.assert_close(
                                next(trainer.model.parameters()),
                                reference,
                                rtol=1e-12,
                                atol=1e-12,
                            )
                            assert (
                                trainer.scheduler.state_dict() == scheduler.state_dict()
                            )
                            assert trainer.updates == (2 if groups else 0)
                            if groups:
                                torch.testing.assert_close(
                                    trainer.optimizer.state[
                                        next(trainer.model.parameters())
                                    ]["momentum_buffer"],
                                    optimizer.state[reference]["momentum_buffer"],
                                    rtol=1e-12,
                                    atol=1e-12,
                                )
                        finally:
                            trainer.data_packer.shutdown_prefetch()
    finally:
        dist.destroy_process_group()


def test_distributed_objectives_prefetch_and_accumulation(tmp_path):
    mp.spawn(_distributed, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
