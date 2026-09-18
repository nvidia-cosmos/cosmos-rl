# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import tempfile
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cosmos_rl.policy.trainer.batching import (
    ExpandedSampleBatching,
    ExpandedTrainingBatch,
    RecoverablePreparationError,
    run_training_step,
)


def trainer_for(minibatches, *, tail="include"):
    return SimpleNamespace(
        batching_contract=ExpandedSampleBatching(partial_tail=tail),
        config=SimpleNamespace(
            train=SimpleNamespace(
                train_policy=SimpleNamespace(mini_batch=2, mu_iterations=2)
            )
        ),
        prepare_training_batch=Mock(
            return_value=ExpandedTrainingBatch(tuple(minibatches))
        ),
        step_expanded_training=Mock(return_value={"updates": len(minibatches)}),
        step_training=Mock(
            side_effect=AssertionError("must consume validated samples")
        ),
    )


def test_fixed_trainer_keeps_existing_entrypoint():
    trainer = SimpleNamespace(step_training=Mock(return_value={"loss": 2}))
    before = Mock()
    assert run_training_step(trainer, before_step=before, rollouts=[1]) == {"loss": 2}
    before.assert_called_once()
    trainer.step_training.assert_called_once_with(rollouts=[1])


def test_variable_episode_expansion_and_partial_tail():
    trainer = trainer_for([[1.0, 2.0], [3.0]])
    assert (
        run_training_step(trainer, rollouts=["episode"], current_step=7)["updates"] == 2
    )
    trainer.step_expanded_training.assert_called_once_with(
        ExpandedTrainingBatch(((1.0, 2.0), (3.0,)), (2, 1), 2), current_step=7
    )


@pytest.mark.parametrize(
    "batches,tail",
    [
        ([], "include"),
        ([[]], "include"),
        ([[1]], "reject"),
        ([[float("nan"), 1]], "include"),
        ([[1], [2, 3]], "include"),
    ],
)
def test_empty_and_nonfinite_data_produces_a_recoverable_plan(batches, tail):
    trainer = trainer_for(batches, tail=tail)
    scheduler = Mock()
    report = run_training_step(trainer, before_step=scheduler, rollouts=["episode"])
    batch = trainer.step_expanded_training.call_args.args[0]
    assert all(batch.global_sample_counts)
    assert scheduler.call_count == bool(batch.minibatches)
    assert report["batching/skipped_update"] == (not batch.minibatches)


def test_expansion_failure_is_recoverable():
    trainer = trainer_for([[1, 2]])
    trainer.prepare_training_batch.side_effect = RecoverablePreparationError(
        "missing completion"
    )
    before = Mock()
    report = run_training_step(
        trainer, before_step=before, rollouts=["missing"], do_save_checkpoint=True
    )
    assert report["batching/preparation_failed"] == 1
    assert report["batching/skipped_update"] == 1
    before.assert_not_called()
    trainer.step_expanded_training.assert_called_once_with(
        ExpandedTrainingBatch((), (), 2), do_save_checkpoint=True
    )


def test_expanded_training_matches_explicit_sample_updates():
    initial = torch.tensor([0.5])
    actual = torch.nn.Parameter(initial.clone())
    expected = torch.nn.Parameter(initial.clone())
    opt = torch.optim.SGD([actual], lr=0.1, momentum=0.9)
    reference = torch.optim.SGD([expected], lr=0.1, momentum=0.9)
    batches = [[torch.tensor(1.0), torch.tensor(2.0)], [torch.tensor(3.0)]]
    trainer = trainer_for(batches)

    def update(parameter, optimizer, samples):
        optimizer.zero_grad()
        loss = ((parameter * torch.stack(samples) - 1) ** 2).mean()
        loss.backward()
        optimizer.step()

    def step(batch, **kwargs):
        for samples in batch.minibatches:
            update(actual, opt, samples)
        return {"updates": len(batch.minibatches)}

    trainer.step_expanded_training = step
    assert run_training_step(trainer, rollouts=["episode"])["updates"] == 2
    for samples in batches:
        update(expected, reference, samples)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        opt.state[actual]["momentum_buffer"],
        reference.state[expected]["momentum_buffer"],
        rtol=0,
        atol=0,
    )


def _distributed_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        for scenario in (
            "one_empty",
            "all_empty",
            "nonfinite",
            "unequal_steps",
            "expansion_error",
            "healthy",
            "mu_mismatch",
            "programming_error",
        ):
            batches = [[1.0, 2.0], [3.0]]
            if scenario == "all_empty" or (scenario == "one_empty" and rank == 1):
                batches = []
            elif scenario == "nonfinite" and rank == 1:
                batches = [[float("inf"), 2.0], [3.0]]
            elif scenario == "unequal_steps" and rank == 1:
                batches = [[1.0, 2.0]]
            trainer = trainer_for(batches)
            if scenario == "expansion_error" and rank == 1:
                trainer.prepare_training_batch.side_effect = (
                    RecoverablePreparationError("missing rollout")
                )
            scheduler = Mock()
            if scenario == "mu_mismatch" and rank == 1:
                trainer.config.train.train_policy.mu_iterations = 3
            if scenario == "programming_error" and rank == 1:
                trainer.prepare_training_batch.side_effect = RuntimeError("bug")
            if scenario in ("mu_mismatch", "programming_error"):
                with pytest.raises(ValueError):
                    run_training_step(
                        trainer, before_step=scheduler, rollouts=["episode"]
                    )
                trainer.step_expanded_training.assert_not_called()
                scheduler.assert_not_called()
                dist.barrier()
                continue
            run_training_step(trainer, before_step=scheduler, rollouts=["episode"])
            trainer.step_expanded_training.assert_called_once()
            batch = trainer.step_expanded_training.call_args.args[0]
            assert batch.mu_iterations == 2
            assert len(batch.minibatches) == (0 if scenario == "all_empty" else 2)
            assert scheduler.call_count == (scenario != "all_empty")
            # Empty ranks must execute the same gradient schedule as their peers.
            for _ in range(batch.mu_iterations):
                for i, samples in enumerate(batch.minibatches):
                    count = torch.tensor(len(samples))
                    dist.all_reduce(count)
                    assert count.item() == batch.global_sample_counts[i]
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("expanded", [False, True])
def test_startup_validation_uses_registered_trainer_contract(monkeypatch, expanded):
    from cosmos_rl.policy.trainer.base import Trainer, TrainerRegistry
    from cosmos_rl.policy.worker.base import PolicyWorkerBase
    from cosmos_rl.policy.trainer.batching import FixedRolloutBatching

    class CustomTrainer(Trainer):
        batching_contract = (
            ExpandedSampleBatching() if expanded else FixedRolloutBatching()
        )

        def prepare_training_batch(self, rollouts):
            pass

        def step_expanded_training(self, batch, **kwargs):
            pass

    monkeypatch.setattr(TrainerRegistry, "get_trainer_cls", lambda _: CustomTrainer)
    worker = SimpleNamespace(
        config=SimpleNamespace(
            train=SimpleNamespace(
                train_batch_per_replica=3,
                train_policy=SimpleNamespace(
                    type="grpo", mini_batch=2, trainer_type="custom"
                ),
            ),
            policy=SimpleNamespace(
                parallelism=SimpleNamespace(
                    dp_shard_size=2, tp_size=1, cp_size=1, pp_size=1
                )
            ),
        ),
        parallel_dims=SimpleNamespace(dp_shard=2),
    )
    if expanded:
        PolicyWorkerBase.check_config(worker)
        worker.config.policy.parallelism.tp_size = 2
        with pytest.raises(ValueError, match="pure data parallelism"):
            PolicyWorkerBase.check_config(worker)
    else:
        with pytest.raises(AssertionError, match="divisible"):
            PolicyWorkerBase.check_config(worker)


def test_two_rank_preflight_never_diverges():
    with tempfile.TemporaryDirectory(prefix="batching-contract-") as directory:
        mp.spawn(
            _distributed_worker, args=(f"{directory}/rendezvous",), nprocs=2, join=True
        )
