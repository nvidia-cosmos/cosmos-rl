# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""The next optimizer update uses the restored LR, not LambdaLR's initial LR."""

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from cosmos_rl.policy.trainer.optm import OptimizersContainer, build_lr_schedulers
from cosmos_rl.policy.trainer.llm_trainer.grpo_trainer import GRPOTrainer
from cosmos_rl.policy.worker.multi_replica_sft_worker import MultiReplicaSFTPolicyWorker
from cosmos_rl.policy.worker.rl_worker import RLPolicyWorker


def _state():
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = OptimizersContainer(
        torch.optim.SGD, [model], [{"lr": 0.1, "momentum": 0.9}]
    )
    config = SimpleNamespace(
        train=SimpleNamespace(
            optm_warmup_steps=4,
            optm_decay_ratio=None,
            optm_decay_type="linear",
            optm_min_lr_factor=0.0,
            optm_warmup_start_factor=0.0,
        )
    )
    scheduler = build_lr_schedulers(optimizer, config, 20)
    for _ in range(3):
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        scheduler.step()
    return model, optimizer, scheduler, config


def _lrs(optimizer):
    return [group["lr"] for opt in optimizer for group in opt.param_groups]


def test_scheduler_reload_restores_effective_optimizer_lr():
    model, optimizer, scheduler, config = _state()
    expected_lr = _lrs(optimizer)
    state = copy.deepcopy(scheduler.state_dict())
    replacement = build_lr_schedulers(optimizer, config, 20)
    assert _lrs(optimizer) == [0.0]
    replacement.load_state_dict(state)
    assert _lrs(optimizer) == expected_lr
    assert replacement.get_last_lr() == expected_lr
    assert replacement.state_dict() == state


def test_first_grpo_update_preserves_loaded_lr_and_parameter_delta():
    model, optimizer, scheduler, config = _state()
    expected_lr = _lrs(optimizer)
    expected_model = copy.deepcopy(model)
    reference = torch.optim.SGD(
        expected_model.parameters(), lr=expected_lr[0], momentum=0.9
    )
    reference.load_state_dict(copy.deepcopy(next(iter(optimizer)).state_dict()))
    trainer = SimpleNamespace(
        lr_schedulers=scheduler,
        lr_schedulers_updated=False,
        build_lr_schedulers=lambda steps: build_lr_schedulers(optimizer, config, steps),
    )
    GRPOTrainer.update_lr_schedulers(trainer, 20)
    assert _lrs(optimizer) == expected_lr
    for module, opt in ((model, optimizer), (expected_model, reference)):
        module.weight.grad = torch.ones_like(module.weight)
        opt.step()
    torch.testing.assert_close(model.weight, expected_model.weight, rtol=0, atol=0)


@pytest.mark.parametrize("method", ["broadcast", "unicast"])
@pytest.mark.parametrize("mismatch", [False, True])
def test_multi_replica_sft_keeps_loaded_scheduler_before_state_sync(
    monkeypatch, method, mismatch
):
    _, optimizer, scheduler, config = _state()
    expected_state = copy.deepcopy(scheduler.state_dict())
    expected_lr = _lrs(optimizer)
    worker = object.__new__(MultiReplicaSFTPolicyWorker)
    worker.trainer = SimpleNamespace(optimizers=optimizer, lr_schedulers=scheduler)
    worker.config = config
    worker.total_steps = None
    worker.loaded_total_steps = 20
    name = f"execute_policy_to_policy_{method}"
    sync = Mock(return_value=False)
    monkeypatch.setattr(RLPolicyWorker, name, sync)
    command = SimpleNamespace(total_steps=21 if mismatch else 20)
    if mismatch:
        with pytest.raises((ValueError, AssertionError)):
            getattr(worker, name)(command)
        sync.assert_not_called()
        assert worker.total_steps is None
    else:
        assert getattr(worker, name)(command) is False
        sync.assert_called_once()
        assert worker.total_steps == 20
    assert worker.trainer.lr_schedulers is scheduler
    assert scheduler.state_dict() == expected_state
    assert _lrs(optimizer) == expected_lr


@pytest.mark.parametrize("loaded_steps", [None, 0, 20])
def test_multi_replica_sft_builds_only_for_fresh_initialization(loaded_steps):
    _, optimizer, _, config = _state()
    worker = object.__new__(MultiReplicaSFTPolicyWorker)
    worker.trainer = SimpleNamespace(optimizers=optimizer, lr_schedulers=None)
    worker.config = config
    worker.total_steps = None
    worker.loaded_total_steps = loaded_steps
    if loaded_steps:
        with pytest.raises(RuntimeError, match="restored scheduler"):
            worker._prepare_lr_schedulers(20)
        assert worker.total_steps is None
    else:
        worker._prepare_lr_schedulers(20)
        assert worker.total_steps == 20
        scheduler = worker.trainer.lr_schedulers
        worker._prepare_lr_schedulers(20)
        assert worker.trainer.lr_schedulers is scheduler
        with pytest.raises(ValueError, match="differs"):
            worker._prepare_lr_schedulers(21)
        assert worker.total_steps == 20
