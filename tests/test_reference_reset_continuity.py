# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Reference resets preserve scheduled LR and next-update checkpoint state."""

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from cosmos_rl.policy.trainer.llm_trainer.grpo_trainer import GRPOTrainer
from cosmos_rl.policy.trainer.optm import OptimizersContainer
from test_scheduler_continuity import _state, _lrs


def trainer_state():
    model, optimizer, scheduler, config = _state()
    config.train.train_policy = SimpleNamespace(
        kl_beta=0.1, reference_reset_interval=1, reset_optimizer_with_reference=True
    )
    config.train.ckpt = SimpleNamespace(export_safetensors=False)
    trainer = GRPOTrainer.__new__(GRPOTrainer)
    trainer.model, trainer.optimizers, trainer.lr_schedulers, trainer.config = (
        model,
        optimizer,
        scheduler,
        config,
    )
    trainer.parallel_dims = SimpleNamespace(pp_enabled=False)
    trainer.reference_state_dict = {
        key: torch.zeros_like(value).cpu() for key, value in model.state_dict().items()
    }

    def rebuild():
        trainer.optimizers = OptimizersContainer(
            torch.optim.SGD, [model], [{"lr": 0.1, "momentum": 0.9}]
        )

    trainer.build_optimizers = rebuild
    return trainer


@pytest.mark.parametrize("step", [3, 9, 18])
def test_reset_optimizer_keeps_current_schedule_and_next_parameter_update(step):
    trainer = trainer_state()
    for _ in range(3, step):
        trainer.model.weight.grad = torch.ones_like(trainer.model.weight)
        trainer.optimizers.step()
        trainer.lr_schedulers.step()
    expected_lr = _lrs(trainer.optimizers)
    scheduler_state = copy.deepcopy(trainer.lr_schedulers.state_dict())
    expected_weight = trainer.model.weight.detach().clone() - expected_lr[0]
    trainer.reference_reset(step)
    assert _lrs(trainer.optimizers) == expected_lr
    assert trainer.lr_schedulers.state_dict() == scheduler_state
    assert trainer.lr_schedulers.get_last_lr() == expected_lr
    assert all(not optimizer.state for optimizer in trainer.optimizers)
    trainer.model.weight.grad = torch.ones_like(trainer.model.weight)
    trainer.optimizers.step()
    torch.testing.assert_close(trainer.model.weight, expected_weight, rtol=0, atol=0)


def test_reference_snapshot_does_not_alias_cpu_offloaded_policy():
    trainer = trainer_state()
    trainer.reference_reset(3)
    before = copy.deepcopy(trainer.reference_state_dict)
    with torch.no_grad():
        trainer.model.weight.add_(10)
    torch.testing.assert_close(trainer.reference_state_dict, before, rtol=0, atol=0)
    assert trainer.reference_reset_step == 3


def test_checkpoint_contains_reference_and_reset_optimizer_for_next_update():
    trainer = trainer_state()
    trainer.reference_reset(3)
    trainer.ckpt_manager = Mock()
    trainer.save_checkpoint(3, 20, 17, is_final=False)
    payload = trainer.ckpt_manager.save_checkpoint.call_args.kwargs
    assert payload["grpo_reference_enabled"]
    assert payload["grpo_reference_reset_step"] == 3
    torch.testing.assert_close(
        payload["grpo_reference_state"], trainer.model.state_dict()
    )
    restored = trainer_state()
    restored._restore_checkpoint_reference(dict(payload))
    torch.testing.assert_close(
        restored.reference_state_dict, trainer.reference_state_dict
    )
    assert restored.reference_reset_step == 3


def test_training_boundary_publishes_after_scheduler_and_reference_reset():
    trainer = trainer_state()
    events = []

    def save(**kwargs):
        events.append((trainer.reference_reset_step, _lrs(trainer.optimizers)))
        assert all(not optimizer.state for optimizer in trainer.optimizers)
        torch.testing.assert_close(
            trainer.reference_state_dict, trainer.model.state_dict()
        )

    trainer.save_checkpoint = save
    trainer._finish_training_batch(4, 20, 16, save=True)
    assert events == [(4, [0.1])]


@pytest.mark.parametrize("mutation", ["keys", "shape", "dtype", "reset_step", "mode"])
def test_reference_metadata_mismatch_fails_after_successful_policy_restore(mutation):
    trainer = trainer_state()
    trainer.reference_reset(3)
    payload = {
        "step": 3,
        "grpo_reference_enabled": True,
        "grpo_reference_reset_step": 3,
        "grpo_reference_state": copy.deepcopy(trainer.reference_state_dict),
    }
    if mutation == "keys":
        payload["grpo_reference_state"] = {}
    elif mutation == "shape":
        payload["grpo_reference_state"]["weight"] = torch.zeros(4)
    elif mutation == "dtype":
        payload["grpo_reference_state"]["weight"] = payload["grpo_reference_state"][
            "weight"
        ].double()
    elif mutation == "reset_step":
        payload["grpo_reference_reset_step"] = 4
    else:
        payload["grpo_reference_enabled"] = False
    with pytest.raises(ValueError):
        trainer._restore_checkpoint_reference(payload)


def test_legacy_checkpoint_cannot_reconstruct_a_reset_reference_from_initial_hf():
    trainer = trainer_state()
    trainer.config.train.train_policy.reference_reset_interval = 4
    trainer._restore_checkpoint_reference({"step": 3})
    with pytest.raises(ValueError, match="cannot reliably resume"):
        trainer._restore_checkpoint_reference({"step": 4})
    trainer.config.train.train_policy.reference_reset_interval = None
    trainer._restore_checkpoint_reference({"step": 9})


def test_pipeline_reference_uses_live_parts_not_original_meta_model():
    trainer = trainer_state()
    trainer.parallel_dims.pp_enabled = True
    trainer.model = torch.nn.Linear(1, 1, device="meta")
    trainer.model_parts = [torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)]
    trainer.model_module_path = ["first", "last"]
    state = trainer._reference_model_state()
    assert set(state) == {"first.weight", "first.bias", "last.weight", "last.bias"}
    assert all(not value.is_meta for value in state.values())
    trainer.reference_state_dict = {
        key: torch.zeros_like(value) for key, value in state.items()
    }
    trainer.config.train.train_policy.reset_optimizer_with_reference = False
    trainer.reference_reset(3)
    torch.testing.assert_close(trainer.reference_state_dict, state)
    trainer.ckpt_manager = Mock()
    trainer.save_checkpoint(3, 20, 17, is_final=False)
    saved = trainer.ckpt_manager.save_checkpoint.call_args.kwargs["model"]
    torch.testing.assert_close(saved, state)
