# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from types import SimpleNamespace

from cosmos_rl.policy.trainer.base import Trainer
from cosmos_rl.policy.trainer.diffusers_trainer.nft_trainer import NFTTrainer
from cosmos_rl.policy.trainer.llm_trainer.grpo_trainer import GRPOTrainer
from cosmos_rl.policy.trainer.vla_trainer.pi_grpo_trainer import PI05GRPOTrainer
from cosmos_rl.policy.trainer.vla_trainer.vla_trainer import OpenVLAGRPOTrainer


class _RecordingCheckpointManager:
    def __init__(self):
        self.checkpoints = []
        self.completed_steps = []

    def save_checkpoint(self, **kwargs):
        self.checkpoints.append(kwargs)

    def save_check(self, *, step):
        self.completed_steps.append(step)


class _FailingCheckpointManager(_RecordingCheckpointManager):
    def save_checkpoint(self, **kwargs):
        raise RuntimeError("checkpoint failed")


class _RecordingEMA:
    def __init__(self):
        self.events = []

    def copy_ema_to(self, params, *, store_temp):
        self.events.append(("copy_ema_to", params, store_temp))

    def copy_temp_to(self, params):
        self.events.append(("copy_temp_to", params))


class _NFTModel:
    def __init__(self):
        self.ema = _RecordingEMA()
        self.state = object()

    def get_trained_model_state_dict(self):
        return self.state


def _checkpoint_config(*, export_safetensors: bool):
    return SimpleNamespace(
        train=SimpleNamespace(
            output_dir="/output",
            param_dtype="float32",
            ckpt=SimpleNamespace(export_safetensors=export_safetensors),
        )
    )


class _TrainerWithoutCheckpointSupport(Trainer):
    def build_optimizers(self):
        pass

    def build_lr_schedulers(self):
        pass

    def step_training(self):
        pass

    def step_validation(self):
        pass

    def export_safetensors(self, *args, **kwargs):
        pass

    def model_load_from_hf(self):
        pass

    def model_resume_from_checkpoint(self):
        pass

    @property
    def pp_loss_fn(self):
        return None


def test_base_checkpoint_hook_raises_only_when_invoked():
    trainer = object.__new__(_TrainerWithoutCheckpointSupport)

    with pytest.raises(NotImplementedError, match="checkpoint saving is not supported"):
        trainer.save_checkpoint(
            current_step=3,
            total_steps=8,
            remain_samples_num=21,
            is_final=True,
        )


def test_grpo_checkpoint_hook_preserves_final_checkpoint_behavior():
    trainer = object.__new__(GRPOTrainer)
    trainer.config = _checkpoint_config(export_safetensors=False)
    trainer.model = object()
    trainer.optimizers = object()
    trainer.lr_schedulers = object()
    trainer.ckpt_manager = _RecordingCheckpointManager()
    exports = []
    trainer.export_safetensors = lambda **kwargs: exports.append(kwargs)

    trainer.save_checkpoint(
        current_step=3,
        total_steps=8,
        remain_samples_num=21,
        is_final=True,
    )

    assert exports == [
        {
            "output_dir": "/output",
            "rel_path": "safetensors/step_3",
            "trainable_only": False,
            "is_final": True,
            "dtype": torch.float32,
        }
    ]
    assert trainer.ckpt_manager.checkpoints == [
        {
            "model": trainer.model,
            "optimizer": trainer.optimizers,
            "scheduler": trainer.lr_schedulers,
            "step": 3,
            "total_steps": 8,
            "remain_samples_num": 21,
            "is_final": True,
        }
    ]
    assert trainer.ckpt_manager.completed_steps == [3]


def test_nft_checkpoint_hook_restores_temporary_ema_weights_after_failure():
    trainer = object.__new__(NFTTrainer)
    trainer.config = _checkpoint_config(export_safetensors=False)
    trainer.config.train.ema_enable = True
    trainer.model = _NFTModel()
    trainer.trainable_params = object()
    trainer.optimizers = object()
    trainer.lr_schedulers = object()
    trainer.ckpt_manager = _FailingCheckpointManager()

    with pytest.raises(RuntimeError, match="checkpoint failed"):
        trainer.save_checkpoint(
            current_step=3,
            total_steps=8,
            remain_samples_num=21,
            is_final=False,
        )

    assert trainer.model.ema.events == [
        ("copy_ema_to", trainer.trainable_params, True),
        ("copy_temp_to", trainer.trainable_params),
    ]


@pytest.mark.parametrize("trainer_cls", [OpenVLAGRPOTrainer, PI05GRPOTrainer])
def test_vla_checkpoint_hooks_preserve_configured_export_behavior(trainer_cls):
    trainer = object.__new__(trainer_cls)
    trainer.config = _checkpoint_config(export_safetensors=False)
    trainer.model = object()
    trainer.optimizers = object()
    trainer.lr_schedulers = object()
    trainer.ckpt_manager = _RecordingCheckpointManager()
    exports = []
    trainer.export_safetensors = lambda **kwargs: exports.append(kwargs)

    trainer.save_checkpoint(
        current_step=3,
        total_steps=8,
        remain_samples_num=21,
        is_final=True,
    )

    assert exports == []
    assert trainer.ckpt_manager.checkpoints == [
        {
            "model": trainer.model,
            "optimizer": trainer.optimizers,
            "scheduler": trainer.lr_schedulers,
            "step": 3,
            "total_steps": 8,
            "remain_samples_num": 21,
            "is_final": True,
        }
    ]
    assert trainer.ckpt_manager.completed_steps == [3]
