# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint commit and controller/trainer boundaries, not just file loading."""

import copy
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cosmos_rl.policy.config import Config
from cosmos_rl.policy.trainer.llm_trainer.grpo_trainer import GRPOTrainer
from cosmos_rl.dispatcher.data.data_fetcher import ControllerDataFetcher
from cosmos_rl.utils.checkpoint import CheckpointMananger
from cosmos_rl.utils.parallelism import ParallelDims


def manager_at(root, *, mode="sync", rank=0, dims=None):
    config = Config.from_dict(
        {
            "train": {
                "output_dir": str(root / "run"),
                "timestamp": "run",
                "ckpt": {
                    "enable_checkpoint": True,
                    "save_mode": mode,
                    "max_keep": 1,
                    "export_safetensors": False,
                },
            }
        }
    )
    return CheckpointMananger(config, dims, global_rank=rank)


def components():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    return model, optimizer, scheduler


def grpo_trainer(manager, kl_beta):
    trainer = object.__new__(GRPOTrainer)
    trainer.config = manager.config
    trainer.config.train.train_policy.kl_beta = kl_beta
    trainer.parallel_dims = SimpleNamespace(pp_enabled=False)
    trainer.ckpt_manager = manager
    trainer.model, trainer.optimizers, trainer.lr_schedulers = components()
    trainer.reference_state_dict = copy.deepcopy(trainer.model.state_dict())
    trainer.reference_reset_step = 0
    trainer.map_w_from_policy_to_rollout = object()
    trainer.model_load_from_hf = Mock()
    trainer.set_model_train = trainer.model.train
    trainer.model_resume_from_checkpoint = lambda: manager.load_checkpoint(
        trainer.model, trainer.optimizers, trainer.lr_schedulers, "unused"
    )[0]
    return trainer


@pytest.mark.parametrize("kl_beta", [0.0, 0.1])
@pytest.mark.parametrize("legacy", [False, True])
def test_real_checkpoint_resume_http_agreement(tmp_path, monkeypatch, kl_beta, legacy):
    from cosmos_rl.dispatcher import run_web_panel as panel

    manager = manager_at(tmp_path)
    trainer = grpo_trainer(manager, kl_beta)
    trainer.save_checkpoint(2, 10, 8, is_final=False)
    path = Path(manager.ckpt_output_dir) / "step_2" / "policy"
    if legacy:
        metadata_path = path / "extra_info_rank_0.pth"
        state = torch.load(metadata_path, weights_only=False)
        for key in (
            "grpo_reference_enabled",
            "grpo_reference_state",
            "grpo_reference_reset_step",
        ):
            state.pop(key)
        torch.save(state, metadata_path)
    trainer.config.train.resume = str(path)
    fetcher = object.__new__(ControllerDataFetcher)
    fetcher.ckpt_extra_info = manager.load_extra_info_from_checkpoint()
    worker_info = trainer.weight_resume()
    monkeypatch.setattr(panel, "controller", SimpleNamespace(data_fetcher=fetcher))
    exits = []

    async def exit_probe():
        exits.append(True)

    monkeypatch.setattr(panel, "_exit_on_resume_mismatch", exit_probe)
    app = FastAPI()
    app.post("/resume")(panel.resume_info)
    with TestClient(app) as client:
        response = client.post("/resume", json={"ckpt_extra_info": worker_info})
        assert response.status_code == 200, response.text
        assert not exits
        for key in ("step", "total_steps", "remain_samples_num"):
            changed = dict(worker_info, **{key: worker_info[key] + 1})
            response = client.post("/resume", json={"ckpt_extra_info": changed})
            assert response.status_code == 409
        assert len(exits) == 3


@pytest.mark.parametrize("failure", [False, True])
def test_retention_waits_for_complete_replacement(tmp_path, monkeypatch, failure):
    manager = manager_at(tmp_path, mode="async")
    model, optimizer, scheduler = components()
    old = Path(manager.save_checkpoint(model, optimizer, scheduler, 1, 10))
    manager._wait_for_pending_async_saves()
    manager.save_check(1, val_score=2.0)
    # Keep a healthy best pointer, but allow retention to select the old step.
    monkeypatch.setattr(manager, "_is_ckpt_dir_linked_as_best", lambda path: False)
    started, release, executor_progress = (threading.Event() for _ in range(3))
    original_save = torch.save

    def save(state, path):
        if "step_2" in str(path) and "model_rank" in str(path):
            started.set()
            assert release.wait(10)
            if failure:
                raise OSError("injected model write failure")
        original_save(state, path)

    monkeypatch.setattr(torch, "save", save)
    try:
        new = Path(manager.save_checkpoint(model, optimizer, scheduler, 2, 10))
        assert started.wait(5)
        manager.save_check(2, val_score=1.0)
        manager.executor.submit(executor_progress.set)
        assert executor_progress.wait(5)
        assert old.exists(), "retention deleted the last committed checkpoint"
        assert not manager.ckpt_path_check(str(new))
        assert manager.best_score == 2.0
        assert Path(manager._best_dir, "checkpoints").resolve() == old.parent
    finally:
        release.set()
        if failure:
            with pytest.raises(OSError, match="injected"):
                manager.finalize()
        else:
            manager.finalize()
    if failure:
        assert old.exists()
        assert not manager.ckpt_path_check(str(new))
    else:
        assert manager.ckpt_path_check(str(new))
        assert manager.best_score == 1.0


def test_other_saving_rank_must_complete_before_retention(tmp_path):
    dims = ParallelDims(
        dp_replicate=1,
        dp_shard=2,
        cp=1,
        tp=1,
        pp=1,
        world_size=2,
        pp_dynamic_shape=False,
    )
    first = manager_at(tmp_path, dims=dims)
    second = manager_at(tmp_path, dims=dims, rank=1)
    model, optimizer, scheduler = components()
    for manager in (first, second):
        manager.save_checkpoint(model, optimizer, scheduler, 1, 10)
    first.save_check(1)
    old = Path(first.ckpt_output_dir) / "step_1"
    path = first.save_checkpoint(model, optimizer, scheduler, 2, 10)
    first.save_check(2, val_score=1.0)
    first.finalize()
    assert old.exists()
    assert not first.ckpt_path_check(path)
    assert first.best_score == float("inf")
    second.save_checkpoint(model, optimizer, scheduler, 2, 10)
    first.save_check(2, val_score=1.0)
    assert not old.exists()
    assert first.best_score == 1.0


@pytest.mark.parametrize(
    "mutation", ["none", "missing", "extra", "parts", "prefixes", "duplicate"]
)
def test_pipeline_restore_requires_complete_stage_state(tmp_path, mutation):
    manager = manager_at(tmp_path)
    model, optimizer, scheduler = components()
    parts, prefixes = [model], ["stage"]
    state = {
        f"stage.{key}": value.detach().clone() + 10
        for key, value in model.state_dict().items()
    }
    if mutation == "missing":
        state.pop("stage.weight")
    elif mutation == "extra":
        state["unowned.weight"] = torch.ones(1)
    elif mutation == "parts":
        parts.append(torch.nn.Linear(2, 1))
    elif mutation == "prefixes":
        prefixes.append("another")
    elif mutation == "duplicate":
        parts.append(torch.nn.Linear(2, 1))
        prefixes.append("stage")
    path = manager.save_checkpoint(state, optimizer, scheduler, 2, 10)
    manager.config.train.resume = path
    if mutation == "none":
        info, _ = manager.load_checkpoint(
            model,
            optimizer,
            scheduler,
            "unused",
            pp_model_parts=parts,
            pp_model_module_paths=prefixes,
        )
        assert info["step"] == 2
        torch.testing.assert_close(model.weight, state["stage.weight"])
    else:
        before = copy.deepcopy(model.state_dict())
        with pytest.raises((ValueError, RuntimeError)):
            manager.load_checkpoint(
                model,
                optimizer,
                scheduler,
                "unused",
                pp_model_parts=parts,
                pp_model_module_paths=prefixes,
            )
        torch.testing.assert_close(model.state_dict(), before)


def test_interleaved_parts_can_share_a_root_prefix(tmp_path):
    manager = manager_at(tmp_path)
    parts = [
        torch.nn.ModuleDict({f"layer{i}": torch.nn.Linear(2, 1)}) for i in range(2)
    ]
    model = torch.nn.ModuleList(parts)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    state = {
        key: value.detach().clone() + 10
        for part in parts
        for key, value in part.state_dict().items()
    }
    manager.config.train.resume = manager.save_checkpoint(
        state, optimizer, scheduler, 2, 10
    )
    manager.load_checkpoint(
        model,
        optimizer,
        scheduler,
        "unused",
        pp_model_parts=parts,
        pp_model_module_paths=["", ""],
    )
    for part in parts:
        for key, value in part.state_dict().items():
            torch.testing.assert_close(value, state[key])


def test_controller_projection_preserves_unknown_contract_fields(tmp_path):
    manager = manager_at(tmp_path)
    model, optimizer, scheduler = components()
    manager.config.train.resume = manager.save_checkpoint(
        model, optimizer, scheduler, 2, 10, application_sampling={"cursor": 17}
    )
    info = manager.load_extra_info_from_checkpoint()
    assert info["application_sampling"] == {"cursor": 17}
