# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A selected checkpoint may not degrade into fresh or older training state."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from cosmos_rl.utils.checkpoint import CheckpointMananger
from cosmos_rl.utils.resume import NoCheckpointFound
from cosmos_rl.policy.trainer.llm_trainer.sft_trainer import SFTTrainer
from cosmos_rl.policy.trainer.llm_trainer.dpo_trainer import DPOTrainer
from cosmos_rl.policy.trainer.llm_trainer.grpo_trainer import GRPOTrainer
from cosmos_rl.policy.trainer.diffusers_trainer.sft_trainer import (
    SFTTrainer as DiffusionSFTTrainer,
)
from cosmos_rl.policy.trainer.diffusers_trainer.nft_trainer import NFTTrainer
from cosmos_rl.dispatcher.data import data_fetcher
from test_checkpoint_discovery import _config, _checkpoint
from test_resume_data_index import _make_rl_config, _RLPromptDataset


TRAINERS = [
    (SFTTrainer, "load_model"),
    (DPOTrainer, "load_model"),
    (GRPOTrainer, "weight_resume"),
    (DiffusionSFTTrainer, "load_model"),
    (NFTTrainer, "weight_resume"),
]


def make_trainer(resume, failure=None):
    return SimpleNamespace(
        config=SimpleNamespace(
            train=SimpleNamespace(
                resume=resume, train_policy=SimpleNamespace(kl_beta=0.0)
            ),
            policy=SimpleNamespace(
                model_safetensor_path=None,
                model_name_or_path="model",
                model_revision=None,
            ),
        ),
        parallel_dims=SimpleNamespace(dp_replicate_enabled=False),
        model=SimpleNamespace(load_hf_weights=Mock(), transformer=Mock(), train=Mock()),
        device="cpu",
        model_resume_from_checkpoint=Mock(
            side_effect=failure, return_value={"total_steps": 20, "step": 2}
        ),
        _restore_checkpoint_reference=Mock(),
        model_load_from_hf=Mock(),
        build_optimizers=Mock(),
        set_model_train=Mock(),
        map_w_from_policy_to_rollout=object(),
    )


@pytest.mark.parametrize("trainer_class,method", TRAINERS)
@pytest.mark.parametrize("resume", [True, "/explicit/checkpoint"])
@pytest.mark.parametrize("failure_type", [ValueError, FileNotFoundError])
def test_restore_errors_never_fall_back_to_fresh_weights(
    trainer_class, method, resume, failure_type
):
    failure = failure_type("selected checkpoint restore failed")
    trainer = make_trainer(resume, failure)
    with pytest.raises(failure_type) as raised:
        getattr(trainer_class, method)(trainer)
    assert raised.value is failure
    trainer.model_load_from_hf.assert_not_called()
    trainer.model.load_hf_weights.assert_not_called()
    trainer.build_optimizers.assert_not_called()


@pytest.mark.parametrize("trainer_class,method", TRAINERS)
@pytest.mark.parametrize("resume", [True, "/explicit/checkpoint"])
def test_only_automatic_discovery_can_start_fresh(trainer_class, method, resume):
    trainer = make_trainer(resume, NoCheckpointFound("no committed checkpoint"))
    if isinstance(resume, str):
        with pytest.raises(NoCheckpointFound):
            getattr(trainer_class, method)(trainer)
        trainer.model_load_from_hf.assert_not_called()
        trainer.model.load_hf_weights.assert_not_called()
        trainer.build_optimizers.assert_not_called()
    else:
        getattr(trainer_class, method)(trainer)
        assert (
            trainer.model_load_from_hf.call_count
            + trainer.model.load_hf_weights.call_count
        ) == 1


@pytest.mark.parametrize("trainer_class,method", TRAINERS)
def test_successful_resume_preserves_metadata_and_weights(trainer_class, method):
    trainer = make_trainer(True)
    restored = getattr(trainer_class, method)(trainer)
    metadata = restored[-1] if isinstance(restored, tuple) else restored
    assert metadata == {"total_steps": 20, "step": 2}
    trainer.model_load_from_hf.assert_not_called()
    trainer.model.load_hf_weights.assert_not_called()
    trainer.build_optimizers.assert_not_called()


def write_checkpoint(root, step):
    path = _checkpoint(root, "previous", step, [0])
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    for name, state in (
        ("model", model.state_dict()),
        ("optimizer", optimizer.state_dict()),
        ("scheduler", scheduler.state_dict()),
        ("extra_info", {"step": step, "total_steps": 20, "remain_samples_num": 16}),
    ):
        torch.save(state, path / f"{name}_rank_0.pth")
    return path


@pytest.mark.parametrize("metadata_only", [False, True])
def test_selected_corrupt_checkpoint_does_not_retry_older_checkpoint(
    tmp_path, metadata_only
):
    write_checkpoint(tmp_path, 1)
    newer = write_checkpoint(tmp_path, 2)
    field = "extra_info" if metadata_only else "optimizer"
    (newer / f"{field}_rank_0.pth").write_bytes(b"corrupt committed artifact")
    manager = CheckpointMananger(_config(tmp_path, "current"))
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    with pytest.raises(Exception, match="corrupt|invalid|pickle|load key"):
        if metadata_only:
            manager.load_extra_info_from_checkpoint()
        else:
            manager.load_checkpoint(model, optimizer, scheduler, "unused")


def test_committed_checkpoint_requires_metadata_file(tmp_path):
    path = write_checkpoint(tmp_path, 1)
    (path / "extra_info_rank_0.pth").unlink()
    manager = CheckpointMananger(_config(tmp_path, "current"))
    with pytest.raises(FileNotFoundError):
        manager.load_extra_info_from_checkpoint()


@pytest.mark.parametrize("metadata_only", [False, True])
def test_missing_output_root_is_typed_discovery_miss(tmp_path, metadata_only):
    manager = CheckpointMananger(_config(tmp_path / "never-created", "current"))
    with pytest.raises(NoCheckpointFound):
        if metadata_only:
            manager.load_extra_info_from_checkpoint()
        else:
            manager.load_checkpoint(Mock(), Mock(), Mock(), "unused")
    assert manager.selected_checkpoint_path is None


@pytest.mark.parametrize("resume", [True, "/explicit/checkpoint"])
def test_controller_only_allows_automatic_discovery_miss(monkeypatch, resume):
    manager = Mock(
        load_extra_info_from_checkpoint=Mock(side_effect=NoCheckpointFound("missing"))
    )
    monkeypatch.setattr(data_fetcher, "CheckpointMananger", lambda config: manager)
    config = _make_rl_config(resume=resume)
    config.train.train_policy.dataloader_num_workers = 0
    config.train.train_policy.dataloader_prefetch_factor = None
    if isinstance(resume, str):
        with pytest.raises(NoCheckpointFound):
            data_fetcher.ControllerDataFetcher(
                config, dataset=_RLPromptDataset(size=10)
            )
    else:
        fetcher = data_fetcher.ControllerDataFetcher(
            config, dataset=_RLPromptDataset(size=10)
        )
        assert fetcher.config.train.resume is False
        assert fetcher.ckpt_extra_info == {}
        indices, _ = next(fetcher.train_dataloader_iter)
        assert list(indices) == [0, 1]


@pytest.mark.parametrize("resume", [True, "/explicit/checkpoint"])
def test_legacy_controller_rejects_corrupt_metadata_without_fresh_start(
    monkeypatch, resume
):
    failure = ValueError("corrupt checkpoint metadata")
    manager = Mock(load_extra_info_from_checkpoint=Mock(side_effect=failure))
    monkeypatch.setattr(data_fetcher, "CheckpointMananger", lambda config: manager)
    config = _make_rl_config(resume=resume)
    config.train.train_policy.dataloader_num_workers = 0
    config.train.train_policy.dataloader_prefetch_factor = None
    with pytest.raises(ValueError, match="corrupt checkpoint"):
        data_fetcher.ControllerDataFetcher(config, dataset=_RLPromptDataset(size=10))


def test_legacy_controller_publishes_the_selected_checkpoint(monkeypatch, tmp_path):
    path = write_checkpoint(tmp_path, 2)
    manager = CheckpointMananger(_config(tmp_path, "current"))
    monkeypatch.setattr(data_fetcher, "CheckpointMananger", lambda config: manager)
    config = _make_rl_config(resume=True)
    config.train.train_policy.dataloader_num_workers = 0
    config.train.train_policy.dataloader_prefetch_factor = None
    fetcher = data_fetcher.ControllerDataFetcher(
        config, dataset=_RLPromptDataset(size=10)
    )
    assert Path(fetcher.config.train.resume) == path.resolve()
    assert fetcher.ckpt_extra_info["step"] == 2
