# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Identity configuration and borrowed SDK run lifecycle, without network I/O."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from cosmos_rl.policy.config import Config, LoggingConfig
from cosmos_rl.policy.config.wfm import CosmosVisionGenConfig
from cosmos_rl.utils.report import wandb_logger as logger


@pytest.fixture
def sdk(monkeypatch):
    sdk = SimpleNamespace(run=None, init=Mock())

    def initialize(**kwargs):
        sdk.run = Mock()
        return sdk.run

    sdk.init.side_effect = initialize
    monkeypatch.setattr(logger, "wandb", sdk)
    monkeypatch.setattr(logger, "wandb_run", None)
    return sdk


@pytest.fixture
def config(tmp_path):
    config = Config()
    config.train.output_dir = str(tmp_path)
    config.train.timestamp = "timestamp"
    return config


@pytest.mark.parametrize("experiment", [None, "", "None", "experiment"])
def test_legacy_identity(sdk, config, experiment):
    config.logging.experiment_name = experiment
    run = logger.init_wandb(config)
    kwargs = sdk.init.call_args.kwargs
    assert kwargs["id"] == "timestamp"
    assert kwargs["resume"] == "allow"
    assert kwargs["name"] == (
        "experiment/timestamp"
        if experiment == "experiment"
        else config.train.output_dir
    )
    logger.log_wandb({"loss": 1}, 2)
    run.log.assert_called_once_with({"loss": 1}, step=2)


@pytest.mark.parametrize("resume", ["allow", "must", "never", "auto", None])
def test_explicit_identity(sdk, config, resume):
    config.logging = LoggingConfig(
        wandb_run_id="attempt-id",
        wandb_run_name="exact-name",
        wandb_resume=resume,
        project_name="project",
        group_name="group",
    )
    logger.init_wandb(config)
    kwargs = sdk.init.call_args.kwargs
    assert {k: kwargs[k] for k in ("id", "name", "resume", "project", "group")} == {
        "id": "attempt-id",
        "name": "exact-name",
        "resume": resume,
        "project": "project",
        "group": "group",
    }
    assert LoggingConfig.model_validate(config.logging.model_dump()) == config.logging


@pytest.mark.parametrize(
    "values",
    [
        {"wandb_run_id": ""},
        {"wandb_run_name": ""},
        {"wandb_resume": "typo"},
    ],
)
def test_invalid_configuration(values):
    with pytest.raises(ValidationError):
        LoggingConfig(**values)


def test_borrow_external_run_and_repeated_initialization(sdk, config):
    external = sdk.run = Mock()
    for _ in range(2):
        assert logger.init_wandb(config) is external
    sdk.init.assert_not_called()
    logger.log_wandb({"loss": 1}, 1)
    external.log.assert_called_once()
    external.finish.assert_not_called()


def test_finished_or_replaced_run_requires_explicit_initialization(sdk, config):
    old = logger.init_wandb(config)
    sdk.run = None
    logger.log_wandb({}, 1)
    assert logger.wandb_run is None
    old.log.assert_not_called()
    replacement = sdk.run = Mock()
    logger.log_wandb({}, 2)
    replacement.log.assert_not_called()
    assert logger.init_wandb(config) is replacement
    logger.log_wandb({}, 3)
    replacement.log.assert_called_once_with({}, step=3)


def test_failed_initialization_clears_old_handle_and_can_retry(sdk, config):
    old = logger.init_wandb(config)
    sdk.run = None
    sdk.init.side_effect = RuntimeError("initialization failed")
    assert logger.init_wandb(config) is None
    assert logger.wandb_run is None
    logger.log_wandb({}, 1)
    old.log.assert_not_called()
    replacement = sdk.run = Mock()
    assert logger.init_wandb(config) is replacement


def test_optional_dependency_absent(monkeypatch, config):
    monkeypatch.setattr(logger, "wandb", None)
    monkeypatch.setattr(logger, "wandb_run", Mock())
    assert logger.init_wandb(config) is None
    logger.log_wandb({}, 1)
    assert logger.wandb_run is None


def test_vision_config_preserves_legacy_identity(sdk):
    config = CosmosVisionGenConfig(job={"name": "vision", "timestamp": "saved"})
    logger.init_wandb(config)
    kwargs = sdk.init.call_args.kwargs
    assert kwargs["name"] == config.job.name
    assert kwargs["id"] == config.job.timestamp
    assert kwargs["resume"] == "allow"
