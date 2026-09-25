# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Reject unsupported trajectory distillation before worker/model startup."""

from pathlib import Path
import tomllib

import pytest
from pydantic import ValidationError

from cosmos_rl.policy.config import Config


def config_data():
    with (Path(__file__).parent / "configs/test_simple_grpo.toml").open("rb") as source:
        return tomllib.load(source)


@pytest.mark.parametrize("mode", ["colocated", "disaggregated"])
@pytest.mark.parametrize("top_k", [0, 4])
@pytest.mark.parametrize("teacher_tokens", [False, True])
@pytest.mark.parametrize("recompute", [False, True])
def test_multiturn_distillation_rejected_before_launch(
    mode, top_k, teacher_tokens, recompute
):
    data = config_data()
    data["mode"] = mode
    data["rollout"]["multi_turn_config"] = {"enable": True}
    data["distillation"] = {
        "enable": True,
        "top_k": top_k,
        "trainer_token_ids_from_teacher": teacher_tokens,
        "rollout_top_k_recompute": recompute,
    }
    with pytest.raises(
        ValidationError, match="Multi-turn distillation is not supported"
    ) as error:
        Config.model_validate(data)
    assert "per-trajectory conversation context and token alignment" in str(error.value)


@pytest.mark.parametrize("top_k", [0, 4])
def test_single_turn_distillation_preserves_existing_configuration(top_k):
    data = config_data()
    data["distillation"] = {"enable": True, "top_k": top_k}
    config = Config.model_validate(data)
    assert config.distillation.enable
    assert config.distillation.top_k == top_k
    assert config.train.train_policy.rollout_as_token_ids
    assert config.train.train_policy.bypass_reward


@pytest.mark.parametrize("multi_turn", [False, True])
def test_without_distillation_preserves_rollout_configuration(multi_turn):
    data = config_data()
    data["rollout"]["multi_turn_config"] = {"enable": multi_turn}
    config = Config.model_validate(data)
    assert not config.distillation.enable
    assert config.rollout.multi_turn_config.enable == multi_turn
    assert not config.train.train_policy.bypass_reward
