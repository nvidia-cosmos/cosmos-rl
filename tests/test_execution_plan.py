# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
import pytest

from cosmos_rl.policy.config import Config


def recipe(policy="sft", **execution):
    return Config.model_validate(
        {
            "train": {"train_policy": {"type": policy}},
            "execution": execution,
        }
    )


def test_legacy_paths_keep_role_and_objective_semantics():
    sft = recipe()
    assert sft.execution_plan.path == "train_from_data"
    assert sft.execution_plan.generators == sft.execution_plan.teachers == 0
    rl = recipe("grpo")
    assert rl.execution_plan.path == "train_from_generation"
    assert rl.execution_plan.sync_actor_weights
    assert rl.execution_plan.policy_gradient == 1


def test_explicit_paths_do_not_allocate_unused_roles():
    sft = recipe(path="train_from_data")
    assert sft.rollout.parallelism.n_init_replicas == 0
    generated = recipe("grpo", path="generate")
    assert generated.policy.parallelism.n_init_replicas == 0
    assert generated.rollout.rollout_only
    assert not generated.execution_plan.sync_actor_weights
    assert Config.model_validate(sft.model_dump()).execution_plan == sft.execution_plan
    assert (
        Config.model_validate(generated.model_dump()).execution_plan
        == generated.execution_plan
    )


def test_opd_keeps_a_student_actor_and_separate_teacher():
    config = Config.model_validate(
        {
            "train": {"train_policy": {"type": "grpo"}},
            "distillation": {"enable": True},
            "execution": {"path": "train_from_generation"},
        }
    )
    assert config.execution_plan.teachers == 1
    assert config.execution_plan.sync_actor_weights


@pytest.mark.parametrize(
    "execution",
    [
        {"path": "invalid"},
        {"path": "generate", "objectives": {"supervised": 1}},
        {"path": "train_from_data", "actor": {"weight_source": "checkpoint"}},
        {"path": "train_from_data", "source": {"kind": "corpus"}},
        {"path": "train_from_data", "objectives": {"supervised": float("nan")}},
        {"path": "train_from_data", "unknown_option": True},
        {"path": "train_from_generation", "actor": {"model_name_or_path": "different"}},
    ],
)
def test_invalid_plans_fail_during_configuration(execution):
    with pytest.raises(ValueError):
        recipe(**execution)


def test_explicit_contradictory_counts_are_not_silently_discarded():
    with pytest.raises(ValueError, match="zero rollout"):
        Config.model_validate(
            {
                "train": {"train_policy": {"type": "sft"}},
                "rollout": {"parallelism": {"n_init_replicas": 2}},
                "execution": {"path": "train_from_data"},
            }
        )


def test_shipped_execution_examples_validate():
    from pathlib import Path
    import toml

    for path in Path("configs/execution").glob("*.toml"):
        Config.model_validate(toml.load(path))


@pytest.mark.parametrize(
    "source,objectives,message",
    [
        ({"kind": "dataset"}, {"policy_gradient": 1}, "corpus source"),
        (
            {"kind": "corpus", "path": "/recorded", "reward_weighting": "nonnegative"},
            None,
            "explicit execution.objectives",
        ),
    ],
)
def test_recorded_objective_requirements_cannot_be_silently_ignored(
    source, objectives, message
):
    with pytest.raises(ValueError, match=message):
        Config.model_validate(
            {
                "execution": {
                    "path": "train_from_data",
                    "source": source,
                    "objectives": objectives,
                },
                "train": {"train_policy": {"type": "sft"}},
                "policy": {"parallelism": {"tp_size": 1}},
            }
        )
