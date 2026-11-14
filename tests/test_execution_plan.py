# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
import pytest

from cosmos_rl.policy.config import Config


@pytest.mark.parametrize("seed", [-1, True, 0.5, 2**63])
def test_record_preparation_seed_is_a_bounded_integer(seed):
    from cosmos_rl.execution.config import SourceConfig

    with pytest.raises(ValueError):
        SourceConfig(kind="corpus", path="/corpus", preparation_seed=seed)


def test_record_preparation_seed_requires_a_corpus():
    from cosmos_rl.execution.config import SourceConfig

    with pytest.raises(ValueError, match="requires source.kind=corpus"):
        SourceConfig(preparation_seed=0)


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


@pytest.mark.parametrize("revision", [None, "main", "v1", "", " " * 40])
def test_resumable_generation_rejects_unknown_or_moving_model_identity(revision):
    with pytest.raises(ValueError, match="immutable"):
        Config.model_validate(
            {
                "train": {"train_policy": {"type": "grpo"}},
                "policy": {"model_revision": revision},
                "execution": {
                    "path": "generate",
                    "output": {
                        "path": "/corpus",
                        "run_id": "run",
                        "generation_fingerprint": "text-v1",
                    },
                },
            }
        )


def test_resumable_generation_accepts_full_commit_and_requires_app_semantics():
    output = {"path": "/corpus", "run_id": "run"}
    data = {
        "train": {"train_policy": {"type": "grpo"}},
        "policy": {"model_revision": "a" * 40},
        "execution": {"path": "generate", "output": output},
    }
    with pytest.raises(ValueError, match="generation_fingerprint"):
        Config.model_validate(data)
    output["generation_fingerprint"] = "text-v1"
    assert Config.model_validate(data).execution.output.resume
    # An independent actor cannot inherit the learner's pinned revision.
    data["execution"]["actor"] = {
        "weight_source": "checkpoint",
        "model_name_or_path": "another-model",
    }
    with pytest.raises(ValueError, match="immutable"):
        Config.model_validate(data)


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


@pytest.mark.parametrize(
    "path,policy",
    [
        ("train_from_data", "sft"),
        ("generate", "grpo"),
        ("train_from_generation", "grpo"),
    ],
)
def test_resolved_execution_config_roundtrips_through_native_launcher_toml(
    path, policy
):
    from pathlib import Path
    import toml
    from cosmos_rl.launcher.utility import dump_config_with_literal_patterns_to_tmpfile

    configuration = recipe(policy, path=path)
    filename = Path(
        dump_config_with_literal_patterns_to_tmpfile(configuration.model_dump())
    )
    try:
        restored = Config.model_validate(toml.load(filename))
        assert restored.execution_plan == configuration.execution_plan
    finally:
        filename.unlink()


def test_recorded_distillation_does_not_allocate_a_live_teacher():
    configuration = Config.model_validate(
        {
            "execution": {
                "path": "train_from_data",
                "source": {"kind": "corpus", "path": "/recorded-targets"},
                "objectives": {"distillation": 1},
            },
            "train": {"train_policy": {"type": "sft"}},
            "policy": {"parallelism": {"tp_size": 1}},
            "distillation": {"enable": True},
        }
    )
    assert configuration.execution_plan.distillation == 1
    assert configuration.execution_plan.teachers == 0
    assert not configuration.distillation.enable


def test_three_independent_roles_fit_a_three_gpu_node():
    from types import SimpleNamespace
    from cosmos_rl.launcher.utility import NodesManager

    manager = NodesManager("replica.sh", ["2", "4", "7"])
    manager.replica_placement(SimpleNamespace(), 1, 1, 1, 1, 1, 1, "replica.sh")
    workers = manager.finalize()
    assert len(workers) == 1
    commands = workers[0].command_items
    assert [item.gpu_devices for item in commands] == ["2", "4", "7"]
    assert all(
        f"--type {role}" in item.command
        for role, item in zip(["policy", "rollout", "reference"], commands)
    )
    with pytest.raises(ValueError, match="integral number"):
        NodesManager("replica.sh", ["0", "1", "2"]).replica_placement(
            SimpleNamespace(), 1, 0, 0, 4, 1, 1, "replica.sh"
        )
