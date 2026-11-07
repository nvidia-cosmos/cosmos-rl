# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from cosmos_rl.execution.models import actor_config, validate_actor_representation
from cosmos_rl.policy.config import Config
from test_prompt_dispatch_liveness import controller, fetch


def config(**actor):
    return Config.model_validate(
        {
            "train": {
                "train_policy": {"type": "grpo", "use_rollout_logprobs_for_loss": True}
            },
            "policy": {
                "model_name_or_path": "student",
                "model_revision": "student-revision",
            },
            "execution": {
                "path": "train_from_generation",
                "actor": {"weight_source": "checkpoint", **actor},
            },
        }
    )


def test_actor_projection_preserves_learner_identity_and_uses_its_own_revision():
    learner = config(model_name_or_path="teacher")
    actor = actor_config(learner)
    assert actor.policy.model_name_or_path == "teacher"
    assert actor.policy.model_revision is None
    assert learner.policy.model_name_or_path == "student"
    assert learner.policy.model_revision == "student-revision"
    assert not learner.execution_plan.sync_actor_weights


def test_token_representation_mismatch_fails_before_model_initialization():
    def tokenizer(name, **kwargs):
        return NS(
            backend_tokenizer=NS(to_str=lambda: name),
            special_tokens_map={},
            chat_template="chat",
        )

    with pytest.raises(ValueError, match="representations differ"):
        validate_actor_representation(config(model_name_or_path="teacher"), tokenizer)

    def shared(*args, **kwargs):
        return NS(
            backend_tokenizer=NS(to_str=lambda: "shared-pipeline"),
            special_tokens_map={},
            chat_template="chat",
        )

    validate_actor_representation(config(model_name_or_path="teacher"), shared)


def test_independent_actor_dispatch_is_bounded_without_learner_version_gates():
    item = controller(lag=1, quota=2)
    item.config = config()
    item.config.train.train_batch_per_replica = 2
    item.config.rollout.n_generation = 1
    item.config.train.train_policy.max_inflight_steps = 2
    item.policy_status_manager.current_step = 40
    item.policy_status_manager.config = item.config
    item.policy_status_manager.total_steps = 100
    payloads = fetch(item, 10)
    assert len(payloads) == 4
    assert all(payload.weight_version == 0 for payload in payloads)
    assert fetch(item, 10) == []
    item.policy_status_manager.samples_on_the_fly -= 1
    assert len(fetch(item, 10)) == 1
    assert item.policy_status_manager.current_step == 40


def test_independent_actor_keeps_original_version_and_never_schedules_weight_sync():
    item = controller(lag=1)
    item.policy_status_manager.config = config()
    item.policy_status_manager.current_step = 40
    rollout = NS(weight_version=0)
    assert item.policy_status_manager.filter_outdated_rollouts([rollout]) == [rollout]
    assert rollout.weight_version == 0
    assert not item.policy_status_manager.should_weight_sync_after_train_ack(40, Mock())


def test_actor_waits_for_initial_learner_registration_without_ending_input():
    item = controller()
    item.config = config()
    item.policy_status_manager.policy_replicas = {}
    payloads, ended = asyncio.run(item._get_batched_prompt_impl(2))
    assert payloads == [] and not ended


@pytest.mark.parametrize("revision", [None, "pinned-student"])
def test_teacher_uses_student_tokenizer_without_mutating_shared_config(
    monkeypatch, revision
):
    from cosmos_rl.reference.worker import teacher_worker

    shared = Config.model_validate(
        {
            "train": {"train_policy": {"type": "grpo"}},
            "policy": {"model_name_or_path": "student", "model_revision": revision},
            "distillation": {"enable": True, "model_name_or_path": "teacher"},
        }
    )
    monkeypatch.setattr(
        teacher_worker.PolicyWorkerBase,
        "__init__",
        lambda self, config, **kwargs: setattr(self, "config", config),
    )
    loaded = []

    def tokenizer(name, **kwargs):
        loaded.append((name, kwargs))
        return name

    monkeypatch.setattr(teacher_worker, "setup_tokenizer", tokenizer)
    monkeypatch.setattr(
        teacher_worker.TeacherWorker, "build_runner", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(teacher_worker.atexit, "register", lambda *args: None)
    worker = teacher_worker.TeacherWorker(shared, NS())
    assert worker.student_tokenizer == "student"
    assert worker.config.policy.model_name_or_path == "teacher"
    assert shared.policy.model_name_or_path == "student"
    assert loaded == [
        ("student", {"revision": revision} if revision is not None else {})
    ]


def test_hf_tokenizer_and_packer_use_the_configured_model_revision(monkeypatch):
    from cosmos_rl.utils import util
    from cosmos_rl.dispatcher.data.packer.decoder_only_llm_data_packer import (
        DecoderOnlyLLMDataPacker,
    )

    loaded = []
    monkeypatch.setattr(util, "_CUSTOM_TOKENIZER_LOADERS", [])

    def load(name, **kwargs):
        loaded.append((name, kwargs))
        return NS(pad_token_id=0)

    monkeypatch.setattr(util.AutoTokenizer, "from_pretrained", load)
    packer = DecoderOnlyLLMDataPacker()
    packer.setup(config())
    assert loaded == [
        ("student", {"trust_remote_code": True, "revision": "student-revision"})
    ]
