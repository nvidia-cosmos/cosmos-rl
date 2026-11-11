# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.execution.corpus import CorpusDataset
from cosmos_rl.execution.sink import CorpusResultSink, generation_metadata
from cosmos_rl.policy.config import Config
from cosmos_rl.rollout.results import CompletionOutcome, RolloutResultEvent
from test_rollout_only import controller


def generation(tmp_path, *, generation=1, count=4):
    item = controller()
    data = item.config.model_dump()
    data["execution"] = {
        "path": "generate",
        "actor": {"weight_source": "checkpoint", "artifact_digest": "test-weights-v1"},
        "output": {
            "path": str(tmp_path),
            "run_id": "stable-run",
            "generation_fingerprint": "test-application-v1",
        },
    }
    data["rollout"].update(
        n_generation=generation, max_inflight_prompts=8, max_prompts=count
    )
    item.config = Config.model_validate(data)
    item.data_fetcher = SimpleNamespace(
        get_batched_prompt=Mock(
            return_value=(
                [RLPayload(prompt_idx=i, prompt=f"input-{i}") for i in range(count)],
                True,
            )
        )
    )
    return item


def sink(item):
    return CorpusResultSink(
        item.config.execution.output,
        actor_digest=(
            item.config.execution.actor.artifact_digest
            if item.config.execution.actor
            else None
        ),
        metadata=generation_metadata(item.config),
    )


def event(item, payload, *, text="answer"):
    return RolloutResultEvent(
        execution_id="attempt",
        replica_name="worker",
        global_rank=0,
        work_id=f"reservation-{payload.generation_occurrence}",
        prompt_idx=payload.prompt_idx,
        occurrence_id=payload.generation_occurrence,
        model_name_or_path=item.config.policy.model_name_or_path,
        model_revision=item.config.policy.model_revision,
        weight_version=0,
        prompt=payload.prompt,
        conversation=None,
        outcomes=tuple(
            CompletionOutcome(
                slot=slot, status="generated", completion=f"{text}-{slot}"
            )
            for slot in range(item.config.rollout.n_generation)
        ),
    )


def test_controller_resume_skips_committed_outputs_and_keeps_logical_limit(tmp_path):
    initial = generation(tmp_path)
    work, ended = initial._get_rollout_only_prompts(4)
    assert ended and len(work) == 4
    writer = sink(initial)
    for payload in work[:2]:
        writer(event(initial, payload))
    resumed = generation(tmp_path)
    pending, ended = resumed._get_rollout_only_prompts(4)
    assert ended
    assert [p.generation_occurrence for p in pending] == [2, 3]
    assert resumed.rollout_only_state.prompts_issued == 2
    assert resumed.rollout_only_state.source_occurrences_seen == 4
    for payload in pending:
        sink(resumed)(event(resumed, payload))
    replay = generation(tmp_path)
    assert replay._get_rollout_only_prompts(4) == ([], True)
    resumed.rollout_only_state.settle(len(pending))
    resumed.finalize_generation()
    assert len(CorpusDataset(tmp_path)) == 4


def test_changed_source_is_rejected_before_reusing_a_logical_occurrence(tmp_path):
    initial = generation(tmp_path)
    initial._get_rollout_only_prompts(1)
    resumed = generation(tmp_path)
    resumed.data_fetcher.get_batched_prompt.return_value = (
        [RLPayload(prompt_idx=0, prompt="changed")],
        True,
    )
    with pytest.raises(ValueError, match="source changed"):
        resumed._get_rollout_only_prompts(1)


def test_partial_prompt_publication_replays_only_missing_commit(tmp_path, monkeypatch):
    initial = generation(tmp_path, generation=2, count=1)
    work, _ = initial._get_rollout_only_prompts(1)
    consumer = sink(initial)
    original = consumer.writer.commit
    count = 0

    def interrupted(record):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("writer died after one slot")
        return original(record)

    monkeypatch.setattr(consumer.writer, "commit", interrupted)
    with pytest.raises(OSError, match="one slot"):
        consumer(event(initial, work[0], text="first"))
    resumed = generation(tmp_path, generation=2, count=1)
    pending, ended = resumed._get_rollout_only_prompts(1)
    assert ended and len(pending) == 1
    sink(resumed)(event(resumed, pending[0], text="second"))
    corpus = CorpusDataset(tmp_path, allow_partial=True)
    rows = [corpus[i] for i in range(len(corpus))]
    assert {row.group_slot: row.targets for row in rows} == {
        0: "first-0",
        1: "second-1",
    }


def test_changed_sampling_configuration_cannot_append_to_existing_run(tmp_path):
    initial = generation(tmp_path)
    initial._get_rollout_only_prompts(1)
    resumed = generation(tmp_path)
    resumed.config.train.train_policy.temperature = 0.5
    with pytest.raises(ValueError, match="another dataset run"):
        resumed._get_rollout_only_prompts(1)


def test_restart_validates_committed_artifacts_before_skipping_input(tmp_path):
    from dataclasses import replace
    import torch

    initial = generation(tmp_path, count=1)
    work, _ = initial._get_rollout_only_prompts(1)
    consumer = sink(initial)
    result = event(initial, work[0])
    consumer(
        replace(
            result, outcomes=(replace(result.outcomes[0], completion=torch.arange(4)),)
        )
    )
    artifact = next((tmp_path / "artifacts").glob("*.pt"))
    artifact.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="length"):
        generation(tmp_path, count=1)._get_rollout_only_prompts(1)


def test_restart_reconciles_identity_without_tensor_io(tmp_path, monkeypatch):
    from dataclasses import replace
    import torch
    from cosmos_rl.execution import corpus

    initial = generation(tmp_path, count=1)
    work, _ = initial._get_rollout_only_prompts(1)
    result = event(initial, work[0])
    sink(initial)(
        replace(
            result, outcomes=(replace(result.outcomes[0], completion=torch.arange(32)),)
        )
    )
    monkeypatch.setattr(
        torch, "load", lambda *a, **k: pytest.fail("controller loaded tensor")
    )
    monkeypatch.setattr(
        corpus, "_file_digest", lambda *a: pytest.fail("controller hashed tensor")
    )
    resumed, ended = generation(tmp_path, count=1)._get_rollout_only_prompts(1)
    assert not resumed and ended


def test_generation_completion_requires_settlement_and_all_slots(tmp_path):
    item = generation(tmp_path, count=1)
    work, ended = item._get_rollout_only_prompts(1)
    assert ended
    with pytest.raises(ValueError, match="unsettled"):
        item.finalize_generation()
    item.rollout_only_state.settle(1)
    with pytest.raises(ValueError, match="count"):
        item.finalize_generation()
    sink(item)(event(item, work[0]))
    item.finalize_generation()
    assert len(CorpusDataset(tmp_path)) == 1
    assert generation(tmp_path, count=1)._get_rollout_only_prompts(1) == ([], True)


@pytest.mark.parametrize("change", ["model", "application"])
def test_changed_immutable_generation_identity_cannot_append(tmp_path, change):
    initial = generation(tmp_path)
    initial._get_rollout_only_prompts(1)
    resumed = generation(tmp_path)
    if change == "model":
        resumed.config.execution.actor.artifact_digest = "test-weights-v2"
    else:
        resumed.config.execution.output.generation_fingerprint = "test-application-v2"
    with pytest.raises(ValueError, match="another dataset run"):
        resumed._get_rollout_only_prompts(1)


@pytest.mark.parametrize("stage", ["journaled", "committed", "completed"])
def test_one_off_generation_allows_unknown_model_but_cannot_reopen_work(
    tmp_path, stage
):
    from cosmos_rl.execution.sink import GenerationLedger

    item = generation(tmp_path)
    data = item.config.model_dump()
    data["execution"]["actor"] = None
    data["policy"]["model_revision"] = None
    data["execution"]["output"].update(resume=False, generation_fingerprint=None)
    item.config = Config.model_validate(data)
    # Producer sink initialization may publish dataset metadata before dispatch.
    sink(item)
    pending, _ = item._get_rollout_only_prompts(1)
    assert len(pending) == 1
    if stage != "journaled":
        sink(item)(event(item, pending[0]))
    if stage == "completed":
        item.rollout_only_state.settle(1)
        # The fixture's source normally has four inputs; close admission here.
        item.rollout_only_state.input_ended = True
        item._generation_ledger.finalize(1)
    with pytest.raises(ValueError, match="One-off generation cannot reopen"):
        GenerationLedger(item.config)


def test_generation_metadata_rechecks_identity_after_configuration_mutation(tmp_path):
    item = generation(tmp_path)
    item.config.execution.actor.artifact_digest = None
    item.config.policy.model_revision = "main"
    with pytest.raises(ValueError, match="immutable"):
        generation_metadata(item.config)
