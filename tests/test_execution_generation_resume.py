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
        "output": {"path": str(tmp_path), "run_id": "stable-run"},
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
        item.config.execution.output, metadata=generation_metadata(item.config)
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
    corpus = CorpusDataset(tmp_path)
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
