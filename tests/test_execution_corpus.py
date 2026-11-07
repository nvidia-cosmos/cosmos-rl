# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from cosmos_rl.execution import corpus as storage
from cosmos_rl.execution.corpus import CorpusDataset, CorpusWriter
from cosmos_rl.execution.records import ModelIdentity, SampleRecord
from cosmos_rl.execution.sink import CorpusResultSink
from cosmos_rl.execution.source import (
    CorpusSource,
    CorpusSupervisedDataset,
    prepare_supervised_record,
)
from cosmos_rl.rollout.results import CompletionOutcome, RolloutResultEvent


def record(index=0, **kwargs):
    return SampleRecord(
        sample_id=str(index),
        occurrence_id=f"occurrence-{index}",
        group_id=f"group-{index // 2}",
        producer=ModelIdentity(role="actor", name="model", revision="pinned"),
        inputs={"image": torch.arange(12).reshape(3, 2, 2)},
        targets=torch.tensor([0.1, 0.2]),
        **kwargs,
    )


def test_owned_tensor_roundtrip_and_typed_mapping(tmp_path):
    sample = record()
    sample.inputs["literal"] = {"kind": "tensor", "path": "ordinary application data"}
    writer = CorpusWriter(tmp_path, run_id="run")
    assert writer.commit(sample)
    sample.inputs["image"].zero_()
    sample.targets.zero_()
    result = CorpusDataset(tmp_path)[0]
    assert result.inputs["image"].sum() == 66
    assert torch.allclose(result.targets, torch.tensor([0.1, 0.2]))
    assert result.inputs["literal"]["path"] == "ordinary application data"


def test_concurrent_commit_and_replay_keep_first_durable_output(tmp_path):
    def commit(_):
        return CorpusWriter(tmp_path, run_id="run").commit(record())

    with ThreadPoolExecutor(4) as workers:
        assert sum(workers.map(commit, range(8))) == 1
    writer = CorpusWriter(tmp_path, run_id="run")
    replay = record()
    replay.targets = torch.tensor([9.0])
    assert not writer.commit(replay)
    assert len(CorpusDataset(tmp_path)) == 1
    assert torch.allclose(CorpusDataset(tmp_path)[0].targets, torch.tensor([0.1, 0.2]))


def test_crash_before_manifest_exposes_no_partial_record(tmp_path, monkeypatch):
    writer = CorpusWriter(tmp_path, run_id="run")
    publish = storage._publish

    def fail_manifest(path, data):
        if path.parent.name == "records":
            raise OSError("crash before manifest publication")
        return publish(path, data)

    with monkeypatch.context() as patch:
        patch.setattr(storage, "_publish", fail_manifest)
        with pytest.raises(OSError, match="crash"):
            writer.commit(record())
    assert len(CorpusDataset(tmp_path)) == 0
    assert writer.commit(record())
    assert len(CorpusDataset(tmp_path)) == 1


def test_corrupt_artifact_and_changed_snapshot_are_rejected(tmp_path):
    writer = CorpusWriter(tmp_path, run_id="run")
    writer.commit(record())
    original = CorpusDataset(tmp_path).revision
    artifact = next((tmp_path / "artifacts").iterdir())
    data = artifact.read_bytes()
    artifact.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    with pytest.raises(ValueError, match="checksum"):
        CorpusDataset(tmp_path)[0]
    artifact.write_bytes(data)
    writer.commit(record(1))
    with pytest.raises(ValueError, match="revision"):
        CorpusDataset(tmp_path, revision=original)


def test_source_progress_commits_at_update_boundary_and_preserves_groups(tmp_path):
    writer = CorpusWriter(tmp_path, run_id="run")
    for i in range(6):
        writer.commit(record(i))
    dataset = CorpusDataset(tmp_path)
    source = CorpusSource(dataset)
    batch = source.reserve(1)
    assert len({r.group_id for r in batch.records}) == 1
    assert len(batch.records) == 2
    before = source.state_dict()
    resumed = CorpusSource(dataset)
    resumed.load_state_dict(before)
    replay = resumed.reserve(1)
    assert [r.sample_id for r in replay.records] == [r.sample_id for r in batch.records]
    source.commit(batch)
    resumed.rollback()
    resumed.load_state_dict(source.state_dict())
    assert {r.group_id for r in resumed.reserve(1).records}.isdisjoint(
        {r.group_id for r in batch.records}
    )


def test_global_shards_do_not_duplicate_partial_final_group_window(tmp_path):
    writer = CorpusWriter(tmp_path, run_id="run")
    for i in range(6):
        writer.commit(record(i))
    sources = [
        CorpusSource(CorpusDataset(tmp_path), rank=r, world_size=2) for r in range(2)
    ]
    ids = []
    while sources[0].cursor < len(sources[0].groups):
        batches = [source.reserve(1) for source in sources]
        ids += [item.sample_id for batch in batches for item in batch.records]
        for source, batch in zip(sources, batches):
            source.commit(batch)
    assert len(ids) == len(set(ids)) == 6


def test_byte_budget_checked_before_loading_a_group(tmp_path, monkeypatch):
    writer = CorpusWriter(tmp_path, run_id="run")
    writer.commit(record())
    dataset = CorpusDataset(tmp_path)
    source = CorpusSource(dataset, max_batch_bytes=1)
    monkeypatch.setattr(
        CorpusDataset, "__getitem__", lambda *_: pytest.fail("loaded oversized data")
    )
    with pytest.raises(ValueError, match="byte budget"):
        source.reserve(1)


def test_result_sink_keeps_all_outcomes_and_resumes_with_new_attempt(tmp_path):
    sink = CorpusResultSink(
        SimpleNamespace(path=tmp_path, run_id="dataset", max_record_bytes=1024**2)
    )
    event = RolloutResultEvent(
        execution_id="attempt-1",
        replica_name="worker",
        global_rank=0,
        work_id="native-reservation",
        prompt_idx=7,
        occurrence_id=0,
        model_name_or_path="teacher",
        model_revision="pinned",
        weight_version=0,
        prompt="question",
        conversation=None,
        outcomes=(
            CompletionOutcome(
                slot=0, status="generated", completion="answer", quality_trainable=False
            ),
            CompletionOutcome(
                slot=1, status="generation_failed", failure_reason="engine failure"
            ),
        ),
    )
    sink(event)
    sink(replace(event, execution_id="attempt-2", work_id="new-reservation"))
    corpus = CorpusDataset(tmp_path)
    assert len(corpus) == 2
    samples = [corpus[i] for i in range(len(corpus))]
    assert {r.annotations["status"] for r in samples} == {
        "generated",
        "generation_failed",
    }
    assert all(r.annotations["score"] is None for r in samples)
    training = CorpusSupervisedDataset(corpus)
    assert len(training) == 1
    assert training[0].annotations["quality_trainable"] is False
    prepared = prepare_supervised_record(
        training[0], SimpleNamespace(sft_process_sample=lambda x: {"messages": x})
    )
    assert prepared["messages"][-1] == {"role": "assistant", "content": "answer"}
    assert prepared["_execution_record"].producer.name == "teacher"


def test_manifest_path_escape_is_rejected(tmp_path):
    writer = CorpusWriter(tmp_path, run_id="run")
    writer.commit(record())
    manifest = next((tmp_path / "records").iterdir())
    payload = json.loads(manifest.read_text())
    payload["record"]["items"]["targets"]["path"] = "../outside.pt"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="escapes"):
        CorpusDataset(tmp_path)[0]


def test_recorded_rl_does_not_accept_missing_behavior_information():
    with pytest.raises(ValueError, match="behavior"):
        record().require_policy_gradient()


def test_group_standardization_uses_complete_original_groups(tmp_path):
    writer = CorpusWriter(tmp_path, run_id="run")
    for i, score in enumerate((1.0, 3.0)):
        item = record(i)
        item.group_size, item.group_slot = 2, i
        item.annotations["score"] = score
        writer.commit(item)
    dataset = CorpusSupervisedDataset(CorpusDataset(tmp_path))
    dataset.validate_policy_groups(advantages="group_standardized")
    values = {
        dataset[i].sample_id: dataset[i].annotations["group_advantage"]
        for i in range(len(dataset))
    }
    assert values["0"] == pytest.approx(-1)
    assert values["1"] == pytest.approx(1)
    assert dataset[0].producer.behavior_version is None


def test_incomplete_policy_group_is_rejected(tmp_path):
    writer = CorpusWriter(tmp_path, run_id="run")
    item = record()
    item.group_size, item.group_slot = 2, 0
    writer.commit(item)
    dataset = CorpusSupervisedDataset(CorpusDataset(tmp_path))
    with pytest.raises(ValueError, match="incomplete"):
        dataset.validate_policy_groups()


def test_record_scoring_reuses_reference_free_callbacks_without_an_engine(tmp_path):
    from cosmos_rl.execution.source import RecordScorer
    from cosmos_rl.policy.config import Config

    configuration = Config.model_validate(
        {
            "train": {"train_policy": {"type": "sft"}},
            "policy": {"parallelism": {"tp_size": 1}},
            "execution": {
                "path": "train_from_data",
                "source": {
                    "kind": "corpus",
                    "path": str(tmp_path),
                    "score_records": True,
                },
            },
        }
    )
    seen = []

    def score(completion, reference, **kwargs):
        seen.append(reference)
        return float(completion.sum()), {"source": "owned-record"}

    scorer = RecordScorer(configuration, SimpleNamespace(), [score])
    item = record()
    scorer(item)
    assert seen == [None]
    assert item.annotations["score"] == pytest.approx(0.3)
