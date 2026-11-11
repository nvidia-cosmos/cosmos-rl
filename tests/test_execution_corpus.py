# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import threading
from types import SimpleNamespace

import pytest
import torch

from cosmos_rl.execution import corpus as storage
from cosmos_rl.execution.corpus import CorpusDataset, CorpusWriter
from cosmos_rl.execution.records import ModelIdentity, SampleRecord
from cosmos_rl.execution.sink import CorpusResultSink
from cosmos_rl.execution.source import (
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


def test_preparation_context_uses_durable_identity_and_epoch_without_global_rng(
    tmp_path,
):
    import random
    import shutil
    import numpy as np

    writer = CorpusWriter(tmp_path / "original", run_id="preparation")
    writer.commit(record())
    writer.finalize()
    shutil.copytree(tmp_path / "original", tmp_path / "relocated")
    original = CorpusSupervisedDataset(
        CorpusDataset(tmp_path / "original"), preparation_seed=17
    )
    relocated = CorpusSupervisedDataset(
        CorpusDataset(tmp_path / "relocated"), preparation_seed=17
    )
    sample = original[0]
    python_state, numpy_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    context = original.preparation_context(sample)
    assert context == relocated.preparation_context(relocated[0])
    assert original.preparation_context(sample) == context
    assert (
        original.preparation_context(sample.model_copy(update={"attempt_id": "retry"}))
        == context
    )
    assert (
        original.preparation_context(
            sample.model_copy(update={"occurrence_id": "another"})
        )
        != context
    )
    original.set_epoch(1)
    assert original.preparation_context(sample) != context
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_owned_tensor_roundtrip_and_typed_mapping(tmp_path):
    sample = record()
    sample.inputs["literal"] = {"kind": "tensor", "path": "ordinary application data"}
    writer = CorpusWriter(tmp_path, run_id="run")
    assert writer.commit(sample)
    sample.inputs["image"].zero_()
    sample.targets.zero_()
    result = CorpusDataset(tmp_path, allow_partial=True)[0]
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
    assert len(CorpusDataset(tmp_path, allow_partial=True)) == 1
    assert torch.allclose(
        CorpusDataset(tmp_path, allow_partial=True)[0].targets, torch.tensor([0.1, 0.2])
    )


def test_duplicate_commit_fences_an_unacknowledged_winner(tmp_path, monkeypatch):
    import os

    writer = CorpusWriter(tmp_path, run_id="duplicate-durability")
    fsync = os.fsync
    fences = []

    def unavailable(fd):
        target = os.readlink(f"/proc/self/fd/{fd}")
        if target == str(tmp_path / "records"):
            fences.append(target)
            raise OSError("manifest directory unavailable")
        return fsync(fd)

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", unavailable)
        # The first failure leaves a linked, unacknowledged manifest. A
        # concurrent duplicate or a retry must not treat visibility as durability.
        for _ in range(2):
            with pytest.raises(OSError, match="manifest directory unavailable"):
                writer.commit(record())
    assert len(fences) == 2
    assert not writer.commit(record())
    writer.finalize(expected_records=1)
    assert len(CorpusDataset(tmp_path)) == 1


def test_finalization_excludes_active_and_future_publications(tmp_path, monkeypatch):
    writer = CorpusWriter(tmp_path, run_id="snapshot-boundary")
    publishing = threading.Event()
    release = threading.Event()
    finalizing = threading.Event()
    publish, flock = storage._publish, storage.fcntl.flock

    def paused(path, data):
        if path.parent.name == "records":
            publishing.set()
            assert release.wait(timeout=5)
        return publish(path, data)

    def observed(fd, flags):
        if flags == storage.fcntl.LOCK_EX:
            finalizing.set()
        return flock(fd, flags)

    monkeypatch.setattr(storage, "_publish", paused)
    monkeypatch.setattr(storage.fcntl, "flock", observed)
    with ThreadPoolExecutor(2) as workers:
        commit = workers.submit(writer.commit, record())
        try:
            assert publishing.wait(timeout=5)
            complete = workers.submit(writer.finalize, expected_records=1)
            assert finalizing.wait(timeout=5)
            assert not complete.done()
            assert not (tmp_path / "complete.json").exists()
        finally:
            release.set()
        assert commit.result(timeout=5)
        # NFS may delay a blocking lock grant even after the holder releases it.
        complete.result(timeout=45)
    assert len(CorpusDataset(tmp_path)) == 1
    with pytest.raises(ValueError, match="completed corpus"):
        writer.commit(record(1))


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
    assert len(CorpusDataset(tmp_path, allow_partial=True)) == 0
    assert writer.commit(record())
    assert len(CorpusDataset(tmp_path, allow_partial=True)) == 1


def test_corrupt_artifact_and_changed_snapshot_are_rejected(tmp_path):
    writer = CorpusWriter(tmp_path, run_id="run")
    writer.commit(record())
    original = CorpusDataset(tmp_path, allow_partial=True).revision
    artifact = next((tmp_path / "artifacts").iterdir())
    data = artifact.read_bytes()
    artifact.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    with pytest.raises(ValueError, match="checksum"):
        CorpusDataset(tmp_path, allow_partial=True)[0]
    artifact.write_bytes(data)
    writer.commit(record(1))
    with pytest.raises(ValueError, match="revision"):
        CorpusDataset(tmp_path, revision=original, allow_partial=True)


def test_manifest_decodes_exactly_the_verified_buffer(tmp_path, monkeypatch):
    from pathlib import Path

    writer = CorpusWriter(tmp_path, run_id="run")
    sample = record()
    sample.targets = "original"
    writer.commit(sample)
    writer.finalize(expected_records=1)
    dataset = CorpusDataset(tmp_path)
    path = dataset.paths[0]
    read_bytes = Path.read_bytes
    reads = []

    def mutate_after_read(target):
        assert target != tmp_path / "dataset.json", (
            "Snapshot must reuse dataset identity"
        )
        data = read_bytes(target)
        if target == path:
            reads.append(target)
            changed = json.loads(data)
            changed["record"]["items"]["targets"] = "changed"
            target.write_text(json.dumps(changed))
        return data

    monkeypatch.setattr(Path, "read_bytes", mutate_after_read)
    assert dataset[0].targets == "original"
    assert reads == [path]
    with pytest.raises(ValueError, match="manifest changed"):
        dataset[0]


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
    corpus = CorpusDataset(tmp_path, allow_partial=True)
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
        CorpusDataset(tmp_path, allow_partial=True)[0]


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
    dataset = CorpusSupervisedDataset(CorpusDataset(tmp_path, allow_partial=True))
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
    dataset = CorpusSupervisedDataset(CorpusDataset(tmp_path, allow_partial=True))
    with pytest.raises(ValueError, match="incomplete"):
        dataset.validate_policy_groups(advantages="group_standardized")


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


def test_complete_snapshot_requires_finalization_and_rejects_new_records(tmp_path):
    writer = CorpusWriter(tmp_path, run_id="run")
    writer.commit(record())
    with pytest.raises(ValueError, match="not complete"):
        CorpusDataset(tmp_path)
    with pytest.raises(ValueError, match="count"):
        writer.finalize(expected_records=2)
    assert not (tmp_path / "complete.json").exists()
    revision = writer.finalize(expected_records=1)
    assert CorpusDataset(tmp_path).revision == revision
    assert writer.finalize(expected_records=1) == revision
    assert not writer.commit(record())
    with pytest.raises(ValueError, match="cannot accept"):
        writer.commit(record(1))


def test_completed_snapshot_rejects_manifest_mutation(tmp_path):
    writer = CorpusWriter(tmp_path, run_id="run")
    writer.commit(record())
    writer.finalize()
    manifest = next((tmp_path / "records").glob("*.json"))
    value = json.loads(manifest.read_bytes())
    value["record"]["items"]["sample_id"] = "changed"
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="manifests changed"):
        CorpusDataset(tmp_path)


def test_failed_completion_publication_is_retryable_without_payload_io(
    tmp_path, monkeypatch
):
    writer = CorpusWriter(tmp_path, run_id="run")
    writer.commit(record())
    original = storage._publish
    with monkeypatch.context() as patch:

        def fail(path, data):
            if path.name == "complete.json":
                raise OSError("disk full")
            return original(path, data)

        patch.setattr(storage, "_publish", fail)
        patch.setattr(
            torch, "load", lambda *a, **k: pytest.fail("completion loaded tensor")
        )
        with pytest.raises(OSError, match="disk full"):
            writer.finalize()
    assert not (tmp_path / "complete.json").exists()
    writer.finalize()
    assert len(CorpusDataset(tmp_path)) == 1


@pytest.mark.parametrize("grouped", [False, True])
def test_supplied_advantages_retain_single_sample_policy_gradient(tmp_path, grouped):
    from cosmos_rl.execution.token_objectives import TokenObjectiveBatch
    from cosmos_rl.execution.config import ExecutionConfig

    item = SampleRecord(
        sample_id="one",
        occurrence_id="one",
        inputs={},
        targets={"token_ids": [0, 1, 2], "label_ids": [0, 1, 2]},
        producer=ModelIdentity(role="actor", name="behavior"),
        group_id="one" if grouped else None,
        group_size=1 if grouped else None,
        group_slot=0 if grouped else None,
        behavior_logprobs=[-1.0, -1.0],
        masks={"policy_gradient": [True, True]},
        annotations={"advantages": [1.0, 1.0]},
    )
    writer = CorpusWriter(tmp_path, run_id="run")
    writer.commit(item)
    writer.finalize()
    dataset = CorpusSupervisedDataset(CorpusDataset(tmp_path))
    dataset.validate_policy_groups(advantages="recorded")
    row = prepare_supervised_record(dataset[0], SimpleNamespace())
    config = SimpleNamespace(
        execution=ExecutionConfig(
            path="train_from_data", objectives={"policy_gradient": 1}
        ),
        policy=SimpleNamespace(model_max_length=3),
    )
    batch = TokenObjectiveBatch([row], config, device="cpu", vocab_size=3)
    assert batch.window.counts == (2.0,)
    logits = torch.zeros(1, 3, 3, requires_grad=True)
    batch.loss(logits, torch.tensor([[0, 1, 2]]), start=0).backward()
    assert logits.grad.abs().sum() > 0
