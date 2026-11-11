"""Durable composition reuses Rollout content without changing its live owners."""

import json

import pytest
import torch

from cosmos_rl.dispatcher.data.schema import Rollout
from cosmos_rl.execution.corpus import CorpusDataset, CorpusWriter
from cosmos_rl.execution.records import ModelIdentity, SampleRecord
from cosmos_rl.execution.storage import FileSystemDataStore


def test_live_defaults_and_borrowed_tensors_survive_durable_composition(tmp_path):
    tensor = torch.arange(24).reshape(6, 4)
    live = Rollout(
        prompt=tensor, completion=tensor[1:], prompt_idx=9, report_metrics={"rate": 3}
    )
    assert (
        live.reward == live.advantage == live.filter_reward == live.weight_version == 0
    )
    record = SampleRecord(sample_id="one", occurrence_id="one", rollout=live)
    assert (
        live.reward == live.advantage == live.filter_reward == live.weight_version == 0
    )
    assert (
        record.rollout.reward
        is record.rollout.advantage
        is record.rollout.weight_version
        is None
    )
    assert record.rollout.prompt is tensor
    assert record.rollout.completion is live.completion
    fields = record.model_dump()["rollout"]
    assert fields["completion"] is live.completion
    assert (
        not {
            "is_end",
            "prompt_idx",
            "teacher_result_uuid",
            "weight_version",
            "producer_model",
            "report_metrics",
        }
        & fields.keys()
    )
    writer = CorpusWriter(tmp_path, run_id="composition")
    writer.commit(record)
    writer.finalize()
    restored = CorpusDataset(tmp_path)[0]
    assert isinstance(restored.rollout, Rollout)
    assert restored.rollout.reward is restored.rollout.advantage is None
    assert torch.equal(restored.rollout.completion, tensor[1:])


def test_explicit_zero_scores_scalar_advantages_and_provenance_roundtrip(tmp_path):
    producer = ModelIdentity(role="actor", name="fixed", behavior_version=7)
    live = Rollout(
        prompt="p",
        completion="c",
        reward=0,
        advantage=0,
        producer_model=producer.model_dump(),
    )
    record = SampleRecord(sample_id="one", occurrence_id="one", rollout=live)
    assert record.producer == producer and record.rollout.weight_version == 7
    writer = CorpusWriter(tmp_path, run_id="zero")
    writer.commit(record)
    restored = CorpusDataset(tmp_path, allow_partial=True)[0]
    assert restored.rollout.reward == restored.rollout.advantage == 0
    assert restored.producer == producer and restored.rollout.weight_version == 7
    with pytest.raises(ValueError, match="identities differ"):
        SampleRecord(
            sample_id="bad",
            occurrence_id="bad",
            rollout=live,
            producer=producer.model_copy(update={"name": "other"}),
        )


@pytest.mark.parametrize(
    "transient", [{"is_end": True}, {"teacher_result_uuid": "redis-request"}]
)
def test_transient_live_results_require_resolution_before_storage(tmp_path, transient):
    with pytest.raises(ValueError, match="durable sample|transient teacher"):
        SampleRecord(sample_id="bad", occurrence_id="bad", rollout=Rollout(**transient))
    record = SampleRecord(
        sample_id="one", occurrence_id="one", rollout=Rollout(completion="value")
    )
    for key, value in transient.items():
        setattr(record.rollout, key, value)
    with pytest.raises(ValueError, match="transient teacher"):
        CorpusWriter(tmp_path, run_id="reject").commit(record)
    assert not list((tmp_path / "records").glob("*.json"))


def test_reference_remains_metadata_through_composed_record(tmp_path, monkeypatch):
    ref = FileSystemDataStore(tmp_path).save(torch.ones(16), max_bytes=65536)
    record = SampleRecord(
        sample_id="one", occurrence_id="one", rollout=Rollout(completion=ref)
    )
    assert record.model_dump()["rollout"]["completion"] is ref
    monkeypatch.setattr(
        torch, "load", lambda *a, **k: pytest.fail("control materialized data")
    )
    monkeypatch.setattr(
        torch, "save", lambda *a, **k: pytest.fail("control rewrote data")
    )
    CorpusWriter(tmp_path, run_id="reference").commit(record)
    assert (
        CorpusDataset(tmp_path, allow_partial=True)[0].rollout.completion.to_dict()
        == ref.to_dict()
    )


def test_pre_release_record_format_is_rejected_explicitly(tmp_path):
    (tmp_path / "dataset.json").write_text(
        json.dumps({"schema_version": 1, "run_id": "old", "metadata": {}})
    )
    with pytest.raises(ValueError, match="regenerate pre-release"):
        CorpusDataset(tmp_path, allow_partial=True)
