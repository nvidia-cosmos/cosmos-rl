# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Durable payload plugins preserve metadata-only control and byte admission."""

import json

import pytest
import torch

from cosmos_rl.execution import corpus
from cosmos_rl.execution.corpus import CorpusDataset, CorpusWriter
from cosmos_rl.execution.records import SampleRecord
from cosmos_rl.execution.source import CorpusSource
from cosmos_rl.execution.storage import (
    DataReference,
    DataSaveError,
    FileSystemDataStore,
    save_data,
)


def test_filesystem_payload_is_saved_once_and_read_by_application(
    tmp_path, monkeypatch
):
    store = FileSystemDataStore(tmp_path)
    data = {"observations": torch.arange(32).reshape(8, 4), "actions": torch.ones(8, 2)}
    reference = save_data(store, data, max_bytes=1024**2)
    data["observations"].zero_()
    sample = SampleRecord(
        sample_id="one", occurrence_id="one", inputs={}, targets=reference
    )
    assert isinstance(sample.model_dump()["targets"], DataReference)
    with monkeypatch.context() as patch:
        patch.setattr(
            torch, "load", lambda *a, **k: pytest.fail("control loaded payload")
        )
        patch.setattr(
            torch, "save", lambda *a, **k: pytest.fail("control rewrote payload")
        )
        patch.setattr(
            corpus, "_file_digest", lambda *a: pytest.fail("control hashed payload")
        )
        assert CorpusWriter(tmp_path, run_id="test").commit(sample)
        read = CorpusDataset(tmp_path)[0]
        assert read.targets.to_dict() == reference.to_dict()
    actual = store.load(read.targets, max_bytes=1024**2)
    assert actual["observations"].sum() == 496
    assert actual["observations"].device.type == "cpu"
    assert not list((tmp_path / "artifacts").glob(".pending-*"))


def test_custom_saver_needs_no_framework_registration(tmp_path):
    class JsonSaver:
        def save(self, value, *, max_bytes):
            encoded = json.dumps(value).encode()
            assert len(encoded) <= max_bytes
            target = tmp_path / "application.json"
            target.write_bytes(encoded)
            return DataReference(
                uri=target.as_uri(),
                format="application.json.v1",
                size_bytes=len(encoded),
            )

        def close(self):
            pass

    reference = save_data(JsonSaver(), {"actions": [1, 2]}, max_bytes=1024)
    assert reference.format == "application.json.v1"
    assert json.loads((tmp_path / "application.json").read_text()) == {
        "actions": [1, 2]
    }


def test_byte_admission_uses_payload_size_instead_of_pointer_size(
    tmp_path, monkeypatch
):
    reference = DataReference(
        uri="custom://large", format="custom", size_bytes=64 * 1024**2
    )
    CorpusWriter(tmp_path, run_id="test", max_record_bytes=128 * 1024**2).commit(
        SampleRecord(sample_id="one", occurrence_id="one", inputs={}, targets=reference)
    )
    dataset = CorpusDataset(tmp_path, max_record_bytes=128 * 1024**2)
    monkeypatch.setattr(
        CorpusDataset, "__getitem__", lambda *a: pytest.fail("loaded oversized data")
    )
    with pytest.raises(ValueError, match="byte budget"):
        CorpusSource(dataset, max_batch_bytes=32 * 1024**2).reserve(1)


def test_filesystem_checks_total_raw_bytes_before_any_copy(tmp_path, monkeypatch):
    store = FileSystemDataStore(tmp_path)
    monkeypatch.setattr(
        torch, "save", lambda *a, **k: pytest.fail("serialized oversized data")
    )
    with pytest.raises(ValueError, match="before serialization"):
        store.save([torch.ones(8), torch.ones(8)], max_bytes=48)
    assert not (tmp_path / "artifacts").exists()


def test_reader_checks_checksums_and_declared_bytes(tmp_path):
    store = FileSystemDataStore(tmp_path)
    reference = store.save(torch.arange(8), max_bytes=1024**2)
    with pytest.raises(ValueError, match="byte budget"):
        store.load(reference, max_bytes=1)
    path = next((tmp_path / "artifacts").glob("*.pt"))
    encoded = path.read_bytes()
    path.write_bytes(bytes([encoded[0] ^ 1]) + encoded[1:])
    with pytest.raises(ValueError, match="checksum"):
        store.load(reference, max_bytes=1024**2)


def test_write_failure_removes_temporary_payload_and_never_commits(
    tmp_path, monkeypatch
):
    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        FileSystemDataStore(tmp_path).save(torch.ones(8), max_bytes=1024**2)
    assert not list(tmp_path.glob("**/.pending-*"))
    assert not list(tmp_path.glob("**/*.pt"))
    assert not (tmp_path / "records").exists()


def test_async_saver_and_oversized_metadata_are_rejected():
    class AsyncSaver:
        async def save(self, value, *, max_bytes):
            return None

    with pytest.raises(DataSaveError, match="synchronously"):
        save_data(AsyncSaver(), None, max_bytes=1024)
    with pytest.raises(ValueError, match="64 KiB"):
        DataReference(
            uri="custom://one",
            format="custom",
            size_bytes=1,
            metadata={"payload": "x" * 65536},
        )


def test_small_contiguous_view_does_not_persist_large_backing_storage(tmp_path):
    store = FileSystemDataStore(tmp_path)
    backing = torch.arange(1024**2)
    reference = store.save(backing[16:24], max_bytes=4096)
    assert reference.size_bytes < 4096
    assert torch.equal(store.load(reference, max_bytes=4096), backing[16:24])


def test_required_storage_failure_is_not_discarded_as_an_episode(tmp_path, monkeypatch):
    from cosmos_rl.rollout.generation_mixin import RolloutGenerationMixin

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(DataSaveError, match="disk full") as failed:
        save_data(FileSystemDataStore(tmp_path), torch.ones(8), max_bytes=1024**2)
    with pytest.raises(DataSaveError, match="disk full"):
        RolloutGenerationMixin._on_generation_error(object(), failed.value, [])
