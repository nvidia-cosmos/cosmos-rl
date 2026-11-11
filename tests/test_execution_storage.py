# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Durable payload plugins preserve metadata-only control and byte admission."""

from cosmos_rl.dispatcher.data.schema import Rollout
import json

import pytest
import torch

from cosmos_rl.execution import corpus
from cosmos_rl.execution.corpus import CorpusDataset, CorpusWriter
from cosmos_rl.execution.records import SampleRecord
from cosmos_rl.execution.source import CorpusSupervisedDataset
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
        sample_id="one",
        occurrence_id="one",
        rollout=Rollout(prompt={}, completion=reference),
    )
    assert isinstance(sample.model_dump()["rollout"]["completion"], DataReference)
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
        read = CorpusDataset(tmp_path, allow_partial=True)[0]
        assert read.rollout.completion.to_dict() == reference.to_dict()
    actual = store.load(read.rollout.completion, max_bytes=1024**2)
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
        SampleRecord(
            sample_id="one",
            occurrence_id="one",
            rollout=Rollout(prompt={}, completion=reference),
        )
    )
    dataset = CorpusDataset(
        tmp_path, max_record_bytes=128 * 1024**2, allow_partial=True
    )
    monkeypatch.setattr(
        CorpusDataset, "__getitem__", lambda *a: pytest.fail("loaded oversized data")
    )
    view = CorpusSupervisedDataset(dataset, max_batch_bytes=32 * 1024**2)
    with pytest.raises(ValueError, match="byte budget"):
        view.check_batch_budget([0])


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
    assert not list((tmp_path / "records").glob("*.json"))


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


@pytest.mark.parametrize(
    "layout", ["contiguous", "view", "transposed", "reversed", "readonly"]
)
def test_numpy_save_finishes_borrowed_reads_and_returns_owned_data(tmp_path, layout):
    import numpy as np

    backing = np.arange(120, dtype=np.float32).reshape(10, 12)
    value = {
        "contiguous": backing,
        "view": backing[2:4],
        "transposed": backing.T,
        "reversed": backing[::-1],
        "readonly": backing.view(),
    }[layout]
    if layout == "readonly":
        value.flags.writeable = False
    expected = value.copy()
    store = FileSystemDataStore(tmp_path)
    reference = store.save(value, max_bytes=4096)
    backing.fill(-1)
    result = store.load(reference, max_bytes=4096)
    np.testing.assert_array_equal(result, expected)
    assert result.flags.writeable and not np.shares_memory(result, backing)
    result.fill(0)
    np.testing.assert_array_equal(store.load(reference, max_bytes=4096), expected)


def test_required_storage_failure_is_not_discarded_as_an_episode(tmp_path, monkeypatch):
    from cosmos_rl.rollout.generation_mixin import RolloutGenerationMixin

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(DataSaveError, match="disk full") as failed:
        save_data(FileSystemDataStore(tmp_path), torch.ones(8), max_bytes=1024**2)
    with pytest.raises(DataSaveError, match="disk full"):
        RolloutGenerationMixin._on_generation_error(object(), failed.value, [])


def test_filesystem_reader_can_remap_an_unchanged_reference(tmp_path):
    import shutil

    original = tmp_path / "original"
    reference = FileSystemDataStore(original).save(torch.arange(8), max_bytes=4096)
    relocated = tmp_path / "relocated"
    shutil.copytree(original, relocated)
    shutil.rmtree(original)
    assert torch.equal(
        FileSystemDataStore(relocated).load(reference, max_bytes=4096), torch.arange(8)
    )


@pytest.mark.parametrize("value", ["x" * 16384, {"text": "é" * 8192}, list(range(64))])
def test_inline_payload_is_charged_on_save_and_load(tmp_path, value):
    store = FileSystemDataStore(tmp_path)
    with pytest.raises(ValueError, match="byte budget"):
        store.save(value, max_bytes=1)
    reference = store.save(value, max_bytes=65536)
    assert reference.size_bytes >= len(corpus._json(value))
    assert store.load(reference, max_bytes=reference.size_bytes) == value
    with pytest.raises(ValueError, match="byte budget"):
        store.load(reference, max_bytes=reference.size_bytes - 1)
    understated = reference.to_dict()
    understated["size_bytes"] = 0
    with pytest.raises(ValueError, match="byte budget"):
        store.load(DataReference(**understated), max_bytes=65536)


def test_mixed_payload_budget_includes_inline_data_and_artifacts(tmp_path):
    store = FileSystemDataStore(tmp_path)
    value = {"data": torch.ones(8), "description": "x" * 1024}
    reference = store.save(value, max_bytes=65536)
    artifact_bytes = sum(
        p.stat().st_size for p in (tmp_path / "artifacts").glob("*.pt")
    )
    assert reference.size_bytes > artifact_bytes + 1024
    with pytest.raises(ValueError, match="inline data"):
        store.save(value, max_bytes=reference.size_bytes - 1)
    assert torch.equal(
        store.load(reference, max_bytes=reference.size_bytes)["data"], value["data"]
    )


def test_reader_requires_no_storage_initialization(tmp_path, monkeypatch):
    reference = FileSystemDataStore(tmp_path).save(torch.ones(8), max_bytes=65536)
    monkeypatch.setattr(
        corpus,
        "_initialize_storage",
        lambda *a: pytest.fail("reader initialized storage"),
    )
    assert torch.equal(
        FileSystemDataStore(tmp_path).load(reference, max_bytes=65536), torch.ones(8)
    )
    old_reference = reference.to_dict()
    old_reference["format"] = "cosmos.artifacts.v1"
    with pytest.raises(ValueError, match="different format"):
        FileSystemDataStore(tmp_path).load(
            DataReference(**old_reference), max_bytes=65536
        )


@pytest.mark.parametrize("failed_ancestor", [0, 1, 2])
def test_new_storage_ancestors_are_fenced_and_retry_failure(
    tmp_path, monkeypatch, failed_ancestor
):
    import os

    root = tmp_path / "new-parent" / "new-corpus"
    store = FileSystemDataStore(root)
    real_fsync = os.fsync
    target = (root, root.parent, root.parent.parent)[failed_ancestor]

    def fail_parent(fd):
        if os.readlink(f"/proc/self/fd/{fd}") == str(target):
            raise OSError("storage parent unavailable")
        return real_fsync(fd)

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", fail_parent)
        # Existing directories from the first failed attempt need the same
        # fence; neither save may acknowledge durable publication.
        for _ in range(2):
            with pytest.raises(DataSaveError, match="storage parent unavailable"):
                save_data(store, torch.ones(8), max_bytes=65536)
        with pytest.raises(OSError, match="storage parent unavailable"):
            CorpusWriter(root, run_id="parent-durability")
    reference = save_data(store, torch.ones(8), max_bytes=65536)
    writer = CorpusWriter(root, run_id="parent-durability")
    writer.commit(
        SampleRecord(
            sample_id="one",
            occurrence_id="one",
            rollout=Rollout(prompt={}, completion=reference),
        )
    )
    # The established layout needs no extra ancestor fsync on subsequent saves
    # and commits; each publication still fences its containing directory.
    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", fail_parent)
        store.save(torch.ones(8), max_bytes=65536)
        assert not writer.commit(
            SampleRecord(
                sample_id="one",
                occurrence_id="one",
                rollout=Rollout(prompt={}, completion=reference),
            )
        )
