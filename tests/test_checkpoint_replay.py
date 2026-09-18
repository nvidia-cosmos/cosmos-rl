# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from cosmos_rl.dispatcher.data.checkpoint_manifest import CheckpointManifest
from cosmos_rl.dispatcher.data.replay import SamplingBoundary, SamplingReplayLedger
from test_controller_resume_adapter import metadata, build, adapter_for, CursorSampler


def boundary(cursor, *, epoch=1):
    return SamplingBoundary(
        epoch=epoch, sampler_state={"cursor": cursor}, remaining_completions=10 - cursor
    )


def test_out_of_order_work_replays_without_skipping_and_preserves_budget():
    ledger = SamplingReplayLedger(boundary(0))
    tokens = [ledger.issue(boundary(i + 1)) for i in range(5)]
    assert ledger.settle(tokens[0])
    assert ledger.settle(tokens[2])
    assert ledger.settle(tokens[4])
    safe = ledger.snapshot()
    assert safe.sampler_state == {"cursor": 1}
    assert safe.remaining_completions == 9  # includes repeats 2 and 4
    state = metadata(
        epoch=safe.epoch,
        sampler_state=safe.sampler_state,
        remaining_completions=safe.remaining_completions,
    )
    resumed = build(adapter_for(state), CursorSampler())
    indices, _ = next(resumed.train_dataloader_iter)
    assert list(indices) == [1, 2]
    assert resumed.remain_samples_num == 9
    fresh = SamplingReplayLedger(safe)
    assert not fresh.settle(tokens[1])  # old execution cannot advance replay
    assert fresh.snapshot() == safe


def test_partial_group_duplicate_and_filtered_completion():
    ledger = SamplingReplayLedger(boundary(0))
    token = ledger.issue(boundary(3), completions=3)
    assert ledger.settle(token, 2)
    assert not ledger.settle(token, 2)
    assert ledger.settle(token, 0)
    assert ledger.snapshot() == boundary(0)
    # The final completion may be a terminal filter/discard instead of training.
    assert ledger.settle(token, 1)
    assert ledger.snapshot() == boundary(3)
    assert not ledger.settle(token, 1)


def test_epoch_transition_and_snapshots_do_not_alias_live_state():
    ledger = SamplingReplayLedger(boundary(0))
    after = boundary(1, epoch=2)
    token = ledger.issue(after)
    after.sampler_state["cursor"] = 999
    ledger.settle(token)
    saved = ledger.snapshot()
    assert saved.epoch == 2
    assert saved.sampler_state == {"cursor": 1}
    saved.sampler_state["cursor"] = 999
    assert ledger.snapshot().sampler_state == {"cursor": 1}


def test_all_outstanding_and_invalid_accounting():
    ledger = SamplingReplayLedger(boundary(0))
    with pytest.raises(ValueError, match="every issued"):
        ledger.issue(boundary(2))
    token = ledger.issue(boundary(1))
    with pytest.raises(ValueError, match="outside"):
        ledger.settle(token, 1)
    assert ledger.snapshot() == boundary(0)


def publish(tmp_path):
    (tmp_path / "trainer-0.pt").write_bytes(b"complete trainer state")
    return CheckpointManifest.publish(
        tmp_path,
        metadata(checkpoint_path=str(tmp_path)),
        required_artifacts={"trainer-0.pt"},
        compatibility={"dataset": "v1", "world_size": 1},
    )


def load(tmp_path, **changes):
    return CheckpointManifest.load(
        tmp_path,
        **(
            {
                "required_artifacts": {"trainer-0.pt"},
                "compatibility": {"dataset": "v1", "world_size": 1},
            }
            | changes
        ),
    )


def test_committed_manifest_round_trip_and_no_overwrite(tmp_path):
    saved = publish(tmp_path)
    assert load(tmp_path) == saved
    with pytest.raises(FileExistsError):
        publish(tmp_path)
    assert not list(tmp_path.glob(".manifest-*"))


def test_interrupted_save_is_not_a_checkpoint(tmp_path):
    (tmp_path / "trainer-0.pt").write_bytes(b"partial")
    with pytest.raises(FileNotFoundError):
        load(tmp_path)
    with pytest.raises(ValueError, match="Missing"):
        CheckpointManifest.publish(
            tmp_path,
            metadata(checkpoint_path=str(tmp_path)),
            required_artifacts={"trainer-0.pt", "trainer-1.pt"},
            compatibility={},
        )
    assert not (tmp_path / "manifest.json").exists()


def test_interrupted_publication_never_exposes_manifest(tmp_path, monkeypatch):
    from unittest.mock import Mock

    monkeypatch.setattr(
        "cosmos_rl.dispatcher.data.checkpoint_manifest.os.link",
        Mock(side_effect=OSError("interrupted publication")),
    )
    with pytest.raises(OSError, match="interrupted publication"):
        publish(tmp_path)
    assert not list(tmp_path.glob(".manifest-*"))
    with pytest.raises(FileNotFoundError):
        load(tmp_path)


def test_same_counters_from_different_save_fail_agreement():
    state = metadata()
    fetcher = build(adapter_for(state), CursorSampler())
    other_save = state.to_checkpoint_extra_info() | {"checkpoint_id": "different-save"}
    with pytest.raises((AssertionError, ValueError)):
        fetcher.validate_after_resume(other_save)


@pytest.mark.parametrize("mutation", ["mixed", "missing", "symlink"])
def test_bad_artifact_rejected(tmp_path, mutation):
    publish(tmp_path)
    artifact = tmp_path / "trainer-0.pt"
    if mutation == "mixed":
        artifact.write_bytes(b"state from a different checkpoint")
    else:
        artifact.unlink()
        if mutation == "symlink":
            artifact.symlink_to(tmp_path / "manifest.json")
    with pytest.raises(ValueError):
        load(tmp_path)


def test_incompatible_dataset_or_missing_required_shard(tmp_path):
    publish(tmp_path)
    with pytest.raises(ValueError, match="incompatible"):
        load(tmp_path, compatibility={"dataset": "v2", "world_size": 1})
    with pytest.raises(ValueError, match="required artifact"):
        load(tmp_path, required_artifacts={"trainer-0.pt", "trainer-1.pt"})


def test_unknown_manifest_schema_and_unsafe_paths(tmp_path):
    publish(tmp_path)
    path = tmp_path / "manifest.json"
    content = json.loads(path.read_text())
    content["schema_version"] = 2
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError):
        load(tmp_path)
    with pytest.raises(ValueError, match="Invalid"):
        CheckpointManifest._artifact(tmp_path, "../outside")
