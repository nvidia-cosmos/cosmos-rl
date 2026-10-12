# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resume discovery does not confer ownership of another run's files."""

import json
from pathlib import Path

import pytest

from cosmos_rl.policy.config import Config
from cosmos_rl.utils.checkpoint import CheckpointMananger
from cosmos_rl.utils.parallelism import ParallelDims


def _config(root, run):
    return Config.from_dict(
        {
            "train": {
                "output_dir": str(root / run),
                "timestamp": run,
                "resume": True,
                "ckpt": {
                    "enable_checkpoint": True,
                    "save_mode": "sync",
                    "max_keep": 1,
                },
            }
        }
    )


def _checkpoint(root, run, step, ranks):
    policy = root / run / "checkpoints" / f"step_{step}" / "policy"
    policy.mkdir(parents=True)
    (policy / "cosmos_config").write_text("{}")
    for rank in ranks:
        (policy / f".rank_{rank}_complete").touch()
    # An unfinished save owns data before it publishes a completion marker.
    (policy / "model_rank_0.pth").write_bytes(b"owned by another writer")
    return policy


def _files(root):
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("saving_ranks", [[], [0]])
def test_discovery_preserves_incomplete_and_other_topology_files(
    tmp_path, saving_ranks
):
    policy = _checkpoint(tmp_path, "previous", 10, saving_ranks)
    before = _files(policy.parent)
    # Both pure DP and two-way sharding are supported; an older one-rank save
    # is not a complete two-shard save, but is not ours to delete either.
    dims = ParallelDims(
        dp_replicate=1,
        dp_shard=2,
        cp=1,
        tp=1,
        pp=1,
        world_size=2,
        pp_dynamic_shape=False,
    )
    manager = CheckpointMananger(_config(tmp_path, "current"), dims)
    assert not manager.ckpt_path_check(str(policy))
    assert _files(policy.parent) == before
    assert manager.saved_ckpt_step_dirs == []


def test_retention_only_deletes_checkpoints_owned_by_current_run(tmp_path):
    previous = _checkpoint(tmp_path, "previous", 1, [0])
    before = _files(previous.parent)
    manager = CheckpointMananger(_config(tmp_path, "current"))
    # Cross-run discovery remains available to resume, but not to retention.
    assert str(previous) in manager.get_latest_ckpt_paths()
    for step in (2, 3):
        _checkpoint(tmp_path, "current", step, [0])
        manager.save_check(step)
    assert _files(previous.parent) == before
    assert not (tmp_path / "current/checkpoints/step_2").exists()
    assert [Path(p).name for p in manager.saved_ckpt_step_dirs] == ["step_3"]


def test_discovery_preserves_current_run_incomplete_save(tmp_path):
    policy = _checkpoint(tmp_path, "current", 10, [])
    before = _files(policy.parent)
    manager = CheckpointMananger(_config(tmp_path, "current"))
    assert manager.saved_ckpt_step_dirs == []
    assert _files(policy.parent) == before


def test_pipeline_plus_replicated_dp_is_rejected_before_checkpointing():
    # The external report's non-contiguous {0,1,4,5} example is not supported.
    with pytest.raises(ValueError, match="dp_replicate must be 1 when pp > 1"):
        ParallelDims(
            dp_replicate=2,
            dp_shard=1,
            cp=1,
            tp=2,
            pp=2,
            world_size=8,
            pp_dynamic_shape=False,
        )


@pytest.mark.parametrize("relative_output", [False, True])
@pytest.mark.parametrize("symlink_output", [False, True])
def test_best_links_resolve_and_retain_protection_after_restart(
    tmp_path, monkeypatch, relative_output, symlink_output
):
    monkeypatch.chdir(tmp_path)
    if symlink_output:
        (tmp_path / "storage").mkdir()
        (tmp_path / "runs").symlink_to(tmp_path / "storage", target_is_directory=True)
    root = Path("runs") if relative_output else tmp_path / "runs"
    config = _config(root, "current")
    config.train.ckpt.export_safetensors = True
    manager = CheckpointMananger(config)
    best = _checkpoint(root, "current", 1, [0]).parent
    exported = root / "current" / "safetensors" / "step_1"
    exported.mkdir(parents=True)
    manager.save_check(1, val_score=0.25)
    assert (root / "best" / "checkpoints").resolve() == best.resolve()
    assert (root / "best" / "safetensors").resolve() == exported.resolve()
    metadata_path = root / "best" / "best_score.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["best_ckpt_abs_dir"] == str(best.resolve())
    # Old versions saved the alias spelling. Accept it when it still identifies
    # the same committed artifact; do not require rewriting old metadata.
    if symlink_output:
        metadata["best_ckpt_abs_dir"] = str(best.absolute())
        metadata_path.write_text(json.dumps(metadata))
    restarted = CheckpointMananger(config)
    assert restarted.best_score == 0.25
    assert restarted.best_ckpt_abs_dir == str(best.resolve())
    _checkpoint(root, "current", 2, [0])
    restarted.save_check(2, val_score=0.5)
    assert best.exists() and exported.exists()
