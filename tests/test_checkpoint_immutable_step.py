# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Same-step final promotion must preserve an immutable resumable snapshot."""

from pathlib import Path

import pytest
import torch

from cosmos_rl.utils.parallelism import ParallelDims
from test_checkpoint_commit_contract import components, manager_at


def saved_files(path):
    return {
        file.name: file.read_bytes() for file in Path(path).iterdir() if file.is_file()
    }


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("conflict", ["model", "optimizer", "scheduler", "progress"])
def test_conflicting_same_step_cannot_modify_committed_state(tmp_path, mode, conflict):
    manager = manager_at(tmp_path, mode=mode)
    model, optimizer, scheduler = components()
    path = manager.save_checkpoint(
        model, optimizer, scheduler, 1, 10, cursor=3, is_final=False
    )
    manager._wait_for_pending_async_saves()
    manager.save_check(1)
    before = saved_files(path)
    cursor = 3
    if conflict == "model":
        with torch.no_grad():
            model.weight.add_(1)
    elif conflict == "optimizer":
        optimizer.param_groups[0]["lr"] *= 2
    elif conflict == "scheduler":
        scheduler.last_epoch += 1
    else:
        cursor += 1
    try:
        with pytest.raises(ValueError, match="immutable"):
            manager.save_checkpoint(
                model, optimizer, scheduler, 1, 10, cursor=cursor, is_final=True
            )
        assert saved_files(path) == before
        assert manager.ckpt_path_check(path)
    finally:
        manager.finalize()


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_final_promotion_reuses_files_and_original_resume_metadata(
    tmp_path, monkeypatch, mode
):
    manager = manager_at(tmp_path, mode=mode)
    model, optimizer, scheduler = components()
    path = manager.save_checkpoint(
        model, optimizer, scheduler, 1, 10, cursor=3, is_final=False
    )
    manager._wait_for_pending_async_saves()
    manager.save_check(1)
    before = saved_files(path)
    # Validation may advance RNG without performing another optimizer update.
    torch.rand(3)
    manager.invalidate_completion_marker(1)  # actual RL final-save hook
    assert manager.ckpt_path_check(path)

    def unexpected_save(*args, **kwargs):
        raise OSError("promotion must not serialize over committed artifacts")

    monkeypatch.setattr(torch, "save", unexpected_save)
    assert (
        manager.save_checkpoint(
            model, optimizer, scheduler, 1, 10, cursor=3, is_final=True
        )
        == path
    )
    manager.save_check(1, val_score=1.0)
    manager.finalize()
    assert saved_files(path) == before
    assert manager.saved_ckpt_step_dirs == [str(Path(path).parent)]
    assert (
        torch.load(Path(path) / "extra_info_rank_0.pth", weights_only=False)["is_final"]
        is False
    )
    manager.config.train.resume = path
    restored_model, restored_optimizer, restored_scheduler = components()
    metadata, _ = manager.load_checkpoint(
        restored_model, restored_optimizer, restored_scheduler, "unused"
    )
    assert metadata["cursor"] == 3
    torch.testing.assert_close(restored_model.state_dict(), model.state_dict())


@pytest.mark.parametrize("fail_upload", [False, True])
def test_final_only_upload_uses_existing_files_and_preserves_them_on_error(
    tmp_path, monkeypatch, fail_upload
):
    from cosmos_rl.utils import checkpoint

    manager = manager_at(tmp_path)
    manager.config.train.ckpt.upload_s3 = "final"
    manager.config.train.ckpt.s3_bucket = "test-bucket"
    manager.ckpt_s3_output_dir = "checkpoints"
    model, optimizer, scheduler = components()
    path = manager.save_checkpoint(model, optimizer, scheduler, 1, 10, is_final=False)
    before = saved_files(path)
    uploads = []

    def upload(**kwargs):
        uploads.append(Path(kwargs["local_file_path"]).name)
        if fail_upload:
            raise OSError("injected upload failure")

    monkeypatch.setattr(checkpoint, "upload_file_to_s3", upload)
    if fail_upload:
        with pytest.raises(OSError, match="injected"):
            manager.save_checkpoint(model, optimizer, scheduler, 1, 10, is_final=True)
    else:
        manager.save_checkpoint(model, optimizer, scheduler, 1, 10, is_final=True)
        assert len(uploads) == 4
    assert uploads
    assert saved_files(path) == before
    assert manager.ckpt_path_check(path)


def test_staggered_final_promotion_never_mixes_rank_attempts(tmp_path):
    dims = ParallelDims(
        dp_replicate=1,
        dp_shard=2,
        cp=1,
        tp=1,
        pp=1,
        world_size=2,
        pp_dynamic_shape=False,
    )
    managers = [manager_at(tmp_path, rank=rank, dims=dims) for rank in range(2)]
    model, optimizer, scheduler = components()
    for manager in managers:
        path = manager.save_checkpoint(
            model, optimizer, scheduler, 1, 10, is_final=False
        )
    before = saved_files(path)
    for manager in managers:
        manager.invalidate_completion_marker(1)
        manager.save_checkpoint(model, optimizer, scheduler, 1, 10, is_final=True)
        assert saved_files(path) == before
        assert manager.ckpt_path_check(path)


def test_partial_rank_artifacts_are_not_overwritten(tmp_path, monkeypatch):
    manager = manager_at(tmp_path)
    model, optimizer, scheduler = components()
    original = torch.save

    def fail_scheduler(state, path):
        if "scheduler_rank" in str(path):
            raise OSError("injected")
        original(state, path)

    with monkeypatch.context() as context:
        context.setattr(torch, "save", fail_scheduler)
        with pytest.raises(OSError, match="injected"):
            manager.save_checkpoint(model, optimizer, scheduler, 1, 10)
    path = Path(manager.ckpt_output_dir) / "step_1" / "policy"
    before = saved_files(path)
    with pytest.raises(ValueError, match="incomplete rank artifacts"):
        manager.save_checkpoint(model, optimizer, scheduler, 1, 10)
    assert saved_files(path) == before
