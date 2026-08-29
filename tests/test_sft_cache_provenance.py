# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest

from cosmos_rl.tools.prewarm_sft_cache import (
    _configure_conversation_video_decoder,
    _prewarm_indices,
    cache_path,
    combined_cache_fingerprint,
    entry_path,
    finalize_marker,
    finish_distributed_prewarm,
    wait_for_finalize_marker,
    write_finalize_marker,
)


def test_conversation_prewarm_uses_training_video_decoder_contract(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        "cosmos_rl.tools.custom_hooks.tao_sft_example.configure_video_decoder",
        lambda custom: calls.append(custom.video_decoder)
        or {"backend": custom.video_decoder},
    )

    result = _configure_conversation_video_decoder(
        SimpleNamespace(video_decoder="torchvision")
    )

    assert result == {"backend": "torchvision"}
    assert calls == ["torchvision"]


class _ConversationDataset:
    def __init__(self, annotation) -> None:
        self.annotation = annotation

    def __len__(self) -> int:
        return len(self.annotation)


def test_conversation_prewarm_keeps_repeated_media_on_one_rank() -> None:
    dataset = _ConversationDataset(
        [
            {"video": "a.mp4"},
            {"video": "b.mp4"},
            {"video": "a.mp4"},
            {"video": "b.mp4"},
            {"video": "a.mp4"},
            {"video": "c.mp4"},
            {"video": "a.mp4"},
        ]
    )

    rank_zero, strategy_zero = _prewarm_indices(dataset, "wts", 0, 2)
    rank_one, strategy_one = _prewarm_indices(dataset, "wts", 1, 2)

    assert strategy_zero == strategy_one == "media_grouped_balanced"
    assert sorted(rank_zero + rank_one) == list(range(len(dataset)))
    for media in ("a.mp4", "b.mp4", "c.mp4"):
        indices = {
            index
            for index, record in enumerate(dataset.annotation)
            if record["video"] == media
        }
        assert indices <= set(rank_zero) or indices <= set(rank_one)


def test_non_conversation_prewarm_retains_rank_striding() -> None:
    dataset = _ConversationDataset([{"video": f"{index}.mp4"} for index in range(7)])

    assert _prewarm_indices(dataset, "aetc", 1, 3) == ([1, 4], "rank_strided")


def test_cache_key_is_deterministic_and_runtime_root_is_preserved(tmp_path) -> None:
    fingerprint = combined_cache_fingerprint("dataset-sha", "model-sha", "processor-sha")
    assert fingerprint == combined_cache_fingerprint("dataset-sha", "model-sha", "processor-sha")
    root = cache_path(tmp_path / "user-cache", "train", fingerprint)
    assert root == (tmp_path / "user-cache" / f"train-{fingerprint}").resolve()
    assert entry_path(root, 10001) == root / "1" / "10001.pt"


def test_different_dataset_or_model_never_reuses_cache() -> None:
    original = combined_cache_fingerprint("dataset-a", "model-a", "processor")
    assert combined_cache_fingerprint("dataset-b", "model-a", "processor") != original
    assert combined_cache_fingerprint("dataset-a", "model-b", "processor") != original


def test_distributed_prewarm_closes_group_before_rank_zero_hashing(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(
        "cosmos_rl.tools.prewarm_sft_cache.torch.distributed.barrier",
        lambda: events.append("barrier"),
    )
    monkeypatch.setattr(
        "cosmos_rl.tools.prewarm_sft_cache.torch.distributed.destroy_process_group",
        lambda: events.append("destroy"),
    )

    assert finish_distributed_prewarm(rank=0, world_size=8, initialized_here=True)
    assert events == ["barrier", "destroy"]


def test_nonzero_rank_does_not_write_manifest(monkeypatch) -> None:
    monkeypatch.setattr(
        "cosmos_rl.tools.prewarm_sft_cache.torch.distributed.barrier", lambda: None
    )
    monkeypatch.setattr(
        "cosmos_rl.tools.prewarm_sft_cache.torch.distributed.destroy_process_group",
        lambda: None,
    )

    assert not finish_distributed_prewarm(rank=3, world_size=8, initialized_here=True)


def test_finalize_marker_is_job_and_split_specific(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TAO_JOB_ID", "job-one")
    train = finalize_marker(tmp_path, "train")
    validation = finalize_marker(tmp_path, "val")
    monkeypatch.setenv("TAO_JOB_ID", "job-two")
    other_job = finalize_marker(tmp_path, "train")

    assert train != validation
    assert train != other_job


def test_nonzero_rank_waits_for_success_marker(tmp_path) -> None:
    marker = tmp_path / "finalize.json"
    write_finalize_marker(marker, status="success")
    wait_for_finalize_marker(marker, timeout_seconds=0.01)
    assert json.loads(marker.read_text())["status"] == "success"


def test_nonzero_rank_propagates_rank_zero_failure(tmp_path) -> None:
    marker = tmp_path / "finalize.json"
    write_finalize_marker(marker, status="failure", error="hash failed")
    with pytest.raises(RuntimeError, match="hash failed"):
        wait_for_finalize_marker(marker, timeout_seconds=0.01)


def test_nonzero_rank_finalize_wait_times_out(tmp_path, monkeypatch) -> None:
    marker = tmp_path / "missing.json"
    monkeypatch.setattr("cosmos_rl.tools.prewarm_sft_cache.time.sleep", lambda _: None)
    with pytest.raises(TimeoutError, match="Timed out"):
        wait_for_finalize_marker(marker, timeout_seconds=0.001)
