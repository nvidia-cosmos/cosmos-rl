# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prewarm and validate a deterministic WTS or AETC SFT preprocessing cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import toml
import torch


def cache_path(cache_root: str | Path, split: str, fingerprint: str) -> Path:
    return Path(cache_root).expanduser().resolve() / f"{split}-{fingerprint}"


def entry_path(root: Path, index: int) -> Path:
    return root / str(index // 10000) / f"{index}.pt"


def combined_cache_fingerprint(dataset: str, model: str, processor: str) -> str:
    return hashlib.sha256(f"dataset={dataset}\nmodel={model}\nprocessor={processor}\n".encode()).hexdigest()


def finish_distributed_prewarm(
    *, rank: int, world_size: int, initialized_here: bool
) -> bool:
    """Close distributed coordination before rank 0 hashes cache entries.

    Hashing a full video cache can exceed NCCL's collective watchdog timeout.
    All ranks therefore synchronize once after writing their entries and then
    tear down the process group. Rank 0 alone performs the filesystem-only
    completeness and provenance pass with no outstanding collective.
    """
    if world_size > 1:
        torch.distributed.barrier()
    if initialized_here:
        torch.distributed.destroy_process_group()
    return rank == 0


def finalize_marker(root: Path, split: str) -> Path:
    """Return a run-specific filesystem marker used after NCCL teardown."""
    job_id = os.environ.get("TAO_JOB_ID", "standalone")
    token = hashlib.sha256(f"job={job_id}\nsplit={split}\n".encode()).hexdigest()[:16]
    return root / f".prewarm-finalize-{token}.json"


def wait_for_finalize_marker(marker: Path, timeout_seconds: float) -> None:
    """Keep nonzero workers alive until rank 0 finishes filesystem hashing.

    Multi-node torchrun may host its c10d rendezvous store on a node that does
    not own global rank 0.  If that node's worker exits while rank 0 is still
    hashing the cache, its local elastic agent can tear down the store and kill
    rank 0 before the provenance manifest is published.  Polling a filesystem
    marker after NCCL teardown keeps every agent alive without retaining an
    NCCL collective that could hit the watchdog timeout.
    """
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if marker.is_file():
            payload = json.loads(marker.read_text(encoding="utf-8"))
            if payload.get("status") == "success":
                return
            raise RuntimeError(
                f"Rank 0 cache finalization failed: {payload.get('error', 'unknown error')}"
            )
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for rank 0 cache finalization marker: {marker}")


def write_finalize_marker(marker: Path, *, status: str, error: str | None = None) -> None:
    payload = {"status": status}
    if error is not None:
        payload["error"] = error
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, marker)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configure_conversation_video_decoder(custom) -> dict[str, object]:
    """Use the same registered decoder contract as normal SFT training."""
    from cosmos_rl.tools.custom_hooks.tao_sft_example import configure_video_decoder

    return configure_video_decoder(custom)


def _prewarm_indices(dataset, kind: str, rank: int, world_size: int) -> tuple[list[int], str]:
    """Assign repeated-media records together without changing cache indices.

    Conversation manifests commonly reuse each video for many questions.  A
    rank-strided prewarm defeats the sparse reader's small in-process media
    cache and decodes the same video once per record.  Grouping only the
    *processing order* keeps every cache entry at its original dataset index,
    while ensuring one rank processes a video's records consecutively.
    """
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(f"invalid distributed coordinates: rank={rank}, world_size={world_size}")
    annotation = getattr(dataset, "annotation", None)
    if kind != "wts" or not isinstance(annotation, list) or len(annotation) != len(dataset):
        return list(range(rank, len(dataset), world_size)), "rank_strided"

    groups: dict[str, list[int]] = {}
    for index, record in enumerate(annotation):
        media = None
        if isinstance(record, dict):
            media = {
                key: record.get(key)
                for key in ("video", "videos", "image", "images")
                if record.get(key) is not None
            }
        if not media:
            key = f"record:{index}"
        else:
            key = "media:" + json.dumps(
                media, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        groups.setdefault(key, []).append(index)

    assignments: list[list[int]] = [[] for _ in range(world_size)]
    loads = [0] * world_size
    for group in groups.values():
        target_rank = min(range(world_size), key=lambda candidate: (loads[candidate], candidate))
        assignments[target_rank].extend(group)
        loads[target_rank] += len(group)
    return assignments[rank], "media_grouped_balanced"


def _build_dataset_and_packer(kind, split, config, raw):
    custom_raw = raw.get("custom", {})
    if kind == "wts":
        from cosmos_rl.tools.custom_hooks.tao_sft_example import CustomConfig, CustomDataset
        from cosmos_rl.dispatcher.data.packer.hf_vlm_data_packer import HFVLMDataPacker

        custom = CustomConfig.model_validate(custom_raw)
        dataset_config = custom.train_dataset if split == "train" else custom.val_dataset
        if dataset_config is None:
            raise ValueError(f"WTS {split} dataset was not supplied")
        _configure_conversation_video_decoder(custom)
        dataset = CustomDataset(
            config=config,
            custom_config=custom,
            annotation_path=dataset_config.annotation_path,
            media_path=dataset_config.media_path,
        )
        packer = HFVLMDataPacker()
    elif kind == "aetc":
        from cosmos_rl.tools.custom_hooks.tao_vl_reason_daft_sft_example import (
            CustomConfig,
            CustomDataset,
            TaoVlReasonHFVLMDataPacker,
        )

        custom = CustomConfig.model_validate(custom_raw)
        dataset_config = custom.train_dataset if split == "train" else custom.val_dataset
        if dataset_config is None:
            raise ValueError(f"AETC {split} dataset was not supplied")
        if custom.video_decoder == "pynvvideocodec":
            from cosmos_rl.utils.pynv_video_reader import register_pynv_video_reader

            register_pynv_video_reader(
                cache_size=custom.video_cache_size,
                video_override_map=custom.video_override_map,
            )
        dataset = CustomDataset(config=config, custom_config=custom, dataset_config=dataset_config)
        packer = TaoVlReasonHFVLMDataPacker()
    else:
        raise ValueError(f"Unsupported dataset kind: {kind}")
    packer.setup(config)
    if kind == "wts":
        dataset.setup(config, getattr(packer, "tokenizer", None))
    else:
        dataset.setup(config)
    return dataset, packer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-kind", required=True, choices=("wts", "aetc"))
    parser.add_argument("--split", required=True, choices=("train", "val"))
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--dataset-fingerprint", required=True)
    parser.add_argument("--model-fingerprint", required=True)
    parser.add_argument("--processor-fingerprint", required=True)
    parser.add_argument("--cache-fingerprint", required=True)
    args = parser.parse_args()

    expected_fingerprint = combined_cache_fingerprint(
        args.dataset_fingerprint, args.model_fingerprint, args.processor_fingerprint
    )
    if args.cache_fingerprint != expected_fingerprint:
        raise ValueError(
            f"cache fingerprint mismatch: supplied={args.cache_fingerprint}, expected={expected_fingerprint}"
        )

    from cosmos_rl.policy.config import Config

    config_path = Path(args.config).expanduser().resolve(strict=True)
    raw = toml.load(config_path)
    config = Config.from_dict(raw)
    dataset, packer = _build_dataset_and_packer(args.dataset_kind, args.split, config, raw)
    root = cache_path(args.cache_dir, args.split, args.cache_fingerprint)
    root.mkdir(parents=True, exist_ok=True)

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    initialized_here = False
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
        initialized_here = True

    marker = finalize_marker(root, args.split)
    if world_size > 1:
        if rank == 0:
            marker.unlink(missing_ok=True)
        torch.distributed.barrier()

    indices, prewarm_strategy = _prewarm_indices(dataset, args.dataset_kind, rank, world_size)
    for index in indices:
        target = entry_path(root, index)
        if target.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        value = packer.sft_process_sample(dataset[index])
        temporary = target.with_suffix(f".rank{rank}.tmp")
        torch.save(value, temporary)
        os.replace(temporary, target)

    write_manifest = finish_distributed_prewarm(
        rank=rank, world_size=world_size, initialized_here=initialized_here
    )
    if not write_manifest:
        wait_for_finalize_marker(
            marker,
            float(os.environ.get("TAO_CACHE_FINALIZE_TIMEOUT_SECONDS", "7200")),
        )
        return 0

    try:
        missing = [index for index in range(len(dataset)) if not entry_path(root, index).is_file()]
        if missing:
            raise RuntimeError(f"Cache prewarm incomplete: {len(missing)} missing entries; first={missing[:10]}")
        entry_hashes = {str(index): _sha256(entry_path(root, index)) for index in range(len(dataset))}
        manifest = {
            "schema_version": 1,
            "dataset_kind": args.dataset_kind,
            "split": args.split,
            "record_count": len(dataset),
            "dataset_fingerprint": args.dataset_fingerprint,
            "model_fingerprint": args.model_fingerprint,
            "processor_fingerprint": args.processor_fingerprint,
            "cache_fingerprint": args.cache_fingerprint,
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "prewarm_schedule": {
                "strategy": prewarm_strategy,
                "world_size": world_size,
            },
            "entry_hashes": entry_hashes,
            "complete": True,
        }
        temporary = root / "cache_provenance.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, root / "cache_provenance.json")
        write_finalize_marker(marker, status="success")
    except Exception as exc:
        write_finalize_marker(marker, status="failure", error=f"{type(exc).__name__}: {exc}")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
