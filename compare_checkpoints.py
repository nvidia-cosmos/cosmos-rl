#!/usr/bin/env python3
"""
Compare two Orbax/JAX-style checkpoint directories, ignoring timestamp fields.

Usage:
  python compare_checkpoints.py <dir_a> <dir_b> [--hash]

By default it compares file tree (relative paths) and file sizes.
If --hash is provided, it also computes sha256 for each file (can be slow).
For _CHECKPOINT_METADATA it parses JSON and ignores init/commit timestamps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass


IGNORE_JSON_KEYS = {"init_timestamp_nsecs", "commit_timestamp_nsecs"}


@dataclass(frozen=True)
class FileInfo:
    rel: str
    size: int
    sha256: str | None = None


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_files(root: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            abs_path = os.path.join(dirpath, fn)
            rel = os.path.relpath(abs_path, root)
            out[rel] = abs_path
    return out


def _load_checkpoint_metadata_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for k in list(data.keys()):
        if k in IGNORE_JSON_KEYS:
            data.pop(k, None)
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    ap.add_argument("--hash", action="store_true", help="also compute sha256 for each file")
    args = ap.parse_args()

    a = os.path.abspath(args.dir_a)
    b = os.path.abspath(args.dir_b)
    if not os.path.isdir(a):
        raise SystemExit(f"not a directory: {a}")
    if not os.path.isdir(b):
        raise SystemExit(f"not a directory: {b}")

    files_a = _iter_files(a)
    files_b = _iter_files(b)
    rel_a = set(files_a.keys())
    rel_b = set(files_b.keys())

    only_a = sorted(rel_a - rel_b)
    only_b = sorted(rel_b - rel_a)

    same = True
    if only_a:
        same = False
        print("Only in A:")
        for p in only_a[:200]:
            print(f"  - {p}")
        if len(only_a) > 200:
            print(f"  ... ({len(only_a) - 200} more)")
    if only_b:
        same = False
        print("Only in B:")
        for p in only_b[:200]:
            print(f"  - {p}")
        if len(only_b) > 200:
            print(f"  ... ({len(only_b) - 200} more)")

    common = sorted(rel_a & rel_b)
    # Special-case _CHECKPOINT_METADATA
    if "_CHECKPOINT_METADATA" in rel_a and "_CHECKPOINT_METADATA" in rel_b:
        meta_a = _load_checkpoint_metadata_json(files_a["_CHECKPOINT_METADATA"])
        meta_b = _load_checkpoint_metadata_json(files_b["_CHECKPOINT_METADATA"])
        if meta_a != meta_b:
            same = False
            print("[DIFF] _CHECKPOINT_METADATA differs (timestamps ignored)")

    # Size / hash checks
    for rel in common:
        if rel == "_CHECKPOINT_METADATA":
            continue
        pa = files_a[rel]
        pb = files_b[rel]
        sa = os.path.getsize(pa)
        sb = os.path.getsize(pb)
        if sa != sb:
            same = False
            print(f"[DIFF] size {rel}: {sa} vs {sb}")
            continue
        if args.hash:
            ha = _sha256(pa)
            hb = _sha256(pb)
            if ha != hb:
                same = False
                print(f"[DIFF] sha256 {rel}: {ha} vs {hb}")

    if same:
        print("RESULT: SAME (timestamps ignored)")
        return 0
    else:
        print("RESULT: DIFFERENT (timestamps ignored)")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())



