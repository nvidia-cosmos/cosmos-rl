# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional local-filesystem publication helper for application checkpoints.

Applications keep their own trainer serialization. The manifest is published
last, after all required files are durable. A directory without a manifest is
not a checkpoint. Remote/object stores can implement the same adapter contract
using their own commit mechanism instead.
"""

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from cosmos_rl.dispatcher.data.resume import ControllerResumeMetadata


class CheckpointManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    metadata: ControllerResumeMetadata
    compatibility: dict[str, Any]
    artifacts: dict[str, str] = Field(min_length=1)

    @staticmethod
    def _sha256(stream) -> str:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _artifact(root: Path, name: str) -> Path:
        # Flat files avoid ambiguous relative paths and symlink escapes.
        if not name or Path(name).name != name or name in {".", "..", "manifest.json"}:
            raise ValueError(f"Invalid checkpoint artifact name: {name!r}")
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or non-regular checkpoint artifact: {name}")
        return path

    @classmethod
    def publish(
        cls,
        root: Path,
        metadata: ControllerResumeMetadata,
        *,
        required_artifacts: set[str],
        compatibility: dict[str, Any],
    ) -> "CheckpointManifest":
        """Commit a new immutable directory; never replace a published save.

        The application must finish and close every shard at the metadata's
        completed-update boundary before calling. Files must not be mutated
        after publication. Only the designated checkpoint coordinator publishes.
        """
        root = Path(root)
        if str(root.resolve()) != str(Path(metadata.checkpoint_path).resolve()):
            raise ValueError("metadata checkpoint path must identify this directory")
        artifacts = {}
        for name in sorted(required_artifacts):
            with cls._artifact(root, name).open("rb") as stream:
                artifacts[name] = cls._sha256(stream)
                os.fsync(stream.fileno())
        manifest = cls(
            metadata=metadata, compatibility=compatibility, artifacts=artifacts
        )
        encoded = json.dumps(
            manifest.model_dump(mode="json"), allow_nan=False, sort_keys=True
        )
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=root, prefix=".manifest-", delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            # Atomic no-replace publication on a POSIX local/shared filesystem.
            os.link(temporary, root / "manifest.json")
            directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary is not None:
                temporary.unlink()
        return manifest

    @classmethod
    def load(
        cls,
        root: Path,
        *,
        required_artifacts: set[str],
        compatibility: dict[str, Any],
    ) -> "CheckpointManifest":
        """Verify before deserializing trainer state or consuming the sampler.

        Required artifacts/compatibility come from the current application, not
        from the manifest being checked (which could describe an incomplete or
        incompatible save). Both controller and trainer call this on load.
        """
        root = Path(root)
        manifest = cls.model_validate_json((root / "manifest.json").read_text())
        if manifest.compatibility != compatibility:
            raise ValueError(
                "Checkpoint dataset/configuration/topology is incompatible"
            )
        if set(manifest.artifacts) != required_artifacts:
            raise ValueError("Checkpoint does not contain the required artifact set")
        if str(root.resolve()) != str(
            Path(manifest.metadata.checkpoint_path).resolve()
        ):
            raise ValueError("Checkpoint manifest identifies a different directory")
        for name, digest in manifest.artifacts.items():
            with cls._artifact(root, name).open("rb") as stream:
                if cls._sha256(stream) != digest:
                    raise ValueError(f"Checkpoint artifact integrity failure: {name}")
        return manifest
