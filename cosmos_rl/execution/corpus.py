# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Immutable local/shared-filesystem corpus with atomic manifest publication.

Only committed manifests are readable. Each manifest is a small one-record
shard; large tensors are content-addressed artifacts, never transport leases.
Concurrent writers use no-replace publication, so the first committed result
wins even when a process dies before acknowledging its work.
"""

import hashlib
import json
import os
from pathlib import Path
import tempfile

from cosmos_rl.execution.records import SampleRecord


def _json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(2 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _publish(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return True
    finally:
        os.unlink(temporary)


class ArtifactCodec:
    def __init__(self, root, max_bytes):
        self.root = Path(root).resolve()
        self.max_bytes = max_bytes
        self.used_bytes = 0

    def _tensor_path(self, value, *, verify=True):
        path = (self.root / value["path"]).resolve()
        if not path.is_relative_to(self.root / "artifacts"):
            raise ValueError("Artifact escapes its corpus")
        declared = value["bytes"]
        if type(declared) is not int or declared < 0:
            raise ValueError("Invalid artifact length")
        self.used_bytes += declared
        if self.used_bytes > self.max_bytes or path.stat().st_size != declared:
            raise ValueError("Artifact exceeds byte budget or has an invalid length")
        if verify and _file_digest(path) != value["sha256"]:
            raise ValueError("Artifact checksum mismatch")
        return path

    def encode(self, value):
        from cosmos_rl.execution.storage import DataReference

        if isinstance(value, DataReference):
            self.used_bytes += value.size_bytes
            if self.used_bytes > self.max_bytes:
                raise ValueError("Corpus record exceeds its artifact byte budget")
            return {"kind": "data_reference", "reference": value.to_dict()}
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise TypeError("Persistent mappings require string keys")
            return {
                "kind": "mapping",
                "items": {k: self.encode(v) for k, v in value.items()},
            }
        if isinstance(value, (list, tuple)):
            return {
                "kind": "tuple" if isinstance(value, tuple) else "list",
                "items": [self.encode(v) for v in value],
            }
        import torch

        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided:
                raise TypeError("Persistent tensors must use a strided layout")
            size = value.numel() * value.element_size()
            self.used_bytes += size
            if self.used_bytes > self.max_bytes:
                raise ValueError("Corpus record exceeds its artifact byte budget")
            tensor = value.detach().to("cpu").contiguous()
            # torch.save preserves backing storage. Compact contiguous slices
            # too, otherwise a tiny view can serialize an entire trajectory.
            if tensor.storage_offset() or tensor.untyped_storage().nbytes() > size:
                tensor = tensor.clone()
            directory = self.root / "artifacts"
            directory.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=directory)
            try:
                with os.fdopen(fd, "wb") as stream:
                    torch.save(tensor, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                serialized = Path(temporary).stat().st_size
                self.used_bytes += serialized - size
                if self.used_bytes > self.max_bytes:
                    raise ValueError(
                        "Corpus record exceeds its serialized artifact byte budget"
                    )
                sha = _file_digest(Path(temporary))
                relative = f"artifacts/{sha}.pt"
                try:
                    os.link(temporary, self.root / relative)
                except FileExistsError:
                    existing = self.root / relative
                    if (
                        existing.stat().st_size != serialized
                        or _file_digest(existing) != sha
                    ):
                        raise ValueError("Existing artifact checksum mismatch")
                directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                os.unlink(temporary)
            return {
                "kind": "tensor",
                "path": relative,
                "sha256": sha,
                "bytes": serialized,
            }
        # NumPy is optional; normalize its scalars and arrays without repr().
        import numpy as np

        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError("Object arrays are not durable tensor data")
            if self.used_bytes + value.nbytes > self.max_bytes:
                raise ValueError("Corpus record exceeds its artifact byte budget")
            return {
                "kind": "ndarray",
                "tensor": self.encode(torch.from_numpy(value.copy())),
            }
        if isinstance(value, np.generic):
            return self.encode(value.item())
        raise TypeError(
            f"Unsupported persistent value: {type(value).__name__}; supply a durable application representation"
        )

    def decode(self, value, *, materialize=True):
        if not isinstance(value, dict):
            return value
        kind = value["kind"]
        if kind == "data_reference":
            from cosmos_rl.execution.storage import DataReference

            reference = DataReference(**value["reference"])
            self.used_bytes += reference.size_bytes
            if self.used_bytes > self.max_bytes:
                raise ValueError("Referenced payload exceeds its byte budget")
            return reference
        if kind == "mapping":
            return {
                k: self.decode(v, materialize=materialize)
                for k, v in value["items"].items()
            }
        if kind in ("list", "tuple"):
            items = [self.decode(v, materialize=materialize) for v in value["items"]]
            return tuple(items) if kind == "tuple" else items
        if kind == "ndarray":
            result = self.decode(value["tensor"], materialize=materialize)
            return result.numpy() if materialize else result
        if kind != "tensor":
            raise ValueError(f"Unknown artifact encoding {kind}")
        path = self._tensor_path(value, verify=materialize)
        if not materialize:
            # Controller resume only needs identity and declared byte/availability
            # metadata. Tensor checksums are verified by the eventual reader.
            return None
        import torch

        return torch.load(path, map_location="cpu", weights_only=True)


class CorpusWriter:
    def __init__(self, root, *, run_id, max_record_bytes=1024**3, metadata=None):
        self.root = Path(root).resolve()
        self.run_id = run_id
        self.max_record_bytes = max_record_bytes
        metadata = _json(
            {"schema_version": 1, "run_id": run_id, "metadata": metadata or {}}
        )
        _publish(self.root / "dataset.json", metadata)
        if (self.root / "dataset.json").read_bytes() != metadata:
            raise ValueError("Output directory belongs to another dataset run")

    def commit(self, record: SampleRecord):
        codec = ArtifactCodec(self.root, self.max_record_bytes)
        payload = codec.encode(record.model_dump())
        data = _json({"schema_version": 1, "run_id": self.run_id, "record": payload})
        if len(data) + codec.used_bytes > self.max_record_bytes:
            raise ValueError("Corpus record exceeds its total byte budget")
        path = self.root / "records" / f"{_digest(record.sample_id.encode())}.json"
        created = _publish(path, data)
        if not created:
            existing = json.loads(path.read_bytes())["record"]["items"]
            incoming = payload["items"]
            if any(
                existing[key] != incoming[key]
                for key in (
                    "sample_id",
                    "occurrence_id",
                    "producer",
                    "inputs",
                    "sampling",
                    "group_id",
                    "group_size",
                    "group_slot",
                )
            ):
                raise ValueError("Output identity conflicts with committed provenance")
        return created

    def close(self):
        # Writes are synchronous and fsynced before returning to the caller.
        pass


def _read_record(root, path, max_bytes, *, materialize=True):
    if path.stat().st_size > max_bytes:
        raise ValueError("Manifest exceeds its byte budget")
    manifest = json.loads(path.read_bytes())
    metadata = json.loads((root / "dataset.json").read_bytes())
    if manifest["schema_version"] != 1 or manifest["run_id"] != metadata["run_id"]:
        raise ValueError("Manifest schema or dataset identity mismatch")
    codec = ArtifactCodec(root, max_bytes - path.stat().st_size)
    return SampleRecord.model_validate(
        codec.decode(manifest["record"], materialize=materialize)
    )


class CorpusDataset:
    """An immutable manifest snapshot usable by existing local dataloaders."""

    def __init__(self, root, *, max_record_bytes=1024**3, revision=None):
        self.root = Path(root).resolve()
        metadata = json.loads((self.root / "dataset.json").read_bytes())
        if metadata.get("schema_version") != 1:
            raise ValueError("Unsupported corpus schema")
        self.paths = sorted((self.root / "records").glob("*.json"))
        self.max_record_bytes = max_record_bytes
        self.manifest_hashes = {}
        digest = hashlib.sha256(_json(metadata))
        for path in self.paths:
            if path.stat().st_size > max_record_bytes:
                raise ValueError("Manifest exceeds its byte budget")
            digest.update(path.name.encode())
            data = path.read_bytes()
            self.manifest_hashes[path] = _digest(data)
            digest.update(data)
        self.revision = digest.hexdigest()
        if revision is not None and revision != self.revision:
            raise ValueError("Corpus revision does not match the configured snapshot")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        if _digest(path.read_bytes()) != self.manifest_hashes[path]:
            raise ValueError("Committed manifest changed after the source snapshot")
        return _read_record(self.root, path, self.max_record_bytes)

    def record_bytes(self, index):
        path = self.paths[index]

        def size(value):
            if isinstance(value, dict):
                if value.get("kind") == "tensor":
                    return value["bytes"]
                if value.get("kind") == "data_reference":
                    return value["reference"]["size_bytes"]
                return sum(size(v) for v in value.values())
            if isinstance(value, list):
                return sum(size(v) for v in value)
            return 0

        return path.stat().st_size + size(json.loads(path.read_bytes()))
