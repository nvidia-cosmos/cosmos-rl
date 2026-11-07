# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Small application-owned durable payload saver and reader interfaces.

Savers run where the payload is owned, before report acknowledgment. References
are metadata, not live transport leases. A saver returns only after publication
is durable; internal asynchronous work must finish before returning. Applications
own formats, credentials, compression, sharding and storage-specific cleanup.
"""

import copy
import inspect
import json
from pathlib import Path
from typing import Any, Protocol


class DataReference:
    """Portable reference with a declared decoded payload size for admission.

    The format identifies the application reader. Metadata must remain small
    JSON; never include tensors, credentials or expiring transport handles.
    ``size_bytes`` must conservatively cover both stored and decoded bytes.
    """

    __slots__ = ("_data",)

    def __init__(self, *, uri, format, size_bytes, metadata=None):
        if (
            not isinstance(uri, str)
            or not uri
            or not isinstance(format, str)
            or not format
        ):
            raise ValueError("A data reference requires a nonempty URI and format")
        if type(size_bytes) is not int or size_bytes < 0:
            raise ValueError("Data reference size_bytes must be nonnegative")
        data = {
            "uri": uri,
            "format": format,
            "size_bytes": size_bytes,
            "metadata": {} if metadata is None else metadata,
        }
        encoded = json.dumps(data, allow_nan=False, separators=(",", ":"))
        if len(encoded.encode()) > 64 * 1024:
            raise ValueError("Data reference metadata exceeds 64 KiB")
        self._data = json.loads(encoded)

    @property
    def uri(self):
        return self._data["uri"]

    @property
    def format(self):
        return self._data["format"]

    @property
    def size_bytes(self):
        return self._data["size_bytes"]

    @property
    def metadata(self):
        return copy.deepcopy(self._data["metadata"])

    def to_dict(self):
        return copy.deepcopy(self._data)


class DataSaver(Protocol):
    def save(self, value: Any, *, max_bytes: int) -> DataReference:
        """Persist owned data; return after durability and all payload reads."""
        ...

    def close(self) -> None:
        """Finish owned operations and propagate failures before shutdown."""
        ...


class DataReader(Protocol):
    def load(self, reference: DataReference, *, max_bytes: int) -> Any:
        """Validate integrity and return owned application data within budget."""
        ...


class DataSaveError(RuntimeError):
    """A required durability operation failed; never discard as a bad episode."""


def save_data(saver: DataSaver, value: Any, *, max_bytes: int) -> DataReference:
    """Validate the synchronous saver boundary without prescribing a format."""
    try:
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("A saver requires a positive byte budget")
        result = saver.save(value, max_bytes=max_bytes)
        if inspect.isawaitable(result):
            if inspect.iscoroutine(result):
                result.close()
            raise TypeError("DataSaver.save must complete synchronously")
        if not isinstance(result, DataReference):
            raise TypeError("DataSaver.save must return a DataReference")
        if result.size_bytes > max_bytes:
            raise ValueError("Saved payload exceeds its byte budget")
        return result
    except Exception as error:
        raise DataSaveError(f"Required durable save failed: {error}") from error


class FileSystemDataStore:
    """Baseline tensor/array saver and reader on a local or shared filesystem.

    Reuses the corpus artifact codec: payload files are fsynced and published
    without replacement before returning. The logical record manifest is committed
    separately by the required result consumer. No payload goes through the
    controller. A custom saver/reader can be passed directly by application code;
    no registry or replacement launcher is needed.
    """

    format = "cosmos.artifacts.v1"

    def __init__(self, root):
        self.root = Path(root).resolve()

    def save(self, value, *, max_bytes):
        from cosmos_rl.execution.corpus import ArtifactCodec
        import numpy as np
        import torch

        def raw_bytes(item):
            if isinstance(item, torch.Tensor):
                return item.numel() * item.element_size()
            if isinstance(item, np.ndarray):
                return item.nbytes
            if isinstance(item, dict):
                return sum(raw_bytes(child) for child in item.values())
            if isinstance(item, (list, tuple)):
                return sum(raw_bytes(child) for child in item)
            if isinstance(item, str) and len(item) > 64 * 1024:
                raise ValueError("Baseline saver metadata exceeds 64 KiB")
            return 0

        if type(max_bytes) is not int or max_bytes <= 0 or raw_bytes(value) > max_bytes:
            raise ValueError("Payload exceeds its byte budget before serialization")

        codec = ArtifactCodec(self.root, max_bytes)
        tree = codec.encode(value)
        return DataReference(
            uri=self.root.as_uri(),
            format=self.format,
            size_bytes=codec.used_bytes,
            metadata={"tree": tree},
        )

    def load(self, reference, *, max_bytes):
        from cosmos_rl.execution.corpus import ArtifactCodec

        if reference.format != self.format or reference.uri != self.root.as_uri():
            raise ValueError("Data reference belongs to a different store or format")
        if reference.size_bytes > max_bytes:
            raise ValueError("Referenced payload exceeds its byte budget")
        codec = ArtifactCodec(self.root, max_bytes)
        value = codec.decode(reference.metadata["tree"])
        if codec.used_bytes != reference.size_bytes:
            raise ValueError("Data reference size does not match its artifacts")
        return value

    def close(self):
        # All writes and fsyncs have completed synchronously.
        pass
