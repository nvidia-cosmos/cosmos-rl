# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Run: cosmos-rl --config configs/rollout-only.toml examples/rollout_only_jsonl.py.

COSMOS_ROLLOUT_RESULTS_DIR selects the output directory (default ./rollout-results).
Each reporting rank owns its own JSONL file. Nested arrays/tensors are copied to
CPU and serialized with values, shape and dtype; opaque objects fail explicitly.
For large media workloads, replace this example with an artifact/reference sink.
"""

import base64
from dataclasses import fields, is_dataclass
import json
import os
from pathlib import Path

import numpy as np
from pydantic import BaseModel
import torch

from cosmos_rl.launcher.worker_entry import main
from cosmos_rl.utils.application_callbacks import CallbackFactory


def json_value(value):
    if isinstance(value, torch.Tensor):
        return {
            "type": "tensor",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.detach().cpu().tolist(),
        }
    if isinstance(value, np.ndarray):
        return {
            "type": "array",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, BaseModel):
        return json_value(value.model_dump())
    if is_dataclass(value):
        return {
            field.name: json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("JSONL result mappings require string keys")
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "encoding": "base64",
            "value": base64.b64encode(value).decode("ascii"),
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"Unsupported result data type: {type(value).__name__}; persist it as an artifact reference"
    )


class JSONLResults:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.output = open(path, "a", encoding="utf-8")

    def __call__(self, event):
        # Finish the copy/write while all borrowed data and refs are valid.
        line = json.dumps(json_value(event), ensure_ascii=False, allow_nan=False)
        self.output.write(line + "\n")
        self.output.flush()

    def close(self):
        self.output.close()


def make_sink(context):
    directory = Path(os.environ.get("COSMOS_ROLLOUT_RESULTS_DIR", "./rollout-results"))
    return JSONLResults(
        directory / f"{context.replica_name}-rank-{context.global_rank}.jsonl"
    )


if __name__ == "__main__":
    main(hook_fns={"rollout_result": CallbackFactory(make_sink)})
