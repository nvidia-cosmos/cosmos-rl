# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU controllers must import with the optional CUDA extensions installed."""

import importlib.util
import os
import subprocess
import sys
import unittest


class TestCPUModelImport(unittest.TestCase):
    def test_cpu_discovery_never_imports_cuda_only_transformer_engine(self):
        # Model discovery must not execute optional native CUDA initialization.
        # Exercise this contract even when Transformer Engine is not installed.
        script = """
import builtins
import sys
import torch
assert not torch.cuda.is_available()
original_import = builtins.__import__
def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    caller = (globals or {}).get("__name__", "")
    if caller == "cosmos_rl.policy.model.deepseek_v3.deepseekv3_mapped" and name.startswith("transformer_engine"):
        raise RuntimeError("CPU discovery attempted CUDA-only extension initialization")
    return original_import(name, globals, locals, fromlist, level)
builtins.__import__ = guarded_import
from cosmos_rl.utils.report.wandb_logger import init_wandb
from cosmos_rl.policy.model import DeepseekV3MoEModel
assert callable(init_wandb)
assert DeepseekV3MoEModel.supported_model_types() == ["deepseek_v3"]
assert not torch.cuda.is_initialized()
"""
        self._assert_cpu_discovery(script)

    def test_optional_transformer_engine_does_not_initialize_cuda(self):
        if importlib.util.find_spec("transformer_engine") is None:
            self.skipTest("Requires the installed optional Transformer Engine package")
        script = """
import sys
import torch
assert not torch.cuda.is_available()
from cosmos_rl.utils.report.wandb_logger import init_wandb
from cosmos_rl.policy.model import DeepseekV3MoEModel
from cosmos_rl.policy.model.deepseek_v3 import deepseekv3_mapped
assert callable(init_wandb)
assert DeepseekV3MoEModel.supported_model_types() == ["deepseek_v3"]
# Other dependencies may import the top-level package without using CUDA.
# DeepSeek must select its fallback, and discovery must not create a context.
assert deepseekv3_mapped._RMSNorm.__module__ == deepseekv3_mapped.__name__
assert not torch.cuda.is_initialized()
print("CPU model discovery with installed Transformer Engine PASS")
"""
        self._assert_cpu_discovery(script)

    def _assert_cpu_discovery(self, script):
        result = subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
