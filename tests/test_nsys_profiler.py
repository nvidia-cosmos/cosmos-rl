import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import toml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_profiler_config_class():
    spec = importlib.util.spec_from_file_location(
        "cosmos_config_under_test",
        REPO_ROOT / "cosmos_rl" / "policy" / "config" / "__init__.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.ProfilerConfig


def _install_profiler_import_stubs():
    logger = mock.Mock()
    logging_mod = types.ModuleType("cosmos_rl.utils.logging")
    logging_mod.logger = logger

    policy_config_mod = types.ModuleType("cosmos_rl.policy.config")
    policy_config_mod.Config = object

    parallelism_mod = types.ModuleType("cosmos_rl.utils.parallelism")
    parallelism_mod.ParallelDims = object

    s3_utils_mod = types.ModuleType("cosmos_rl.utils.s3_utils")
    s3_utils_mod.upload_folder_to_s3 = mock.Mock()

    api_client_mod = types.ModuleType("cosmos_rl.dispatcher.api.client")
    api_client_mod.APIClient = object

    protocol_mod = types.ModuleType("cosmos_rl.dispatcher.protocol")

    class SetTracePathRequest:
        def __init__(self, replica_name, trace_path, global_rank):
            self.replica_name = replica_name
            self.trace_path = trace_path
            self.global_rank = global_rank

    protocol_mod.SetTracePathRequest = SetTracePathRequest

    stubs = {
        "cosmos_rl": types.ModuleType("cosmos_rl"),
        "cosmos_rl.utils": types.ModuleType("cosmos_rl.utils"),
        "cosmos_rl.utils.logging": logging_mod,
        "cosmos_rl.policy": types.ModuleType("cosmos_rl.policy"),
        "cosmos_rl.policy.config": policy_config_mod,
        "cosmos_rl.utils.parallelism": parallelism_mod,
        "cosmos_rl.utils.s3_utils": s3_utils_mod,
        "cosmos_rl.dispatcher": types.ModuleType("cosmos_rl.dispatcher"),
        "cosmos_rl.dispatcher.api": types.ModuleType("cosmos_rl.dispatcher.api"),
        "cosmos_rl.dispatcher.api.client": api_client_mod,
        "cosmos_rl.dispatcher.protocol": protocol_mod,
    }
    old_modules = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    return old_modules


def _restore_modules(old_modules):
    for name, module in old_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _load_profiler_class():
    old_modules = _install_profiler_import_stubs()
    try:
        spec = importlib.util.spec_from_file_location(
            "cosmos_profiler_under_test",
            REPO_ROOT / "cosmos_rl" / "utils" / "profiler.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module.CosmosProfiler
    finally:
        _restore_modules(old_modules)


def _load_nvtx_range():
    spec = importlib.util.spec_from_file_location(
        "cosmos_nvtx_under_test",
        REPO_ROOT / "cosmos_rl" / "utils" / "nvtx.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.nvtx_range


ProfilerConfig = _load_profiler_config_class()
CosmosProfiler = _load_profiler_class()
nvtx_range = _load_nvtx_range()


class TestNsightProfilerConfig(unittest.TestCase):
    def test_nsys_config_round_trip(self):
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".toml", delete=False) as f:
            toml.dump(
                {
                    "redis": "12808",
                    "train": {
                        "output_dir": "./outputs/nsys-test",
                        "train_policy": {"type": "sft", "dataset": {"name": ""}},
                    },
                    "profiler": {
                        "enable_nsys": True,
                        "nsys_target_roles": ["policy", "rollout"],
                        "nsys_output_dir": "./outputs/nsys",
                        "nsys_output_prefix": "smoke",
                        "nsys_trace": "cuda,nvtx",
                        "nsys_extra_args": ["--sample=none"],
                    },
                },
                f,
            )
            path = f.name
        try:
            cfg = ProfilerConfig.model_validate(toml.load(path)["profiler"])
            self.assertTrue(cfg.enable_nsys)
            self.assertEqual(cfg.nsys_target_roles, ["policy", "rollout"])
            self.assertEqual(cfg.nsys_output_prefix, "smoke")
            self.assertEqual(cfg.nsys_trace, "cuda,nvtx")
            self.assertEqual(cfg.nsys_extra_args, ["--sample=none"])
        finally:
            os.unlink(path)

    def test_default_nsys_trace_includes_cuda_and_nvtx(self):
        cfg = ProfilerConfig()
        traces = {item.strip() for item in cfg.nsys_trace.split(",")}
        self.assertIn("cuda", traces)
        self.assertIn("nvtx", traces)

    def test_invalid_nsys_role_fails_fast(self):
        with self.assertRaises(ValueError):
            ProfilerConfig.model_validate(
                {
                    "enable_nsys": True,
                    "nsys_target_roles": ["policy", "trainer"],
                }
            )


class TestNsightProfilerRuntime(unittest.TestCase):
    def _profiler(self):
        config = SimpleNamespace(
            profiler=SimpleNamespace(
                enable_profiler=False,
                enable_nsys=True,
                sub_profiler_config=SimpleNamespace(
                    wait_steps=1,
                    warmup_steps=1,
                    active_steps=2,
                    rank_filter=[0],
                    record_shape=False,
                    profile_memory=False,
                    with_stack=False,
                    with_modules=False,
                ),
            ),
            train=SimpleNamespace(
                output_dir="./outputs/run",
                ckpt=SimpleNamespace(upload_s3=False),
            ),
        )
        parallel_dims = SimpleNamespace(global_rank=0)
        return CosmosProfiler(config, parallel_dims, "replica", mock.Mock())

    @mock.patch("torch.cuda.is_available", return_value=True)
    @mock.patch("torch.cuda.nvtx")
    @mock.patch("torch.cuda.cudart")
    def test_nsys_capture_window_uses_cuda_profiler_api(
        self, mock_cudart, mock_nvtx, _mock_is_available
    ):
        cudart = mock.Mock()
        mock_cudart.return_value = cudart
        profiler = self._profiler()

        profiler.start_nsys()
        profiler.maybe_start_nsys()
        cudart.cudaProfilerStart.assert_not_called()

        profiler.step()
        profiler.maybe_start_nsys()
        cudart.cudaProfilerStart.assert_not_called()

        profiler.step()
        profiler.maybe_start_nsys()
        cudart.cudaProfilerStart.assert_called_once()
        mock_nvtx.range_push.assert_called_once()

        profiler.step()
        cudart.cudaProfilerStop.assert_not_called()

        profiler.step()
        cudart.cudaProfilerStop.assert_called_once()
        mock_nvtx.range_pop.assert_called_once()
        self.assertTrue(profiler.check_finished())

    @mock.patch("torch.cuda.is_available", return_value=True)
    @mock.patch("torch.cuda.nvtx")
    def test_nvtx_range_pushes_and_pops(self, mock_nvtx, _mock_is_available):
        with nvtx_range("cosmos.policy.test"):
            pass
        mock_nvtx.range_push.assert_called_once_with("cosmos.policy.test")
        mock_nvtx.range_pop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
