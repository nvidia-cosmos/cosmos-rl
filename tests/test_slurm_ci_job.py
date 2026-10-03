# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submission/mount contracts for the existing full-suite Slurm runner."""

import os
import json
from pathlib import Path
from importlib.util import find_spec
import subprocess

import pytest


SCRIPT = Path(find_spec("cosmos_rl").origin).parent / "tools/slurm/cosmos_rl_ci_job.sh"


def candidate(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    wheel = tmp_path / "cosmos_rl-0.4.6-py3-none-any.whl"
    wheel.touch()
    deps = tmp_path / "deps"
    deps.mkdir()
    return repo, wheel, deps


@pytest.mark.parametrize(
    "cached_paths,readonly", [(False, False), (True, False), (True, True)]
)
def test_wheel_submission_preserves_identity_and_offline_dependencies(
    tmp_path, cached_paths, readonly
):
    repo, wheel, deps = candidate(tmp_path)
    cache = tmp_path / "hf-cache"
    cache.mkdir()
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--container",
            str(tmp_path / "image.sqsh"),
            "--repo-root-path",
            str(repo),
            "--package-wheel",
            str(wheel),
            "--test-deps-dir",
            str(deps),
            "--hf-cache",
            str(cache),
            *(["--cached-model-paths"] if cached_paths else []),
            *(["--readonly-model-cache", str(deps)] if readonly else []),
            "--output-root-path",
            str(tmp_path / "out"),
            "--slurm-partition",
            "test",
            "--slurm-account",
            "test",
            "--dry-run",
        ],
        env={k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert f"COSMOS_CI_PACKAGE_WHEEL={wheel}" in result.stdout
    assert f"COSMOS_CI_TEST_DEPS_DIR={deps}" in result.stdout
    assert f"COSMOS_CI_HF_CACHE={cache}" in result.stdout
    assert f"COSMOS_CI_CACHED_MODEL_PATHS={int(cached_paths)}" in result.stdout
    assert f"COSMOS_CI_READONLY_MODEL_CACHE={deps if readonly else ''}" in result.stdout
    assert "--gres=gpu:8" in result.stdout
    assert "--time=2:30:00" in result.stdout
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "installed,readonly", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("exit_code", [0, 42])
def test_job_mounts_and_full_suite_entrypoint(tmp_path, installed, readonly, exit_code):
    repo, wheel, deps = candidate(tmp_path)
    binary = tmp_path / "bin"
    binary.mkdir()
    srun = binary / "srun"
    srun.write_text(f"#!/bin/bash\nprintf '%s\\n' \"$@\"\nexit {exit_code}\n")
    srun.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{binary}:{os.environ['PATH']}",
        "SLURM_JOB_ID": "123",
        "COSMOS_CI_OUTPUT_DIR": str(tmp_path / "out"),
        "COSMOS_CI_SLURM_DIR": str(tmp_path / "out/slurm"),
        "COSMOS_CI_CONTAINER": str(tmp_path / "image.sqsh"),
        "COSMOS_CI_REPO_ROOT": str(repo),
        "COSMOS_CI_SUBMIT_USER": "test",
        "COSMOS_CI_SCRATCH": str(tmp_path / "scratch"),
        "COSMOS_CI_PACKAGE_WHEEL": str(wheel) if installed else "",
        "COSMOS_CI_TEST_DEPS_DIR": str(deps),
        "COSMOS_CI_CACHED_MODEL_PATHS": "1" if readonly else "0",
        "COSMOS_CI_READONLY_MODEL_CACHE": str(deps) if readonly else "",
    }
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == exit_code
    output = result.stdout
    if installed:
        assert f"{repo}/tests:/workspace/cosmos-rl/tests:ro" in output
        assert f"{repo}:/opt/cosmos-rl" not in output
        assert f"{wheel}:/ci-wheel/{wheel.name}:ro" in output
    else:
        assert f"{repo}:/opt/cosmos-rl" in output
    assert f"{deps}:/ci-test-deps:ro" in output
    if readonly:
        assert f"{deps}:/ci-model-cache:ro" in output
        assert "model_cache=/ci-model-cache" in output
        assert 'ci_link_cached_models "$model_cache" "$PWD"' in output
    else:
        assert f"{deps}:/ci-model-cache:ro" not in output
    assert "bash tests/run_test.sh" in output
    assert "unset PYTHONPATH" in output
    assert "--no-deps --force-reinstall" in output
    assert "pytest>=8,<9" in output
    assert "ucxx-cu12>=0.40.0" in output
    assert "import torch, ucxx, pytest" in output
    assert "TEST_LOG_DIR=/ci-results/test-logs" in output
    assert ("CI PASSED" in output) == (exit_code == 0)


@pytest.mark.parametrize("cached_paths,exists", [(False, True), (True, False)])
def test_readonly_models_require_alias_mode_and_existing_directory(
    tmp_path, cached_paths, exists
):
    repo, wheel, deps = candidate(tmp_path)
    model_cache = deps if exists else tmp_path / "missing"
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--container",
            "image.sqsh",
            "--package-wheel",
            str(wheel),
            "--repo-root-path",
            str(repo),
            "--hf-cache",
            str(tmp_path / "runtime"),
            "--readonly-model-cache",
            str(model_cache),
            *(["--cached-model-paths"] if cached_paths else []),
            "--output-root-path",
            str(tmp_path / "out"),
            "--slurm-partition",
            "test",
            "--slurm-account",
            "test",
            "--dry-run",
        ],
        env={k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "--readonly-model-cache requires" in result.stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("relation", ["same", "parent", "child"])
def test_readonly_models_cannot_also_be_exposed_through_writable_cache(
    tmp_path, relation
):
    repo, wheel, deps = candidate(tmp_path)
    cache = (
        deps
        if relation == "same"
        else deps / "runtime"
        if relation == "child"
        else tmp_path
    )
    cache.mkdir(exist_ok=True)
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--container",
            "image.sqsh",
            "--package-wheel",
            str(wheel),
            "--repo-root-path",
            str(repo),
            "--hf-cache",
            str(cache),
            "--readonly-model-cache",
            str(deps),
            "--cached-model-paths",
            "--output-root-path",
            str(tmp_path / "out"),
            "--slurm-partition",
            "test",
            "--slurm-account",
            "test",
            "--dry-run",
        ],
        env={k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "must not overlap" in result.stderr
    assert not (tmp_path / "out").exists()


def test_wheel_requires_candidate_tests(tmp_path):
    _, wheel, _ = candidate(tmp_path)
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--container",
            "image.sqsh",
            "--package-wheel",
            str(wheel),
            "--output-root-path",
            str(tmp_path / "out"),
            "--slurm-partition",
            "test",
            "--slurm-account",
            "test",
            "--dry-run",
        ],
        env={k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "requires an existing wheel" in result.stderr


def cache_alias_command(cache, workspace):
    source = SCRIPT.read_text()
    helper = (
        "ci_link_cached_models() {"
        + source.split("ci_link_cached_models() {", 1)[1].split("\nusage()", 1)[0]
    )
    return [
        "bash",
        "-c",
        helper + '\nci_link_cached_models "$1" "$2"',
        "bash",
        str(cache),
        str(workspace),
    ]


def cached_model(tmp_path):
    cache, workspace = tmp_path / "hub", tmp_path / "workspace"
    model = cache / "models--Qwen--test-model"
    revision = "a" * 40
    snapshot = model / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"fixture weights")
    (model / "refs").mkdir()
    (model / "refs/main").write_text(revision)
    workspace.mkdir()
    return cache, workspace, model, snapshot


def test_cached_model_alias_is_revision_pinned_and_idempotent(tmp_path):
    cache, workspace, _, snapshot = cached_model(tmp_path)
    for _ in range(2):
        subprocess.run(cache_alias_command(cache, workspace), check=True)
        assert (workspace / "Qwen/test-model").is_symlink()
        assert (workspace / "Qwen/test-model").resolve() == snapshot


@pytest.mark.parametrize("invalid", ["config-only", "empty", "broken-symlink"])
def test_incomplete_single_file_checkpoint_fails_before_aliasing(tmp_path, invalid):
    cache, workspace, _, snapshot = cached_model(tmp_path)
    checkpoint = snapshot / "model.safetensors"
    checkpoint.unlink()
    if invalid == "empty":
        checkpoint.touch()
    elif invalid == "broken-symlink":
        checkpoint.symlink_to(snapshot / "missing-blob")
    result = subprocess.run(cache_alias_command(cache, workspace), capture_output=True)
    assert result.returncode != 0
    assert b"incomplete cached model" in result.stderr
    assert not (workspace / "Qwen/test-model").exists()


@pytest.mark.parametrize(
    "variant", ["complete", "missing", "empty", "bad-json", "empty-map", "outside"]
)
def test_sharded_cache_requires_every_indexed_weight(tmp_path, variant):
    cache, workspace, _, snapshot = cached_model(tmp_path)
    (snapshot / "model.safetensors").unlink()
    shard = snapshot / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"fixture shard")
    mapping = {"weight": shard.name}
    if variant == "missing":
        shard.unlink()
    elif variant == "empty":
        shard.write_bytes(b"")
    elif variant == "empty-map":
        mapping = {}
    elif variant == "outside":
        mapping = {"weight": "../outside.safetensors"}
    (snapshot / "model.safetensors.index.json").write_text(
        "not json" if variant == "bad-json" else json.dumps({"weight_map": mapping})
    )
    result = subprocess.run(cache_alias_command(cache, workspace), capture_output=True)
    assert (result.returncode == 0) == (variant == "complete")
    assert (workspace / "Qwen/test-model").exists() == (variant == "complete")


@pytest.mark.parametrize(
    "invalid", ["revision", "missing-config", "collision", "parent-symlink"]
)
def test_invalid_cache_or_existing_workspace_data_is_not_overwritten(tmp_path, invalid):
    cache, workspace, model, snapshot = cached_model(tmp_path)
    if invalid == "revision":
        (model / "refs/main").write_text("../../outside")
    elif invalid == "missing-config":
        (snapshot / "config.json").unlink()
    elif invalid == "collision":
        (workspace / "Qwen").mkdir()
        (workspace / "Qwen/test-model").write_text("preserve me")
    else:
        outside = tmp_path / "outside"
        outside.mkdir()
        (workspace / "Qwen").symlink_to(outside, target_is_directory=True)
    result = subprocess.run(cache_alias_command(cache, workspace))
    assert result.returncode != 0
    if invalid == "collision":
        assert (workspace / "Qwen/test-model").read_text() == "preserve me"
    else:
        assert not (workspace / "Qwen/test-model").exists()


def test_cached_model_paths_require_disposable_wheel_workspace(tmp_path):
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--container",
            "image.sqsh",
            "--output-root-path",
            str(tmp_path / "out"),
            "--slurm-partition",
            "test",
            "--slurm-account",
            "test",
            "--cached-model-paths",
            "--dry-run",
        ],
        env={k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "requires --package-wheel and --hf-cache" in result.stderr
