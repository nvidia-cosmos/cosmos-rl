# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise real torchrun exit folding, not just a mocked child exit."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("worker_exit", [0, 1, 86])
def test_torchrun_preserves_only_explicit_fatal_status(tmp_path, worker_exit):
    script = tmp_path / "worker.py"
    script.write_text(f"import os\nos._exit({worker_exit})\n")
    marker = tmp_path / "fatal-transport"
    env = dict(os.environ)
    env["COSMOS_FATAL_TRANSPORT_FILE"] = str(marker)
    repo = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(repo)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cosmos_rl.launcher.torchrun",
            "--standalone",
            "--nproc-per-node=1",
            str(script),
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == worker_exit, result.stderr
    assert marker.exists() == (worker_exit == 86)


def test_expiration_retains_live_cache_before_fatal_exit(monkeypatch):
    import queue
    import threading
    from cosmos_rl.utils.payload_transport import prefetch_mixin

    packer = prefetch_mixin.PrefetchDataPackerMixin()
    packer._transport_strategy = object()
    packer._prefetch_deadline_lock = threading.Lock()
    packer._prefetch_timers = {7: object()}
    packer._prefetch_shutdown = threading.Event()
    packer._prefetch_result_queue = queue.Queue()
    cache = {"live": object()}
    packer._prefetch_cache = cache

    def fatal(context):
        assert packer._prefetch_cache is cache
        assert packer._prefetch_shutdown.is_set()
        raise SystemExit(86)

    monkeypatch.setattr(prefetch_mixin, "fail_transport", fatal)
    with pytest.raises(SystemExit, match="86"):
        packer._expire_prefetch(7, 1)


def test_slurm_fatal_marker_terminates_running_steps(tmp_path):
    from test_slurm_multinode_exit import _extract_monitor_loop

    marker = tmp_path / "fatal-transport"
    marker.touch()
    script = (
        """
set -u
log() { :; }
sleep 30 & pid_policy=$!
sleep 30 & pid_rollout=$!
sleep 30 & pid_controller=$!
trap 'kill "$pid_policy" "$pid_rollout" "$pid_controller" 2>/dev/null || true' EXIT
"""
        + _extract_monitor_loop()
        + '\necho "status=$status"\n'
    )
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        timeout=10,
        env={**os.environ, "COSMOS_FATAL_TRANSPORT_FILE": str(marker)},
    )
    assert result.returncode == 0, result.stderr
    assert "status=86" in result.stdout
