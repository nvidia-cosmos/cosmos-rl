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
    # A stale deployment variable must not re-enable filesystem notification.
    assert not marker.exists()


def test_expiration_retains_live_cache_before_fatal_exit(monkeypatch):
    import queue
    import threading
    from types import SimpleNamespace
    from cosmos_rl.utils.payload_transport import prefetch_mixin

    packer = prefetch_mixin.PrefetchDataPackerMixin()
    packer._transport_strategy = object()
    packer._prefetch_deadline_lock = threading.Lock()
    packer._prefetch_timers = {
        7: SimpleNamespace(
            _prefetch_deadline=0, _prefetch_timeout=1, cancel=lambda: None
        )
    }
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
