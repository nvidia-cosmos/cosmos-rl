# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Native producer/shared-owner canaries in isolated, finite subprocesses."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import cosmos_rl
from cosmos_rl.utils.payload_transport.ucxx import UCXX_AVAILABLE


@pytest.mark.skipif(not UCXX_AVAILABLE, reason="UCXX extra is required")
@pytest.mark.parametrize(
    "case", ["shared", "producer-healthy", "producer-timeout", "partial-start"]
)
def test_native_producer_and_shared_context(case):
    package = Path(cosmos_rl.__file__).resolve().parent
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("ucxx_context_canary.py")),
            "--case",
            case,
        ],
        env={
            **os.environ,
            "PYTHONPATH": str(package.parent),
            "EXPECTED_PACKAGE_ROOT": str(package),
            "UCX_TLS": "tcp,self",
            "UCX_RNDV_THRESH": "1024",
        },
        capture_output=True,
        text=True,
        timeout=45,
    )
    log = result.stdout + result.stderr
    if case == "producer-timeout":
        assert result.returncode == 86, log
        assert f"UCXX_PRODUCER_NATIVE_ISSUED case={case} accepted=True" in log
        assert "[Transport FATAL] UCXX producer slot" in log
        assert (
            "native request completion timed out" in log
            or "accepted operation deadline expired" in log
        )
    else:
        assert result.returncode == 0, log
        marker = {
            "shared": "UCXX_SHARED_CONTEXT_NATIVE_PASS",
            "partial-start": "UCXX_PARTIAL_START_NATIVE_PASS",
            "producer-healthy": "UCXX_PRODUCER_NATIVE_PASS",
        }[case]
        assert marker in log
