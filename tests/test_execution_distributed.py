# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import subprocess
import sys


def test_real_distributed_objective_update_and_failure_agreement():
    probe = Path(__file__).with_name("execution_ddp_probe.py")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(probe),
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"gradient_equivalence": true' in result.stdout
    assert '"peer_preparation_failure": true' in result.stdout
    assert '"empty_rank": true' in result.stdout
    assert '"malformed_teacher_agreement": true' in result.stdout
