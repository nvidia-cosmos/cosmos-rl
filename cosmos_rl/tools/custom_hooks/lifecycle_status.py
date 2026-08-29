# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable terminal TAO lifecycle status writes.

TAO Core's shared logger remains responsible for progress events. Terminal
events are appended directly so a successful worker cannot leave a final
RUNNING record because of logger state inherited by the distributed worker.
"""

import json
import os
from datetime import datetime
from typing import Mapping


_TERMINAL_STATUSES = frozenset({"SUCCESS", "FAILURE"})


def is_lifecycle_status_owner(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether this process owns the entrypoint terminal status.

    Policy workers emit progress through ``TAOStatusLogger``. The controller
    waits for those workers and is therefore the only process that can append a
    terminal lifecycle record after the distributed run has actually ended.
    Direct, role-less rank-zero execution uses the same ownership contract.
    """
    env = os.environ if environ is None else environ
    role = env.get("COSMOS_ROLE", "")
    node_rank = int(env.get("NODE_RANK", "0"))
    local_rank = int(env.get("LOCAL_RANK", env.get("RANK", "0")))
    return role in {"", "Controller"} and node_rank == 0 and local_rank == 0


def append_terminal_status(filename: str, status: str, message: str) -> None:
    """Append and verify one terminal TAO JSONL record."""
    if status not in _TERMINAL_STATUSES:
        raise ValueError(f"terminal TAO status must be one of {_TERMINAL_STATUSES}, got {status!r}")

    now = datetime.now()
    record = {
        "date": f"{now.month}/{now.day}/{now.year}",
        "time": f"{now.hour}:{now.minute}:{now.second}",
        "status": status,
        "verbosity": "INFO" if status == "SUCCESS" else "ERROR",
        "message": message,
    }
    os.makedirs(os.path.dirname(os.path.realpath(filename)), exist_ok=True)
    with open(filename, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
        stream.flush()
        os.fsync(stream.fileno())

    with open(filename, encoding="utf-8") as stream:
        final_line = next(line for line in reversed(stream.readlines()) if line.strip())
    final_record = json.loads(final_line)
    if final_record.get("status") != status:
        raise RuntimeError(
            f"terminal TAO status verification failed: expected {status}, "
            f"found {final_record.get('status')!r}"
        )
