# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve explicit fatal transport status across torchrun's exit-code folding."""

from cosmos_rl.utils.transport_failure import FATAL_TRANSPORT_EXIT_CODE
import os
from pathlib import Path


def main():
    from torch.distributed.run import main as torchrun_main
    from torch.distributed.elastic.multiprocessing.errors import ChildFailedError

    try:
        torchrun_main()
    except ChildFailedError as error:
        if any(
            failure.exitcode == FATAL_TRANSPORT_EXIT_CODE
            for failure in error.failures.values()
        ):
            # Slurm's srun may wait for other tasks even after this node exits.
            # Notify the batch supervisor through its existing shared run
            # directory. This is supervisor I/O, never watchdog-thread I/O.
            marker = os.environ.get("COSMOS_FATAL_TRANSPORT_FILE")
            if marker:
                try:
                    Path(marker).touch(exist_ok=True)
                except OSError as notification_error:
                    # Preserve the fatal classification even when the shared
                    # supervisor channel is unavailable; do not call it an
                    # ordinary application exit. Cross-node containment then
                    # depends on the scheduler's configured failure policy.
                    import sys

                    print(
                        f"Fatal transport notification failed: {notification_error}",
                        file=sys.stderr,
                    )
            raise SystemExit(FATAL_TRANSPORT_EXIT_CODE) from error
        raise


if __name__ == "__main__":
    main()
