# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve explicit fatal transport status across torchrun's exit-code folding."""

from cosmos_rl.utils.transport_failure import FATAL_TRANSPORT_EXIT_CODE


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
            raise SystemExit(FATAL_TRANSPORT_EXIT_CODE) from error
        raise


if __name__ == "__main__":
    main()
