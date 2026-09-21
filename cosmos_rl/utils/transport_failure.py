# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend-neutral terminal transport failure, not ordinary transfer rejection.

Use only when native completion cannot be established and continuing could reuse
live storage or distributed state. No native cleanup runs on this path.
"""

import os
from typing import NoReturn


FATAL_TRANSPORT_EXIT_CODE = 86


class TransportUnusableError(RuntimeError):
    """The backend cannot establish completion; this worker must not continue."""


def fail_transport(context: str) -> NoReturn:
    # Avoid logging-handler locks. Even a diagnostic write failure must exit.
    try:
        os.set_blocking(2, False)
        os.write(
            2,
            f"[Transport FATAL] {context[:2048]}; exiting without native cleanup\n".encode(),
        )
    finally:
        os._exit(FATAL_TRANSPORT_EXIT_CODE)
