#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single-process NCCL payload-transport preflight diagnostic.

The NCCL payload transport moves rollout tensors GPU->GPU via NCCL P2P with a
Redis control plane.  A *full* 2-rank send/recv needs two GPUs / two processes
and is covered by ``tests/test_nccl_e2e.py``.  This tool is the cheap,
single-process preflight you run on a compute node *before* burning a full
allocation: it verifies the environment pieces that, when broken, make the
transport fail at runtime -- CUDA, ``pynccl`` (unique-ID creation), the in-tree
``nccl`` package registration, and Redis reachability -- and prints the NCCL
tuning env for the record.

Usage::

    python -m cosmos_rl.tools.profiler.check_nccl [-v]
    # Redis endpoint (optional): COSMOS_TEST_REDIS_HOST / COSMOS_TEST_REDIS_PORT
    #   or COSMOS_REDIS_HOST / COSMOS_REDIS_PORT; defaults to 127.0.0.1:6379.

Exit code 0 when all *critical* checks pass (CUDA + pynccl + in-tree package);
Redis is reported as a warning (control plane may live elsewhere).
"""

from __future__ import annotations

import os


def _redis_endpoint():
    host = (
        os.environ.get("COSMOS_TEST_REDIS_HOST")
        or os.environ.get("COSMOS_REDIS_HOST")
        or "127.0.0.1"
    )
    port = int(
        os.environ.get("COSMOS_TEST_REDIS_PORT")
        or os.environ.get("COSMOS_REDIS_PORT")
        or 6379
    )
    return host, port


def check_nccl(verbose: bool = False) -> int:
    ok = True

    # 1. Torch + CUDA -------------------------------------------------------
    print("=== Torch / CUDA ===")
    try:
        import torch

        print(f"  torch:        {torch.__version__}")
        cuda_ok = torch.cuda.is_available()
        print(f"  cuda avail:   {cuda_ok}")
        if cuda_ok:
            n = torch.cuda.device_count()
            print(f"  device count: {n}")
            print(f"  cuda/runtime: {torch.version.cuda}")
            if verbose:
                for i in range(n):
                    print(f"    cuda:{i} -> {torch.cuda.get_device_name(i)}")
            if n < 2:
                print(
                    "  ** only 1 CUDA device visible -- 2-rank P2P needs >= 2 "
                    "(the full E2E will self-skip here) **"
                )
        else:
            print("  FAIL: CUDA not available; NCCL P2P cannot run")
            ok = False
    except Exception as e:
        print(f"  FAIL: torch import/CUDA query failed: {e}")
        ok = False

    # 2. pynccl -------------------------------------------------------------
    print("\n=== pynccl ===")
    try:
        from cosmos_rl.utils import pynccl

        uid = pynccl.create_nccl_uid()
        print(f"  create_nccl_uid: OK (len={len(uid)})")
        if hasattr(pynccl, "get_nccl_timeout_ms"):
            print(f"  default timeout: {pynccl.get_nccl_timeout_ms()} ms")
    except Exception as e:
        print(f"  FAIL: pynccl unavailable: {e}")
        ok = False

    # 3. In-tree NCCL payload-transport package ----------------------------
    print("\n=== In-tree NCCL payload transport ===")
    try:
        from cosmos_rl.utils.payload_transport import PayloadTransportRegistry
        from cosmos_rl.utils.payload_transport.nccl import (  # noqa: F401
            NCCL_COMPLETION_PREFIX,
            NCCLDataPackerMixin,
            NCCLRolloutMixin,
            NcclPayloadTransport,
        )

        transport = PayloadTransportRegistry.get_optional("nccl")
        if isinstance(transport, NcclPayloadTransport):
            print(
                f"  registered:   nccl (completion_prefix="
                f"{transport.completion_prefix!r})"
            )
        else:
            print("  FAIL: 'nccl' transport not registered")
            ok = False
        print("  mixins:       NCCLRolloutMixin + NCCLDataPackerMixin import OK")
    except Exception as e:
        print(f"  FAIL: in-tree nccl package import failed: {e}")
        ok = False

    # 4. Redis control plane (warning, not fatal) --------------------------
    print("\n=== Redis control plane ===")
    host, port = _redis_endpoint()
    try:
        import redis

        client = redis.Redis(host=host, port=port, socket_connect_timeout=2)
        client.ping()
        print(f"  reachable:    {host}:{port} (PING OK)")
    except Exception as e:
        print(
            f"  WARN: Redis at {host}:{port} not reachable ({type(e).__name__}: {e}); "
            "the control plane may run elsewhere -- verify the worker's endpoint."
        )

    # 5. NCCL tuning env (informational) -----------------------------------
    print("\n=== NCCL env (informational) ===")
    for var in (
        "NCCL_DEBUG",
        "NCCL_IB_HCA",
        "NCCL_SOCKET_IFNAME",
        "NCCL_P2P_LEVEL",
        "NCCL_BUFFSIZE",
        "CUDA_VISIBLE_DEVICES",
    ):
        print(f"  {var + ':':<22s} {os.environ.get(var, '(not set)')}")

    print()
    if ok:
        print(
            "RESULT: PASS -- NCCL transport preflight OK "
            "(full 2-rank send/recv validated by tests/test_nccl_e2e.py)."
        )
    else:
        print("RESULT: FAIL -- see the checks above.")
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="NCCL payload-transport preflight diagnostic."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="per-device detail"
    )
    args = parser.parse_args()
    sys.exit(check_nccl(verbose=args.verbose))
