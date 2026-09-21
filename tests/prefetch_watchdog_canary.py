# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in canary: park one consumer after real NCCL rendezvous, lock held.

This models a native call that never returns while releasing the GIL. It does
not claim to reproduce the CUDA driver deadlock or call empty_cache itself.
"""

import hashlib
import os
from pathlib import Path
import threading
import time


def patch_strategy(module):
    cls = module.NCCLTransportStrategy
    original = cls._rendezvous_one
    after = int(os.environ.get("PREFETCH_HANG_AFTER_BATCH", "2"))

    def rendezvous(self, ref, pynccl):
        prepared = original(self, ref, pynccl)
        if prepared is not None and self._steps >= after:
            # Replica-local ranks are all zero in this deployment. Elect one
            # consumer through the real run-scoped Redis, not a rank guess.
            elected = getattr(self, "_canary_hang_target", None)
            if elected is None:
                elected = self._redis.set(
                    self._prefix + ":canary-prefetch-hang",
                    self._receiver_replica,
                    nx=True,
                    ex=3600,
                )
            if elected:
                self._canary_hung = True
                print(
                    f"[PREFETCH-INJECT] t={time.time():.6f} "
                    f"replica={self._receiver_replica} completed={self._steps} "
                    f"transfer={ref['transfer_id']} comm={prepared[0]} "
                    f"recv_lock_held={self._recv_lock.locked()}",
                    flush=True,
                )
                threading.Event().wait()
        return prepared

    cls._rendezvous_one = rendezvous
    original_sync = cls.sync_fetch

    def sync_fetch(self, ref):
        if getattr(self, "_canary_hung", False):
            print("[PREFETCH-INJECT] UNSAFE FALLBACK", flush=True)
        return original_sync(self, ref)

    cls.sync_fetch = sync_fetch
    print(
        f"[PREFETCH-INJECT] strategy={module.__file__} "
        f"sha256={hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}",
        flush=True,
    )


def patch_scheduler(module):
    cls = module.PrefetchDataPackerMixin
    original = cls.start_prefetch
    after = int(os.environ.get("PREFETCH_HANG_AFTER_BATCH", "2"))
    timeout = float(os.environ.get("PREFETCH_HANG_TIMEOUT_S", "15"))

    def start(self, rollouts):
        strategy = self._transport_strategy
        if strategy is not None and getattr(strategy, "_steps", 0) >= after:
            # Preserve generous cold-start budgets. Only shorten the deadline
            # once real batches have transferred successfully on this consumer.
            if os.environ.get("PREFETCH_HANG_SINGLE_WATCHDOG") == "1":
                if not hasattr(strategy, "_canary_hang_target"):
                    strategy._canary_hang_target = bool(
                        strategy._redis.set(
                            strategy._prefix + ":canary-prefetch-hang",
                            strategy._receiver_replica,
                            nx=True,
                            ex=3600,
                        )
                    )
                    print(
                        f"[PREFETCH-INJECT] replica={strategy._receiver_replica} "
                        f"elected={strategy._canary_hang_target} "
                        f"hostname={os.uname().nodename}",
                        flush=True,
                    )
                # Keep peers alive well beyond the elected worker's deadline,
                # so Slurm must terminate them instead of their own watchdogs.
                self._prefetch_timeout_s = (
                    timeout if strategy._canary_hang_target else 300.0
                )
            else:
                self._prefetch_timeout_s = timeout
        return original(self, rollouts)

    cls.start_prefetch = start
    print(
        f"[PREFETCH-INJECT] scheduler={module.__file__} "
        f"sha256={hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()} "
        f"after_batch={after} timeout_s={timeout}",
        flush=True,
    )


def main():
    """Wrap an application's normal role entrypoint; never enable in production."""
    import argparse
    import runpy
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("entrypoint", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.entrypoint.is_file():
        parser.error("entrypoint must name an existing application script")
    if not args.clean:
        from cosmos_rl.utils.payload_transport.nccl import strategy
        from cosmos_rl.utils.payload_transport import prefetch_mixin

        os.environ.setdefault("PREFETCH_HANG_SINGLE_WATCHDOG", "1")
        patch_strategy(strategy)
        patch_scheduler(prefetch_mixin)
    sys.argv = [str(args.entrypoint), *args.arguments]
    sys.path.insert(0, str(args.entrypoint.resolve().parent))
    runpy.run_path(str(args.entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
