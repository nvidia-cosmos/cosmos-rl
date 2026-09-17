"""CPU-only fault injection: native hangs must not enter fallback or teardown."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("action", ["collect", "no-collect", "shutdown"])
def test_nccl_prefetch_timeout_exits_without_cleanup(action):
    # A separate process exercises the real os._exit, rather than mocking away
    # the only operation capable of terminating an uninterruptible native call.
    code = r"""
import threading
import sys
import ctypes
from cosmos_rl.utils.payload_transport.prefetch_mixin import PrefetchDataPackerMixin
from cosmos_rl.utils.payload_transport.nccl.strategy import NCCLTransportStrategy

class HungStrategy(NCCLTransportStrategy):
    def fetch_batch(self, tasks):
        with lock:
            print("receive lock held", flush=True)
            # ctypes.CDLL releases the GIL, like the NCCL wrapper. Block
            # inside an actual native call, not merely a Python sleep.
            ctypes.CDLL(None).pause()

    def before_join(self):
        # A concurrent explicit shutdown must not disarm the fetch watchdog
        # before its native abort returns.
        threading.Event().wait()

    def sync_fetch(self, ref):
        raise AssertionError("timeout must not enter fallback")

lock = threading.Lock()
packer = PrefetchDataPackerMixin()
packer.set_transport_strategy(HungStrategy())
packer._setup_prefetch(prefetch_timeout=0.2)
packer.start_prefetch([{"_nccl": True}])
if sys.argv[1] == "collect":
    packer.wait_prefetch()
elif sys.argv[1] == "shutdown":
    packer.shutdown_prefetch()
else:
    # Model a trainer stuck in CUDA/collective work, never collecting.
    threading.Event().wait()
raise AssertionError("timeout returned")
"""
    result = subprocess.run(
        [sys.executable, "-c", code, action],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1, result.stderr
    assert "receive lock held" in result.stdout
    assert "NCCL payload FATAL" in result.stderr
    assert "fallback and reuse disabled" in result.stderr
    assert "Traceback" not in result.stderr
