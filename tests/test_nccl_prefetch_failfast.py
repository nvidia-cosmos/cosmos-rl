"""CPU-only fault injection: native hangs must not enter fallback or teardown."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("action", ["collect", "no-collect", "shutdown"])
@pytest.mark.parametrize("backend", ["nccl", "ucxx"])
@pytest.mark.parametrize("mode", ["plain", "prepared-fetch", "prepared-cpu"])
def test_prefetch_timeout_exits_without_cleanup(action, backend, mode):
    # A separate process exercises the real os._exit, rather than mocking away
    # the only operation capable of terminating an uninterruptible native call.
    code = r"""
import threading
import sys
import ctypes
from cosmos_rl.utils.payload_transport.prefetch_mixin import PrefetchDataPackerMixin
from cosmos_rl.utils.payload_transport.nccl.strategy import NCCLTransportStrategy
from cosmos_rl.utils.payload_transport.ucxx.strategy import UCXXTransportStrategy

base = NCCLTransportStrategy if sys.argv[2] == "nccl" else UCXXTransportStrategy
class HungStrategy(base):
    def filter_prefetch_tasks(self, rollouts):
        return list(enumerate(rollouts))

    def fetch_batch(self, tasks):
        if sys.argv[3] == "prepared-cpu":
            return {}
        with lock:
            print("receive lock held", flush=True)
            entered.set()
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
entered = threading.Event()
packer = PrefetchDataPackerMixin()
packer.set_transport_strategy(HungStrategy())
packer._setup_prefetch(prefetch_timeout=0.2)
def prepare():
    print("CPU preparation entered", flush=True)
    entered.set()
    ctypes.CDLL(None).pause()
if sys.argv[3] == "plain":
    packer.start_prefetch([{"_nccl": True}])
else:
    future = packer.start_prepared_prefetch([{"_nccl": True}], prepare)
assert entered.wait(5), "fault injection did not start"
if sys.argv[1] == "collect":
    if sys.argv[3] == "plain":
        packer.wait_prefetch()
    else:
        future.result()
elif sys.argv[1] == "shutdown":
    packer.shutdown_prefetch()
else:
    # Model a trainer stuck in CUDA/collective work, never collecting.
    threading.Event().wait()
raise AssertionError("timeout returned")
"""
    result = subprocess.run(
        [sys.executable, "-c", code, action, backend, mode],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 86, result.stderr
    marker = (
        "CPU preparation entered" if mode == "prepared-cpu" else "receive lock held"
    )
    assert marker in result.stdout
    assert "Transport FATAL" in result.stderr
    assert "fallback and reuse disabled" in result.stderr
    assert "Traceback" not in result.stderr


def test_prepared_transport_unusable_exits_without_consumption():
    code = r"""
import threading
from cosmos_rl.utils.payload_transport.prefetch_mixin import PrefetchDataPackerMixin
from cosmos_rl.utils.transport_failure import TransportUnusableError

class Packer(PrefetchDataPackerMixin):
    def _filter_prefetch_tasks(self, rollouts):
        return [(0, "ref")]
    def _fetch_batch(self, tasks):
        raise TransportUnusableError("injected terminal transport failure")

packer = Packer()
packer._setup_prefetch(prefetch_timeout=60)
packer.start_prepared_prefetch([], lambda: None)
threading.Event().wait()
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 86, result.stderr
    assert "injected terminal transport failure" in result.stderr
