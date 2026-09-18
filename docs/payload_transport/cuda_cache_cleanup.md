# Opportunistic CUDA cache cleanup during payload transport

Applications sometimes flush the CUDA allocator after every rollout wave even
though training can continue without that flush. Such opportunistic cleanup
must be distinguished from intentional memory release, such as giving GPU
memory to another allocator after unloading a model.

## Optional per-wave or periodic cleanup

Use `cosmos_rl.utils.cuda_cache.maybe_empty_cuda_cache()` **only when skipping
the flush is acceptable**. It returns `True` if the PyTorch call ran, and `False`
when the optional request was skipped. It does not report freed bytes.

```python
from cosmos_rl.utils.cuda_cache import maybe_empty_cuda_cache

# Optional housekeeping: no other component depends on memory being returned.
maybe_empty_cuda_cache()
```

NCCL/UCXX producer and consumer setup, and generic prefetch startup, suppress
these optional requests before starting background work. Custom transports
bypassing those paths must call `suppress_opportunistic_cuda_cache_cleanup()`
before native/background startup.

An idle local queue does not prove native completion across streams and peers.
Optional suppression therefore remains sticky after close or failed setup.
It does **not** prohibit intentional memory release, intercept PyTorch calls,
or change existing model-load, checkpoint-resume, simulator, and phase-transition
cleanup sites. This PR deliberately does not guess which existing calls are
unnecessary; downstream per-wave callers must explicitly adopt the helper.

## Intentional memory release

Keep the existing `torch.cuda.empty_cache()` call when memory must be returned.
Its owner must first establish a safe boundary: stop admitting new work, complete
the relevant communication with participating peers, join owned background
operations, and ensure their GPU work has finished. A local synchronization or
Python lock alone does not prove that boundary. Do not synchronize into an
unmatched collective and assume it will complete.

This change neither implements a distributed quiescence protocol nor certifies
existing release sites as safe. It preserves their behavior instead of silently
turning an intentional release into a no-op. If the owner cannot establish a
safe boundary, defer the handoff or use process isolation; do not relabel it as
optional when another component depends on the released memory.

## Scope and tradeoffs

- Only explicitly opted-in cleanup is suppressed. Existing in-tree direct calls
  and third-party engines are unchanged, so this is not complete prevention of
  every concurrent allocator-cleanup interaction.
- Optional callers may retain allocator-reserved memory longer; PyTorch can
  reuse it, but other allocators may not. This is not a memory budget or OOM fix.
- Suppression covers both NCCL and UCXX, regardless of network transport. No
  live RDMA validation is implied.
- The startup transition waits for a helper flush already in progress. Skipped
  calls perform no CUDA synchronization, distributed barrier, or transport wait.
- A fork inherits optional suppression with a fresh lock; this does not make
  CUDA-after-fork supported. There is no delayed replay of skipped requests.
- Allocator-internal reclamation and native fault recovery remain outside scope.

The portable `tests/cuda_cache_cleanup_canary.py` checks optional suppression
alongside real background transfers and intentional release after full teardown.
It does not reproduce the original allocator deadlock or certify a full workload.
