# Optional CUDA cache cleanup during payload transport

Use `cosmos_rl.utils.cuda_cache.empty_cuda_cache()` instead of direct
`torch.cuda.empty_cache()` calls in Cosmos extensions. It returns `True` if the
PyTorch call ran, and `False` when optional cleanup was suppressed. It does not
report freed bytes or release live tensors.

Before asynchronous payload transport starts, cleanup behaves as before. NCCL
and UCXX producer/consumer setup, and the backend-neutral prefetch scheduler,
permanently suppress explicit allocator flushing in their process. All in-tree
explicit cache-cleanup calls use this policy. Custom transports that bypass
these setup paths must call `suppress_cuda_cache_cleanup()` before starting any
native/background work.

Suppression and flushing share a lock only to serialize startup against a flush
already in progress. The lock is **not** a claim that native communication can
be made safe by locking an `empty_cache()` call. No new CUDA synchronization,
cross-rank barrier, transport pause, or teardown wait is introduced by skipped
cleanup. Existing pre-start flushes can still block as they could before.

## Why suppression rather than an idle-queue check

An empty queue or completed Python future does not establish completion of all
GPU streams or remote peers. Flushing the caching allocator can interact with
concurrent communication. Without a proven global idle boundary, automatically
flushing between waves or after a local close would reintroduce that risk.

The policy therefore stays suppressed after successful close, failed setup, and
failed/timed-out teardown. Requests are skipped, not queued for later replay.
There is deliberately no force/reset option. Process exit reclaims resources;
a new process starts with a fresh policy. A fork inherits suppression and gets
a fresh policy lock; this does not make using CUDA after fork supported.

## Tradeoffs and scope

- Allocator-cached memory remains reusable by PyTorch but may remain reserved
  from other GPU users longer. This is not a device-memory budget or OOM fix.
- The policy is process-wide, conservatively covering all devices and both
  transport backends, including UCXX configurations using RDMA.
- Direct application or third-party `torch.cuda.empty_cache()` calls bypass the
  helper. They must be removed or migrated; PyTorch is not monkey-patched.
- CUDA allocator-internal reclamation on allocation pressure, third-party
  engine cleanup, and arbitrary native transport faults are not controlled.
- This prevention change is independent of transport deadline containment and
  resource-lifetime teardown changes; those remain necessary for other faults.

The original successful run without per-wave flushing supports this prevention
policy, but does not prove the precise native deadlock mechanism or guarantee
that every transport stall has the same cause.
