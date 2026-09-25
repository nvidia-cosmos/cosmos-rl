# Completion is part of the prefetch deadline

The independent timer and the background worker use the same absolute deadline.
Before publishing a fetched or CPU-prepared result, the worker checks whether
that deadline has expired. A delayed timer callback cannot turn late completion
into success merely because the worker cancels the timer first.

Proven pre-receive consumer backpressure keeps its existing deadline extension.
Once work completes within the permitted budget, later result collection is not
active transport work and does not cause a timeout.

Expired work cannot be consumed, used for synchronous fallback, or made reusable
by a later successful return. The terminal path retains returned payloads,
prepared aliases and receive-budget charges until process exit. Explicit close
after deadline failure does not enter backend cleanup. Strategy-backed packers
exit with status 86; legacy strategy-less packers retain their timeout exception
behavior. This does not make native cancellation safe or provide job-wide failure
propagation.

Portable checks deliberately prevent the timer callback from running:

```bash
torchrun --standalone --nproc-per-node=2 tests/prefetch_completion_canary.py --case healthy
torchrun --standalone --nproc-per-node=2 tests/prefetch_completion_canary.py --case late --prepared
```

Run healthy/late cases both with and without `--prepared`, on one or two nodes.
The CUDA late case asserts a real pending copy before returning after the injected
deadline; supervisors require the worker's own fatal exit. `--cpu` tests the same
completion protocol without CUDA. This is not a transport-pair or scheduler
propagation test.
