# Training collective lifetime and unused gradients

Expanded schedules must agree over every rank whose gradients are coupled,
including separate policy replicas. Dynamic unweighted and weighted schedules
both reconcile the full cohort. Fixed unweighted schedules exchange their
signature once upfront and follow it without per-update metadata exchanges.
Recoverable empty/bad input can contribute an empty slot before agreement;
programming errors and uncertain native completion are not data filtering.

`HighAvailabilitylNccl.operation_scope()` pins a mesh with a local reentrant lock.
Expanded updates and opt-in VLA objective updates hold it from agreement/counts
through optimizer completion. The gradient helper also holds it across all
buckets; P2P batches resolve peer ranks under the same lock. This is not a new
distributed barrier. Custom adapters own an equivalent stable group lifetime.
Sealed expanded schedules reject membership or native-communicator changes.

Readiness may be retried before native work is issued. Once native calls have
started, an error or unknown completion makes the communicator terminal:
`CollectiveOperationError` propagates, no operation is replayed, and later calls
cannot continue through a rebuilt communicator. Operands remain quarantined
until process exit because an error/abort alone does not prove all device access
has stopped. Collective error reporting makes one timed HTTP attempt, not the
operational retry/backoff chain. It cannot suppress the native error.
All calls in a native batch consume one remaining-time budget; late native or
stream success is still a deadline failure. This is a local terminal contract,
not an atomic commit protocol across processes: a peer can finish earlier, but
the failed cohort must restart from its last globally committed checkpoint.

Callers must not commit an optimizer update after this exception. Earlier
buckets may already contain reduced gradients; those are not a resumable update.
Restart from a committed checkpoint. This is not transparent collective recovery,
CUDA-driver recovery, or cross-node launcher failure propagation.

## Stable gradient layout

All replicas must supply the same ordered trainable parameters and compatible
local DTensor layouts. Buckets follow parameter dtype/order/size, not the local
`grad is None` subset. Missing local gradients occupy zero-filled slots. One
used flag per parameter travels in the existing gradient reduction; there is no
additional used-mask collective. A globally used parameter receives the reduced
gradient even on a locally unused rank. A globally unused parameter retains
`grad=None`, preserving optimizer momentum/weight-decay skip semantics. Frozen
parameters are omitted consistently by `requires_grad`.

AVG remains the default; SUM is also supported. Other reduction operations are
rejected. Existing FP32 reduction arithmetic is retained. Temporary buckets are
limited by FP32 storage size (200 MiB by default), except that a single larger
parameter is still indivisible. Flags incur storage and a small host read after
completion. This is not a total-memory bound. Strict SUM mode retains its existing
complete-gradient and participant checks for compact trainable subsets.

## Portable validation

```bash
python -m pytest -q tests/test_collective_operation.py
python -m pytest -q tests/test_collective_error_reporting.py
python -m pytest -q tests/test_distributed_kv_store.py
python -m pytest -q tests/test_training_collective_contract.py
torchrun --standalone --nproc-per-node=4 tests/training_collective_gpu_canary.py healthy
torchrun --standalone --nproc-per-node=2 tests/training_collective_gpu_canary.py partial
torchrun --standalone --nproc-per-node=2 tests/training_collective_gpu_canary.py missing-peer
```

The four-rank canary uses two DP ranks per policy replica and real native HA
NCCL between replicas. It checks weighted/unweighted, dynamic/fixed, prefetch
on/off, empty/recoverable inputs, optimizer/scheduler parity, mixed-dtype missing
gradients and replicated DTensor gradient materialization. Fault arms prove
injection was reached and require terminal failure without replay or optimizer
commit; an unrelated crash, skip, or allocation timeout is not success. Use a
finite outer timeout. These tests do not replace full-model/FSDP or multi-node
validation for deployment topologies.
