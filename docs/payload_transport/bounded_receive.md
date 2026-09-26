# Bounded NCCL payload reception

This opt-in mode limits **receiver-device payload storage**, not total device
memory. Configuration names and defaults are provisional pending maintainer
agreement. UCXX and Redis do not implement this budget.

```toml
[custom]
nccl_receive_budget_bytes = 8589934592  # example: 8 GiB, not a workload recommendation
nccl_receive_admission_timeout = 30.0  # seconds; excludes an active consumer's final use
```

Omitting `nccl_receive_budget_bytes`, or setting it to `0`, preserves the
existing concurrent NCCL receive path. A positive integer enables bounded
reception. No disk or host-payload spill is performed.

## Admission and accounting

The batch API must return the whole decoded batch. For each payload in input
order, let D be its decoded storage bytes and R its wire bytes (including the
32-byte device header). Before contacting any sender, reserve:

```
sum(D) + max(R)
```

Reservations for other batches, including unreleased consumer leases, also
count against the same per-strategy budget. This admission rule prevents a
partially received batch from occupying all capacity while waiting for space
that can only be released after handing the complete batch to its consumer.
A batch that cannot fit fails immediately with the required bytes and suggested
remedies. Waiting solely for an already handed-off consumer is safe backpressure,
not a native transport failure: it has no receive deadline and shutdown wakes it
immediately. An uncollected/queued reservation does not qualify for that exemption;
its admission timeout remains finite and actionable.

After admission, grow the wire workspace using currently available capacity,
up to `sum(R)`, without waiting while holding a reservation. Receive as many
consecutive payloads concurrently as fit that workspace. Each sender rendezvous
is immediately followed by its matching receive enqueue; independent NCCL
communicators are not grouped. Receive and decode completion are observed before
advancing to the next window. The existing alignment clones remain; raw-buffer
aliases disappear when the window's receive helper returns.
Reservations shrink to actual unique decoded storage at handoff. Truncated views
are charged for their **full backing storage**, and aliases of the same storage
are counted once. Peak reservations are conservative admission figures, distinct
from peak charged tensor bytes.

Compute on the prior batch and reception of the next still overlap when both
fit. A minimal budget yields single-payload windows; additional available bytes
allow concurrent receives. Workspace stays reserved until batch handoff. This
avoids an admission cycle but can temporarily reduce capacity available to other
requests. Measure throughput before selecting a production budget.

The cap excludes model/optimizer tensors, autograd state, consumer-created
copies/minibatch caches, CPU control-plane/header copies, NCCL internal memory,
and CUDA allocator reserved memory/rounding. Allocations can still OOM because
of those excluded users or fragmentation. Multiple strategies/devices have
independent budgets; this is not a process-wide cap.

## Consumer release contract

Bounded reception requires the prefetch API. A synchronous cache miss fails
explicitly instead of creating an untracked second receive. Repeated cache reads
are supported until final release. Call the following from the consumer thread:

```python
packer.start_prefetch(rollouts)
packer.wait_prefetch()
# Resolve/read the batch as many times as required.
# Training, reward and visualization readers may use additional CUDA streams.
# Drop every external payload/tensor/view alias when its final use is enqueued.
packer.release_prefetch(streams=(training_stream, reward_stream, visualization_stream))
```

Explicit producer `MISSING` replies and references rejected before rendezvous
are recorded as safe outcomes in that same batch lease. Resolving them returns the existing packer's
missing-sample outcome without trying an unleased synchronous receive. This also
works during background preparation and for an entirely rejected batch. Outcome
metadata expires at release, so it cannot suppress a later batch's unknown miss.
Absence from the cache alone is not proof of rejection: unknown misses, closed
admission, accepted payload identity mismatch, and uncertain native completion
remain errors. A completed receive with the wrong header does not prove that the
accepted stream can be reused. No broad exception is converted into a dropped sample.

`release_prefetch` records completion events on all supplied reader streams and
the current stream on each payload device, then waits before clearing the cache
and returning its budget. Event failure leaves the lease and its charge intact.
The lease independently retains its original backing storages even if a
consumer removes or replaces payload-dictionary entries. Those storages remain
charged until release. Consumer-created views do not extend the contract automatically: **every reader
must be finished or represented by a supplied stream, and no external alias may
be used or retained after release**. An external alias retained after release
violates the budget contract. Calls to release/collect must be serialized on the
consumer thread.

### Prepared batches (#753)

This builds on merged #753 and its #747 watchdog dependency. Prepared prefetch
retains the received lease independently of its thread-local cache. Consuming
the preparation future transfers that lease to the ordinary consumer cache;
it does not return capacity. The handoff waits until the background worker has
dropped its own prepared aliases. After final training readers and after dropping
prepared aliases, standalone consumers call `release_prefetch(streams=...)`. This also
applies when preparation raises: discard the failed prepared state and release
the handed-off lease, or shut down the packer before consumption.

Both `shutdown_prefetch()` and the owned `close_transport()` path release
unconsumed queued/prepared leases only after the worker exits. A close timeout
retains those leases, and a release failure keeps the affected lease owned.
Already-consumed leases remain the consumer's responsibility. Do not retain
prepared futures/results across shutdown. CPU preparation completing is not
proof that all payload aliases have finished. No automatic early release is
inferred for independent CPU copies. Memory created by preparation remains
outside the transport budget.

### Generic expanded-training integration

The policy worker's `run_training_step` integrates final release for
`ExpandedSampleBatching`: inline preparation starts/collects a batch-owned receive,
and prepared prefetch transfers the same lease. After the complete update returns,
the entrypoint drops its preparation/minibatch/future aliases and releases the
consumer cache. Both dynamic and fixed schedules are supported, with or without
background preparation. The default/current CUDA streams and `trainer.train_stream`
are included; override `trainer.training_payload_streams()` for additional reward,
visualization, or other readers.

An expanded trainer must not retain samples or views past its step return, and
its report must not retain autograd state or alias received storage. Invalid
tensor reports are rejected without returning capacity. Unexpected preparation,
training, or release errors retain the lease; they do not prove final use.
The default/unbounded path is unchanged. A fixed/legacy trainer using this worker
entrypoint is rejected before rendezvous when the budget is enabled: it needs an
explicit consumer integration, not an inferred release boundary.

Closing the backend does not release a consumer-owned batch. Its cache lease
remains available for `release_prefetch(streams=...)` after final use; supply the
reader streams just as during normal operation.

The next prefetch may be started before release to overlap communication and
compute. Release the old batch before `wait_prefetch`/`collect_prefetch` for the
next one; attempting to replace an unreleased cache raises an actionable error.
`release_prefetch` preserves the rollout double-buffer scheduling state.

### Interrupted receive ownership

An allocation failure after an earlier receive was posted is not an ordinary
missing episode. Until receive completion has been observed, bounded reception
retains raw storage, communicator pins, transfer streams and the entire admitted
reservation. Enqueue/event/completion failures have the same terminal contract.
The worker exits through the backend-neutral fatal transport path; no native
cleanup or retry is attempted there. Quarantined ownership persists to process
exit even if an intermediate caller catches the exception. An abort or a sync
of an unrelated compute stream is not proof that a transfer stopped using memory.

A decode failure after proven receive completion can release storage only after
decode copies complete too. Failure to establish that completion retains the
allocated decode storage and charge, not just its accounting counter.

The background worker drops its result reference immediately after queue
handoff. Shutdown closes admission and uses the existing communicator abort/join
lifecycle. It releases unconsumed queued batches after the worker stops. It does
**not** infer final use of a batch already handed to a consumer: the consumer
must still release that batch. Memory/prefetch errors are fatal to the attempted
collection and never silently turn into empty episodes. The shared independent
watchdog covers queueing, receive and preparation even without consumption.
Only a pre-receive wait whose entire reservation belongs to a handed-off consumer
is exempt. Native work has not started in that phase; releasing capacity starts a
fresh transport/preparation deadline, independently observed even when the consumer
never waits for the result. Native hangs are not exempt, and strategy-backed
expiry exits without native cleanup. Do not retry a timed-out
packer. Existing per-transfer recovery/fallback behavior remains in the receive
helper. Failure to prove CUDA completion closes admission and retains the
reservation; restart that receiver rather than reusing its capacity.

Downstream custom loops must honor the expanded final-reader contract or wire
their standalone release boundary explicitly, and manage their own caches.
No application-specific reader lifetime is inferred by this Cosmos-only change.

## Telemetry

`NCCLTransportStrategy.receive_memory_stats()` returns:

- `budget_bytes`, `reserved_bytes`, `peak_reserved_bytes`;
- `raw_receive_bytes`, `decoded_leased_bytes`;
- `current_tensor_bytes`, `peak_tensor_bytes`;
- `admission_waits` (number of admissions that had to wait).

Tensor charges remain conservative until completion is observed. Each completed
bounded batch logs its snapshot. Allocation/decode failures log transfer ID,
schema, and budget attribution; receive-buffer allocation failures also preserve
the existing communicator resynchronization behavior.

## Standalone PR acceptance

Local CPU tests use real tensor allocation and unpacking with faked NCCL traffic.
They cover variable sizes, odd field alignment, backing-storage accounting,
repeated reads/batches, admission pressure/timeouts, shutdown, decode OOM cleanup,
worker reference release, stream-event ordering and lease retention on errors.
A CUDA test covers views and multiple reader streams when hardware is available.
These CPU tests alone are not evidence of a GPU remedy or workload-level
loss equivalence.

The PR can be validated without NDAS. Run the following on two GPUs, and
require zero skipped acceptance tests:

```sh
python -m pytest tests/test_receive_memory.py tests/test_receive_memory_cuda.py tests/test_nccl_e2e.py -v
python tests/receive_memory_cluster_probe.py --steps 20 --output results.json
torchrun --standalone --nproc-per-node=2 tests/receive_outcome_canary.py
```

The outcome canary also supports a two-node torchrun launch. It uses actual Redis
rendezvous and GPU NCCL transfers, injects missing buffers and pre-rendezvous
schema rejection, and
checks both entirely rejected and mixed batches with preparation on/off. Each
case must release its lease, avoid fallback receives, and complete the next
healthy batch. A separate real-receive header injection must close admission and
raise a terminal error, without trying to reuse the pair. Test-only Gloo barriers coordinate cases; this does not claim
peer-failure propagation or recovery from uncertain GPU completion.

`test_receive_memory_cuda.py` uses real Redis/NCCL, three producer endpoints with
variable and unaligned schemas, and a standalone consumer. It compares 20 SGD
steps against direct-input data, loss and gradients, both with the budget disabled
and enabled. It also asserts next-batch completion during a pending CUDA reader,
backpressure until a slow reader releases capacity, cancellation during admission,
oversized rejection before contacting a sender, real CUDA allocation cleanup
after an injected decode error, missing-payload recovery, and storage retention
when a consumer mutates its payload dictionary. The injected error is not a
physical device OOM. All batch leases and charges must be released at final use.

The repeated-transfer probe compares peak allocated memory and mean fetch time
at batch sizes 16/32/64 for 20 iterations per mode. It checks each process exit,
payload values, repeated reads, stream completion, budget usage, and lower
steady-state receiver peaks. It has no NDAS dependency.

## Downstream workload evaluation (outside the PR gate)

A deployment may separately evaluate its pinned model/data/image with budget
disabled and enabled. That evaluates the application's final-reader integration
and workload-specific memory savings; it does not block the standalone Cosmos
PR. For such an evaluation, record separately for **every trainer**:

- progress through all iterations and matching data/loss within existing tolerance;
- raw, decoded/leased, reservation and charged-tensor peaks;
- CUDA allocated/reserved peaks and device-wide memory, independently of payload accounting;
- OOM count and allocation context, admission waits, step time and throughput;
- recovery under slow readers, producer failure, cancellation and repeated restart.

Include all final NDAS readers in the release contract. Compare sustained receive
peaks and throughput with the motivating workload, without per-wave cache flushes.
Keep source/image hashes, job configuration, and per-trainer logs with results.
## Previous serial-window validation (not evidence for the concurrent revision)

The previous serial implementation completed successfully on two H100 80GB GPUs.
All **41 contract/acceptance/round-trip tests passed with zero skips**,
including standalone data/loss/gradient equivalence, real prefetch overlap,
backpressure, admission cancellation, injected decode-error cleanup, and dictionary
mutation while a CUDA reader remained pending. The lease now holds unique backing
storage independently of the mutable payload dictionaries.

The real Redis/NCCL synthetic producer/consumer completed 20 batches per
baseline/bounded case at 16/32/64. Steady-state receiver allocated peaks (MiB)
were 768.7/272.2, 1537.4/528.5, and 3074.7/1040.9 respectively. Mean bounded
fetch-time increases were 3.6%, 4.0%, and 3.9%. This measures the combined incremental
receive and explicit-release change with roughly 16 MiB payloads. Peak memory
is PyTorch receiver allocation, and fetch time is not application throughput.
Separate standalone SGD tests establish exact data/loss/gradient equivalence;
separate delayed-reader tests establish actual prefetch overlap and backpressure.

The original NDAS workload and its final-reader integration were not exercised;
they are outside the standalone PR gate, and no NDAS OOM-remedy claim is made.
The validation report accompanying the change contains source hashes, commands,
job accounting, per-iteration metrics, and test XML.
