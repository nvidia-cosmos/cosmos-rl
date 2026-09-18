# Transport failure containment

The strategy-backed prefetch scheduler owns operation deadlines. Backends own
buffer lifetimes and ordinary transfer-error handling; supervisors own process
containment. RDMA and TCP are network mechanisms below a backend, not separate
failure policies.

- Completed transfers hand off payload ownership normally.
- An ordinary backend error does not gain new retry or recovery semantics here.
  Existing backend handling remains responsible for safe release and fallback.
- A backend may raise `TransportUnusableError` when safe native completion cannot
  be established. This is terminal, not a failed episode eligible for fallback.
- An overdue strategy-backed fetch has not established completion. The scheduler
  disables reuse and exits through the same fatal path for NCCL, UCXX and future
  strategies. It does not invoke backend cancellation/cleanup on the watchdog
  thread or release possibly live cached storage first.
- Legacy packers without a transport strategy retain their terminal timeout
  exception behavior. They are not covered by the native-strategy guarantee.

This initial contract deliberately has no cancellation callback on expiration:
such a callback could hang the watchdog itself. Backend-supported safe recovery
must finish within the operation deadline. General asynchronous cancellation and
live replica replacement are not implemented.

## Supervisor boundary

Exit code 86 is reserved for an unusable transport. Cosmos's torchrun wrapper
preserves that classification from `ChildFailedError`, instead of flattening it
to exit 1. The CLI launcher contains these explicit failures; its pre-existing
handling of ordinary errors and coordinated controller shutdown remains intact.
External launchers must preserve and supervise this status themselves. The MPI
launch path is not covered by the torchrun exit-code bridge.

No Slurm propagation mechanism is added. Native Slurm templates, task-exit policy,
retry and autoresume behavior remain unchanged. There is no shared fatal marker,
extra polling loop or new kill-on-bad-exit flag. Surviving tasks may remain blocked
until the allocation reaches its configured time limit, so deployments relying
on this fallback must set a finite limit. This PR does not guarantee prompt
cross-node termination or correct job-level failure classification on every
existing early-shutdown path. Worker fatal status and allocation status are
different guarantees. Scheduler-specific containment is deferred.

The process exit intentionally bypasses final checkpointing and native cleanup.
Python scheduling is still required. This is not an OS-level watchdog and cannot
guarantee termination of unkillable kernel tasks.

## Related changes

Transport lifecycle ownership (#751) and bounded receiver memory (#754) prevent
unsafe freeing and reduce memory pressure; neither proves native completion
after a hang. This PR does not change cache cleanup, promise transport recovery,
or introduce a job-wide memory budget. Cancellation and backend parity must be
validated separately from allocation containment; TCP tests are not RDMA tests.

## Validation scope

A current-revision live NCCL control completed 20 training steps with all four
workers exiting zero. An injected lock-held stall after real rendezvous produced
fatal exit 86 at its 15-second deadline; the local CLI terminated its cohort and
exited nonzero, with no unsafe fallback. Both arms ran on one node with separate
policy and rollout processes. UCXX deadline behavior is covered by native-call
subprocess tests, not a live RDMA canary. No cross-node containment or combined
adoption with the related lifecycle/memory PRs is established by this validation.
