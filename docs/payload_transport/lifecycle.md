# Payload transport ownership and close

Registry backends are shared descriptors, not per-worker resource owners.
`PayloadTransport.close_data_packer(packer, timeout=5)` and
`close_producer(producer, timeout=5)` close the corresponding owned instance.
The default backend is inert; stateful extensions should implement these hooks.
Composed packers expose `close_transport(timeout=5)` directly.

Worker attachment records distinct training/validation packers. If attachment
fails, the partially initialized packer and earlier packers are closed in reverse
order. Explicitly selected backends propagate attachment errors, including errors
other than ImportError/RuntimeError. Failed cleanup remains owned and visible.
Policy execution closes its packers before distributed teardown; rollout control
closes the configured producer backend rather than checking for a UCXX method.

NCCL producer setup rolls back partial registry, communicator, executor and
listener acquisitions. Listener subscription happens before setup can report
success. Both NCCL and UCXX composition install strategy ownership before setup
and roll back strategy/prefetch startup failures. Setup and reattachment are
worker-owner operations, not concurrent with another setup operation.

Close stops admission, invokes backend cancellation, waits for owned work, and
only then releases storage. The NCCL producer joins senders before clearing
buffers. Entries with live leases or incomplete/unqueryable GPU events stay
owned rather than being freed merely because a timeout expired. Both ordinary
eviction and final cleanup follow this rule.

## Bounds and failure semantics

A close operation runs once, including blocking native teardown. The caller's
deadline covers its wait for the entire operation, not just Python thread joins.
On timeout the operation and resources remain owned; another close waits for
that same operation. It never starts competing cleanup or frees a live buffer.
Repeated successful close is inert. Reattachment requires successful completion;
a timed-out or failed close is not permission to reuse the transport.

This is not native fault recovery. A stuck CUDA/NCCL call cannot be cancelled by
a Python timeout. A worker whose close fails must fail the job and must not resume
training; process-level containment remains the launcher's responsibility. This
change does not incorporate the independent watchdog/launcher PR.

Healthy explicit close joins owned work without depending on interpreter exit.
Borrowed Redis clients and shared stream pools are not destroyed by producer
close. The NCCL registry adapter closes its own consumer control-plane client.
The deprecated arbitrary `post_redis_injection` path retains its compatibility
behavior; custom resources created by that hook require an application lifecycle
implementation and are not covered by in-tree strategy rollback guarantees.

## Validation scope

CPU tests cover individual acquisition failures, repeated/concurrent close,
timeout ownership, worker two-packer rollback, pending senders, and incomplete GPU
events represented by controlled test doubles. Live cross-node producer/consumer
cleanup and UCXX native teardown still require canary validation before release.
