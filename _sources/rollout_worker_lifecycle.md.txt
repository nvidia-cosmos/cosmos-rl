# Standalone rollout resource ownership

`LLMRolloutWorker.execute()` owns teardown on both normal return and work failure.
`destroy_worker()` explicitly calls the control worker's shutdown method; it
does not depend on object deletion or interpreter exit. The control constructor
also unwinds its acquired resources if initialization fails before returning.

Shutdown first closes work admission and joins the command, teacher, prompt
prefetch and asynchronous generation producers. Reward executor work receives a
bounded drain before its packers are closed; a timeout retains the executor and
dependent resources. Producer transport cleanup precedes weight-sync
teardown/abort, preserving the existing transport ordering and timeout semantics.
After dependent work drains, the worker calls the backend's `shutdown()`, closes
attached packer transports, and calls each distinct packer's optional `close()`.
Aliased training/validation packers are closed once. Heartbeat cleanup and
controller unregistration are attempted even when another cleanup action fails.

The engine and packers remain alive while their producers may still use them.
A failed join/drain is an error, not permission to free those resources. A
cleanup error after successful work propagates to the caller. If work already
failed, that original exception is preserved and cleanup failures are logged.
Repeated teardown does not repeat successful close operations; a failed teardown
remains failed rather than silently becoming success.

An asynchronous scheduler exception is raised in the owning control thread
instead of appearing as an indefinitely empty completion queue. If the failure
is first observed during shutdown, teardown reports it after the scheduler has
quiesced and its dependent resources have been safely released.

The entrypoint records the identity of a default process group it creates. It
destroys that group only after dependent cleanup, including failed construction.
A pre-existing caller-owned group is preserved, as is a replacement group with
a different identity. If dependent cleanup cannot complete, the group is retained
and the failure is reported; application/process supervision still owns final
containment. No new distributed stop protocol or global shutdown deadline is added.
TRT-LLM's executor retains responsibility for its own distributed initialization;
the wrapper delegates to its existing engine shutdown API.

## Backend and packer responsibilities

Backends implement the existing `RolloutBase.shutdown()` method. It must tolerate
partial initialization and repeated calls. `RolloutBase` invokes it when
`post_init_hook()` fails. Synchronous/asynchronous vLLM and TRT-LLM delegate to
their existing engine/executor shutdown methods, including an acquired engine
whose initialization did not finish. Simulator-specific cleanup belongs inside
the custom backend. `RolloutGenerationMixin.shutdown_generation()` reports a
still-running preparation thread instead of releasing resources under it.
After that timeout, setup remains closed: complete `shutdown_generation()`
after the callback exits before calling `setup_generation()` again.

`BaseDataPacker` has **no general close method**. Stateless built-in packers need
none; resource-owning custom packers can expose `close()`. Existing transport
cleanup remains separate and is used without adding a callback registry. Passing
a packer into the standalone worker transfers its cleanup responsibility to that
worker. Do not share that instance with an independently running owner.

A factory/constructor that raises before returning an object must unwind the
resources it acquired internally: callers cannot close an object they never
received. This includes custom packer factories and constructors overriding the
base backend constructor. Resources already returned to the control worker are
unwound there if later setup fails.

Colocated rollout control borrows shared policy model/packer state; it does not
take standalone ownership of those objects. This change does not redesign the
colocated policy/checkpoint owner's lifecycle. Generation, training objectives,
weight-sync admission and checkpoint semantics are unchanged.
