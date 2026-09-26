# UCXX completion and storage ownership

Each native read attempt has one finite deadline covering endpoint checkout,
connection, protocol requests and retirement. An independent watchdog also covers
synchronous fallback; the caller does not need to poll `wait_prefetch()`.

A Python task timeout or cancellation does not prove that UCXX stopped accessing
its buffers. The consumer waits without cancelling native request tasks. On
uncertain completion it retains the tasks, endpoint and all operands until
process exit, latches a terminal transport failure and exits with status 86.
It does not rotate ports, close a live endpoint, recycle pinned storage, or turn
that failure into a missing sample. Late completion cannot make the client usable
again. Completed connection failures can still rotate ports, and clean stale-slot
rejections remain recoverable.

Decoded host arrays are validated before device work. Pinned buffers return to
the pool only after a recorded CUDA event proves all copy/decode work finished
within its deadline. Copy errors do not launch a fallback copy against uncertain
storage. An allocation OOM is recoverable only if a bounded device fence proves
earlier work drained; otherwise it is terminal. CPU copies own their storage and
do not leave views into a recycled buffer. Episode-length metadata is read on
the host, avoiding a separate unbounded device synchronization.

The native-contract distinction is grounded in UCXX 0.47's implementation:
[endpoint send/recv](https://github.com/rapidsai/ucxx/blob/v0.47.00/python/ucxx/ucxx/_lib_async/endpoint.py)
await request completion, while the
[request wrapper](https://github.com/rapidsai/ucxx/blob/v0.47.00/python/ucxx/ucxx/_lib/libucxx.pyx)
requests cancellation during destruction. Cancelling that waiter is not the
same as waiting for native completion.

Portable two-rank tests, also usable with two-node `torchrun`:

```bash
torchrun --standalone --nproc-per-node=2 tests/ucxx_operation_canary.py --case healthy
torchrun --standalone --nproc-per-node=2 tests/ucxx_operation_canary.py --case stale --prefetch
torchrun --standalone --nproc-per-node=2 tests/ucxx_operation_canary.py --case warm-stall --prefetch
```

Run each case both with and without `--prefetch`. The warm-stall arm completes a
first payload and then leaves an accepted payload pending. In prefetch mode the
consumer deliberately does not collect its result; its own native watchdog must
still exit. Supervisors require that receiver exit and marker before terminating
the fixture peer. Healthy/stale arms verify later payloads and previous decoded
storage after pinned-buffer reuse.

The fixture above checks consumer containment, not transparent recovery. It
uses real UCXX endpoints and the production consumer, not the production producer.

## Producer sends and shared-context retirement

An accepted producer slot has one finite `UCXXBufferConfig.send_timeout` budget
(default 30 seconds), covering status and payload sends. The operation retains
the server, shared-memory slot, endpoint and uncancelled native waiter. Only
completed sends consume the slot; an uncertain error, cancellation or timeout
terminates the process without releasing it for reuse.

Idle headers keep one receive waiter across polling intervals. Each connection
closes on its originating event loop, observes that waiter finishing, and only
then releases its receive buffer. Listener shutdown includes active connection
handshakes and handlers. Unexpected loop failure with published resources is
terminal; closing an event loop is not a native cancellation fence.

Producers and clients lease Cosmos' process-global UCXX context for their whole
owned lifetime, including idle pooled endpoints. Stopping one producer must not
reset another producer/client's worker. The last owner resets only after its
endpoints and requests have retired; admission and final reset are serialized.
`stop_server(timeout=...)` has one join budget across all server threads, not a
fresh timeout per thread. A failed join retains the shared-memory owner.

Python endpoint `close()` can conceal a native close timeout. Cosmos retains the
native endpoint handle, checks its status after close, and observes its owned
request waiters before releasing storage. A completed peer connection reset is
normal; an endpoint timeout is not. UCXX 0.50/0.51 does not expose the worker's
native cancellation-count query, so an untracked context cannot claim drain
from cancellation scheduling or an elapsed sleep. Unknown completion fails
closed. Direct external UCXX users must not bypass this ownership protocol or
reset a context while Cosmos owns it. This is not general ownership tracking for
arbitrary third-party raw UCXX calls.

The close distinction follows UCXX's
[native endpoint implementation](https://github.com/rapidsai/ucxx/blob/v0.51.01/cpp/src/endpoint.cpp),
where blocking cancellation/close can exhaust its attempts without throwing.

Portable actual-producer controls:

```bash
python tests/ucxx_context_canary.py --case shared
python tests/ucxx_context_canary.py --case partial-start
torchrun --standalone --nproc-per-node=2 tests/ucxx_producer_canary.py --case healthy
torchrun --standalone --nproc-per-node=2 tests/ucxx_producer_canary.py --case healthy --prefetch
UCX_RNDV_THRESH=1024 torchrun --standalone --nproc-per-node=2 tests/ucxx_producer_canary.py --case producer-timeout
```

Use a working routed UCX interface/transport for two-node runs. The fault case
receives the accepted status but deliberately withholds the payload receive;
its supervisors require the producer's own fatal exit and phase marker before
stopping the idle peer. `--cpu` supports local harness checks; it does not replace
GPU validation. The shared-context test owns two producers on four loops plus
a pooled client and verifies continued reads after retiring the first producer.
The partial-start test verifies bounded cleanup and a fresh healthy restart.

No scheduler-wide failure propagation, transparent peer recovery, unconditional
cache-cleanup prevention, UCXX receive-memory budget or untested RDMA guarantee
is claimed by these TCP ownership tests.
