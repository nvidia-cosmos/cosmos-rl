# In-tree NCCL Payload Transport

By default Cosmos-RL ships rollout completion payloads (token IDs,
log-probs, reward/action tensors, …) from rollout workers to the
controller over **Redis streams**. For large payloads — e.g. VLA policies
that emit high-dimensional action tensors — the Redis data path is a
bottleneck.

**NCCL payload transfer** is an opt-in backend that moves the payload
tensors **GPU→GPU with NCCL point-to-point** while Redis stays the
*control plane*: receivers publish per-transfer requests, senders
acknowledge them, per-pair NCCL unique-IDs are exchanged, and cleanup
messages retire producer buffers after their native readers finish.

Enable it with:

```toml
[custom]
payload_transfer = "nccl"     # one of "redis", "nccl", "ucxx"
# optional tunables (defaults shown):
nccl_prefetch_timeout = 30.0      # requested outer prefetch budget (s)
nccl_read_max_attempts = 2        # pre-acceptance attempts; not accepted-send replay
nccl_recv_timeout = 5.0          # warm operation budget (s)
nccl_first_transfer_timeout = 30.0 # cold operation budget (s)
nccl_prefetch_batch_hint = 8       # used to size the outer budget floor
```

The legacy boolean `[custom].nccl_payload_transfer = true` still resolves
to `"nccl"` (deprecated alias).

The effective prefetch budget is at least `nccl_prefetch_batch_hint *
nccl_read_max_attempts * nccl_first_transfer_timeout`: 480 seconds with these
defaults, not 30. This outer budget does not extend an individual accepted
operation's deadline or permit unsafe retries. The independent watchdog and
completion-time check preserve terminal failure even if the consumer never
collects its result. See [completion deadlines](../prefetch_completion_deadline.md).

## Components

| Module | Role |
| --- | --- |
| `nccl/protocol.py` | Redis key / channel builders, transfer-id parsing (pure strings). |
| `nccl/transport.py` | `NcclPayloadTransport` backend: registration, `attach_data_packer`, controller-side discard cleanup. |
| `nccl/rendezvous.py` | Per-transfer request/ack handshake + per-pair unique-ID exchange over Redis. |
| `nccl/comm_cache.py` | Lazy 2-rank communicator cache: LRU cap, bounded concurrent init, health quarantine. |
| `nccl/buffer_registry.py` | Producer GPU send-buffer registry with bounded backpressure + idempotent free. |
| `nccl/streams.py` | Per-process transfer-stream pool + CUDA event helpers. |
| `nccl/header.py` | Self-describing payload header stamped on every transfer, and the receiver's check of it. |
| `utils/trajectory.py` | Flat trajectory schema (`TensorSpec`) shared by sender + receiver. |
| `nccl/mixins.py` | `NCCLRolloutMixin` — producer. |
| `nccl/strategy.py` | `NCCLTransportStrategy` — the consumer's rendezvous + recv engine. |
| `nccl/data_packer_mixin.py` | `NCCLDataPackerMixin` — trainer-side consumer (subclass of `PrefetchDataPackerMixin`). |

## Per-transfer flow

1. The producer packs the payload, records its compute-stream ready event and
   registers the backing buffer.
2. The receiver creates a unique operation identity and deadline. Redis
   compare-and-set seals acceptance against cancellation and duplicate delivery.
3. Acceptance retains the actual UID value and producer payload lease through
   queueing, communicator initialization, enqueue and device completion.
4. Accepted sends leave a per-pair FIFO. The producer waits for the ready event,
   enqueues a standalone send and proves device completion before releasing its
   lease/pin. The receiver proves completion and checks the header before decode.
5. Explicit pre-acceptance rejection can skip unavailable data. Accepted failure,
   late acceptance, corrupt identity or uncertain native completion is terminal;
   it cannot become an ordinary missing episode or a new send attempt.

`TensorSpec` describes the flat schema; the same layout is used to pack
(producer) and unpack (consumer).

## Payload framing and per-pair ordering

A payload on the wire is a header followed by the schema region:

```
[ 32-byte header ][ schema entry_size bytes ]
  magic | version | payload_nbytes | transfer_key
```

`transfer_key` is a 64-bit digest of the `transfer_id`.  The receiver checks
both it and `payload_nbytes` before unpacking. A disagreement after acceptance is
terminal (`nccl/header.py`), even if the receive event has completed: completed
bytes alone do not prove the ordered stream is reusable.

The check exists because a cached 2-rank communicator is an **ordered stream
with no tags**: its k-th `nccl_send` is taken by the k-th `nccl_recv`, and
nothing in the data plane says which transfer a buffer holds.  If the two ends
ever disagree about how many transfers have crossed a pair, every payload after
that point lands in the previous one's buffer.  With one fixed schema that is
invisible; with a per-payload schema the receiver slices a foreign buffer at
its own offsets and returns decoded garbage.

Two rules keep the two ends in step, and the header catches anything that
still slips through:

- **Sends leave in accept order.** The producer accepts on its single pub/sub
  listener thread and queues each accepted send on a *per-pair* FIFO drained by
  at most one pool task, so pool workers can never launch two of a pair's sends
  out of order.  Pool width still bounds how many *pairs* transfer at once.
- **Accepted work is not replayed after failure.** Failure to honor an accepted
  send/receive or a late acceptance is terminal. Abort/cache invalidation alone
  does not prove native work stopped touching buffers or that the stream can
  safely restart. Fresh-UID negotiation is a pre-acceptance operation, not
  recovery of an uncertain accepted transfer.

## Rendezvous state machine

Every invocation has its own operation ID. Redis atomic state transitions make
acceptance, cancellation and duplicate/lost-reply handling refer to that same
operation, not an uncorrelated response key.

```
REQUESTED -> ACCEPTED -> COMPLETE / FAILED
          -> MISSING / NEED_UID / CANCELLED  (before acceptance)
```

`ACCEPTED` is not success. Lost replies are resolved against the same operation;
late acceptance cannot be silently discarded and retried. Bounded peer-outcome
observers run independently of blocked native callers, and hard deadlines do not
depend on Redis making progress. See the
[operation contract](../transfer_operation_contract.md).

## Failure and ownership boundaries

- **Bounded failure, not recovery:** enqueue return is not device completion.
  Unknown completion retains buffers, streams and communicator pins until
  terminal process exit. Strategy-backed terminal failure exits with status 86;
  cross-node job failure propagation is not guaranteed. Use a finite job limit.
- **Explicit safe rejection:** MISSING/no-schema outcomes survive prefetch
  without an uncapped synchronous refetch. Unknown absence is not equivalent to
  a known pre-acceptance rejection. See [receive leases](bounded_receive.md).
- **Quarantine is not proof of recovery:** consumers consult endpoint health.
  Producer setup-failure diagnostics and invalidation use the actual
  `(sender_rank, receiver_replica, receiver_rank)` cache key. The producer does
  not use that cooldown as an admission gate; accepted-operation ownership and
  deadlines govern its sends.
- **Registry capacity is not a hard GPU-memory limit:** retired entries with
  pending native readers retain their storage. Uncertain work cannot be freed
  merely to meet an entry-count or memory target.

## Communicator scaling

Comm count is `O(rollout_ranks × trainer_ranks)`; each 2-rank comm costs
tens of MB + a QP. Controls:

- **Consumer-driven pair set** — only pairs that actually transfer get a comm.
- **Live-comm cap + LRU eviction** — bounded live comms; LRU aborted beyond.
- **Bounded concurrent init** — a semaphore caps simultaneous `create_nccl_comm`.
- **Pair-scoped communicators** — identities and FIFO order are per pair, but
  this is not dead-peer failure isolation. Shared streams, GPU resources and
  process-fatal outcomes can affect other pairs.

## Dedicated transfer stream

Payload NCCL runs on its own **low-priority** CUDA stream(s), separate from
both the compute stream and the weight-sync stream. This permits overlap but
does not guarantee independence from shared-device contention. Correctness is
event-based (mirroring `activation_offloading`'s s0/s1 hand-off):

- **Sender** records a ready-event on the compute stream after the
  trajectory tensor is produced; the transfer stream `wait_event`s it
  before `nccl_send`, and the buffer is held until a send-complete event.
- **Receiver** records a recv-complete event after `nccl_recv`; the training
  consumer waits on it before reading.

The default producer pool has one transfer stream shared by pairs. A stalled
native operation can block later work on it; separate host tasks do not remove
that dependency. Independent operation deadlines contain uncertain failure by
terminating the worker, not by promising other pairs continue safely.

## Cleanup semantics

`NcclPayloadTransport.completion_prefix = "nccl:"` stays active. When the
controller discards outdated rollouts whose `completion` is a
`"nccl:<transfer_id>"` string, `PayloadTransportRegistry.handle_discarded`
dispatches to `publish_cleanup_for_discarded`, which publishes on the
producer's `:nccl_cleanup` channel to retire the buffer. Cleanup and eviction are
idempotent bookkeeping operations, not proof of native completion. A buffer is
reclaimable only after its outstanding readers finish; uncertain readers keep
their storage owned until terminal process exit.

## Deprecated: the `redis_client` / `post_redis_injection` packer contract

Before the in-tree mixin, NCCL-aware data packers exposed a
`redis_client` attribute and an optional `post_redis_injection()` hook
(PR #670). `attach_data_packer` still honors that path as a **deprecated
fallback** so in-flight downstream forks keep working, but new code should
subclass `NCCLDataPackerMixin` (which exposes `_setup_nccl_data_packer`,
the path `attach_data_packer` prefers). The legacy path logs a deprecation
warning and will be removed no earlier than two minor releases after the
in-tree transport lands.
