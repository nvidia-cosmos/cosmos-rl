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
messages free producer GPU buffers when the controller discards a rollout.

Enable it with:

```toml
[custom]
payload_transfer = "nccl"     # one of "redis", "nccl", "ucxx"
# optional tunables (defaults shown):
nccl_prefetch_timeout = 30.0  # per-batch prefetch wait ceiling (s)
nccl_read_max_attempts = 2    # attempts per transfer (initial + retries)
nccl_recv_timeout = 5.0       # per-recv / per-rendezvous budget (s)
```

The legacy boolean `[custom].nccl_payload_transfer = true` still resolves
to `"nccl"` (deprecated alias).

## Components

| Module | Role |
| --- | --- |
| `nccl/protocol.py` | Redis key / channel builders, transfer-id parsing (pure strings). |
| `nccl/transport.py` | `NcclPayloadTransport` backend: registration, `attach_data_packer`, controller-side discard cleanup. |
| `nccl/rendezvous.py` | Per-transfer request/ack handshake + per-pair unique-ID exchange over Redis. |
| `nccl/comm_cache.py` | Lazy 2-rank communicator cache: LRU cap, bounded concurrent init, health quarantine. |
| `nccl/buffer_registry.py` | Producer GPU send-buffer registry with bounded backpressure + idempotent free. |
| `nccl/streams.py` | Per-process transfer-stream pool + CUDA event helpers. |
| `nccl/schema.py` | Flat trajectory schema (reuses `TensorSpec`) shared by sender + receiver. |
| `nccl/mixins.py` | `NCCLRolloutMixin` — producer. |
| `nccl/data_packer_mixin.py` | `NCCLDataPackerMixin` — trainer-side consumer (subclass of `PrefetchDataPackerMixin`). |

## Per-transfer flow

```
Rollout worker (producer)                 Policy trainer (consumer)
NCCLRolloutMixin                          NCCLDataPackerMixin (PrefetchDataPackerMixin)
  write_to_buffer(traj):                    prefetch a batch of refs:
    pack -> fixed-schema GPU buffer           for each ref:
    record ready-event (compute stream)         if pair comm not cached: mint uid, SET pair_uid_key
    register(transfer_id, buf)                  DEL resp_key; PUBLISH :nccl_req {transfer_id,...}
    return {"_nccl": True, ...,                 poll resp_key up to nccl_recv_timeout:
            "completion": "nccl:<id>"}
  serve loop (bounded sender pool):
    on :nccl_req:                               ACCEPTED -> get_or_create pair comm; nccl_recv
      entry = registry.get(transfer_id)         MISSING  -> drop episode (buffer recycled)
      if missing: SET resp_key=MISSING          timeout  -> CANCELLED: retry, then quarantine
      else: SET resp_key=ACCEPTED
        transfer_stream.wait_event(ready)     grouped recvs issued in one ncclGroupStart/End
        nccl_send(buf) on transfer stream     record recv-complete event -> gates training read
  cleanup subscriber:
    on :nccl_cleanup {transfer_id}:
      registry.free(transfer_id)
```

`TensorSpec` describes the flat schema; the same layout is used to pack
(producer) and unpack (consumer).

## Rendezvous state machine

Each `transfer_id` has exactly **one** intended receiver, so there is no
multi-winner election — a plain request/ack with bilateral timeouts
replaces UCXX's Lua-CAS rendezvous.

```
                 receiver publishes :nccl_req
   (implicit) ─────────────────────────────────► REQUESTED
                                                    │
                 sender: buffer present ────────────┼──► ACCEPTED  (both build comm, send/recv)
                 sender: buffer recycled ───────────┼──► MISSING   (receiver drops episode)
                 receiver: no reply in timeout ─────┴──► CANCELLED (retry ≤ max_attempts, then quarantine)
```

All three terminal states are idempotent: the receiver clears `resp_key`
before each attempt and consumes it on read, so a late/duplicate reply
from an abandoned attempt is discarded.

## Robustness (TODO-8 wedge-fixes)

- **Finite recv timeout** — every `nccl_recv` and rendezvous poll is bounded
  by `nccl_recv_timeout`, so a wedged sender engages retry/skip quickly.
- **Health-aware quarantine** — a transient failure quarantines the
  `(rollout_idx, sender_rank)` endpoint with a cooldown (`comm_cache`);
  the episode is dropped/retried next round instead of wedging the batch.
- **Layered retry ceiling** — per-ref fresh-call retry (`nccl_read_max_attempts`)
  × the base `PrefetchDataPackerMixin` per-batch multi-round, so a wedged
  transfer is dropped via `_on_resolve_failed` rather than hanging.
- **Defensive state** — buffer-registry register/free and the rendezvous
  three-state are idempotent; a stale cleanup or duplicate request cannot
  double-free or orphan a buffer.
- **Producer backpressure** — the send-buffer registry has bounded
  capacity: when full it blocks, then evicts the oldest un-sent buffer, so
  rollout-generation prefetch cannot outrun demand-driven sends and OOM the
  rollout GPU.

## Communicator scaling

Comm count is `O(rollout_ranks × trainer_ranks)`; each 2-rank comm costs
tens of MB + a QP. Controls:

- **Consumer-driven pair set** — only pairs that actually transfer get a comm.
- **Live-comm cap + LRU eviction** — bounded live comms; LRU aborted beyond.
- **Bounded concurrent init** — a semaphore caps simultaneous `create_nccl_comm`.
- **2-rank isolation** — a dead replica only aborts its own pair comms, never
  a global mesh.

## Dedicated transfer stream

Payload NCCL runs on its own **low-priority** CUDA stream(s), separate from
both the compute stream and the weight-sync stream, so payload kernels
overlap compute and never false-serialize against either. Correctness is
event-based (mirroring `activation_offloading`'s s0/s1 hand-off):

- **Sender** records a ready-event on the compute stream after the
  trajectory tensor is produced; the transfer stream `wait_event`s it
  before `nccl_send`, and the buffer is held until a send-complete event.
- **Receiver** records a recv-complete event after `nccl_recv`; the training
  consumer waits on it before reading.

Streams give GPU-side overlap only; shared-NIC bandwidth arbitration vs
weight-sync is a measure-and-tune concern, with the finite recv timeout +
backpressure as the safety valve.

## Cleanup semantics

`NcclPayloadTransport.completion_prefix = "nccl:"` stays active. When the
controller discards outdated rollouts whose `completion` is a
`"nccl:<transfer_id>"` string, `PayloadTransportRegistry.handle_discarded`
dispatches to `publish_cleanup_for_discarded`, which publishes on the
producer's `:nccl_cleanup` channel so the rollout worker frees the GPU
buffer immediately. (Even without a controller cleanup — e.g. when the
completion is carried as dict metadata rather than the string form — the
producer registry's bounded capacity + oldest-first eviction guarantees
buffers are reclaimed.)

## Deprecated: the `redis_client` / `post_redis_injection` packer contract

Before the in-tree mixin, NCCL-aware data packers exposed a
`redis_client` attribute and an optional `post_redis_injection()` hook
(PR #670). `attach_data_packer` still honors that path as a **deprecated
fallback** so in-flight downstream forks keep working, but new code should
subclass `NCCLDataPackerMixin` (which exposes `_setup_nccl_data_packer`,
the path `attach_data_packer` prefers). The legacy path logs a deprecation
warning and will be removed no earlier than two minor releases after the
in-tree transport lands.
