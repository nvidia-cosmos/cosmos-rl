# Trainer payload prefetch

`train.prefetch_payloads = true` opts a compatible policy worker into depth-one
payload lookahead. It is disabled by default. It uses the selected payload
transport and stock command handlers; it is separate from rollout prompt
prefetch and does not change the training objective or optimizer schedule.

## Supported execution

Overlap requires disaggregated GRPO, centralized rollout metadata, pure data
parallelism, NCCL or UCXX payload transport and permitted policy-version lag.
Weight-sync intervals including one are supported. Distillation and strict on-policy
execution use the synchronous path. Other unsupported modes/topologies also
log a synchronous fallback. Enabling the setting without an attached
prefetch-capable packer in a supported mode is an error.

The controller reserves an eligible complete next batch when available,
including after the current command has been dispatched. It publishes an
independent `PayloadPrefetchCommand`; the existing background control reader
delivers it to every rank through the separate TCPStore control channel. It
does not enter training collectives or wait behind the executing training
command. Current training never waits for future production to fill the pipeline.

Staleness is checked against the intended next pre-update step. Input receive
can cross a checkpoint or weight-publication boundary: it neither changes the
model nor counts as completed training. The ordinary command sequence still
publishes weights N before optimizer update N+1. A checkpoint records only
committed progress; speculative input is not serialized into it. No new
reservation is made during active validation, after requested stop, or beyond
the final horizon. Already-admitted work retains its ordinary owner.

## Progress and completion

A train ACK still means that the corresponding command completed. Reserving B
while training A does not advance the optimizer, scheduler, training step or
completed-sample accounting. B is consumed by its own ordinary command, with
its original rollout metadata and an explicit reservation identity. Admission
does not repurpose A's step number for B or require an early ACK.

Warmup fetches and trains the first batch once. Exhaustion and requested stop
drain admitted work using ordinary training commands. Unissued incomplete
inputs are not fabricated into updates. Terminal completion always checks for
an empty pipeline, including when checkpoint saving is disabled; it refuses
to ACK pending or failed work. Teardown never performs training.

The final `DataFetchCommand` closes notification admission and checks for
pending work before sending its success ACK. Managed received batches retain
their storage until the worker's bounded final-reader drain succeeds; a
reader timeout cannot release the storage or send a success ACK. Returned
metrics must not alias received storage. Standalone training releases a
successfully consumed received batch even when no receive-byte budget is set.

A checkpoint records completed training, not the reserved future batch.
Resume starts with an empty pipeline; uncommitted work may be fetched or
generated again. This is checkpoint continuation, not exact asynchronous
replay or durable controller-queue recovery.

## Notification delivery

Notifications contain original rollout metadata, a stable batch identity,
intended step and fixed process incarnations. Each pure-DP rank selects its
original shard locally. The controller removes the batch into an explicit
reservation; it cannot be independently redistributed or removed again.
Publication uses the existing immutable Redis receipt protocol. A partial or
uncertain multi-replica publication is terminal, not permission to issue a
replacement reservation.

The ordinary training command carries the same metadata as a fallback: missed
lookahead may lose overlap, never samples. Exact duplicate notifications are
idempotent; changed content, unknown stale identities and incarnation mismatch
fail closed. The receiver retains only one future payload slot and a bounded
duplicate-receipt window. A reader ahead of the training loop waits with a
finite deadline for the preceding slot to be consumed. Only the training
thread installs a result in the current cache.

Background control delivery uses bounded I/O and recipient agreement separate
from training collectives. It does not retry uncertain delivery under another
sequence number. Notification errors are surfaced at training readiness or
completion agreement; no success ACK is sent after a known failure. Admission
is closed and the owned reader is stopped before transport teardown. Controller
restart or changed membership is not elastic recovery; resume starts a fresh
empty pipeline from the committed checkpoint.

## Custom trainer contract

Keep the ordinary training method: train the rollout batch it receives and
return after enqueueing its final payload readers. A synchronous
`start_prefetch(rollouts)` / `wait_prefetch()` pair may remain; inside managed
training it reuses the worker's current cache and cannot consume the future
batch by mistake. Use the same rollout objects passed by the worker.

Remove downstream command/deferred-buffer orchestration before enabling the
managed lifecycle. `collect_prefetch()` and `defer_prefetch()` deliberately
reject use inside managed training. Do not independently submit another
prepared/payload future: there is one shared outstanding slot. CPU sample
expansion remains the trainer's existing batching contract; this setting does
not automatically move arbitrary trainer preparation into a background thread.

The current and future caches have separate owners. The worker owns a dedicated
CUDA fetch stream for next-batch allocation, copy and decode. A background
Python thread alone does not select a different CUDA stream. The fetch future
becomes ready only after that stream finishes, under the existing independent
prefetch watchdog; training never reads a partially decoded payload.
A rejected reference is
not synchronously re-fetched during managed training. Existing trainer
empty-data/batching rules still determine the training contribution; this
feature does not invent zero-loss samples or convert a fixed-batch trainer into
an expanded/weighted trainer.

Before returning, drop aliases that must not outlive the command. Override
`training_payload_streams()` to declare additional CUDA readers beyond the
worker training/current streams. The worker fences final readers and releases
the current lease before the next command consumes its future. Insufficient
receive-memory credit can serialize the next fetch rather than increase the
configured memory bound. Producer capacity/backpressure remains transport-owned.
Final-reader fencing uses the training collective timeout budget, not the
short process-teardown timeout. Configure the existing collective deadlines
for the largest legitimate asynchronous training step.

Optimizer metrics count actual `step()` calls when the optimizer layout is
recognized. Override `payload_prefetch_optimizers()` for a custom layout.
Unrecognized/replaced optimizers report the counter as unavailable, not zero.

## Failures and observability

Readiness checks cover metadata, identity and fetch failures before training.
All ranks must follow the same collective schedule. Mesh membership is held
stable for each operation and cannot change under an admitted reservation.
Uncertain native completion remains terminal under the existing transport
deadline/watchdog contract; this feature does not recover a failed transport,
partially completed optimizer update or lost policy participant. A failed
command sends no successful train ACK. It does not add launcher-level failure
propagation; use finite job time limits.

Reports include `prefetch/trained_step`, `prefetch/delivered_step`,
`prefetch/outstanding_batches`, `prefetch/fetch_latency_s`,
`prefetch/exposed_receive_wait_s`, lookahead hit/miss counters and per-optimizer
step counters. Logs include
current and next reservation IDs. A completed command may contain multiple
optimizer calls, or none under an existing agreed skip policy. Do not equate
admissions or command count with useful optimizer work.

NCCL and UCXX periodic and final transport totals account for each successful
background fetch, including managed prepared batches. Payload counts and bytes
describe the native receive results, not the number of distinct cache keys or
optimizer updates. Fetch latency excludes subsequent CPU batch preparation.
The legacy completion hook remains available, but cache promotion does not
count the same transfer again. Failed fetches are not counted as completed
batches; successful receives still count if later preparation fails.

On checkpoint resume, the initial policy-to-rollout transfer carries the
controller's restored weight version, regardless of registration order. The
rollout must adopt that version along with the weights before serving resumed
prompts; a singleton asynchronous rollout cannot rely on a later native R2R
broadcast to correct an unversioned bootstrap transfer.

`tests/trainer_prefetch_payload_canary.py` runs matched OFF/ON/barrier controls
through the stock controller, background readers and worker with real NCCL or
UCXX payloads and SGD. The next batch is released only after current training
starts. A separate native NCCL channel publishes GPU snapshots at
`--sync-interval` boundaries, checking that snapshot N excludes update N+1.
This synthetic weight channel is not production P2R/VLA qualification. It emits
fetch/compute intervals, useful update counts, end-to-end throughput and CUDA
memory samples, including warmup and final consumption. Its producer and
consumer are fixture ranks, not a training mesh. The separate four-rank
`tests/trainer_prefetch_cohort_canary.py` exercises two policy replicas. Run
both controls; a positive overlap span alone is not proof of a throughput win.

For device-level evidence, run the payload canary with `--trace-dir DIRECTORY`
and inspect transfer/copy activity against the annotated OFF/ON/barrier training
kernels. Profiling is optional and changes timing; keep throughput controls
unprofiled. For example, use `--steps 32 --compute 256` with the trace option.
