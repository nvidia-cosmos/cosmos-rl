# Expanded samples and matching collective participation

Ordinary trainers retain `FixedRolloutBatching` and existing startup checks.
Custom GRPO trainers opt in with
`batching_contract = ExpandedSampleBatching(partial_tail="include")`.
Collection counts refer to completions/episodes; `mini_batch` refers to expanded
training samples. Their counts need not divide each other.

Collection still must shard evenly across the full data-parallel mesh, including
replicated DP: `train_batch_per_replica % dp_world_size == 0`. Startup checks
retain that dispatch constraint; expanded dispatch also rejects uneven command
counts before dequeuing, rather than silently rounding down. For example six
episodes over two ranks is valid even with sample minibatches of size two;
three episodes over two ranks is not yet supported by collection dispatch.

Implement two methods:

- `prepare_training_batch(rollouts) -> ExpandedTrainingBatch`: return ordered
  local minibatches without training collectives or optimizer/scheduler changes.
  Convert known bad/unavailable data errors to `RecoverablePreparationError`.
  Prefer excluding individual bad episodes while retaining good ones. An exception
  makes the whole local preparation empty for this update.
- `step_expanded_training(batch, **kwargs)`: execute `batch.mu_iterations`
  passes over every slot in `batch.minibatches`, including empty local slots.
  Each slot must use the same collective schedule on all coupled ranks.

## Schedule agreement

Expanded trainers seal their configuration once, before the first preparation.
They may call `agree_batching_schedule(trainer)` earlier, after the replica's
process group is initialized. The worker otherwise does this lazily on the first
update. The agreement includes scheduling mode, minibatch size, tail policy and
`mu_iterations`. Configuration and process-group changes require a new trainer.

By default, expanded updates use one replica-local metadata exchange after preparation.
It communicates local minibatch sizes and configured `mu_iterations`; it is
not a per-minibatch barrier and does not synchronize independent replicas.
This exchange is needed because variable local data determines participation.
Fixed-rollout trainers do not pay this cost.

The schedule has enough slots for the longest local plan. Missing slots become
empty contributions, not duplicated samples. Slots empty on every rank are
removed consistently. Configuration disagreement about `mu_iterations` remains
an error, not an invitation to silently change the learning algorithm.

Nonfinite values in supported numeric/tensor/list/mapping representations cause
the affected sample to be excluded, not the job to fail. Applications with
intentional missing-value markers must sanitize or encode them during preparation.
Opaque objects require application validation. With `partial_tail="include"`,
remaining partial batches participate; with `"reject"`, undersized local
batches contribute zero. Discarded samples and recoverable preparation failures
are exposed as `batching/*` metrics.

## Empty contributions and weighting

An empty rank must execute the same forward/backward/reduction operations as
its peers. A real DDP/FSDP trainer may need a masked dummy forward/backward;
simply skipping backward or assigning zero gradients is not generally sufficient.
This is a trainer contract, not an automatic conversion of arbitrary trainers.

`batch.global_sample_counts[i]` counts actual contributing samples in slot i.
For an equally weighted sample objective with **averaged** distributed gradients,
multiply the local **sum** loss by
`batch.mean_gradient_scale(i, world_size)` (world size / global sample count).
Token-weighted objectives or sum-reduction trainers must use their own weighting.
All ranks step optimizers/schedulers identically for globally nonempty slots,
including ranks with zero local contribution.

If all ranks are empty, the schedule has zero slots. Scheduler setup is skipped;
the trainer is still called for checkpoint/control work and must not advance the
optimizer or scheduler. The worker may acknowledge a consumed-but-skipped update;
this does not claim an optimization step occurred.

## Optional fixed schedule

Declare, for example:

```python
batching_contract = ExpandedSampleBatching(
    partial_tail="include", fixed_minibatches=4
)
```

Once the initial agreement is sealed, every update executes exactly four slots
for each configured mu iteration. There are **no per-update schedule exchanges**.
Short/empty preparations are padded with empty local contributions, including
recoverable preparation failures. Excess minibatches are a contract error, never
silently truncated: the trainer must bound preparation or manage its own carryover.
Unexpected post-agreement failures use the normal worker/cohort failure path.

This mode intentionally cannot discover globally empty slots or updates without
additional communication. It never removes slots based on local emptiness, and
scheduler setup is invoked even for locally empty data. With a fixed optimizer
schedule, zero gradients can still change parameters via momentum/weight decay.
Choose this mode only when those semantics are acceptable, or implement a
consistent trainer-owned skip protocol using existing collectives.

`global_sample_counts` is `None` in fixed mode, and `mean_gradient_scale()` rejects
use rather than inventing actual sample counts. A trainer can use an explicit
fixed nominal denominator (changing the objective when samples are missing), or
obtain true counts through its own existing collective protocol. Use dynamic
mode for automatic valid-sample counts and globally empty update skipping.

## Background preparation with payload prefetch

Expanded trainers may call `trainer.prefetch_training_batch(next_rollouts)` on
the training thread when the next owned batch becomes available. The attached
data packer must have an active `PrefetchDataPackerMixin` prefetcher. This replaces
the plain `start_prefetch` submission for that batch: the existing background
worker fetches its payloads, calls `prepare_training_batch`, validates CPU output,
and filters samples before publishing a prepared result.

The regular worker's `run_training_step` consumes that result when it receives
the same rollout objects; without a submission it prepares synchronously.
Submit the next batch while consuming/training the current batch to overlap CPU
preparation with GPU work. This is opt-in custom-trainer/prefetch-scheduler
integration, not an automatic change to existing fixed-rollout trainers.
It does not fetch future controller commands, change ACK timing, shift update
numbers, or replay the cold-start batch. The caller must supply the next batch
in the original command order; this cannot create overlap if none is available.

Only one unconsumed prepared batch is permitted. The submitted rollout objects
and fetched storage remain owned until preparation/consumption completes; do not
mutate them. Background resolution uses a thread-local cache, not the cache used
by the currently training batch. Exceptions surface at consumption, where the
normal dynamic/fixed recovery policy applies; waiting uses the packer's finite
prefetch timeout. Packer shutdown cancels queued work and joins running work with
its existing bounded timeout; it cannot forcibly interrupt a hung Python/native
preparation function. Such timeouts remain fatal infrastructure errors.

Preparation must be CPU-only, thread-safe and independent of mutable model,
optimizer, scheduler and shared RNG state. Use batch-local RNG if needed. Move
prepared CPU tensors to GPU on the training thread. Configuration agreement,
per-update agreement, scheduler setup and all training collectives stay on that
thread. Quality decisions affecting group advantages still belong before
advantage computation (#752), not in this late sample-preparation stage.

## Boundaries and validation

Initial integration remains pure-data-parallel GRPO, including colocated RL.
TP/CP/PP and SFT require separate participation integration. Configuration errors,
programming errors and CUDA/collective failures are not recoverable data errors.
Preparation must return: schedule agreement cannot recover a hung rank.

Tests cover uneven counts, partial tails, empty ranks, all-empty updates,
nonfinite samples, recoverable preparation failure, and configuration disagreement.
Run `torchrun --standalone --nproc-per-node=2 tests/trainer_batching_canary.py`
for CUDA/NCCL, or add `--cpu` for Gloo. The canary compares parameters, SGD
momentum and scheduler state with explicit global sample updates over two
mu iterations, including zero-contribution ranks and globally empty slots.
Fixed-mode tests additionally execute successive healthy, uneven, empty and
failed-preparation updates with exactly one schedule agreement and verify
numerical parity against an explicit fixed-denominator reference.
It validates the protocol and test trainer, not arbitrary custom trainer loops.
