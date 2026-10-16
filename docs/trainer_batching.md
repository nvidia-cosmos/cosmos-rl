# Expanded samples and matching collective participation

Ordinary trainers retain `FixedRolloutBatching` and their minibatch rules.
Custom GRPO trainers opt in with
`batching_contract = ExpandedSampleBatching(partial_tail="include")`.
Collection counts refer to completions/episodes; `mini_batch` refers to expanded
training samples. Their counts need not divide each other.

Collection still must shard evenly across the full data-parallel mesh, including
replicated DP: `train_batch_per_replica % dp_world_size == 0`. Startup checks
retain that dispatch constraint; all RL dispatch paths also reject uneven command
counts before dequeuing, rather than silently rounding down. For example six
episodes over two ranks is valid even with sample minibatches of size two;
three episodes over two ranks is not yet supported by collection dispatch.

Fixed-rollout GRPO startup checks include both sharded and replicated DP, so
`train_batch_per_replica` must be divisible by `dp_world_size * mini_batch`. A runtime
command whose collection count is not divisible by the full DP size is rejected
on every rank before consuming queues or entering scatter. The worker never
rounds a command down and leaves its remainder behind. SFT's per-DP-worker batch
interpretation and valid OpenVLA/PI05 loss/minibatch behavior are unchanged.
These checks reject unsupported collection counts; they do not turn empty or
failed expanded preparation into a fatal error or add token weighting.

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

Dynamic expanded updates gather metadata within each replica, then reconcile it
across the existing inter-policy gradient communicator when replicas share
gradients. This applies with or without objective weighting: a MAX header agrees
width/errors/update identity and a SUM carries slot counts (plus objective counts
when requested). Neither is a per-minibatch barrier. Independent trainers without
a shared gradient communicator do not synchronize with one another.
The variable data determines participation; fixed-rollout trainers retain their
own existing schedule rather than entering expanded preflight.

The schedule has enough slots for the longest local plan. Missing slots become
empty contributions, not duplicated samples. Slots empty on every rank are
removed consistently for the legacy per-slot interface. Objective-weighted
windows retain their slot boundaries and skip globally empty optimizer windows.
Configuration disagreement about `mu_iterations` remains
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
For accumulation or episode weighting, use the objective-window interface below.
Sum-reduction trainers must not apply an averaging-reduction compensation factor.
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

Without objective weighting, once the initial agreement is sealed, every update
executes exactly four slots for each configured mu iteration. There are
**no per-update schedule exchanges**.
The upfront agreement covers all coupled replicas, not just each replica's DP
group. Native membership is pinned for each entire update. Configuration or
cohort changes require a new sealed schedule; they are not renegotiated locally.
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

Without objective weighting, `global_sample_counts` is `None` in fixed mode, and `mean_gradient_scale()` rejects
use rather than inventing actual sample counts. A trainer can use an explicit
fixed nominal denominator (changing the objective when samples are missing), or
obtain true counts through its own existing collective protocol. Use dynamic
mode for automatic valid-sample counts and globally empty update skipping.

## Sample- and episode-weighted optimizer windows

Opt in explicitly:

```python
batching_contract = ExpandedSampleBatching(
    partial_tail="include",
    objective_weighting="episode",  # or "sample"
    accumulation_steps=4,
)
```

Preparation supplies `ExpandedTrainingBatch(minibatches, episode_ids=ids)`.
`ids` mirrors the minibatch/sample structure and contains integer or string
episode identities local to this rank. Filtering removes the corresponding
identity too. Sample weighting does not require identities. For episode weighting,
an episode must be wholly owned by one rank and fit within one optimizer window;
it may span that window's microbatches. Reusing an identity across optimizer
windows is rejected before training. Splitting one physical episode across ranks
is not supported: rank-local IDs cannot establish shared episode ownership.

The sample objective is the mean of retained scalar sample losses. The episode
objective is the mean of nonempty episodes' means of retained scalar sample
losses. Empty episodes contribute no denominator; filtering renormalizes the
retained data. These definitions intentionally differ for unequal episode lengths.
Token weighting is not introduced.

`batch.objective_windows` describes each optimizer update. For each mu iteration,
consume every window as follows:

1. If `window.global_count == 0`, all ranks skip optimizer and scheduler updates.
2. Otherwise zero gradients once and execute every slot in `window.slots`,
   including empty local slots with the model's required dummy participation.
3. Produce one scalar loss per retained sample in preparation order. Backpropagate
   `window.loss(losses, start=offset)` for each slice.
   Advance `offset` by the local number of samples, not padded dummy rows.
4. Step optimizer and scheduler once after all slots. Do **not** divide by the
   number of accumulation slots again.

The default `gradient_divisor` compensates for averaging across the agreed DP
group and policy-replica gradient cohort. Override it explicitly for sum or mixed
reductions: use one for all-sum reductions, or the product of only the averaging
stages' sizes. It must match the actual gradient reductions, never an unrelated
global world size. This interface rejects the old per-slot `mean_gradient_scale`.

Dynamic mode adds counts to the existing preflight exchange: one integer per
optimizer window, not per-sample identities or tensors. Fixed schedules retain
their configured slot count but opting into exact weighting adds this per-update
count exchange and permits globally empty optimizer windows to skip. Fixed mode
without objective weighting remains exchange-free after startup.

When gradients also average across policy replicas, pass the existing
`inter_policy_nccl` gradient communicator to `run_training_step` (the worker
already does this). A one-time agreement seals the cohort's configuration and
membership. Each update then agrees schedule width, preparation errors and step
with a small MAX reduction, followed by a SUM of slot/window counts. The same
agreement protects unweighted dynamic schedules; unweighted fixed schedules use
only the upfront signature. A zero-width
update omits the latter. All replicas therefore use the same slot schedule,
denominator and empty-window decision; an entirely empty replica participates
when another has data. There are no per-microbatch count collectives. This is
independent of the payload transfer backend. Missing configured multi-replica
communicators and observed membership/communicator changes are rejected; elastic
cohort changes are not supported by the sealed schedule.

The native HA mesh is held through count agreement, all training buckets and
optimizer completion using a local reentrant lock, not another collective.
Custom communicator adapters must provide an equivalent stable group lifetime.
An uncertain native failure is terminal for that communicator and is not replayed
with partially modified buffers. See [training_collectives.md](training_collectives.md).

CPU preparation, identity filtering, and normalization semantics are the same
with synchronous preparation and background payload prefetch. Count agreement,
backward, clipping and optimizer updates stay on the training thread. Custom
trainers must consume this interface; merely declaring an objective does not
rewrite their loss loops. The fixed-rollout LLM trainers retain existing loss
semantics. OpenVLA and PI05 integration is described in [objectives.md](objectives.md).

## Background preparation with payload prefetch

Expanded trainers may call `trainer.prefetch_training_batch(next_rollouts)` on
the training thread when the next owned batch becomes available. When the attached
data packer has an active `PrefetchDataPackerMixin` prefetcher, this replaces
the plain `start_prefetch` submission for that batch: the existing background
worker fetches its payloads, calls `prepare_training_batch`, validates CPU output,
and filters samples before publishing a prepared result.

The same caller works with prefetch disabled or with a packer that has no
prefetch capability: `prefetch_training_batch` returns `False` without doing
preparation or allocating a pending batch. The normal training entrypoint then
prepares/filters synchronously. It returns `True` when background submission
succeeds. Both modes use the same preparation, filtering and schedule logic;
callers must not skip training when the return value is `False`.

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
by the currently training batch. Ordinary preparation exceptions surface at
consumption, where the normal dynamic/fixed recovery policy applies.
Both submission paths share the independent watchdog from PR #747. Its deadline
covers queueing, fetch and CPU preparation, even if training never consumes the
result. Completion disarms the watchdog before publishing the result, so delayed
consumption of completed work is safe. Strategy-backed timeouts and terminal
transport errors exit without native cleanup; they are not recoverable data
errors. Strategy-less legacy packers become terminally unusable on timeout.
Packer shutdown cancels queued work but does not disarm running work's watchdog
until the worker exits; it cannot forcibly interrupt a hung Python/native call.

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
