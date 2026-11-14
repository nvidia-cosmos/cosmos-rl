# Sample- and episode-weighted objectives

An optimizer accumulation window has one denominator. Averaging independently
normalized microbatch losses is incorrect when their contributing counts differ.
For scalar sample losses `l`, use either:

- **Sample:** sum of retained sample losses / global retained sample count.
- **Episode:** sum of each nonempty episode's mean sample loss / global nonempty
  episode count.

`ObjectiveWindow` prepares CPU weights after filtering and scales loss slices
against a shared global count. It does not synchronize, change model collective
participation, clip gradients, or step optimizers. Averaged gradient reductions
require multiplication by their participant count before backward; sum reductions
do not. All-empty optimizer windows skip both optimizer and scheduler, including
momentum and weight-decay updates. A locally empty rank must still participate in
the model's forward/backward schedule when another rank has data.

## Expanded trainers

See [trainer_batching.md](trainer_batching.md) for the opt-in objective windows,
episode identity contract, count agreement and prefetch integration. This path
supports uneven/empty local batches without duplicate samples. It is layered on
the expanded batching contract and inherits its pure-DP topology restrictions.
Counts, schedules and skip decisions include every policy replica participating
in the supplied gradient communicator, not just the local DP group. The default
loss multiplier compensates both averaging stages. This does not depend on the
payload transfer backend and does not support changing the sealed gradient cohort.

## OpenVLA and PI05 GRPO

Leave `[vla].objective_weighting` unset to preserve the trainer's existing
normalization. OpenVLA retains its within-episode valid-component mean, including
the existing weight of a partially valid final action chunk. PI05 also retains
its existing microbatch normalization by default. The legacy path adds no count
exchange or extra collation pass and keeps its loss reporting.

Explicitly set `[vla].objective_weighting` to `"episode"` or `"sample"` to opt in.
A sample is one action chunk's scalar mean over its valid action components.
Padding contributes neither a sample nor denominator mass. Episode weighting
then averages these chunk objectives within each nonempty episode.

The opt-in path corrects PI05's sum of independently normalized microbatch means,
whose gradient magnitude depended on `training_chunk_size`. Both trainers use one
objective window for the whole update. It also removes OpenVLA's division by zero
for empty episodes and the second division of reported loss by episode count.
Compared with the old OpenVLA element-weighted episode mean, a partially valid
final chunk now has the same sample weight as another valid chunk. This is an
explicit opt-in action-chunk definition, not token weighting. It is not applied
to existing configurations automatically.

The opted-in fixed-rollout VLA loops have no expanded preflight to reuse. They exchange a
small count/shape report once per update within the DP mesh, then sum counts over
the existing inter-replica communicator. This also aligns PI05's padded chunk
count across DP ranks before FSDP forward/backward. The gradient scaling compensates
both averaging stages. No count exchange is added per microbatch. Loss reporting
remains a rank-local scaled contribution, not a newly reduced global metric.

Existing fixed-rollout dispatch must provide matching episode slots within each
DP replica. A zero-contribution rank uses fully masked episode slots, not missing
slots; slot disagreement is detected before backward. This is not an automatic
conversion of VLA to variable-slot expanded scheduling. Built-in packers obtain
counts in a streaming CPU mask-only pass, using the same mask construction as
training collation. Images, observations and actions are not collated for counting;
each trained episode is fully collated only once. No batch of padded episodes is
retained. Custom packers without `policy_logprob_masks` keep the original collation
fallback; a custom mask helper must exactly match that packer's training masks.
Mask values/dtypes, padding and training losses are unchanged. Checkpoint behavior
is kept even when an all-empty update skips optimization.

The opt-in VLA path supports pure data parallelism; TP/PP/CP/EP layouts are
rejected before count collectives until their reduction semantics are validated.
Legacy one-dimensional custom PI05 masks, including empty masks, remain accepted.
The native gradient mesh is pinned from count agreement through optimizer
completion. This adds no count exchange or mesh lock to the default VLA path.

The shared gradient reducer now keeps stable trainable-parameter buckets even
when local gradients are missing. Missing contributions are zero, while globally
unused parameters keep `grad=None` and their optimizer skip behavior. No default
OpenVLA or PI05 loss/within-episode formula is changed.

## Validation

`tests/test_objective_weighting.py` compares real two-rank CPU/DDP updates against
an independent single-process reference, including momentum, weight decay,
scheduler state, repeated mu iterations, uneven episodes, empty ranks, all-empty
windows, filtering, fixed/dynamic schedules and prefetch on/off.
`tests/test_objective_cohort.py` uses four real Gloo processes arranged as two DP
ranks in each of two policy replicas. It compares the combined gradient and
optimizer state with a global reference, including an entirely empty replica,
recoverable preparation failures, successive updates, and configuration/step
disagreements before forward/backward. Its adapter exercises the gradient
communicator's reductions; it does not validate native NCCL recovery.
`tests/test_vla_objective_weighting.py` executes the actual OpenVLA/PI05 loops with
CPU model and clock substitutes to check both objectives at several chunk sizes
and all-empty updates. Compatibility tests verify the default legacy formulas,
partially valid final chunks, and absence of extra collation/count collectives
when the new objective is unset. These are numerical regressions, not full simulator/GPU
lifecycle validation.
Mask-only preparation tests compare built-in training masks against independent
legacy formulas, including partial final chunks and PI05 finish-step clamping.
Loop tests assert one full collation per trained episode for both weighting modes
and the default path, and no count preparation on the default path.
