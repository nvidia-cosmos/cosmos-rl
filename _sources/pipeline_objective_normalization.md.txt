# Pipeline minibatches preserve existing objectives

SFT's existing non-pipeline loss averages valid target-token losses within each
minibatch, then averages those minibatches for the optimizer step. Pipeline
microbatching must only split that computation; it must not change its weighting.

The last stage counts valid targets for the whole minibatch before invoking the
schedule. Each microbatch contributes its loss sum over that common denominator
and the existing optimizer-step factor. Configured DP-token balancing reduces
the denominator once for the whole minibatch, not separately for each slice.
The existing CP-local-mean fallback is preserved. Empty-target slices contribute
finite zero. This is not a new token-weighted accumulation mode.

The SFT schedule must not divide these gradients again. SFT's multi-stage
builder explicitly disables PyTorch's optional gradient scaling, as does GRPO's
normalized loss adapter. The patched 1F1B schedule detects its
existing scaling override by capability rather than lexicographic version
comparison. GPipe training defaults remain unchanged unless explicitly overridden.
Forward-only validation also leaves existing gradients untouched. Patched
schedules can contain backward actions even without a loss on older PyTorch
versions. The stage skips all such compute, send/receive and scaling actions
before touching backward-only state. Training still delegates those actions.
Patched
schedules restore the stage's backward flag on every step and rebuild shared
stage infrastructure after another schedule has used it; validation's receive
buffers and chunk counts are not valid training-backward state.

For SFT, the DeepSeek builder uses the training minibatch size, not the entire
optimizer batch. SFT validation uses its own configured batch size and no training
loss function; disabled SFT validation creates no unused schedule.

The compatibility wrapper also matches send/receive dtypes in PyTorch's neighbor
initialization handshake; affected releases otherwise send an integer scalar into
a floating-point receive allocation.

`tests/pipeline_objective_canary.py` runs the actual `SFTTrainer.step_training()`
method with tiny real pipeline modules. It compares losses, gradients, parameters
and SGD momentum with the unsplit non-pipeline objective over repeated optimizer
updates and train/validate/train transitions. It exercises equal/unequal target
lengths, multiple minibatch/microbatch counts, both patched schedules and the
single-/multi-stage builder. Run with two-rank `torchrun`, optionally `--cpu`.

## GRPO and GSPO

Each pipeline chunk contributes to the existing whole-minibatch denominator for
the configured sequence-mean or token reduction. Existing DP-token balancing
collects its denominator once per minibatch. Entropy and positive NLL retain
their separate local token denominators. No new token-weighted objective is added.

The loss adapter wraps the materialized final stage, not the unused root/meta
model. Reference snapshots and swaps likewise use the local materialized parts.
Reference and old-policy passes use a forward-only schedule. Cache slots retain
both minibatch and chunk identities; repeated metadata forwards do not append
duplicates, and synthetic training shape-inference forwards cannot populate the
old policy. Ragged rollout logprobs are converted to masked row-aligned tensors
before native input splitting. Reports sum the normalized chunk contributions,
preserving the existing per-minibatch loss/KL and entropy reporting means.

When minibatch dimensions change between completed steps, rebuild stage metadata
and use smaller equal-sized chunks. Do not cache every historical shape, pad
examples or silently drop tail rows. Native schedule restrictions still apply:
in particular, a one-row minibatch cannot meet a two-stage single-stage schedule's
minimum chunk count. This repair does not promise arbitrary dynamic partitions
or agreement between DP peers; those require a separate distributed contract.

`tests/grpo_pipeline_canary.py` drives the actual trainer with native GPipe, 1F1B
and interleaved schedules. It compares repeated gradients, parameters and SGD
momentum with the existing non-PP objective, including reference/old phases,
unequal/empty responses, short supported tails, two mu iterations, multiple
optimizer chunks, rollout-provided old logprobs and behavior correction. The CPU
matrix additionally checks reported loss, KL and entropy. Run with two-rank
`torchrun`, optionally `--cpu`.

This does not change OpenVLA/PI05 formulas or checkpoint persistence. Production
model kernels and combined TP/CP/FSDP topologies need their own integration gates.
