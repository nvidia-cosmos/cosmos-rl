# Rollout collection and expanded sample batching

Ordinary trainers retain `FixedRolloutBatching` and existing startup divisibility
checks. For a custom GRPO trainer that expands completions, declare
`batching_contract = ExpandedSampleBatching(partial_tail="include")` (or
`"reject"`) and implement both:

- `prepare_training_batch(rollouts) -> ExpandedTrainingBatch`: expand without
  training collectives or optimizer/scheduler changes. Return actual, ordered
  minibatches of samples, not lazy handles. Expansion errors are exchanged with
  every rank before any rank enters training.
- `step_expanded_training(batch, **kwargs)`: consume that validated batch and
  implement the application's algorithm, gradient weighting, optimizer and
  scheduler semantics. Cosmos does not call legacy `step_training` on this path.

`train_batch_per_replica` remains a **completion collection count**;
`train.train_policy.mini_batch` is the **post-expansion sample minibatch size**.
Only the final sample minibatch may be partial. Cosmos never silently drops or
pads a tail. All ranks must have the same number of minibatches; local tail sizes
may differ, so the trainer must implement appropriate objective weighting.

The worker enforces a collective preflight after expansion. Empty batches on
one or all ranks, nonfinite numeric/tensor inputs, preparation errors and
different minibatch participation counts fail on all ranks before trainer
collectives or scheduler setup. This is an error, not a fabricated training
update or a successful ACK. The application may choose a different input plan
before reporting completions; the worker does not silently skip a rank.

Initial support is pure data parallelism and GRPO only. TP/CP/PP and SFT are
rejected for this contract. All ranks must use the same trainer and configuration.
Preparation must terminate: the preflight does not recover a rank stuck in
native I/O or CUDA. It checks the supplied sample representation, not future
losses or opaque application objects; finite gradients and custom collective
ordering remain trainer obligations.

Tests cover fixed-size behavior, variable episodes, partial tails, missing
completion errors, numerical SGD/momentum parity, and real two-rank Gloo
agreement for empty/nonfinite/unequal plans. `tests/trainer_batching_canary.py`
adds two-rank CUDA/NCCL numerical and failure-path validation.
