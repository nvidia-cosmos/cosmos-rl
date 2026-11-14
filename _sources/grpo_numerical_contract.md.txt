# GRPO sample and numerical contracts

Optimizer chunks keep indices into the original rollout batch. Advantages,
teacher/behavior logprobs, completions and positive-example flags use that same
identity. Cached reference/old logprobs allocate slots for the actual minibatches,
including uneven chunks and token-budget partitions. A fully skipped teacher
chunk still reaches its existing reduction boundary; this does not add a new
collective or a new missing-teacher training objective.

Sequence packing changes physical layout, not the identities used by existing
sequence-mean losses, GSPO ratios or off-policy masks. Masked-token cumulative
boundaries are retained separately from attention/token boundaries. Every
selected logit retains its next-token target, including EOS when EOS equals
padding. Empty responses contribute zero rather than an invalid index or a NaN;
the existing sequence-count denominator is retained. Packing remains disabled
for pipeline parallelism.

A positive `entropy_coeff` now differentiates the effective-token entropy.
The calculation is FP32 and recomputes chunks during backward to avoid retaining
another full token-by-vocabulary probability matrix. This costs recomputation
only when entropy regularization is enabled. Logging metrics are detached;
coefficient zero preserves the non-regularized objective.

These fixes do not add token-weighted accumulation or change default OpenVLA/PI05
losses. DPO target-position masks are shifted to logit positions; reference-policy
semantics retain the default reference-free mode, with the separately documented
opt-in reference mode. Pipeline normalization and cache ownership are covered in
`pipeline_objective_normalization.md`; reference-reset persistence remains a
separate checkpoint contract.

Portable tests (select `COSMOS_ALIGNMENT_DEVICE=cuda:0` for the GPU reference gate):

```bash
python -m pytest -q tests/test_grpo_numerical_contract.py
python -m pytest -q tests/test_trainer_sample_alignment.py
```

The tests drive actual trainer entrypoints with tiny models and compare targets,
losses, gradients and optimizer updates. They cover packed/unpacked, full/indexed
logits, unequal/empty responses, GRPO/GSPO, off-policy masks, positive NLL,
reference/old-logprob phases, teacher data, two mu iterations and uneven/dynamic
minibatches. They are not a full-model, arbitrary-topology training benchmark.
