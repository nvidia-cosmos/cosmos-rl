# Trainer sample and target alignment

GRPO optimization chunks partition one collected batch without changing sample
identity. Static and token-balanced partitioners return chunk-relative indices;
the trainer converts these once to full-batch indices before caching the
arrangement. Advantages, teacher/behavior log probabilities, completion metadata
and positive-NLL flags consequently follow the same sample in every phase and
mu iteration. Disabling optimization chunking keeps the existing path.

DPO packers mark response **target** positions, as does the optional chosen
response SFT term. The response log-probability sum shifts that mask left once
before the shared next-token scoring utility, and normalizes it to boolean
indexing. The last logit has no following target and must not contribute. The
input batch/mask is not mutated. This corrects target alignment, not the choice
of DPO/BCO algorithm or reference-model semantics.

`tests/test_trainer_sample_alignment.py` calls the actual trainer entrypoints
with a tiny model and distinct sample metadata. It covers static/reordered
chunks, cached phases, teacher and rollout log probabilities, positive-NLL flags,
two mu iterations, and the no-chunk control. DPO values and logits gradients are
compared with a naive next-token gather, including padding and an empty response.

Run on CPU with `python -m pytest -q tests/test_trainer_sample_alignment.py`.
For GPU numerical validation, set `COSMOS_ALIGNMENT_DEVICE=cuda`; CUDA absence
then fails instead of skipping. Export the checkout root in `PYTHONPATH` when
using a Python environment editable-installed to another checkout.

No OpenVLA/PI05 trainer, objective formula, collective schedule, token-weighted
feature, or checkpoint/transport interface changes are included. Sequence-packing
boundaries and other numerical audit findings remain separate follow-up work.
