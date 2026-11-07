# Execution workflows

Local implementation of the design discussed on 6 November 2026. This branch
is intentionally unpushed while the design is under review.

## Execution paths

`[execution].path` selects `train_from_data`, `train_from_generation`, or
`generate`. Existing recipes retain their defaults. New execution options are
strictly validated; absent role counts are filled from the selected path,
while explicitly contradictory counts fail.

`train_from_data` uses the native SFT worker. Its training objective can be
supervised, recorded policy gradient, recorded distillation, or a combination.
The first token-objective implementation requires one learner replica with
pure data parallelism, materialized logits and unpacked fixed-schedule batches.
Legacy SFT and OPD paths keep their existing implementations.

## Corpus format

`execution.source.kind = "corpus"` reads immutable committed manifests from
`execution.source.path`. Tensors are owned content-addressed artifacts with
checksums. Corpus revisions cover the manifest snapshot; an explicitly pinned
revision rejects a changed dataset. Temporary writes are invisible to readers.

The corpus keeps generation failures and quality annotations. The supervised
dataset view excludes generation failures, retains low-quality generated
targets, and calls the existing packer for text targets. Application packers
can implement `prepare_record(record)` for observations, actions and other
representations. Preprocessed target mappings contain the packer's normal
fields, including `label_ids` for token training. Original model identity,
behavior probabilities and group identities remain attached to each record.

The exact distributed corpus sampler uses fully masked occurrences to align
the schedule rather than duplicating contributing examples. A transactional
`CorpusSource` separately supports group-preserving reservation and source
cursor commit at an optimizer boundary.

## Generation output

`[execution.output]` specifies `path`, a stable logical `run_id`, and an optional
`max_record_bytes`. A required synchronous consumer persists outputs before
native report settlement. A record ID combines run ID, controller-issued input
occurrence and original completion slot. Concurrent publication is atomic and
the first committed output wins. A new controller attempt does not change the
logical output identity. Applications can still register their own required
consumer instead of configuring the built-in output sink.

## Local implementation progress

- Integrated the latest local heads of 783, 784 and 785 and reconciled reward
  cleanup with generation result delivery. Existing PR branches are untouched.
- Added execution configuration and role projection, corpus identities,
  owned artifact storage, atomic publication, bounded group input, native SFT
  corpus loading and checkpoint source compatibility.
- Added opt-in token loss composition with separate masks and one distributed
  denominator per objective over the optimizer window.
- Actor checkpoint configuration and synchronization guards are being wired
  and validated. Production resume, model compatibility, native trainer
  qualification and Slurm coverage are still in progress.

Validation completed during implementation: 65 execution/rollout controller
regressions passed; 26 execution-plan, corpus and objective regressions passed.
These are interim results, not qualification of the final branch.

## Remaining qualification

Exercise the actual SFT training loop and resume metadata; test distributed
gradient equivalence, empty rank schedules, generation restart and source
reconciliation; complete independent actor startup and provenance checks;
bound teacher traffic; add runnable examples and wire new scenarios into the
existing full Slurm CI. Conditional extensions such as EMA teachers and arbitrary
streaming topologies remain outside the first supported slice of the design.
