# Checkpoint discovery and ownership

Automatic resume may inspect previous timestamped output directories. Discovery
is read-only: missing completion markers can mean an unfinished save or a
different parallel layout, not permission to delete checkpoint files. Candidates
that cannot be loaded are left intact. An incomplete candidate may be skipped
before selection. Once a committed checkpoint is selected, an artifact/restore
failure is fatal, even in automatic mode: loading may already have changed model,
optimizer, scheduler or RNG state. Trying an older checkpoint or base weights
would not roll those changes back. Missing metadata is a load error, not an empty
contract. The legacy RL controller publishes its selection to workers.

`train.ckpt.max_keep` applies to completed checkpoints owned by the current
run's checkpoint output directory, not every directory visited during automatic
resume. Starting a new timestamped run does not adopt deletion authority over
previous runs. Incomplete saves are also left intact for inspection or explicit
cleanup. This can retain more files than the previous cross-run pruning policy.
Use one writer job per output directory; this does not add a shared-directory
writer lock or change which compatible candidate automatic resume selects.

Retention and best-checkpoint links are published only after all expected saving
ranks' completion markers exist. Async save submission is not a commit. Pending
housekeeping is revisited at save checks, owned-writer joins and finalization;
an incomplete remote rank can leave extra checkpoints and the previous best
link intact until a later check. No new distributed wait or background polling
is introduced. Failed replacements never authorize deleting the last good save.
Best checkpoint and safetensors links use absolute targets, including when the
configured output directory is relative. They resolve independently of the
link's parent directory, preserving best-score discovery and retention protection
after restart. This does not relocate or repair pre-existing broken links.
Best-score metadata and retention compare canonical artifact paths, so a
symlinked output root does not lose best protection on restart. Older metadata
using the alias spelling remains valid if it resolves to the same artifact.

Committed rank artifacts are immutable. Saving the same step again verifies the
model, optimizer, scheduler and shared progress against the existing files
before performing uploads or hooks. A conflict fails without touching the save;
partial rank artifacts from a failed attempt cannot be overwritten in place.
Final promotion reuses the verified files, including their original RNG and
`is_final` metadata. The new `is_final` argument selects final-only uploads and
hooks; it does not rewrite resume state. This costs reads and local tensor
comparisons on repeated saves, not new distributed collectives. DTensor comparison
checks layout and the local shard only. The caller must still provide a coherent
snapshot boundary, and one writer job must own the output directory. This is
not a protocol for replacing a step or recovering an interrupted multi-rank save.

Controller resume metadata excludes the three trainer-owned GRPO reference
fields. The trainer validates their mode, reset step and complete tensor state
before reporting its remaining shared metadata. Unknown application fields are
not filtered, and shared progress/sampling mismatches remain fatal. Reference
tensors are not sent through the controller JSON API. Legacy pre-reset checkpoints
retain the existing compatibility rules.

Pipeline resume requires matching part/prefix counts and complete, nonoverlapping
stage keys before loading any stage. Readable checkpoints with missing/unowned
keys are errors, not partial weight initialization. Interleaved parts may share
an empty root prefix when their parameter names are disjoint.

Supported layouts still determine completion from their expected saving ranks.
Pipeline parallelism together with replicated data parallelism remains rejected
by `ParallelDims`; this change does not add that topology or reshard checkpoints.

The first colocated training update uses one global remaining-sample snapshot
and one checkpoint decision for all policy replicas, including epoch boundaries.
During rollout shutdown, an engine/scheduler error does not skip bounded heartbeat
cleanup and unregister; the original error propagates after that cleanup.

Portable regressions (also included in `tests/run_test.sh`):

```bash
python -m pytest -q tests/test_checkpoint_discovery.py
python -m pytest -q tests/test_checkpoint_commit_contract.py
python -m pytest -q tests/test_checkpoint_immutable_step.py
python -m pytest -q tests/test_resume_selection_contract.py
python -m pytest -q tests/test_checkpoint.py
python -m pytest -q tests/test_colocated_first_checkpoint.py
python -m pytest -q tests/test_vla_shutdown.py
```

These cover incomplete and foreign-topology file preservation, current-run
retention, best-checkpoint protection, rejected unsupported topology, shared
initial save metadata, and injected backend shutdown failures. They complement
the [snapshot and GPU resume tests](checkpoint_snapshot.md), not a claim of
full multi-topology or simulator validation.
