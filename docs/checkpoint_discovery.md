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
