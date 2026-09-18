# Prefetch deadline fault injection

`tests/prefetch_watchdog_canary.py` publishes the injection functions used by
the distributed canary. The optional command-line wrapper runs an existing
application role entrypoint with those functions installed:

```bash
python tests/prefetch_watchdog_canary.py --clean APP_ENTRY.py APP_ARGUMENTS
PREFETCH_HANG_AFTER_BATCH=2 PREFETCH_HANG_TIMEOUT_S=15 \
  python tests/prefetch_watchdog_canary.py APP_ENTRY.py APP_ARGUMENTS
```

Use an isolated, finite NCCL-payload workload with its normal launcher and
controller/rollout/policy topology. If a launcher requires a single role script,
use a test-only entry shim that calls `patch_strategy` and `patch_scheduler`
before running the normal application entrypoint. The wrapper is not a Slurm
submission script and does not configure the application or allocate GPUs.

After successful transfers, a run-scoped Redis election selects one consumer.
The fixture parks that consumer after real NCCL rendezvous while its receive
lock is held. Its prefetch deadline is shortened; peer deadlines stay at 300
seconds so any earlier peer termination can be distinguished from another
watchdog expiration. A threading wait releases the GIL. This models
a blocked native call; it does not reproduce a CUDA driver deadlock or make
concurrent cache cleanup safe.

Require the clean control to complete its training horizon and release the
allocation. For the fault arm, require exactly one elected receiver, a watchdog
fatal exit (86 through the Cosmos torchrun wrapper) and no `UNSAFE FALLBACK`
marker. Local CLI checks additionally require termination of its launched peers.
For cross-node checks, use a finite Slurm time limit and record worker failure
separately from allocation outcome. Peers may remain blocked until that limit;
this PR does not add cross-node propagation. Disable relaunch/autoresume in the
test configuration; the PR does not change those scheduler policies. A scheduler
timeout is not evidence of prompt cohort termination. See
`transport_failure_contract.md` for the precise containment boundary.

The injection functions were cluster-tested against the earlier NCCL-only
watchdog. The revised generic fatal-status and local containment path still needs
live validation; those older results do not establish its correctness. The
standalone wrapper is an additional convenience interface. Never install this
fixture into production startup paths.
