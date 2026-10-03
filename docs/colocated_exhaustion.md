# Colocated batch preparation and exhaustion

Colocated workers prepare their batches after the dispatcher issues DataFetch.
Dispatch is not permission to optimize: every original policy replica must
report a complete local batch before any replica starts the update.

Within a replica, ranks refill together using the minimum available capacity.
Prompt/generation decisions also agree before entering backend collectives.
Centralized reward reports are gathered at the explicit reporting boundary,
preserving callback-major/rank-major ordering without assuming every rank emits
the same number of callbacks. Filtering and training objectives are unchanged.

When the dataset genuinely ends before a complete batch exists, all replicas
discard that unstarted batch. The controller restores its progress debit,
cancels only that unstarted dispatch, and sends the existing terminal checkpoint
command for the last completed optimizer update. The configured training horizon
is retained. Logs identify dataset exhaustion as the successful early-stop
reason; it must not be confused with reaching the configured target step.

Preparation reports are tied to registered process sessions and the sealed
dispatch step, horizon, and participant set. Exact retries are idempotent;
conflicting reports, membership changes, and uncertain execution are rejected.
No partial optimizer update is rolled back. A missing/dead participant is not
silently removed from the cohort. The finite preparation deadline and existing
job timeout contain failure; this is not elastic recovery.

This adds a control-plane preparation rendezvous per colocated batch and small
rank-local-group readiness exchanges during refill. It does not change the
disaggregated or colocated-separated loop, loss normalization, or native cache
cleanup policy. Deploy controller and workers from the same revision; old
workers cannot bypass preparation by sending a successful training ACK.

Regression entry points:

- `tests/test_colocated_batch_capacity.py`: negative control on the old code.
- `tests/test_colocated_exhaustion.py`: cancellation, progress, retry, identity,
  validation and ordering contracts.
- `tests/colocated_exhaustion_canary.py --backend gloo|nccl --case
  healthy|exhausted|refill|empty [--centralized]`: two replicas with two ranks each,
  real HTTP agreement and gradient collectives; generation/training are narrow
  test doubles, not a model-quality or end-to-end simulator qualification.
- `tests/test_colocated.py`: real colocated training, final-step rejection/refill,
  and genuine exhaustion after two of three configured updates. The exhaustion
  control checks both replicas stop at update two and verifies the saved model,
  optimizer and committed checkpoint metadata retain the configured horizon.
