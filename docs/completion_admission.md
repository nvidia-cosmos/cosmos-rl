# Completion selection and accounting

## Quality selection: identical in colocated and disaggregated modes

Applications return `RolloutResult.completion_trainable` and optional aligned
`completion_drop_reasons` from their rollout producer. This existing interface
is independent of deployment mode and uses the selected algorithm's
`minimum_trainable_completions`, not a hard-coded GRPO rule.

The shared worker/reward path performs these steps:

1. Compute reward telemetry (including excluded completions where supported).
2. Resolve the producer's quality mask for each original prompt group.
3. Exclude the entire group if too few eligible completions remain.
4. Compute advantages using **only eligible rewards**.
5. Select all completion-aligned fields consistently and report outcomes.

For example, rejecting reward `100` from `[0, 1, 100]` must normalize `[0, 1]`,
not retain the first two advantages computed from all three rewards. Rejected
members of one prompt must not be replaced with members of another prompt to
satisfy the minimum. Local and remote rewards honor this ordering; bypassed
rewards select the same members with zero rewards/advantages. Validation ignores
training masks and continues evaluating every completion.

There is deliberately **no controller quality callback**: received training
payloads already carry group-derived advantages. Controller extraction rejects
an unconsumed mask instead of silently training it or filtering too late.
Late staleness and terminal-job cleanup remain separate lifecycle decisions.

Colocated workers inherit the same producer and reward implementation. Their
controller queues only selected completions, aggregates numeric telemetry, and
does not sum routing IDs or originating weight versions. Both centralized and
uncentralized colocated ingestion follow this contract; existing queue-based
collection continues until enough accepted samples are available.

## Optional identified reporting

`completion_admission=True` on controller `main` / `Controller.setup` enables
per-completion replay-protected settlement for custom producers implementing
the contract below. It does not install quality policy. The ordinary producer
mask path does **not** require this option and retains report-level discard
accounting. Identified settlement currently supports disaggregated, non-DAPO RL;
that accounting limitation does not restrict quality selection in either mode.

`RolloutRequest.completion_identities` contains one `CompletionIdentity` per
accepted completion in payload/group order. Allocate monotonically increasing
sequences **before generation**, scoped to source replica incarnation and global
rank; retain the originating `weight_version`. A restart uses a fresh replica
name. Retries reuse both identity and payload reference.

Report generation failures, quality exclusions and all members of insufficient
groups through `completion_failures`. Their identities come from the same
allocation, not from renumbering the surviving group. An optional `payload`
carries the rejected completion's transport reference for controller cleanup;
it never enters the training buffer. Without it, resources remain producer-owned.
Do not attach an accepted consumer's reference to a rejected identity or share
one disposable reference across distinct completion identities.

The controller settles and refills each rejected identity once using existing
versioned accounting and cleanup. A failure followed by a late payload settles
once and releases the late payload once. Replays of accepted completions never
release buffers a consumer may still own. Legacy discard/admission metric reports
must not be mixed with identified settlement.

Duplicate identities within a report, mismatched/future versions, invalid source
ranks and reports from departed replicas fail before accounting mutation. Malformed
or expired reports do not transfer new ownership. Replay state retains at most
4096 sequences per live source rank; sequences below the window are rejected,
never considered new. Producers must bound reordering. This is not a durable
controller-restart journal.

Admission and replica changes share the controller lifecycle lock. Closed
admission discards without reopening prompt capacity. An infrastructure error
after settlement starts poisons admission and requires job/controller restart;
automatic fatal propagation remains an outstanding follow-up.

## Downstream adoption

Applications should decide quality while producing the completion group and
populate the mask before submitting it to rewards. They must not discard
individual members in controller ingest or trainer preprocessing after advantage
normalization. Existing whole-group pre-save gates can emit an all-false mask;
partial gates emit the appropriate member mask. Keep driving-specific policy
and diagnostic retention in the application. Quality-policy errors must not
silently admit an unchecked group.

## Validation scope

`tests/test_completion_admission_modes.py` drives real worker generation-result
processing, local rewards, reporting and ingestion for disaggregated and both
colocated queue modes. It checks partial/all rejection, insufficient groups,
algorithm-specific minima, aligned advantages, and rejection accounting.
Existing tests cover remote/bypassed rewards and validation behavior.

`tests/completion_admission_canary.py` is a disaggregated integration fixture
requiring the RL-Gym companion modules. It marks producer masks before reward
processing, reports rejected references for cleanup, and replays each report.
Its identities are allocated at generation return, so it does not test generation
failure reporting. The previous late-controller-rejection live result does not
validate this revised design. Fresh live validation and standard producer identity
integration remain required before marking the PR ready.
