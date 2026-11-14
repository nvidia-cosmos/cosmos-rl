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

For step-bounded runs with validation, sampler exhaustion is not a prerequisite
for terminal cleanup. The controller closes admission and releases buffered tail
payloads after the final dispatch's sealed ACK set and final validation round
are complete, with no outstanding dispatch, active validation iterator or
payload lookahead. A missing training ACK cannot be replaced by live READY
status or a completed validation receipt. No extra synthetic training step is
issued to finish this path.

Colocated workers inherit the same producer and reward implementation. Their
controller queues only selected completions, aggregates numeric telemetry, and
does not sum routing IDs or originating weight versions. Both centralized and
uncentralized colocated ingestion follow this contract; existing queue-based
collection continues until enough accepted samples are available.

### Colocated terminal-sample accounting

Fetching a prompt charges its requested generations even in colocated mode.
Accepted payloads stay on the existing local queue path. Quality exclusions and
structurally failed/empty generations instead send an accounting-only report
through the registered producer's ordered HTTP receipt stream. No accepted
tensors, rewards or advantages cross that reporting path; healthy reports add
no HTTP request or training collective.

The controller validates the source incarnation, rank, sequence and current
preparing dispatch before retiring the count. It decrements both the global
in-flight charge and that dispatch's outstanding fetched balance. This makes
lost-ACK retries idempotent and prevents an exhaustion cancellation from
retiring the same dropped samples again. Training ACKs still settle only the
actual training batch; any accepted queue surplus remains charged. A fresh
discard cannot retire a batch whose preparation agreement is already sealed.

Each new dispatch records the charged local input carried from earlier updates.
Existing strict on-policy purges report `discarded_queued_samples` against that
carried balance, without changing which samples are trained or their ordering.
Dataset exhaustion cancels both remaining carried input and current fetched
input; acknowledged drops cannot be counted again. Once preparation is sealed,
a queue-only discard can retire only surplus outside the optimizer reservation.

On clean exit, each rank retires its own unused local queues before one final
CPU completion fence permits source unregistration. Empty ranks participate too:
rank zero must not unregister while a peer's queued-input receipt is pending.
Failure to confirm any purge prevents clean completion. This adds no per-update
training collective and leaves accepted completion reporting on its local path.

For example, six fetched samples with two exclusions leave four charged for
training, then zero after the training ACK. Six healthy samples with a batch of
four leave two queued samples charged until reused, purged or retired at clean
exit. Delivery failure stops the source rather than admitting healthy work after
uncertain accounting.

## Optional identified reporting

Set `[rollout].completion_admission = true` to enable the standard worker's
pre-generation reservations and the controller's replay-protected settlement.
Synchronous generation and the asynchronous task scheduler preserve identities
through failures, reward processing, selection, serialization and HTTP retries.
Exhausted report delivery raises rather than silently losing reservations.
Custom producers may instead enable `completion_admission=True` on controller
`main` / `Controller.setup` and implement the same contract.

This does not install quality policy or force a quality mask onto legacy
unmasked groups. The ordinary producer mask path does **not** require identified
reporting and retains report-level discard accounting. Identified settlement
supports disaggregated, non-DAPO RL; this accounting limitation does not restrict
quality selection in either mode. Unsupported configurations fail at startup.

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

Replacement prompts do not consume still-unissued normal prompt quota. This is
essential when a producer reports a failure before other producers finish
fetching their initial batch, especially with zero allowed staleness.

Admission and replica changes share the controller lifecycle lock. Closed
admission discards without reopening prompt capacity. An infrastructure error
after settlement starts terminates the controller process with exit code 86.
It cannot be converted into a retryable HTTP error or leave a poisoned controller
alive. Atomic validation errors remain ordinary errors before mutation. Process
restart and cross-node scheduler cancellation remain deployment responsibilities.

## Application callbacks and admission commit observations

Register application behavior through the existing `hook_fns` argument on
`worker_entry.main` / role entrypoints. A custom data packer is optional:

```python
from cosmos_rl.utils.application_callbacks import (
    CallbackFactory, CallbackRegistration,
)


def annotate(payload, result):
    return [
        {"sample": payload.prompt_idx, "slot": slot,
         "admitted": slot in result.selected_indices}
        for slot in range(len(result.rewards))
    ]


def finish_reads(payload, *, is_validation, error):
    # Finish application readers and release only their reward-owned views.
    # Worker/packer outputs remain live for later result/policy consumers.
    pass


hook_fns = {
    "reward_annotation": annotate,
    "reward_read_complete": finish_reads,
    "admission_committed": CallbackRegistration(
        handler=log_admission_events, policy="best_effort", queue_capacity=64
    ),
}
```

`reward_annotation(payload, result)` runs after native rewards, admission and
advantages have been computed. `RewardResult` contains tuples in original
completion order: rewards, filter rewards, advantages (`None` for rejected
members), and borrowed native report metrics. `selected_indices` identifies
native admitted members, or an empty tuple when the group is insufficient.
Native admission precedes whole-group dynamic sampling, buffering, outdated
filtering, training and ACKs. **Accepted does not mean trained or consumed.**
Treat the payload and result as borrowed read-only views valid until return.
The callback may return detached annotations; it cannot change rewards,
advantages, eligibility, reservations, replacement capacity or counters.
Application quality policy still belongs in the producer's completion mask.

Return `None` or one dictionary per original slot. Each dictionary supports only
finite JSON primitives and string keys, with at most 4096 encoded bytes, eight
nesting levels and 1024 nodes. The group's complete diagnostic envelope is
limited to 65536 UTF-8 bytes. Values are copied without invoking arbitrary
serializers; tensors, arrays, object handles and transport leases are rejected.
Native producer-owned reports are unchanged. Selected payloads carry
`completion_diagnostics`; `original_index` survives selection, and
`rejected_completion_diagnostics` retains excluded slots. Identified reports align
accepted diagnostics with identities and include rejected diagnostics in each
`CompletionFailure`, even when native reporting removes the rejected payload.

`admission_committed(events)` observes `AdmissionCommitted` events whose explicit
`stage` is `"admission_committed"`. Each event contains source replica/rank,
stable identity, native disposition and detached diagnostics, with no payload or
accounting operation. Native rejection accounting, cleanup and replay-window
commit precede this observation. It reports admission only: later buffering,
outdated filtering, training, ACKs and eventual policy consumption keep their
existing native owners. Duplicate identities and late cleanup-only payloads
produce no second event, including reordered and all-rejected reports.

A plain observer callable or `CallbackRegistration(..., policy="required")`
runs synchronously under the controller lifecycle lock; failures poison admission
and retain the existing fatal controller behavior (exit 86). Accounting is
already committed and cannot be repeated to repair an observer failure. Use this
policy only for application observations that must succeed for the role to
continue. `completion_settled` and `CompletionSettlement` remain deprecated draft
aliases; use `admission_committed` and `AdmissionCommitted` in new code. Registering
both observer names is rejected.

An explicit `policy="best_effort"` observer receives a deep detached snapshot on
a separate daemon worker through a bounded nonblocking queue. Factory creation
and observer execution run off the controller thread. A full queue drops/logs the
event; ordinary factory/handler errors are logged and dropped. These diagnostics
cannot block the controller lifecycle lock or poison admission. They may be lost
on process failure or bounded shutdown and must not own mandatory persistence,
capacity, counters or payload cleanup. This API is not an exactly-once journal
across controller restart. Best-effort mode is forbidden for borrowed reward
annotations, final-reader cleanup and required result consumers.

For worker-local state, register `CallbackFactory(create)` in place of a plain
callable, or wrap it in `CallbackRegistration`. `create(context)` returns the
callback for that hook. `CallbackContext` supplies role, replica name, global rank,
validation flag, PID and a lifetime-unique executing worker ID. Factories bind
once per hook/process/thread, including after fork; annotation and cleanup
registrations are separate bindings. Plain callables can be invoked concurrently
by reward threads and must be thread-safe. Factories are the preferred boundary
for clients/caches that cannot be shared between workers. Registrations/factories
must be serializable when a launcher uses process workers. Factory creation and
callbacks must finish synchronously at required borrowed boundaries; applications
must synchronize any asynchronous readers before returning.

Role shutdown quiesces reward workers before closing its callback registries.
An optional required callback object's `close()` runs once on its owner shutdown
thread; best-effort instance close runs on its telemetry worker. Process workers
register their own finalizers on first use, so a parent-initialized singleton
still binds and closes child-owned instances. Process-finalizer errors are
reported by multiprocessing after tasks finish; mandatory per-task payload
cleanup must use the strict `reward_read_complete` boundary rather than defer to
process-exit finalization. Bounded telemetry close can abandon a stalled daemon
with only detached metadata, leaving payload and transport ownership independent.

`reward_read_complete(payload, *, is_validation, error)` always runs synchronously
and strictly after reward evaluation and annotation, including failures. Registered
application cleanup runs first, followed by `packer.on_reward_read_complete` for
packer-owned readers; both boundaries and every submitted group are attempted.
Each releases only its own reward-reader references. Native evaluation errors
remain primary; otherwise the first cleanup error fails the task and secondary
errors are logged. `BaseDataPacker.annotate_reward_result` remains a compatibility
fallback when no annotation hook is registered. Packer-owned cleanup remains a
supported independent ownership boundary and is never suppressed by registration.

Cleanup receives each submitted group's original payload, including groups not
evaluated because an earlier group failed, and must tolerate repetition. It must
retain worker/packer output references needed by later result consumers,
trainable colocated views, policy leases and transport readers. Tasks canceled
before starting notify their submitting role with `CancelledError`; submission
failures finish all not-yet-submitted groups. Failed cancellation cleanup fails
shutdown and retains its resource owners. Cleanup does not grant permission to
release outputs that are still consumed by a required `rollout_result` callback.

Annotation and final-reader cleanup work for local, remote and bypassed rewards
in colocated and disaggregated training execution. With rollout-only execution,
the training/admission-oriented `reward_annotation` callback is deliberately
skipped; `reward_read_complete` still runs before the required `rollout_result`
consumer, retaining output ownership. Use that result consumer for optional
scores, producer masks/reasons and scorer metrics. Transient remote transport/server errors
retain readers for polling. Invalid responses, slot-count mismatches, nonretryable
HTTP errors and annotation failures finish pending readers with the original
error. Remote shutdown serializes against active readers and notifies pending
groups with `CancelledError`. Callback registries still close after quiescence if
another shutdown step fails, while the original/first error remains primary.

Identified admission observations retain the native disaggregated non-DAPO
restriction. Colocated callbacks expose native admission and detached diagnostics;
they do not promise per-identity controller events. A colocated consumer captures
fully rejected diagnostics during annotation before native reporting removes the
payload. Validation ignores training masks, exposes every original slot, uses its
own callback context/packer, and has no training identities, refill capacity or
`admission_committed` events.

`tests/test_application_callbacks.py` checks required/best-effort policies,
nonblocking bounded telemetry, thread lifetimes and actual fork worker creation
and close. `tests/test_reward_callbacks.py` checks registration without custom
packers, both cleanup owners and first errors, detached snapshots, replay
suppression, original slots and local/remote/bypass failure/shutdown behavior.
`tests/test_completion_admission_modes.py` exercises both colocated queue modes
and disaggregated reporting/ingestion. These CPU regressions do not certify
application-specific CUDA streams or distributed transport readers.

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

`tests/test_completion_reporting.py` checks pre-generation reservations, reward
pickling, stable identities, asynchronous failures, delivery exhaustion and real
subprocess exit after settlement/filter/buffering faults. These suites run in CI.

`tests/completion_admission_canary.py` requires the RL-Gym companion modules.
Run it as the launcher role entrypoint from the companion directory using
`tests/configs/completion_admission_disaggregated.toml` (two policy/two rollout
replicas) and `tests/configs/completion_admission_colocated.toml` (one policy
replica). Set `ADMISSION_CANARY_MODE` accordingly so the fixture selects the
appropriate packer. It uses standard identity reporting, injects a generation
failure and pre-reward quality rejection, then replays reports. Both configurations
run 20 steps. The test algorithm explicitly supports the companion's single-member
groups; GRPO's minimum is not weakened.

A separate disaggregated run with `ADMISSION_CANARY_SETTLEMENT_FAULT=1` must exit
nonzero after logging `Completion settlement failed; terminating controller`.
Use a finite outer timeout and distinguish a timeout from the expected failure.
