# Controller-owned completion admission

Pass a `CompletionAdmission` instance as `completion_admission=` to controller
`main` or `Controller.setup`. Its synchronous `admit(context, rollout)` returns
`CompletionDisposition(outcome="accepted")` or
`CompletionDisposition(outcome="rejected", reason="quality")`. Application
quality policy stays in the adapter; counters, cleanup and refill stay in Cosmos.
The adapter receives a copy and must not mutate controller state or perform
blocking work. No process-global hook is installed.

This opt-in path currently supports disaggregated, non-DAPO GRPO. Default
ingestion, DAPO and validation are unchanged without an adapter. Configuring an
adapter requires a producer using the identified reporting contract below;
legacy metric-only failure reports cannot be mixed with it.

## Producer identity and ownership

`RolloutRequest.completion_identities` contains one `CompletionIdentity` per
extracted completion in payload/group order. Each identifies its originating
`weight_version` and a monotonically allocated `sequence` scoped to the source
replica incarnation and source global rank. Allocate it **before generation**.
Every new replica incarnation must use a fresh replica name. A retry reuses the
identity and the same payload reference; it must not publish a different buffer
under an already reported identity.

Report generation failures through `completion_failures`, whose entries contain
the same identity and a diagnostic reason. A failure followed by a late payload
settles only once; the late payload is released once and never admitted. A retry
of an accepted completion never releases a buffer a consumer may still own.
Each completion owns its own transport reference; sharing one disposable
reference across distinct completion identities is not supported.

Duplicate identities within a report, missing identities, changed versions,
future versions, invalid source ranks and reports from departed/ended replicas
are rejected before settlement. Diagnostic reasons do not affect identity.
Rejected malformed/expired reports have not transferred new payload ownership;
the producer remains responsible for their resources.

Replay state holds at most 4096 sequences per active source rank. Out-of-order
reports are allowed within that window. Older sequences are rejected as expired,
including unseen old sequences: eviction never grants permission to settle them
again. Producers must bound outstanding/reordered work accordingly. This state
is not a durable controller-restart journal.

## Accounting and synchronization

Admission runs before buffer or staleness accounting, under the controller's
lifecycle lock. The entire report is evaluated before mutations. Callback errors
return an error with no settlement; unchanged reports may be retried. Concurrent
ingestion and replica registration/unregistration use that same lock. Validation
uses its existing separate route and does not invoke training admission.

Accepted members of partial groups continue through ordinary staleness filtering
and buffering. Rejections use `settle_discarded_samples`, existing transport
cleanup dispatch, and its versioned refill hook. Different originating versions
are settled separately. Closed admission discards without invoking application
policy or reopening prompt capacity. Reason counters are bounded to 64 labels
plus an `other` bucket.

An infrastructure exception after settlement begins poisons admission and blocks
further ingestion; it is not treated as a retryable callback error. It requires
controller/job restart, not continued training on partially settled state.
Transport cleanup retains each backend's existing delivery/lease semantics;
this interface does not add reliable transport recovery.

## Validation still required

The focused CPU tests cover replay bounds, malformed reports, partial/all
rejection, versioned settlement, closed admission, callback failures and late
payloads. A live strict-version refill run and full producer integration are
required before release.
