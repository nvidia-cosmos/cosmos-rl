# Controller dispatch and accounting

A real training step seals its participant set before publishing one immutable
Redis plan containing its rollouts and DataFetch commands. A lost Redis reply
retries the same operation identity and returns the original stream IDs. Exact
ACK retries are no-ops; changed schedules, reports or nonparticipants are rejected
before status and reservation mutation. A later registration is not retroactively
part of the already-issued update.

Redis Lua isolates publication but does not roll back writes. A marker is reserved
before appending; a surviving incomplete marker is terminal, not replayable.
Publication requires `noeviction`, and the script checks the Redis server
incarnation. Restart, ambiguous partial writes and exhausted publication deadlines
cannot silently create another logical dispatch. Completed markers expire only
after the operation's publication deadline. This is not durable queue recovery.

Controller writes use a bounded, single-writer outbox. Membership, dispatch and
accounting remain owned by the HTTP event loop; the writer only performs Redis
I/O. A queued operation retains its original deadline. Submission acknowledges
ownership, not delivery. Writer failure is observed independently and terminates
the controller rather than continuing with a partially issued step. Heartbeats
remain responsive while Redis is waiting. Ordinary worker Redis clients remain
synchronous. Shutdown waits for the owned outbox within a finite budget.

Per-step reports are detached before optional logging. Logging failure cannot
replay dynamic-sampling accounting or retain prior reports. SFT train/validation
ACK groups are separate by step and phase. Registration retries validate identity
without replaying initialization. Monitor failure and unsupported policy
scale-to-zero are explicit terminal outcomes, not transparent recovery.

Ordinary RUNNING-mode dispatch uses the same original-ACK barrier as drain.
Transient READY/REDUCED flags cannot authorize another update while a sealed
dispatch is outstanding, and issuing the final command is not job completion.
New participants cannot join an unsettled update. Losing an original participant
or replacing its active mesh marks that execution uncertain; it is not elastic
optimizer recovery. A trainer whose final ACK is recorded may still exit before
the remaining final ACKs. Healthy between-update membership changes remain
supported; autonomous SFT has the stricter fixed-cohort boundary below.

Admission-time staleness estimates are rechecked against the actual pre-update
weight version before batch publication. A smaller cohort or uneven rank-local
queue can otherwise make previously accepted work stale. Revalidation preserves
queue order, never dispatches a partial batch, and settles stale entries once.
Discarded obsolete prompt slots stay spent, matching admission-time filtering.
Configured lag remains authoritative: allowing one old step still permits it.
No new divisibility restriction on generation count or default objective change.

## Multi-replica SFT membership

Autonomous SFT seals its original cohort before the first mesh publication.
It cannot prove an idle boundary between optimizer steps, so later joins and
replacement processes are rejected and an active mesh rebuild is terminal,
even with the same replica names. Registration retries retain their process
session; policy ACKs carry that session and reporting rank. Matching controller
and worker builds are required for multi-replica SFT. This does not change the
single-replica SFT protocol, loss, optimizer or gradient reduction.

The latest ACK advances observed progress, but it is not cohort completion.
Successful completion requires the original final ACK set and no retained
unsettled groups. A trainer with its final receipt recorded may exit before its
peers; its receipt remains owned by the original group. Departure before that
receipt marks execution uncertain without shrinking the set or decrementing
unfinished sample accounting. Expired receipts are rejected rather than
recreating an old step. The existing successful early-stop/checkpoint protocol
is preserved. There is no elastic recovery, fabricated ACK or peer restart;
remaining workers may require the finite job timeout.

`tests/test_sft_membership.py` covers publication, departure/join timing,
incarnation/rank fencing, exact/changed/expired receipts, bounded pending groups,
interleaved progress and early-stop compatibility. The full-launcher fixture
uses real SFT training, a tiny generated Llama, and ordinary HTTP/Redis paths:

```bash
python tests/sft_membership_live.py --prepare /tmp/sft-membership-assets
SFT_MEMBERSHIP_CASE=healthy timeout --kill-after=15s 5m \
  cosmos-rl --config /tmp/sft-membership-assets/sft.toml \
  --policy 2 --rollout 0 tests/sft_membership_live.py
```

Use fresh asset/output directories for each case. `late-join` and `replacement`
must reject the injection and complete two real updates on both trainers.
`departure` and `rebuild` must emit the explicit `SFT_MEMBERSHIP_INCOMPLETE`
marker and terminate the controller; a timeout or unrelated failure is not a
passing containment result. These probes do not establish cross-node launcher
failure propagation.

When removal of a rollout leaves only ended replicas, reevaluate the existing
end-of-data drain after rebuilding the survivor mesh. In non-validation runs,
accepted full batches still train, issued updates retain their complete ACK
requirement, and only the unusable tail is released. Synthetic completion keeps
the original checkpoint horizon without inventing an optimizer update. This
does not recover unreported results or settle reservations owned by the departed
process. Validation-enabled exhaustion keeps active validation and issued
trainer ACKs as barriers, then drains complete accepted batches. It validates
the last real committed weight version (or reuses that version's completed
round) before synthetic completion. An early final round carries its explicit
identity even off the periodic schedule, without changing the checkpoint horizon.
After completion ACKs and policy exit, ordinary STOP releases the rollout workers.
An issued update missing a departed trainer's ACK remains incomplete even when
all surviving trainers appear ready. It cannot be certified successful or
replayed by this drain path; finite job limits remain required for such loss.

Portable two-rank controller/worker optimizer probe:

```bash
torchrun --standalone --nproc-per-node=2 tests/dispatch_commit_canary.py --case healthy
torchrun --standalone --nproc-per-node=2 tests/dispatch_commit_canary.py --case lost-reply
torchrun --standalone --nproc-per-node=2 tests/dispatch_commit_canary.py --case partial
torchrun --standalone --nproc-per-node=2 tests/dispatch_commit_canary.py --case departure
torchrun --standalone --nproc-per-node=2 tests/dispatch_commit_canary.py --case pending-ack
torchrun --standalone --nproc-per-node=2 tests/dispatch_commit_canary.py --case stale-surplus
```

`--cpu` is available for local checks. Rank zero owns Redis/controller state;
rank one uses the production policy-worker data-fetch path and a tiny optimizer
whose three updates are checked against a scalar reference. The canary substitutes
Gloo messages for HTTP ACKs and observes the partial-publication fatal callback
without exiting its fixture process. It is not a full production model run or
proof of scheduler-wide failure propagation.
The pending-ACK arm substitutes a transient READY flag before the real worker
receipt and verifies no extra command is published. The stale-surplus arm drops
two previously admitted old entries without an extra optimizer step; all three
real updates retain scalar-reference parity. These are controller-boundary
probes, not full elastic optimizer recovery tests.
The departure case checks one real update followed by an ACK-gated synthetic
completion, with no extra optimizer/scheduler call; it does not exercise native
mesh reconstruction or recover lost producer payloads.

## Validation delivery

The controller seals a validation round's reporting ranks before publishing its
validating weight command. Each issued prompt receives a round-local work ID;
dataset indices are not identities because custom samplers may repeat them.
Completion requires all issued work and a terminal receipt from every sealed
reporter, including reporters with empty assignments. It does not depend on the
dataset's nominal length. Completed payload groups and sampler iterators are
released.

One producer per rollout replica serializes fetches. Identical retries replay its
last response without advancing the sampler; each reporting rank similarly
serializes report receipts. Changed, out-of-order, unknown or stale work is
rejected before settlement. The controller keeps the latest completed terminal
receipt per reporter (bounded to 4096 sources), so a lost final reply can be
acknowledged after the next round starts. This is in-memory deduplication, not
durable recovery of crashed producers. A sampler exception is terminal because
its iterator may already have advanced.

Controller and workers must use the matching validation protocol; an older
unfenced command is not silently treated as successful validation. SFT retains
its separate validation-ACK path. Normal, async and colocated workers preserve
the round identity through local/remote rewards. TRT passes it across its
process boundary and applies validation/STOP in order, outside active delivery.
Validation HTTP operations have bounded requests and raise on exhausted retries;
rank-zero fetch rejection uses the existing prompt broadcast to release peers.
This does not promise scheduler-wide failure propagation or repair the independent
async generation/weight-adoption phase-isolation contract.

`tests/validation_delivery_canary.py` supports `healthy`, `empty`, `lost-fetch`,
`lost-report` and `fetch-rejected` under two-rank torchrun. Pass `--device cpu`
for local Gloo, or `--device cuda` for native CUDA/NCCL work, and
`--expected-package-root /path/to/installed/cosmos_rl` to assert source identity.
It uses real HTTP routes, sampler, receipt handling and fault injection, but
controlled arithmetic instead of a full model or simulator. The rejection case
also exercises the actual worker prompt broadcast on both ranks.

`tests/validation_drain_canary.py` supports `zero`, `early`, `periodic`, `initial`
and `nominal` with the same two-rank/device/package-root options. Production
Redis commands drive the policy worker's actual fetch and optimizer path;
validation receipts and training ACK retries settle exactly once. It checks
zero/partial/full accepted tails, active initial/periodic validation, synthetic
completion without an extra optimizer step, and the unchanged checkpoint
horizon. Gloo substitutes for HTTP receipts and weight transfer in this probe;
it does not establish native P2R/R2R recovery or a full-model training run.

## Training report delivery

Each reporting rank registers a process-incarnation identifier and serializes
its HTTP reports through one ordered receipt stream. The controller retains
only its last acknowledged response. Lost-reply retries return that response
without repeating admission, discard settlement or dynamic-sampling statistics.
Changed, expired, out-of-order and unregistered reports are rejected. End-of-data
is a report in the same stream: its exact retry is acknowledged, but that rank
cannot submit more work afterwards. Retiring a source discards its receipts;
reusing its replica name does not let an old incarnation report again.

Admission and receipt publication hold the controller's existing lifecycle lock,
so removal cannot interleave with report settlement. A failure after mutation
starts poisons the receipt and marks the controller terminal; the independent
monitor observes that error. No partially applied report is replayed. This is
in-memory deduplication, not a transaction rollback or durable payload recovery.
HTTP requests have a finite per-attempt timeout and at most three passes over
the configured controller endpoints; exhaustion raises and makes the client
unusable for subsequent reports. This is not an absolute operation deadline.

Controller and workers must use matching registration/report protocols. Normal
workers register their reporting ranks; TRT-LLM transfers its unused rank-zero
report stream to the wrapper through the existing IPC queue, disabling reporting
on the delegating client. The wrapper's reporter capability seals replica-wide
end-of-data without waiting for inner ranks that never submit reports. Colocated
training continues to use direct local admission, without this HTTP retry layer.
These changes do not alter filtering or training objectives.

Training prompt fetches use a separate ordered receipt stream under the same
registered incarnation. An exact retry returns an immutable copy of the original
prompts/end flag without advancing the sampler, assigning another weight-version
quota, or incrementing in-flight reservations twice. An empty throttled response
is acknowledged too; the next new request uses the next sequence. SFT retains
its existing fetch path. Sampler failure is terminal because an iterator may
already have advanced, and a retired source cannot refetch into a replacement.
This does not recover work after a producer process disappears.

Rank-zero rejection is relayed through the existing prompt broadcast. With
background prefetch, the client retains its terminal delivery failure and the
consumer observes it instead of reporting successful end-of-data. These checks
do not require the separate async-generation phase-isolation change. General
non-HTTP preparation failures and backend teardown retain their own audit scope.

`tests/rollout_report_canary.py` exercises real HTTP/controller admission under
two-rank torchrun with `healthy`, `lost-reply`, `changed-report`,
`partial-settlement` and `retired-source`. It accepts the same device and
expected-package-root arguments as the validation probes. Controlled CPU/CUDA
arithmetic supplies the result; it is not a full-model training or native TRT
engine test, and does not demonstrate recovery of lost producer work.

`tests/training_fetch_canary.py` uses the same two-rank/device/package-root
arguments with `healthy`, `lost-fetch`, `sampler-failure` and `fetch-rejected`.
It exercises actual HTTP/controller reservation logic and the worker prompt
broadcast. The sampler-failure case consumes a prompt before raising; retry
cannot consume another. Controlled arithmetic is not a training optimizer test.

Reservation settlement on producer departure, fixed-cohort SFT containment,
and use-time staleness filtering have the scoped tests described above. They do
not establish elastic recovery, durable queue replay, or an atomic distributed
optimizer commit. This change does not claim exact replay after restart or alter
trainer objectives.
