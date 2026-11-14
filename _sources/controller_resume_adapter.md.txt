# Application checkpoint metadata and sampling restoration

## Consistent checkpoint recovery

The adapter restores metadata and a sampling stream. Application-owned trainer
formats remain application-owned. The optional `SamplingReplayLedger` and
`CheckpointManifest` helpers implement conservative sampling progress and
publish-last, verified local-file checkpoints without serializing rollout queues.

The contract is **consistent resumption, not exact replay**. Restore training
state and a safe sampling position associated with the same committed checkpoint.
Discard old in-flight work and regenerate unfinished rollouts after restart.
Some repeated work is acceptable; work must not be skipped merely because it
was issued before the checkpoint.

Required properties:

1. All required trainer shards restore the same completed-update boundary,
   including model, optimizer, scheduler, trainer RNG and any mutable
   reference-model/application state needed by the training algorithm.
2. The saved sampling state represents a safe replay position, not simply the
   latest fetched cursor. It includes the epoch and effective sampler state
   (permutation, cursor and private RNG where relevant). Out-of-order completion
   may require a conservative rewind so no uncommitted work is skipped. Repeated
   prompts after that rewind are allowed; progress counters must reflect the
   chosen replay policy rather than treating issued work as committed.
3. Controller state and trainer artifacts share a verified checkpoint identity.
   Publish the checkpoint only after all required artifacts are durable; reject
   incomplete or mixed saves and incompatible dataset/configuration state.
4. Restart with fresh transient rollout/transport state and reject results from
   the old execution. Require worker/controller metadata agreement before
   resumed training proceeds.

There is no requirement to persist rollout payload queues or partially generated
rollouts, drain all outstanding rollouts, reproduce generation outputs, or impose
deterministic asynchronous scheduling. Future completion order, batch composition
and numerical trajectory may differ from an uninterrupted run.

Acceptance requires fresh-process restoration of the committed training state
and safe sampling progress, followed by successful new updates. Exercise
outstanding/out-of-order work, conservative replay, partial tails, filtering,
and interrupted saves. Verify that uncommitted work is not skipped, repeated
work is allowed, and stale results cannot be admitted after restart. Negative
tests must reject incompatible dataset/configuration, mixed checkpoint artifacts
and missing required state. Exact future output or parameter parity is not an
acceptance requirement.

These are explicit application integration contracts, not automatic conversion
of native Cosmos checkpoints. The application connects issuance and completed
training to its sampling state and saves the custom trainer's artifacts. Cosmos
owns restoration ordering, initializes its counters from the chosen replay
boundary, and fences old rollout reports when an adapter is configured.

## Restoration interface

Pass an instance implementing `ControllerResumeAdapter` to controller `main`,
`Controller.setup`, or `ControllerDataFetcher`. No registry or module-global
callback is required. Without an adapter the existing Cosmos checkpoint loader
and legacy metadata behavior are unchanged.

```python
from cosmos_rl.dispatcher.data.resume import ControllerResumeMetadata
from cosmos_rl.dispatcher.run_web_panel import main

class ApplicationResume:
    def load_metadata(self, config):
        # Read and validate the application's checkpoint, not its live sampler.
        # A string config.train.resume is an explicit path and wins over discovery.
        saved = read_application_metadata(config.train.resume, config.train.output_dir)
        if saved is None:
            return None  # legal only for an automatic discovery miss
        return ControllerResumeMetadata(
            checkpoint_path=saved.path,
            checkpoint_id=saved.checkpoint_id,
            completed_training_steps=saved.cosmos_steps,
            completed_optimizer_updates=saved.optimizer_updates,
            remaining_completions=saved.remaining_completions,
            epoch=saved.epoch,
            sampling_owner="sampler",
            sampler_state=saved.cursor_state,
        )

    def restore_sampler(self, sampler, metadata):
        sampler.load_state_dict(metadata.sampler_state)

main(dataset=my_dataset, sampler=my_sampler_factory, resume_adapter=ApplicationResume())
```

The example's checkpoint decoder, sampler, and dataset are application-owned.
The existing Trainer methods still restore model, optimizer, scheduler and RNG
state; this adapter does not replace them or prescribe an on-disk format.
The custom trainer's existing `weight_resume()` method must return
`metadata.to_checkpoint_extra_info()` for the checkpoint it actually loaded.
The existing worker/controller resume agreement compares that dictionary
exactly; an adapter does not disable this check. Decode the same metadata in
both processes and do not report controller counters for different model state.

## Metadata contract (schema version 1)

- `completed_training_steps`: completed Cosmos training commands; initializes
  the controller step. Do not equate this with optimizer updates when a trainer
  performs multiple updates per command.
- `completed_optimizer_updates`: actual optimizer-update count, retained as
  `optimizer_updates` in controller checkpoint extra information.
- `remaining_completions`: remaining training work in generated completions,
  including the `rollout.n_generation` multiplier, not prompts or minibatches.
- `epoch`: one-based current sampling epoch.
- `checkpoint_path`: the selected checkpoint, propagated into `train.resume`
  in controller configuration so custom trainers can load the same selection.
- `checkpoint_id`: immutable shared save identity, returned in worker/controller
  agreement. Reusing a directory or step number is not a sufficient identity.
- `sampling_owner`: `sampler`, or `batch_sampler` when a batch sampler exists.
  The latter owns its entire stream, including any nested sampler. Cosmos calls
  restoration once on that owner, never separately on two inconsistent cursors.
- `sampler_state`: opaque application dictionary. Cosmos does not inspect it.

Counters are strict nonnegative integers, epoch must be positive, and unknown
schema versions/fields are rejected. Step zero is a valid checkpoint, not a
discovery miss. Legacy checkpoint formats remain supported by the default loader;
an adapter must explicitly migrate its application's legacy metadata into v1.

## Safe sampling progress without saved rollout payloads

`SamplingReplayLedger` accepts application-owned `SamplingBoundary` snapshots:
one-based epoch, effective sampler state, and remaining generated completions
from that position. Initialize it from the starting/restored boundary. After
issuing a prompt or batch, call `issue(after, completions=...)`; retain its token
alongside the transient work. Each generated completion has an index within
that token. Call `settle(token, index)` only when the corresponding optimizer
update is complete, or when it is terminally discarded under the application's
sampling policy. Receipt, admission, and dispatch alone are not settlement.

The ledger advances only across a contiguous fully settled prefix. For example,
if batches A and C have trained while B remains outstanding, the snapshot is
before B. Restart regenerates B and may repeat C. Remaining work includes that
repeated suffix; it is not the number of outstanding requests. This keeps replay
from consuming a smaller budget that would otherwise omit later dataset items.
Configured hard training-step limits still apply, as in ordinary training.

Capture `snapshot()` under the application's sampling/checkpoint synchronization
boundary, with trainer counters from the same completed update. A copied snapshot
does not advance the live sampler and cannot be overwritten by subsequent cursor
changes. Never acknowledge an update that is absent from the saved trainer state.
Construct a fresh ledger on restart; old tokens and duplicate settlements do not
advance it. It retains metadata for issued work after the earliest unresolved
request, so callers must bound issuance rather than indefinitely running past a
stalled gap. No rollout outputs or transport handles enter the snapshot.

For stateful data loaders, the boundary must cover prefetched work as well. Use
zero loader workers as in the reference canary, or supply an application snapshot
that includes that prefetched input state. The ledger cannot infer opaque loader
or sampler semantics; neither it nor the adapter probes them.

## Complete checkpoint publication

`CheckpointManifest.publish(root, metadata, required_artifacts=..., compatibility=...)`
is an optional POSIX local/shared-filesystem helper, not a trainer serializer:

1. Save and close every required trainer shard at the selected completed-update
   boundary, including optimizer/scheduler/RNG and application state. Save the
   shared `checkpoint_id` and completed counters inside those artifacts too.
2. Wait for all required shard writes to succeed. The designated coordinator
   passes their expected names and the matching safe sampling metadata to
   `publish`. Files are flushed and hashed before an atomic no-replace manifest
   publication; a directory without `manifest.json` is not resumable.
3. Both the controller adapter and trainer call `CheckpointManifest.load` with
   the **current application's** required artifact set and compatibility values
   (dataset/preprocessing identity, relevant settings and topology). Load rejects
   missing/mixed/corrupt artifacts and incompatible metadata before restoration.
4. The trainer verifies the checkpoint ID and completed counters in its decoded
   state against the manifest, restores training state, then returns
   `metadata.to_checkpoint_extra_info()` through the existing resume agreement.

Use immutable directories/files: publication does not permit replacing an
existing manifest. Hash validation cannot certify the semantic contents of an
arbitrary trainer file, so the application must serialize and check all required
state. Metadata in this optional helper must be JSON-serializable; other physical
formats and object stores may implement the adapter's commit/validation contract
directly. This does not add automatic migration or change native checkpointing.

## New execution, not recovered worker processes

Start fresh controller, workers and transport resources on resume. With an
adapter configured, `Controller.setup` generates a new `controller_execution_id`
even when a saved configuration contains an old one. Built-in network worker
clients copy this ID from their startup configuration into rollout reports.
Old/missing IDs receive non-retryable HTTP 410 before payload extraction, metrics,
end-of-stream handling or transport cleanup can modify the new attempt.

Custom network producers must construct `APIClient` with
`controller_execution_id=config.controller_execution_id`. Do not refresh that
ID in a running old worker. It is an attempt fence, not an authentication secret.
The direct colocated client lives inside the fresh controller/worker process and
does not have cross-attempt network reports. Without an adapter, the fence is
disabled and legacy reporting behavior remains unchanged.

## Reproducible validation

`tests/controller_resume_replay_canary.py` provides save and resume phases in
fresh two-rank process groups (`--device cpu` for Gloo, `--device cuda` for NCCL).
It deliberately trains batches A and C while B remains outstanding, publishes
all trainer shards plus a conservative sampling boundary, restores parameters,
optimizer momentum, scheduler and RNG, then trains B and repeats C. It verifies
the saved state and successful new updates, not identical future trajectories.
The fixture demonstrates application-owned sharding/serialization; it is not a
full launcher or large-model sharded-checkpoint test.

## Ordering and errors

With `train.resume=false`, the adapter is not called. With resume enabled,
metadata is loaded once after samplers have been constructed. Returning `None`
for automatic discovery bootstraps fresh training and sets `train.resume=false`;
returning `None` for an explicit path raises `FileNotFoundError`. Corrupt metadata
and provider/restoration exceptions propagate rather than silently restarting.
The provider must validate completed/atomic saves and honor explicit-path
precedence; remote storage and checkpoint discovery remain its responsibility.

Cosmos sets the epoch, then invokes `restore_sampler`, then constructs the
DataLoader and its first iterator. No live sampler is probed for batch shape,
length or skip arithmetic on the adapter path. Restored state is not overwritten
by another initial `set_epoch`. Normal subsequent epoch transitions still call
the sampler's `set_epoch`; custom samplers own those semantics. Use zero loader
workers when application state must not be advanced by PyTorch prefetch.

The adapter requires controller-owned sampling (`is_rl=True`, including the
existing multi-replica SFT controller path). Worker-owned SFT sampling must use
its trainer's existing restoration contract. Application-sharded cursors can
live in the effective sampler but are not inferred from remaining counts.
