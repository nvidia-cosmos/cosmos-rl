# Application checkpoint metadata and sampling restoration

## Deterministic continuation: required follow-up

The adapter below restores metadata and a sampling stream. It is **not yet a
coordinated checkpoint protocol**. A cursor captured while rollout work is
outstanding is not necessarily the position corresponding to the saved model.
Do not use the current sampling/optimizer canary as evidence of deterministic
asynchronous job continuation.

The intended contract is a checkpoint containing all state required to continue
from one committed boundary, without serializing completed rollout queues or
partially generated rollouts. The proposed simplest boundary is:

1. Pause new prompt issuance and membership changes.
2. Finish or explicitly settle outstanding prompts and issued training updates;
   ensure there are no unconsumed rollouts or loader-prefetched samples omitted
   from the sampling snapshot. Partial batches and filtered/stale completions
   need an explicit settlement policy, not a drain that waits forever for a full
   batch. On timeout, fail the checkpoint without publishing it.
3. Freeze controller/sampling state and have every required trainer shard save
   the same completed-update boundary. Persist model, optimizer, scheduler,
   trainer RNG, and any mutable reference-model/application state.
4. Publish one immutable manifest only after all artifacts are durable. It must
   bind the controller state and all trainer shards to a common checkpoint ID,
   with artifact integrity checks. Incomplete saves must not be discoverable as
   resumable checkpoints.
5. Resume issuance only after committing the checkpoint. On restart, verify
   the manifest and execution compatibility, restore all state, and require
   worker/controller agreement before training can proceed.

Controller state must include the epoch, effective sampler/batch-sampler state
(including permutation, cursor and private RNG), counters, dataset/preprocessing
identity, and relevant configuration/topology. Controller RNG and request-ID or
seed-allocation counters must also be saved if they affect subsequent work.
Rollout generation must use reproducible per-request seeds or restore its RNG
state; a sampler cursor alone does not restore stochastic generation.

An empty queue at save time removes the need to persist rollout payloads, but
does **not** by itself make future asynchronous execution deterministic. Exact
continuation additionally requires reproducible admission/batch ordering and
weight-version assignment, as well as deterministic kernels and compatible
execution settings. If those cannot be enforced, the guarantee must be stated
as consistent checkpoint recovery, not bit-for-bit continuation.

Acceptance requires fresh-process restart parity of subsequent prompt IDs,
generation seeds/outputs, batch membership, policy versions, optimizer updates,
scheduler state, parameters and RNG-sensitive behavior. Exercise nonempty work
at checkpoint request, partial tails, filtering, all shard acknowledgements,
and interrupted saves. Negative tests must reject incompatible dataset/config,
mixed checkpoint artifacts and missing state before the next update.

This save-time coordination and end-to-end validation remain to be implemented;
the interfaces below describe the current restore-only implementation.

## Current restoration interface

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
- `sampling_owner`: `sampler`, or `batch_sampler` when a batch sampler exists.
  The latter owns its entire stream, including any nested sampler. Cosmos calls
  restoration once on that owner, never separately on two inconsistent cursors.
- `sampler_state`: opaque application dictionary. Cosmos does not inspect it.

Counters are strict nonnegative integers, epoch must be positive, and unknown
schema versions/fields are rejected. Step zero is a valid checkpoint, not a
discovery miss. Legacy checkpoint formats remain supported by the default loader;
an adapter must explicitly migrate its application's legacy metadata into v1.

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
