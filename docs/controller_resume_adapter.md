# Application checkpoint metadata and sampling restoration

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
