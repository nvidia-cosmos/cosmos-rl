# Execution workflows

Three explicit execution paths reuse the native controller, SFT/GRPO trainers, rollout backends, packers,
reward functions and teacher channel. Existing recipes keep their defaults.

## Infrastructure scope

Batch size, SFT microbatch size, rollout/learner allocation, generation count,
policy freshness and weight-sync cadence are user-owned workload settings.
Execution infrastructure consumes those settings; it does not tune them.
Zero counts for absent roles express the selected execution path, not a
rollout/learner capacity optimization.

Evaluate infrastructure optimizations with the workload fixed: source bytes,
sample order, preparation, model, precision, objectives, batch geometry,
topology and freshness rules. Removing redundant reads, copies or serialization,
and reducing scheduling or communication overhead within those contracts, are
in scope. Application models, image resizing, augmentation, reward/loss choices
and payload encodings belong in application adapters. Example recipes and
qualification fixtures demonstrate the interfaces; their configuration changes
are not generic Cosmos-RL speedups.

Storage retains synchronous immutable publication and the exclusive metadata
publication/finalization lock. Initialization durably establishes the corpus
directory layout, including newly created ancestors. Every metadata publication
fences its directory, including duplicate retries after a previous fsync failure.
Loader/preparation exceptions fail the whole learner cohort even when their
exception message is empty.

No speedup is claimed from the rejected reporting or storage experiments.
The design Page retains their measurements and historical qualification evidence.
PR qualification uses the exact final source and installed wheel.

## Select a path

`[execution].path` selects a data source and worker relationship. It does not
turn a logged reward into a training objective.

| Path | Roles | Objectives |
| --- | --- | --- |
| `train_from_data` | Learner; no rollout worker | Native SFT/PT, or declared recorded objectives |
| `train_from_generation` | Learner and rollout workers; optional teacher | Existing GRPO and OPD settings |
| `generate` | Rollout workers; no learner or teacher | No optimizer; optional scoring and a required result sink |

New execution fields reject unknown keys. Absent unused role counts become
zero; contradictory explicit counts, including launcher overrides, fail.
The launcher validates the configuration before allocating GPUs. Legacy SFT,
GRPO and OPD configurations need no execution section.
Placement accepts any positive local GPU count, so learner, actor and teacher
can occupy three distinct GPUs. A replica spanning nodes must divide exactly
across their GPU allocations; unsupported partial-node splits fail at startup.

```bash
# Direct local dataset; the example supplies four illustrative paragraphs.
cosmos-rl --config configs/execution/pretrain.toml examples/pretrain_local_text.py

# Commit generated records, then use the resulting dataset for student SFT.
cosmos-rl --config configs/execution/generate.toml
cosmos-rl --config configs/execution/train-corpus.toml

# A frozen checkpoint actor supplies original behavior likelihoods.
cosmos-rl --config configs/execution/independent-actor.toml
```

The examples use Qwen2.5-0.5B. Choose a model, dataset, output directory and
resource sizes appropriate for the application. Independent generation and
corpus SFT can use different model architectures when the durable boundary is
text or another representation that the student packer understands.

## Records and preparation

`cosmos_rl.execution.records.SampleRecord` is a versioned envelope around
application data:

- `sample_id`, `occurrence_id`, and optional original group identity, size and
  slot identify content independently of controller reservations.
- `rollout` holds the existing `Rollout` content: prompt, completion, token IDs,
  behavior/teacher probabilities, reward and advantages. `masks` and
  `annotations` retain per-objective and dataset metadata.
- `producer` records the actor/checkpoint, optional artifact and representation
  revision, and its actual behavior version when known. A learner step never
  replaces a recorded behavior identity.
- `sampling` and `attempt_id` distinguish the logical output from the attempt
  that produced it. Unknown revisions remain explicit for one-off generation.
  Resumable output requires an immutable producer identity as described below.

Records use composition: `SampleRecord(rollout=Rollout(prompt=..., completion=...),
sample_id=..., occurrence_id=...)`. The v2 storage serializer defines a stable
subset of Rollout content. It excludes prompt indices, stream markers, runtime
telemetry and transient teacher handles; resolve teacher targets before saving.
Native Rollout defaults remain unchanged. At the durable boundary, absent
rewards/advantages stay `None`, distinct from explicit zero. Tensor and
`DataReference` values pass through the envelope without copying payloads.
Pre-release flat v1 corpora must be regenerated; their revisions/checkpoints
cannot be treated as the same dataset after this format change.

The native SFT packer handles generated text targets. A preprocessed mapping
contains the packer's usual fields: for the decoder packer these are
`token_ids` and `label_ids`. Application packers may implement
`prepare_record(record)` to return their trainer input mapping. For example,
an observation/action adapter can put camera tensors from `record.rollout.prompt`
alongside action targets from `record.rollout.completion`. The attached
`_execution_record` retains provenance and objective views.

For continuous actions and other application losses, select an existing
registered trainer with `train.train_policy.trainer_type`. It opts in by
declaring `recorded_objectives`, a set containing the objective names it
implements. The worker checks enabled terms before source or model setup.
Each registered class declares this support explicitly, including subclasses.
The trainer implements the native SFT worker's step/checkpoint interface and
owns model-specific mathematics, normalization units, masks and collective
participation. It can reuse `execution.objectives.ObjectiveWindow` for shared
window counts and scaling. There is no requirement to manufacture token labels.

A custom `prepare_record` must honor
`record.annotations["empty_contribution"]`: preserve a valid model input while
giving every objective zero contributing mass. That flag also remains in the
attached record. The built-in token adapter masks `label_ids` automatically.
Checkpoint compatibility includes the selected trainer and packer identities.

Application packers may also implement `export_record(record)` for generation.
It must return a `SampleRecord` with the same sample/group identities and
producer. Resolve transient application handles into owned tensors or durable
values there. Serialization and persistence finish before the result callback
returns and before the borrowed generation buffers are released.

## Durable corpus

`CorpusWriter` publishes immutable manifests and content-addressed tensor
artifacts on a shared filesystem. Each manifest is a one-record shard. Artifact
writes finish before an atomic, no-replace manifest publication; temporary
files do not enter the dataset. Readers check lengths and checksums and load
owned CPU data. Arbitrary Python objects and object arrays require an application
adapter; the codec does not serialize them with pickle.

After all writers drain, call `writer.finalize(expected_records=N)` to publish
`complete.json`: the exact manifest membership, hashes, count and corpus revision.
Manifest publication and finalization share a filesystem lock; payload IO stays
outside that lock. Completed corpora reject new records. `close()` only closes
owned writes and never claims that a multi-writer dataset is complete.

`execution.source.kind = "corpus"` reads committed records from `source.path`.
Training requires a completed snapshot by default. Its revision covers dataset
metadata and manifest contents; `source.revision` can pin that snapshot.
For intentional partial training, set `source.allow_partial=true` and pin
`source.revision`; use `CorpusDataset(path, allow_partial=True)` to inspect an
unfinished run. All learner ranks agree the selected revision and source-opening
failures before model setup. Compact metadata read at snapshot creation supplies
filtering, group validation and byte admission without repeated manifest scans.
Each record read verifies and decodes the same manifest buffer against the
snapshot's dataset identity; it does not reopen the manifest or dataset metadata.
Unavailable artifacts, changed manifests and checksum failures are errors.
The supervised view excludes generation failures while retaining quality
annotations and successful targets that an RL quality gate rejected.

The initial native corpus path supports one learner replica with pure data
parallelism. Its exact sampler pads schedules with fully masked occurrences,
without duplicating contributing examples. A global optimizer window may have
empty local contributions. Native checkpoint metadata pins corpus revision,
objectives, topology, optimizer batch and microbatch sizes, shuffle seed, model
and preparation length.
Resume uses the native optimizer-boundary step and deterministic sampler
progress; changing that compatibility state is rejected.
Relocating the same corpus or changing `max_prefetch_bytes` preserves that
identity. Prepared-batch limits remain pinned because they can affect grouping.
Older location-sensitive fingerprints can migrate with their original settings
when the checkpoint also declares its microbatch size; a new checkpoint then
carries the portable identity. Recorded checkpoints created before microbatch
pinning are rejected for exact resume because the missing value cannot be
verified. Use their weights to start a new training run instead. Exact continuation assumes
deterministic preparation and repeatable model/optimizer kernels. To use
reproducible stochastic preparation, opt into
`execution.source.preparation_seed` and implement
`prepare_record(record, *, context)`. The frozen `RecordPreparationContext`
from `cosmos_rl.execution.source` supplies `seed` and `epoch`. The v1 seed
includes the configured seed, pinned corpus revision, durable sample/occurrence
IDs and training epoch; it excludes rank, loader worker, path and prefetch
timing. Changing the seed rejects checkpoint continuation.

```python
def prepare_record(self, record, *, context):
    generator = torch.Generator().manual_seed(context.seed)
    # Use this local generator for crop/noise/sampling operations.
    # NumPy default_rng(context.seed) and random.Random(context.seed) work too.
    return self.prepare_with_generator(record, generator)
```

Do not reseed process-wide randomness inside the adapter: with zero loader
workers, it shares the learner process. On this opt-in path, native train and
validation DataLoaders use an independent generator so iterator construction
does not advance the model's checkpointed RNG. The existing epoch lifecycle
updates preparation before native nonpersistent workers start. Preparation and
collation must otherwise be reproducible; random record scoring or external
state still requires an application contract. Pin augmentation settings through
`record_preparation_state()`. Existing adapters retain `prepare_record(record)`
and their checkpoint identity when the seed is unset.

For the native text replay recipe, set `train.deterministic=true` and export
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before launching workers. Attention now
auto-selects FA2 in this opt-in mode; ordinary training retains hardware-based
selection. An explicit `train.fa_version=3` is still honored (subject to existing
compile compatibility) and warns that replay is unqualified. Startup logs the
resolved version, request, deterministic flag and compile setting. The pinned
H100/FA3 environment failed repeated native updates even with the deterministic
flag; this does not establish a limitation for every FA3 build. PyTorch's
deterministic mode currently uses warnings for unsupported operations. Neither
that flag nor the preparation seed guarantees bitwise replay across arbitrary
models, kernels, hardware, batch geometry or library versions. Retain the same
qualified environment when comparing checkpoints.

`train.train_policy.mini_batch` and `train.train_batch_per_replica` remain
user-owned settings. The historical microbatch comparison changes workload
configuration and does not establish a generic infrastructure speedup. BF16
batch geometry can change numerical updates. Recorded checkpoint metadata pins
microbatch size separately from the optimizer batch; a microbatch-only change
is rejected on resume. Preserve the original full training configuration for
continuation.

`tests/microbatch_numerics_probe.py` compares native BF16 GPT/FSDP2 loss,
gradients and AdamW updates against an independent functional FP64 reference
at several microbatch sizes. It checks declared numerical bounds, including a
wrong-gradient scaling control; it does not require bitwise BF16 equivalence.
The sibling RL-Gym `scripts/qualify_microbatch_quality.py` uses disjoint cached
ShareGPT examples and fixed evaluation geometry to compare convergence and
held-out loss separately from per-update numerical agreement.

The standalone tokenized-text recipe is
`configs/execution/train-text-throughput.toml`, with the public entry point
`examples/train_recorded_text.py`. Set `execution.source.path` to a completed
corpus and `policy.model_name_or_path` to the desired checkpoint, then run from
the Cosmos-RL checkout:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 \
  cosmos-rl --config configs/execution/train-text-throughput.toml \
  examples/train_recorded_text.py
```

This recipe consumes `token_ids` and optional `label_ids` in inline record
targets or filesystem `DataReference` targets. Explicit labels retain SFT
prompt masks; omitted labels supervise every token for PT. Raw generated text
can use the existing `train-corpus.toml` recipe instead. The example recipe uses
optimizer batch two, microbatch two, two loader workers and prefetch factor one.
It initializes no rollout workers. Its settings are explicit example choices;
generic defaults are unchanged. The historical microbatch throughput difference
does not measure a Cosmos-RL code optimization or prescribe a user's batch size.

`source.max_prefetch_bytes` bounds serialized records admitted to native loader
batches. The worker divides it across the active batch and configured prefetch
slots and checks the batch before loading tensors. Oversized records fail;
the implementation does not spill them implicitly. The legacy dataset cache is
unsupported for corpus sources because it lacks their revision identity.

## Explicit recorded objectives

The token adapter uses the native SFT training loop with fixed-schedule batches
and materialized logits. Set `train.sequence_packing = true` to reuse native
packing within each microbatch on a compatible model. Attention segments and
positions remain separate; the last logit of one record never predicts the next
record's first token. Behavior probabilities, teacher targets, reward weights
and objective masks retain their original next-token positions. Denominators
still cover the entire distributed optimizer window. Packing metadata is
reconstructed from the same deterministic source batch after resume.

TP/PP/CP/EP, multiple learner replicas, dynamic load balancing and fused cross
entropy remain rejected for this adapter. Native legacy trainers retain their
own support. `train/packing_active`, `train/useful_input_tokens` and
`train/model_input_tokens` report actual packing and padding work, including a
model's existing fallback when packing is unsupported.

```toml
[execution]
path = "train_from_data"

[execution.source]
kind = "corpus"
path = "/shared/recorded-trajectories"
advantages = "group_standardized"

[execution.objectives]
supervised = 0.5
policy_gradient = 0.5
policy_clip = 0.2
```

Each coefficient defaults to zero. Disabled terms require no otherwise-unused
behavior probabilities or teacher targets. Each enabled term has its own
boolean mask over next-token positions and its own denominator over the whole
distributed optimizer window. Microbatch partitioning does not change those
denominators. All-empty windows skip optimizer and scheduler updates; locally
empty ranks participate when their peers contribute.

Supervised loss is next-token cross entropy. Optional
`source.reward_weighting = "nonnegative"` weights it by finite nonnegative
record scores and normalizes by contributing weighted token mass. Scores alone
do not enable this rule or an RL objective.

Recorded policy loss uses the clipped importance ratio against **original**
behavior log probabilities. It requires producer identity, an explicit aligned
`masks.policy_gradient`, and aligned `annotations.advantages`, or the explicit
group-standardization option above. This is an opt-in clipped off-policy token
objective; it does not claim to reproduce on-policy GRPO from old trajectories.
The application adapter owns token/representation alignment.

Supplied aligned advantages do not require groups or a minimum group size.
Generation failures and RL-quality-rejected completions remain ineligible.
For group standardization, groups must declare all original slots. It
uses finite committed scores from the original eligible group. Generation
failures and quality-rejected completions contribute no RL mass; standardization
groups with fewer than two eligible completions contribute no RL mass. Their valid
supervised targets remain eligible. Complete group statistics are prepared
before distributing individual records to training ranks.

Recorded distillation requires teacher identity in `annotations.teacher` and
finite normalized dense `annotations.teacher_logprobs` aligned with the student
next-token vocabulary. It uses forward KL from teacher to student, normalized
by contributing tokens. Missing or incompatible contributing targets fail;
arbitrary tokenizer mappings and sparse target codecs need an explicit adapter.
Mixed token losses share one vocabulary normalization; the CE-only path retains
its native cross entropy implementation. This changes no objective coefficients,
masks, normalization units or optimizer boundaries.

For scalar record scoring, set `source.score_records = true` and provide
`source.reward_function` or application `reward_fns`. This reuses the native
reward implementation over owned records without constructing an inference
backend. Initially this runs in the learner process with
`dataloader_num_workers = 0`. A scorer revision can be recorded in the
annotations. Group standardization uses committed scores, so rescore and commit
a new corpus before requesting it; transient per-record rescoring and group
standardization cannot be combined.

### Application-defined temporal batches

Observation/action trainers may use the optional helpers in
`cosmos_rl.execution.sequences`. A `SequenceSample` holds application-owned,
time-aligned tensor fields, named boolean loss masks and optional episode ends.
The application decides how observations, actions and targets align; no token
representation or temporal-policy implementation is imposed.

```python
from cosmos_rl.execution.sequences import SequenceSample, iter_sequence_batches

sample = SequenceSample(
    record.sample_id,
    fields={"observations": observations, "actions": actions},
    loss_masks={"supervised": valid_actions},
    episode_ends=terminated | truncated,
)
for batch in iter_sequence_batches(
    [sample], window_steps=32, context_steps=8, batch_size=4,
    max_bytes=256 * 1024**2, device=learner_device,
):
    # batch.fields, valid_mask, loss_masks, reset_mask and windows are available.
    # The trainer owns forward/backward, global normalization and optimizer step.
    ...
```

Windows never cross episode ends. Each step belongs to one contributing window;
overlapping context has no loss mass. Every window starts fresh model state,
with context serving as warmup. A recurrent application that carries state
between windows needs its own explicit state contract. Chunking changes temporal
context relative to a full episode; compare gradients against a reference using
the same declared windows, rather than claiming full-episode equivalence.

The iterator checks exact padded tensor bytes before allocating a batch and
flushes at its count or byte limit. A single oversized window fails. Compatible
field shapes/dtypes and objective names are required within a batch; applications
own bucketing for heterogeneous representations. Source tensors stay immutable.
The budget covers one prepared batch, excluding loaded source records, model,
optimizer and application workspace. `source.max_prefetch_bytes` continues to
bound admitted records; optional `source.max_prepared_bytes` is available for an
application trainer's staging budget. A custom reader may implement efficient
chunked or mapped reads; the filesystem baseline loads a whole admitted record.

Packers may override `record_preparation_state()` with JSON-serializable
windowing, mask and weighting settings. These are pinned in corpus checkpoints;
changing them on resume fails. Preparation finishes within an optimizer window,
so an unfinished window never advances committed source progress.

The bundled RL-Gym packer supplies `prepare_sequence`; its pointwise MLP trainer
accumulates all window gradients before one optimizer step. Application options
`recorded_window_steps`, `recorded_context_steps` and
`recorded_window_batch_size` select its temporal preparation. Its loss remains
supervised MSE, optionally weighted by per-episode discounted returns.

## Generation and restart

### Application owned durable payloads

Heavy observation/action applications can implement
`execution.storage.DataSaver.save(value, max_bytes=...) -> DataReference` and
the matching `DataReader.load(reference, max_bytes=...)`. Applications supply
these objects through their registered backend or packer; there is no storage
registry or new launcher. The reference contains a URI, format, conservative
stored/decoded `size_bytes`, and at most 64 KiB of JSON metadata. Use metadata
references for large inputs as well as outputs.

`FileSystemDataStore` supplies the baseline on local/shared storage:

```python
from cosmos_rl.execution.storage import FileSystemDataStore, save_data

store = FileSystemDataStore("/shared/corpus")
# Run in the producer, while it owns the trajectory and its buffers.
reference = save_data(store, trajectory, max_bytes=2 * 1024**3)
# The packer attaches the reference as SampleRecord.rollout.completion. The required
# result consumer commits its small manifest before report settlement.
owned_trajectory = store.load(reference, max_bytes=2 * 1024**3)
```

The baseline checks aggregate tensor/array bytes before serialization, writes
through temporary files, fsyncs, and publishes immutable content-addressed
artifacts without replacement. It compacts tensor views that would otherwise
serialize a larger backing storage. Checksums stream through bounded buffers;
no full serialized byte buffer is built. A saver returns only when the payload
is durable and all reads of borrowed buffers have finished. Internal writer
concurrency must be bounded and joined before returning. `save_data` raises
`DataSaveError` on a storage or contract failure; callers must propagate it.
The standard generation mixin treats it as fatal rather than discarding an
episode. Slow storage directly backpressures the producer.

Corpus manifests preserve references without resolving them. Corpus byte
admission counts declared payload bytes, not just small pointers; compressed
formats must account conservatively for decoded memory too. Application packers
resolve references through their reader, which validates integrity and returns
owned training data. Controller restart reconciles identity and reference
metadata without loading, hashing or serializing payloads. Tensor checksums are
verified by readers; full integrity scrubbing belongs outside controller startup.
The baseline filesystem reader can remap the reference URI to a configured root
on another machine, while checking the same relative content-addressed artifacts.

The RL-Gym application example uses its modular simulator, rewards, MLP and
optimizer with these interfaces. `scripts/qualify_execution_workflows.py` in
the sibling RL-Gym checkout qualifies the public launcher, generation restart,
supervised and return-weighted updates, checkpoint continuation and optional
500 MB/1 GB payload cases. The toy return-weighted MSE objective is application
supervision; it is not advertised as likelihood-based policy gradient.

`execution.output` specifies `path`, a stable logical `run_id`, and optional
`max_record_bytes`. It installs a required synchronous consumer. Applications
can instead register their own required result consumer, following its ownership
and failure contract.

Output is resumable by default. Supply a full immutable model commit revision
or `execution.actor.artifact_digest`, plus
`execution.output.generation_fingerprint`. The application owns that fingerprint:
change it when simulator/environment, scorer, representation or export semantics
change. Include semantic versions/settings, rather than operational settings
such as log verbosity, profiling or artificial sink delays. The controller
compares these small declarations; it never loads or hashes model weights.
An artifact digest is a caller/backend assertion of immutable content, not
automatic detection of edits to a local checkpoint under a stale declaration.
Moving branches/tags and a bare model path are insufficient for resumable output.
For a one-off run, set `execution.output.resume=false`; unknown model provenance
is allowed, but reopening journaled or committed work is rejected. Use a fresh
output path and logical run ID for the next one-off run. Existing corpora without
these identity fields cannot be resumed under the new contract.

Before dispatch, the controller journals the input occurrence. Output IDs
combine logical run ID, source occurrence and original completion slot. The
first committed output wins; duplicate attempts cannot replace it with another
sample or different provenance. All completion outcomes, including generation
failures, scores and quality decisions, remain visible in the corpus.

On restart, committed manifest metadata is validated and complete occurrences are
skipped. Partially committed prompts may generate again, filling missing slots
while retaining earlier commits. Logical prompt limits include already-seen
source occurrences. Changed input at an existing occurrence or changed run
metadata is rejected. This supports replayable finite datasets; it is not a
checkpoint protocol for a mutable external stream.
After input admission closes and every completion settles, the controller
finalizes the exact dataset snapshot before normal shutdown. A required
completion-publication failure fails execution. Early requested stops leave
an unfinished corpus for explicit inspection or partial training; it is
resumable only when `execution.output.resume=true` and its identity is pinned.

The controller bounds outstanding prompts globally across rollout workers.
Writing the required sink blocks report settlement and therefore admission of
more work. A slow sink cannot silently drop data. Source exhaustion ends the
job only after outstanding results are settled. Tensor and manifest byte limits
apply to each committed record; failed partial writes can leave unreferenced
artifacts for a future offline cleanup tool.

The baseline reference format is `cosmos.artifacts.v2`. Its declared size includes
both serialized artifacts and inline values, keys and encoding metadata. Save
and load enforce the same budget before returning data. Pre-release v1 references
underreported inline bytes and are rejected by this reader; regenerate those
local qualification datasets. Custom saver/reader formats remain application-owned.

## Independent actor and teacher

`execution.actor.weight_source = "checkpoint"` initializes the rollout engine
from its own model and disables P2R/R2R weight synchronization and learner-version
staleness filtering. Dispatch remains bounded by outstanding work. The actor's
immutable behavior version stays zero as the learner advances.

The initial independent-actor training path is disaggregated shared-queue GRPO,
with original rollout likelihoods enabled, off-policy operation, and no validation
or retry budget tied to learner versions. Different checkpoints require identical
fast-tokenizer pipelines, special-token semantics and chat templates. Broader
alignment requires an application/algorithm adapter.
Using original rollout likelihoods also preserves the sampled token IDs;
decoding and retokenizing them can change the sequence those probabilities describe.

Student-actor OPD uses its existing trainer and synchronization path. Teacher
configuration is copied before projection, and the teacher retains the student
tokenizer identity independently. Explicit execution recipes have bounded
teacher request count, aggregate pending request bytes, individual request
bytes and individual result bytes. `distillation.max_completed_bytes` adds an
aggregate retained-result limit (256 MiB by default for explicit execution).
The same limit bounds packed bytes in each learner inbox. Fetching a result
keeps its Redis capacity charged until training consumes or cancels it. A
waiting update takes priority over prefetched replies; local future copies may
be dropped and reread from Redis. If the update itself exceeds the inbox budget,
all ranks use the existing coordinated skip before model or optimizer work.
Retirement retries failed acknowledgments without recreating local entries.
These limits bound logical packed targets, not decoded Python objects, model
memory or total process RSS; Redis and the inbox can hold copies of the same
charged result, and one read may be in flight.
The pending-request count limit also bounds retained-result count. Publication
checks both limits atomically before changing request ownership. Capacity
exhaustion fails clearly; configure enough capacity for the application's cohort.
Acknowledgment/cancellation removes retained bodies and accounting once; expiry
pruning removes both together. Late completion cannot resurrect a cancelled or
consumed result. Small deduplication markers retain their TTL lifecycle.
Legacy settings remain compatible. Existing
teacher deadlines and cohort agreement decide an update together when targets
are delayed, missing or malformed. Completed results retain their existing TTL
and acknowledgment lifecycle.
After the last learner and rollout depart, the controller terminates the
teacher replicas independently of the final weight synchronization path.

EMA/external teacher updates, streaming between independently scaled jobs,
checkpointed mixtures of separate datasets and additional objective topologies
remain conditional extensions from phase 4 of the design.

## Validation

CPU qualification exercises the actual SFT loop with camera observations and
token targets, mixed gradients, zero-weight dependencies, all-empty windows,
checkpoint source compatibility and optimizer/scheduler resume equivalence.
Two real Gloo/DDP processes compare gradients and momentum with an independent
reference and check preparation/configuration failure agreement.

The application-trainer test feeds variable-length continuous observation/action
trajectories through the actual corpus dataloader, without token labels. Its
application-owned MSE and Gaussian policy losses match an independent gradient
and optimizer reference. The same qualification helper also passed using the
local RL-Gym checkout's actual `MLP` and `SimpleRLDataPacker`; this is an adapter
and model check, not a full RL-Gym simulator run.

Corpus tests cover atomic publication, concurrent attempts, interrupted writes,
partial-prompt restart, changed sources, corrupt artifacts, byte admission and
preservation of diagnostic outcomes. Actor/teacher tests cover independent model
projection, original versions, synchronization suppression and bounded teacher
traffic.

`tests/execution_workflow_canary.py` runs nine real GPU scenarios through the
public launcher and installed package: direct PT, generation, restart, generated
corpus SFT, mixed recorded losses, an independent actor, and OPD. It uses a tiny
random Qwen2 model with a cached production tokenizer to qualify infrastructure;
it is not a model-quality or live-throughput benchmark. The full Slurm CI runs
this canary alongside the existing suite.
