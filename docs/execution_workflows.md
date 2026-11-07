# Execution workflows

This implements the initial supported phases of the execution infrastructure
[design](https://chatgpt.com/space/page_9dd048a396f48191a8a04bce375c84af).
It reuses the native controller, SFT/GRPO trainers, rollout backends, packers,
reward functions and teacher channel. Existing recipes keep their defaults.

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
- `inputs`, `targets`, `masks` and `annotations` hold text, tensors, arrays or
  nested durable values. Original behavior probabilities remain attached.
- `producer` records the actor/checkpoint, optional artifact and representation
  revision, and its actual behavior version when known. A learner step never
  replaces a recorded behavior identity.
- `sampling` and `attempt_id` distinguish the logical output from the attempt
  that produced it. Unknown revisions remain unknown; pin revisions or provide
  an artifact digest when reproducibility requires an immutable model identity.

The native SFT packer handles generated text targets. A preprocessed mapping
contains the packer's usual fields: for the decoder packer these are
`token_ids` and `label_ids`. Application packers may implement
`prepare_record(record)` to return their trainer input mapping. For example,
an observation/action adapter can put camera tensors from `record.inputs`
alongside action targets from `record.targets`. The attached
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

`execution.source.kind = "corpus"` reads committed records from `source.path`.
The source takes an immutable manifest snapshot. Its revision covers dataset
metadata and manifest contents; `source.revision` can pin that snapshot.
Unavailable artifacts, changed manifests and checksum failures are errors.
The supervised view excludes generation failures while retaining quality
annotations and successful targets that an RL quality gate rejected.

The initial native corpus path supports one learner replica with pure data
parallelism. Its exact sampler pads schedules with fully masked occurrences,
without duplicating contributing examples. A global optimizer window may have
empty local contributions. Native checkpoint metadata pins corpus revision,
objectives, topology, batch geometry, shuffle seed, model and preparation length.
Resume uses the native optimizer-boundary step and deterministic sampler
progress; changing that compatibility state is rejected.

`CorpusSource` separately supplies group-preserving transactional reservations
for application trainers: reserve one bounded window, commit its cursor only at
the optimizer boundary, or roll it back. `state_dict` saves committed progress.
A reservation in flight does not advance the durable cursor.

`source.max_prefetch_bytes` bounds serialized records admitted to native loader
batches. The worker divides it across the active batch and configured prefetch
slots and checks the batch before loading tensors. Oversized records fail;
the implementation does not spill them implicitly. The legacy dataset cache is
unsupported for corpus sources because it lacks their revision identity.

## Explicit recorded objectives

The initial token adapter uses the native SFT training loop with unpacked,
fixed-schedule batches and materialized logits. TP/PP/CP/EP, multiple learner
replicas, sequence packing, dynamic load balancing and fused cross entropy are
rejected for this adapter. Native legacy trainers retain their own support.

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

Groups must declare a complete set of original slots. Group standardization
uses finite committed scores from the original eligible group. Generation
failures and quality-rejected completions contribute no RL mass; groups with
fewer than two eligible completions contribute no RL mass. Their valid
supervised targets remain eligible. Complete group statistics are prepared
before distributing individual records to training ranks.

Recorded distillation requires teacher identity in `annotations.teacher` and
finite normalized dense `annotations.teacher_logprobs` aligned with the student
next-token vocabulary. It uses forward KL from teacher to student, normalized
by contributing tokens. Missing or incompatible contributing targets fail;
arbitrary tokenizer mappings and sparse target codecs need an explicit adapter.

For scalar record scoring, set `source.score_records = true` and provide
`source.reward_function` or application `reward_fns`. This reuses the native
reward implementation over owned records without constructing an inference
backend. Initially this runs in the learner process with
`dataloader_num_workers = 0`. A scorer revision can be recorded in the
annotations. Group standardization uses committed scores, so rescore and commit
a new corpus before requesting it; transient per-record rescoring and group
standardization cannot be combined.

## Generation and restart

`execution.output` specifies `path`, a stable logical `run_id`, and optional
`max_record_bytes`. It installs a required synchronous consumer. Applications
can instead register their own required result consumer, following its ownership
and failure contract.

Before dispatch, the controller journals the input occurrence. Output IDs
combine logical run ID, source occurrence and original completion slot. The
first committed output wins; duplicate attempts cannot replace it with another
sample or different provenance. All completion outcomes, including generation
failures, scores and quality decisions, remain visible in the corpus.

On restart, committed artifacts are validated and complete occurrences are
skipped. Partially committed prompts may generate again, filling missing slots
while retaining earlier commits. Logical prompt limits include already-seen
source occurrences. Changed input at an existing occurrence or changed run
metadata is rejected. This supports replayable finite datasets; it is not a
checkpoint protocol for a mutable external stream.

The controller bounds outstanding prompts globally across rollout workers.
Writing the required sink blocks report settlement and therefore admission of
more work. A slow sink cannot silently drop data. Source exhaustion ends the
job only after outstanding results are settled. Tensor and manifest byte limits
apply to each committed record; failed partial writes can leave unreferenced
artifacts for a future offline cleanup tool.

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
bytes and individual result bytes. Legacy settings remain compatible. Existing
teacher deadlines and cohort agreement decide an update together when targets
are delayed, missing or malformed. Completed results retain their existing TTL
and acknowledgment lifecycle.

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

`tests/execution_workflow_canary.py` runs seven real GPU scenarios through the
public launcher and installed package: direct PT, generation, restart, generated
corpus SFT, mixed recorded losses, an independent actor, and OPD. It uses a tiny
random Qwen2 model with a cached production tokenizer to qualify infrastructure;
it is not a model-quality or live-throughput benchmark. The full Slurm CI runs
this canary alongside the existing suite.
