# Rollout-only execution

Set `[rollout].rollout_only = true` and
`[policy.parallelism].n_init_replicas = 0` to run generation and reward
calculation without a trainer. This is an explicit execution mode: an ordinary
training job whose policy replicas disappear keeps its existing end-of-rollouts
behavior. The controller rejects policy and reference registration in this mode.

A complete example is [`configs/rollout-only.toml`](../configs/rollout-only.toml):

```bash
cosmos-rl --config configs/rollout-only.toml
```

The example loads Qwen2.5-0.5B-Instruct and evaluates up to 32 GSM8K test
prompts on one GPU, with one completion per prompt. It needs the usual vLLM
installation, model/dataset access and Redis executable. Set
`policy.model_name_or_path` to a compatible local checkpoint to use local
weights. Custom entry scripts can register their own backend, dataset, packer
and reward functions through the normal `launch_worker` API; pass the script
as the launcher's positional script argument, for example
`cosmos-rl --config my-rollout.toml my_entry.py`.

## Initialization and versions

Each worker constructs its backend normally, including `post_init_hook()`.
Before starting its prompt producer or generation loop, it initializes the
engine with `load_format="auto"` and calls `post_init_engine_hook()` once through
the existing lazy initialization path. Re-entering that path after successful
initialization does not recreate the engine or repeat its hook. A custom backend
must set `_engine_initialized` after its successful `init_engine()`, as required
by the regular lazy initialization contract. Its `auto` load path must make its
checkpoint or application-owned model ready for inference.

No policy replica, initial weight transfer or rollout weight-sync communicator
is needed. Prompts and generated results use model version **0**, which denotes
the unchanged local model for this execution; there are no optimizer steps or
policy versions. Both synchronous generation and vLLM asynchronous generation
skip trainer version gates. Prompt preparation/prefetch continues to use the
existing bounded worker queues.

## Work, results and completion

`rollout.max_inflight_prompts` (default **128**) bounds outstanding prompts
across all workers, including generation, queued rewards and unacknowledged
reports. Each prompt reserves `n_generation` completion slots. Workers use the
normal controller-issued identities, ordered fetch/report receipts and failure
settlement: a retried HTTP report releases capacity once. A full inflight budget
returns an empty batch with `is_end=false` so producers retry after capacity is
released. Dataloader batches larger than the requested amount are kept as an
undispatched remainder and cannot overrun either admission limit.

Execution ends when the dataset completes `train.epoch` passes, or the optional
positive `rollout.max_prompts` limit is reached. The limit counts dispatched
prompts, including prompts whose generation fails; it does not request
replacements based on quality annotations. An empty dataset ends without any generation call.
`train.max_num_steps` does not define a rollout-only horizon.

## Generation and optional scoring

Rollout-only generation does not prepare a training objective: it never computes
advantages, applies GRPO group minima, normalizes scores or selects shared-prefix
training tokens. Every generated completion is retained, including an empty
string and completions marked low quality. Producer `completion_trainable` masks
and `completion_drop_reasons` remain annotations for the application result
consumer; they do not remove rollout-only outputs. Missing reserved completions
are separate generation-failure outcomes.

Set `[train.train_policy].bypass_reward = true` for pure generation. This skips
local reward construction, scorer tokenizer initialization, remote scoring setup
and reward executors. Public result scores are **None**, even though the existing
transport uses zero reward/advantage placeholders. No reward function needs to
be configured in this mode; `reward_function = {}` is accepted with
`bypass_reward = true`.

With `bypass_reward = false`, the ordinary configured reward functions or
`launch_worker(reward_fns=[...])` score every generated output, independently of
its quality annotation. Scorers receive the existing `(completion, reference,
**kwargs)` contract; `reference` may be **None**, so reference-free functions do
not need a fabricated label. Group scoring remains available as a scoring API,
without a training group minimum. Existing remote multimodal scoring is also
supported. Scorer errors, malformed scores and exhausted remote request retries
are strict failures that terminate the job; they never become zero scores or
silently discarded completions.

## Application result consumer

Pass `hook_fns={"rollout_result": consumer}` through the existing launcher entry
API to consume outputs. A plain callable is synchronous and required. A
`CallbackFactory` can create a consumer once in the owning process/thread using
`CallbackContext(role="rollout", replica_name=..., global_rank=...)`. A
`CallbackRegistration` must use `policy="required"` at this boundary; best-effort
queues are rejected because results and payload references are borrowed.

The consumer receives one frozen `RolloutResultEvent` per reserved prompt:

- `execution_id`, native `work_id`, and per-outcome `slot` provide stable attempt
  and completion identities. The controller starts a fresh execution UUID for
  every rollout-only run; dataset indices can repeat between epochs.
- `model_name_or_path`, `model_revision` and `weight_version=0` identify the
  configured checkpoint/model. A path/revision is an identity string, not a
  claim that local files have been content-hashed.
- `prompt`, `conversation`, `extra_info`, and each outcome's completion,
  conversation, token IDs, log probabilities and scorer metrics expose the
  original data or application references.
- Each `CompletionOutcome` has `status="generated"` or `"generation_failed"`,
  an optional score, producer quality flag/reason, and a failure reason when
  generation did not produce the reserved slot. Quality flags are separate from
  generation status. An unscored generated output has `score=None`.

The reporting rank invokes the consumer after optional scoring and before
`data_packer.get_rollout_output()`, HTTP reporting and transport cleanup. Its
borrowed lifetime ends when the call returns. The envelope is frozen, but nested
objects/tensor storage remain borrowed and must be treated as read-only. Copy or
persist the data synchronously before returning; never retain it in an
application queue. Outputs retain their worker/packer ownership through the
consumer call. Reward-reader cleanup hooks, when composed with application
callbacks, release only temporary reward-reader ownership and must not release
outputs still owned by the rollout worker. Rollout-only does not invoke the
training/admission `reward_annotation` hook. When composed with application
callbacks, both registered and packer `reward_read_complete` hooks still run
once at the reward-reader boundary, including scoring failures and unscored
bypass, before result publication. Scorer metrics and original quality masks
are exposed through `rollout_result` instead of training diagnostics.

Consumer exceptions fail the run before packing or settlement. Generation
exceptions attempt a failure event for each affected prompt, preserving the
original generation error if a consumer also fails. Async generation reports its
normal terminal empty-result failures through the same consumer boundary.
Native HTTP retries reuse their report receipt after consumption and do not
repeat the consumer. There is no persistence guarantee across process death or
an application consumer failure; applications that resume their own writes
should deduplicate by `(execution_id, work_id, slot)`. Applications own result
persistence and scoring semantics.

The runnable JSONL entry script copies tensors/arrays with values, shape and
dtype, retains structured artifact references, and rejects opaque objects rather
than writing a lossy representation:

```bash
COSMOS_ROLLOUT_RESULTS_DIR=./results cosmos-rl --config configs/rollout-only.toml examples/rollout_only_jsonl.py
```

Each reporting rank owns a separate JSONL file. Its factory opens the file in the
executing rollout process, and its `close()` flushes/releases application state
during worker teardown. For large media, adapt the sink to write artifacts and
persist durable references instead of inline arrays.

Reports remain receipt-checked and counted, then transport payloads are released
after consumption. Results never enter policy training buffers or consume a
training sample budget. The controller logs counts and outstanding completion
slots; the application consumer supplies the durable outputs.

## Stop and cleanup

After all initial rollout replicas have registered, a controller-side
`await controller.request_stop("reason")` closes new prompt admission, preserves
the first stop reason, and lets admitted generation/scoring work drain. It does not request a training checkpoint. After every reporter of
every registered initial rollout replica has acknowledged its final report and
posted its end signal, the controller broadcasts the normal NCCL-free STOP.
Workers remain command participants until that broadcast, preserving multi-rank
shutdown ordering. The controller exits after workers unregister.

Initialization, generation, asynchronous scheduler and transport failures use the
existing worker failure and resource-ownership paths. Engine, reward workers,
packer transports and packers are closed on success and failure, including
partial initialization. Cleanup failures remain errors; dependent resources are
retained when a producer cannot safely drain. The launcher fails and contains
peer processes on any nonzero rollout-only worker exit, even if the controller
has already exited after that worker unregistered. Normal training retains its
existing controller-supervised recovery policy. See
[`rollout_worker_lifecycle.md`](rollout_worker_lifecycle.md) for ownership details.

## Initial supported contract

The first contract supports disaggregated execution with a non-DAPO GRPO rollout
configuration, one or more rollout replicas, synchronous or vLLM asynchronous
generation, and custom backends using `DisaggregatedRolloutControlWorker`.
Trainer-oriented validation rounds, distillation, checkpoint resume, completion
admission, rank-affine dispatch, asynchronous R2R weight sync and the separate
TRT-LLM worker are rejected. This avoids silently accepting options whose
lifecycle still depends on trainers. Multi-DP backends must support empty DP
batches to consume an uneven final batch, as in ordinary training.

All controller and worker processes must run the same build. Model changes
require a new execution. This mode does not add elastic replacement or checkpoint
recovery for an already admitted prompt.
