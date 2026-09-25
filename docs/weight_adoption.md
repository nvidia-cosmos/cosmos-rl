# Async weight adoption

Training preparation futures are training-only. Validation uses its own inline
preparation and leaves a training future with an equal dataset index untouched.
An asynchronous worker marks local prompt consumption complete once, reports
drain once after acknowledgment, and remains a command participant until STOP.
Disaggregated synchronous weight-update callbacks also defer final validation
and shutdown to the main loop, after the outer generation has returned. This
prevents nested generation and stops further training work after final validation.
WST full-state R2R writes record receipt of non-trainable parameters before a
later trainable-only P2R, but cancelled/failed writes never publish that receipt.
The async scheduler preserves training/validation phase identity and selects
the matching data packer. Validation first drains accepted training generation
and training reward reports; equal prompt indices do not mix the two phases.
An empty, failed or cancelled generation yields one terminal result, including
legacy prompts without completion identities. Training uses its existing discard
path. Validation checks the configured completion count, including partial or
surplus results, and fails explicitly instead of waiting indefinitely or inventing
rewards. Queue ownership spans dequeue through terminal publication,
so a task not yet registered as active cannot be mistaken for a drained phase.
Queued tasks abandoned at scheduler shutdown also receive terminal outcomes;
delivery/retry of those outcomes remains the caller's responsibility.
The async vLLM adapter drains every child completion before returning a failed
prompt. Successful siblings are not left generating after another child fails;
the existing all-or-nothing failed-prompt behavior is preserved. Cancelling the
parent still cancels and awaits its child gather.

For the async-engine live-weight path, P2R and R2R handlers stop scheduler
admission and drain active request tasks before writing. Pause admission and
task registration share a lock, and each nested/overlapping context owns its
pause; one context cannot resume another. Waiting from the scheduler's own loop
fails explicitly rather than deadlocking it. Pauses do not serialize writers;
the worker main loop already serializes these handlers.

After request drain, vLLM acknowledges a device synchronization RPC in each
backend worker process. This matters for cancelled requests: sending an abort
does not itself prove GPU completion. The writer then waits for its inference
stream (including P2R's joined copy-back stream) before admission resumes.
Backend-fence, write and completion errors poison the worker's live-weight state
and leave admission paused. The waits use `COSMOS_ROLLOUT_CMD_WAIT_TIMEOUT`;
an RPC timeout retains the pending future rather than cancelling native work.
Other async backends must implement the device-fence contract explicitly; the
base implementation raises instead of claiming that a no-op is safe.

`rollout.async_r2r_sync` is opt-in and supports one rank per rollout replica.
Multi-rank replicas are rejected before worker initialization: background command
routing does not currently coordinate those ranks. Buffer adoption remains opt-in;
synchronous-engine delivery retains its existing full/trainable selection except
that local readiness cannot override the command's selected tensor set. An
unseeded trainable-only receiver fails before native entry instead of switching
to a full transfer while peers send a subset. A successful full-state path
records frozen-weight receipt regardless of the command's optimization hint.
This adds no synchronization collective and does not prove peer failure
propagation; use finite job limits if another peer has already entered native
work. Default OpenVLA/PI05 training objectives are unchanged.

The async buffer preserves the live state dictionary's storage topology. Tied
weights, overlapping views, offsets and strides share one cloned backing
storage instead of independent per-key copies. Plain strided tensors are
required; unsupported distributed, sparse, quantized or lazy conjugate/negative
tensors fail explicitly. Cloning backing storage can retain bytes outside an
individual parameter view, as required to preserve its aliases.

Ownership has two device fences and a short CPU publication lock:

1. P2R/R2R buffer writers wait for the previous adoption's final buffer read.
2. Adoption waits for the completed writer's event before copying on the
   inference stream. The next writer waits for that copy's completion event.
3. While a writer owns the buffer, adoption keeps the previous live version; it
   does not block a generation thread behind a peer barrier. Initial generation
   waits for an adopted buffer, and prompt gating attempts adoption first.

Initial snapshot copies are fenced too. Failed transfers or adoption scheduling
poison the buffer, request worker shutdown and prohibit further adoption/reuse.
This is not transport recovery or a global job-abort mechanism.

Received and adopted versions are distinct. `current_weight_version` advances
when the copy is scheduled before inference, not when a background receive
finishes. Generation results retain the version captured at generation start.
In inference-sync mode later forwards can use newer policies: the recorded value
is a conservative oldest-version bound for staleness, not exact single-policy
attribution. Identified completions keep their original reservation identity.

The built-in generation and inference callback wrappers order forward work on
the inference stream. Custom background/auxiliary streams must establish their
own final-reader contract. Arbitrary external request producers and exact-version
validation pinning are outside this scheduler-owned live-write contract. Default
OpenVLA/PI05 objectives are unchanged.

The single-producer prompt-prefetch loop distinguishes clean exhaustion from a
failed producer. Its final batch must be validated and queued before publishing
end-of-data. Unexpected setup/payload failures remain observable on the main
thread; they cannot masquerade as an empty, completed dataset. Throttled fetches
and request retries use bounded, shutdown-interruptible backoff. This does not
repair empty rank-local batches in the separate multi-rank fetch path.

Portable validation:

```bash
python -m pytest -q tests/test_weight_adoption.py
python -m pytest -q tests/test_async_rollout_phases.py
python -m pytest -q tests/test_async_weight_pause.py
COSMOS_WEIGHT_DEVICE=cuda:0 python -m pytest -q tests/test_weight_adoption.py
torchrun --standalone --nproc-per-node=2 tests/weight_adoption_canary.py
torchrun --standalone --nproc-per-node=2 tests/weight_adoption_canary.py --packed
torchrun --standalone --nproc-per-node=2 tests/r2r_selection_canary.py
torchrun --standalone --nproc-per-node=2 tests/r2r_selection_canary.py --packed
RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29763 \
  python tests/async_rollout_phase_canary.py --config tests/configs/test_simple_grpo.toml --live-weight-fence
```

The canary uses actual R2R NCCL and generation/forward entrypoints over twelve
versions, with delayed adoption reads. Gloo replaces only its Redis phase
barrier; P2R seeding is a local staged-write fixture, not a full controller/P2R
protocol test. Separate tests inject pending/failed writers, preserve old live
weights, exercise both writer wrappers and verify conservative result stamps.
The selection canary runs the actual synchronous worker handler over four
full/trainable/full-state updates and checks every tensor on both ranks. Its
invalid-state control prohibits local native entry; it does not simulate a
healthy peer already blocked in a collective or assert global job termination.
The phase tests use the actual scheduler, worker validation/reporting and reward
queue with a small async tensor engine, not a full vLLM/IPC-engine deployment.
Set `COSMOS_REQUIRE_CUDA=1` to require real CUDA work in those phase tests.
The separate async phase canary requires vLLM and runs the actual engine with
dummy model weights: overlapping training work, delayed reward reports, one
generation fault and validation with reused prompt indices/different completion
counts, then a child failure while its sibling enters real generation.
With `--live-weight-fence`, it also exercises healthy/cancelled native requests,
delayed reads in the actual backend process, delayed writes through shared CUDA
IPC model storage, and backend-RPC failure that keeps admission paused.
Request/reward and write fixtures remain local; this is not a full controller,
native P2R/R2R transport or pretrained-model accuracy test.
