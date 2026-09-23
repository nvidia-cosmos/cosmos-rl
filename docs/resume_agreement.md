# Checkpoint resume agreement

Controller and worker checkpoint extra information is a correctness contract,
not a logging envelope. The controller uses `step` and `remain_samples_num` to
schedule resumed training and reconstruct sampling; the trainer restores model,
optimizer and scheduler state corresponding to those counters. `total_steps`
also affects the training horizon/scheduler. Different values mean that the two
sides disagree about the state being resumed, so continuing is unsafe.

The existing exact-dictionary agreement is preserved, including application
fields. Unknown fields cannot safely be ignored: Cosmos cannot determine which
ones affect a custom trainer or sampler. Applications must not put rank-local
state or diagnostics in this shared contract. The native checkpoint reader
already restores and excludes rank-local `rng_state` before agreement.

This check is necessary, not sufficient: matching counters do not prove equal
model contents, dataset identity, or a consistent distributed checkpoint. Custom
checkpoint formats should include a shared immutable checkpoint/manifest identity
in their agreement metadata and verify the corresponding artifacts on load.

A mismatch raises `ResumeMetadataMismatch`, including under Python `-O`. The
controller returns HTTP 409 and exits with status 1 after sending the response.
The client does not retry that conflict; transient connection failures retain
their existing retries. Successful agreement is unchanged. No launcher or Slurm
propagation behavior is added: allocation-wide cleanup remains the launcher's or
scheduler's responsibility.

This change does not alter checkpoint discovery, legacy fallback-to-base-weight
behavior, or the successful-resume command ordering. It detects disagreement;
it does not certify a resume when both sides report empty metadata.
