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

Native automatic discovery distinguishes absence from failure. Only
`NoCheckpointFound`, raised before selecting or loading a committed checkpoint,
permits an automatic (`resume = true`) fresh start. An explicit path never
falls back. A missing/corrupt artifact, incompatible state, or loader/hook error
after selection propagates instead of loading an older candidate or base weights.
This applies to the built-in LLM SFT/DPO/GRPO and diffusion SFT/NFT trainers.

The legacy RL controller publishes its selected native checkpoint path to the
workers, just as the opt-in custom resume adapter does. No checkpoint disables
automatic resume before dispatch; corrupt metadata does not. This is not a
transactional rollback, a new distributed restore protocol, or a custom-format
adapter for SFT. Resume agreement alone still cannot certify a resume when both
sides report empty metadata.
