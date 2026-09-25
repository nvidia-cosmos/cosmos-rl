# DPO reference-policy mode

The default DPO trainer remains reference-free. Its preference loss uses the
policy's chosen-minus-rejected response log-probability, and its `bco_pair`
quality term retains the existing zero-baseline formula. This is not a claim
of exact TRL BCO compatibility; no running reward baseline is introduced.

Set the following to enable frozen-reference log-ratios:

```toml
[train.train_policy]
dpo_reference_policy = true
```

The reference is the initial loaded policy, frozen for the run. For `sigmoid`,
the preference margin becomes
`beta * ((log_pi_chosen - log_ref_chosen) - (log_pi_rejected - log_ref_rejected))`.
The `bco_pair` rewards use the same policy/reference log-ratios; the optional
SFT term still uses only the policy's chosen-response cross entropy.
GRPO reference-reset settings do not reset the DPO reference.

The reference forward runs without gradients and in evaluation mode before the
differentiable policy forward. A CPU reference snapshot avoids keeping a second
full model on the GPU, at the cost of CPU storage, weight copies and an additional
forward per batch. Policy weights, buffers, modes and RNG are restored even if
the reference forward raises. Pipeline parallelism remains unsupported, as it
was before this option.

DPO implements the worker's checkpoint hook for both modes. In reference-based
mode each saving rank includes its frozen reference with the normal policy,
optimizer, scheduler and RNG checkpoint. Resume requires the same mode and a
complete matching reference state. Enabling it on an older reference-free
checkpoint, disabling it on a reference-based checkpoint, or losing reference
tensors fails instead of silently changing the training objective or restarting
from model-directory weights. Disable checkpointing only if continuation is not
required.

Portable checks:

```bash
python -m pytest -q tests/test_dpo_reference_policy.py
torchrun --standalone --nproc-per-node=2 tests/dpo_reference_canary.py
torchrun --standalone --nproc-per-node=2 tests/dpo_reference_canary.py --reference-free
```

The canary compares real FSDP updates with an unsharded global-batch reference,
including reconstruction of a new trainer from sharded serialized state. It is
not an exact replay guarantee for asynchronous rollout scheduling.
