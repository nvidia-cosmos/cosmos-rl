# Learning-rate continuity on resume

Restoring scheduler metadata alone is insufficient: `LambdaLR` construction
sets optimizer parameter-group learning rates to the initial schedule value.
`LRSchedulersContainer.load_state_dict` now applies the restored epoch using
the current schedule and updates both optimizer LRs and `get_last_lr()`, without
advancing the scheduler. GRPO's first controller-informed schedule rebuild
therefore preserves the correct next-update LR.

Multi-replica SFT keeps an already-restored scheduler when preparing state
broadcast/unicast. It checks the controller's total-step horizon against loaded
metadata before changing any scheduler or worker state. A resumed run with a
missing scheduler is an error, not fresh warmup.

GRPO reference resets retain the effective scheduled LR when rebuilding the
optimizer. The reset clears optimizer history as configured; it does not restart
warmup or decay. The reference is an owned CPU copy, including with CPU model
offload. Pipeline checkpoints/reference snapshots use live stage parts under
their checkpoint names rather than an unused original model.

A training-batch boundary advances the scheduler, performs any configured
reference/optimizer reset, then checkpoints the state needed by the next update.
Each saving rank persists the active reference and its reset step. Reload checks
reference mode, keys, shapes, dtypes and reset-step bounds. Older checkpoints
without reference state may reconstruct the initial model-directory reference
only if no configured reset could have occurred; otherwise resume fails.

This does not provide exact replay of asynchronous rollout scheduling.
Direct checkpoint loading already restores optimizer state after scheduler
construction; the repaired failure is a subsequent scheduler rebuild/reset.

Portable checks:

```bash
export PYTHONPATH="$PWD"
python -m pytest -q tests/test_scheduler_continuity.py
python -m pytest -q tests/test_reference_reset_continuity.py
torchrun --standalone --nproc-per-node=2 tests/scheduler_resume_gpu_canary.py save ROOT
torchrun --standalone --nproc-per-node=2 tests/scheduler_resume_gpu_canary.py resume ROOT
```

Use the same fresh directory for both phases. The native two-rank canary compares
the first two resumed updates, effective learning rates, parameters, momentum,
and scheduler state against uninterrupted training. It uses rank-owned test
checkpoints and is not validation of topology-changing resume.
Use `--cpu` on both canary invocations for a local Gloo harness check; that does
not replace the CUDA/NCCL gate. The canary rejects a different imported checkout.
Add `--reference-reset` to both phases, using another fresh directory, to verify
two exact post-reset resumed updates including reference weights and the cleared
optimizer history. Its anchored toy loss tests continuation state, not GRPO
objective semantics or production model accuracy.
