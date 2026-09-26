# Asynchronous checkpoint snapshot ownership

Async checkpoints stage an owned CPU snapshot of model, optimizer, scheduler,
RNG, and supplied application metadata before submitting background writes.
This includes tensors already on CPU (for example Adam's step counter), tensors
inside lists and tuples, NumPy arrays, and other mutable metadata. Ordered model
state dictionaries retain their version metadata. Dictionary entries containing
meta tensors remain omitted, as before.

Staging is synchronous: the training/application owner must keep a coherent
snapshot boundary across these components while calling `save_checkpoint()`.
After it returns, subsequent training or sampling mutations cannot change the
queued snapshot. The cost is an owned CPU copy of previously aliased CPU state;
this does not change loss definitions, optimizer updates, or resume ordering.

This is not exact replay, restoration of in-flight work, or a distributed
checkpoint transaction. Existing per-rank completion markers still follow
successful serialization. Existing final-save and same-step-rewrite error
propagation remains unchanged.

## Regression and GPU validation

```bash
python -m pytest -q tests/test_checkpoint_snapshot.py tests/test_checkpoint.py
COSMOS_REQUIRE_CUDA=1 python -m pytest -q tests/test_checkpoint_snapshot.py
```

The snapshot test pauses the actual asynchronous writer, performs another Adam
and scheduler update, mutates application sampling state, then releases the
writer. Saved weights, optimizer counters/moments, scheduler, and application
state must remain at the earlier step. It runs on CPU and CUDA; the GPU gate
refuses to silently skip when CUDA was required. The original implementation
fails this test for both CPU weights and CPU optimizer counters with CUDA weights.

For the existing distributed controller-resume contract, run two fresh process
groups against the same new checkpoint directory:

```bash
torchrun --standalone --nproc-per-node=2 tests/controller_resume_gpu_canary.py save CHECKPOINT_DIR
torchrun --standalone --nproc-per-node=2 tests/controller_resume_gpu_canary.py resume CHECKPOINT_DIR
```

That canary checks rank-local sampling and optimizer continuation through the
public controller adapter; it is not a full simulator training run.
