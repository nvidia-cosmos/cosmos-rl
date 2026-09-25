# Empty data-parallel rollout slices

A nonempty controller batch may contain fewer prompts than rollout DP slices.
Every rank retains one queue slot for the global batch, even if its local slice
is empty. Each slot carries the maximum required weight version of the original
batch, so all ranks use the same version gate and fetch/command cadence. This
applies with preparation prefetch on or off and to colocated consumption.

Skipping a forward is not safe for every backend. `RolloutBase` defaults
`supports_empty_dp_batches = False`. A backend may explicitly set it to `True`
only if skipping an empty DP slice cannot omit a forward collective required by
any nonempty slice. Unknown/custom backends must opt in; existing sharded VLA
backends remain unsupported. OpenVLA/PI05 forward and objective behavior is not
changed. There is no backend-name heuristic or automatically generated dummy input.

The existing global prompt broadcast lets every rank reject an unsupported
partial batch before scatter, preparation or forward, without an additional
agreement collective. Supported empty slots do no preparation/generation and
create no prompts, rewards or reservation decrements. Empty validation reporters
still emit a terminal report. A globally empty controller response remains a
throttle/end response, not a fabricated queue slot.

Portable checks:

```bash
python -m pytest -q tests/test_empty_rollout_batch.py
torchrun --standalone --nproc-per-node=2 tests/empty_rollout_batch_canary.py
torchrun --standalone --nproc-per-node=2 tests/empty_rollout_batch_canary.py --cuda
```

The distributed canary uses actual worker queues, fetch/command collectives and
an explicitly independent toy backend. It verifies exactly-once prompt generation,
mixed-version gates and prefetch ordering, not real VLA empty-forward support.
Full participation by sharded backends requires a separate backend-defined
neutral-input/collective contract; this change does not implement that feature.
