# Model-export ownership and publication

The trainer owns its background safetensors writer/upload task. Worker shutdown
waits for that owner before stopping heartbeat liveness or unregistering; the
generic worker error path also joins it before transport/process-group teardown.
Writer/upload errors are re-raised on join and checkpoint-finalization errors
are not converted into a successful exit. Both independent writer owners are
drained even when one fails. Unexpected export failure does not excuse essential
worker cleanup.

LLM exports take independent CPU snapshots. Each pipeline stage writes its own
shards on the training thread and all ranks agree completion before rank 0
publishes the index or uploads. This avoids background export collectives racing
training collectives. File writes can add latency at a PP export boundary;
single-stage exports retain background writing. Pipeline-parallel LoRA export
is rejected before writes because the single-adapter format needs an explicit
cross-stage merge; silently overwriting one stage with another is not supported.

LLM and Diffusers exporters write a sibling `*.incomplete-*` directory and rename
it to the requested destination only after all local files/configuration are
ready. A manifest cannot name an absent LLM shard. Failed writes leave the prior
destination intact. Replacing an existing destination first moves it to a unique
`*.previous-*` recovery sibling. Replacement has a brief missing-path window,
not a half-written-directory window; a failed rename restores the prior path.
An abrupt process crash between renames can leave that recovery sibling instead
of the final path. These siblings are retained for inspection/recovery, not
silently deleted. Concurrent writers to the same destination are unsupported.
Remote uploads occur after local publication and are not transactional with it.

SFT epoch-based checkpoint cadence counts a full rank-local loader epoch, with
no second DP division. Mid-epoch resume does not shorten that cadence; explicit
step-based frequency is unchanged.

Portable validation:

```sh
python -m pytest tests/test_model_export_lifecycle.py tests/test_resume_data_index.py
torchrun --standalone --nproc-per-node=2 tests/model_export_canary.py --device cuda --case healthy --output /tmp/export-healthy
torchrun --standalone --nproc-per-node=2 tests/model_export_canary.py --device cuda --case shard-failure --output /tmp/export-failure
```

Use fresh output directories. The canary validates actual files and both ranks,
including a delayed stage and an injected shard-write failure. It does not test
external storage-service availability or indefinite native filesystem stalls.
