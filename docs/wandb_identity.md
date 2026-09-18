# W&B identity and run ownership

Applications can supply launcher-owned identity using ordinary configuration:

```toml
[logging]
logger = ["wandb"]
project_name = "training"
group_name = "experiment"
wandb_run_id = "attempt-123"
wandb_run_name = "My training attempt"
wandb_resume = "allow"
```

`wandb_run_id` overrides `train.timestamp` for W&B only. `wandb_run_name` is an
exact display name without a timestamp suffix. Neither changes checkpoint paths,
training timestamps, or distributed run identity. Launcher environment parsing
belongs to the application.

Defaults preserve existing behavior: the timestamp is the ID, the display name
is `experiment_name/timestamp` (or the output directory), and resume is `allow`.
Resume accepts `allow`, `must`, `never`, `auto`, or Python/JSON `None`/`null` to
leave resuming disabled. W&B defines their semantics in its
[initialization API](https://docs.wandb.ai/models/ref/python/functions/init).
Project and group already use `project_name` and `group_name`. Vision-generation
configs retain their existing `job` identity behavior.

`init_wandb` borrows an already-active `wandb.run`, returns it, and does not
reconfigure or finish it. Its creator owns that run's identity and lifetime;
configuration overrides apply only when Cosmos initializes a new run. Logging
invalidates a cached handle when the SDK global run is finished or replaced.
Call `init_wandb` explicitly to adopt a replacement. Failed initialization clears
the previous cached handle and retains existing best-effort error logging.

This logger targets the SDK's single active global run. Concurrent independent
W&B runs (`reinit="create_new"`) are not supported by this global logger; use
application-owned run handles for that use case.
