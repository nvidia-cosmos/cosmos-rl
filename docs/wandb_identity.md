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
defer to SDK settings, including environment configuration. `None` does not
guarantee that resuming is disabled. W&B defines these policies in its
[initialization API](https://docs.wandb.ai/models/ref/python/functions/init).
Project and group already use `project_name` and `group_name`. Vision-generation
configs retain their existing `job` identity behavior.

`init_wandb` borrows an already-active `wandb.run`, returns it, and does not
reconfigure or finish it. Its creator owns that run's identity and lifetime;
configuration overrides apply only when Cosmos initializes a new run. Logging
invalidates a cached handle when the SDK global run is finished or replaced.
Call `init_wandb` explicitly to adopt a replacement. Failed initialization clears
the previous cached handle and retains existing best-effort error logging.

Initialization remains best-effort for every resume policy. For example, if
`wandb_resume = "must"` causes the SDK to reject initialization, Cosmos logs the
error and continues without W&B logging; it does not fail training. This setting
governs W&B resume behavior, not a requirement that telemetry be available for
training to proceed. Application-owned runs retain their application's policy.

An embedding application can instead explicitly bind its own SDK run:

```python
from cosmos_rl.utils.report.wandb_logger import init_wandb

init_wandb(config, run=application_run)
```

This binds the existing reporting path without initializing, renaming, configuring
or finishing that run, even when it differs from `wandb.run`. Identity overrides
are ignored for borrowed runs. Scalar/media values, steps and SDK default commit
ordering are forwarded unchanged. Only one reporting target is bound at a time;
this is not a multi-run router. The application owns the explicitly supplied
run's lifetime and must rebind before finishing it. Calling `init_wandb(config)`
without `run` restores the legacy active-global-run behavior. An invalid explicit
handle raises `TypeError` and clears the previous binding.
