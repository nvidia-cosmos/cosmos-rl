# Application-requested successful stop

On the controller event loop, applications may call:

```python
accepted = await controller.request_stop("application budget reached")
```

The reason must be nonempty. The first request wins; subsequent requests and
requests after completion return `False`. `True` acknowledges acceptance, not
finished shutdown. This is successful early completion, not a transport-error
handler or an immediate process kill. Application thresholds and quality policy
remain application-owned. A colocated controller facade forwards requests to
the authoritative controller through `POST /api/request_stop`; API clients may
also call `request_stop(reason)`.

The same operation supports disaggregated and colocated RL (including
colocated-separated execution), and single- and multi-replica SFT. RL requires
initialized policy and rollout replicas; SFT only requires initialized policy
replicas. Requests before initialization fail without changing admission.
Membership is frozen after a
request: new registrations are rejected and a changed policy cohort cannot be
certified as successful completion. Crash recovery and elastic stop are not
provided by this operation.

For disaggregated RL, the controller closes training prompt allocation and result admission, releases
buffered transport references through existing cleanup, and uses terminal
settlement for late results. It does not issue another training update. Already
issued updates must complete their ACK accounting; already active validation
continues, including its required weight synchronization. No new validation is
created just because an application requests stop.

Colocated RL finishes the already-issued iteration, including any prompts it
still needs, then consumes the terminal command before generating another
iteration. It also observes terminal commands while waiting for weight sync.
The initial iteration may already have been issued when stop is requested.

SFT owns its training loop instead of consuming RL update commands. Each
replica leader therefore rendezvous at `POST /api/training_boundary` before
each update, and broadcasts the shared decision to its ranks. A stop does not
revoke an already-granted update: that update and its active validation finish
before the next boundary stops all replicas. This adds one controller round trip
per replica per update. Step or membership disagreement fails the boundary;
a missing participant cannot produce successful completion.

At that safe boundary, RL's existing `TrainingCompleteCommand` protocol saves
the actual completed step (including zero) and waits for all completion ACKs.
The configured training horizon is preserved, not overwritten to force an exit.
Checkpoint failures retain the existing all-rank failure agreement and cannot
send a success ACK. Custom trainers must implement the existing
`save_checkpoint(..., is_final=True)` and `invalidate_checkpoint_completion`
methods. This feature does not define a new application checkpoint format.
SFT uses its existing final-checkpoint interface, flushes asynchronous saves,
and acknowledges completion only after all ranks agree that saving succeeded.
The controller requires every participating replica's final acknowledgement.
Checkpointing remains optional when disabled by configuration.

The stop reason is exposed as `policy_status_manager.stop_reason` and logged
when accepted. Worker/cohort failure handling remains the launcher's
responsibility; see the separate watchdog/launcher PR #747. A missing participant
or stalled operation must not be presented as a successful requested stop.
