# Application-requested successful stop

On the controller event loop, applications may call:

```python
accepted = await controller.request_stop("application budget reached")
```

The reason must be nonempty. The first request wins; subsequent requests and
requests after completion return `False`. `True` acknowledges acceptance, not
finished shutdown. This is successful early completion, not a transport-error
handler or an immediate process kill. Application thresholds and quality policy
remain application-owned. No new HTTP endpoint is exposed.

Supported scope is initialized **disaggregated GRPO** policy replicas, including
registered custom GRPO trainers. Other modes and requests before policy
initialization fail without changing admission. Membership is frozen after a
request: new registrations are rejected and a changed policy cohort cannot be
certified as successful completion. Crash recovery and elastic stop are not
provided by this operation.

The controller closes training prompt allocation and result admission, releases
buffered transport references through existing cleanup, and uses terminal
settlement for late results. It does not issue another training update. Already
issued updates must complete their ACK accounting; already active validation
continues, including its required weight synchronization. No new validation is
created just because an application requests stop.

At that safe boundary, the existing `TrainingCompleteCommand` protocol saves
the actual completed step (including zero) and waits for all completion ACKs.
The configured training horizon is preserved, not overwritten to force an exit.
Checkpoint failures retain the existing all-rank failure agreement and cannot
send a success ACK. Custom trainers must implement the existing
`save_checkpoint(..., is_final=True)` and `invalidate_checkpoint_completion`
methods. This feature does not define a new application checkpoint format.

The stop reason is exposed as `policy_status_manager.stop_reason` and logged
when accepted. Worker/cohort failure handling remains the launcher's
responsibility; see the separate watchdog/launcher PR #747. A missing participant
or stalled operation must not be presented as a successful requested stop.
