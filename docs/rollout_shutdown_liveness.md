# Rollout heartbeat lifetime during shutdown

Normal STOP, final synchronous/colocated broadcast and deferred weight-sync
shutdown stop the main loop without stopping heartbeat liveness. The worker
retains heartbeats while joining owned helpers and releasing engine/simulator
resources, then makes the existing single bounded best-effort unregister request.
A `finally` path stops and joins the heartbeat even when engine cleanup or
unregister raises. Shutdown is idempotent; the async scheduler remains the owner
of its engine teardown.

Liveness during cleanup is finite. Before payload-server or engine teardown, a shared
monotonic deadline grants one `COSMOS_HEARTBEAT_TIMEOUT` interval (at least one
second), without renewing it between cleanup phases. The separate heartbeat process observes that deadline; if cleanup is
stuck, it stops heartbeats rather than indefinitely presenting the worker as
live. This is not successful cleanup, transport recovery or process termination;
the controller's normal expiration and finite job limit remain the fallback.
Heartbeat requests try each controller address once, checking the stop flag and
grace before every attempt. The periodic heartbeat loop supplies retries, not
the operational request backoff chain. An already-issued HTTP request retains
its bounded timeout; it cannot be recalled. Parent-death cleanup remains armed.

Terminal weight-sync failure still stops heartbeats immediately. The clean
shutdown path never clears/restarts a heartbeat stopped by such a fault.

`tests/test_rollout_heartbeat_shutdown.py` uses the actual heartbeat subprocess
and HTTP client to verify liveness through controlled cleanup and independent
grace expiration. `tests/test_vla_shutdown.py` covers ordering, idempotence and
exception paths; `tests/test_simulator_shutdown.py` covers real owned-child
kill/reap behavior. These are lifetime tests, not simulator numerical changes.
