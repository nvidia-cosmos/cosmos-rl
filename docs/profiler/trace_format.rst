Trace log line format
=====================

This page documents the structured ``[Trace] ...`` log line grammar
emitted by :mod:`cosmos_rl.utils.trace` and consumed by the offline
profiler (``cosmos_rl.tools.profiler``).  Components that wish to
participate in profiling should emit lines in this exact format so the
analyzer can parse them.

Grammar
-------

Each trace line is a single newline-terminated string with the
following grammar (EBNF-ish)::

    trace_line  := "[Trace]" SP fields
    fields      := field (SP field)*
    field       := key "=" value
    key         := identifier            ; ASCII letters, digits, underscores
    value       := number | identifier | quoted_string

Required fields (in this order):

* ``thread`` — logical thread / role name
  (``"trainer"``, ``"rollout"``, ``"controller"``, ``"ucxx_prefetch"``).
* ``op`` — operation name; the analyzer's opcode registry keys off this
  value.

Optional well-known fields (when present, parsed as floats in
milliseconds):

* ``start`` — start time relative to the process trace zero.
* ``end`` — end time, same reference frame.
* ``waited_ms`` — time spent blocked (e.g. waiting on a prefetch).
* ``transfer_ms`` — wire-time spent on a UCXX transfer.
* ``copy_ms`` — host-to-device or device-to-device copy time.

Optional well-known integer / size fields:

* ``count`` — number of items (rollouts, slots, episodes).
* ``bytes`` — total bytes transferred.
* ``iter`` — current training iteration.

Optional identity field (auto-emitted when
:func:`cosmos_rl.utils.trace.set_worker_id` has been called):

* ``worker`` — short tag identifying the producing process.

All other ``key=value`` pairs are passed through to the analyzer as
free-form metadata.

Example
-------

::

    [Trace] worker=policy_0 thread=trainer op=step_training start=12.3 end=42.7 iter=42
    [Trace] worker=policy_0 thread=trainer op=trainer_forward start=14.1 end=22.0
    [Trace] worker=policy_0 thread=trainer op=trainer_backward start=22.0 end=33.5
    [Trace] worker=rollout_0 thread=ucxx_prefetch op=ucxx_fetch start=8.0 end=11.5 transfer_ms=2.1 copy_ms=1.0 count=4 bytes=1048576

Producing trace lines
---------------------

Use :func:`cosmos_rl.utils.trace.format_trace`:

.. code-block:: python

    from cosmos_rl.utils.trace import (
        format_trace, get_trace_time, set_worker_id,
    )
    from cosmos_rl.utils.logging import logger

    set_worker_id("policy_0")

    start = get_trace_time()
    # ... do work ...
    end = get_trace_time()
    logger.debug(format_trace(
        thread="trainer", op="step_training",
        start=start, end=end, iter=current_iter,
    ))

Recommended opcodes
-------------------

The analyzer ships with an opcode registry; new transports can register
their own opcodes via the analyzer API.  The core set used by
upstream cosmos-rl trainers includes:

* ``step_training`` — full training iteration boundary.
* ``trainer_forward`` / ``trainer_backward`` / ``trainer_optimizer`` —
  per-phase timing within a training step.
* ``rollout_processing`` — rollout decode/prep before training.
* ``batch_preparation`` — batch concatenation.
* ``prefetch_submit`` / ``prefetch_collect`` /
  ``deferred_prefetch_collect`` — UCXX prefetch lifecycle (see
  :mod:`cosmos_rl.utils.payload_transport.ucxx`).
* ``ucxx_fetch`` — bg-thread UCXX fetch with ``transfer_ms`` /
  ``copy_ms`` / ``count`` / ``bytes`` fields.

Stability
---------

The grammar above is stable: new well-known field names may be added,
but existing field names will not change semantics.  Free-form
``key=value`` fields are always permitted.
