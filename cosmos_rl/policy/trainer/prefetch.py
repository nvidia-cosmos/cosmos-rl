# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Worker-owned, depth-one payload lookahead; training ACKs never mean admission.

The control reader admits immutable notifications; only the training thread
consumes payloads or updates progress. Commands retain their original update
coordinates. A notification never authorizes optimizer work.
"""

from dataclasses import dataclass
from contextlib import contextmanager, nullcontext
import time
import threading
import hashlib
from collections import OrderedDict
from functools import wraps

import msgpack

import torch
import torch.distributed as dist

from cosmos_rl.utils import distributed as dist_util


def prefetch_fallback_reason(config):
    """One policy shared by controller and worker; never relax staleness."""
    if not getattr(config.train, "prefetch_payloads", False):
        return "disabled"
    policy = config.train.train_policy
    if config.mode != "disaggregated" or policy.type != "grpo":
        return "requires disaggregated GRPO"
    if getattr(policy, "uncentralized_training", False):
        return "requires centralized rollout metadata delivery"
    if policy.on_policy or policy.allowed_outdated_steps == 0:
        return "strict on-policy execution"
    if getattr(config.distillation, "enable", False):
        return "teacher preparation is not a payload-only operation"
    dims = config.policy.parallelism
    if any(getattr(dims, name) != 1 for name in ("tp_size", "cp_size", "pp_size")):
        return "requires a pure data-parallel policy"
    if config.custom.get("payload_transfer", "redis") not in ("nccl", "ucxx"):
        return "no separately fetched payload transport"
    return None


@dataclass
class PendingPayloadBatch:
    identity: str
    step: int
    rollouts: tuple
    future: object


def _serialized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._condition:
            return method(self, *args, **kwargs)

    return call


class TrainerPayloadPrefetch:
    """Exactly one future batch, independent of the trainer's optimizer loop."""

    def __init__(self, packer, *, device=None):
        self.packer = packer
        # A background Python thread still defaults to the training CUDA
        # stream. Isolate allocation/copy/decode too, not only native receives.
        self.fetch_stream = (
            torch.cuda.Stream(device=device)
            if device is not None and torch.device(device).type == "cuda"
            else None
        )
        self.pending = None
        self.completed_step = None
        self.failed = False
        self.fetch_latency_s = 0.0
        self.cohort_identity = None
        self._condition = threading.Condition(threading.RLock())
        self.active_step = None
        self._closed = False
        self._receipts = OrderedDict()
        self.lookahead_hits = 0
        self.lookahead_misses = 0

    def _check_failure(self):
        if self.failed:
            raise RuntimeError("Trainer payload prefetch is terminal")
        if self.pending is not None and self.pending.future is not None:
            if self.pending.future.done():
                self.pending.future.result()  # Propagate a failed receive before ACK.

    def _start_pending(self):
        pending = self.pending
        if (
            not self._closed
            and pending is not None
            and pending.future is None
            and self.active_step is not None
            and pending.step == self.active_step + 1
            and self.packer._prepared_prefetch_future is None
        ):
            pending.future = self._prefetch(pending.rollouts)

    @_serialized
    def notify(self, notification, rollouts, *, start_fetch=True):
        """Admit one immutable future batch, without touching the live cache.

        A reader can run ahead of the training loop. Wait (bounded, releasing
        the lock) for the previous admitted batch's ordinary command to take
        ownership, rather than buffering an unbounded sequence of futures.
        """
        if self._closed:
            return
        self._check_failure()
        identity, step = notification.batch_id, notification.global_step
        digest = hashlib.sha256(
            msgpack.packb(
                (
                    identity,
                    step,
                    notification.replica_name,
                    notification.cohort,
                    notification.rollouts,
                )
            )
        ).hexdigest()
        if not identity or type(step) is not int or step <= 0:
            raise ValueError("Invalid payload notification identity or step")
        if identity in self._receipts:
            if self._receipts[identity] != (step, digest):
                raise ValueError("Payload notification identity changed content")
            return
        if self.completed_step is not None and step <= self.completed_step:
            raise ValueError("Unknown stale payload notification")
        if self.active_step is not None and step <= self.active_step:
            raise ValueError("Payload notification does not name a future batch")
        deadline = time.monotonic() + self.packer._prefetch_timeout_s
        while self.pending is not None or (
            self.active_step is not None and step > self.active_step + 1
        ):
            if self.pending is not None and self.pending.step >= step:
                raise ValueError("Conflicting or reordered payload notification")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Payload notification admission timed out")
            self._condition.wait(remaining)
            if self._closed:
                return
            self._check_failure()
            # The main command can supply the same fallback metadata while
            # this reader waits for the previous batch's slot. Recheck after
            # waking so that concurrent duplicate delivery remains idempotent.
            if identity in self._receipts:
                if self._receipts[identity] != (step, digest):
                    raise ValueError("Payload notification identity changed content")
                return
        if self.active_step is not None and step != self.active_step + 1:
            raise ValueError("Nonconsecutive payload notification")
        self.pending = PendingPayloadBatch(identity, step, tuple(rollouts), None)
        self._receipts[identity] = (step, digest)
        while len(self._receipts) > 2:
            self._receipts.popitem(last=False)
        if start_fetch:
            self._start_pending()

    @_serialized
    def stop_notifications(self):
        self._closed = True
        self._condition.notify_all()

    @_serialized
    def fail_notification(self):
        self.failed = True
        self._condition.notify_all()

    @_serialized
    def check(self):
        """Surface a failed reader even while the main loop has no command."""
        self._check_failure()

    def _prefetch(self, rollouts):
        if self.fetch_stream is None:
            return self.packer.prefetch_payload_batch(rollouts)
        return self.packer.prefetch_payload_batch(rollouts, stream=self.fetch_stream)

    @_serialized
    def validate(self, command):
        self._check_failure()
        if (
            self.completed_step is not None
            and command.global_step != self.completed_step + 1
        ):
            raise ValueError("Nonconsecutive or duplicate training command")
        if self.pending is None:
            if command.prefetched_batch_id is not None:
                raise ValueError("Training command names an absent prefetched batch")
        elif (
            self.pending.step == command.global_step + 1
            and command.prefetched_batch_id is None
        ):
            return  # Notification arrived before current command execution.
        elif (command.prefetched_batch_id, command.global_step) != (
            self.pending.identity,
            self.pending.step,
        ):
            raise ValueError("Training command does not own the prefetched batch")

    @_serialized
    def take(self, command, dispatch):
        self.validate(command)
        identity = command.prefetched_batch_id
        start = time.monotonic()
        if self.pending is not None and self.pending.step == command.global_step:
            pending = self.pending
            overlapped = pending.future is not None
            if identity != pending.identity or command.global_step != pending.step:
                raise ValueError("Training command does not own the prefetched batch")
            if pending.future is None:
                pending.future = self._prefetch(pending.rollouts)
            self.packer.consume_payload_batch(pending.future)
            self.fetch_latency_s = pending.future.payload_timings["fetch_latency_s"]
            self.pending = None
            rollouts = pending.rollouts
            self.lookahead_hits += int(overlapped)
            self.lookahead_misses += int(not overlapped)
        else:
            if identity is not None:
                raise ValueError("Training command names an absent prefetched batch")
            rollouts = tuple(dispatch())
            future = self._prefetch(rollouts)
            self.packer.consume_payload_batch(future)
            self.fetch_latency_s = future.payload_timings["fetch_latency_s"]
            self.lookahead_misses += 1
        self.active_step = command.global_step
        self._start_pending()
        # A notification can be waiting for this step even if there was no
        # pending payload to consume. Wake it on every active-step advance.
        self._condition.notify_all()
        return rollouts, time.monotonic() - start

    @_serialized
    def submit_next(self, command, dispatch, *, defer_fetch=False):
        if command.prefetch_next_batch_id is None:
            return
        if self.pending is not None or self.failed:
            raise RuntimeError("Only one future payload batch is permitted")
        if command.global_step >= command.total_steps:
            raise ValueError("Cannot prefetch beyond the training horizon")
        rollouts = tuple(dispatch())
        future = None if defer_fetch else self._prefetch(rollouts)
        self.pending = PendingPayloadBatch(
            command.prefetch_next_batch_id, command.global_step + 1, rollouts, future
        )

    @_serialized
    def complete(self, step):
        self._check_failure()
        self.completed_step = step

    @_serialized
    def drain(self):
        """Unconditional terminal fence, never training from process cleanup.

        The controller drains admitted batches with real DataFetch commands
        before publishing TrainingComplete. A pending batch here is a protocol
        error, not permission to drop it or fabricate a completion ACK.
        """
        if self.failed or self.pending is not None:
            raise RuntimeError("Trainer completion before admitted payloads drained")
        self.stop_notifications()


def cohort_payload_call(worker, callback):
    """Agree recoverable Python outcomes before the next training collective.

    Native uncertainty remains terminal under the transport watchdog; it is
    never converted into an empty batch. No vote fabricates successful training.
    """
    result, error = None, None
    try:
        result = callback()
    except Exception as caught:
        error = caught
    if cohort_payload_max(worker, int(error is not None)):
        worker.payload_prefetch.failed = True
        raise RuntimeError(
            "Payload lifecycle failed on the training cohort; no update ACK"
        ) from error
    return result


def cohort_payload_max(worker, value):
    vote = torch.tensor([value], dtype=torch.int32)
    if dist.is_initialized():
        vote = dist_util.all_reduce_tensor_object_cpu(vote, op=dist.ReduceOp.MAX)
    vote = vote.to(worker.device)
    worker.inter_policy_nccl.allreduce(vote, vote, op=dist.ReduceOp.MAX)
    vote = vote.cpu()
    if dist.is_initialized():
        vote = dist_util.all_reduce_tensor_object_cpu(vote, op=dist.ReduceOp.MAX)
    return int(vote.item())


@contextmanager
def payload_cohort_scope(worker):
    """Keep one mesh through readiness, training and completion accounting.

    Reuse the shared operation scope when available. On older communicators,
    holding the same reentrant mesh lock prevents a failed collective from
    rebuilding/replaying against fewer peers. Failure is terminal for this
    pipeline; it does not provide elastic recovery.
    """
    pipeline = getattr(worker, "payload_prefetch", None)
    if pipeline is None:
        yield
        return
    comm = worker.inter_policy_nccl
    scope = getattr(comm, "operation_scope", None)
    if scope is None:
        ready = getattr(comm, "wait_comm_ready", None)
        if ready is not None:
            ready()
        guard = getattr(comm, "build_mesh_lock", nullcontext())
    else:
        guard = scope()
    try:
        with guard:
            ready = getattr(comm, "is_comm_ready", None)
            if ready is not None and not ready.is_set():
                raise RuntimeError(
                    "Payload training mesh changed before operation entry"
                )
            identity = (
                getattr(comm, "comm_idx", None),
                tuple(sorted(getattr(comm, "replica_name_to_rank", {}).items())),
            )
            if (
                pipeline.cohort_identity is not None
                and pipeline.cohort_identity != identity
            ):
                raise RuntimeError(
                    "Payload training mesh changed with an admitted batch"
                )
            # Notifications can arrive during or between commands. Pin from
            # first execution, not only when a future happens to be present.
            pipeline.cohort_identity = identity
            yield
    except BaseException:
        pipeline.failed = True
        raise


@contextmanager
def observe_optimizer_steps(trainer):
    """Count actual optimizer calls, never substitute delivered batch counts."""
    from cosmos_rl.policy.trainer.base import Trainer

    getter = getattr(trainer, "payload_prefetch_optimizers", None)
    if getter is None:

        def getter():
            return Trainer.payload_prefetch_optimizers(trainer)

    optimizers, counts = (), []
    handles = []
    report = {"prefetch/optimizer_counter_available": 0}
    try:
        try:
            optimizers = tuple(getter())
            counts = [0] * len(optimizers)
            for index, optimizer in enumerate(optimizers):

                def observed(_optimizer, _args, _kwargs, index=index):
                    counts[index] += 1

                handles.append(optimizer.register_step_post_hook(observed))
        except Exception:
            # Instrumentation is optional, never a rank-local training gate.
            # Unknown/custom optimizer layouts report unavailable, not zero.
            optimizers = ()
        yield report
    finally:
        for handle in handles:
            handle.remove()
        try:
            stable = bool(optimizers) and tuple(getter()) == optimizers
        except Exception:
            stable = False
        report["prefetch/optimizer_counter_available"] = int(stable)
        if stable:
            report.update(
                {
                    f"prefetch/optimizer_{index}_steps": value
                    for index, value in enumerate(counts)
                }
            )
