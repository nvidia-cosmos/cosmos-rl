# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU unit tests for the multi-rank end-of-data shutdown protocol.

Covers the corner cases enumerated in ``rollout_multirank_shutdown.md``:

  * Corner 1 (P2R-gates-the-stop) and corners 2/3 (stop never emitted /
    R2R ``in_mesh`` gate) -- neutralised on the controller side by
    excluding ``status.ended`` rollouts from ``trigger_weight_sync`` so
    no P2R recv / R2R broadcast is ever issued to a worker that is
    leaving ``main_loop``.
  * Corner 4 (lockstep invariant) and corner E (``dp>1`` uneven tail) --
    handled by the controller's explicit ``StopCommand``: it is delivered
    over the redis command channel and broadcast across ranks inside
    ``consume_one_command``, so every rank sets the shutdown signal at the
    same collective call and leaves ``main_loop`` in lockstep.  (This
    replaced the former Option-C per-iteration drain vote.)

The true cross-rank lockstep itself is exercised by the GPU integration
test ``test_process_flow.py`` (bounded so a regression fails fast); these
tests pin the decision logic that test depends on, on CPU.
"""

import threading
import unittest
import asyncio
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, AsyncMock

from cosmos_rl.rollout.worker.rollout_control import (
    DisaggregatedRolloutControlWorker,
)
from cosmos_rl.rollout.worker import weight_sync
from cosmos_rl.rollout.worker.weight_sync import r2r_barrier
from cosmos_rl.dispatcher.command import Command, StopCommand, TrainingCompleteCommand
from cosmos_rl.dispatcher.status import (
    JobPhase,
    PolicyStatusManager,
    RolloutStatusManager,
    need_weight_sync,
    should_broadcast_stop,
    should_coalesce_skip,
)
from cosmos_rl.dispatcher.protocol import MESH_NAMES, RegisterRequest, Role


# ---------------------------------------------------------------------------
# Controller -- final-step weight-sync suppression (unified end-of-data stop)
# ---------------------------------------------------------------------------
class TestNeedWeightSync(unittest.TestCase):
    """``need_weight_sync``: the controller must NOT weight-sync on the final
    training step of a non-validation run.  That sync is never consumed (no
    step N+1) and, issued as an "ending signal", races rollout self-terminate
    at end-of-data -> orphaned P2R recv -> ``ncclCommAbort`` hang.  Rollouts
    now stop via the unified prompt-stream ``is_end`` path instead.  Validation
    runs keep the last-step sync (it drives the final validation)."""

    def test_non_validation_final_step_suppressed(self):
        # Final step lands on an interval boundary, yet must NOT sync.
        self.assertFalse(
            need_weight_sync(
                step=10,
                total_steps=10,
                sync_weight_interval=1,
                validation_enabled=False,
                validation_freq=None,
            )
        )
        # Even with a coarse interval the final step never syncs.
        self.assertFalse(
            need_weight_sync(
                step=10,
                total_steps=10,
                sync_weight_interval=5,
                validation_enabled=False,
                validation_freq=None,
            )
        )

    def test_non_validation_interval_step_still_syncs(self):
        # Non-final interval boundary -> normal sync (weights consumed by the
        # next generation).
        self.assertTrue(
            need_weight_sync(
                step=5,
                total_steps=10,
                sync_weight_interval=5,
                validation_enabled=False,
                validation_freq=None,
            )
        )

    def test_non_validation_non_interval_step_no_sync(self):
        self.assertFalse(
            need_weight_sync(
                step=4,
                total_steps=10,
                sync_weight_interval=5,
                validation_enabled=False,
                validation_freq=None,
            )
        )

    def test_validation_final_step_kept(self):
        # Validation on -> the last-step sync is retained to drive the final
        # validation even when it is not an interval boundary.
        self.assertTrue(
            need_weight_sync(
                step=10,
                total_steps=10,
                sync_weight_interval=5,
                validation_enabled=True,
                validation_freq=3,
            )
        )

    def test_validation_freq_step_syncs(self):
        self.assertTrue(
            need_weight_sync(
                step=6,
                total_steps=10,
                sync_weight_interval=5,
                validation_enabled=True,
                validation_freq=3,
            )
        )


# ---------------------------------------------------------------------------
# Explicit STOP command -- the unified, NCCL-free end-of-job stop
# ---------------------------------------------------------------------------
class TestStopCommandSerialization(unittest.TestCase):
    """``StopCommand`` must survive the redis pack/depack round-trip and
    resolve back to its own subclass (so ``get_rollout_command_handler``
    dispatches to ``handle_stop``)."""

    def test_pack_depack_roundtrip(self):
        cmd = StopCommand("replica-7")
        restored = Command.depack(cmd.pack())
        self.assertIsInstance(restored, StopCommand)
        self.assertEqual(restored.replica_name, "replica-7")
        self.assertEqual(restored.command_type, "STOP")


class TestStopCommandHandler(unittest.TestCase):
    """``handle_stop`` must break ``main_loop`` out of *any* branch -- a
    normal drain, an empty queue, or the weight-version-gate spin that no
    longer clears once weight syncs stop -- by setting both shutdown
    signals.  It must not touch NCCL (the whole point of the redis-channel
    delivery)."""

    @staticmethod
    def _worker():
        return SimpleNamespace(
            replica_name="replica-0",
            shutdown_signal=threading.Event(),
            shutdown_mp_signal=threading.Event(),
        )

    def test_sets_both_shutdown_signals(self):
        worker = self._worker()
        self.assertFalse(worker.shutdown_signal.is_set())
        DisaggregatedRolloutControlWorker.handle_stop(worker, StopCommand("replica-0"))
        self.assertTrue(worker.shutdown_signal.is_set())
        self.assertTrue(worker.shutdown_mp_signal.is_set())


class TestShouldBroadcastStop(unittest.TestCase):
    """``should_broadcast_stop`` gates the controller's one-shot end-of-job
    STOP broadcast.  It must fire exactly when the policy side is genuinely
    done (all policy replicas gone *after* we saw some, training finished,
    non-validation, not already sent) and stay silent for every other
    transition -- notably cold start, transient scale-to-zero, and
    validation runs."""

    # The canonical "policy just finished" state -> the one case that fires.
    _FIRE = dict(
        n_policy=0,
        had_policy_replicas=True,
        stop_broadcast_sent=False,
        validation_enabled=False,
        training_finished=True,
        all_rollouts_ended=False,
    )

    def test_fires_at_genuine_end_of_job(self):
        self.assertTrue(should_broadcast_stop(**self._FIRE))

    def test_silent_while_policy_still_present(self):
        # n_policy > 0: policy still running -> never stop the rollouts.
        self.assertFalse(should_broadcast_stop(**{**self._FIRE, "n_policy": 2}))

    def test_silent_on_cold_start(self):
        # n_policy == 0 but no policy ever seen -> startup, not end-of-job.
        self.assertFalse(
            should_broadcast_stop(**{**self._FIRE, "had_policy_replicas": False})
        )
        self.assertFalse(
            should_broadcast_stop(
                **{
                    **self._FIRE,
                    "had_policy_replicas": False,
                    "all_rollouts_ended": True,
                }
            )
        )

    def test_silent_on_transient_scale_to_zero(self):
        # Policy gone but training not finished (rolling restart /
        # scale-to-zero) and rollouts still live: stopping would be wrong.
        self.assertFalse(
            should_broadcast_stop(
                **{
                    **self._FIRE,
                    "training_finished": False,
                    "all_rollouts_ended": False,
                }
            )
        )

    def test_fires_when_rollouts_ended_but_total_steps_inflated(self):
        # Post-is_end recompute can leave total_steps > current_step (buffer
        # backlog the policy already walked away from).  STOP must still fire
        # once every rollout has checked out and the policy is gone.
        self.assertTrue(
            should_broadcast_stop(
                **{
                    **self._FIRE,
                    "training_finished": False,
                    "all_rollouts_ended": True,
                }
            )
        )

    def test_suppressed_for_validation_runs(self):
        # Validation keeps the final weight sync + R2R replica_should_stop;
        # STOP would race that path.
        self.assertFalse(
            should_broadcast_stop(**{**self._FIRE, "validation_enabled": True})
        )

    def test_one_shot_not_resent(self):
        # Already broadcast once -> never again.
        self.assertFalse(
            should_broadcast_stop(**{**self._FIRE, "stop_broadcast_sent": True})
        )


class TestShouldCoalesceSkip(unittest.TestCase):
    """``should_coalesce_skip`` implements depth-1 drop-to-latest.  It skips
    (coalesces) a weight-sync round only when coalescing is enabled, the round
    is not forced, and a previously issued round is still in flight
    (``last_staged_step > max_adopted_version``); otherwise it issues."""

    # Canonical "a round is still in flight" state -> the one case that skips.
    _SKIP = dict(
        coalesce_enabled=True,
        forced=False,
        last_staged_step=10,
        max_adopted_version=7,
    )

    def test_skips_while_round_in_flight(self):
        # Issued step 10, rollouts only adopted 7 -> in flight -> skip.
        self.assertTrue(should_coalesce_skip(**self._SKIP))

    def test_issues_when_rollouts_caught_up(self):
        # Rollouts adopted the last staged version -> issue (drop-to-latest).
        self.assertFalse(
            should_coalesce_skip(**{**self._SKIP, "max_adopted_version": 10})
        )
        # Adopted even beyond -> still issue.
        self.assertFalse(
            should_coalesce_skip(**{**self._SKIP, "max_adopted_version": 12})
        )

    def test_first_sync_issues_naturally(self):
        # Nothing staged yet (last_staged = -1) -> never skip, no special-case.
        self.assertFalse(
            should_coalesce_skip(
                coalesce_enabled=True,
                forced=False,
                last_staged_step=-1,
                max_adopted_version=-1,
            )
        )

    def test_never_coalesces_when_disabled(self):
        # Feature off -> behave like the unconditional every-interval sync.
        self.assertFalse(
            should_coalesce_skip(**{**self._SKIP, "coalesce_enabled": False})
        )

    def test_forced_round_always_issues(self):
        # Validation-trigger steps override the skip even with a round in flight.
        self.assertFalse(should_coalesce_skip(**{**self._SKIP, "forced": True}))


class TestWeightSyncForced(unittest.TestCase):
    """``_weight_sync_forced`` forces a sync only on validation-trigger steps."""

    def _mgr(self, validation_enable=False, freq=0):
        mgr = PolicyStatusManager()
        mgr.config = SimpleNamespace(
            validation=SimpleNamespace(enable=validation_enable, freq=freq),
        )
        return mgr

    def test_not_forced_without_validation(self):
        mgr = self._mgr(validation_enable=False)
        self.assertFalse(mgr._weight_sync_forced(step=10, total_steps=100))

    def test_forced_on_validation_freq_step(self):
        mgr = self._mgr(validation_enable=True, freq=10)
        self.assertTrue(mgr._weight_sync_forced(step=10, total_steps=100))
        self.assertFalse(mgr._weight_sync_forced(step=11, total_steps=100))

    def test_forced_on_final_step_with_validation(self):
        mgr = self._mgr(validation_enable=True, freq=10)
        self.assertTrue(mgr._weight_sync_forced(step=100, total_steps=100))


# ---------------------------------------------------------------------------
# Controller -- policy-seen state is set by /register, not monitor polling
# ---------------------------------------------------------------------------
class TestRegisterTracksPolicySeen(unittest.TestCase):
    def test_policy_register_sets_persistent_seen_flag(self):
        """Short tests can register and unregister between monitor scans."""
        from cosmos_rl.dispatcher import run_web_panel

        original_any_seen = run_web_panel._replicas_were_registered
        original_policy_seen = run_web_panel._policy_replicas_were_registered
        run_web_panel._replicas_were_registered = False
        run_web_panel._policy_replicas_were_registered = False
        request = RegisterRequest(
            replica_name="policy-0",
            role=Role.POLICY,
            mesh_names=MESH_NAMES,
            global_rank=0,
            host_ip="127.0.0.1",
            host_name="localhost",
            ranks=[0, 0, 0, 0],
            group_size=[1, 1, 1, 1],
        )
        try:
            with patch.object(
                run_web_panel.controller, "register", new=AsyncMock()
            ) as register_mock:
                asyncio.run(run_web_panel.register(request))
            register_mock.assert_awaited_once()
            self.assertTrue(run_web_panel._replicas_were_registered)
            self.assertTrue(run_web_panel._policy_replicas_were_registered)
        finally:
            run_web_panel._replicas_were_registered = original_any_seen
            run_web_panel._policy_replicas_were_registered = original_policy_seen


# ---------------------------------------------------------------------------
# Controller -- trigger_weight_sync excludes ended rollouts
# ---------------------------------------------------------------------------
def _replica(name, ended, start_time):
    return SimpleNamespace(
        name=name,
        start_time=start_time,
        status=SimpleNamespace(ended=ended),
    )


class _RolloutMgrStub:
    def __init__(self, replicas):
        self._replicas = replicas
        self.rollout_atoms_in_replica = 1

    def get_all_atoms_arrived_replicas(self):
        return list(self._replicas)


class TestTriggerWeightSyncExcludesEnded(unittest.TestCase):
    """Corners 1-3: the controller must not issue P2R/R2R to a rollout
    that has already POSTed ``is_end`` (and is self-terminating), unless
    validation is enabled (then the rollout stays for the final
    validation and must keep receiving the sync)."""

    def _run(self, replicas, validation_enabled):
        mgr = PolicyStatusManager()
        mgr.config = SimpleNamespace(
            validation=SimpleNamespace(enable=validation_enabled)
        )
        mgr.policy_atoms_in_replica = 1
        mgr.redis_handler = object()
        policy_replica = _replica("policy-0", ended=False, start_time=0)
        rollout_mgr = _RolloutMgrStub(replicas)

        with (
            patch(
                "cosmos_rl.dispatcher.command.PolicyToRolloutUnicastCommand.trigger"
            ) as p2r,
            patch(
                "cosmos_rl.dispatcher.command.RolloutToRolloutBroadcastCommand.trigger"
            ) as r2r,
        ):
            mgr.trigger_weight_sync(
                policy_replica, rollout_mgr, current_step=10, total_steps=10
            )
        return p2r, r2r

    def test_single_ended_replica_no_sync(self):
        """1 replica, ended, validation off -> early return, no P2R/R2R.

        This is the failing-test topology (single rollout replica, intra-
        replica TP): once it ends, the controller stops syncing entirely
        and the rollout stops via the controller's explicit STOP command.
        """
        p2r, r2r = self._run(
            [_replica("r0", ended=True, start_time=1)], validation_enabled=False
        )
        p2r.assert_not_called()
        r2r.assert_not_called()

    def test_mixed_excludes_only_ended(self):
        """Live + ended replicas, validation off -> sync targets only the
        live replica (ended dropped from P2R target and R2R recipients)."""
        live = _replica("r-live", ended=False, start_time=1)
        ended = _replica("r-ended", ended=True, start_time=0)
        p2r, r2r = self._run([live, ended], validation_enabled=False)
        p2r.assert_called_once()
        r2r.assert_called_once()
        # P2R unicast target is the (only) live replica.
        self.assertIs(p2r.call_args.kwargs["dst_replica"], live)
        # R2R recipient list excludes the ended replica.
        dst_replicas = r2r.call_args.kwargs["dst_replicas"]
        self.assertEqual(dst_replicas, [live])

    def test_ended_included_when_validation_enabled(self):
        """Validation on -> exclusion disabled; ended replica still synced
        (it stays alive to serve the final validation)."""
        ended = _replica("r-ended", ended=True, start_time=0)
        p2r, r2r = self._run([ended], validation_enabled=True)
        p2r.assert_called_once()
        r2r.assert_called_once()
        self.assertIs(p2r.call_args.kwargs["dst_replica"], ended)
        self.assertEqual(r2r.call_args.kwargs["dst_replicas"], [ended])

    def test_no_ended_unchanged_behavior(self):
        """No ended replicas, validation off -> all replicas synced
        (behaviour identical to before the exclusion)."""
        a = _replica("r-a", ended=False, start_time=0)
        b = _replica("r-b", ended=False, start_time=1)
        p2r, r2r = self._run([a, b], validation_enabled=False)
        p2r.assert_called_once()
        r2r.assert_called_once()
        self.assertIs(p2r.call_args.kwargs["dst_replica"], a)
        self.assertEqual(r2r.call_args.kwargs["dst_replicas"], [a, b])


class TestRolloutStatusAllEnded(unittest.TestCase):
    def test_empty_rollout_manager_is_not_all_ended(self):
        rsm = RolloutStatusManager()
        self.assertFalse(rsm.all_rollouts_ended())

    def test_all_ended_requires_at_least_one_rollout(self):
        rsm = RolloutStatusManager()
        rsm.rollout_replicas = {
            "r0": SimpleNamespace(status=SimpleNamespace(ended=True)),
            "r1": SimpleNamespace(status=SimpleNamespace(ended=True)),
        }
        self.assertTrue(rsm.all_rollouts_ended())


# ---------------------------------------------------------------------------
# Cross-replica R2R barrier (async_r2r_sync="generation"/"inference")
# ---------------------------------------------------------------------------
class _FakeRedis:
    """Minimal Redis stand-in for the R2R barrier counter + pub/sub."""

    def __init__(self, incr_result=1, get_result=None):
        self._incr_result = incr_result
        self._get_result = get_result if get_result is not None else incr_result
        self.published = []
        self.incr_calls = 0
        self._pubsub = _FakePubSub()

    def incr(self, key):
        self.incr_calls += 1
        return self._incr_result

    def expire(self, key, ttl):
        pass

    def get(self, key):
        return self._get_result

    def publish(self, channel, msg):
        self.published.append((channel, msg))

    def pubsub(self):
        return self._pubsub


class _FakePubSub:
    def __init__(self):
        self.messages = []

    def subscribe(self, channel):
        pass

    def get_message(self, timeout=None):
        return self.messages.pop(0) if self.messages else None

    def unsubscribe(self, channel):
        pass

    def close(self):
        pass


def _barrier_worker(redis, cached_world_size, mesh=None, stop_set=False):
    wst = SimpleNamespace(_stop=threading.Event())
    if stop_set:
        wst._stop.set()
    return SimpleNamespace(
        _r2r_redis=redis,
        _r2r_world_size=cached_world_size,
        _r2r_barrier_prefix="test:r2r",
        replica_name_to_rank=mesh if mesh is not None else {},
        _weight_sync_thread=wst,
    )


class TestR2RBarrierWorldSize(unittest.TestCase):
    """The generation-mode shutdown bug: ``_r2r_world_size`` is frozen at
    setup, so when a replica reaches end-of-data and the controller drops it
    from the broadcast set, the survivors kept waiting for a participant that
    will never arrive (120 s timeout per step -> wedged shutdown).  The fix
    drives the barrier off the controller's authoritative per-round count
    (``len(dst_replica_names)``), falling back to the live mesh size."""

    def test_expected_world_size_overrides_stale_cache(self):
        """With one replica ended, the controller ships a count of 6 while the
        cached size is the stale 7.  The 6th (last) arriver must publish "go"
        immediately -- proving the barrier used 6, not 7 (which would block)."""
        redis = _FakeRedis(incr_result=6)
        worker = _barrier_worker(redis, cached_world_size=7)
        r2r_barrier(worker, weight_step=80, expected_world_size=6)
        self.assertEqual(redis.published, [("test:r2r:go:80", "go")])

    def test_falls_back_to_live_mesh_when_no_expected(self):
        """No explicit count -> use the live mesh size (2), not the stale
        cached value (5).  Last arriver (count==2) publishes "go"."""
        redis = _FakeRedis(incr_result=2)
        worker = _barrier_worker(redis, cached_world_size=5, mesh={"a": 0, "b": 1})
        r2r_barrier(worker, weight_step=10)
        self.assertEqual(redis.published, [("test:r2r:go:10", "go")])

    def test_skips_when_single_participant(self):
        """world_size <= 1 -> no barrier at all (counter never touched)."""
        redis = _FakeRedis(incr_result=1)
        worker = _barrier_worker(redis, cached_world_size=5)
        r2r_barrier(worker, weight_step=1, expected_world_size=1)
        self.assertEqual(redis.incr_calls, 0)
        self.assertEqual(redis.published, [])

    def test_stop_event_aborts_wait(self):
        """Teardown backstop: a waiting (non-last) worker must abort the
        barrier wait when its WeightSyncThread is asked to stop, so
        ``wst.stop()`` / ``destroy_distributed()`` are never blocked for the
        full timeout.  Patch the timeout small so a regression fails fast
        instead of busy-looping for 120 s."""
        redis = _FakeRedis(incr_result=1, get_result=1)  # 1/2 -> waiter
        worker = _barrier_worker(redis, cached_world_size=2, stop_set=True)
        with patch.object(weight_sync, "_R2R_BARRIER_TIMEOUT_S", 2):
            r2r_barrier(worker, weight_step=5, expected_world_size=2)
        # Aborted as a waiter: never published "go".
        self.assertEqual(redis.published, [])


# ---------------------------------------------------------------------------
# Teardown: in-flight R2R broadcast abort (async_r2r_sync="generation")
# ---------------------------------------------------------------------------
class _FakeCudaEvent:
    """Stand-in for ``torch.cuda.Event`` that never completes (or always does)."""

    def __init__(self, completed):
        self._completed = completed

    def record(self, stream=None):
        pass

    def query(self):
        return self._completed


class TestNcclAbortAll(unittest.TestCase):
    """The teardown gap: an async R2R grouped broadcast is enqueued on the
    WeightSyncThread stream and pynccl only bounds the *enqueue* phase, so a
    broadcast whose peer departed hangs on the device with no watchdog.
    ``nccl_abort_all`` is the primitive that unwedges it at teardown."""

    def test_aborts_all_registered_and_empties_registry(self):
        from cosmos_rl.utils import pynccl

        # Registry is empty on CPU; register a few fakes.
        idxs = [
            pynccl._COMM_REGISTRY.register(object(), rank=r, world_size=4)
            for r in range(3)
        ]
        self.assertEqual(sorted(pynccl._COMM_REGISTRY.all_indices()), sorted(idxs))
        with patch.object(pynccl, "_nccl") as fake_nccl:
            n = pynccl.nccl_abort_all()
        self.assertEqual(n, 3)
        self.assertEqual(fake_nccl.ncclCommAbort.call_count, 3)
        # Registry fully drained -> a second call is a no-op.
        self.assertEqual(pynccl._COMM_REGISTRY.all_indices(), [])
        with patch.object(pynccl, "_nccl"):
            self.assertEqual(pynccl.nccl_abort_all(), 0)


class TestBoundedDrainOrAbort(unittest.TestCase):
    """The shared teardown guard used by both ``WeightSyncThread.stop()`` (its
    own stream) and the rollout ``work()`` teardown (inference_stream): bounded-
    wait for in-flight GPU work and, on timeout (a peer departed mid-collective),
    abort all NCCL comms so the subsequent sync / ``destroy_distributed`` cannot
    wedge.  Returns True if drained cleanly, False if it timed out + aborted."""

    def _drain(self, completed):
        from cosmos_rl.utils import pynccl

        with (
            patch(
                "cosmos_rl.utils.pynccl.torch.cuda.Event",
                lambda: _FakeCudaEvent(completed),
            ),
            patch("cosmos_rl.utils.pynccl.nccl_abort_all") as fake_abort,
        ):
            result = pynccl.bounded_drain_or_abort(
                object(), timeout_s=0.05, context="test"
            )
        return result, fake_abort

    def test_aborts_when_stream_never_drains(self):
        result, fake_abort = self._drain(completed=False)
        fake_abort.assert_called_once()
        self.assertFalse(result)

    def test_no_abort_when_stream_drains(self):
        result, fake_abort = self._drain(completed=True)
        fake_abort.assert_not_called()
        self.assertTrue(result)


class TestCleanupPayloadServer(unittest.TestCase):
    """Teardown stops the rollout's UCXX output server (``cleanup_ucxx``)
    before the NCCL abort, so a trainer still pulling this replica's final
    output cannot wedge ``ncclCommAbort``.  The hook is ``getattr``-guarded so
    backends without a UCXX server are unaffected."""

    def _call(self, rollout):
        fake_self = SimpleNamespace(rollout=rollout, replica_name="r-test")
        DisaggregatedRolloutControlWorker._cleanup_payload_server(fake_self)

    def test_calls_cleanup_when_present(self):
        rollout = SimpleNamespace(cleanup_ucxx=MagicMock())
        self._call(rollout)
        rollout.cleanup_ucxx.assert_called_once_with()

    def test_noop_when_backend_has_no_server(self):
        # Backend without cleanup_ucxx -> no error, nothing to call.
        self._call(SimpleNamespace())
        self._call(None)

    def test_swallows_cleanup_exception(self):
        rollout = SimpleNamespace(
            cleanup_ucxx=MagicMock(side_effect=RuntimeError("boom"))
        )
        # Must not propagate: teardown has to continue to the NCCL abort.
        self._call(rollout)
        rollout.cleanup_ucxx.assert_called_once_with()


class TestRolloutCheckoutGate(unittest.TestCase):
    """Root-cause fix: on all-policy-dead the controller must wait for rollout
    replicas to check out (post end) before self-SIGTERM, otherwise stragglers
    wedge on a dead controller and survivors orphan the final R2R broadcast.
    ``_await_rollout_checkout`` is the bounded gate."""

    @staticmethod
    def _controller(ended_flags):
        replicas = {
            f"r{i}": SimpleNamespace(status=SimpleNamespace(ended=bool(e)))
            for i, e in enumerate(ended_flags)
        }
        rsm = SimpleNamespace(
            rollout_replicas=replicas,
            maintain_life_status=MagicMock(),
        )
        rsm.all_rollouts_ended = lambda: all(
            r.status.ended for r in rsm.rollout_replicas.values()
        )
        return SimpleNamespace(
            rollout_status_manager=rsm, policy_status_manager=object()
        )

    def test_returns_true_when_all_already_ended(self):
        from cosmos_rl.dispatcher import run_web_panel

        ctrl = self._controller([True, True, True])
        ev = threading.Event()
        self.assertTrue(
            run_web_panel._await_rollout_checkout(
                ctrl, ev, timeout_s=5.0, scan_interval_s=0.01
            )
        )
        # All ended up front -> never had to pump life-status.
        ctrl.rollout_status_manager.maintain_life_status.assert_not_called()

    def test_returns_true_when_no_rollouts_exist(self):
        from cosmos_rl.dispatcher import run_web_panel

        ctrl = self._controller([])
        ev = threading.Event()
        self.assertTrue(
            run_web_panel._await_rollout_checkout(
                ctrl, ev, timeout_s=5.0, scan_interval_s=0.01
            )
        )
        ctrl.rollout_status_manager.maintain_life_status.assert_not_called()

    def test_waits_then_true_when_straggler_checks_out(self):
        from cosmos_rl.dispatcher import run_web_panel

        ctrl = self._controller([True, False])
        rsm = ctrl.rollout_status_manager

        # Straggler posts end on the first maintain_life_status pump.
        def _flip(_pol):
            rsm.rollout_replicas["r1"].status.ended = True

        rsm.maintain_life_status.side_effect = _flip
        ev = threading.Event()
        self.assertTrue(
            run_web_panel._await_rollout_checkout(
                ctrl, ev, timeout_s=5.0, scan_interval_s=0.01
            )
        )
        rsm.maintain_life_status.assert_called()

    def test_times_out_when_rollout_never_checks_out(self):
        from cosmos_rl.dispatcher import run_web_panel

        ctrl = self._controller([True, False])
        ev = threading.Event()
        # Deadline in the past -> forces the abnormal-path return without hanging.
        self.assertFalse(
            run_web_panel._await_rollout_checkout(
                ctrl, ev, timeout_s=0.0, scan_interval_s=0.01
            )
        )

    def test_external_shutdown_signal_breaks_wait(self):
        from cosmos_rl.dispatcher import run_web_panel

        ctrl = self._controller([False])
        ev = threading.Event()
        ev.set()  # already signaled -> wait() returns True immediately
        self.assertFalse(
            run_web_panel._await_rollout_checkout(
                ctrl, ev, timeout_s=5.0, scan_interval_s=0.01
            )
        )


class TestFinalValidationCompletion(unittest.TestCase):
    def test_validation_report_completion_clears_controller_state(self):
        data_fetcher = SimpleNamespace(
            val_datasize=2,
            val_dataloader=None,
            activated_val_iter=object(),
            activated_val_tqdm=SimpleNamespace(update=MagicMock()),
        )

        def _clear_validation_status():
            data_fetcher.activated_val_iter = None
            data_fetcher.activated_val_tqdm = None

        data_fetcher.clear_validation_status = MagicMock(
            side_effect=_clear_validation_status
        )
        psm = SimpleNamespace(
            val_report_data={},
            data_fetcher=data_fetcher,
            config=SimpleNamespace(
                validation=SimpleNamespace(n_generation=1),
                logging=SimpleNamespace(logger=[]),
            ),
            total_steps=2,
            custom_logger_fns=[],
            try_trigger_data_fetch_and_training=MagicMock(),
        )
        psm._expected_validation_rollout_count = (
            PolicyStatusManager._expected_validation_rollout_count.__get__(psm)
        )
        psm._reported_validation_rollout_count = (
            PolicyStatusManager._reported_validation_rollout_count.__get__(psm)
        )
        psm.validation_report_validation_results = (
            PolicyStatusManager.validation_report_validation_results.__get__(psm)
        )
        rollouts = [
            SimpleNamespace(reward=1.0, report_metrics=None),
            SimpleNamespace(reward=0.0, report_metrics=None),
        ]
        with patch("cosmos_rl.dispatcher.status.logger.info") as info:
            psm.validation_report_validation_results(
                validation_step=2,
                validation_results=[rollouts],
                rollout_status_manager=SimpleNamespace(),
            )
        data_fetcher.clear_validation_status.assert_called_once()
        self.assertIsNone(data_fetcher.activated_val_iter)
        psm.try_trigger_data_fetch_and_training.assert_called_once()
        self.assertTrue(
            any("Validation finished" in call.args[0] for call in info.call_args_list)
        )

    def test_validation_prompt_fetch_after_policy_shutdown_serves_active_validation(
        self,
    ):
        from cosmos_rl.dispatcher.controller import Controller

        class NoPolicyStatus:
            def __len__(self):
                return 0

        psm = NoPolicyStatus()
        data_fetcher = SimpleNamespace(
            activated_val_iter=object(),
            get_batched_prompt=MagicMock(return_value=(["payload"], False)),
        )
        controller = SimpleNamespace(
            policy_status_manager=psm, data_fetcher=data_fetcher
        )
        method = Controller._get_batched_prompt_impl.__get__(controller)
        payloads, is_end = asyncio.run(method(8, validation_step=2))
        self.assertEqual(payloads, ["payload"])
        self.assertFalse(is_end)
        data_fetcher.get_batched_prompt.assert_called_once_with(
            8,
            2,
            None,
            weight_version=None,
        )

    def test_validation_prompt_fetch_after_validation_exhausted_signals_end(self):
        from cosmos_rl.dispatcher.controller import Controller

        class NoPolicyStatus:
            def __len__(self):
                return 0

        controller = SimpleNamespace(
            policy_status_manager=NoPolicyStatus(),
            data_fetcher=SimpleNamespace(activated_val_iter=None),
        )
        method = Controller._get_batched_prompt_impl.__get__(controller)
        payloads, is_end = asyncio.run(method(8, validation_step=2))
        self.assertEqual(payloads, [])
        self.assertTrue(is_end)

    def test_non_validation_prompt_fetch_after_policy_shutdown_is_training_end(self):
        from cosmos_rl.dispatcher.controller import Controller

        class NoPolicyStatus:
            def __len__(self):
                return 0

        psm = NoPolicyStatus()
        controller = SimpleNamespace(policy_status_manager=psm)
        method = Controller._get_batched_prompt_impl.__get__(controller)
        payloads, is_end = asyncio.run(method(8))
        self.assertEqual(payloads, [])
        self.assertTrue(is_end)

    def test_finalize_logs_error_if_validation_is_still_active(self):
        from cosmos_rl.dispatcher import run_web_panel

        class EmptyPolicyStatus:
            def __len__(self):
                return 0

            def training_finished(self):
                return True

        class EmptyManager:
            def __len__(self):
                return 0

        original_controller = run_web_panel.controller
        original_server = run_web_panel.server
        server = SimpleNamespace(should_exit=False)
        psm = EmptyPolicyStatus()
        psm.data_fetcher = SimpleNamespace(activated_val_iter=object())
        run_web_panel.controller = SimpleNamespace(
            policy_status_manager=psm,
            rollout_status_manager=EmptyManager(),
            teacher_result_manager=set(),
            is_rl=True,
        )
        run_web_panel.server = server
        try:
            with patch("cosmos_rl.dispatcher.run_web_panel.logger.error") as error:
                self.assertTrue(run_web_panel._maybe_finalize("clean unregister of r0"))
        finally:
            run_web_panel.controller = original_controller
            run_web_panel.server = original_server
        self.assertIn("validation is still active", error.call_args.args[0])
        self.assertTrue(server.should_exit)


# ---------------------------------------------------------------------------
# Rollout -- fail-fast command poll after StopCommand
# ---------------------------------------------------------------------------
class TestQueryCommandStopsAfterStop(unittest.TestCase):
    """``query_command_from_controller`` must not block on a second XREAD after
    ``StopCommand`` is enqueued."""

    def test_stops_poll_loop_after_stop_command(self):
        stop_cmd = StopCommand("replica-0")
        subscribe_calls = []

        def fake_subscribe(replica_name):
            subscribe_calls.append(replica_name)
            return [stop_cmd.pack()]

        worker = SimpleNamespace(
            replica_name="replica-0",
            shutdown_signal=threading.Event(),
            redis_controller=SimpleNamespace(subscribe_command=fake_subscribe),
            _command_queue=Queue(),
        )
        DisaggregatedRolloutControlWorker.query_command_from_controller(worker)
        self.assertEqual(subscribe_calls, ["replica-0"])
        self.assertEqual(worker._command_queue.qsize(), 1)
        self.assertIsInstance(worker._command_queue.get(), StopCommand)


# ---------------------------------------------------------------------------
# Controller -- JobPhase end-of-data model
# ---------------------------------------------------------------------------
class TestJobPhaseEnterDraining(unittest.TestCase):
    def test_first_is_end_enters_draining_and_recomputes(self):
        recompute_args = []

        def _recompute(explicit_num_remaining_samples=None):
            recompute_args.append(explicit_num_remaining_samples)

        psm = SimpleNamespace(
            job_phase=JobPhase.RUNNING,
            total_pending_rollouts=lambda: 8,
            recompute_total_steps=_recompute,
        )
        psm.enter_draining_phase = PolicyStatusManager.enter_draining_phase.__get__(psm)
        psm.enter_draining_phase()
        self.assertEqual(psm.job_phase, JobPhase.DRAINING)
        self.assertEqual(recompute_args, [8])

    def test_second_enter_stays_draining(self):
        recompute_args = []

        def _recompute(explicit_num_remaining_samples=None):
            recompute_args.append(explicit_num_remaining_samples)

        psm = SimpleNamespace(
            job_phase=JobPhase.DRAINING,
            total_pending_rollouts=lambda: 4,
            recompute_total_steps=_recompute,
        )
        psm.enter_draining_phase = PolicyStatusManager.enter_draining_phase.__get__(psm)
        psm.enter_draining_phase()
        self.assertEqual(psm.job_phase, JobPhase.DRAINING)
        self.assertEqual(recompute_args, [])


class TestJobPhaseWeightSync(unittest.TestCase):
    def test_draining_suppresses_weight_sync(self):
        psm = SimpleNamespace(
            job_phase=JobPhase.DRAINING,
            total_steps=10,
            config=SimpleNamespace(
                train=SimpleNamespace(sync_weight_interval=1),
                validation=SimpleNamespace(enable=False, freq=None),
            ),
        )
        rsm = SimpleNamespace(
            all_rollouts_ended=lambda: False,
            rollout_replicas={},
        )
        psm.should_weight_sync_after_train_ack = (
            PolicyStatusManager.should_weight_sync_after_train_ack.__get__(psm)
        )
        self.assertFalse(psm.should_weight_sync_after_train_ack(2, rsm))

    def test_running_defers_to_need_weight_sync(self):
        psm = SimpleNamespace(
            job_phase=JobPhase.RUNNING,
            total_steps=10,
            config=SimpleNamespace(
                train=SimpleNamespace(sync_weight_interval=5),
                validation=SimpleNamespace(enable=False, freq=None),
            ),
        )
        ended = SimpleNamespace(status=SimpleNamespace(ended=True), start_time=1.0)
        live = SimpleNamespace(status=SimpleNamespace(ended=False), start_time=2.0)
        rsm = SimpleNamespace(
            all_rollouts_ended=lambda: False,
            rollout_replicas={"r0": ended, "r1": live},
            get_all_atoms_arrived_replicas=lambda: [ended, live],
        )
        psm.should_weight_sync_after_train_ack = (
            PolicyStatusManager.should_weight_sync_after_train_ack.__get__(psm)
        )
        psm._weight_sync_rollout_targets = (
            PolicyStatusManager._weight_sync_rollout_targets.__get__(psm)
        )
        self.assertTrue(psm.should_weight_sync_after_train_ack(5, rsm))

    def test_final_validation_sync_when_rollouts_already_ended(self):
        """Regression: final validation must not be skipped after prompt is_end."""
        ended = SimpleNamespace(status=SimpleNamespace(ended=True), start_time=1.0)
        psm = SimpleNamespace(
            job_phase=JobPhase.RUNNING,
            total_steps=2,
            config=SimpleNamespace(
                train=SimpleNamespace(sync_weight_interval=1),
                validation=SimpleNamespace(enable=True, freq=1),
            ),
        )
        rsm = SimpleNamespace(
            all_rollouts_ended=lambda: True,
            rollout_replicas={"r0": ended},
            get_all_atoms_arrived_replicas=lambda: [ended],
        )
        psm.should_weight_sync_after_train_ack = (
            PolicyStatusManager.should_weight_sync_after_train_ack.__get__(psm)
        )
        psm._weight_sync_rollout_targets = (
            PolicyStatusManager._weight_sync_rollout_targets.__get__(psm)
        )
        self.assertTrue(psm.should_weight_sync_after_train_ack(2, rsm))

    def test_non_final_sync_suppressed_when_rollouts_ended(self):
        ended = SimpleNamespace(status=SimpleNamespace(ended=True), start_time=1.0)
        psm = SimpleNamespace(
            job_phase=JobPhase.RUNNING,
            total_steps=2,
            config=SimpleNamespace(
                train=SimpleNamespace(sync_weight_interval=1),
                validation=SimpleNamespace(enable=True, freq=1),
            ),
        )
        rsm = SimpleNamespace(
            all_rollouts_ended=lambda: True,
            rollout_replicas={"r0": ended},
            get_all_atoms_arrived_replicas=lambda: [ended],
        )
        psm.should_weight_sync_after_train_ack = (
            PolicyStatusManager.should_weight_sync_after_train_ack.__get__(psm)
        )
        self.assertFalse(psm.should_weight_sync_after_train_ack(1, rsm))


class TestJobPhaseFinishDraining(unittest.TestCase):
    def test_finish_triggers_training_complete_when_buffer_empty(self):
        training_complete = []
        recompute_calls = []

        def _recompute(explicit_num_remaining_samples=None):
            recompute_calls.append(explicit_num_remaining_samples)
            psm.total_steps = psm.current_step

        def _trigger_training_complete():
            training_complete.append(True)

        psm = SimpleNamespace(
            total_steps=10,
            current_step=9,
            policy_replicas={},
            total_pending_rollouts=lambda: 0,
            all_ready_or_reduced=lambda: True,
            recompute_total_steps=_recompute,
            trigger_training_complete=_trigger_training_complete,
            rollout_buffer=SimpleNamespace(queue=SimpleNamespace(clear=lambda: None)),
            config=SimpleNamespace(train=SimpleNamespace(train_batch_per_replica=8)),
            get_all_atoms_arrived_replicas=lambda: [object()],
        )
        psm.finish_draining_phase = PolicyStatusManager.finish_draining_phase.__get__(
            psm
        )
        rsm = SimpleNamespace(all_rollouts_ended=lambda: True)
        psm.finish_draining_phase(rsm)
        self.assertEqual(recompute_calls, [0])
        self.assertEqual(psm.total_steps, 10)
        self.assertEqual(training_complete, [True])

    def test_finish_happen_to_finish_skips_when_buffer_empty(self):
        """When the buffer is already drained, no TrainingComplete is needed."""
        training_complete = []

        psm = SimpleNamespace(
            total_steps=3,
            current_step=3,
            total_pending_rollouts=lambda: 0,
            recompute_total_steps=lambda **kw: None,
            trigger_training_complete=lambda: training_complete.append(True),
            rollout_buffer=SimpleNamespace(queue=SimpleNamespace(clear=lambda: None)),
            config=SimpleNamespace(train=SimpleNamespace(train_batch_per_replica=8)),
            get_all_atoms_arrived_replicas=lambda: [object()],
        )
        psm.finish_draining_phase = PolicyStatusManager.finish_draining_phase.__get__(
            psm
        )
        psm.finish_draining_phase(SimpleNamespace(all_rollouts_ended=lambda: True))
        self.assertEqual(training_complete, [])

    def test_finish_buffered_rollouts_triggers_training_complete(self):
        """Regression: buffered rollouts at the final step must still dispatch."""
        cleared = []
        training_complete = []
        queue = SimpleNamespace(clear=lambda: cleared.append(True))

        psm = SimpleNamespace(
            total_steps=3,
            current_step=3,
            policy_replicas={},
            total_pending_rollouts=lambda: 8,
            all_ready_or_reduced=lambda: True,
            recompute_total_steps=lambda **kw: None,
            trigger_training_complete=lambda: training_complete.append(True),
            rollout_buffer=SimpleNamespace(queue=queue),
            config=SimpleNamespace(train=SimpleNamespace(train_batch_per_replica=8)),
            get_all_atoms_arrived_replicas=lambda: [object()],
        )
        psm.finish_draining_phase = PolicyStatusManager.finish_draining_phase.__get__(
            psm
        )
        psm.finish_draining_phase(SimpleNamespace(all_rollouts_ended=lambda: True))
        self.assertEqual(cleared, [True])
        self.assertEqual(psm.total_steps, 4)
        self.assertEqual(training_complete, [True])

    def test_finish_defers_when_buffer_can_fill_remaining_steps(self):
        """Do not fire TrainingComplete while a full step of rollouts waits."""
        training_complete = []

        psm = SimpleNamespace(
            total_steps=3,
            current_step=2,
            policy_replicas={},
            total_pending_rollouts=lambda: 16,
            all_ready_or_reduced=lambda: True,
            trigger_training_complete=lambda: training_complete.append(True),
            rollout_buffer=SimpleNamespace(
                queue=SimpleNamespace(
                    clear=lambda: (_ for _ in ()).throw(
                        AssertionError("buffer should not be cleared")
                    )
                )
            ),
            config=SimpleNamespace(train=SimpleNamespace(train_batch_per_replica=8)),
            get_all_atoms_arrived_replicas=lambda: [object()],
        )

        def _recompute(**_kw):
            psm.total_steps = 2

        psm.recompute_total_steps = _recompute
        psm.finish_draining_phase = PolicyStatusManager.finish_draining_phase.__get__(
            psm
        )
        psm.finish_draining_phase(SimpleNamespace(all_rollouts_ended=lambda: True))
        self.assertEqual(training_complete, [])


class TestJobPhaseValidationBypass(unittest.TestCase):
    def test_on_rollout_is_end_noop_when_validation_enabled(self):
        psm = SimpleNamespace(
            job_phase=JobPhase.RUNNING,
            config=SimpleNamespace(validation=SimpleNamespace(enable=True)),
            enter_draining_phase=lambda: (_ for _ in ()).throw(
                AssertionError("must not enter draining")
            ),
            finish_draining_phase=lambda rsm: (_ for _ in ()).throw(
                AssertionError("must not finish draining")
            ),
        )
        psm.on_rollout_is_end = PolicyStatusManager.on_rollout_is_end.__get__(psm)
        psm.on_rollout_is_end(SimpleNamespace())
        self.assertEqual(psm.job_phase, JobPhase.RUNNING)


class TestTrainingCompleteCommand(unittest.TestCase):
    def test_pack_depack_roundtrip(self):
        cmd = TrainingCompleteCommand(
            "policy-0",
            global_step=3,
            total_steps=3,
            remain_samples_num=0,
        )
        restored = Command.depack(cmd.pack())
        self.assertIsInstance(restored, TrainingCompleteCommand)
        self.assertEqual(restored.replica_name, "policy-0")
        self.assertEqual(restored.global_step, 3)
        self.assertEqual(restored.total_steps, 3)
        self.assertEqual(restored.command_type, "TRAINING_COMPLETE")

    def test_trigger_training_complete_dispatch(self):
        triggered = []
        replica = SimpleNamespace(
            name="policy-0",
            sub_profiler_config=SimpleNamespace(
                do_profile=False,
                active_steps=None,
                rank_filter=None,
                record_shape=None,
                profile_memory=None,
                with_stack=None,
                with_modules=None,
            ),
        )

        def fake_trigger(**kwargs):
            triggered.append(kwargs)

        psm = SimpleNamespace(
            data_fetcher=SimpleNamespace(activated_val_iter=None),
            current_step=2,
            total_steps=3,
            remain_samples_num=0,
            dispatched_rollouts_by_step={},
            config=SimpleNamespace(
                validation=SimpleNamespace(enable=False, freq=1),
                train=SimpleNamespace(
                    ckpt=SimpleNamespace(
                        enable_checkpoint=False, save_freq=1, save_freq_in_epoch=None
                    ),
                    epoch=1,
                ),
            ),
            redis_handler=object(),
            get_all_atoms_arrived_replicas=lambda: [replica],
            training_finished=lambda: False,
            check_checkpoint_saving=lambda n: False,
            set_status=lambda name, status: None,
        )
        psm.trigger_training_complete = (
            PolicyStatusManager.trigger_training_complete.__get__(psm)
        )
        with patch.object(TrainingCompleteCommand, "trigger", fake_trigger):
            psm.trigger_training_complete()
        self.assertEqual(psm.current_step, 3)
        self.assertEqual(psm.dispatched_rollouts_by_step[3], 0)
        self.assertEqual(len(triggered), 1)
        self.assertEqual(triggered[0]["global_step"], 3)

    def test_finish_draining_calls_training_complete(self):
        calls = []

        def _trigger():
            calls.append(True)

        psm = SimpleNamespace(
            total_steps=10,
            current_step=9,
            policy_replicas={},
            total_pending_rollouts=lambda: 0,
            all_ready_or_reduced=lambda: True,
            recompute_total_steps=lambda **kw: setattr(
                psm, "total_steps", psm.current_step
            ),
            trigger_training_complete=_trigger,
            rollout_buffer=SimpleNamespace(queue=SimpleNamespace(clear=lambda: None)),
            config=SimpleNamespace(train=SimpleNamespace(train_batch_per_replica=8)),
            get_all_atoms_arrived_replicas=lambda: [object()],
        )
        psm.finish_draining_phase = PolicyStatusManager.finish_draining_phase.__get__(
            psm
        )
        psm.finish_draining_phase(SimpleNamespace(all_rollouts_ended=lambda: True))
        self.assertEqual(calls, [True])

    def test_finish_draining_defers_while_policy_in_flight(self):
        calls = []

        psm = SimpleNamespace(
            total_steps=1,
            current_step=1,
            policy_replicas={"p0": SimpleNamespace(status="running")},
            total_pending_rollouts=lambda: 8,
            all_ready_or_reduced=lambda: False,
            recompute_total_steps=lambda **kw: None,
            trigger_training_complete=lambda: calls.append(True),
            rollout_buffer=SimpleNamespace(queue=SimpleNamespace(clear=lambda: None)),
            config=SimpleNamespace(train=SimpleNamespace(train_batch_per_replica=8)),
            get_all_atoms_arrived_replicas=lambda: [object()],
        )
        psm.finish_draining_phase = PolicyStatusManager.finish_draining_phase.__get__(
            psm
        )
        psm.finish_draining_phase(SimpleNamespace(all_rollouts_ended=lambda: True))
        self.assertEqual(calls, [])


class TestOnRolloutIsEndSequence(unittest.TestCase):
    def test_enter_once_finish_only_when_all_ended(self):
        enter_count = []
        finish_count = []

        psm = SimpleNamespace(
            job_phase=JobPhase.RUNNING,
            config=SimpleNamespace(validation=SimpleNamespace(enable=False)),
            enter_draining_phase=lambda: (
                enter_count.append(1),
                setattr(psm, "job_phase", JobPhase.DRAINING),
            ),
            finish_draining_phase=lambda rsm: finish_count.append(
                rsm.all_rollouts_ended()
            ),
        )
        psm.on_rollout_is_end = PolicyStatusManager.on_rollout_is_end.__get__(psm)

        rsm_partial = SimpleNamespace(all_rollouts_ended=lambda: False)
        psm.on_rollout_is_end(rsm_partial)
        self.assertEqual(enter_count, [1])
        self.assertEqual(finish_count, [False])
        self.assertEqual(psm.job_phase, JobPhase.DRAINING)

        rsm_all = SimpleNamespace(all_rollouts_ended=lambda: True)
        psm.on_rollout_is_end(rsm_all)
        self.assertEqual(enter_count, [1])
        self.assertEqual(finish_count, [False, True])


class TestTrainAckDuringPartialDrain(unittest.TestCase):
    def test_train_ack_suppresses_weight_sync_while_draining(self):
        psm = SimpleNamespace(
            job_phase=JobPhase.DRAINING,
            total_steps=10,
            config=SimpleNamespace(
                train=SimpleNamespace(sync_weight_interval=1),
                validation=SimpleNamespace(enable=False, freq=None),
            ),
        )
        rsm = SimpleNamespace(
            all_rollouts_ended=lambda: False,
            rollout_replicas={
                "r0": SimpleNamespace(status=SimpleNamespace(ended=True)),
                "r1": SimpleNamespace(status=SimpleNamespace(ended=True)),
            },
        )
        psm.should_weight_sync_after_train_ack = (
            PolicyStatusManager.should_weight_sync_after_train_ack.__get__(psm)
        )
        self.assertFalse(psm.should_weight_sync_after_train_ack(2, rsm))


class TestEndOfDataRecomputeUsesBufferOnly(unittest.TestCase):
    """Regression: ``enter_draining_phase`` uses ``total_pending_rollouts`` only."""

    def test_recompute_uses_buffer_count_on_first_is_end(self):
        from cosmos_rl.dispatcher import run_web_panel

        recompute_args = []

        def _recompute(explicit_num_remaining_samples=None):
            recompute_args.append(explicit_num_remaining_samples)

        psm = SimpleNamespace(
            job_phase=JobPhase.RUNNING,
            config=SimpleNamespace(validation=SimpleNamespace(enable=False)),
            total_steps=10,
            current_step=10,
            total_pending_rollouts=lambda: 8,
            recompute_total_steps=_recompute,
            finish_draining_phase=lambda rsm: None,
        )
        psm.enter_draining_phase = PolicyStatusManager.enter_draining_phase.__get__(psm)
        psm.on_rollout_is_end = PolicyStatusManager.on_rollout_is_end.__get__(psm)
        rsm = SimpleNamespace(
            rollout_end=lambda name: None,
            all_rollouts_ended=lambda: False,
        )
        fake_controller = SimpleNamespace(
            rollout_status_manager=rsm,
            policy_status_manager=psm,
        )
        req = SimpleNamespace(
            is_end=True, src_replica_name="r0", payloads=[], metrics={}
        )
        with patch.object(run_web_panel, "controller", fake_controller):
            asyncio.run(run_web_panel.put_rollout_group(req))
        self.assertEqual(recompute_args, [8])
        self.assertEqual(psm.job_phase, JobPhase.DRAINING)


if __name__ == "__main__":
    unittest.main()
