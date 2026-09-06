# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""The rollout global mesh must not commit to a collective it cannot finish.

``ncclCommInitRank`` is a collective over a fixed membership snapshot, and the
rebuild that calls it is triggered BY membership changing.  If any member of
the snapshot departs before reaching its own call, every survivor blocks in the
collective -- alive, still heartbeating, with the GIL released so no signal
handler runs.  A whole allocation is lost with no error and no traceback.

Three defences are covered here:

* the collective is skipped when the job can never use it -- decided by the
  CONTROLLER and carried on the command, because the same predicate evaluated
  independently per worker can disagree, and a collective that only some ranks
  enter is itself the hang;
* mesh RANKS are still assigned either way, because data dispatch reads them;
* when the collective is built, the wait is bounded and the resulting error
  names the membership it was waiting on.

The handler-registration cases exist because a previous revision of this fix
inserted a method between the ``@register_rollout_command_handler`` decorator
and ``build_global_mesh``, silently re-pointing ``BuildMeshCommand`` at the
wrong function.  Every behavioural test still passed, because they all call
``build_global_mesh`` directly and never go through the registry.
"""

import importlib.util
import os
import pathlib
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cosmos_rl.comm.base import CommMixin
import msgpack

from cosmos_rl.dispatcher.command import (
    BuildMeshCommand,
    Command,
    CommandType,
)
from cosmos_rl.dispatcher.status import RolloutStatusManager
from cosmos_rl.rollout.worker.rollout_control import DisaggregatedRolloutControlWorker
from cosmos_rl.utils import constant
from cosmos_rl.utils import pynccl_wrapper


@pytest.fixture(autouse=True)
def _restore_warn_once_flag():
    """``_warned_no_init_config`` is process-global and CI runs files together.

    Leaving it set would silently disable the warning for any later test in the
    same interpreter, and such a test would pass because nothing warned --
    indistinguishable from passing because the warning was right.
    """
    saved = pynccl_wrapper.NCCLLibrary._warned_no_init_config
    yield
    pynccl_wrapper.NCCLLibrary._warned_no_init_config = saved


def _worker(replica_name: str = "rollout-0"):
    """A stub carrying only what ``build_global_mesh`` actually reads.

    ``_mesh_rebuild_ready`` is a real Event, not None: the real worker assigns
    a fresh one immediately before enqueueing the command, and that is the only
    path that delivers a ``BuildMeshCommand``.  Modelling it as None would take
    the ``is not None`` false branch on every case and leave
    ``mesh_ready.set()`` unexercised -- and failing to set it wedges the
    command loop, which is the same class of hang this change exists to remove.
    """
    return SimpleNamespace(
        state=SimpleNamespace(prompt_consume_end=lambda: False),
        parallel_dims=SimpleNamespace(world_size=1),
        replica_name=replica_name,
        _weight_sync_thread=None,
        _mesh_rebuild_ready=threading.Event(),
        get_group_unique_key=lambda mapping: "rollout_mesh_key",
        query_nccl_unique_id_from_controller=MagicMock(return_value=[1, 2, 3]),
        api_client=SimpleNamespace(
            post_nccl_comm_initiator=MagicMock(),
            post_nccl_comm_acceptor=MagicMock(return_value=[1, 2, 3]),
            post_nccl_comm_error=MagicMock(),
        ),
    )


# A two-member mesh: the second member is what makes the collective real.  The
# single-member case returns before any of this.
_TWO_MEMBERS = {"rollout-0": 0, "rollout-1": 1}


def _mesh_command(mesh_is_used: bool) -> BuildMeshCommand:
    return BuildMeshCommand(dict(_TWO_MEMBERS), mesh_is_used=mesh_is_used)


class TestHandlerStaysRegistered:
    """The decorator must still be attached to the function that handles it."""

    def test_build_mesh_command_dispatches_to_build_global_mesh(self):
        handler = CommMixin.get_rollout_command_handler(BuildMeshCommand)
        assert handler is DisaggregatedRolloutControlWorker.build_global_mesh

    def test_registered_handler_accepts_the_command_argument(self):
        # The dispatcher calls handler(self, command).  A handler taking only
        # self raises TypeError there; the worker then never sets
        # _mesh_rebuild_ready, so its command loop spins forever and even STOP
        # is never delivered.
        handler = CommMixin.get_rollout_command_handler(BuildMeshCommand)
        worker = _worker()

        with (
            patch("cosmos_rl.rollout.worker.rollout_control.create_nccl_comm"),
            patch("cosmos_rl.rollout.worker.rollout_control.create_nccl_uid"),
        ):
            handler(worker, _mesh_command(mesh_is_used=False))

        assert worker._mesh_rebuild_ready.is_set()


class TestControllerDecidesWhetherTheMeshIsUsed:
    """One decision per rebuild, taken where the whole topology is visible.

    Evaluating this per worker is unsafe: the controller mutates
    ``policy.parallelism.n_init_replicas`` in place as policy replicas register
    and each rollout reads the config independently at startup, so two workers
    can disagree -- which is exactly the partial-participation hang.
    """

    def _manager(self, configured: int):
        manager = object.__new__(RolloutStatusManager)
        manager.config = SimpleNamespace(
            policy=SimpleNamespace(
                parallelism=SimpleNamespace(n_init_replicas=configured)
            )
        )
        return manager

    def test_no_policy_replicas_configured_or_registered(self):
        manager = self._manager(configured=0)
        assert manager._rollout_mesh_is_used() is False

    def test_configured_but_not_yet_registered_still_counts(self):
        # Weights are coming; the mesh is needed before they arrive.
        manager = self._manager(configured=2)
        assert manager._rollout_mesh_is_used() is True

    def test_decision_is_carried_on_the_command(self):
        manager = self._manager(configured=0)
        manager.redis_handler = MagicMock()
        manager.data_fetcher = MagicMock()
        replica = SimpleNamespace(
            name="rollout-0",
            start_time=0,
            all_atoms_arrived=True,
            status=SimpleNamespace(mesh_rank=None),
        )

        with patch.object(BuildMeshCommand, "trigger") as trigger:
            manager.trigger_rebuild_mesh([replica])

        assert trigger.call_args.kwargs["mesh_is_used"] is False
        # Mesh-size bookkeeping is NOT conditional: data dispatch needs it even
        # when no communicator is built.
        manager.data_fetcher.set_rollout_global_mesh_size.assert_called_once_with(1)

    def test_the_decision_survives_the_wire(self):
        # It is only authoritative if it actually reaches the workers; the
        # command is msgpack-packed through Redis.
        packed = _mesh_command(mesh_is_used=False).pack()
        from cosmos_rl.dispatcher.command import Command

        assert Command.depack(packed).mesh_is_used is False


class TestSkippingTheCollective:
    def test_unused_mesh_skips_the_communicator_and_the_uid_exchange(self):
        worker = _worker()

        with (
            patch(
                "cosmos_rl.rollout.worker.rollout_control.create_nccl_comm"
            ) as create_comm,
            patch(
                "cosmos_rl.rollout.worker.rollout_control.create_nccl_uid"
            ) as create_uid,
        ):
            DisaggregatedRolloutControlWorker.build_global_mesh(
                worker, _mesh_command(mesh_is_used=False)
            )

        create_comm.assert_not_called()
        # Posting a uid nobody consumes would leave a peer that DID participate
        # waiting on a rendezvous that never completes.
        create_uid.assert_not_called()
        worker.api_client.post_nccl_comm_initiator.assert_not_called()
        worker.query_nccl_unique_id_from_controller.assert_not_called()

    def test_skipping_still_assigns_mesh_ranks(self):
        """Data dispatch reads ``rank_in_rollout_repicas``, not the comm.

        ``data_dispatch_as_rank_in_mesh`` selects prompts by
        ``prompt_idx % mesh_size == rank_in_mesh``.  Suppressing the whole
        BuildMeshCommand -- rather than only its collective -- would silently
        break prompt dispatch.
        """
        worker = _worker(replica_name="rollout-1")

        with (
            patch("cosmos_rl.rollout.worker.rollout_control.create_nccl_comm"),
            patch("cosmos_rl.rollout.worker.rollout_control.create_nccl_uid"),
        ):
            DisaggregatedRolloutControlWorker.build_global_mesh(
                worker, _mesh_command(mesh_is_used=False)
            )

        assert worker.rank_in_rollout_repicas == 1
        assert worker.replica_name_to_rank == _TWO_MEMBERS

    def test_skipping_releases_the_command_loop(self):
        # Every return path must set the event; the loop waiting on it has no
        # exit other than a STOP it cannot receive while blocked.
        worker = _worker()

        with (
            patch("cosmos_rl.rollout.worker.rollout_control.create_nccl_comm"),
            patch("cosmos_rl.rollout.worker.rollout_control.create_nccl_uid"),
        ):
            DisaggregatedRolloutControlWorker.build_global_mesh(
                worker, _mesh_command(mesh_is_used=False)
            )

        assert worker._mesh_rebuild_ready.is_set()

    def test_a_command_without_the_field_still_builds(self):
        # An older controller does not send mesh_is_used.  Building a mesh
        # nobody uses costs a communicator; skipping one somebody needs wedges
        # the job, so an absent field must mean "build".
        worker = _worker()
        legacy = SimpleNamespace(replica_name_to_rank=dict(_TWO_MEMBERS))

        with (
            patch(
                "cosmos_rl.rollout.worker.rollout_control.create_nccl_comm",
                return_value=11,
            ) as create_comm,
            patch(
                "cosmos_rl.rollout.worker.rollout_control.create_nccl_uid",
                return_value=[1, 2, 3],
            ),
        ):
            DisaggregatedRolloutControlWorker.build_global_mesh(worker, legacy)

        create_comm.assert_called_once()


class TestBuildingTheCollective:
    def _build(self, create_comm, worker=None):
        worker = worker or _worker()
        with (
            patch(
                "cosmos_rl.rollout.worker.rollout_control.create_nccl_comm", create_comm
            ),
            patch(
                "cosmos_rl.rollout.worker.rollout_control.create_nccl_uid",
                return_value=[1, 2, 3],
            ),
        ):
            DisaggregatedRolloutControlWorker.build_global_mesh(
                worker, _mesh_command(mesh_is_used=True)
            )
        return worker

    def test_used_mesh_builds_the_communicator_and_releases_the_loop(self):
        create_comm = MagicMock(return_value=11)
        worker = self._build(create_comm)

        create_comm.assert_called_once()
        assert worker.global_commnicator_idex == 11
        assert worker._mesh_rebuild_ready.is_set()

    def test_the_wait_is_bounded(self):
        """An unset budget resolves to COSMOS_NCCL_TIMEOUT_MS -- ten minutes.

        That much silence per rebuild cannot be told apart from a wedged job.
        """
        create_comm = MagicMock(return_value=11)
        self._build(create_comm)

        assert (
            create_comm.call_args.kwargs["timeout_ms"]
            == constant.COSMOS_ROLLOUT_MESH_BUILD_TIMEOUT_MS
        )

    def test_timeout_is_reported_but_does_not_kill_the_replica(self):
        """The deadline means a member never arrived -- a topology problem.

        The CONTROLLER fixes that by issuing another rebuild, so dying here
        throws away the thing that would have recovered the replica, and each
        death is another departure that kills the next survivor. Same cascade
        the weight-sync fence used to cause one line above. The policy side
        already reports-and-returns; this matches it.
        """
        create_comm = MagicMock(side_effect=TimeoutError("enqueue timed out"))
        worker = _worker()

        # Must not raise.
        self._build(create_comm, worker=worker)

        # No communicator, and the command loop is released either way.
        assert worker.global_commnicator_idex == -1
        assert worker._mesh_rebuild_ready.is_set()

        # The controller is told, so it can rebuild.
        worker.api_client.post_nccl_comm_error.assert_called_once()
        reported = worker.api_client.post_nccl_comm_error.call_args.args[1]
        message = str(reported)
        # The deadline expiring IS the diagnosis, so it carries the evidence:
        # without the membership an operator cannot tell a departed member
        # from a fabric fault.
        assert "rollout-0" in message and "rollout-1" in message
        assert "world_size=2" in message
        assert str(constant.COSMOS_ROLLOUT_MESH_BUILD_TIMEOUT_MS) in message
        assert isinstance(reported.__cause__, TimeoutError)

    def test_a_non_timeout_failure_is_not_relabelled_as_a_departed_member(self):
        # pynccl re-raises the real cause rather than synthesising a timeout,
        # so a fabric or argument error must propagate as itself instead of
        # being reported as a peer that never arrived.
        create_comm = MagicMock(
            side_effect=RuntimeError("NCCL: asynchronous error 2 reported")
        )

        with pytest.raises(RuntimeError) as excinfo:
            self._build(create_comm)

        assert "departed" not in str(excinfo.value)
        assert "asynchronous error" in str(excinfo.value)


class TestTimeoutResolution:
    """The call site must not quietly take the operator's knob away.

    ``_get_timeout_ms`` returns an explicit caller value verbatim, so passing
    one OVERRIDES ``COSMOS_NCCL_TIMEOUT_MS`` instead of tightening it.
    """

    def _resolve(self, env):
        keys = ("COSMOS_ROLLOUT_MESH_BUILD_TIMEOUT_MS", "COSMOS_NCCL_TIMEOUT_MS")
        with patch.dict(os.environ, env, clear=False):
            for key in keys:
                if key not in env:
                    os.environ.pop(key, None)
            return constant._resolve_rollout_mesh_build_timeout_ms()

    def test_default_is_tighter_than_the_nccl_default(self):
        assert self._resolve({}) < 600000

    def test_default_clears_the_measured_worst_case_with_margin(self):
        # Measured on this hardware: 44 mesh builds, worst case 10s at 7
        # replicas and 9s at 5, so rank count barely moves it. 60s is ~6x that.
        # Safe only because the deadline is not fatal -- see the constant's
        # docstring. Anything at or below the observed worst case would kill
        # healthy replicas.
        assert self._resolve({}) >= 30000
        assert self._resolve({}) < 300000

    def test_an_operator_raised_nccl_timeout_wins(self):
        assert self._resolve({"COSMOS_NCCL_TIMEOUT_MS": "900000"}) == 900000

    def test_an_explicit_mesh_budget_wins_over_both(self):
        resolved = self._resolve(
            {
                "COSMOS_NCCL_TIMEOUT_MS": "900000",
                "COSMOS_ROLLOUT_MESH_BUILD_TIMEOUT_MS": "45000",
            }
        )
        assert resolved == 45000


class TestUnboundedInitFallbackIsAnnounced:
    """``ncclCommInitRank`` is the one creation path that cannot be bounded.

    ``ncclCommInitRankConfig`` is optional, and without it creation blocks
    until every rank joins.  The abort watchdog is not armed for creation --
    the communicator it would abort does not exist yet -- so a peer that never
    arrives wedges the thread for the life of the process.  Degrading to that
    silently from a debug line leaves an operator with no way to know.
    """

    def _library_without_init_config(self):
        library = object.__new__(pynccl_wrapper.NCCLLibrary)
        library._funcs = {}  # no ncclCommInitRankConfig symbol
        library.ncclCommInitRank = MagicMock(return_value="comm")
        return library

    def test_fallback_warns_and_still_creates_the_communicator(self):
        pynccl_wrapper.NCCLLibrary._warned_no_init_config = False
        library = self._library_without_init_config()

        with patch.object(pynccl_wrapper, "logger") as log:
            comm = library.ncclCommInitRankConfig(2, pynccl_wrapper.ncclUniqueId(), 0)

        assert comm == "comm"
        library.ncclCommInitRank.assert_called_once()
        assert log.warning.call_count == 1
        assert "unbounded" in log.warning.call_args.args[0].lower()

    def test_fallback_warning_is_emitted_once_per_process(self):
        # One line per communicator would bury it; every rebuild creates one.
        pynccl_wrapper.NCCLLibrary._warned_no_init_config = False
        library = self._library_without_init_config()

        with patch.object(pynccl_wrapper, "logger") as log:
            for _ in range(3):
                library.ncclCommInitRankConfig(2, pynccl_wrapper.ncclUniqueId(), 0)

        assert log.warning.call_count == 1
        assert library.ncclCommInitRank.call_count == 3


class TestTheFieldReachesTheWorkerSafely:
    """The controller's decision is worthless if it does not survive the wire,
    and dangerous if it breaks the workers that receive it.
    """

    def test_trigger_forwards_the_decision(self):
        # The seam between the decision and the command. Dropping this kwarg
        # would make every worker see the default and silently turn the whole
        # guard into a no-op, with every test still green.
        replica = SimpleNamespace(
            name="rollout-0",
            start_time=0,
            all_atoms_arrived=True,
            status=SimpleNamespace(mesh_rank=None),
        )
        redis = MagicMock()

        BuildMeshCommand.trigger([replica], redis_handler=redis, mesh_is_used=False)

        packed = redis.publish_command.call_args.args[0]
        assert Command.depack(packed).mesh_is_used is False

    def test_default_is_true_so_an_older_controller_still_builds(self):
        # Backward compatibility rests entirely on this default: an old packed
        # dict has no key, from_dict passes none, the ctor supplies True.
        assert BuildMeshCommand({"rollout-0": 0}).mesh_is_used is True
        revived = BuildMeshCommand.from_dict(
            {
                "replica_name_to_rank": {"rollout-0": 0},
                "scope": 0,
                "command_type": CommandType.BUILD_MESH,
                "uuid_value": "u",
            }
        )
        assert revived.mesh_is_used is True

    def test_the_default_is_omitted_from_the_wire(self):
        """A new key is a hard break for an older worker.

        ``from_dict`` does ``cls(**dict_v)``, so a surplus kwarg reaches
        ``Command.__init__`` as a TypeError -- raised by ``depack`` outside the
        command loop's try, killing the worker's command thread so it stops
        receiving everything including STOP. Omitting the default keeps the
        wire identical for every job that has policy replicas.
        """
        default_keys = set(msgpack.unpackb(BuildMeshCommand({"r": 0}).pack()))
        assert "mesh_is_used" not in default_keys

        explicit = msgpack.unpackb(
            BuildMeshCommand({"r": 0}, mesh_is_used=False).pack()
        )
        assert explicit["mesh_is_used"] is False

    def test_omitting_it_does_not_lose_the_meaning(self):
        for value in (True, False):
            command = BuildMeshCommand({"r": 0}, mesh_is_used=value)
            assert Command.depack(command.pack()).mesh_is_used is value


class TestBroadcastWithoutAMeshFailsLoudly:
    """A skipped mesh leaves comm_idx at -1, and every R2R path funnels here.

    Guarding one of the three call sites is not enough: the async path
    swallows exceptions as a generic task failure, so an unguarded -1 becomes
    a bare KeyError and the replica runs on, silently never syncing weights.
    """

    def test_grouped_broadcast_refuses_a_missing_communicator(self):
        from cosmos_rl.rollout.worker import weight_sync

        worker = SimpleNamespace(
            rank_in_rollout_repicas=0,
            replica_name_to_rank={"rollout-0": 0},
            global_commnicator_idex=-1,
        )

        with pytest.raises(RuntimeError) as excinfo:
            weight_sync.do_nccl_broadcast_grouped(worker, "rollout-0", None)

        assert "no global mesh communicator" in str(excinfo.value)


class TestMalformedEnvDoesNotBreakImport:
    """The budget is resolved at import time, so a bad value must not be fatal.

    ``COSMOS_NCCL_TIMEOUT_MS=`` (exported empty) is a common shell habit, and
    raising here would take down every process -- controller included --
    before any NCCL work exists. pynccl parses the same variable lazily and
    tolerates it, so being stricter earlier would be a regression.
    """

    @pytest.mark.parametrize("bad", ["", "   ", "not-a-number", "10min"])
    def test_a_bad_value_falls_back_to_the_default(self, bad):
        with patch.dict(os.environ, {"COSMOS_NCCL_TIMEOUT_MS": bad}, clear=False):
            os.environ.pop("COSMOS_ROLLOUT_MESH_BUILD_TIMEOUT_MS", None)
            from_bad_value = constant._resolve_rollout_mesh_build_timeout_ms()

        # A bad value must resolve to exactly what NO value resolves to, and
        # that has to be computed with the environment cleared rather than
        # captured at import: a module-level snapshot would silently inherit an
        # ambient COSMOS_NCCL_TIMEOUT_MS and make this pass for the wrong
        # reason. Comparing to a literal would just re-encode the constant.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COSMOS_NCCL_TIMEOUT_MS", None)
            os.environ.pop("COSMOS_ROLLOUT_MESH_BUILD_TIMEOUT_MS", None)
            assert from_bad_value == constant._resolve_rollout_mesh_build_timeout_ms()


class TestReplicaLossDoesNotCascade:
    """A departed peer must not make the rebuild fatal to the survivors.

    Measured before this: killing ONE rollout replica produced twelve
    "Weight-sync work did not drain before rollout mesh rebuild" failures and
    six unregistrations, and the job wedged. The chain is:

    * the departing peer makes an in-flight R2R raise, latching _task_failed;
    * fence() therefore returns False and latches _fence_failed;
    * _fence_failed short-circuits every later fence() BEFORE the drain, so
      the rebuild is refused without even attempting one;
    * build_global_mesh raised, killing the survivor, which triggered the next
      rebuild, which killed the next survivor.

    The rebuild is the recovery. It must not be vetoed by the failure it
    exists to recover from.
    """

    def _worker_with_failed_sync(self):
        worker = _worker()
        wst = MagicMock()
        wst.fence.return_value = False
        worker._weight_sync_thread = wst
        return worker, wst

    def test_a_failed_fence_does_not_kill_the_survivor(self):
        worker, wst = self._worker_with_failed_sync()

        with (
            patch(
                "cosmos_rl.rollout.worker.rollout_control.create_nccl_comm",
                return_value=7,
            ) as create_comm,
            patch(
                "cosmos_rl.rollout.worker.rollout_control.create_nccl_uid",
                return_value=[1, 2, 3],
            ),
        ):
            DisaggregatedRolloutControlWorker.build_global_mesh(
                worker, _mesh_command(mesh_is_used=True)
            )

        wst.reset_for_rebuild.assert_called_once_with()
        create_comm.assert_called_once()
        assert worker.global_commnicator_idex == 7
        assert worker._mesh_rebuild_ready.is_set()

    def test_a_clean_fence_discards_nothing(self):
        worker = _worker()
        wst = MagicMock()
        wst.fence.return_value = True
        worker._weight_sync_thread = wst

        with (
            patch(
                "cosmos_rl.rollout.worker.rollout_control.create_nccl_comm",
                return_value=7,
            ),
            patch(
                "cosmos_rl.rollout.worker.rollout_control.create_nccl_uid",
                return_value=[1, 2, 3],
            ),
        ):
            DisaggregatedRolloutControlWorker.build_global_mesh(
                worker, _mesh_command(mesh_is_used=True)
            )

        wst.reset_for_rebuild.assert_not_called()


class TestResetForRebuildClearsTheLatch:
    """``_fence_failed`` is sticky, and that stickiness is what cascaded."""

    def _thread(self):
        from cosmos_rl.rollout.worker.weight_sync import WeightSyncThread

        wst = object.__new__(WeightSyncThread)
        wst._worker = SimpleNamespace(replica_name="rollout-0")
        wst._fence_failed = True
        wst._task_failed = True
        wst._fenced_seq = 3
        return wst

    def test_it_reports_and_clears_a_latched_failure(self):
        from cosmos_rl.rollout.worker.weight_sync import WeightSyncThread

        wst = self._thread()
        with patch.object(WeightSyncThread, "fence", return_value=True) as fence:
            had_failure = wst.reset_for_rebuild()

        assert had_failure is True
        fence.assert_called_once_with()
        assert wst._fence_failed is False
        assert wst._task_failed is False

    def test_the_latch_is_cleared_before_the_drain(self):
        # fence() short-circuits on _fence_failed and returns False without
        # draining, so clearing afterwards would report failure having done
        # no work at all -- which is exactly the bug.
        from cosmos_rl.rollout.worker.weight_sync import WeightSyncThread

        wst = self._thread()
        seen = {}

        def _record_state(*args, **kwargs):
            seen["fence_failed"] = wst._fence_failed
            seen["task_failed"] = wst._task_failed
            return True

        with patch.object(WeightSyncThread, "fence", _record_state):
            wst.reset_for_rebuild()

        assert seen == {"fence_failed": False, "task_failed": False}

    def test_a_failing_drain_still_leaves_the_thread_usable(self):
        # fence() aborts NCCL on its timeout paths, so the old communicator is
        # already dead. Staying latched would only refuse the next rebuild too.
        from cosmos_rl.rollout.worker.weight_sync import WeightSyncThread

        wst = self._thread()

        def _failing_fence(*args, **kwargs):
            # The real fence() ends with `self._fence_failed = not result`, so
            # a mock that merely returns False leaves nothing to clear and the
            # assertions below would hold no matter what reset does.
            wst._fence_failed = True
            return False

        with patch.object(WeightSyncThread, "fence", _failing_fence):
            had_failure = wst.reset_for_rebuild()

        assert had_failure is True
        assert wst._fence_failed is False
        assert wst._task_failed is False


class TestWaitCommReadyIsBounded:
    """The trainer must not spin forever waiting for a mesh that failed.

    ``__execute_build_mesh`` reports a build failure and RETURNS rather than
    raising, leaving ``is_comm_ready`` cleared. Every caller that took
    ``wait_comm_ready``'s default then span in an unbounded
    ``while not set: sleep(0.1)`` -- silently, with no diagnostic.

    ``broadcast()`` makes it worse by calling ``get_replica_rank`` (which takes
    the default) BEFORE ``__do_nccl_op_with_retry``, so the hang sat one frame
    above the timeout-and-retry machinery written to handle exactly this.
    """

    def _comm(self):
        from cosmos_rl.utils.distributed import HighAvailabilitylNccl

        comm = object.__new__(HighAvailabilitylNccl)
        comm.is_comm_ready = threading.Event()
        comm.is_comm_ready.clear()
        comm.default_timeout_ms = 150
        comm.replica_name = "policy-0"
        comm.global_rank = 0
        # Real attribute, read by __log_prefix when the timeout message is
        # built; omitting it would make these tests fail on a missing stub
        # rather than on the behaviour under test.
        comm.replica_name_to_rank = {}
        return comm

    def test_the_default_wait_times_out_instead_of_spinning(self):
        comm = self._comm()
        started = time.monotonic()

        with pytest.raises(TimeoutError):
            comm.wait_comm_ready()

        # Bounded by default_timeout_ms, not unbounded.
        assert time.monotonic() - started < 5.0

    def test_an_explicit_timeout_is_still_honoured(self):
        comm = self._comm()
        with pytest.raises(TimeoutError):
            comm.wait_comm_ready(timeout=0.05)

    def test_a_ready_comm_returns_immediately(self):
        comm = self._comm()
        comm.is_comm_ready.set()
        comm.wait_comm_ready()  # must not raise


class TestPolicyMeshCollectiveIsBounded:
    """The policy mesh has the same shape as the rollout mesh and the same risk.

    It is a collective over a controller snapshot, issued by a rebuild that is
    triggered BY membership changing, so a policy replica that departs before
    reaching its own call blocks the rest. Unlike the rollout side the failure
    is caught and reported rather than fatal, but an unset budget still means
    ten minutes of silence first.
    """

    def _comm(self):
        from cosmos_rl.utils.distributed import HighAvailabilitylNccl

        comm = object.__new__(HighAvailabilitylNccl)
        comm.replica_name = "policy-0"
        comm.replica_name_to_rank = {}
        comm.global_rank = 0
        comm.comm_idx = -1
        comm.is_single_peer = threading.Event()
        comm.is_comm_ready = threading.Event()
        comm.is_first_time_build_mesh = True
        comm.api_client = MagicMock()
        comm.api_client.post_nccl_comm_acceptor.return_value = [1, 2, 3]
        return comm

    def test_build_mesh_passes_an_explicit_timeout(self):
        from cosmos_rl.utils import distributed as dist_mod

        comm = self._comm()
        command = BuildMeshCommand({"policy-0": 1, "policy-1": 0})

        with patch.object(dist_mod, "create_nccl_comm", return_value=5) as create_comm:
            dist_mod.HighAvailabilitylNccl._HighAvailabilitylNccl__execute_build_mesh(
                comm, command
            )

        assert (
            create_comm.call_args.kwargs["timeout_ms"]
            == constant.COSMOS_ROLLOUT_MESH_BUILD_TIMEOUT_MS
        )
        assert comm.is_comm_ready.is_set()


class TestEveryCommunicatorHandshakeIsBounded:
    """No ``create_nccl_comm`` call site may omit ``timeout_ms``.

    Structural rather than behavioural, deliberately. Two of the call sites
    live in modules that cannot be exercised here -- ``trtllm_worker`` needs
    tensorrt_llm, which is why nothing under tests/ has ever imported it -- and
    an unbounded handshake is the exact defect this change exists to remove, so
    it should not be able to reappear in a module the suite happens not to
    reach.

    Parsed with ``ast`` from the INSTALLED package rather than imported, so it
    covers modules whose dependencies are absent and works in CI, which has no
    source tree.

    ``payload_transport`` is excluded: it creates two-rank comms under its own
    rendezvous, retry, quarantine and cold-start machinery.
    """

    EXCLUDED = {"comm_cache.py"}

    def _package_root(self):
        spec = importlib.util.find_spec("cosmos_rl")
        assert spec is not None and spec.origin, "cosmos_rl is not importable"
        return pathlib.Path(spec.origin).resolve().parent

    def _call_sites(self):
        import ast

        sites = []
        for path in self._package_root().rglob("*.py"):
            if path.name in self.EXCLUDED:
                continue
            try:
                tree = ast.parse(path.read_text(errors="ignore"))
            except SyntaxError:  # pragma: no cover - not our code to fix
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                name = getattr(fn, "id", None) or getattr(fn, "attr", None)
                if name != "create_nccl_comm":
                    continue
                kwargs = {kw.arg for kw in node.keywords}
                sites.append((path.name, node.lineno, "timeout_ms" in kwargs))
        return sites

    def test_call_sites_are_found_at_all(self):
        # A rename would otherwise make the check below vacuously pass.
        sites = self._call_sites()
        assert len(sites) >= 5, f"expected several call sites, found {sites}"

    def test_no_call_site_omits_the_timeout(self):
        unbounded = [
            f"{name}:{line}"
            for name, line, bounded in self._call_sites()
            if not bounded
        ]
        assert not unbounded, (
            "these create_nccl_comm call sites pass no timeout_ms and would "
            f"inherit the 10-minute default: {unbounded}"
        )


class TestTrtllmMirrorsTheVllmContract:
    """The trtllm backend registers its own BuildMeshCommand handler.

    It therefore does NOT inherit any of the vLLM worker's protections, and it
    cannot be imported here (tensorrt_llm is absent), so this reads the
    installed source. Structural, but it is the only thing standing between a
    refactor and a silently unbounded second backend.
    """

    def _source(self):
        spec = importlib.util.find_spec("cosmos_rl")
        path = (
            pathlib.Path(spec.origin).resolve().parent
            / "rollout"
            / "trtllm_rollout"
            / "trtllm_worker.py"
        )
        assert path.is_file(), f"trtllm_worker.py not found at {path}"
        return path.read_text(errors="ignore")

    def test_it_honours_the_controller_decision(self):
        assert 'getattr(build_mesh_command, "mesh_is_used", True)' in self._source()

    def test_it_guards_the_broadcast_against_a_missing_communicator(self):
        # Without this, a skipped mesh hands -1 to the comm registry and raises
        # a bare KeyError instead of saying what is wrong.
        assert "self.global_commnicator_idex < 0" in self._source()
