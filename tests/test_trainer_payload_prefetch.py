# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock
import threading

import pytest

from cosmos_rl.dispatcher.command import DataFetchCommand
from cosmos_rl.dispatcher.status import PolicyStatus
from cosmos_rl.policy.trainer.prefetch import (
    TrainerPayloadPrefetch,
    prefetch_fallback_reason,
)
from cosmos_rl.utils.payload_transport.prefetch_mixin import PrefetchDataPackerMixin
import test_terminal_drain_protocol as drain_fixture


def test_real_communicator_lock_supports_lifecycle_scope(monkeypatch, packer):
    from contextlib import nullcontext
    import torch
    from cosmos_rl.policy.trainer.prefetch import payload_cohort_scope
    from cosmos_rl.utils import distributed

    # Construct the production lock, not a test-supplied RLock. No background
    # mesh commands or GPU operations are needed to expose non-reentrancy.
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    comm = distributed.HighAvailabilitylNccl("p0", 0, Mock())
    comm.is_comm_ready.set()
    comm.comm_idx = 1
    comm.replica_name_to_rank = {"p0": 0, "p1": 1}
    comm.max_retry = 1
    native = Mock()
    monkeypatch.setattr(distributed, "nccl_allreduce", native)
    monkeypatch.setattr(
        distributed, "nccl_timeout_watchdog", lambda **kwargs: nullcontext()
    )
    worker = SimpleNamespace(
        payload_prefetch=TrainerPayloadPrefetch(packer), inter_policy_nccl=comm
    )
    with payload_cohort_scope(worker):
        # Fail promptly on a plain Lock instead of hanging the regression.
        assert comm.build_mesh_lock.acquire(blocking=False), (
            "mesh lock is not reentrant"
        )
        comm.build_mesh_lock.release()
        value = torch.zeros(1)
        comm.allreduce(value, value, torch.distributed.ReduceOp.MAX)
    native.assert_called_once()


def test_native_receive_waits_for_allocation_stream_before_write(monkeypatch):
    import torch
    import cosmos_rl.utils.pynccl as pynccl
    from cosmos_rl.utils.payload_transport.nccl import strategy
    from cosmos_rl.utils.payload_transport.nccl.comm_cache import CommCache
    from test_nccl_payload_pairing import _consumer, _consumer_ref, _schema

    cache = CommCache(build_fn=lambda u, r: 55, abort_fn=lambda c: None)
    ref = _consumer_ref("a", _schema())
    pair = strategy._pair_key(ref, 0)
    cache.get_or_create(pair, uid_chars=[1], local_rank=1)
    consumer = _consumer(cache, warm_pairs={pair})
    consumer._streams = SimpleNamespace(acquire=lambda: "receive-stream")
    order = []

    def allocate(ref, module):
        order.append("allocate")
        return 55, strategy._alloc_recv_buffer(ref["layout"], None)

    def record(stream=None):
        order.append(("record", stream))
        return SimpleNamespace(
            name="allocation-ready" if stream is None else "receive-done",
            query=lambda: True,
        )

    def receive(*args, **kwargs):
        assert order == [
            "allocate",
            ("record", None),
            ("wait", "receive-stream", "allocation-ready"),
        ]
        order.append("receive")

    consumer._rendezvous_one = allocate
    monkeypatch.setattr(strategy, "record_event", record)
    monkeypatch.setattr(
        strategy,
        "wait_event",
        lambda stream, event: order.append(("wait", stream, event.name)),
    )
    decoded = torch.tensor([1])
    monkeypatch.setattr(
        strategy, "_verify_and_unpack", lambda *a, **kw: {"value": decoded}
    )
    monkeypatch.setattr(pynccl, "nccl_recv", receive)
    result, _, _ = consumer._fetch_all([(0, ref)])
    assert result[0]["value"] is decoded
    result.release()


class Base:
    def get_policy_input(
        self, sample=None, rollout_output=None, n_ignore_prefix_tokens=0, **kwargs
    ):
        return rollout_output


class Packer(PrefetchDataPackerMixin, Base):
    def _should_intercept(self, value):
        return isinstance(value, str)

    def _cache_key(self, value):
        return value

    def _fetch_batch(self, tasks):
        return {ref: ref + "-payload" for _, ref in tasks}


@pytest.fixture
def packer():
    result = Packer()
    result._setup_prefetch(prefetch_timeout=5)
    try:
        yield result
    finally:
        result.shutdown_prefetch()


def cmd(step, *, current=None, next_=None):
    return DataFetchCommand(
        "policy",
        1,
        step,
        10,
        10 - step,
        prefetched_batch_id=current,
        prefetch_next_batch_id=next_,
    )


@pytest.mark.parametrize("fail_fence", [False, True])
def test_payload_fetch_stream_fences_before_future_readiness(
    monkeypatch, packer, fail_fence
):
    from contextlib import contextmanager
    import torch

    fence_entered, finish = threading.Event(), threading.Event()
    order = []
    training_thread = threading.get_ident()

    def synchronize():
        order.append("fence")
        assert packer._prefetch_timers  # watchdog covers the final copy/decode fence
        fence_entered.set()
        assert finish.wait(2)
        if fail_fence:
            raise RuntimeError("copy fence failed")

    stream = SimpleNamespace(synchronize=synchronize)
    allocate_stream = Mock(return_value=stream)
    monkeypatch.setattr(torch.cuda, "Stream", allocate_stream)

    @contextmanager
    def scope(selected):
        assert selected is stream
        assert threading.get_ident() != training_thread
        order.append("enter")
        try:
            yield
        finally:
            order.append("exit")

    monkeypatch.setattr(torch.cuda, "stream", scope)
    original = packer._fetch_batch

    def fetch(tasks):
        assert order == ["enter"]
        order.append("fetch")
        return original(tasks)

    monkeypatch.setattr(packer, "_fetch_batch", fetch)
    pipeline = TrainerPayloadPrefetch(packer, device=torch.device("cuda:1"))
    allocate_stream.assert_called_once_with(device=torch.device("cuda:1"))
    packer._prefetch_cache = {"current": "still-owned"}
    try:
        pipeline.submit_next(cmd(1, next_="next"), lambda: ["b"])
        future = pipeline.pending.future
        assert fence_entered.wait(2)
        assert not future.done()
        assert packer._prefetch_cache == {"current": "still-owned"}
        finish.set()
        if fail_fence:
            with pytest.raises(RuntimeError, match="copy fence failed"):
                future.result(timeout=2)
            assert packer._prefetch_cache == {"current": "still-owned"}
        else:
            future.result(timeout=2)
            pipeline.take(cmd(2, current="next"), lambda: pytest.fail("refetched"))
            assert packer._prefetch_cache == {"b": "b-payload"}
        assert order == ["enter", "fetch", "fence", "exit"]
    finally:
        finish.set()


@pytest.mark.parametrize("count", [0, 1, 2, 5])
def test_exact_once_warmup_rotation_and_final_drain(packer, count):
    pipeline = TrainerPayloadPrefetch(packer)
    source = iter([str(i)] for i in range(count))
    dispatch = Mock(side_effect=lambda: next(source))
    consumed = []
    for index in range(count):
        command = cmd(
            index + 1,
            current=str(index) if index else None,
            next_=str(index + 1) if index + 1 < count else None,
        )
        rollouts, _ = pipeline.take(command, dispatch)
        pipeline.submit_next(command, dispatch)
        with packer.payload_batch_scope(rollouts):
            # Existing custom trainers' start/wait pairs cannot collect B as A.
            packer.start_prefetch(rollouts)
            packer.wait_prefetch()
            consumed.append(packer.get_policy_input(rollout_output=rollouts[0]))
        pipeline.complete(index + 1)
    pipeline.drain()
    assert consumed == [str(i) + "-payload" for i in range(count)]
    assert dispatch.call_count == count
    assert pipeline.pending is None


def test_fetch_next_overlaps_current_compute_without_cache_replacement(packer):
    pipeline = TrainerPayloadPrefetch(packer)
    first, _ = pipeline.take(cmd(1), lambda: ["a"])
    entered, finish = threading.Event(), threading.Event()
    original = packer._fetch_batch

    def fetch(tasks):
        entered.set()
        assert finish.wait(2)
        return original(tasks)

    packer._fetch_batch = fetch
    try:
        pipeline.submit_next(cmd(1, next_="b-id"), lambda: ["b"])
        assert entered.wait(2)
        with packer.payload_batch_scope(first):
            assert packer.get_policy_input(rollout_output="a") == "a-payload"
            assert not pipeline.pending.future.done()
            packer.start_prefetch(first)
            packer.wait_prefetch()
        pipeline.complete(1)
    finally:
        finish.set()
    second, _ = pipeline.take(cmd(2, current="b-id"), Mock())
    assert second == ("b",)
    assert packer.get_policy_input(rollout_output="b") == "b-payload"


def test_pending_batch_blocks_completion_and_wrong_identity(packer):
    pipeline = TrainerPayloadPrefetch(packer)
    pipeline.take(cmd(1), lambda: ["a"])
    pipeline.submit_next(cmd(1, next_="b-id"), lambda: ["b"])
    pipeline.complete(1)
    with pytest.raises(RuntimeError, match="before admitted"):
        pipeline.drain()
    with pytest.raises(ValueError, match="does not own"):
        pipeline.take(cmd(2, current="wrong-id"), Mock())
    with pytest.raises(ValueError, match="duplicate"):
        pipeline.take(cmd(1), Mock())
    pipeline.take(cmd(2, current="b-id"), Mock())


def manager(count=8):
    result, replica = drain_fixture.TestTerminalMatrix._manager(count)
    config = result.config
    result.total_steps = 10
    config.train.prefetch_payloads = True
    config.train.sync_weight_interval = 4
    config.train.train_policy.type = "grpo"
    config.train.train_policy.on_policy = False
    config.train.train_policy.allowed_outdated_steps = 10
    config.train.train_policy.coalesce_weight_sync = False
    config.distillation = SimpleNamespace(enable=False)
    config.policy = SimpleNamespace(
        parallelism=SimpleNamespace(tp_size=1, cp_size=1, pp_size=1)
    )
    config.custom = {"payload_transfer": "nccl"}
    replica.atoms = {
        "rank0": SimpleNamespace(global_rank=0, report_session_id="test-session")
    }
    return result, replica


def advance(status, replica):
    # Isolate dispatch from ACK reporting; separate protocol tests exercise ACKs.
    status.dispatched_rollouts_by_step.clear()
    status.status[replica.name] = PolicyStatus.READY
    status.try_trigger_data_fetch_and_training()


def test_controller_reserves_b_without_training_accounting_then_consumes_it():
    status, replica = manager()
    status.try_trigger_data_fetch_and_training()
    assert status.current_step == 1 and status.remain_samples_num == 18
    identity = status._payload_lookahead[0]
    assert status.samples_on_the_fly == 8  # Admission is not a train ACK.
    assert status.total_pending_rollouts() == 6
    assert status.rollout_buffer.qsize() == 4
    assert list(status.training_dispatches) == [1]
    advance(status, replica)
    assert status.current_step == 2 and status.remain_samples_num == 16
    assert status._payload_lookahead[0] != identity
    assert status.rollout_buffer.qsize() == 2
    assert status.total_pending_rollouts() == 4


@pytest.mark.parametrize("boundary", ["validation", "horizon", "on_policy"])
def test_barriers_do_not_reserve_a_future_batch(boundary):
    status, _ = manager()
    if boundary == "checkpoint":
        status.config.train.ckpt.enable_checkpoint = True
    elif boundary == "weight":
        status.current_step = 3
    elif boundary == "validation":
        status.config.validation.enable = True
        status.data_fetcher.validation_activate_dataloader = lambda step: setattr(
            status.data_fetcher, "activated_val_iter", object()
        )
    elif boundary == "horizon":
        status.total_steps = 1
    elif boundary == "on_policy":
        status.config.train.train_policy.on_policy = True
    status.try_trigger_data_fetch_and_training()
    assert status._payload_lookahead is None
    assert status.rollout_buffer.qsize() == 6


@pytest.mark.parametrize("sync_interval", [1, 3])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_weight_sync_and_checkpoint_do_not_block_input_admission(
    sync_interval, checkpoint
):
    status, _ = manager()
    status.config.train.sync_weight_interval = sync_interval
    status.current_step = sync_interval - 1
    status.config.train.ckpt.enable_checkpoint = checkpoint
    status.try_trigger_data_fetch_and_training()
    assert status._payload_lookahead is not None


def test_requested_stop_keeps_admitted_b_and_cleans_only_unissued_tail():
    status, replica = manager()
    status.policy_init_done = True
    status.try_trigger_data_fetch_and_training()
    identity = status._payload_lookahead[0]
    assert status.request_stop("stop at completed work")
    assert status._payload_lookahead[0] == identity
    assert status.rollout_buffer.empty()
    advance(status, replica)
    assert status.current_step == 2 and status._payload_lookahead is None


def test_membership_change_cannot_reassign_prefetched_work():
    status, replica = manager()
    status.try_trigger_data_fetch_and_training()
    replica.name = "replacement"
    status.status = {replica.name: PolicyStatus.READY}
    with pytest.raises(RuntimeError, match="membership changed"):
        advance(status, replica)


def test_disabled_configuration_does_not_require_new_config_fields():
    assert (
        prefetch_fallback_reason(SimpleNamespace(train=SimpleNamespace())) == "disabled"
    )


def test_unexpected_checkpoint_defers_reserved_fetch(packer):
    pipeline = TrainerPayloadPrefetch(packer)
    pipeline.take(cmd(1), lambda: ["a"])
    fetch = Mock(wraps=packer._fetch_batch)
    packer._fetch_batch = fetch
    pipeline.submit_next(cmd(1, next_="b-id"), lambda: ["b"], defer_fetch=True)
    assert pipeline.pending.future is None
    assert fetch.call_count == 0
    packer.finish_payload_batch()
    pipeline.complete(1)
    result, _ = pipeline.take(cmd(2, current="b-id"), Mock())
    assert result == ("b",) and fetch.call_count == 1


def test_rejected_reference_does_not_refetch_on_training_thread(packer):
    packer._fetch_batch = lambda tasks: {}
    packer._sync_fetch = Mock(side_effect=AssertionError("unbounded fallback"))
    pipeline = TrainerPayloadPrefetch(packer)
    rollouts, _ = pipeline.take(cmd(1), lambda: ["missing"])
    with packer.payload_batch_scope(rollouts):
        assert packer.get_policy_input(rollout_output="missing") is None
    packer._sync_fetch.assert_not_called()


def test_finish_releases_current_owner_without_consuming_next(packer):
    pipeline = TrainerPayloadPrefetch(packer)
    pipeline.take(cmd(1), lambda: ["a"])
    pipeline.submit_next(cmd(1, next_="b"), lambda: ["b"])
    future = pipeline.pending.future
    old_cache = packer._prefetch_cache
    packer.finish_payload_batch()
    assert packer._prefetch_cache == {}
    assert old_cache == {}  # A completed Future must not retain A at final use.
    assert packer._prepared_prefetch_future is future
    pipeline.complete(1)
    pipeline.take(cmd(2, current="b"), Mock())
    assert packer.get_policy_input(rollout_output="b") == "b-payload"


def test_optimizer_observation_never_changes_custom_training():
    import torch
    from cosmos_rl.policy.trainer.prefetch import observe_optimizer_steps

    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    trainer = SimpleNamespace(optimizer=optimizer)
    with observe_optimizer_steps(trainer) as report:
        parameter.backward()
        optimizer.step()
    assert report == {
        "prefetch/optimizer_counter_available": 1,
        "prefetch/optimizer_0_steps": 1,
    }
    trainer.payload_prefetch_optimizers = Mock(side_effect=ValueError("custom layout"))
    with observe_optimizer_steps(trainer) as report:
        optimizer.step()
    assert report == {"prefetch/optimizer_counter_available": 0}


def test_cohort_scope_pins_reserved_batch_membership(packer):
    from cosmos_rl.policy.trainer.prefetch import payload_cohort_scope

    pipeline = TrainerPayloadPrefetch(packer)
    ready = threading.Event()
    ready.set()
    comm = SimpleNamespace(
        build_mesh_lock=threading.RLock(),
        is_comm_ready=ready,
        comm_idx=1,
        replica_name_to_rank={"p0": 0, "p1": 1},
    )
    worker = SimpleNamespace(payload_prefetch=pipeline, inter_policy_nccl=comm)
    with payload_cohort_scope(worker):
        pipeline.take(cmd(1), lambda: ["a"])
        pipeline.submit_next(cmd(1, next_="b"), lambda: ["b"])
        pipeline.complete(1)
    comm.comm_idx = 2
    with pytest.raises(RuntimeError, match="changed with an admitted batch"):
        with payload_cohort_scope(worker):
            pytest.fail("must not train under another communicator")
    assert pipeline.failed


def test_discard_protects_both_current_and_reserved_payloads(monkeypatch):
    from cosmos_rl.dispatcher.status import (
        PayloadTransportRegistry,
        PolicyStatusManager,
    )

    status, _ = manager(6)
    status.try_trigger_data_fetch_and_training()
    current = list(status._payload_training_rollouts)
    reserved = status._payload_lookahead[2]
    assert len(current) == len(reserved) == 2
    cleanup = Mock(return_value=False)
    monkeypatch.setattr(PayloadTransportRegistry, "handle_discarded", cleanup)
    PolicyStatusManager._publish_payload_transport_cleanup(
        status, [current[0], reserved[0]], []
    )
    assert cleanup.call_args.args[1] == [*current, *reserved]


def test_lookahead_rechecks_the_next_preupdate_version():
    status, _ = manager(4)
    status.current_step = 1
    status.config.train.train_policy.allowed_outdated_steps = 1
    status.try_trigger_data_fetch_and_training()
    # Version zero is valid for step two (pre-update version one), but not
    # eligible for a reservation for step three (pre-update version two).
    assert status.current_step == 2
    assert status._payload_lookahead is None
    assert status.rollout_buffer.empty()
    assert status.filter_records["outdated"] == 2
    assert status.samples_on_the_fly == 2
