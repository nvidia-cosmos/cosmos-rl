# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Independent control delivery while the stock training handler is running."""

import asyncio
import copy
from queue import Queue, Empty
import threading
from types import SimpleNamespace

import msgpack
import pytest
import torch

from cosmos_rl.dispatcher.command import (
    Command,
    DataFetchCommand,
    PayloadPrefetchCommand,
)
from cosmos_rl.dispatcher.data.schema import Rollout
from cosmos_rl.policy.trainer.prefetch import TrainerPayloadPrefetch
from dispatch_commit_canary import worker as make_worker
from test_trainer_payload_prefetch import Packer, manager, cmd


def notification(step=2, identity="batch-b", refs=("b",)):
    return PayloadPrefetchCommand(
        "policy-0",
        identity,
        step,
        (("policy-0", ((0, "test-session"),)),),
        [
            Rollout(prompt_idx=i, completion=ref).model_dump()
            for i, ref in enumerate(refs)
        ],
    )


def test_synchronous_take_wakes_future_notification(monkeypatch):
    packer = Packer()
    packer._setup_prefetch(prefetch_timeout=5)
    pipeline = TrainerPayloadPrefetch(packer)
    waiting, admitted = threading.Event(), threading.Event()
    failures = []
    original_wait = pipeline._condition.wait

    def wait(timeout=None):
        waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(pipeline._condition, "wait", wait)
    pipeline.take(cmd(1), lambda: [Rollout(prompt_idx=0, completion="a")])
    pipeline.complete(1)
    packer.finish_payload_batch()

    def notify():
        try:
            pipeline.notify(
                notification(3, "third", ("c",)),
                (Rollout(prompt_idx=2, completion="c"),),
            )
            admitted.set()
        except Exception as error:
            failures.append(error)

    reader = threading.Thread(target=notify)
    reader.start()
    try:
        assert waiting.wait(1)
        # No pending batch: this must still wake the reader when active_step
        # advances, rather than waiting the entire admission timeout.
        pipeline.take(cmd(2), lambda: [Rollout(prompt_idx=1, completion="b")])
        assert admitted.wait(1), "notification slept until its timeout"
        assert not failures
        assert pipeline.pending.step == 3
    finally:
        pipeline.stop_notifications()
        reader.join(2)
        packer.finish_payload_batch()
        packer.shutdown_prefetch()
        assert not reader.is_alive()


@pytest.fixture
def worker():
    instance, acks = make_worker(torch.device("cpu"), None)
    instance.device = torch.device("cpu")
    instance.train_stream = None
    instance.api_client._report_session_id = "test-session"
    instance.inter_policy_nccl = SimpleNamespace(allreduce=lambda *_args, **_kw: None)
    instance.config.train.train_policy.data_dispatch_as_rank_in_mesh = False
    instance.data_packer = Packer()
    instance.data_packer._setup_prefetch(prefetch_timeout=5)
    instance.payload_prefetch = TrainerPayloadPrefetch(instance.data_packer)
    instance.shutdown_signal = threading.Event()
    instance.fetch_command_buffer = Queue()
    instance.kv_store = SimpleNamespace(
        broadcast_command_bounded=lambda command, **_: command
    )
    wire = Queue()

    def subscribe(_name):
        try:
            return [wire.get(timeout=0.02)]
        except Empty:
            return []

    instance.redis_controller = SimpleNamespace(subscribe_command=subscribe)
    failures = []

    def read():
        try:
            asyncio.run(instance.fetch_command())
        except Exception as error:
            failures.append(error)

    reader = threading.Thread(target=read)
    instance.fetch_command_thread = reader
    reader.start()
    try:
        yield instance, acks, wire, failures
    finally:
        instance.payload_prefetch.stop_notifications()
        instance.shutdown_signal.set()
        reader.join(2)
        assert not reader.is_alive()
        instance.data_packer.shutdown_prefetch()


@pytest.mark.parametrize("sync_interval", [1, 3])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_late_batch_fetches_during_stock_training_without_early_ack(
    worker, sync_interval, checkpoint
):
    instance, acks, wire, reader_failures = worker
    status, _ = manager(4)
    status.total_steps = 3
    status.config.train.sync_weight_interval = sync_interval
    status.config.train.ckpt.enable_checkpoint = checkpoint
    status.config.rollout = SimpleNamespace(include_stop_str_in_output=False)
    initial = [status.rollout_buffer.get_nowait() for _ in range(4)]
    for rollout in initial:
        rollout.completion = f"sample-{rollout.prompt_idx}"
    for rollout in initial[:2]:
        status.rollout_buffer.put_nowait(rollout)

    def publish(plan):
        for _stream, kind, raw in plan.entries:
            if kind == "rollout":
                instance.data_queue.put_nowait(
                    Rollout.model_validate(msgpack.unpackb(raw))
                )
            else:
                wire.put_nowait(raw)

    status.redis_handler.publish_plan = publish
    running, finish, fetching_b = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    original_train, original_fetch = (
        instance.trainer.step_training,
        instance.data_packer._fetch_batch,
    )

    def fetch(tasks):
        if any(ref == "sample-2" for _, ref in tasks):
            fetching_b.set()
        return original_fetch(tasks)

    def train(**kwargs):
        result = original_train(**kwargs)
        if kwargs["current_step"] == 1:
            running.set()
            assert finish.wait(3)
        return result

    instance.data_packer._fetch_batch = fetch
    instance.trainer.step_training = train
    status.try_trigger_data_fetch_and_training()
    first = instance.fetch_command_buffer.get(timeout=2)
    assert isinstance(first, DataFetchCommand) and first.prefetch_next_batch_id is None
    errors = []

    def execute():
        try:
            instance.execute_data_fetch(first)
        except Exception as error:
            errors.append(error)

    training = threading.Thread(target=execute)
    training.start()
    try:
        assert running.wait(2)
        assert not acks and not fetching_b.is_set()
        for rollout in initial[2:]:
            status.put_rollout(rollout)
        assert fetching_b.wait(2)
        assert training.is_alive() and not acks
        assert status.current_step == 1 and status.remain_samples_num == 18
        assert status.samples_on_the_fly == 4
        assert instance.fetch_command_buffer.empty()  # No ordinary command N+1.
        assert set(instance.data_packer._prefetch_cache) == {"sample-0", "sample-1"}
        assert instance.trainer.updates == instance.trainer.scheduler_calls == 1
        snapshot = instance.trainer.weight.detach().clone()
    finally:
        finish.set()
        training.join(3)
    assert not training.is_alive() and not errors and not reader_failures
    assert len(acks) == 1 and acks[0][1] == 1
    # Dispatch the next ordinary command through real controller accounting.
    # Snapshot/update ordering is separately exercised by weight-publication
    # controls; this test verifies receipt alone did not perform update N+1.
    from test_trainer_payload_prefetch import advance

    advance(status, status.policy_replicas[instance.replica_name])
    second = instance.fetch_command_buffer.get(timeout=2)
    assert second.prefetched_batch_id == instance.payload_prefetch.pending.identity
    torch.testing.assert_close(snapshot, instance.trainer.weight)
    instance.execute_data_fetch(second)
    assert (
        len(acks) == 2
        and instance.trainer.updates == instance.trainer.scheduler_calls == 2
    )
    assert instance.payload_prefetch.pending is None
    assert instance.payload_prefetch.lookahead_hits == 1


def test_missing_notification_is_reconstructed_by_ordinary_command(worker):
    instance, acks, _wire, failures = worker
    for index in range(2):
        instance.data_queue.put(Rollout(prompt_idx=index, completion=f"a{index}"))
    instance.execute_data_fetch(DataFetchCommand("policy-0", 2, 1, 3, 4))
    notice = notification(refs=("b0", "b1"))
    instance.execute_data_fetch(
        DataFetchCommand(
            "policy-0",
            2,
            2,
            3,
            2,
            prefetched_batch_id=notice.batch_id,
            payload_notification=notice._serialize(),
        )
    )
    assert len(acks) == 2 and not failures
    assert instance.payload_prefetch.lookahead_hits == 0
    assert instance.payload_prefetch.lookahead_misses == 2
    # A delayed exact duplicate cannot refetch or resurrect the consumed batch.
    instance.receive_payload_notification(notice)
    assert instance.payload_prefetch.pending is None


def test_duplicate_and_changed_identity(worker):
    instance, _, _, _ = worker
    notice = notification()
    instance.receive_payload_notification(notice)
    pending = instance.payload_prefetch.pending
    instance.receive_payload_notification(Command.depack(notice.pack()))
    assert instance.payload_prefetch.pending is pending
    changed = copy.deepcopy(notice)
    changed.rollouts[0]["completion"] = "another-payload"
    with pytest.raises(ValueError, match="changed content"):
        instance.receive_payload_notification(changed)


def test_wrong_process_incarnation_fails_closed(worker):
    instance, _, _, _ = worker
    notice = notification()
    notice.cohort = (("policy-0", ((0, "restarted-process"),)),)
    with pytest.raises(ValueError, match="incarnation"):
        instance._handle_background_command(notice)
    assert instance.payload_prefetch.failed
    with pytest.raises(RuntimeError, match="terminal"):
        instance.payload_prefetch.complete(1)


def test_receiver_shutdown_unblocks_bounded_next_notification(worker):
    instance, _, _, _ = worker
    pipeline = instance.payload_prefetch
    pipeline.take(cmd(1), lambda: ["a"])
    instance.receive_payload_notification(notification())
    entered, done = threading.Event(), threading.Event()

    def receive():
        entered.set()
        instance.receive_payload_notification(notification(3, "batch-c", ("c",)))
        done.set()

    reader = threading.Thread(target=receive)
    reader.start()
    assert entered.wait(1)
    pipeline.stop_notifications()
    reader.join(1)
    assert done.is_set() and not reader.is_alive()
    assert pipeline.pending.identity == "batch-b"


def test_future_failure_prevents_completed_update_ack(worker):
    instance, _, _, _ = worker
    pipeline = instance.payload_prefetch
    pipeline.take(cmd(1), lambda: ["a"])
    instance.data_packer._fetch_batch = lambda tasks: (_ for _ in ()).throw(
        ValueError("receive failed")
    )
    instance.receive_payload_notification(notification())
    with pytest.raises(ValueError, match="receive failed"):
        pipeline.pending.future.result(timeout=2)
    with pytest.raises(ValueError, match="receive failed"):
        pipeline.complete(1)


def test_background_control_uses_no_training_collective_and_bounds_retention(
    monkeypatch,
):
    from cosmos_rl.utils.distributed import DistKVStore

    def forbidden(*args, **kwargs):
        raise AssertionError("Background control entered a training collective")

    monkeypatch.setattr(torch.distributed, "broadcast_object_list", forbidden)
    monkeypatch.setattr(torch.distributed, "barrier", forbidden)
    store = torch.distributed.HashStore()
    stop = threading.Event()
    peers = []
    for rank in range(2):
        peer = DistKVStore.__new__(DistKVStore)
        peer.rank, peer.world_size, peer.counter = rank, 2, 0
        peer.local_store, peer.shutdown_event = store, stop
        peers.append(peer)
    for index in range(8):
        result, errors = [], []

        def receive():
            try:
                result.append(peers[1].broadcast_command_bounded(None, timeout_s=1))
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=receive)
        thread.start()
        notice = notification(index + 2, f"batch-{index}")
        sent = peers[0].broadcast_command_bounded(notice, timeout_s=1)
        thread.join(2)
        assert not thread.is_alive() and not errors
        assert sent.pack() == result[0].pack() == notice.pack()
        assert store.num_keys() == 3  # one command plus two delivery receipts
    stop.set()
    assert peers[1].broadcast_command_bounded(None, timeout_s=1) is None


def test_missing_background_recipient_times_out_without_retry():
    from cosmos_rl.utils.distributed import DistKVStore

    peer = DistKVStore.__new__(DistKVStore)
    peer.rank, peer.world_size, peer.counter = 0, 2, 0
    peer.local_store = torch.distributed.HashStore()
    peer.shutdown_event = threading.Event()
    with pytest.raises(TimeoutError, match="recipient"):
        peer.broadcast_command_bounded(notification(), timeout_s=0.01)
    assert peer.counter == 0


def test_future_slot_wait_is_bounded(worker):
    instance, _, _, _ = worker
    pipeline = instance.payload_prefetch
    pipeline.take(cmd(1), lambda: ["a"])
    instance.receive_payload_notification(notification())
    pipeline.packer._prefetch_timeout_s = 0.01
    with pytest.raises(TimeoutError, match="admission"):
        instance.receive_payload_notification(notification(3, "c", ("c",)))
    assert pipeline.pending.identity == "batch-b"


def test_ordinary_command_must_match_reserved_metadata_count(worker):
    instance, acks, _, _ = worker
    notice = notification(refs=("b0", "b1"))
    command = DataFetchCommand(
        "policy-0",
        4,
        2,
        3,
        2,
        prefetched_batch_id=notice.batch_id,
        payload_notification=notice._serialize(),
    )
    with pytest.raises(RuntimeError, match="no update ACK"):
        instance.execute_data_fetch(command)
    assert not acks and instance.trainer.updates == 0


def test_idle_pipeline_surfaces_reader_failure(worker):
    instance, _, _, _ = worker
    instance.payload_prefetch.fail_notification()
    with pytest.raises(RuntimeError, match="terminal"):
        instance.payload_prefetch.check()


def test_shutdown_joins_notification_reader_before_transport_close(worker):
    instance, _, _, _ = worker
    reader = instance.fetch_command_thread
    instance.shutdown_mp_signal = threading.Event()
    # The integrated supervision helper owns this same reader. The standalone
    # path joins it directly; both must finish before packer retirement.
    instance._owned_worker_threads = SimpleNamespace(
        close=lambda timeout: reader.join(timeout)
    )
    retired = []

    def close():
        assert not reader.is_alive()
        assert instance.payload_prefetch._closed
        retired.append(True)

    class Finished(Exception):
        pass

    instance.data_packer.shutdown_nccl_data_packer = close
    instance.val_data_packer = None
    instance.inter_policy_nccl.shutdown = lambda: (_ for _ in ()).throw(Finished())
    with pytest.raises(Finished):
        instance.handle_shutdown()
    assert retired == [True]
