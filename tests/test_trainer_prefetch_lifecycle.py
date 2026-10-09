# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Actual controller dispatch, stock worker handlers and real optimizer updates."""

from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import msgpack
import pytest
import torch

from cosmos_rl.dispatcher.command import (
    Command,
    DataFetchCommand,
    PayloadPrefetchCommand,
    TrainingCompleteCommand,
)
from cosmos_rl.dispatcher.data.schema import Rollout
from cosmos_rl.dispatcher.status import RolloutStatusManager
from cosmos_rl.policy.trainer.prefetch import TrainerPayloadPrefetch
from dispatch_commit_canary import worker as make_worker
from test_trainer_payload_prefetch import Packer, manager


@pytest.mark.parametrize("reason", ["rejected", "nonfinite", "preparation"])
def test_empty_update_preserves_next_payload_and_optimizer_accounting(reason):
    from cosmos_rl.policy.trainer.batching import (
        ExpandedSampleBatching,
        ExpandedTrainingBatch,
        RecoverablePreparationError,
    )

    instance, acknowledgements = make_worker(torch.device("cpu"), None)
    instance.device = torch.device("cpu")
    instance.train_stream = None
    instance.inter_policy_nccl = SimpleNamespace(
        allreduce=lambda *_args, **_kwargs: None,
        wait_comm_ready=lambda: None,
        world_size=lambda: 1,
        replica_name_to_rank={"policy-0": 0},
    )
    instance.config.train.train_policy.data_dispatch_as_rank_in_mesh = False
    packer = instance.data_packer = Packer()
    packer._setup_prefetch(prefetch_timeout=5)
    pipeline = instance.payload_prefetch = TrainerPayloadPrefetch(packer)
    weight = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([weight], lr=0.1)
    scheduler_calls = []

    def prepare(rollouts):
        if rollouts[0].prompt_idx == 0:
            if reason == "preparation":
                raise RecoverablePreparationError("missing input")
            if reason == "nonfinite":
                return ExpandedTrainingBatch(((float("nan"),),))
            assert all(
                packer.get_policy_input(rollout_output=r.completion) is None
                for r in rollouts
            )
            return ExpandedTrainingBatch(())
        assert all(
            packer.get_policy_input(rollout_output=r.completion) is not None
            for r in rollouts
        )
        return ExpandedTrainingBatch(((2.0, 3.0),))

    def train(batch, **kwargs):
        for samples in batch.minibatches:
            optimizer.zero_grad()
            (weight * (sum(samples) / len(samples))).backward()
            optimizer.step()
        return {}

    instance.trainer = SimpleNamespace(
        batching_contract=ExpandedSampleBatching(partial_tail="include"),
        config=SimpleNamespace(
            train=SimpleNamespace(
                train_policy=SimpleNamespace(mini_batch=2, mu_iterations=1)
            )
        ),
        data_packer=packer,
        optimizer=optimizer,
        prepare_training_batch=prepare,
        step_expanded_training=train,
        update_lr_schedulers=lambda total: scheduler_calls.append(total),
    )
    original_fetch = packer._fetch_batch
    packer._fetch_batch = lambda tasks: original_fetch(
        [(i, ref) for i, ref in tasks if not ref.startswith("rejected")]
    )
    try:
        for index in range(4):
            instance.data_queue.put(
                Rollout(
                    prompt_idx=index,
                    completion=f"{'rejected' if index < 2 else 'valid'}-{index}",
                    reward=0.0,
                )
            )
        instance.execute_data_fetch(
            DataFetchCommand("policy-0", 2, 1, 2, 2, prefetch_next_batch_id="next")
        )
        assert weight.item() == 1.0 and not scheduler_calls
        assert acknowledgements[-1][-1]["batching/skipped_update"] == 1
        assert acknowledgements[-1][-1]["prefetch/optimizer_0_steps"] == 0
        assert pipeline.pending.identity == "next"
        instance.execute_data_fetch(
            DataFetchCommand("policy-0", 2, 2, 2, 0, prefetched_batch_id="next")
        )
        assert scheduler_calls == [2] and len(acknowledgements) == 2
        assert acknowledgements[-1][-1]["prefetch/optimizer_0_steps"] == 1
        torch.testing.assert_close(weight, torch.tensor(0.75))
        pipeline.drain()
    finally:
        packer.shutdown_prefetch()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("count", [0, 1, 2, 5, 6])
@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("requested_stop", [False, True])
def test_real_dispatch_ack_and_terminal_drain(
    enabled, count, checkpoint, requested_stop
):
    status, replica = manager(count)
    status.total_steps = 3
    status.config.train.prefetch_payloads = enabled
    status.config.train.ckpt.enable_checkpoint = checkpoint
    status.config.train.ckpt.save_freq = 2
    status.policy_init_done = True
    # No real rollout peers in this fixture; transport/weight-sync gets native gates.
    status.should_weight_sync_after_train_ack = lambda *_: False
    rollout_status = RolloutStatusManager()
    rollout_status.rollout_replicas = {}
    rollout_status.all_rollouts_ended = lambda: True
    status.data_fetcher.activated_val_iter = None
    instance, acknowledgements = make_worker(torch.device("cpu"), None)
    instance.device = torch.device("cpu")
    instance.train_stream = None
    instance.inter_policy_nccl = SimpleNamespace(
        allreduce=lambda *_args, **_kwargs: None
    )
    instance.replica_name = replica.name
    instance.api_client._report_session_id = "test-session"
    instance.world_size = instance.dp_world_size = 1
    instance.parallel_dims.get_rank_in_dim = lambda *_: 0
    instance.config.train.train_policy.data_dispatch_as_rank_in_mesh = False
    instance.data_packer = Packer()
    instance.data_packer._setup_prefetch(prefetch_timeout=5)
    instance.payload_prefetch = (
        TrainerPayloadPrefetch(instance.data_packer) if enabled else None
    )
    instance.trainer.save_checkpoint = Mock()
    instance.trainer.invalidate_checkpoint_completion = Mock()
    instance.trainer.finish_checkpoint_writes = Mock()
    saved = []
    original_train = instance.trainer.step_training

    def train(**kwargs):
        for rollout in kwargs["rollouts"]:
            assert (
                instance.data_packer.get_policy_input(rollout_output=rollout.completion)
                == rollout.completion + "-payload"
            )
        report = original_train(**kwargs)
        if kwargs["do_save_checkpoint"]:
            saved.append(kwargs["current_step"])
        return report

    instance.trainer.step_training = train
    # Existing default path uses synchronous resolution; enabling lookahead must
    # change neither the numerical samples nor the real optimizer count.
    instance.data_packer._sync_fetch = lambda ref: ref + "-payload"
    pending = deque()

    def publish(plan):
        for _stream, kind, data in plan.entries:
            if kind == "rollout":
                instance.data_queue.put(Rollout(**msgpack.unpackb(data)))
            else:
                pending.append(Command.depack(data))

    def publish_command(data, _replica):
        pending.append(Command.depack(data))

    status.redis_handler.publish_plan = publish
    status.redis_handler.publish_command = publish_command
    for rollout in status.rollout_buffer.queue:
        rollout.completion = f"sample-{rollout.prompt_idx}"
    try:
        status.finish_draining_phase(rollout_status)
        terminal = []
        while pending:
            command = pending.popleft()
            if isinstance(command, PayloadPrefetchCommand):
                instance.receive_payload_notification(command)
                continue
            if isinstance(command, TrainingCompleteCommand):
                # CPU-only fixture substitutes just the checkpoint agreement;
                # handler order and checkpoint invocation remain production code.
                from unittest.mock import patch

                with patch(
                    "cosmos_rl.policy.worker.rl_worker.dist_util.all_reduce_tensor_object_cpu",
                    side_effect=lambda tensor, **_: tensor,
                ):
                    instance.execute_training_complete(command)
                terminal.append(command)
            else:
                instance.execute_data_fetch(command)
                if requested_stop and command.global_step == 1:
                    status.request_stop("test budget reached")
            ack = acknowledgements.pop(0)
            status.train_ack(*ack, rollout_status_manager=rollout_status)
        expected = min(count // 2, 3)
        if requested_stop and expected:
            expected = min(expected, 2 if enabled else 1)
        assert instance.trainer.updates == instance.trainer.scheduler_calls == expected
        assert status.current_step == expected
        assert status.training_finished()
        assert status.samples_on_the_fly == 0
        assert status._payload_lookahead is None
        if instance.payload_prefetch:
            instance.payload_prefetch.drain()
        assert not acknowledgements and not pending
        if checkpoint and terminal:
            instance.trainer.save_checkpoint.assert_called_once()
            assert (
                instance.trainer.save_checkpoint.call_args.kwargs["current_step"]
                == expected
            )
        elif not checkpoint:
            instance.trainer.save_checkpoint.assert_not_called()
            assert not saved
    finally:
        instance.data_packer.shutdown_prefetch()


def standalone_worker():
    instance, acks = make_worker(torch.device("cpu"), None)
    instance.device = torch.device("cpu")
    instance.train_stream = None
    instance.inter_policy_nccl = SimpleNamespace(allreduce=lambda *a, **k: None)
    instance.config.train.train_policy.data_dispatch_as_rank_in_mesh = False
    instance.data_packer = Packer()
    instance.data_packer._setup_prefetch(prefetch_timeout=5)
    instance.payload_prefetch = TrainerPayloadPrefetch(instance.data_packer)
    return instance, acks


def deliver(instance, first):
    for index in (first, first + 1):
        instance.data_queue.put(
            Rollout(prompt_idx=index, completion=f"sample-{index}", reward=0.0)
        )


def test_resume_starts_empty_at_committed_update(tmp_path):
    """Real optimizer/parameter reload; untrained reservation is replayable."""
    instance, acks = standalone_worker()
    resumed, resumed_acks = standalone_worker()
    try:
        deliver(instance, 0)
        deliver(instance, 2)
        instance.signal_handler = SimpleNamespace(
            signals_received=lambda: [True], release=Mock()
        )
        instance.execute_data_fetch(
            DataFetchCommand(
                "policy-0", 2, 1, 3, 4, prefetch_next_batch_id="uncommitted"
            )
        )
        assert instance.payload_prefetch.pending.future is None
        trainer = instance.trainer
        checkpoint = tmp_path / "checkpoint.pt"
        torch.save(
            dict(
                weight=trainer.weight.detach(),
                optimizer=trainer.optimizer.state_dict(),
                updates=trainer.updates,
                scheduler_calls=trainer.scheduler_calls,
                reference_weight=trainer.reference_weight,
                reference_momentum=trainer.reference_momentum,
                step=1,
                remain_samples_num=4,
            ),
            checkpoint,
        )
        saved = torch.load(checkpoint, weights_only=True)
        assert saved["step"] == 1 and saved["remain_samples_num"] == 4
        resumed.trainer.weight.data.copy_(saved["weight"])
        resumed.trainer.optimizer.load_state_dict(saved["optimizer"])
        for key in (
            "updates",
            "scheduler_calls",
            "reference_weight",
            "reference_momentum",
        ):
            setattr(resumed.trainer, key, saved[key])
        # A restored worker receives fresh metadata, not the old reservation ID.
        deliver(resumed, 2)
        resumed.execute_data_fetch(DataFetchCommand("policy-0", 2, 2, 3, 2))
        instance.execute_data_fetch(
            DataFetchCommand("policy-0", 2, 2, 3, 2, prefetched_batch_id="uncommitted")
        )
        for worker in (instance, resumed):
            deliver(worker, 4)
            worker.execute_data_fetch(DataFetchCommand("policy-0", 2, 3, 3, 0))
            worker.payload_prefetch.drain()
        torch.testing.assert_close(instance.trainer.weight, resumed.trainer.weight)
        assert instance.trainer.updates == resumed.trainer.updates == 3
        assert len(acks) == 3 and len(resumed_acks) == 2
    finally:
        instance.data_packer.shutdown_prefetch()
        resumed.data_packer.shutdown_prefetch()


@pytest.mark.parametrize("after_optimizer", [False, True])
def test_failed_training_never_acks_or_reuses_pipeline(after_optimizer):
    instance, acks = standalone_worker()
    train = instance.trainer.step_training

    def fail(**kwargs):
        if after_optimizer:
            train(**kwargs)
        raise RuntimeError("injected trainer error")

    instance.trainer.step_training = fail
    try:
        deliver(instance, 0)
        deliver(instance, 2)
        with pytest.raises(RuntimeError, match="injected trainer error"):
            instance.execute_data_fetch(
                DataFetchCommand("policy-0", 2, 1, 3, 4, prefetch_next_batch_id="b")
            )
        assert not acks and instance.payload_prefetch.failed
        assert instance.trainer.updates == int(after_optimizer)
        with pytest.raises(RuntimeError):
            instance.payload_prefetch.drain()
    finally:
        instance.data_packer.shutdown_prefetch()


def test_final_reader_uses_training_budget_not_shutdown_budget(monkeypatch):
    instance, acks = standalone_worker()
    instance.train_stream = object()
    instance.inter_policy_nccl.default_timeout_ms = 120000
    drain = Mock(return_value=True)
    monkeypatch.setenv("COSMOS_TEARDOWN_DRAIN_TIMEOUT_S", "0.01")
    monkeypatch.setattr(
        "cosmos_rl.policy.worker.rl_worker.bounded_drain_or_abort", drain
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    try:
        deliver(instance, 0)
        instance.execute_data_fetch(DataFetchCommand("policy-0", 2, 1, 3, 4))
        assert len(acks) == 1 and drain.call_count == 1
        assert 119 < drain.call_args.args[1] <= 120
    finally:
        instance.data_packer.shutdown_prefetch()


@pytest.mark.parametrize("late_future", [False, True])
def test_final_data_fetch_closes_admission_before_ack(late_future: bool) -> None:
    worker, acks = standalone_worker()
    try:
        for step in [1, 2]:
            deliver(worker, 2 * (step - 1))
            worker.execute_data_fetch(
                DataFetchCommand("policy-0", 2, step, 3, 2 * (3 - step))
            )
        original = worker.trainer.step_training

        def train(**kwargs):
            result = original(**kwargs)
            if late_future:
                worker.payload_prefetch.notify(
                    SimpleNamespace(
                        batch_id="past-final",
                        global_step=4,
                        replica_name="policy-0",
                        cohort=(),
                        rollouts=[],
                    ),
                    (),
                    start_fetch=False,
                )
            return result

        worker.trainer.step_training = train
        deliver(worker, 4)
        if late_future:
            with pytest.raises(RuntimeError, match="Payload lifecycle failed"):
                worker.execute_data_fetch(DataFetchCommand("policy-0", 2, 3, 3, 0))
            assert len(acks) == 2 and worker.payload_prefetch.failed
        else:
            assert worker.execute_data_fetch(DataFetchCommand("policy-0", 2, 3, 3, 0))
            assert len(acks) == 3 and worker.payload_prefetch._closed
            assert worker.payload_prefetch.pending is None
        assert worker.trainer.updates == 3
    finally:
        worker.payload_prefetch.stop_notifications()
        worker.data_packer.shutdown_prefetch()
