# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Training owns received batches even when admission has no byte limit."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from cosmos_rl.policy.trainer.batching import (
    ExpandedSampleBatching,
    ExpandedTrainingBatch,
    FixedRolloutBatching,
    run_training_step,
)
from cosmos_rl.utils.payload_transport.prefetch_mixin import PrefetchDataPackerMixin
from cosmos_rl.utils.payload_transport.receive_memory import (
    ReceiveBudget,
    ReceivedBatch,
    ReceiveMemoryError,
)


class BasePacker:
    def get_policy_input(
        self, sample=None, rollout_output=None, n_ignore_prefix_tokens=0, **kwargs
    ):
        return rollout_output["x"]


class Packer(PrefetchDataPackerMixin, BasePacker):
    def __init__(self, bounded):
        self.batches = []
        self._transport_strategy = SimpleNamespace(
            _receive_budget=ReceiveBudget(64, 1) if bounded else None,
            before_join=lambda: None,
            shutdown=lambda: None,
        )
        self._setup_prefetch(prefetch_timeout=2)

    def _filter_prefetch_tasks(self, rollouts):
        return [(index, rollout.completion) for index, rollout in enumerate(rollouts)]

    def _should_intercept(self, value):
        return isinstance(value, str)

    def _cache_key(self, value):
        return value

    def _fetch_batch(self, tasks):
        tensors = {ref: {"x": torch.tensor([1.0, 2.0])} for _, ref in tasks}
        size = 8 * len(tensors)
        budget = self._transport_strategy._receive_budget
        if budget is not None:
            budget.reserve(size)
            budget.attribute(decoded=size)
        batch = ReceivedBatch(tensors, budget, size)
        self.batches.append(batch)
        return batch

    def _sync_fetch(self, ref):
        raise AssertionError("Training must use its received batch")

    def _on_prefetch_complete(self, *args):
        pass

    def finish(self):
        self.release_prefetch()
        self.close_transport(timeout=2)


def trainer(packer, expanded):
    parameter = torch.nn.Parameter(torch.tensor(0.25))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)

    def prepare(rollouts):
        packer.start_prefetch(rollouts)
        packer.wait_prefetch()
        values = tuple(
            float(value)
            for rollout in rollouts
            for value in packer.get_policy_input(rollout_output=rollout.completion)
        )
        return ExpandedTrainingBatch((values,))

    def update(batch, **kwargs):
        optimizer.zero_grad()
        loss = parameter * sum(batch.minibatches[0])
        loss.backward()
        optimizer.step()
        return {"loss": float(loss.detach())}

    candidate = SimpleNamespace(
        batching_contract=ExpandedSampleBatching(partial_tail="include")
        if expanded
        else FixedRolloutBatching(),
        data_packer=packer,
        config=SimpleNamespace(
            train=SimpleNamespace(
                train_policy=SimpleNamespace(mini_batch=2, mu_iterations=1)
            )
        ),
        prepare_training_batch=prepare,
        step_expanded_training=update,
        step_training=lambda rollouts, **kwargs: update(prepare(rollouts)),
    )
    return candidate, parameter, optimizer


@pytest.mark.parametrize(
    "bounded,expanded,managed",
    [
        (False, False, False),
        (False, True, False),
        (False, True, True),
        (True, True, False),
        (True, True, True),
    ],
)
def test_two_updates_release_each_receive_once(bounded, expanded, managed):
    packer = Packer(bounded)
    candidate, parameter, optimizer = trainer(packer, expanded)
    try:
        for step in range(2):
            rollouts = [SimpleNamespace(completion=f"ref-{step}")]
            if managed:
                packer.start_prefetch(rollouts)
                packer.wait_prefetch()
            scope = packer.payload_batch_scope(rollouts) if managed else nullcontext()
            with scope:
                run_training_step(candidate, rollouts=rollouts)
                if managed:
                    assert not packer.batches[-1].released
            if managed:
                packer.finish_payload_batch()
            assert packer.batches[-1].released
            assert packer._prefetch_cache == {}
        assert len(packer.batches) == 2
        assert float(optimizer.state[parameter]["step"]) == 2
        assert float(parameter.detach()) != 0.25
    finally:
        packer.finish()


@pytest.mark.parametrize("managed", [False, True])
def test_unbounded_returned_alias_keeps_receive_owned(managed):
    packer = Packer(False)
    candidate, _, _ = trainer(packer, False)
    rollouts = [SimpleNamespace(completion="ref")]

    def leak(**kwargs):
        candidate.prepare_training_batch(rollouts)
        return {"borrowed": packer._prefetch_cache["ref"]["x"]}

    candidate.step_training = leak
    try:
        if managed:
            packer.start_prefetch(rollouts)
            packer.wait_prefetch()
        scope = packer.payload_batch_scope(rollouts) if managed else nullcontext()
        with scope:
            with pytest.raises(ReceiveMemoryError):
                run_training_step(candidate, rollouts=rollouts)
        assert not packer.batches[-1].released
    finally:
        packer.finish()


def test_failed_update_keeps_receive_owned():
    packer = Packer(False)
    candidate, _, _ = trainer(packer, False)
    rollouts = [SimpleNamespace(completion="ref")]

    def fail(**kwargs):
        candidate.prepare_training_batch(rollouts)
        raise ValueError("failed update")

    candidate.step_training = fail
    try:
        with pytest.raises(ValueError, match="failed update"):
            run_training_step(candidate, rollouts=rollouts)
        assert not packer.batches[-1].released
    finally:
        packer.finish()


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("stalled", [False, True])
def test_managed_receive_waits_for_worker_reader_deadline(
    monkeypatch, bounded, stalled
):
    from cosmos_rl.dispatcher.command import DataFetchCommand
    from cosmos_rl.policy.trainer.prefetch import TrainerPayloadPrefetch
    from cosmos_rl.utils import pynccl, transport_failure
    from test_trainer_prefetch_lifecycle import deliver, standalone_worker

    worker, acks = standalone_worker()
    worker.data_packer.shutdown_prefetch()
    packer = worker.data_packer = Packer(bounded)
    candidate, _, optimizer = trainer(packer, expanded=bounded)
    candidate.optimizer = optimizer
    candidate.config.train.train_policy.mini_batch = 4
    candidate.update_lr_schedulers = lambda total: None
    worker.trainer = candidate
    worker.payload_prefetch = TrainerPayloadPrefetch(packer)
    worker.train_stream = candidate.train_stream = object()
    worker.inter_policy_nccl = SimpleNamespace(
        allreduce=lambda *args, **kwargs: None,
        wait_comm_ready=lambda: None,
        world_size=lambda: 1,
        replica_name_to_rank={"policy-0": 0},
        default_timeout_ms=1,
    )
    trace = []

    class ReaderTimeout(RuntimeError):
        pass

    class ReaderEvent:
        def record(self, stream):
            assert stream is worker.train_stream
            trace.append("record")

        def query(self):
            trace.append("query")
            return not stalled

        def synchronize(self):
            trace.append("synchronize")
            assert not stalled and "query" in trace, (
                "Unbounded release preceded the worker reader deadline"
            )

    def fail(context):
        assert context == "Trainer payload reader completion is uncertain"
        raise ReaderTimeout(context)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "Event", ReaderEvent)
    monkeypatch.setattr(pynccl, "nccl_abort_all", lambda: trace.append("abort"))
    monkeypatch.setattr(transport_failure, "fail_transport", fail)
    try:
        deliver(worker, 0)
        command = DataFetchCommand("policy-0", 2, 1, 3, 4)
        if stalled:
            with pytest.raises(ReaderTimeout):
                worker.execute_data_fetch(command)
            assert "query" in trace and "abort" in trace
            assert "synchronize" not in trace and not acks
            assert not packer.batches[-1].released
            assert packer._prefetch_cache is packer.batches[-1]
        else:
            worker.execute_data_fetch(command)
            assert trace.index("query") < trace.index("synchronize")
            assert packer.batches[-1].released and packer._prefetch_cache == {}
            assert len(acks) == 1 and "abort" not in trace
    finally:
        packer.finish()
