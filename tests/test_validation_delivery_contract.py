# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real sampler -> API -> receipts -> report-handler validation contracts."""

import asyncio
import threading
from collections import deque
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cosmos_rl.dispatcher import run_web_panel
from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.dispatcher.protocol import ValidationReportRequest
from cosmos_rl.dispatcher.status import PolicyStatusManager
from cosmos_rl.rollout.validation import ValidationSession
from test_dispatch_fetch_bounds import fetcher


def manager(indices, reporters=(0,), extra_replicas=()):
    source = fetcher(rollout_size=1)
    source.config.train.train_policy.data_dispatch_as_rank_in_mesh = False
    source.val_dataloader = [
        ([index], [RLPayload(prompt_idx=index, prompt=str(index))]) for index in indices
    ]
    source.val_datasize = 100  # Deliberately unlike actual sampler work.
    source.val_iters = {}
    source.activated_val_step = None
    source.activated_val_iter = None
    source.activated_val_tqdm = None
    instance = PolicyStatusManager()
    instance.data_fetcher = source
    instance.config = SimpleNamespace(
        validation=SimpleNamespace(
            enable=True, val_before_train=True, freq=1, n_generation=1
        ),
        train=SimpleNamespace(train_policy=SimpleNamespace(type="grpo")),
        logging=SimpleNamespace(logger=[]),
    )
    instance.custom_logger_fns = [Mock()]
    instance.try_trigger_data_fetch_and_training = Mock()
    replica = SimpleNamespace(
        name="a",
        atoms={
            rank: SimpleNamespace(global_rank=rank, validation_reporter=True)
            for rank in reporters
        },
    )
    replicas = [replica] + [
        SimpleNamespace(name=name, atoms=replica.atoms) for name in extra_replicas
    ]
    round_id = instance.prepare_validation_round(2, 10, replicas)
    return instance, round_id, replica


class EndpointClient:
    """Each request reaches the real route twice, as with a lost HTTP reply."""

    def get_next_prompt(self, size, **kwargs):
        first = asyncio.run(run_web_panel.get_batched_prompt(size, **kwargs))
        second = asyncio.run(run_web_panel.get_batched_prompt(size, **kwargs))
        assert first == second
        assert isinstance(second, dict), second.body
        return [payload.model_dump() for payload in second["payloads_list"]], second[
            "is_end"
        ]

    def post_validation_report(self, request):
        # Independent deserialization per delivery, not a shared mutated object.
        wire = request.model_dump_json()
        for _ in range(2):
            response = asyncio.run(
                run_web_panel.validation_report(
                    ValidationReportRequest.model_validate_json(wire)
                )
            )
            assert isinstance(response, dict), response.body


def bind_controller(monkeypatch, instance):
    monkeypatch.setattr(
        run_web_panel,
        "controller",
        SimpleNamespace(
            config=instance.config,
            policy_status_manager=instance,
            rollout_status_manager=Mock(),
        ),
    )


@pytest.mark.parametrize("indices", [[], [7], [7, 7], [0, 1, 2, 3, 4]])
def test_real_fetch_and_report_retries_settle_actual_work(monkeypatch, indices):
    instance, round_id, _ = manager(indices)
    bind_controller(monkeypatch, instance)
    session = ValidationSession(EndpointClient(), round_id, 2, "a", 0)
    received = []
    while True:
        raw, end = session.fetch(2)
        payloads = [RLPayload.model_validate(value) for value in raw]
        received.extend(payload.prompt_idx for payload in payloads)
        for payload in payloads:
            payload.rewards = [1.0]
            payload.advantages = [0.0]
        if payloads:
            session.report(payloads)
        if end:
            break
    assert received == indices
    assert not instance.validation_round.complete
    session.report([], is_end=True)
    assert instance.validation_round.complete
    assert instance.validation_round.reported_prompts == len(indices)
    assert instance.data_fetcher.activated_val_iter is None
    assert not instance.data_fetcher.val_iters
    assert not instance.val_report_data
    instance.try_trigger_data_fetch_and_training.assert_called_once()

    assert instance.custom_logger_fns[0].call_count == int(bool(indices))


def test_old_step_cannot_clear_new_round_and_retained_final_ack_is_idempotent(
    monkeypatch,
):
    instance, round_id, replica = manager([])
    bind_controller(monkeypatch, instance)
    session = ValidationSession(EndpointClient(), round_id, 2, "a", 0)
    session.fetch(2)
    session.report([], is_end=True)
    next_id = instance.prepare_validation_round(3, 10, [replica])
    assert next_id != round_id
    stale = ValidationReportRequest(
        src_replica_name="a",
        src_global_rank=0,
        validation_step=2,
        validation_round_id=round_id,
        report_sequence=0,
        payloads=[],
        is_end=True,
    )
    response = asyncio.run(run_web_panel.validation_report(stale))
    assert isinstance(response, dict)  # Lost final ACK, not new work.
    stale.is_end = False
    response = asyncio.run(run_web_panel.validation_report(stale))
    assert response.status_code == 409
    assert instance.data_fetcher.activated_val_step == 3
    assert not instance.validation_round.complete
    instance.try_trigger_data_fetch_and_training.assert_called_once()


def test_empty_reporter_and_sealed_membership(monkeypatch):
    instance, round_id, replica = manager([], reporters=(0, 1))
    bind_controller(monkeypatch, instance)
    first = ValidationSession(EndpointClient(), round_id, 2, "a", 0)
    first.fetch(2)
    first.report([], is_end=True)
    assert not instance.validation_round.complete
    replica.atoms.pop(1)
    with pytest.raises(ValueError, match="change its reporters"):
        instance.prepare_validation_round(2, 10, [replica])
    second = ValidationSession(EndpointClient(), round_id, 2, "a", 1)
    second.report([], is_end=True)
    assert instance.validation_round.complete


def test_short_extracted_group_does_not_consume_receipt(monkeypatch):
    instance, round_id, _ = manager([7])
    bind_controller(monkeypatch, instance)
    batch = instance.fetch_validation_prompts(2, 2, None, round_id, "a", 0)
    payload = batch.payloads[0]
    payload.rewards, payload.advantages = [1.0], []
    request = ValidationReportRequest(
        src_replica_name="a",
        src_global_rank=0,
        validation_step=2,
        validation_round_id=round_id,
        report_sequence=0,
        payloads=[payload],
    )
    response = asyncio.run(run_web_panel.validation_report(request))
    assert response.status_code == 409
    assert instance.validation_round.reported_prompts == 0
    assert not instance.val_report_data


def test_round_is_sealed_before_weight_sync_publication(monkeypatch):
    from cosmos_rl.dispatcher import command

    instance, round_id, replica = manager([])
    instance._weight_sync_rollout_targets = lambda _: [replica]
    instance.policy_atoms_in_replica = 1
    instance.redis_handler = Mock()
    seen = []

    def publish(**kwargs):
        assert instance.validation_round.round_id == round_id
        assert instance.validation_round.reporters == {("a", 0)}
        seen.append(kwargs)

    monkeypatch.setattr(command.PolicyToRolloutUnicastCommand, "trigger", publish)
    monkeypatch.setattr(command.RolloutToRolloutBroadcastCommand, "trigger", publish)
    instance.trigger_weight_sync(
        Mock(), SimpleNamespace(rollout_atoms_in_replica=1), 2, 10
    )
    assert len(seen) == 2
    assert seen[-1]["validation_round_id"] == round_id
    replica.atoms[1] = SimpleNamespace(global_rank=1, validation_reporter=True)
    with pytest.raises(ValueError, match="change its reporters"):
        instance.trigger_weight_sync(
            Mock(), SimpleNamespace(rollout_atoms_in_replica=2), 2, 10
        )
    assert len(seen) == 2


def test_validation_command_and_wrapper_instruction_keep_round_identity():
    import pickle
    from cosmos_rl.dispatcher.command import Command, RolloutToRolloutBroadcastCommand
    from cosmos_rl.rollout.trtllm_rollout.trtllm_common import ValidationInstruction

    command = RolloutToRolloutBroadcastCommand(
        "a", ["a", "b"], 2, 10, False, validation_round_id="round"
    )
    restored = Command.depack(command.pack())
    assert restored.validation_round_id == "round"
    instruction = ValidationInstruction(
        restored.weight_step, restored.total_steps, restored.validation_round_id
    )
    assert pickle.loads(pickle.dumps(instruction)).validation_round_id == "round"


@pytest.mark.parametrize("mode", ["sync", "async", "colocated"])
@pytest.mark.parametrize("indices", [[], [7], [7, 7], [0, 1, 2, 3, 4]])
def test_worker_collection_rewards_and_terminal_delivery(monkeypatch, mode, indices):
    # Real worker control flow and local rewards; generation is a controlled
    # boundary. This does not claim a live async engine or simulator run.
    from cosmos_rl.reward.local_calculator import LocalRewardCalculator
    from cosmos_rl.rollout.schema import RolloutResult
    from cosmos_rl.rollout.worker.rollout_control import (
        DisaggregatedRolloutControlWorker,
    )
    from cosmos_rl.rollout.worker.colocated.rollout_control import (
        ColocatedRolloutControlWorker,
    )
    from cosmos_rl.utils import distributed as dist_utils
    from test_completion_admission import _TestAlgo

    instance, round_id, _ = manager(indices)
    bind_controller(monkeypatch, instance)
    monkeypatch.setattr(dist_utils, "broadcast_object_cpu", lambda value: value)
    cls = (
        ColocatedRolloutControlWorker
        if mode == "colocated"
        else DisaggregatedRolloutControlWorker
    )
    worker = object.__new__(cls)
    worker.api_client = EndpointClient()
    worker.replica_name, worker.global_rank = "a", 0
    worker.current_step = worker.current_weight_version = 2
    worker.validation_round_id = round_id
    worker.validation_flag = threading.Event()
    worker.validation_flag.set()
    worker.val_batch_size = 2
    worker.val_data_packer = worker.inference_stream = None
    worker.should_report = True
    worker._is_async_rollout = mode == "async"
    worker.is_diffusers = False
    worker.config = SimpleNamespace(
        train=SimpleNamespace(
            local_dataset=True,
            train_policy=SimpleNamespace(data_dispatch_as_rank_in_mesh=False),
        ),
        rollout=SimpleNamespace(
            mode="async" if mode == "async" else "sync",
            multi_turn_config=SimpleNamespace(enable=False),
        ),
    )
    worker.parallel_dims = SimpleNamespace(mesh={"dp": SimpleNamespace(size=lambda: 1)})
    worker._prompt_fetch_lock = threading.Lock()
    worker.rank_in_rollout_repicas = 0
    worker.data_fetcher = SimpleNamespace(
        get_payload_by_index=lambda index, **kw: None if "attr" in kw else str(index),
        query_reference_answer=lambda *args: "answer",
    )
    worker._call_rollout_generation = lambda **kw: [
        RolloutResult(completions=["answer"]) for _ in kw["payloads"]
    ]
    calculator = object.__new__(LocalRewardCalculator)
    calculator.val_rl_algo = _TestAlgo([1.0])
    pending = deque()

    def enqueue(payloads, is_validation, step):
        assert is_validation and step == 2
        if payloads:
            pending.append(
                (*calculator.compute_validation_rewards(payloads, step), False)
            )

    def dequeue_rewards(**kwargs):
        return pending.popleft() if pending else (None, False, -1, True)

    worker.reward_dispatcher = SimpleNamespace(
        enqueue_rewards_cal=enqueue,
        dequeue_rewards_cal=dequeue_rewards,
    )
    worker.report_rollouts = dequeue_rewards
    completed = []

    def schedule(tasks):
        completed.extend(
            SimpleNamespace(
                payload=task.payload, result=RolloutResult(completions=["answer"])
            )
            for task in tasks
        )

    def get_all():
        result = list(completed)
        completed.clear()
        return result

    worker.scheduler = SimpleNamespace(
        is_busy=lambda: False,
        max_concurrent_requests=2,
        pending_tasks=lambda: 0,
        active_tasks=lambda: 0,
        is_idle=lambda: not completed,
        get_all=get_all,
        put_rollout_batch=schedule,
    )
    worker._prompt_queue = Queue()
    worker.do_validation()
    assert instance.validation_round.complete
    assert instance.validation_round.reported_prompts == len(indices)
    assert not worker.validation_flag.is_set()
    assert not instance.val_report_data
    instance.try_trigger_data_fetch_and_training.assert_called_once()

    # A duplicate validating command must not generate a second result set.
    worker.validation_flag.set()
    worker._call_rollout_generation = Mock(side_effect=AssertionError("regenerated"))
    worker.do_validation()
    assert not worker.validation_flag.is_set()
    worker._call_rollout_generation.assert_not_called()
    instance.try_trigger_data_fetch_and_training.assert_called_once()


@pytest.mark.parametrize(
    "backend,tp,pp,rank,expected",
    [
        ("vllm", 0, 0, 0, False),
        ("vllm", 0, 1, 2, True),
        ("vllm", 1, 1, 3, False),
        ("vllm_async", 0, 1, 2, True),
        ("trtllm", 0, 0, 0, True),
        ("trtllm", 0, 1, 2, False),
    ],
)
def test_registration_advertises_actual_validation_reporter(
    monkeypatch, backend, tp, pp, rank, expected
):
    from cosmos_rl.comm import base
    from cosmos_rl.dispatcher.protocol import Role

    class Mesh:
        mesh_dim_names = ["tp", "pp"]

        def __getitem__(self, name):
            return SimpleNamespace(
                get_local_rank=lambda: {"tp": tp, "pp": pp}[name],
                size=lambda: 2,
            )

    worker = SimpleNamespace(
        parallel_dims=SimpleNamespace(mesh=Mesh()),
        config=SimpleNamespace(validation=SimpleNamespace(enable=True)),
        role=Role.ROLLOUT,
        backend=backend,
        global_rank=rank,
        replica_name="a",
        api_client=Mock(),
        heartbeat_trigger=Mock(),
        unregister_from_controller=Mock(),
    )
    monkeypatch.setattr(base, "get_local_ip", lambda: ("127.0.0.1", "host"))
    monkeypatch.setattr(base.dist, "barrier", Mock())
    monkeypatch.setattr(base.mp, "Process", Mock())
    monkeypatch.setattr(base.mp, "Event", threading.Event)
    monkeypatch.setattr(base.atexit, "register", Mock())
    base.CommMixin.register_to_controller(worker)
    assert (
        worker.api_client.register.call_args.kwargs["validation_reporter"] is expected
    )
