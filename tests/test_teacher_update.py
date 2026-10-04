# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import ast
from contextlib import nullcontext
from datetime import timedelta
import logging
import os
from pathlib import Path
import threading
from types import MethodType, SimpleNamespace
from unittest.mock import Mock
from typing import List, Optional

import msgpack
import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import cosmos_rl
from cosmos_rl.dispatcher.data.schema import Rollout
from cosmos_rl.policy.trainer import teacher_update
from cosmos_rl.utils.teacher_results import TeacherResultInbox


def fetch_method():
    path = (
        Path(cosmos_rl.__file__).parent / "policy/trainer/llm_trainer/grpo_trainer.py"
    )
    tree = ast.parse(path.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "fetch_teacher_logprobs"
    )
    namespace = dict(
        List=List,
        Optional=Optional,
        Rollout=Rollout,
        msgpack=msgpack,
        np=np,
        constant=SimpleNamespace(COSMOS_TEACHER_RESULT_GET_TIMEOUT=0.03),
        logger=logging.getLogger("teacher-test"),
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace[method.name]


def trainer_for(rank, scenario, device):
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            train=SimpleNamespace(
                train_policy=SimpleNamespace(collect_rollout_logprobs=True)
            ),
            distillation=SimpleNamespace(
                enable=True,
                top_k=0,
                trainer_token_ids_from_teacher=scenario.startswith("teacher_token_"),
            ),
        ),
        parallel_dims=SimpleNamespace(pp_cp_tp_coord=(0, 1)),
        teacher_results=TeacherResultInbox(),
        fetched_teacher_uuids=set(),
        device=device,
        steps=0,
        scheduler_steps=0,
        reference_resets=0,
        saved=0,
    )
    trainer.fetch_teacher_logprobs = MethodType(fetch_method(), trainer)

    def clear():
        trainer.teacher_results.retire(trainer.fetched_teacher_uuids)
        trainer.fetched_teacher_uuids.clear()

    trainer.clear_teacher_result_cache = clear
    trainer.save_checkpoint = lambda **_: setattr(trainer, "saved", trainer.saved + 1)
    trainer.parameter = torch.nn.Parameter(torch.tensor(1.0, device=device))
    trainer.optimizer = torch.optim.SGD([trainer.parameter], lr=0.1)
    trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, 1, gamma=0.5)
    rollouts = []
    for i in range(2):
        identity = f"{rank}-{i}"
        rollout = Rollout(
            prompt_idx=i,
            teacher_result_uuid=identity,
            prompt_logprobs=[],
            completion_logprobs=[[-1.0]],
            prompt_token_ids=[[2]],
            completion_token_ids=[[1]],
        )
        trainer.teacher_results.admit(identity)
        missing = scenario == "all_missing" or (
            rank == 3
            and (scenario == "one_missing" or (scenario == "partial" and i == 1))
        )
        if not missing:
            payload = {
                "prompt_token_ids": [[2]],
                "completion_token_ids": [[1]],
                "teacher_logprobs": [
                    [float("nan") if scenario == "malformed" and rank == 3 else -1.0]
                ],
            }
            if scenario == "length_mismatch" and rank == 3:
                payload["teacher_logprobs"].append([-1.0])
            if scenario == "teacher_token_width" and rank == 3:
                payload["completion_token_ids"] = [[1, 3, 4]]
            if scenario == "teacher_token_empty" and rank == 3:
                payload["completion_token_ids"] = [[]]
            if scenario == "malformed" and rank == 2:
                payload = []  # Valid msgpack, but not a teacher-result mapping.
            trainer.teacher_results.complete(identity, msgpack.packb(payload))
        rollouts.append(rollout)
    return trainer, rollouts


def cohort_worker(rank, path, backend):
    teacher_update.constant = SimpleNamespace(
        COSMOS_GLOO_TIMEOUT=90, COSMOS_TEACHER_RESULT_GET_TIMEOUT=0.03
    )
    device = torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(device)
    dist.init_process_group(
        backend,
        init_method=f"file://{path}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=90),
    )
    local_groups = [
        dist.new_group(ranks, backend=backend) for ranks in ([0, 1], [2, 3])
    ]
    cross_groups = [
        dist.new_group(ranks, backend=backend) for ranks in ([0, 2], [1, 3])
    ]
    local = local_groups[rank // 2]
    cross = cross_groups[rank % 2]
    all_reduce = dist.all_reduce
    teacher_update.dist = SimpleNamespace(
        is_initialized=lambda: True,
        ReduceOp=dist.ReduceOp,
        all_reduce=lambda tensor, op: all_reduce(tensor, op=op, group=local),
    )
    comm = SimpleNamespace(
        operation_scope=nullcontext,
        allreduce=lambda send, recv, op: all_reduce(recv, op=op, group=cross),
    )
    native_comm = None
    if backend == "nccl":
        from cosmos_rl.utils.distributed import HighAvailabilitylNccl
        from cosmos_rl.utils.pynccl import create_nccl_comm, create_nccl_uid, nccl_abort

        ids = [create_nccl_uid() if rank < 2 else None]
        dist.broadcast_object_list(ids, src=rank % 2, group=cross)
        native_comm = HighAvailabilitylNccl.__new__(HighAvailabilitylNccl)
        native_comm.replica_name = f"replica-{rank // 2}"
        native_comm.replica_name_to_rank = {"replica-0": 0, "replica-1": 1}
        native_comm.global_rank = rank % 2
        native_comm.default_timeout_ms = 30000
        native_comm.max_retry = 1
        native_comm.build_mesh_lock = threading.RLock()
        native_comm._operation_depth = 0
        native_comm.is_comm_ready = threading.Event()
        native_comm.is_comm_ready.set()
        native_comm.is_single_peer = threading.Event()
        native_comm.comm_idx = create_nccl_comm(ids[0], rank // 2, 2, timeout_ms=30000)
        comm = native_comm

    @teacher_update.teacher_update_boundary
    def train(
        trainer,
        rollouts,
        current_step,
        total_steps,
        remain_samples_num,
        inter_policy_nccl,
        is_master_replica,
        do_save_checkpoint=False,
    ):
        assert all(rollout.teacher_logprobs is not None for rollout in rollouts)
        # A rank-local skip here would hang the real gradient collectives.
        trainer.optimizer.zero_grad()
        (trainer.parameter * (rank + 1)).backward()
        all_reduce(trainer.parameter.grad, group=local)
        comm.allreduce(
            trainer.parameter.grad, trainer.parameter.grad, dist.ReduceOp.SUM
        )
        trainer.parameter.grad.div_(4)
        trainer.optimizer.step()
        trainer.scheduler.step()
        trainer.steps += 1
        trainer.scheduler_steps += 1
        trainer.reference_resets += 1
        return {"updated": 1}

    try:
        for scenario in (
            "healthy",
            "one_missing",
            "partial",
            "all_missing",
            "malformed",
            "length_mismatch",
            "teacher_token_width",
            "teacher_token_empty",
            "healthy_after",
        ):
            trainer, rollouts = trainer_for(rank, scenario, device)
            result = train(trainer, rollouts, 1, 2, 0, comm, True, True)
            healthy = scenario in ("healthy", "healthy_after")
            assert (
                trainer.steps,
                trainer.scheduler_steps,
                trainer.reference_resets,
            ) == ((1, 1, 1) if healthy else (0, 0, 0))
            assert trainer.parameter.item() == pytest.approx(0.75 if healthy else 1.0)
            assert trainer.optimizer.param_groups[0]["lr"] == (0.05 if healthy else 0.1)
            assert trainer.saved == (0 if healthy else 1)
            assert result == (
                {"updated": 1}
                if healthy
                else {"train_step": 1, "train/teacher_update_skipped": 1}
            )
            assert not trainer.teacher_results._results
            print(
                f"TEACHER_COHORT_PASS rank={rank} scenario={scenario} steps={trainer.steps}",
                flush=True,
            )
    finally:
        if native_comm is not None:
            nccl_abort(native_comm.comm_idx)
        dist.destroy_process_group()


def test_two_replicas_with_two_ranks_agree_before_any_optimizer_work(tmp_path):
    backend = os.environ.get("COSMOS_TEST_TEACHER_BACKEND", "gloo")
    if backend == "nccl":
        assert torch.cuda.device_count() >= 4, "Native teacher gate requires four GPUs"
    mp.spawn(
        cohort_worker, args=(str(tmp_path / "cohort"), backend), nprocs=4, join=True
    )


@pytest.mark.parametrize("timeout", [-1, 3601, float("nan"), float("inf")])
def test_invalid_wait_configuration_fails_before_rank_local_fetch(monkeypatch, timeout):
    monkeypatch.setattr(
        teacher_update.constant, "COSMOS_TEACHER_RESULT_GET_TIMEOUT", timeout
    )
    trainer = SimpleNamespace(fetch_teacher_logprobs=Mock(), device="cpu")
    comm = SimpleNamespace(allreduce=Mock())
    with pytest.raises(ValueError, match="Teacher timeout"):
        teacher_update.agree_teacher_update(trainer, [], comm)
    trainer.fetch_teacher_logprobs.assert_not_called()
    comm.allreduce.assert_not_called()


@pytest.mark.parametrize("value", [None, [], 1, True, "bad", b"bad", {}])
def test_actual_grpo_entrypoint_skips_unusable_decoded_result(value):
    from cosmos_rl.policy.trainer.llm_trainer.grpo_trainer import GRPOTrainer

    trainer, rollouts = trainer_for(0, "healthy", torch.device("cpu"))
    trainer.teacher_results.complete("0-0", msgpack.packb(value))
    comm = SimpleNamespace(
        operation_scope=nullcontext, allreduce=lambda *args, **kwargs: None
    )
    result = GRPOTrainer.step_training(trainer, rollouts, 1, 2, 0, comm, True, True)
    assert result == {"train_step": 1, "train/teacher_update_skipped": 1}
    assert trainer.parameter.item() == 1.0
    assert trainer.optimizer.state == {}
    assert trainer.optimizer.param_groups[0]["lr"] == 0.1
    assert trainer.steps == trainer.scheduler_steps == trainer.reference_resets == 0
    assert not trainer.teacher_results._results


def test_actual_grpo_entrypoint_skips_before_model_work(monkeypatch):
    monkeypatch.setattr(
        teacher_update.constant, "COSMOS_TEACHER_RESULT_GET_TIMEOUT", 0.03
    )
    from cosmos_rl.policy.trainer.llm_trainer.grpo_trainer import GRPOTrainer

    trainer, rollouts = trainer_for(3, "one_missing", torch.device("cpu"))
    comm = SimpleNamespace(
        operation_scope=nullcontext, allreduce=lambda *args, **kwargs: None
    )
    # The test object deliberately has no model, parallel mesh or CUDA state.
    result = GRPOTrainer.step_training(trainer, rollouts, 1, 2, 0, comm, True, True)
    assert result["train/teacher_update_skipped"] == 1
    assert trainer.saved == 1 and trainer.parameter.item() == 1.0
    assert (
        trainer.optimizer.state == {} and trainer.optimizer.param_groups[0]["lr"] == 0.1
    )
    assert not trainer.teacher_results._results


def test_teacher_wait_cannot_outlast_peer_collective_deadlines(monkeypatch):
    monkeypatch.setattr(
        teacher_update.constant, "COSMOS_TEACHER_RESULT_GET_TIMEOUT", 1800
    )
    monkeypatch.setattr(teacher_update.constant, "COSMOS_GLOO_TIMEOUT", 600)
    assert (
        teacher_update.teacher_wait_budget(SimpleNamespace(default_timeout_ms=600000))
        == 300
    )
    assert (
        teacher_update.teacher_wait_budget(SimpleNamespace(default_timeout_ms=2000))
        == 1
    )
    monkeypatch.setattr(
        teacher_update.constant, "COSMOS_TEACHER_RESULT_GET_TIMEOUT", 0.05
    )
    assert (
        teacher_update.teacher_wait_budget(SimpleNamespace(default_timeout_ms=2000))
        == 0.05
    )


def test_native_mesh_cannot_rebuild_inside_the_update_scope():
    from cosmos_rl.utils.distributed import HighAvailabilitylNccl

    comm = HighAvailabilitylNccl.__new__(HighAvailabilitylNccl)
    comm.wait_comm_ready = lambda: None
    comm.is_comm_ready = threading.Event()
    comm.is_comm_ready.set()
    comm.build_mesh_lock = threading.RLock()
    comm._operation_depth = 0
    attempted = threading.Event()
    changed = threading.Event()

    def rebuild():
        attempted.set()
        with comm.build_mesh_lock:
            changed.set()

    with comm.operation_scope():
        thread = threading.Thread(target=rebuild)
        thread.start()
        assert attempted.wait(1) and not changed.wait(0.02)
        with comm.operation_scope():
            assert not changed.is_set()
    thread.join(1)
    assert changed.is_set()


def test_uncertain_collective_is_not_retried_inside_sealed_update(monkeypatch):
    from cosmos_rl.utils import distributed

    comm = distributed.HighAvailabilitylNccl.__new__(distributed.HighAvailabilitylNccl)
    comm.wait_comm_ready = lambda **_: None
    comm.is_comm_ready = threading.Event()
    comm.is_comm_ready.set()
    comm.is_single_peer = threading.Event()
    comm.build_mesh_lock = threading.RLock()
    comm._operation_depth = 0
    comm.default_timeout_ms = 100
    comm.comm_idx = 1
    comm.max_retry = 3
    comm.replica_name = "replica"
    comm.global_rank = 0
    comm.replica_name_to_rank = {"replica": 0}
    comm.api_client = SimpleNamespace(post_nccl_comm_error=lambda *_: None)
    calls = []

    def fail(**_):
        calls.append(True)
        raise RuntimeError("injected native failure")

    monkeypatch.setattr(distributed, "nccl_timeout_watchdog", lambda **_: nullcontext())
    monkeypatch.setattr(distributed, "nccl_allreduce", fail)
    with pytest.raises(RuntimeError, match="sealed update|completion is uncertain"):
        with comm.operation_scope():
            comm.allreduce(torch.tensor(1), torch.tensor(0), dist.ReduceOp.SUM)
    assert calls == [True] and not comm.is_comm_ready.is_set()


def test_collation_cannot_fabricate_missing_targets():
    from cosmos_rl.policy.trainer.llm_trainer.grpo_trainer import GRPOTrainer

    trainer = SimpleNamespace(
        config=SimpleNamespace(
            train=SimpleNamespace(
                train_policy=SimpleNamespace(collect_rollout_logprobs=False)
            )
        )
    )
    with pytest.raises(ValueError, match="agreed before collation"):
        GRPOTrainer.collate_teacher_logprobs(
            trainer, [Rollout(prompt_idx=0)], [object()], 1
        )


@pytest.mark.parametrize("logging_sinks", [[], ["console"], ["wandb"]])
def test_actual_controller_ack_retires_skips_before_next_healthy_update(
    monkeypatch, logging_sinks
):
    from cosmos_rl.dispatcher import status

    manager = status.PolicyStatusManager()
    manager.config = SimpleNamespace(
        mode="disaggregated",
        train=SimpleNamespace(
            train_policy=SimpleNamespace(type="grpo", on_policy=False),
            train_batch_per_replica=2,
        ),
        validation=SimpleNamespace(enable=True),
        logging=SimpleNamespace(logger=logging_sinks),
    )
    manager.total_steps = 10
    replicas = [
        SimpleNamespace(name=name, start_time=0, all_atoms_arrived=True, in_mesh=True)
        for name in ("p0", "p1")
    ]
    manager.policy_replicas = {replica.name: replica for replica in replicas}
    manager.should_weight_sync_after_train_ack = lambda *_: False
    manager.try_trigger_data_fetch_and_training = Mock()
    records = []
    manager.custom_logger_fns = [
        Mock(side_effect=RuntimeError("injected logging failure")),
        lambda report, step: records.append((step, dict(report))),
    ]
    monkeypatch.setattr(status, "log_wandb", Mock())
    for step in (1, 2, 3):
        manager.current_step = step
        manager.status = {
            replica.name: status.PolicyStatus.RUNNING for replica in replicas
        }
        manager.samples_on_the_fly = 4
        manager.dispatched_rollouts_by_step[step] = 4
        # The dispatch follow-up additionally seals actual ACK recipients.
        seal = getattr(manager, "_seal_training_dispatch", None)
        if seal is not None:
            seal(replicas, 10, 4)
        manager.train_report_data[step] = {
            "train/reward_mean": 0.0,
            "train/reward_std": 0.0,
            "train/reward_max": 0.0,
            "train/reward_min": 0.0,
            "rollout/completion_length_mean": 1.0,
            "rollout/completion_length_max": 1,
        }
        if step == 2:
            report = {"train_step": step, "train/teacher_update_skipped": 1}
        else:
            report = {
                "train_step": step,
                "train/loss_avg": 3.0,
                "train/loss_max": 3.0,
                "train/learning_rate": 0.1,
                "train/iteration_time": 1.0,
            }
        for replica in replicas:
            manager.train_ack(
                replica.name,
                step,
                10,
                False,
                report,
                SimpleNamespace(replica_scaling_log={}),
            )
        assert manager.samples_on_the_fly == 0
        assert manager.all_ready()
        assert manager.report_data_list == []
        if step == 2:
            assert records[-1][0] == 2
            assert records[-1][1]["train/teacher_update_skipped"] == 1
            assert "train/loss_avg" not in records[-1][1]
        elif logging_sinks:
            assert manager.report_data_list == []
            assert manager.train_report_data[step]["train/loss_avg"] == 3.0
            assert "train/teacher_update_skipped" not in manager.train_report_data[step]
    assert manager.try_trigger_data_fetch_and_training.call_count == 3
