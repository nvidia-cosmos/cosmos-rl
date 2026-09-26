# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute real TRT validation control flow without importing the TRT engine."""

import ast
import copy
import threading
from pathlib import Path
from queue import Queue, Empty
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import Mock

import pytest

import cosmos_rl
from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.rollout.validation import ValidationSession
from cosmos_rl.rollout.trtllm_rollout.trtllm_common import (
    ValidationInstruction,
    ShutdownInstruction,
    RolloutWrapperInstruction,
)


def validation_branch():
    path = (
        Path(cosmos_rl.__file__).parent
        / "rollout/trtllm_rollout/trtllm_rollout_wrapper.py"
    )
    tree = ast.parse(path.read_text())
    wrapper = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TRTLLMRolloutWrapper"
    )
    main = next(
        node
        for node in wrapper.body
        if isinstance(node, ast.FunctionDef) and node.name == "main_loop"
    )
    branch = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "self.validation_event.is_set()"
    )
    # Preserve break/continue semantics with a single synthetic iteration.
    body = copy.deepcopy(branch.body)
    function = ast.FunctionDef(
        name="run_validation",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="self")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[
            ast.For(
                target=ast.Name(id="iteration", ctx=ast.Store()),
                iter=ast.Tuple(elts=[ast.Constant(None)], ctx=ast.Load()),
                body=body,
                orelse=[],
            )
        ],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))

    def apply(payload, result, **kwargs):
        payload.completions = result.completions

    namespace = dict(
        Queue=Queue,
        List=List,
        Any=Any,
        RLPayload=RLPayload,
        ValidationSession=ValidationSession,
        normalize_rollout_results=lambda values: values,
        apply_rollout_result_to_payload=apply,
    )
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["run_validation"]


def lifecycle_method(name):
    path = (
        Path(cosmos_rl.__file__).parent
        / "rollout/trtllm_rollout/trtllm_rollout_wrapper.py"
    )
    tree = ast.parse(path.read_text())
    wrapper = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TRTLLMRolloutWrapper"
    )
    method = copy.deepcopy(
        next(
            node
            for node in wrapper.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
    )
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = dict(
        Empty=Empty,
        ShutdownInstruction=ShutdownInstruction,
        ValidationInstruction=ValidationInstruction,
        RolloutWrapperInstruction=RolloutWrapperInstruction,
    )
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize(
    "values, kept",
    [([], []), (["only"], []), (["a", "", "c"], [0, 2]), (["a", "b", "c"], [0, 1, 2])],
)
def test_real_training_branch_preserves_and_settles_original_slots(values, kept):
    from cosmos_rl.dispatcher.protocol import RolloutRequest
    from cosmos_rl.rollout.schema import RolloutResult
    from cosmos_rl.reward.admission import (
        normalize_rollout_results,
        select_rollout_result_completions,
        apply_rollout_result_to_payload,
    )

    path = (
        Path(cosmos_rl.__file__).parent
        / "rollout/trtllm_rollout/trtllm_rollout_wrapper.py"
    )
    wrapper = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "TRTLLMRolloutWrapper"
    )
    method = copy.deepcopy(
        next(
            node
            for node in wrapper.body
            if isinstance(node, ast.FunctionDef) and node.name == "main_loop"
        )
    )
    method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = dict(
        List=List,
        Any=Any,
        RLPayload=RLPayload,
        RolloutResult=RolloutResult,
        RolloutRequest=RolloutRequest,
        logger=Mock(),
        normalize_rollout_results=normalize_rollout_results,
        select_rollout_result_completions=select_rollout_result_completions,
        apply_rollout_result_to_payload=apply_rollout_result_to_payload,
    )
    exec(compile(module, str(path), "exec"), namespace)
    source = RLPayload(
        training_work_id="controller:0", training_completion_slots=[0, 1, 2]
    )
    queued, reports = [], []
    flags = dict(fetch=False, consume=False)
    worker = SimpleNamespace(
        rollout=SimpleNamespace(
            rollout_config=SimpleNamespace(
                multi_turn_config=SimpleNamespace(enable=False)
            ),
            rollout_generation=lambda **kwargs: [RolloutResult(completions=values)],
        ),
        rollout_wrapper_event=threading.Event(),
        cosmos_replica_name_queue=Queue(),
        shutdown_signal=threading.Event(),
        validation_event=threading.Event(),
        replica_name="source",
        _prompt_queue=Queue(),
        batch_size=1,
        sampling_params=None,
        data_packer=None,
        data_fetcher=None,
        config=SimpleNamespace(
            train=SimpleNamespace(train_policy=SimpleNamespace(bypass_reward=False))
        ),
        state=SimpleNamespace(
            prompt_fetch_end=lambda: flags["fetch"],
            prompt_consume_end=lambda: flags["consume"],
            set_prompt_fetch_end=lambda: flags.update(fetch=True),
            set_prompt_consume_end=lambda: flags.update(consume=True),
        ),
        consume_lifecycle_instruction=lambda: None,
        bind_report_source=lambda source: None,
        report_rollouts=lambda: (None, False, None, True),
        reward_dispatcher=SimpleNamespace(
            enqueue_rewards_cal=lambda payloads, *args, **kwargs: queued.extend(
                payloads
            )
        ),
        api_client=SimpleNamespace(
            post_rollout_completion=lambda request: reports.append(request)
        ),
    )
    worker.cosmos_replica_name_queue.put({"replica_name": "source"})
    worker.rollout_wrapper_event.set()
    worker.request_new_prompts = lambda size, queue: (queue.put([source]), True)[1]
    worker.send_end_signal = lambda: (worker.shutdown_signal.set(), True)[1]
    namespace["main_loop"](worker)
    assert [
        slot for payload in queued for slot in payload.training_completion_slots
    ] == kept
    assert [value for payload in queued for value in payload.completions] == [
        values[index] for index in kept
    ]
    assert [r.slot for report in reports for r in report.training_rejections] == [
        i for i in range(3) if i not in kept
    ]
    assert all(
        report.metrics["discarded_samples"] == len(report.training_rejections)
        for report in reports
    )


@pytest.mark.parametrize("batches", [0, 1, 3])
@pytest.mark.parametrize("end_with_payload", [False, True])
@pytest.mark.parametrize("shutdown_after", [False, True])
def test_each_generated_validation_prompt_is_reported_once(
    batches, end_with_payload, shutdown_after
):
    fetched, pending, reports = [], [], []

    def request(n, queue, **kwargs):
        index = len(fetched)
        if index == batches:
            return True
        payload = RLPayload(prompt_idx=index, prompt=str(index))
        fetched.append(payload)
        queue.put([payload])
        return end_with_payload and index == batches - 1

    def enqueue(payloads, is_validation, step):
        if payloads:
            pending.append(
                (
                    [payload.model_copy(deep=True) for payload in payloads],
                    is_validation,
                    step,
                    False,
                )
            )

    def dequeue(**kwargs):
        return pending.pop(0) if pending else (None, True, 1, True)

    worker = SimpleNamespace(
        validation_round_id="test-round",
        _lifecycle_commands=Queue(),
        shutdown_signal=Mock(),
        shutdown_mp_signal=Mock(),
        validation_event=Mock(wraps=threading.Event()),
        validation_step=1,
        val_batch_size=2,
        request_new_prompts=request,
        rollout=SimpleNamespace(
            rollout_generation=lambda **kwargs: [
                SimpleNamespace(completions=["answer"], completed_conversations=None)
            ]
        ),
        val_data_packer=None,
        data_fetcher=None,
        val_sampling_params=None,
        reward_dispatcher=SimpleNamespace(
            enqueue_rewards_cal=enqueue,
            dequeue_rewards_cal=dequeue,
        ),
        report_rollouts=dequeue,
        replica_name="replica",
        api_client=SimpleNamespace(post_validation_report=reports.append),
    )
    if shutdown_after:
        worker._lifecycle_commands.put(ShutdownInstruction())
    validation_branch()(worker)
    actual = [payload.prompt_idx for report in reports for payload in report.payloads]
    assert actual == list(range(batches)), actual
    assert reports[-1].is_end and reports[-1].payloads == []
    assert sum(report.is_end for report in reports) == 1
    assert [report.report_sequence for report in reports] == list(range(len(reports)))
    assert all(report.validation_round_id == "test-round" for report in reports)
    worker.shutdown_signal.set.assert_not_called()
    worker.shutdown_mp_signal.set.assert_not_called()
    lifecycle_method("consume_lifecycle_instruction")(worker)
    assert worker.shutdown_signal.set.call_count == int(shutdown_after)
    assert worker.shutdown_mp_signal.set.call_count == int(shutdown_after)
    worker.validation_event.clear.assert_called_once()


def test_ipc_cannot_overwrite_active_validation_or_overtake_it_with_shutdown():
    event = threading.Event()
    event.set()
    worker = SimpleNamespace(
        validation_event=event,
        validation_step=1,
        validation_round_id="first",
        cosmos_weight_sync_queue=Queue(),
        _lifecycle_commands=Queue(),
        shutdown_signal=threading.Event(),
        shutdown_mp_signal=threading.Event(),
        rollout_wrapper_event=threading.Event(),
    )
    worker.cosmos_weight_sync_queue.put(ValidationInstruction(2, 2, "next"))
    worker.cosmos_weight_sync_queue.put(ShutdownInstruction())
    worker.cosmos_weight_sync_queue.put(None)
    lifecycle_method("life_control_loop")(worker)
    assert worker.validation_step == 1 and worker.validation_round_id == "first"
    lifecycle_method("consume_lifecycle_instruction")(worker)
    assert worker._lifecycle_commands.qsize() == 2
    assert not worker.shutdown_signal.is_set()
    event.clear()  # Current round has finished delivering its final receipt.
    lifecycle_method("consume_lifecycle_instruction")(worker)
    assert worker.validation_step == 2 and worker.validation_round_id == "next"
    assert event.is_set() and not worker.shutdown_signal.is_set()
    event.clear()
    lifecycle_method("consume_lifecycle_instruction")(worker)
    assert worker.shutdown_signal.is_set() and worker.shutdown_mp_signal.is_set()
