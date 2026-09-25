# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PP microbatching must preserve the existing non-PP minibatch objective."""

import os
import ast
import math
import __future__
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import cosmos_rl

from cosmos_rl.policy.kernel.loss import CrossEntropyLoss
from cosmos_rl.policy.trainer.llm_trainer.sft_trainer import SFTTrainer, async_safe_ce
from cosmos_rl import patch as pipeline_patch


@pytest.mark.parametrize("microbatch", [1, 2])
@pytest.mark.parametrize("unequal_lengths", [False, True])
def test_sft_pipeline_loss_and_gradient_match_unsplit_minibatch(
    monkeypatch, microbatch, unequal_lengths
):
    monkeypatch.setattr(torch, "compile", lambda function: function)
    device = torch.device(os.environ.get("COSMOS_ALIGNMENT_DEVICE", "cpu"))
    if device.type == "cuda":
        assert torch.cuda.is_available()
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            train=SimpleNamespace(
                train_batch_per_replica=8,
                train_policy=SimpleNamespace(
                    mini_batch=4,
                    enable_dp_load_balancing=False,
                    balance_dp_token=False,
                ),
            ),
            policy=SimpleNamespace(
                parallelism=SimpleNamespace(pp_micro_batch_size=microbatch)
            ),
        ),
        parallel_dims=SimpleNamespace(dp_shard_enabled=False, cp_enabled=False),
    )
    labels = torch.tensor([[0, 1, 2, 3, 4]] * 4, device=device)
    if unequal_lengths:
        labels[1, 2:] = -100
        labels[2, 3:] = -100
        labels[3, 1:] = -100
    logits = (
        torch.linspace(-3, 4, 140, device=device)
        .sin()
        .reshape(4, 5, 7)
        .requires_grad_()
    )
    reference = logits.detach().clone().requires_grad_()
    # This conditional lets the same negative control run against the prior head.
    if hasattr(SFTTrainer, "_prepare_pp_loss"):
        SFTTrainer._prepare_pp_loss(trainer, labels, loss_scaling_factor=0.5)
    loss_fn = SFTTrainer.pp_loss_fn.fget(trainer)
    actual = sum(
        loss_fn(logits[start : start + microbatch], labels[start : start + microbatch])
        for start in range(0, 4, microbatch)
    )
    expected = async_safe_ce(
        reference, labels, CrossEntropyLoss(), loss_scaling_factor=0.5
    )
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, reference.grad)


def test_scalar_only_correction_does_not_preserve_unequal_token_means():
    logits = torch.linspace(-3, 4, 140).sin().reshape(4, 5, 7)
    labels = torch.tensor([[0, 1, 2, 3, 4]] * 4)
    labels[1, 2:] = -100
    labels[2, 3:] = -100
    labels[3, 1:] = -100
    reference = async_safe_ce(
        logits, labels, CrossEntropyLoss(), loss_scaling_factor=0.5
    )
    scalar_only = sum(
        async_safe_ce(
            logits[start : start + 1],
            labels[start : start + 1],
            CrossEntropyLoss(),
            loss_scaling_factor=1 / 8,
        )
        for start in range(4)
    )
    assert not torch.allclose(scalar_only, reference)


@pytest.mark.parametrize("version", ["2.7.0", "2.9.1+cu128", "2.10.0"])
@pytest.mark.parametrize("name", ["Schedule1F1B", "ScheduleGPipe"])
def test_schedule_preserves_existing_scaling_defaults_by_capability(
    monkeypatch, version, name
):
    calls = []

    def initialize(self, *args, scale_grads=True):
        calls.append(scale_grads)

    monkeypatch.setattr(torch, "__version__", version)
    monkeypatch.setattr(
        getattr(pipeline_patch, "Original" + name), "__init__", initialize
    )
    getattr(pipeline_patch, name)()
    assert calls == [name == "ScheduleGPipe"]


def test_schedule_supports_older_constructor_without_scale_grads(monkeypatch):
    calls = []

    def initialize(self):
        calls.append(True)

    monkeypatch.setattr(pipeline_patch.OriginalSchedule1F1B, "__init__", initialize)
    pipeline_patch.Schedule1F1B()
    assert calls == [True]


@pytest.mark.parametrize("scaling", [None, False, True])
def test_builder_only_overrides_gradient_scaling_when_requested(monkeypatch, scaling):
    from cosmos_rl.utils.pipelining import pipelining_utils as pp

    class Schedule(pp.PipelineScheduleSingle):
        def __init__(self, stage, n_microbatches, loss_fn, scale_grads=True):
            self.scaling = scale_grads

        def _step_microbatches(self, *args, **kwargs):
            raise AssertionError("Constructor-only test")

    monkeypatch.setattr(pp, "get_schedule_class", lambda name: Schedule)
    monkeypatch.setattr(pp, "PipelineStage", lambda *args, **kwargs: object())
    schedule = pp.build_pipeline_schedule(
        pp_mesh=SimpleNamespace(
            get_local_rank=lambda: 0, size=lambda: 2, get_group=lambda: None
        ),
        batch_size=4,
        num_stages=2,
        schedule_str="test",
        microbatch_size=1,
        model_parts=[torch.nn.Identity()],
        device=torch.device("cpu"),
        loss_fn=lambda output, target: output.sum(),
        scale_grads=scaling,
    )
    assert schedule.scaling is (True if scaling is None else scaling)


def test_neighbor_initialization_uses_matching_wire_dtypes(monkeypatch):
    ops = [
        SimpleNamespace(tensor=torch.zeros(1)),
        SimpleNamespace(tensor=torch.tensor(1)),
    ]
    monkeypatch.setattr(
        pipeline_patch.OriginalPipelineStage,
        "_get_init_p2p_neighbors_ops",
        lambda self: ops,
        raising=False,
    )
    stage = object.__new__(pipeline_patch.PipelineStage)
    actual = stage._get_init_p2p_neighbors_ops()
    assert all(
        op.tensor.dtype == torch.int64 and op.tensor.numel() == 1 for op in actual
    )
    assert [op.tensor.item() for op in actual] == [0, 1]


def test_forward_only_stage_does_not_execute_legacy_backward_action(monkeypatch):
    calls = []
    monkeypatch.setattr(
        pipeline_patch.OriginalPipelineStage,
        "backward_one_chunk",
        lambda self, *a, **kw: calls.append(True),
    )
    stage = object.__new__(pipeline_patch.PipelineStage)
    stage.patched = False
    stage.has_backward = False
    stage.backward_one_chunk(0)
    assert not calls
    stage.has_backward = True
    stage.backward_one_chunk(0)
    assert calls == [True]


def test_dynamic_schedule_sets_backward_flag_at_each_step():
    states = []
    stage = SimpleNamespace(
        has_backward=False, clear_runtime_states=lambda: None, is_last=False
    )
    schedule = SimpleNamespace(
        _stage=stage,
        _has_backward=True,
        _n_microbatches=1,
        pp_dynamic_shape_enabled=False,
        _split_inputs=lambda args, kwargs: ([args], [kwargs]),
        _step_microbatches=lambda *args: states.append(stage.has_backward),
    )
    for backward in (True, False, True):
        schedule._has_backward = backward
        pipeline_patch.step_func(schedule, position_ids=torch.zeros(1, 3))
    assert states == [True, False, True]
    with torch.no_grad(), pytest.raises(RuntimeError, match="gradients"):
        pipeline_patch.step_func(schedule)


@pytest.mark.parametrize(
    "method,forward_result",
    [
        ("backward_weight_one_chunk", None),
        ("get_bwd_recv_ops", []),
        ("get_bwd_send_ops", []),
        ("scale_grads", None),
    ],
)
def test_forward_only_stage_skips_all_backward_actions(
    monkeypatch, method, forward_result
):
    calls = []
    sentinel = object()

    def original(self, *args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(pipeline_patch.OriginalPipelineStage, method, original)
    stage = object.__new__(pipeline_patch.PipelineStage)
    stage.has_backward = False
    action = getattr(stage, method)
    assert action(0) == forward_result
    assert not calls
    stage.has_backward = True
    assert action(0) is sentinel
    assert calls == [((0,), {})]


def test_schedule_switch_invalidates_shared_stage_infrastructure():
    states = []
    stage = SimpleNamespace(
        has_backward=False, clear_runtime_states=lambda: None, is_last=False
    )

    def make(backward):
        schedule = SimpleNamespace(
            _stage=stage,
            _has_backward=backward,
            _stage_initialized=False,
            _n_microbatches=2,
            pp_dynamic_shape_enabled=False,
            _split_inputs=lambda args, kwargs: ([args], [kwargs]),
        )

        def step(*args):
            states.append((schedule._stage_initialized, stage.has_backward))
            schedule._stage_initialized = True

        schedule._step_microbatches = step
        return schedule

    train, validation = make(True), make(False)
    for schedule in (train, train, validation, train):
        pipeline_patch.step_func(schedule, position_ids=torch.zeros(1, 3))
    assert states == [(False, True), (True, True), (False, False), (False, True)]


@pytest.mark.parametrize("empty", [False, True])
def test_sft_whole_minibatch_dp_count_is_reduced_once(monkeypatch, empty):
    monkeypatch.setattr(torch, "compile", lambda function: function)
    group = SimpleNamespace(size=lambda: 2)
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            train=SimpleNamespace(train_policy=SimpleNamespace(balance_dp_token=True))
        ),
        parallel_dims=SimpleNamespace(
            cp_enabled=False,
            dp_shard_enabled=True,
            mesh={"dp_shard": SimpleNamespace(get_group=lambda: group)},
        ),
    )
    labels = torch.tensor([[0, 1, 2], [0, -100, -100]])
    if empty:
        labels.fill_(-100)
    reductions = []

    def reduce_count(tensor, *, group):
        reductions.append(int(tensor))
        tensor.add_(3)

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce_count)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    SFTTrainer._prepare_pp_loss(trainer, labels, loss_scaling_factor=0.5)
    loss_fn = SFTTrainer.pp_loss_fn.fget(trainer)
    logits = torch.linspace(-2, 2, 18).sin().reshape(2, 3, 3).requires_grad_()
    reference = logits.detach().clone().requires_grad_()
    actual = sum(loss_fn(logits[i : i + 1], labels[i : i + 1]) for i in range(2))
    expected = torch.nn.functional.cross_entropy(
        reference[:, :-1].reshape(-1, 3), labels[:, 1:].reshape(-1), reduction="sum"
    ) / ((labels[:, 1:] != -100).sum() + 3)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, reference.grad)
    assert reductions == [0 if empty else 2]


@pytest.mark.parametrize("validation", [False, True])
@pytest.mark.parametrize("policy", ["sft", "grpo"])
def test_deepseek_schedule_uses_actual_training_and_validation_batches(
    monkeypatch, validation, policy
):
    from cosmos_rl.utils.pipelining import pipelining_utils

    # Execute the real wiring function without importing optional MoE/TE kernels.
    # Only model partitioning/FSDP and the schedule constructor are stand-ins;
    # the native schedule builder is exercised separately by the two-rank canary.
    path = (
        Path(cosmos_rl.__file__).resolve().parent
        / "policy/model/deepseek_v3/parallelize.py"
    )
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "parallelize"
    )
    part = SimpleNamespace()
    mesh = object()
    namespace = {
        "os": os,
        "math": math,
        "torch": torch,
        "_init_meshes": lambda dims: {"default": {"pp": mesh}},
        "_get_device_info": lambda: ("cpu", None),
        "pipeline_model": lambda *args: [part],
        "_apply_fsdp": lambda *args: None,
        "pre_parallelize_sanity_check": lambda function: function,
    }
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    calls = []
    monkeypatch.setattr(
        pipelining_utils,
        "build_pipeline_schedule",
        lambda **kwargs: calls.append(kwargs) or kwargs,
    )
    monkeypatch.setattr(
        pipelining_utils, "generate_split_points", lambda **kwargs: ["a", "b", "c"]
    )
    config = SimpleNamespace(
        train=SimpleNamespace(
            train_batch_per_replica=16,
            train_policy=SimpleNamespace(mini_batch=4, type=policy),
        ),
        policy=SimpleNamespace(
            model_gradient_checkpointing=False,
            parallelism=SimpleNamespace(pp_micro_batch_size=2),
        ),
        validation=SimpleNamespace(enable=validation, batch_size=6),
    )
    dims = SimpleNamespace(
        ep=1,
        tp=1,
        pp=2,
        pp_enabled=True,
        ep_enabled=False,
        cp_enabled=False,
        pp_schedule="1F1B",
        pp_layers_per_stage=None,
    )
    model = SimpleNamespace(
        config=SimpleNamespace(n_routed_experts=1, n_dense_layers=1, n_layers=2)
    )
    loss = object()
    train, val = namespace["parallelize"](model, dims, config, loss)
    is_sft = policy == "sft"
    assert train["batch_size"] == 4
    assert train["loss_fn"] is loss
    assert train["scale_grads"] is False
    assert len(calls) == 1 + int(validation or not is_sft)
    if validation or not is_sft:
        assert val["batch_size"] == (6 if is_sft else 4)
        assert val["loss_fn"] is None
    else:
        assert val is None
