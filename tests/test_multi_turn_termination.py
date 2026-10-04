# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Execute the actual multi-turn method with controlled engine results."""

import ast
import copy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import List
from unittest.mock import Mock, patch

import pytest
import torch
import cosmos_rl
from cosmos_rl.dispatcher.data.schema import ChatMessage, ConversationType, RLPayload
from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.policy.config import MultiTurnRolloutConfig


def load_method():
    source = Path(cosmos_rl.__file__).parent / "rollout/vllm_rollout/vllm_rollout.py"
    cls = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "vLLMRollout"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "rollout_generation_multi_turn"
    )
    namespace = dict(
        torch=torch,
        copy=copy,
        List=List,
        RLPayload=RLPayload,
        ConversationType=ConversationType,
        RolloutResult=RolloutResult,
        BaseDataPacker=object,
        apply_vllm_gather_logprobs_patch=lambda: None,
        logger=Mock(),
    )
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace[method.name]


def run(
    responses,
    *,
    prompt_length=2,
    max_tokens=8,
    max_turns=3,
    validation=False,
    generations=1,
    model_max_length=512,
    val_max_tokens=None,
    append_empty=True,
):
    engine = Mock()
    calls = []

    def generate(*, prompts, sampling_params, use_tqdm):
        index = len(calls)
        calls.append((copy.deepcopy(prompts), copy.deepcopy(sampling_params)))
        response = responses[min(index, len(responses) - 1)]
        return [
            SimpleNamespace(
                prompt_token_ids=list(range(prompt_length)),
                outputs=[
                    SimpleNamespace(
                        text=response, token_ids=[101, 102], cumulative_logprob=-0.2
                    )
                ],
            )
        ]

    engine.generate.side_effect = generate
    worker = SimpleNamespace(
        _engine_initialized=True,
        rollout_engine=engine,
        sampling_params=SimpleNamespace(n=generations, max_tokens=max_tokens),
        val_sampling_params=SimpleNamespace(
            n=generations, max_tokens=val_max_tokens or max_tokens
        ),
        rollout_config=SimpleNamespace(
            max_response_length=max_tokens,
            multi_turn_config=SimpleNamespace(max_assistant_turns=max_turns),
        ),
        config=SimpleNamespace(
            distillation=SimpleNamespace(top_k=0, rollout_top_k_recompute=False),
            policy=SimpleNamespace(model_max_length=model_max_length),
        ),
        get_prompt_logprobs_and_token_ids=lambda result, **kwargs: ([], []),
        get_completion_logprobs_and_token_ids=lambda result, **kwargs: (
            [[-0.1], [-0.1]],
            [[101], [102]],
        ),
    )

    def extend(conversation, responses, ground_truth):
        if not responses[0] and not append_empty:
            return conversation
        conversation.append(ChatMessage(role="assistant", content=responses[0]))
        if responses[0].startswith("tool:"):
            conversation.append(ChatMessage(role="tool", content="tool observation"))
        return conversation

    packer = SimpleNamespace(
        get_rollout_input=lambda conversation: conversation,
        rollout_collate_fn=lambda items: items,
        extend_conversation=extend,
    )
    payload = RLPayload(
        prompt_idx=0, conversation=[ChatMessage(role="user", content="question")]
    )
    with (
        patch.object(torch.cuda, "current_stream", return_value=None),
        patch.object(torch.cuda, "stream", side_effect=lambda _: nullcontext()),
    ):
        result = load_method()(worker, [payload], None, packer, validation)
    return result, calls, payload


@pytest.mark.parametrize("validation", [False, True])
def test_final_answer_finishes_without_another_assistant_turn(validation):
    result, calls, original = run(["final answer"], validation=validation)
    assert len(calls) == 1
    assert result[0].completions == ["final answer"]
    assert len(original.conversation) == 1


def test_long_prompt_does_not_spend_the_response_token_budget():
    result, calls, _ = run(["tool: calculate", "final answer"], prompt_length=100)
    assert len(calls) == 2
    assert result[0].completions == ["final answer"]


def test_turn_limit_still_bounds_a_tool_loop():
    _, calls, _ = run(["tool: repeat"], max_turns=2)
    assert len(calls) == 2


def test_one_turn_final_control():
    result, calls, _ = run(["final answer"], max_turns=1)
    assert len(calls) == 1
    assert result[0].completions == ["final answer"]


@pytest.mark.parametrize("validation", [False, True])
def test_turn_limit_rewards_assistant_not_tool_observation(validation):
    result, calls, _ = run(["tool: repeat"], max_turns=1, validation=validation)
    assert len(calls) == 1
    assert result[0].completions == ["tool: repeat"]
    assert result[0].completed_conversations[0][-1].role == "tool"


@pytest.mark.parametrize("prompt_length", [10, 11])
def test_model_context_limit_stops_at_boundary(prompt_length):
    result, calls, _ = run(
        ["tool: repeat"],
        prompt_length=prompt_length,
        max_tokens=64,
        model_max_length=12,
    )
    assert len(calls) == 1
    assert result[0].completions == ["tool: repeat"]


def test_response_cap_is_selected_per_phase_not_spent_by_prompt():
    result, calls, _ = run(
        ["tool: calculate", "answer"],
        prompt_length=100,
        validation=True,
        max_tokens=8,
        val_max_tokens=3,
    )
    assert len(calls) == 2
    assert all(call[1].max_tokens == 3 for call in calls)
    assert result[0].completions == ["answer"]


def test_each_sample_starts_from_its_own_conversation():
    result, calls, original = run(["answer"], generations=2)
    assert len(calls) == 2
    assert all(len(call[0][0]) == 1 for call in calls)
    assert len(original.conversation) == 1
    assert result[0].completions == ["answer", "answer"]


def test_empty_answer_without_packer_extension_is_terminal():
    result, calls, _ = run([""], append_empty=False)
    assert len(calls) == 1
    assert result[0].completions == [""]


@pytest.mark.parametrize("turns", [0, -1])
def test_nonpositive_turn_limit_rejected_before_generation(turns):
    with pytest.raises(ValueError, match="greater than 0"):
        MultiTurnRolloutConfig(max_assistant_turns=turns)
