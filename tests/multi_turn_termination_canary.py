# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Actual vLLM multi-turn adapter with deterministic tool-continuation fixtures.

Dummy weights exercise engine integration, not model accuracy or tool quality.
The engine generates every response; the packer chooses when to append a tool
observation so termination does not depend on random model text.
"""

import argparse
from types import SimpleNamespace

import torch
from vllm import LLM, SamplingParams

from cosmos_rl.dispatcher.data.schema import ChatMessage, RLPayload
from cosmos_rl.policy.config import Config
from cosmos_rl.rollout.vllm_rollout.vllm_rollout import vLLMRollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    args = parser.parse_args()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    engine = LLM(
        model=args.model,
        load_format="dummy",
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=512,
        gpu_memory_utilization=0.45,
        seed=42,
    )
    tokenizer = engine.get_tokenizer()
    config = Config()
    config.policy.model_max_length = 512
    config.rollout.multi_turn_config.enable = True
    config.rollout.max_response_length = 8
    config.train.train_policy.collect_rollout_logprobs = False
    config.train.train_policy.rollout_as_token_ids = False
    config.distillation.top_k = 0
    worker = object.__new__(vLLMRollout)
    worker.config = config
    worker.rollout_config = config.rollout
    worker._engine_initialized = True
    worker.sampling_params = SamplingParams(
        n=2, max_tokens=8, min_tokens=8, ignore_eos=True, logprobs=0
    )
    worker.val_sampling_params = SamplingParams(
        n=2, max_tokens=4, min_tokens=4, ignore_eos=True, logprobs=0
    )

    for validation in (False, True):
        for scenario, tools, max_turns, expected_turns in (
            ("final", 0, 3, 1),
            ("tool_then_final", 1, 3, 2),
            ("turn_limit", 99, 2, 2),
        ):
            config.rollout.multi_turn_config.max_assistant_turns = max_turns
            outputs = []
            cap = 4 if validation else 8

            def generate(*args, **kwargs):
                assert kwargs["sampling_params"].max_tokens == cap
                result = engine.generate(*args, **kwargs)
                assert len(result) == len(result[0].outputs) == 1
                assert len(result[0].prompt_token_ids) > cap
                assert len(result[0].outputs[0].token_ids) == cap
                outputs.append(result[0].outputs[0].text)
                return result

            def extend(conversation, responses, ground_truth):
                conversation.append(ChatMessage(role="assistant", content=responses[0]))
                turn = sum(message.role == "assistant" for message in conversation)
                if turn <= tools:
                    conversation.append(
                        ChatMessage(role="tool", content="Observed result: 4.")
                    )
                return conversation

            def prepare(conversation):
                tokens = tokenizer.apply_chat_template(
                    [message.model_dump() for message in conversation],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                return {"prompt_token_ids": tokens}

            worker.rollout_engine = SimpleNamespace(generate=generate)
            packer = SimpleNamespace(
                get_rollout_input=prepare,
                rollout_collate_fn=lambda prompts: prompts,
                extend_conversation=extend,
            )
            payload = RLPayload(
                prompt_idx=0,
                conversation=[
                    ChatMessage(role="user", content="Compute two plus two. " * 8)
                ],
            )
            results = worker.rollout_generation_multi_turn(
                [payload], torch.cuda.current_stream(), packer, validation
            )
            assert len(results) == 1 and len(outputs) == 2 * expected_turns
            assert results[0].completions == [outputs[expected_turns - 1], outputs[-1]]
            assert len(payload.conversation) == 1
            for conversation in results[0].completed_conversations:
                assert (
                    sum(message.role == "assistant" for message in conversation)
                    == expected_turns
                )
            print(
                f"MULTI_TURN_ENGINE_PASS validation={validation} case={scenario} "
                f"samples=2 turns={expected_turns} response_cap={cap}",
                flush=True,
            )
    torch.cuda.synchronize()
    print("MULTI_TURN_ENGINE_GATE_PASS", flush=True)


if __name__ == "__main__":
    main()
