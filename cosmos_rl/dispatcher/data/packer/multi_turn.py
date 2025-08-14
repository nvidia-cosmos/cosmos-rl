# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
For multi-turn messages we provide some helper functions to check and convert to token ids

For the chat template, we follow the chatml format.
To design a powerfull prompt, you can refer https://huggingface.co/docs/transformers/main/en/tasks/prompting
"""

from transformers import AutoTokenizer
from typing import List
from cosmos_rl.dispatcher.data.schema import ConversationType, ChatMessage


def process_conversation_with_chat_template(
    tokenizer: AutoTokenizer,
    conversation: ConversationType,
    enable_thinking: bool = False,
    tools: list[dict] = None,
) -> tuple[List[int], List[int]]:
    """
    Process the multi-turn conversation to token ids and loss mask.

    Args:
        tokenizer: the tokenizer to use
        conversation: the conversation to process
        enable_thinking: whether to enable thinking
        tools: the tools to use

    Returns:
        concat_tokens: the concatenated tokens
        concat_loss_mask: the concatenated loss mask
    """
    concat_tokens = []
    concat_loss_mask = []

    st = 0
    for i, msg in enumerate(conversation):
        assert msg.role in [
            "user",
            "assistant",
            "system",
            "tool",
        ], "Unknown role: {}".format(msg.role)
        if msg.role == "system":
            assert i == 0, "System message should be the first message"

        # find next assistant message, or process the full sample
        if msg.role != "assistant" and i < len(conversation) - 1:
            st = i
            continue

        if st == 0:
            prev_applied_text = ""
        else:
            prev_applied_text = tokenizer.apply_chat_template(
                conversation[: st + 1],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
                tools=tools,
            )

        cur_applied_text = tokenizer.apply_chat_template(
            conversation[: i + 1],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
            tools=tools,
        )

        # Get tokens for the current message only
        if msg.role == "assistant":
            prev_applied_text_with_generation_prompt = tokenizer.apply_chat_template(
                conversation[: st + 1],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                tools=tools,
            )
            generation_prompt_text = prev_applied_text_with_generation_prompt[
                len(prev_applied_text) :
            ]
            generation_prompt_tokens = tokenizer.encode(
                generation_prompt_text,
                add_special_tokens=False,
            )
            _message_tokens = tokenizer.encode(
                cur_applied_text[len(prev_applied_text) :],
                add_special_tokens=False,
            )
            concat_tokens += generation_prompt_tokens + _message_tokens
            concat_loss_mask += [0] * (len(generation_prompt_tokens)) + [1] * (
                len(_message_tokens)
            )
        else:
            mesage_tokens = tokenizer.encode(
                cur_applied_text[len(prev_applied_text) :],
                add_special_tokens=False,
            )
            concat_tokens += mesage_tokens
            concat_loss_mask += [0] * len(mesage_tokens)
        # move the start index to this processed assistant message
        st = i

    return concat_tokens, concat_loss_mask


def add_user_message(messages: ConversationType, content: str) -> ConversationType:
    """
    Add a user message to the conversation.
    """
    messages += [ChatMessage(role="user", content=content)]
    return messages


def add_assistant_message(messages: ConversationType, content: str) -> ConversationType:
    """
    Add an assistant message to the conversation.
    """
    messages += [ChatMessage(role="assistant", content=content)]
    return messages


def add_tool_response_messages(
    messages: ConversationType, tool_response: str
) -> ConversationType:
    """
    Add a tool response message to the conversation.
    """
    messages += [ChatMessage(role="tool", content=tool_response)]
    return messages
