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

import copy

from transformers import AutoTokenizer
from typing import List, Optional
from cosmos_rl.dispatcher.data.schema import ConversationType, ChatMessage
from cosmos_rl.utils.logging import logger


def get_token_ids_and_loss_mask_from_conversation(
    tokenizer: AutoTokenizer,
    conversation: ConversationType,
    chat_template: Optional[str] = None,
    enable_thinking: bool = False,
    tools: list[dict] = None,
    unmasked_roles: List[str] = ["assistant"],
) -> tuple[List[int], List[int]]:
    """
    Process the multi-turn conversation to token ids and loss mask.

    This function will first replace the unmasked role's message with a placeholder,
    and then replace it with the real message token to generating the final token_ids,
    while also generating an accurate loss_mask.

    Args:
        tokenizer: the tokenizer to use
        conversation: the conversation to process
        enable_thinking: whether to enable thinking
        tools: the tools to use
        unmasked_roles: the roles to unmask, default is only assistant

    Returns:
        concat_tokens_ids: the concatenated tokens
        concat_loss_mask: the concatenated loss mask
    """

    conversation = copy.deepcopy(conversation)
    full_token_ids = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        chat_template=chat_template,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
        tools=tools,
    )
    pad_run_length = 10
    placeholder_tokens = tokenizer.pad_token * pad_run_length
    placeholder_token_ids = [tokenizer.pad_token_id] * pad_run_length

    unmasked_role_contents = []
    # valid the conversation and replace the unmasked roles with placeholder tokens
    for i, msg in enumerate(conversation):
        assert msg.role in [
            "user",
            "assistant",
            "system",
            "tool",
        ], "Unknown role: {}".format(msg.role)
        if msg.role == "system":
            assert i == 0, "System message should be the first message"

        if msg.role in unmasked_roles:
            unmasked_role_contents.append(msg.content)
            msg.content = placeholder_tokens

    if len(unmasked_role_contents) == 0:
        # nothing to replace, return the original token_ids
        return full_token_ids, [0] * len(full_token_ids)

    token_ids_template = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        chat_template=chat_template,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
        tools=tools,
    )

    # Split the token_ids_template by placeholder tokens ids
    token_ids_parts = []
    _idx = 0
    _last_idx = 0
    while _idx < len(token_ids_template) - pad_run_length:
        if token_ids_template[_idx : _idx + pad_run_length] == placeholder_token_ids:
            token_ids_parts.append(token_ids_template[_last_idx:_idx])
            _last_idx = _idx + pad_run_length
            _idx = _last_idx
        else:
            _idx += 1
    token_ids_parts.append(token_ids_template[_last_idx:])

    assert (
        len(token_ids_parts) == len(unmasked_role_contents) + 1
    ), "Number of token ids parts should be the same as the number of unmasked role contents + 1"

    concat_token_ids = []
    concat_loss_mask = []
    for i in range(len(unmasked_role_contents)):
        concat_token_ids.extend(token_ids_parts[i])
        concat_loss_mask.extend([0] * len(token_ids_parts[i]))

        _raw_context_token_ids = tokenizer.encode(
            unmasked_role_contents[i],
            add_special_tokens=False,
        )
        concat_token_ids.extend(_raw_context_token_ids)
        concat_loss_mask.extend([1] * len(_raw_context_token_ids))

    concat_token_ids.extend(token_ids_parts[-1])
    concat_loss_mask.extend([0] * len(token_ids_parts[-1]))

    if full_token_ids != concat_token_ids:
        logger.warning(
            "Full token ids and concat token ids are not the same, use concat token ids instead"
        )
        logger.debug(
            f"Full token ids: {full_token_ids}\nConcat token ids: {concat_token_ids}"
        )
        full_token_ids = concat_token_ids
    return full_token_ids, concat_loss_mask


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
