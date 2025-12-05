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

from typing import List, Any, Dict, Optional, Tuple, Union
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """
    A chat message item of a conversation.
    """

    role: str = Field(default=None, choices=["system", "user", "assistant", "tool"])

    """
    For text message content,
    ```python
    "What do you see in this video?"
    ```

    For MultiModel message content,
    We support those types of content for multi-model context:
    ```python
    [
        {"type": "text", "text": "What do you see in this video?"},
        {"type": "image", "url": "https://example.com/image.png"},
        {"type": "video", "url": "https://example.com/video.mp4"},
    ]
    ```
    """
    content: str | List[Dict[str, Any]] = ""


ConversationType = List[ChatMessage]


class RLPayload(BaseModel):
    """
    The payload schema of RL sample.
    """

    prompt: Optional[Union[ConversationType, str]] = Field(
        default=None, description="The input prompt for the rollout."
    )

    prompt_idx: int = Field(
        default=-1, description="The index of the prompt for the rollout."
    )

    conversation: Optional[ConversationType] = Field(
        default=None, description="The input conversation for the rollout."
    )

    reference_answer: Optional[str] = Field(
        default=None, description="The reference answer for the rollout."
    )

    weight_version: int = Field(
        default=0, description="The weight version for the rollout."
    )

    # For rollout generation result, we add following fields:
    # In tensor native, video, or any other mode, it can be a list of any type of objects.
    # The object type can be defined by the `rollout_generation` implementation.
    # For non-text objects, it will be converted by the `get_rollout_output` of `data_packer` into final serializable format.
    completions: Optional[List[Union[str, Any]]] = Field(
        default=None,
        description="The generated completions for the prompt, In multi-turn conversation, it is a list of last message for each turn.",
    )

    completed_conversations: Optional[List[ConversationType]] = Field(
        default=None,
        description="The original input conversation for the rollout, In multi-turn conversation, it is a list of conversation history for each turn.",
    )

    n_ignore_prefix_tokens: Optional[List[int]] = Field(
        default=None,
        description="The number of prefix tokens to ignore when computing reward.",
    )

    rewards: Optional[List[float]] = Field(
        default=None, description="The reward for each completion."
    )

    advantages: Optional[List[float]] = Field(
        default=None, description="The advantage for each completion."
    )

    # Whether the rollout is valid for dynamic sampling
    valid: Optional[bool] = Field(
        default=True, description="Whether the rollout is valid."
    )

    # For metrics collection
    filter_rewards: Optional[List[float]] = Field(
        default=None, description="The filter reward for each completion."
    )

    completion_token_ids: Optional[List[List[int]]] = Field(
        default=None, description="The token ids of each completion."
    )

    completion_logprobs: Optional[List[List[float]]] = Field(
        default=None, description="The logprobs of each completion."
    )

    @staticmethod
    def collate_fn(
        batch: List["IdxAndRLPayload"],
    ) -> tuple[List[int], List["RLPayload"]]:
        idx_list = []
        payload_list = []

        for idx, payload in batch:
            idx_list.append(idx)
            payload_list.append(payload)

        return idx_list, payload_list


# When we use iter(dataset), we can get the index of the payload in this way
IdxAndRLPayload = Tuple[int, RLPayload]


class Rollout(BaseModel):
    prompt: Optional[Union[ConversationType, str]] = Field(
        default=None, description="The input prompt for the rollout."
    )

    prompt_idx: int = Field(
        default=-1, description="The index of the prompt for the rollout."
    )

    conversation: Optional[ConversationType] = Field(
        default=None, description="The input conversation for the rollout."
    )

    completion: str = Field(
        default="", description="The generated completion for the rollout."
    )

    completed_conversation: Optional[ConversationType] = Field(
        default=None, description="The generated conversation for the rollout."
    )

    is_end: bool = Field(
        default=False, description="Whether the rollout is the last one."
    )

    reward: float = Field(default=0.0, description="The reward for the rollout.")

    advantage: float = Field(default=0.0, description="The advantage for the rollout.")

    n_ignore_prefix_tokens: int = 0

    filter_reward: float = Field(
        default=0.0, description="The filter reward for the rollout."
    )

    completion_token_ids: Optional[List[int]] = Field(
        default=None, description="The token ids of each completion."
    )

    completion_logprobs: Optional[List[float]] = Field(
        default=None, description="The logprobs of each completion."
    )

    weight_version: int = Field(
        default=0, description="The weight version for the rollout."
    )
