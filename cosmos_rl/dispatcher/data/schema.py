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

from typing import List, Any, Dict, Optional, Tuple
from pydantic import BaseModel, Field, model_validator

ConversationType = List["ChatMessage"]

# When we use iter(dataset), we can get the index of the payload in this way
IdxAndRLPayload = Tuple[int, "RLPayload"]


class ChatMessage(BaseModel):
    """
    A chat message item of a conversation.
    """

    role: str = Field(default=None, choices=["system", "user", "assistant"])

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


class RLPayload(BaseModel):
    """
    The payload schema of RL sample.
    """

    prompt: Optional[str] = Field(
        default=None, description="The prompt for the rollout."
    )

    conversation: Optional[List[ChatMessage]] = Field(
        default=None, description="The conversation for the rollout."
    )

    reference_answer: Optional[str] = Field(
        default=None, description="The reference answer for the rollout."
    )

    weight_version: int = Field(
        default=0, description="The weight version for the rollout."
    )

    @model_validator(mode="after")
    def check_params_value(self):
        assert self.prompt or self.conversation, "Must set prompt or conversation"
        return self

    @staticmethod
    def collate_fn(
        batch: List[IdxAndRLPayload],
    ) -> tuple[List[int], List["RLPayload"]]:
        idx_list = []
        payload_list = []

        for idx, payload in batch:
            idx_list.append(idx)
            payload_list.append(payload)

        return idx_list, payload_list
