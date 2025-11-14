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

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel
from cosmos_rl.dispatcher.data.schema import ConversationType


class RolloutResult(BaseModel):
    # The input prompt for the completions
    prompt: Optional[Union[str, ConversationType]] = None

    # The original input prompt in conversation format
    conversation: Optional[ConversationType] = None

    # The generated completions for the prompt, In multi-turn conversation, it is a list of last message for each turn.
    completions: List[str]

    # The generated conversation history for the prompt.
    completed_conversations: Optional[List[ConversationType]] = None
    
    # VLA-specific fields (optional, used by VLA rollouts)
    log_probs: Optional[List[List[float]]] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    rewards: Optional[List[float]] = None
    episode_length: Optional[int] = None
    environment_info: Optional[Dict[str, Any]] = None
    
    # VLA trajectory data (for training)
    # Contains: input_ids, attention_mask, pixel_values, responses, old_log_prob
    vla_trajectory: Optional[Dict[str, Any]] = None  # Contains step-by-step data