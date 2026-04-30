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

"""Identity ``WeightMapper`` for non-LLM RL policies.

Cosmos-RL's :class:`~cosmos_rl.policy.model.base.WeightMapper` is built
around HuggingFace causal-LM conventions: parameter renaming, fused-KV
splitting, expert-stacking for MoE, and so on.  Small RL policies (an
MLP for a Gymnasium task, for example) do not need any of this — every
parameter has the same name on the policy and rollout sides, and there
is no tensor-parallel splitting because the world size is typically 1.

:class:`IdentityWeightMapper` provides a one-line drop-in for that case.
Combine it with :func:`cosmos_rl.policy.model.base.ModelRegistry.register_model`
to register an arbitrary :class:`torch.nn.Module`-derived RL policy with
cosmos-rl::

    from cosmos_rl.policy.model.base import ModelRegistry
    from cosmos_rl.policy.model.identity_weight_mapper import (
        IdentityWeightMapper,
    )

    class GymMLP(BaseModel):
        @staticmethod
        def supported_model_types():
            return ["gym_mlp"]
        ...

    ModelRegistry.register_model(GymMLP, IdentityWeightMapper)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

try:
    from transformers import AutoConfig
except ImportError:  # pragma: no cover - keeps import-time light when transformers absent
    AutoConfig = None  # type: ignore[assignment]

from cosmos_rl.policy.model.base import WeightMapper
from cosmos_rl.utils.logging import logger


class IdentityWeightMapper(WeightMapper):
    """A trivial :class:`WeightMapper` for single-rank, non-sharded models.

    Behavior:

    * Policy → HF and rollout → HF name maps are identity.
    * No parameter splitting (returns the input ``(name, tensor)``
      unchanged in a single-element list).
    * The base-class implementations of every other hook are accepted
      as-is, which is correct for an MLP-sized model that lives entirely
      on rank 0.

    The constructor accepts ``hf_config=None`` so callers that haven't
    set up a HuggingFace config (the typical Gymnasium case) can still
    instantiate the mapper without contortions.
    """

    def __init__(self, hf_config: Optional["AutoConfig"] = None):
        # Skip the parent constructor: it logs the wrong backend and
        # assumes ``hf_config`` is non-None.  We replicate the minimum
        # state ourselves.
        self.config = hf_config
        self.backend = "vllm"
        logger.info(
            f"[{type(self).__name__}] Initialized identity weight mapper "
            "(non-sharded RL policy)"
        )

    def policy_map_local_key_to_hf_key(self, name: str) -> str:
        return name

    def rollout_map_local_key_to_hf_key(self, name: str) -> str:
        return name

    def rollout_split_local_key_n_param_to_hf_key_n_param(
        self,
        param_name: str,
        param: torch.Tensor,
    ) -> List[Tuple[str, torch.Tensor]]:
        return [(self.rollout_map_local_key_to_hf_key(param_name), param)]


__all__ = ["IdentityWeightMapper"]
