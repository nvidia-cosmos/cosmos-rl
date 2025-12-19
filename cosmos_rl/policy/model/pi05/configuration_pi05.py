# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal, Optional

from transformers import PretrainedConfig


class Pi05Config(PretrainedConfig):
    """
    HuggingFace-native config for Cosmos-RL PI05 model.

    This enables:
      - AutoConfig.from_pretrained(<repo_or_path>) to work when config.json contains
        {"model_type": "pi05", ...} (and also "pi0" when registered to this same config class)
      - Cosmos-RL ModelRegistry.build_model to route model_type == "pi05" to PI05.
    """

    model_type = "pi05"
    # Specify tokenizer class since Pi05 uses Gemma-based language model
    tokenizer_class = "GemmaTokenizer"

    def __init__(
        self,
        *,
        pi05: bool = True,
        action_dim: int = 32,
        action_horizon: int = 10,
        paligemma_variant: str = "gemma_2b",
        action_expert_variant: str = "gemma_300m",
        dtype: Literal["bfloat16", "float32"] = "bfloat16",
        # Standard HF fields (optional)
        torch_dtype: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(torch_dtype=torch_dtype, **kwargs)
        self.pi05 = pi05
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.paligemma_variant = paligemma_variant
        self.action_expert_variant = action_expert_variant
        self.dtype = dtype


