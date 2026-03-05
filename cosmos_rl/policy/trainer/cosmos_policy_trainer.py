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
Minimal trainer for Cosmos-Policy evaluation via the cosmos-rl framework.

Builds :class:`CosmosPolicy` directly (no ``ModelRegistry``), exposes it
as ``self.model`` for the rollout via ``get_policy_model()``, and stubs out
all training-related methods so the colocated loop can proceed without
actually running gradient updates.
"""

from __future__ import annotations

from typing import Optional

import torch

from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.trainer.base import Trainer, TrainerRegistry
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.utils.logging import logger


@TrainerRegistry.register(trainer_type="cosmos_policy")
class CosmosPolicyTrainer(Trainer):
    """Eval-only trainer for Cosmos-Policy.

    Builds ``CosmosPolicy`` from ``config.custom`` and exposes it as
    ``self.model``.  Training methods are no-ops.
    """

    def __init__(
        self,
        config: CosmosConfig,
        parallel_dims: ParallelDims,
        train_stream: Optional[torch.cuda.Stream] = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            parallel_dims=parallel_dims,
            train_stream=train_stream,
            **kwargs,
        )

        from cosmos_rl.policy.model.wfm.cosmos_policy import (
            CosmosPolicy,
            CosmosPolicyConfig,
        )

        custom = getattr(config, "custom", {}) or {}
        valid = {f.name for f in CosmosPolicyConfig.__dataclass_fields__.values()}
        cfg = CosmosPolicyConfig(**{k: v for k, v in custom.items() if k in valid})

        logger.info("[CosmosPolicyTrainer] Building CosmosPolicy ...")
        self.model = CosmosPolicy.from_config(cfg)
        self.model.eval()
        logger.info("[CosmosPolicyTrainer] Model ready.")

    # ------------------------------------------------------------------
    # Abstract method stubs (eval-only, no training)
    # ------------------------------------------------------------------

    def build_optimizers(self):
        pass

    def build_lr_schedulers(self):
        pass

    def step_training(self, *args, **kwargs):
        return {}

    def step_validation(self, *args, **kwargs):
        return {}

    def export_safetensors(
        self, output_dir, rel_path, trainable_only=False, is_final=False, dtype=None
    ):
        pass

    def model_load_from_hf(self):
        pass

    def model_resume_from_checkpoint(self, *args, **kwargs):
        pass
