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

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

import torch
from torch.distributed._composable.fsdp import FSDPModule

from cosmos_rl.utils.logging import logger


class FastRefModelUpdater:
    """
    This class is used to manage reference model for GRPO training.
    Similar to FastEmaModelUpdater but simpler - it just copies weights from current model to reference model
    when needed, and provides context switching capabilities.
    """

    def __init__(self):
        # Flag to indicate whether the cache is taken or not. Useful to avoid cache overwrite
        self.is_cached = False

    def copy_to(self, src_model: torch.nn.Module, tgt_model: torch.nn.Module) -> None:
        """Copy weights from source model to target model."""
        for tgt_params, src_params in zip(
            tgt_model.parameters(), src_model.parameters()
        ):
            tgt_params.data.copy_(src_params.data)

    def cache(self, parameters: Any, is_cpu: bool = False) -> None:
        """Save the current parameters for restoring later.

        Args:
            parameters (iterable): Iterable of torch.nn.Parameter to be temporarily stored.
            is_cpu (bool): Whether to store the cache on CPU.
        """
        assert (
            self.is_cached is False
        ), "Ref cache is already taken. Did you forget to restore it?"
        device = "cpu" if is_cpu else "cuda"
        self.collected_params = [param.clone().to(device) for param in parameters]
        self.is_cached = True

    def restore(self, parameters: Any) -> None:
        """Restore the parameters in self.collected_params.

        Useful to switch to reference model parameters without affecting the
        original optimization process. Store the parameters before copy_to().
        After inference, use this to restore the current parameters.

        Args:
            parameters (iterable): Iterable of torch.nn.Parameter to be updated with the stored parameters.
        """
        assert self.is_cached, "Ref cache is not taken yet."
        for c_param, param in zip(self.collected_params, parameters, strict=False):
            param.data.copy_(c_param.data.type_as(param.data))
        self.collected_params = []
        # Release the cache after we call restore
        self.is_cached = False


class DTensorFastRefModelUpdater:
    """
    DTensor version of FastRefModelUpdater for distributed training.
    Similar to DTensorFastEmaModelUpdater but for reference model management.
    """

    def __init__(self):
        # Flag to indicate whether the cache is taken or not. Useful to avoid cache overwrite
        self.is_cached = False

    def copy_to(self, src_model: torch.nn.Module, tgt_model: torch.nn.Module) -> None:
        """Copy weights from source model to target model using DTensor local views."""
        with torch.no_grad():
            for tgt_params, src_params in zip(
                tgt_model.parameters(), src_model.parameters()
            ):
                tgt_params.to_local().data.copy_(src_params.to_local().data)

    @torch.no_grad()
    def cache(self, parameters: Any, is_cpu: bool = False) -> None:
        """Cache current parameters using DTensor local views."""
        assert (
            self.is_cached is False
        ), "Ref cache is already taken. Did you forget to restore it?"
        device = "cpu" if is_cpu else "cuda"
        self.collected_params = [
            param.to_local().clone().to(device) for param in parameters
        ]
        self.is_cached = True

    @torch.no_grad()
    def restore(self, parameters: Any) -> None:
        """Restore cached parameters using DTensor local views."""
        assert self.is_cached, "Ref cache is not taken yet."
        for c_param, param in zip(self.collected_params, parameters, strict=False):
            param.to_local().copy_(c_param.data.type_as(param.data))
        self.collected_params = []
        # Release the cache after we call restore
        self.is_cached = False


@contextmanager
def ref_scope(
    model,
    ref_model: torch.nn.Module,
    ref_worker: FastRefModelUpdater,
    enabled: bool = False,
    context: str = None,
    is_cpu: bool = False,
) -> Generator[None, None, None]:
    """Context manager to switch to reference model weights for GRPO.

    Args:
        model: The main model to switch weights for
        ref_model: The reference model containing the weights to switch to
        ref_worker: The reference model updater utility
        enabled: Whether reference model switching is enabled
        context: Optional context string for logging
        is_cpu: Whether to cache weights on CPU
    """
    if enabled:
        # Handle FSDP resharding
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.reshard()

        # Cache current weights and copy reference weights
        ref_worker.cache(model.parameters(), is_cpu=is_cpu)
        ref_worker.copy_to(src_model=ref_model, tgt_model=model)

        if context is not None:
            logger.info(f"{context}: Switched to reference weights")

    try:
        yield None
    finally:
        if enabled:
            # Handle FSDP resharding
            for module in model.modules():
                if isinstance(module, FSDPModule):
                    module.reshard()

            # Restore original weights
            ref_worker.restore(model.parameters())

            if context is not None:
                logger.info(f"{context}: Restored current weights")
