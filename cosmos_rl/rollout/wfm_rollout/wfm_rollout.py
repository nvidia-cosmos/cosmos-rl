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

import torch

from typing import Optional, Tuple, List, Callable

from cosmos_rl.rollout.rollout_base import RolloutBase
from cosmos_rl.policy.config.wfm import CosmosVisionGenConfig

from cosmos_rl.utils.logging import logger

from cosmos_rl.policy.model.wfm.sampler import (
    Sampler,
    COMMON_SOLVER_OPTIONS,
)

from cosmos_rl.utils.wfm.utils import (
    arch_invariant_rand,
)

from cosmos_rl.utils.wfm.context_parallel import (
    broadcast_split_tensor,
    cat_outputs_cp,
)


class WFMRollout(RolloutBase):
    def __init__(self, config: CosmosVisionGenConfig):
        super().__init__(config, None, torch.cuda.current_device())

    def post_init_hook(self, **kwargs):
        pass

    def rollout_generation(
        self,
        data_batch: dict[str, torch.Tensor],
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Optional[Tuple] = None,
        n_sample: Optional[int] = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        solver_option: COMMON_SOLVER_OPTIONS = "2ab",
        x_sigma_max: Optional[torch.Tensor] = None,
        sigma_max: Optional[float] = None,
        callback_fns: Optional[List[Callable]] = None,
    ):
        self.model._normalize_video_databatch_inplace(data_batch)
        self.model._augment_image_dim_inplace(data_batch)
        is_image_batch = self.model.is_image_batch(data_batch)
        input_key = (
            self.model.input_image_key if is_image_batch else self.model.input_data_key
        )
        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:]
            state_shape = [
                self.model.config.state_ch,
                self.model.tokenizer.get_latent_num_frames(_T),
                _H // self.model.tokenizer.spatial_compression_factor,
                _W // self.model.tokenizer.spatial_compression_factor,
            ]

        x0_fn = self.model.get_x0_fn_from_batch(
            data_batch, guidance, is_negative_prompt=is_negative_prompt
        )

        if x_sigma_max is None:
            x_sigma_max = (
                arch_invariant_rand(
                    (n_sample,) + tuple(state_shape),
                    torch.float32,
                    self.model.tensor_kwargs["device"],
                    seed,
                )
                * self.model.sde.sigma_max
            )

        if self.model.net.is_context_parallel_enabled:
            x_sigma_max = broadcast_split_tensor(
                x_sigma_max,
                seq_dim=2,
                process_group=self.model.get_context_parallel_group(),
            )

        if sigma_max is None:
            sigma_max = self.model.sde.sigma_max

        if self.model.config.rl.enabled:
            samples = self.sampler(
                x0_fn,
                x_sigma_max,
                num_steps=num_steps,
                sigma_max=sigma_max,
                sigma_min=self.model.sde.sigma_min,
                # TODO: this is also used in sampling,
                # which might not be preferred.
                S_churn=self.model.rl_solver_cfg.s_churn,
                S_min=self.model.rl_solver_cfg.s_t_min,
                S_max=self.model.rl_solver_cfg.s_t_max,
                S_noise=self.model.rl_solver_cfg.s_noise,
                solver_option=solver_option,
                callback_fns=callback_fns,
            )
        else:
            samples = self.sampler(
                x0_fn,
                x_sigma_max,
                num_steps=num_steps,
                sigma_max=sigma_max,
                sigma_min=self.model.sde.sigma_min,
                solver_option=solver_option,
                callback_fns=callback_fns,
            )
        if self.model.net.is_context_parallel_enabled:
            samples = cat_outputs_cp(
                samples, seq_dim=2, cp_group=self.model.get_context_parallel_group()
            )

        return samples

    def init_engine(self, model: torch.nn.Module):
        # WFM rollout directly use Pytorch model as the naive forward.
        # it doesn't need inference framework like vllm or trtllm.
        if not self._engine_initialized:
            self.model = model
            self.sampler = Sampler()
            self._engine_initialized = True
            logger.info("[Rollout] WFM engine initialized.")

    def get_underlying_model(self):
        if not self._engine_initialized:
            return None
        else:
            return self.model
