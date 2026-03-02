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
Cosmos Policy sampler – port of cosmos-policy's ``CosmosPolicySampler``
using cosmos-rl sampler primitives.

Key differences from the base ``Sampler``:

* ``sample_clean=True`` always: ``num_steps - 1`` solver iterations followed
  by one clean denoising step.
* Special ``num_steps == 1`` path: a single direct denoising call at
  ``sigma_max``, skipping the ODE solver entirely.
"""

from __future__ import annotations

from typing import Callable

import torch

from cosmos_rl.policy.model.wfm.sampler import (
    SolverConfig,
    differential_equation_solver,
    get_rev_ts,
    is_multi_step_fn_supported,
    is_runge_kutta_fn_supported,
)


class CosmosPolicySampler:
    """Port of cosmos-policy's CosmosPolicySampler using cosmos-rl sampler primitives."""

    @torch.no_grad()
    def __call__(
        self,
        x0_fn: Callable,
        x_sigma_max: torch.Tensor,
        num_steps: int = 5,
        sigma_max: float = 80.0,
        sigma_min: float = 4.0,
        solver_option: str = "2ab",
        rho: float = 7.0,
    ) -> torch.Tensor:
        in_dtype = x_sigma_max.dtype

        def float64_x0_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return x0_fn(x.to(in_dtype), t.to(in_dtype)).to(torch.float64)

        is_multistep = is_multi_step_fn_supported(solver_option)
        is_rk = is_runge_kutta_fn_supported(solver_option)
        assert is_multistep or is_rk

        solver_cfg = SolverConfig(
            s_churn=0, s_t_max=float("inf"), s_t_min=0, s_noise=1,
            is_multi=is_multistep, rk=solver_option, multistep=solver_option,
        )

        effective_steps = max(num_steps - 1, 1) if num_steps > 1 else num_steps
        solver_order = 1 if solver_cfg.is_multi else int(solver_cfg.rk[0])
        num_timestamps = effective_steps // solver_order
        sigmas_L = get_rev_ts(sigma_min, sigma_max, num_timestamps, rho).to(x_sigma_max.device)

        if num_steps > 1:
            denoised = differential_equation_solver(float64_x0_fn, sigmas_L, solver_cfg)(
                x_sigma_max.to(torch.float64)
            )
            ones = torch.ones(denoised.size(0), device=denoised.device, dtype=denoised.dtype)
            denoised = float64_x0_fn(denoised, sigmas_L[-1] * ones)
        else:
            denoised = x_sigma_max.to(torch.float64)
            ones = torch.ones(denoised.size(0), device=denoised.device, dtype=denoised.dtype)
            denoised = float64_x0_fn(denoised, sigmas_L[0] * ones)

        return denoised.to(in_dtype)
