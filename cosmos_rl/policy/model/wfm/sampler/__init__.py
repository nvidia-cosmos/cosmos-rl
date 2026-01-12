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
Standalone implementation of sampling algorithms from res_sampler.py
This module contains the core classes for wfm model sampling.
"""

import math
import attrs
import torch
from torch import Tensor
import numpy as np
from statistics import NormalDist
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

COMMON_SOLVER_OPTIONS = Literal["2ab", "2mid", "1euler"]


# ============================================================================
# Utility Functions
# ============================================================================


def _is_attrs_instance(obj: object) -> bool:
    """
    Helper function to check if an object is an instance of an attrs-defined class.

    Args:
        obj: The object to check.

    Returns:
        bool: True if the object is an instance of an attrs-defined class, False otherwise.
    """
    return hasattr(obj, "__attrs_attrs__")


def make_freezable(cls):
    """
    A decorator that adds the capability to freeze instances of an attrs-defined class.

    NOTE: This requires the wrapped attrs to be defined with attrs.define(slots=False) because we need
    to hack on a "_is_frozen" attribute.

    This decorator enhances an attrs-defined class with the ability to be "frozen" at runtime.
    Once an instance is frozen, its attributes cannot be changed. It also recursively freezes
    any attrs-defined objects that are attributes of the class.

    Usage:
        @make_freezable
        @attrs.define(slots=False)
        class MyClass:
            attribute1: int
            attribute2: str

        obj = MyClass(1, 'a')
        obj.freeze()  # Freeze the instance
        obj.attribute1 = 2  # Raises AttributeError

    Args:
        cls: The class to be decorated.

    Returns:
        The decorated class with added freezing capability.
    """

    if not hasattr(cls, "__dict__"):
        raise TypeError(
            "make_freezable cannot be used with classes that do not define __dict__. Make sure that the wrapped "
            "class was defined with `@attrs.define(slots=False)`"
        )

    original_setattr = cls.__setattr__

    def setattr_override(self, key, value) -> None:  # noqa: ANN001
        """
        Override __setattr__ to allow modifications during initialization
        and prevent modifications once the instance is frozen.
        """
        if hasattr(self, "_is_frozen") and self._is_frozen and key != "_is_frozen":
            raise AttributeError("Cannot modify frozen instance")
        original_setattr(self, key, value)  # type: ignore

    cls.__setattr__ = setattr_override  # type: ignore

    def freeze(self: object) -> None:
        """
        Freeze the instance and all its attrs-defined attributes.
        """
        for _, value in attrs.asdict(self, recurse=False).items():
            if _is_attrs_instance(value) and hasattr(value, "freeze"):
                value.freeze()
        self._is_frozen = True  # type: ignore

    cls.freeze = freeze  # type: ignore

    return cls


# ============================================================================
# Multistep Functions
# ============================================================================


def common_broadcast(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    ndims1 = x.ndim
    ndims2 = y.ndim

    common_ndims = min(ndims1, ndims2)
    for axis in range(common_ndims):
        assert x.shape[axis] == y.shape[axis], "Dimensions not equal at axis {}".format(
            axis
        )

    if ndims1 < ndims2:
        x = x.reshape(x.shape + (1,) * (ndims2 - ndims1))
    elif ndims2 < ndims1:
        y = y.reshape(y.shape + (1,) * (ndims1 - ndims2))

    return x, y


def batch_mul(x: Tensor, y: Tensor) -> Tensor:
    x, y = common_broadcast(x, y)
    return x * y


def phi1(t: torch.Tensor) -> torch.Tensor:
    """
    Compute the first order phi function: (exp(t) - 1) / t.

    Args:
        t: Input tensor.

    Returns:
        Tensor: Result of phi1 function.
    """
    input_dtype = t.dtype
    t = t.to(dtype=torch.float64)
    return (torch.expm1(t) / t).to(dtype=input_dtype)


def phi2(t: torch.Tensor) -> torch.Tensor:
    """
    Compute the second order phi function: (phi1(t) - 1) / t.

    Args:
        t: Input tensor.

    Returns:
        Tensor: Result of phi2 function.
    """
    input_dtype = t.dtype
    t = t.to(dtype=torch.float64)
    return ((phi1(t) - 1.0) / t).to(dtype=input_dtype)


def reg_x0_euler_step(
    x_s: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    x0_s: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a regularized Euler step based on x0 prediction.

    Args:
        x_s: Current state tensor.
        s: Current time tensor.
        t: Target time tensor.
        x0_s: Prediction at current time.

    Returns:
        Tuple[Tensor, Tensor]: Updated state tensor and current prediction.
    """
    coef_x0 = (s - t) / s
    coef_xs = t / s
    return batch_mul(coef_x0, x0_s) + batch_mul(coef_xs, x_s), x0_s


def res_x0_rk2_step(
    x_s: torch.Tensor,
    t: torch.Tensor,
    s: torch.Tensor,
    x0_s: torch.Tensor,
    s1: torch.Tensor,
    x0_s1: torch.Tensor,
) -> torch.Tensor:
    """
    Perform a residual-based 2nd order Runge-Kutta step.

    Args:
        x_s: Current state tensor.
        t: Target time tensor.
        s: Current time tensor.
        x0_s: Prediction at current time.
        s1: Intermediate time tensor.
        x0_s1: Prediction at intermediate time.

    Returns:
        Tensor: Updated state tensor.

    Raises:
        AssertionError: If step size is too small.
    """
    s = -torch.log(s)
    t = -torch.log(t)
    m = -torch.log(s1)

    dt = t - s
    assert not torch.any(
        torch.isclose(dt, torch.zeros_like(dt), atol=1e-6)
    ), "Step size is too small"
    assert not torch.any(
        torch.isclose(m - s, torch.zeros_like(dt), atol=1e-6)
    ), "Step size is too small"

    c2 = (m - s) / dt
    phi1_val, phi2_val = phi1(-dt), phi2(-dt)

    # Handle edge case where t = s = m
    b1 = torch.nan_to_num(phi1_val - 1.0 / c2 * phi2_val, nan=0.0)
    b2 = torch.nan_to_num(1.0 / c2 * phi2_val, nan=0.0)

    return batch_mul(torch.exp(-dt), x_s) + batch_mul(
        dt, batch_mul(b1, x0_s) + batch_mul(b2, x0_s1)
    )


def order2_fn(
    x_s: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    x0_s: torch.Tensor,
    x0_preds: torch.Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    impl the second order multistep method in https://arxiv.org/pdf/2308.02157
    Adams Bashforth approach!
    """
    if x0_preds:
        x0_s1, s1 = x0_preds[0]
        x_t = res_x0_rk2_step(x_s, t, s, x0_s, s1, x0_s1)
    else:
        x_t = reg_x0_euler_step(x_s, s, t, x0_s)[0]
    return x_t, [(x0_s, s)]


MULTISTEP_FNs = {
    "2ab": order2_fn,
}


def is_multi_step_fn_supported(name: str) -> bool:
    """
    Check if the multistep method is supported.
    """
    return name in MULTISTEP_FNs


# ============================================================================
# Runge-Kutta Functions
# ============================================================================
def reg_eps_euler_step(
    x_s: torch.Tensor, s: torch.Tensor, t: torch.Tensor, eps_s: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a regularized Euler step based on epsilon prediction.

    Args:
        x_s: Current state tensor.
        s: Current time tensor.
        t: Target time tensor.
        eps_s: Epsilon prediction at current time.

    Returns:
        Tuple[Tensor, Tensor]: Updated state tensor and current x0 prediction.
    """
    return x_s + batch_mul(eps_s, t - s), x_s + batch_mul(eps_s, 0 - s)


def rk1_euler(
    x_s: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x0_fn: Callable
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a first-order Runge-Kutta (Euler) step.

    Recommended for wfm models with guidance or model undertrained
    Usually more stable at the cost of a bit slower convergence.

    Args:
        x_s: Current state tensor.
        s: Current time tensor.
        t: Target time tensor.
        x0_fn: Function to compute x0 prediction.

    Returns:
        Tuple[Tensor, Tensor]: Updated state tensor and x0 prediction.
    """
    x0_s = x0_fn(x_s, s)
    return reg_x0_euler_step(x_s, s, t, x0_s)


def rk2_mid_stable(
    x_s: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x0_fn: Callable
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a stable second-order Runge-Kutta (midpoint) step.

    Args:
        x_s: Current state tensor.
        s: Current time tensor.
        t: Target time tensor.
        x0_fn: Function to compute x0 prediction.

    Returns:
        Tuple[Tensor, Tensor]: Updated state tensor and x0 prediction.
    """
    s1 = torch.sqrt(s * t)
    x_s1, _ = rk1_euler(x_s, s, s1, x0_fn)

    x0_s1 = x0_fn(x_s1, s1)
    return reg_x0_euler_step(x_s, s, t, x0_s1)


def rk2_mid(
    x_s: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x0_fn: Callable
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a second-order Runge-Kutta (midpoint) step.

    Args:
        x_s: Current state tensor.
        s: Current time tensor.
        t: Target time tensor.
        x0_fn: Function to compute x0 prediction.

    Returns:
        Tuple[Tensor, Tensor]: Updated state tensor and x0 prediction.
    """
    s1 = torch.sqrt(s * t)
    x_s1, x0_s = rk1_euler(x_s, s, s1, x0_fn)

    x0_s1 = x0_fn(x_s1, s1)

    return res_x0_rk2_step(x_s, t, s, x0_s, s1, x0_s1), x0_s1


def rk_2heun_naive(
    x_s: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x0_fn: Callable
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a naive second-order Runge-Kutta (Heun's method) step.
    Impl based on rho-rk-deis solvers, https://github.com/qsh-zh/deis
    Recommended for wfm models without guidance and relative large NFE

    Args:
        x_s: Current state tensor.
        s: Current time tensor.
        t: Target time tensor.
        x0_fn: Function to compute x0 prediction.

    Returns:
        Tuple[Tensor, Tensor]: Updated state tensor and current state.
    """
    x_t, x0_s = rk1_euler(x_s, s, t, x0_fn)
    eps_s = batch_mul(1.0 / s, x_t - x0_s)
    x0_t = x0_fn(x_t, t)
    eps_t = batch_mul(1.0 / t, x_t - x0_t)

    avg_eps = (eps_s + eps_t) / 2

    return reg_eps_euler_step(x_s, s, t, avg_eps)


def rk_2heun_edm(
    x_s: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x0_fn: Callable
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a naive second-order Runge-Kutta (Heun's method) step.
    Impl based no EDM second order Heun method

    Args:
        x_s: Current state tensor.
        s: Current time tensor.
        t: Target time tensor.
        x0_fn: Function to compute x0 prediction.

    Returns:
        Tuple[Tensor, Tensor]: Updated state tensor and current state.
    """
    x_t, x0_s = rk1_euler(x_s, s, t, x0_fn)
    x0_t = x0_fn(x_t, t)

    avg_x0 = (x0_s + x0_t) / 2

    return reg_x0_euler_step(x_s, s, t, avg_x0)


def rk_3kutta_naive(
    x_s: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x0_fn: Callable
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a naive third-order Runge-Kutta step.
    Impl based on rho-rk-deis solvers, https://github.com/qsh-zh/deis
    Recommended for wfm models without guidance and relative large NFE

    Args:
        x_s: Current state tensor.
        s: Current time tensor.
        t: Target time tensor.
        x0_fn: Function to compute x0 prediction.

    Returns:
        Tuple[Tensor, Tensor]: Updated state tensor and current state.
    """
    c2, c3 = 0.5, 1.0
    a31, a32 = -1.0, 2.0
    b1, b2, b3 = 1.0 / 6, 4.0 / 6, 1.0 / 6

    delta = t - s

    s1 = c2 * delta + s
    s2 = c3 * delta + s
    x_s1, x0_s = rk1_euler(x_s, s, s1, x0_fn)
    eps_s = batch_mul(1.0 / s, x_s - x0_s)
    x0_s1 = x0_fn(x_s1, s1)
    eps_s1 = batch_mul(1.0 / s1, x_s1 - x0_s1)

    _eps = a31 * eps_s + a32 * eps_s1
    x_s2, _ = reg_eps_euler_step(x_s, s, s2, _eps)

    x0_s2 = x0_fn(x_s2, s2)
    eps_s2 = batch_mul(1.0 / s2, x_s2 - x0_s2)

    avg_eps = b1 * eps_s + b2 * eps_s1 + b3 * eps_s2
    return reg_eps_euler_step(x_s, s, t, avg_eps)


RK_FNs = {
    "1euler": rk1_euler,
    "2mid": rk2_mid,
    "2mid_stable": rk2_mid_stable,
    "2heun_edm": rk_2heun_edm,
    "2heun_naive": rk_2heun_naive,
    "3kutta_naive": rk_3kutta_naive,
}


def is_runge_kutta_fn_supported(name: str) -> bool:
    """
    Check if the specified Runge-Kutta function is supported.

    Args:
        name: Name of the Runge-Kutta method.

    Returns:
        bool: True if the method is supported, False otherwise.
    """
    return name in RK_FNs


# ============================================================================
# Configuration Classes
# ============================================================================


@make_freezable
@attrs.define(slots=False)
class SolverConfig:
    is_multi: bool = False
    rk: str = "2mid"
    multistep: str = "2ab"
    # following parameters control stochasticity, see EDM paper
    # BY default, we use deterministic with no stochasticity
    s_churn: float = 0.0
    s_t_max: float = float("inf")
    s_t_min: float = 0.05
    s_noise: float = 1.0


@make_freezable
@attrs.define(slots=False)
class SolverTimestampConfig:
    nfe: int = 50
    t_min: float = 0.002
    t_max: float = 80.0
    order: float = 7.0
    is_forward: bool = False  # whether generate forward or backward timestamps


@make_freezable
@attrs.define(slots=False)
class SamplerConfig:
    solver: SolverConfig = attrs.field(factory=SolverConfig)
    timestamps: SolverTimestampConfig = attrs.field(factory=SolverTimestampConfig)
    sample_clean: bool = True  # whether run one last step to generate clean image


def get_rev_ts(
    t_min: float,
    t_max: float,
    num_steps: int,
    ts_order: Union[int, float],
    is_forward: bool = False,
) -> torch.Tensor:
    """
    Generate a sequence of reverse time steps.

    Args:
        t_min (float): The minimum time value.
        t_max (float): The maximum time value.
        num_steps (int): The number of time steps to generate.
        ts_order (Union[int, float]): The order of the time step progression.
        is_forward (bool, optional): If True, returns the sequence in forward order. Defaults to False.

    Returns:
        torch.Tensor: A tensor containing the generated time steps in reverse or forward order.

    Raises:
        ValueError: If `t_min` is not less than `t_max`.
        TypeError: If `ts_order` is not an integer or float.
    """
    if t_min >= t_max:
        raise ValueError("t_min must be less than t_max")

    if not isinstance(ts_order, (int, float)):
        raise TypeError("ts_order must be an integer or float")

    step_indices = torch.arange(num_steps + 1, dtype=torch.float64)
    time_steps = (
        t_max ** (1 / ts_order)
        + step_indices / num_steps * (t_min ** (1 / ts_order) - t_max ** (1 / ts_order))
    ) ** ts_order

    if is_forward:
        return time_steps.flip(dims=(0,))

    return time_steps


def fori_loop(
    lower: int, upper: int, body_fun: Callable[[int, Any], Any], init_val: Any
) -> Any:
    """
    Implements a for loop with a function.

    Args:
        lower: Lower bound of the loop (inclusive).
        upper: Upper bound of the loop (exclusive).
        body_fun: Function to be applied in each iteration.
        init_val: Initial value for the loop.

    Returns:
        The final result after all iterations.
    """
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val


def differential_equation_solver(
    x0_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    sigmas_L: torch.Tensor,
    solver_cfg: SolverConfig,
    callback_fns: Optional[List[Callable]] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Creates a differential equation solver function.

    Args:
        x0_fn: Function to compute x0 prediction.
        sigmas_L: Tensor of sigma values with shape [L,].
        solver_cfg: Configuration for the solver.
        callback_fns: Optional list of callback functions.

    Returns:
        A function that solves the differential equation.
    """
    num_step = len(sigmas_L) - 1

    if solver_cfg.is_multi:
        update_step_fn = get_multi_step_fn(solver_cfg.multistep)
    else:
        update_step_fn = get_runge_kutta_fn(solver_cfg.rk)

    eta = min(solver_cfg.s_churn / (num_step + 1), math.sqrt(1.2) - 1)

    def sample_fn(input_xT_B_StateShape: torch.Tensor) -> torch.Tensor:
        """
        Samples from the differential equation.

        Args:
            input_xT_B_StateShape: Input tensor with shape [B, StateShape].

        Returns:
            Output tensor with shape [B, StateShape].
        """
        ones_B = torch.ones(
            input_xT_B_StateShape.size(0),
            device=input_xT_B_StateShape.device,
            dtype=torch.float64,
        )

        def step_fn(
            i_th: int, state: Tuple[torch.Tensor, Optional[List[torch.Tensor]]]
        ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
            input_x_B_StateShape, x0_preds = state
            sigma_cur_0, sigma_next_0 = sigmas_L[i_th], sigmas_L[i_th + 1]

            # algorithm 2: line 4-6
            if solver_cfg.s_t_min < sigma_cur_0 < solver_cfg.s_t_max:
                hat_sigma_cur_0 = sigma_cur_0 + eta * sigma_cur_0
                input_x_B_StateShape = input_x_B_StateShape + (
                    hat_sigma_cur_0**2 - sigma_cur_0**2
                ).sqrt() * solver_cfg.s_noise * torch.randn_like(input_x_B_StateShape)
                sigma_cur_0 = hat_sigma_cur_0

            if solver_cfg.is_multi:
                x0_pred_B_StateShape = x0_fn(input_x_B_StateShape, sigma_cur_0 * ones_B)
                output_x_B_StateShape, x0_preds = update_step_fn(
                    input_x_B_StateShape,
                    sigma_cur_0 * ones_B,
                    sigma_next_0 * ones_B,
                    x0_pred_B_StateShape,
                    x0_preds,
                )
            else:
                output_x_B_StateShape, x0_preds = update_step_fn(
                    input_x_B_StateShape,
                    sigma_cur_0 * ones_B,
                    sigma_next_0 * ones_B,
                    x0_fn,
                )

            if callback_fns:
                for callback_fn in callback_fns:
                    callback_fn(**locals())

            return output_x_B_StateShape, x0_preds

        x_at_eps, _ = fori_loop(0, num_step, step_fn, [input_xT_B_StateShape, None])
        return x_at_eps

    return sample_fn


# ============================================================================
# Main Classes and Functions
# ============================================================================


def get_multi_step_fn(name: str) -> Callable:
    if name in MULTISTEP_FNs:
        return MULTISTEP_FNs[name]
    methods = "\n\t".join(MULTISTEP_FNs.keys())
    raise RuntimeError("Only support multistep method\n" + methods)


def get_runge_kutta_fn(name: str) -> Callable:
    """
    Get the specified Runge-Kutta function.

    Args:
        name: Name of the Runge-Kutta method.

    Returns:
        Callable: The specified Runge-Kutta function.

    Raises:
        RuntimeError: If the specified method is not supported.
    """
    if name in RK_FNs:
        return RK_FNs[name]
    methods = "\n\t".join(RK_FNs.keys())
    raise RuntimeError(f"Only support the following Runge-Kutta methods:\n\t{methods}")


class Sampler(torch.nn.Module):
    def __init__(self, cfg: Optional[SamplerConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = SamplerConfig()
        self.cfg = cfg

    @torch.no_grad()
    def forward(
        self,
        x0_fn: Callable,
        x_sigma_max: torch.Tensor,
        num_steps: int = 35,
        sigma_min: float = 0.002,
        sigma_max: float = 80,
        rho: float = 7,
        S_churn: float = 0,
        S_min: float = 0,
        S_max: float = float("inf"),
        S_noise: float = 1,
        solver_option: str = "2ab",
        callback_fns: Optional[List[Callable]] = None,
    ) -> torch.Tensor:
        in_dtype = x_sigma_max.dtype

        def float64_x0_fn(
            x_B_StateShape: torch.Tensor, t_B: torch.Tensor
        ) -> torch.Tensor:
            return x0_fn(x_B_StateShape.to(in_dtype), t_B.to(in_dtype)).to(
                torch.float64
            )

        is_multistep = is_multi_step_fn_supported(solver_option)
        is_rk = is_runge_kutta_fn_supported(solver_option)
        assert (
            is_multistep or is_rk
        ), f"Only support multistep or Runge-Kutta method, got {solver_option}"

        solver_cfg = SolverConfig(
            s_churn=S_churn,
            s_t_max=S_max,
            s_t_min=S_min,
            s_noise=S_noise,
            is_multi=is_multistep,
            rk=solver_option,
            multistep=solver_option,
        )
        timestamps_cfg = SolverTimestampConfig(
            nfe=num_steps, t_min=sigma_min, t_max=sigma_max, order=rho
        )
        sampler_cfg = SamplerConfig(
            solver=solver_cfg, timestamps=timestamps_cfg, sample_clean=True
        )

        return self._forward_impl(
            float64_x0_fn, x_sigma_max, sampler_cfg, callback_fns
        ).to(in_dtype)

    @torch.no_grad()
    def _forward_impl(
        self,
        denoiser_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        noisy_input_B_StateShape: torch.Tensor,
        sampler_cfg: Optional[SamplerConfig] = None,
        callback_fns: Optional[List[Callable]] = None,
    ) -> torch.Tensor:
        """
        Internal implementation of the forward pass.

        Args:
            denoiser_fn: Function to denoise the input.
            noisy_input_B_StateShape: Input tensor with noise.
            sampler_cfg: Configuration for the sampler.
            callback_fns: List of callback functions to be called during sampling.

        Returns:
            torch.Tensor: Denoised output tensor.
        """
        sampler_cfg = self.cfg if sampler_cfg is None else sampler_cfg
        solver_order = (
            1 if sampler_cfg.solver.is_multi else int(sampler_cfg.solver.rk[0])
        )
        num_timestamps = sampler_cfg.timestamps.nfe // solver_order

        sigmas_L = get_rev_ts(
            sampler_cfg.timestamps.t_min,
            sampler_cfg.timestamps.t_max,
            num_timestamps,
            sampler_cfg.timestamps.order,
        ).to(noisy_input_B_StateShape.device)

        denoised_output = differential_equation_solver(
            denoiser_fn, sigmas_L, sampler_cfg.solver, callback_fns=callback_fns
        )(noisy_input_B_StateShape)

        if sampler_cfg.sample_clean:
            # Override denoised_output with fully denoised version
            ones = torch.ones(
                denoised_output.size(0),
                device=denoised_output.device,
                dtype=denoised_output.dtype,
            )
            denoised_output = denoiser_fn(denoised_output, sigmas_L[-1] * ones)

        return denoised_output


class EDMSDE:
    def __init__(
        self,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        sigma_max: float = 80.0,
        sigma_min: float = 0.002,
    ):
        self.gaussian_dist = NormalDist(mu=p_mean, sigma=p_std)
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min

    def sample_t(self, batch_size: int) -> torch.Tensor:
        cdf_vals = np.random.uniform(size=(batch_size))
        samples_interval_gaussian = [
            self.gaussian_dist.inv_cdf(cdf_val) for cdf_val in cdf_vals
        ]

        log_sigma = torch.tensor(samples_interval_gaussian, device="cuda")
        return torch.exp(log_sigma)

    def marginal_prob(
        self, x0: torch.Tensor, sigma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """This is trivial in the base class, but may be used by derived classes in a more interesting way"""
        return x0, sigma


class EDMScaling:
    def __init__(self, sigma_data: float = 0.5):
        self.sigma_data = sigma_data

    def __call__(
        self, sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        c_noise = 0.25 * sigma.log()
        return c_skip, c_out, c_in, c_noise

    def sigma_loss_weights(self, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2


class RectifiedFlowScaling:
    def __init__(
        self,
        sigma_data: float = 1.0,
        t_scaling_factor: float = 1.0,
        loss_weight_uniform: bool = True,
    ):
        assert (
            abs(sigma_data - 1.0) < 1e-6
        ), "sigma_data must be 1.0 for RectifiedFlowScaling"
        self.t_scaling_factor = t_scaling_factor
        self.loss_weight_uniform = loss_weight_uniform
        if loss_weight_uniform is False:
            # using huan lin suggested one here. which put more weight on the middle of the timesteps.
            self.num_steps = 1000
            t = torch.linspace(0, 1, self.num_steps)
            y = torch.exp(-2 * (t - 0.5) ** 2)
            shift = y - y.min()
            weights = shift * (
                self.num_steps / shift.sum()
            )  # make sure the avg weights is 1.0
            self.weights = weights

    def __call__(
        self, sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t = sigma / (sigma + 1)
        c_skip = 1.0 - t
        c_out = -t
        c_in = 1.0 - t
        c_noise = t * self.t_scaling_factor
        return c_skip, c_out, c_in, c_noise

    def sigma_loss_weights(self, sigma: torch.Tensor) -> torch.Tensor:
        if self.loss_weight_uniform:
            return (1.0 + sigma) ** 2 / sigma**2
        else:
            t = sigma / (sigma + 1)
            index = (t * self.num_steps).round().long()
            # Clamp index to valid range [0, num_steps-1] to avoid out of bounds
            index = torch.clamp(index, 0, self.num_steps - 1)
            weights_on_device = self.weights.to(sigma.device)
            return weights_on_device[index].type_as(sigma)
