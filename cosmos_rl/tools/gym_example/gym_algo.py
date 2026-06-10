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

"""Toy policy-gradient helpers for the Gymnasium Classic Control demo.

These are deliberately the simplest objects that turn a single
trajectory's ``(observations, actions, rewards)`` into a scalar loss
the trainer can backprop through. They exist to validate the
trajectory-iteration contract end-to-end on CPU, **not** to be a
competitive RL implementation.

What this module is **not**:

* Not PPO, A2C, or REINFORCE-with-baseline. There is no value-function
  bootstrap, no GAE, no per-step advantage, no policy-ratio clipping.
* Not numerically robust. The MSE-weighted-by-returns formulation is a
  toy: it minimizes the squared difference between the network's raw
  output and the (possibly one-hot) sampled action, scaled by the
  return at each step. Useful as a teaching object that **does**
  apply gradients in the direction of higher-return trajectories;
  not useful as a benchmark.

Promoting these helpers to a top-level ``cosmos_rl/algorithms/``
package is deliberately deferred until a second example would consume
them, per the trajectory-iteration feature doc.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch


def compute_returns(rewards: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
    """Compute the discounted Monte-Carlo return at each step of a trajectory.

    For a 1-D reward vector ``r[0], r[1], ..., r[T-1]``, returns the
    1-D vector ``R[t] = sum_{k=0..T-1-t} gamma**k * r[t+k]``. The last
    element is just ``r[T-1]``.

    The implementation walks the rewards in reverse so it's
    ``O(T)`` and avoids materializing a power-of-gamma matrix.

    Args:
        rewards: 1-D float tensor of per-step rewards. Must have
            ``rewards.dim() == 1``. Length 0 returns an empty tensor.
        gamma: Discount factor in ``[0, 1]``. ``gamma = 1.0`` gives
            undiscounted returns (sum-of-future-rewards).

    Returns:
        1-D float tensor of the same length as ``rewards``, containing
        the discounted return at each step.

    Raises:
        ValueError: if ``rewards`` is not 1-D or ``gamma`` is outside
            ``[0, 1]``.
    """
    if rewards.dim() != 1:
        raise ValueError(
            f"compute_returns expects a 1-D rewards tensor, got shape {tuple(rewards.shape)}"
        )
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must lie in [0, 1], got {gamma}")
    T = rewards.shape[0]
    if T == 0:
        return rewards.clone()
    returns = torch.empty_like(rewards)
    running = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for t in range(T - 1, -1, -1):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def compute_simple_pg_loss(
    predicted_actions: torch.Tensor,
    target_actions: torch.Tensor,
    returns: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Toy policy-gradient loss: MSE between predicted and target actions,
    weighted by the per-step return.

    For a trajectory of length ``T``:

    * ``predicted_actions`` is the policy network's raw output for each
      observation: shape ``[T, action_dim]`` for continuous, ``[T, num_actions]``
      for discrete (logits, one-hot-targeted below).
    * ``target_actions`` is the action the rollout actually took:
      shape ``[T]`` (int64 indices) for discrete, ``[T, action_dim]`` for
      continuous.
    * ``returns`` is the discounted return at each step: shape ``[T]``.

    Discrete targets are one-hot-encoded internally so the same MSE
    formula applies for both action types. The loss is a length-mean of
    the per-step MSE scaled by the return, so longer episodes do not
    swamp shorter ones at the trajectory level. (Cross-trajectory
    weighting is the trainer's concern, per the trajectory-iteration
    feature doc.)

    The returned metrics dict carries floats only (no graph nodes), so
    the trainer can stash them across the step without holding the
    autograd graph.

    Args:
        predicted_actions: ``[T, K]`` float tensor of policy outputs.
        target_actions: ``[T]`` int64 tensor (discrete) or ``[T, K]``
            float tensor (continuous).
        returns: ``[T]`` float tensor of per-step returns.

    Returns:
        ``(loss, metrics)`` where ``loss`` is a 0-D tensor with grad
        attached and ``metrics`` is a dict with ``loss`` (float),
        ``mean_return`` (float), ``num_steps`` (int).

    Raises:
        ValueError: on shape mismatch.
    """
    if predicted_actions.dim() != 2:
        raise ValueError(
            f"predicted_actions must be 2-D [T, K], got shape {tuple(predicted_actions.shape)}"
        )
    if returns.dim() != 1 or returns.shape[0] != predicted_actions.shape[0]:
        raise ValueError(
            f"returns shape {tuple(returns.shape)} does not match "
            f"predicted_actions[0]={predicted_actions.shape[0]}"
        )

    T, K = predicted_actions.shape

    if target_actions.dim() == 1:
        # Discrete: one-hot the target action indices to align with logits.
        if target_actions.shape[0] != T:
            raise ValueError(
                f"target_actions shape {tuple(target_actions.shape)} does not match T={T}"
            )
        targets_onehot = torch.zeros_like(predicted_actions)
        targets_onehot.scatter_(1, target_actions.long().unsqueeze(1), 1.0)
        targets = targets_onehot
    elif target_actions.dim() == 2:
        if target_actions.shape != predicted_actions.shape:
            raise ValueError(
                f"continuous target_actions shape {tuple(target_actions.shape)} "
                f"does not match predicted_actions shape {tuple(predicted_actions.shape)}"
            )
        targets = target_actions.to(predicted_actions.dtype)
    else:
        raise ValueError(
            f"target_actions must be 1-D (discrete) or 2-D (continuous), "
            f"got shape {tuple(target_actions.shape)}"
        )

    per_step_mse = ((predicted_actions - targets) ** 2).mean(dim=1)  # [T]
    weighted = per_step_mse * returns
    loss = weighted.mean()

    metrics: Dict[str, Any] = {
        "loss": float(loss.detach().item()),
        "mean_return": float(returns.detach().mean().item()),
        "num_steps": int(T),
    }
    return loss, metrics


__all__ = ["compute_returns", "compute_simple_pg_loss"]
