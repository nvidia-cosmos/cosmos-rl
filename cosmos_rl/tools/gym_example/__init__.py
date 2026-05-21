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

"""Gymnasium Classic Control example for cosmos-rl.

This package demonstrates how to drive a non-LLM RL workload through
the cosmos-rl pipeline using the extension hooks added by the Gym API
MR (``register_tokenizer_loader``, ``register_local_model_config``,
``TensorDataPacker``, ``IdentityWeightMapper``).

Reference environments (Gymnasium Classic Control suite):

* **CartPole-v1** (primary) — discrete 2-action control, 4-dim
  observation; the canonical "hello world" of RL.
* **Pendulum-v1** (noted) — continuous 1-dim action over a 3-dim
  observation; useful for validating continuous-action wiring.

See ``README.md`` in this directory for a launch walkthrough.
"""

from cosmos_rl.tools.gym_example.gym_algo import (
    compute_returns,
    compute_simple_pg_loss,
)
from cosmos_rl.tools.gym_example.gym_data_packer import GymDataPacker
from cosmos_rl.tools.gym_example.gym_policy import (
    GymMLPConfig,
    GymPolicy,
    register_gym_policy,
)
from cosmos_rl.tools.gym_example.gym_rollout import (
    GymRolloutEngine,
    rollout_episode,
)
from cosmos_rl.tools.gym_example.gym_rollout_backend import GymRolloutBackend
from cosmos_rl.tools.gym_example.gym_trainer import GymTrainer

__all__ = [
    "GymDataPacker",
    "GymMLPConfig",
    "GymPolicy",
    "GymRolloutBackend",
    "GymRolloutEngine",
    "GymTrainer",
    "compute_returns",
    "compute_simple_pg_loss",
    "register_gym_policy",
    "rollout_episode",
]
