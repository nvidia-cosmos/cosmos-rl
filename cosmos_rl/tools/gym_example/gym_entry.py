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

"""Launch entry script for the Gymnasium Classic Control demo.

Importing this module triggers all the registrations cosmos-rl needs
to resolve a ``[train.train_policy].trainer_type = "gym_pg"`` /
``[rollout].backend = "gym"`` config:

* ``GymTrainer`` registers under ``"gym_pg"`` via
  :func:`TrainerRegistry.register`.
* ``GymRolloutBackend`` registers under ``"gym"`` via
  :func:`RolloutRegistry.register`.
* :func:`register_gym_policy` wires ``.toml`` config paths to a
  :class:`NoOpTokenizer` and a :class:`GymMLPConfig` factory.

The entry script also defines a trivial seed dataset and reward
function so the launcher's ``launch_worker(dataset=..., reward_fns=...)``
contract is satisfied. The dataset is **not** training data — for an
RL workload the "dataset" is a stream of initial conditions the
dispatcher hands to the rollout backend, which in turn drives the
gymnasium env via :class:`GymRolloutEngine`.

Launch (when the launcher path is fully wired — see the package
README for the current scope of launcher-runnable vs.
contract-validation-runnable):

    cosmos-rl --config cosmos_rl/tools/gym_example/configs/cartpole_colocated.toml \\
              cosmos_rl/tools/gym_example/gym_entry.py
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset

from cosmos_rl.launcher.worker_entry import main as launch_worker
from cosmos_rl.tools.gym_example.gym_data_packer import GymDataPacker
from cosmos_rl.tools.gym_example.gym_policy import register_gym_policy

# Importing these triggers their @register decorators, wiring the
# trainer_type / rollout backend strings to their classes.
from cosmos_rl.tools.gym_example.gym_rollout_backend import (  # noqa: F401
    GymRolloutBackend,
)
from cosmos_rl.tools.gym_example.gym_trainer import GymTrainer  # noqa: F401

# Wire .toml-based policy resolution into cosmos-rl's tokenizer /
# model-config registries. Idempotent under repeated import.
register_gym_policy()


_DEFAULT_DATASET_SIZE = 1024


class GymSeedDataset(Dataset):
    """Trivial dataset of seeds for the gym demo.

    Each item is a JSON-encoded ``{"seed": i}`` so the dispatcher can
    feed a deterministic initial-condition stream to the rollout
    backend. ``GymDataPacker.get_rollout_input`` JSON-decodes the
    prompt back into a dict of init kwargs for the gym env.

    Args:
        size: Number of seeds in the dataset.
        seed_offset: Added to each index so disjoint configs (train
            vs. validation) can avoid seed collisions.
    """

    def __init__(self, size: int = _DEFAULT_DATASET_SIZE, seed_offset: int = 0):
        self._size = int(size)
        self._seed_offset = int(seed_offset)

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"prompt": json.dumps({"seed": idx + self._seed_offset})}


def gym_episode_reward(
    completion: Any,
    *args: Any,
    **kwargs: Any,
) -> float:
    """Return the (undiscounted) total episode reward.

    Cosmos-rl's reward functions are called per completion. For the
    gym demo, ``completion`` is the trajectory dict produced by
    :class:`GymRolloutEngine` (see :class:`GymDataPacker`); we sum the
    valid prefix of ``rewards``. Anything else (a string, a missing
    dict, an error) returns ``0.0`` so the dispatcher can keep going
    without crashing on a malformed rollout.
    """
    if not isinstance(completion, dict):
        return 0.0
    rewards = completion.get("rewards")
    if rewards is None:
        return 0.0
    ep_len = completion.get("episode_length")
    if ep_len is not None:
        if hasattr(ep_len, "item"):
            try:
                ep_len = int(ep_len.item())
            except Exception:
                ep_len = None
        elif hasattr(ep_len, "__len__") and len(ep_len) == 1:
            ep_len = int(ep_len[0])
        else:
            try:
                ep_len = int(ep_len)
            except Exception:
                ep_len = None
    sliced = rewards[:ep_len] if ep_len is not None else rewards
    if hasattr(sliced, "sum"):
        try:
            return float(np.asarray(sliced).sum())
        except Exception:
            pass
    try:
        return float(sum(float(r) for r in sliced))
    except Exception:
        return 0.0


def main(
    *,
    dataset_size: int = _DEFAULT_DATASET_SIZE,
    val_dataset_size: int = 32,
    reward_fns: Optional[List[Any]] = None,
) -> None:
    """Hand the gym pieces to cosmos-rl's launcher entry."""
    rfns = reward_fns if reward_fns is not None else [gym_episode_reward]
    launch_worker(
        dataset=GymSeedDataset(size=dataset_size),
        val_dataset=GymSeedDataset(size=val_dataset_size, seed_offset=10_000),
        data_packer=GymDataPacker(),
        val_data_packer=GymDataPacker(),
        reward_fns=rfns,
        val_reward_fns=rfns,
    )


__all__ = [
    "GymSeedDataset",
    "gym_episode_reward",
    "main",
]


if __name__ == "__main__":
    main()
