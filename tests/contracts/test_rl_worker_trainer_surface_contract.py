# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RL-path trainer surface contract.

The cosmos-rl ``RLPolicyWorker`` (the worker class behind both
``ColocatedRLControlWorker`` and the disaggregated RL path) reaches
into ``self.trainer`` and ``self.trainer.model`` for a number of
attributes and methods that aren't declared on the abstract
:class:`Trainer` base class.  This implicit interface lives in
``cosmos_rl/policy/worker/rl_worker.py`` (greppable by
``self.trainer.\\w+`` / ``self.trainer.model.\\w+``); when a new RL
trainer misses one of these, the launcher surfaces the failure deep
inside worker bring-up (e.g. ``AttributeError: 'GymTrainer' object
has no attribute 'load_model'``), which is hard to debug.

This module pins that contract for **CPU-runnable RL trainers**
(today: ``gym_pg`` only).  As more lightweight RL trainers ship,
add their factories to :data:`CPU_RUNNABLE_RL_TRAINER_FACTORIES` and
the contract auto-applies.

Trainers that take the SFT path (``sft``, ``pi_sft``, ``diffusers_sft``)
or the diffusion-NFT path (``diffusion_nft``, ``cosmos_policy``) have
a different worker (``SFTPolicyWorker``, etc.) with a different
surface; this contract intentionally does not cover them.
"""

from __future__ import annotations

import unittest
from typing import Any, Callable, Dict


# --- Contract definition (the universal helper) -------------------


def assert_rl_worker_trainer_surface(trainer: Any) -> None:
    """Assert ``trainer`` exposes the surface ``RLPolicyWorker`` reads.

    Mirrors the attribute and method reads in
    ``cosmos_rl/policy/worker/rl_worker.py``.  Reusable from any
    trainer-specific test file (e.g. ``test_gym_example.py``'s
    ``TestGymTrainerLauncherSurface`` calls this helper so the
    contract has exactly one definition site).

    Args:
        trainer: A constructed RL trainer instance.

    Raises:
        AssertionError: With a message naming the precise gap.
    """
    # ---- trainer-level instance attributes -----------------------
    for attr in ("model", "weight_mapper", "map_w_from_policy_to_rollout"):
        if not hasattr(trainer, attr):
            raise AssertionError(
                f"{type(trainer).__name__} is missing trainer.{attr!s}; "
                f"rl_worker.py reads it during weight-sync setup. "
                f"Set it in __init__ (see GymTrainer for an example)."
            )

    # ---- trainer-level methods -----------------------------------
    for method in (
        "step_training",
        "weight_resume",
        "update_lr_schedulers",
        "sync_all_states",
    ):
        if not callable(getattr(trainer, method, None)):
            raise AssertionError(
                f"{type(trainer).__name__} is missing trainer.{method}() "
                f"(rl_worker.py calls it).  Stub it as a no-op if the "
                f"trainer doesn't need the underlying behavior."
            )

    # ---- model-level instance attributes -------------------------
    model = trainer.model
    for attr in ("trainable_params", "weight_sync_transforms", "weight_mapper"):
        if not hasattr(model, attr):
            raise AssertionError(
                f"{type(trainer).__name__}.model "
                f"({type(model).__name__}) is missing model.{attr}; "
                f"rl_worker.py reads it via ``trainable_params = "
                f"self.trainer.model.trainable_params`` and the loop "
                f"over ``self.trainer.model.weight_sync_transforms``. "
                f"Attach via a helper like "
                f"``GymTrainer._attach_weight_sync_surface``."
            )

    # ---- shape sanity --------------------------------------------
    if not isinstance(trainer.map_w_from_policy_to_rollout, dict):
        raise AssertionError(
            f"{type(trainer).__name__}.map_w_from_policy_to_rollout "
            f"is {type(trainer.map_w_from_policy_to_rollout).__name__}; "
            "rl_worker.py iterates it as a dict (``.items()``)."
        )
    transforms = list(trainer.model.weight_sync_transforms)
    if transforms and not isinstance(transforms[0], tuple):
        raise AssertionError(
            f"model.weight_sync_transforms must yield (name, "
            f"tensor_or_callable) tuples; got "
            f"{type(transforms[0]).__name__} as the first element."
        )


# --- CPU-runnable RL trainer factories ----------------------------


def _build_gym_pg_trainer():
    """Build a ``GymTrainer`` on CPU.  Mirrors the factory in
    ``test_trainer_metrics_contract.py`` but returns just the
    trainer (no rollouts needed for surface checks)."""
    import torch

    from cosmos_rl.policy.config import Config as CosmosConfig
    from cosmos_rl.tools.gym_example.gym_data_packer import GymDataPacker
    from cosmos_rl.tools.gym_example.gym_policy import GymMLPConfig, GymPolicy
    from cosmos_rl.tools.gym_example.gym_trainer import GymTrainer
    from cosmos_rl.utils.parallelism import ParallelDims

    return GymTrainer(
        CosmosConfig(),
        ParallelDims(dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, world_size=1),
        device=torch.device("cpu"),
        data_packer=GymDataPacker(),
        policy=GymPolicy(GymMLPConfig(obs_dim=4, action_dim=2, discrete=True)),
    )


CPU_RUNNABLE_RL_TRAINER_FACTORIES: Dict[str, Callable[[], Any]] = {
    "gym_pg": _build_gym_pg_trainer,
}


# --- Tests --------------------------------------------------------


class TestRLWorkerTrainerSurfaceContract(unittest.TestCase):
    """Every CPU-runnable RL trainer satisfies the rl_worker surface."""

    def test_surface_per_cpu_runnable_rl_trainer(self):
        for trainer_type, build in CPU_RUNNABLE_RL_TRAINER_FACTORIES.items():
            with self.subTest(trainer_type=trainer_type):
                trainer = build()
                assert_rl_worker_trainer_surface(trainer)


if __name__ == "__main__":
    unittest.main()
