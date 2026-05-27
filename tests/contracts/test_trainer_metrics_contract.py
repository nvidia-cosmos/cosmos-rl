# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trainer per-step metrics contract.

The cosmos-rl controller's ``train_ack`` (see
``cosmos_rl/dispatcher/status.py:1040-1080``) reads five required keys
out of every trainer's ``step_training`` return value:

* ``train/loss_avg``       -- ``np.mean`` over per-replica reports
* ``train/loss_max``       -- ``np.max`` over per-replica reports
* ``train/learning_rate``  -- read from the first replica's report
* ``train/iteration_time`` -- ``np.mean`` over per-replica reports
* ``train_step``           -- read from the first replica's report
                              (note: NO ``train/`` prefix; mirrors what
                              SFT/GRPO/VLA trainers populate from the
                              ``current_step`` kwarg ``rl_worker``
                              passes to ``step_training``).

Missing any one of these triggers a ``KeyError`` inside ``train_ack``,
silently dropping that step's metrics (the controller logs a warning
and the run continues but loses data).  This file pins that contract
down as a unit test so a new trainer can't quietly regress it.

Scope:

* :class:`TestTrainerStaticContract` runs over **every** registered
  trainer and asserts the abstract ``step_training`` method exists on
  the class.  This is cheap and catches "forgot to implement".
* :class:`TestTrainerMetricsDynamicContract` actually invokes
  ``step_training`` on a CPU-runnable trainer (``gym_pg``) with
  synthetic rollouts and asserts the returned dict is a superset of
  the required key set.

Other trainers (``sft``, ``grpo``, ``pi_sft``, ``grpo_pi05``,
``grpo_vla``, ``diffusers_sft``, ``diffusion_nft``, ``cosmos_policy``,
``dpo``) are intentionally out of the dynamic check today: they need
HuggingFace weights, GPU device meshes, vllm, or a real dataset to
construct.  When they get a CPU-runnable smoke harness, add them to
``CPU_RUNNABLE_TRAINER_TYPES`` below.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, FrozenSet, Tuple

import numpy as np


# Source of truth: cosmos_rl/dispatcher/status.py train_ack (the
# direct dict subscripts -- not the ``.get(...)`` reads).
REQUIRED_REPORT_KEYS: Tuple[str, ...] = (
    "train/loss_avg",
    "train/loss_max",
    "train/learning_rate",
    "train/iteration_time",
    # ``train_step`` is read at status.py:1080 via direct subscript,
    # NOT via ``.get(...)``; missing it raises KeyError. It also
    # lives outside the ``train/`` namespace -- a footgun worth
    # documenting because a casual ``data.startswith("train/")``
    # filter on a wandb logger would silently drop it.
    "train_step",
)

# Keys that the controller reads with ``.get(key, 0)`` -- not required,
# but documented here so trainer authors know which extras the
# controller surfaces if they're populated.
OPTIONAL_REPORT_KEYS: FrozenSet[str] = frozenset(
    {
        "train/kl_loss_avg",
        "train/kl_loss_max",
        "train/grad_norm",
        "train/entropy",
    }
)


def assert_step_training_metrics_contract(metrics: Dict[str, Any]) -> None:
    """Helper: assert ``metrics`` satisfies the controller's contract.

    Reused by trainer-specific tests (e.g. ``test_gym_example.py``)
    so the contract has exactly one definition site.

    Args:
        metrics: The dict returned by ``trainer.step_training(...)``.

    Raises:
        AssertionError: With a message naming each missing or
            non-finite key, so failures point at the precise gap.
    """
    if not isinstance(metrics, dict):
        raise AssertionError(
            f"step_training must return a dict; got {type(metrics).__name__}"
        )
    missing = [k for k in REQUIRED_REPORT_KEYS if k not in metrics]
    if missing:
        raise AssertionError(
            f"step_training metrics missing required keys {missing}; "
            f"present keys: {sorted(metrics.keys())}"
        )
    non_finite = [
        k
        for k in REQUIRED_REPORT_KEYS
        if not np.isfinite(np.asarray(metrics[k], dtype=float))
    ]
    if non_finite:
        raise AssertionError(
            f"step_training metrics contained non-finite values for {non_finite}"
        )


class TestTrainerStaticContract(unittest.TestCase):
    """Every registered trainer must declare ``step_training``."""

    def _registered_trainers(self):
        # Import lazily so the test file doesn't trigger heavy
        # trainer-module imports at collection time.  Each import is
        # wrapped in a try/except: missing extras (e.g. ``apex`` for
        # diffusers) shouldn't fail the registry sweep.
        from cosmos_rl.policy.trainer.base import TrainerRegistry

        for module in (
            "cosmos_rl.policy.trainer.llm_trainer.sft_trainer",
            "cosmos_rl.policy.trainer.llm_trainer.grpo_trainer",
            "cosmos_rl.tools.gym_example.gym_trainer",
        ):
            try:
                __import__(module)
            except Exception:
                # Optional dependencies; the registry sweep is
                # best-effort and surfaces whatever did import.
                pass
        return dict(TrainerRegistry._TRAINER_REGISTRY)

    def test_every_registered_trainer_has_step_training(self):
        registry = self._registered_trainers()
        self.assertTrue(registry, "TrainerRegistry is empty after imports")
        for trainer_type, trainer_cls in registry.items():
            with self.subTest(trainer_type=trainer_type):
                self.assertTrue(
                    callable(getattr(trainer_cls, "step_training", None)),
                    f"{trainer_cls.__name__} ({trainer_type!r}) does not "
                    "expose a callable ``step_training``; cosmos-rl "
                    "rl_worker / sft_worker depend on this method.",
                )


# Add new entries here as more CPU-runnable trainers ship; the
# corresponding factory must build a ``Trainer`` instance and a
# representative ``rollouts`` argument for ``step_training``.
CPU_RUNNABLE_TRAINER_TYPES: Tuple[str, ...] = ("gym_pg",)


def _build_gym_pg_trainer_and_rollouts():
    """Build a ``GymTrainer`` and two synthetic rollouts on CPU."""
    import torch

    from cosmos_rl.dispatcher.data.schema import Rollout
    from cosmos_rl.policy.config import Config as CosmosConfig
    from cosmos_rl.tools.gym_example.gym_data_packer import GymDataPacker
    from cosmos_rl.tools.gym_example.gym_policy import (
        GymMLPConfig,
        GymPolicy,
    )
    from cosmos_rl.tools.gym_example.gym_rollout import (
        ACTIONS,
        EPISODE_LENGTH,
        OBSERVATIONS,
        REWARDS,
        TERMINATED,
        TRUNCATED,
    )
    from cosmos_rl.tools.gym_example.gym_trainer import GymTrainer
    from cosmos_rl.utils.parallelism import ParallelDims

    rng = np.random.default_rng(0)
    max_steps = 8

    def _rollout(ep_len: int) -> Rollout:
        return Rollout(
            prompt="{}",
            completion={
                OBSERVATIONS: rng.standard_normal((max_steps, 4)).astype(np.float32),
                ACTIONS: rng.integers(0, 2, size=(max_steps,)).astype(np.int64),
                REWARDS: np.ones((max_steps,), dtype=np.float32),
                TERMINATED: np.zeros((max_steps,), dtype=np.bool_),
                TRUNCATED: np.zeros((max_steps,), dtype=np.bool_),
                EPISODE_LENGTH: np.array([ep_len], dtype=np.int64),
            },
        )

    config = CosmosConfig()
    parallel_dims = ParallelDims(
        dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, world_size=1
    )
    trainer = GymTrainer(
        config,
        parallel_dims,
        device=torch.device("cpu"),
        data_packer=GymDataPacker(),
        policy=GymPolicy(GymMLPConfig(obs_dim=4, action_dim=2, discrete=True)),
    )
    return trainer, [_rollout(ep_len=3), _rollout(ep_len=5)]


_CPU_RUNNABLE_FACTORIES = {
    "gym_pg": _build_gym_pg_trainer_and_rollouts,
}


class TestTrainerMetricsDynamicContract(unittest.TestCase):
    """Actually invoke ``step_training`` and verify the returned keys.

    Parameterized over :data:`CPU_RUNNABLE_TRAINER_TYPES`; trainers
    that need GPU / heavy deps stay in the static check until they
    get a CPU smoke-test harness.
    """

    def test_metrics_contract_per_cpu_runnable_trainer(self):
        for trainer_type in CPU_RUNNABLE_TRAINER_TYPES:
            with self.subTest(trainer_type=trainer_type):
                factory = _CPU_RUNNABLE_FACTORIES[trainer_type]
                trainer, rollouts = factory()
                metrics = trainer.step_training(rollouts)
                assert_step_training_metrics_contract(metrics)


if __name__ == "__main__":
    unittest.main()
