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

"""Tests for the Gymnasium Classic Control example (MR4)."""

import os
import tempfile
import unittest

import numpy as np
import torch

from cosmos_rl.dispatcher.data.packer.tensor_data_packer import (
    ACTIONS,
    EPISODE_LENGTH,
    OBSERVATIONS,
    REWARDS,
    TERMINATED,
    TRUNCATED,
)
from cosmos_rl.tools.gym_example import (
    GymDataPacker,
    GymMLPConfig,
    GymPolicy,
    GymRolloutEngine,
    register_gym_policy,
    rollout_episode,
)


# ---------------------------------------------------------------------------
# A minimal stand-in for gymnasium.Env so tests work without installing the
# `gymnasium` extra.
# ---------------------------------------------------------------------------


class _FakeDiscreteEnv:
    """4-dim observation, 2-dim discrete action; episode ends after
    ``terminate_after`` steps with a small per-step reward of ``+1``.

    Faithful enough to validate the rollout engine's contract
    (reset returns ``(obs, info)``, step returns
    ``(obs, reward, terminated, truncated, info)``)."""

    def __init__(self, obs_dim: int = 4, terminate_after: int = 5):
        self.obs_dim = obs_dim
        self.terminate_after = terminate_after
        self._step = 0
        self._rng = np.random.default_rng(0)

    def reset(self, *, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self._step = 0
        return self._rng.standard_normal(self.obs_dim).astype(np.float32), {}

    def step(self, action):
        self._step += 1
        terminated = self._step >= self.terminate_after
        truncated = False
        return (
            self._rng.standard_normal(self.obs_dim).astype(np.float32),
            1.0,
            terminated,
            truncated,
            {},
        )

    def close(self):
        pass


class _FakeContinuousEnv:
    """3-dim observation, 1-dim continuous action; ends after
    ``terminate_after`` steps with reward = -|action|."""

    def __init__(self, action_dim: int = 1, terminate_after: int = 4):
        self.action_dim = action_dim
        self.terminate_after = terminate_after
        self._step = 0

    def reset(self, *, seed=None):
        self._step = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        self._step += 1
        terminated = self._step >= self.terminate_after
        return (
            np.zeros(3, dtype=np.float32),
            -float(np.abs(np.asarray(action)).sum()),
            terminated,
            False,
            {},
        )

    def close(self):
        pass


# ---------------------------------------------------------------------------
# GymPolicy
# ---------------------------------------------------------------------------


class TestGymPolicyDiscrete(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.cfg = GymMLPConfig(obs_dim=4, action_dim=2, hidden_dim=8, discrete=True)
        self.policy = GymPolicy(self.cfg)

    def test_forward_shape(self):
        out = self.policy(torch.zeros(3, 4))
        self.assertEqual(out.shape, (3, 2))

    def test_act_returns_action_and_logprob(self):
        action, logp = self.policy.act(torch.zeros(1, 4))
        self.assertEqual(action.shape, (1,))
        self.assertEqual(action.dtype, torch.int64)
        self.assertEqual(logp.shape, (1,))
        self.assertTrue(torch.all(action >= 0) and torch.all(action < 2))

    def test_value_head_emits_scalar_per_obs(self):
        v = self.policy.value(torch.zeros(5, 4))
        self.assertEqual(v.shape, (5,))


class TestGymPolicyContinuous(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.cfg = GymMLPConfig(obs_dim=3, action_dim=1, hidden_dim=8, discrete=False)
        self.policy = GymPolicy(self.cfg)

    def test_head_emits_two_action_dim_values(self):
        # Continuous head outputs (mean, log_std) so total = 2*action_dim.
        out = self.policy(torch.zeros(2, 3))
        self.assertEqual(out.shape, (2, 2))

    def test_act_returns_float_action(self):
        action, logp = self.policy.act(torch.zeros(1, 3))
        self.assertEqual(action.shape, (1, 1))
        self.assertEqual(action.dtype, torch.float32)
        self.assertEqual(logp.shape, (1,))


# ---------------------------------------------------------------------------
# GymDataPacker
# ---------------------------------------------------------------------------


class TestGymDataPacker(unittest.TestCase):
    def test_get_rollout_input_decodes_json_prompt(self):
        p = GymDataPacker()
        out = p.get_rollout_input({"prompt": '{"seed": 42}'})
        self.assertEqual(out, {"seed": 42})

    def test_get_rollout_input_passthrough_dict_prompt(self):
        p = GymDataPacker()
        out = p.get_rollout_input({"prompt": {"seed": 7, "deterministic": True}})
        self.assertEqual(out, {"seed": 7, "deterministic": True})

    def test_get_rollout_input_empty_prompt_returns_empty_dict(self):
        p = GymDataPacker()
        self.assertEqual(p.get_rollout_input({"prompt": ""}), {})
        self.assertEqual(p.get_rollout_input({}), {})

    def test_get_rollout_input_non_json_falls_back_to_dict(self):
        p = GymDataPacker()
        out = p.get_rollout_input({"prompt": "not-json-but-truthy"})
        self.assertEqual(out, {"prompt": "not-json-but-truthy"})

    def test_policy_compute_max_len_uses_episode_length(self):
        p = GymDataPacker()
        traj = [
            {OBSERVATIONS: np.zeros((10, 4)), EPISODE_LENGTH: 7},
            {OBSERVATIONS: np.zeros((10, 4)), EPISODE_LENGTH: 3},
        ]
        self.assertEqual(p.policy_compute_max_len(traj), 7)

    def test_policy_compute_max_len_falls_back_to_obs_shape(self):
        p = GymDataPacker()
        traj = [{OBSERVATIONS: np.zeros((9, 4))}]
        self.assertEqual(p.policy_compute_max_len(traj), 9)


class TestGymDataPackerTrajectoryProtocol(unittest.TestCase):
    """``GymDataPacker`` satisfies the ``TrajectoryPacker`` Protocol so
    a trainer composing :class:`TrajectoryExpansionMixin` can iterate
    rollouts at the rollout / chunk / transition scope.
    """

    def _make_rollout(self, ep_len: int, max_steps: int = 10):
        from cosmos_rl.dispatcher.data.schema import Rollout

        traj = {
            OBSERVATIONS: np.arange(max_steps * 4, dtype=np.float32).reshape(
                max_steps, 4
            ),
            ACTIONS: np.arange(max_steps, dtype=np.int64),
            REWARDS: np.arange(max_steps, dtype=np.float32),
            TERMINATED: np.zeros((max_steps,), dtype=np.bool_),
            TRUNCATED: np.zeros((max_steps,), dtype=np.bool_),
            EPISODE_LENGTH: np.array([ep_len], dtype=np.int64),
        }
        return Rollout(prompt="{}", completion=traj)

    def test_satisfies_trajectory_packer_protocol(self):
        from cosmos_rl.dispatcher.data.packer.trajectory_packer import (
            TrajectoryPacker,
        )

        self.assertIsInstance(GymDataPacker(), TrajectoryPacker)

    def test_num_transitions_matches_episode_length(self):
        p = GymDataPacker()
        self.assertEqual(p.num_transitions(self._make_rollout(ep_len=5)), 5)
        self.assertEqual(p.num_transitions(self._make_rollout(ep_len=0)), 0)
        self.assertEqual(
            p.num_transitions(self._make_rollout(ep_len=10, max_steps=10)), 10
        )

    def test_num_transitions_accepts_python_int(self):
        from cosmos_rl.dispatcher.data.schema import Rollout

        traj = {
            OBSERVATIONS: np.zeros((4, 4), dtype=np.float32),
            ACTIONS: np.zeros((4,), dtype=np.int64),
            REWARDS: np.zeros((4,), dtype=np.float32),
            TERMINATED: np.zeros((4,), dtype=np.bool_),
            TRUNCATED: np.zeros((4,), dtype=np.bool_),
            EPISODE_LENGTH: 3,
        }
        self.assertEqual(
            GymDataPacker().num_transitions(Rollout(prompt="{}", completion=traj)),
            3,
        )

    def test_iter_transitions_yields_ep_len_dicts(self):
        p = GymDataPacker()
        rollout = self._make_rollout(ep_len=3, max_steps=10)
        steps = list(p.iter_transitions(rollout))
        self.assertEqual(len(steps), 3)
        for t, step in enumerate(steps):
            self.assertEqual(set(step.keys()), {"observation", "action", "reward"})
            self.assertEqual(int(step["action"]), t)
            self.assertEqual(float(step["reward"]), float(t))

    def test_iter_transitions_skips_padding(self):
        """Padded positions (``[ep_len:]``) must not be yielded."""
        p = GymDataPacker()
        rollout = self._make_rollout(ep_len=2, max_steps=10)
        steps = list(p.iter_transitions(rollout))
        self.assertEqual(len(steps), 2)
        actions = [int(s["action"]) for s in steps]
        self.assertEqual(actions, [0, 1])  # not 0..9

    def test_iter_chunks_default_body_collects_transitions(self):
        """``iter_chunks`` is inherited from the Protocol's default body
        (collects ``iter_transitions`` into chunks of size ``chunk_size``).
        """
        p = GymDataPacker()
        rollout = self._make_rollout(ep_len=5, max_steps=10)
        chunks = list(p.iter_chunks(rollout, chunk_size=2))
        self.assertEqual([len(c) for c in chunks], [2, 2, 1])  # 5 -> 2+2+1

    def test_iter_rollouts_default_body_yields_single_batch(self):
        p = GymDataPacker()
        rollouts = [self._make_rollout(ep_len=2) for _ in range(3)]
        batches = list(p.iter_rollouts(rollouts))
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), 3)

    def test_iter_rollouts_empty_input_yields_zero_batches(self):
        self.assertEqual(list(GymDataPacker().iter_rollouts([])), [])

    def test_trajectory_in_extra_info_fallback(self):
        """If a transport-aware override stashes the trajectory in
        ``extra_info`` instead of ``completion``, the protocol still
        works."""
        from cosmos_rl.dispatcher.data.schema import Rollout

        traj = {
            OBSERVATIONS: np.zeros((3, 4), dtype=np.float32),
            ACTIONS: np.array([7, 8, 9], dtype=np.int64),
            REWARDS: np.zeros((3,), dtype=np.float32),
            TERMINATED: np.zeros((3,), dtype=np.bool_),
            TRUNCATED: np.zeros((3,), dtype=np.bool_),
            EPISODE_LENGTH: 3,
        }
        # completion is a stub (e.g. a UCXX handle), trajectory is in extra_info.
        rollout = Rollout(prompt="{}", completion="ucxx://slot/42", extra_info=traj)
        p = GymDataPacker()
        self.assertEqual(p.num_transitions(rollout), 3)
        actions = [int(s["action"]) for s in p.iter_transitions(rollout)]
        self.assertEqual(actions, [7, 8, 9])


# ---------------------------------------------------------------------------
# Toy PG algorithm helpers (gym_algo.py)
# ---------------------------------------------------------------------------


class TestComputeReturns(unittest.TestCase):
    def test_undiscounted_returns_match_reverse_cumsum(self):
        from cosmos_rl.tools.gym_example.gym_algo import compute_returns

        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        got = compute_returns(rewards, gamma=1.0)
        # R[t] = sum_{k>=t} r[k]; for [1,2,3,4] -> [10, 9, 7, 4].
        self.assertTrue(torch.allclose(got, torch.tensor([10.0, 9.0, 7.0, 4.0])))

    def test_discounted_returns_hand_computed_reference(self):
        from cosmos_rl.tools.gym_example.gym_algo import compute_returns

        rewards = torch.tensor([1.0, 1.0, 1.0])
        got = compute_returns(rewards, gamma=0.5)
        # R[2] = 1
        # R[1] = 1 + 0.5 * 1 = 1.5
        # R[0] = 1 + 0.5 * 1.5 = 1.75
        self.assertTrue(torch.allclose(got, torch.tensor([1.75, 1.5, 1.0])))

    def test_empty_rewards_returns_empty(self):
        from cosmos_rl.tools.gym_example.gym_algo import compute_returns

        out = compute_returns(torch.empty(0))
        self.assertEqual(tuple(out.shape), (0,))

    def test_rejects_non_1d(self):
        from cosmos_rl.tools.gym_example.gym_algo import compute_returns

        with self.assertRaises(ValueError):
            compute_returns(torch.zeros((3, 2)))

    def test_rejects_gamma_out_of_range(self):
        from cosmos_rl.tools.gym_example.gym_algo import compute_returns

        with self.assertRaises(ValueError):
            compute_returns(torch.tensor([1.0]), gamma=-0.1)
        with self.assertRaises(ValueError):
            compute_returns(torch.tensor([1.0]), gamma=1.01)


class TestComputeSimplePGLoss(unittest.TestCase):
    def test_discrete_loss_is_finite_and_has_grad(self):
        from cosmos_rl.tools.gym_example.gym_algo import compute_simple_pg_loss

        T, K = 5, 2
        preds = torch.zeros((T, K), requires_grad=True)
        targets = torch.tensor([0, 1, 0, 1, 0])
        returns = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        loss, metrics = compute_simple_pg_loss(preds, targets, returns)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss.requires_grad)
        self.assertEqual(metrics["num_steps"], T)
        self.assertEqual(metrics["mean_return"], 1.0)
        loss.backward()
        self.assertIsNotNone(preds.grad)

    def test_continuous_loss_matches_mse_when_returns_one(self):
        from cosmos_rl.tools.gym_example.gym_algo import compute_simple_pg_loss

        preds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], requires_grad=True)
        targets = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
        returns = torch.ones(2)
        loss, _ = compute_simple_pg_loss(preds, targets, returns)
        # per-step MSE = mean(squared diff per dim) = 1.0 each, mean = 1.0
        self.assertTrue(torch.allclose(loss, torch.tensor(1.0)))

    def test_zero_returns_yield_zero_loss(self):
        from cosmos_rl.tools.gym_example.gym_algo import compute_simple_pg_loss

        preds = torch.tensor([[1.0, -1.0], [2.0, 0.5]], requires_grad=True)
        targets = torch.tensor([0, 1])
        returns = torch.zeros(2)
        loss, _ = compute_simple_pg_loss(preds, targets, returns)
        self.assertEqual(float(loss), 0.0)

    def test_negative_returns_flip_sign_relative_to_positive(self):
        """Sanity: same trajectory with returns of -1 yields the
        negative of the positive-returns case (toy semantic)."""
        from cosmos_rl.tools.gym_example.gym_algo import compute_simple_pg_loss

        preds = torch.tensor([[0.0, 0.0]], requires_grad=True)
        targets = torch.tensor([0])
        loss_pos, _ = compute_simple_pg_loss(preds, targets, torch.tensor([1.0]))
        loss_neg, _ = compute_simple_pg_loss(preds, targets, torch.tensor([-1.0]))
        self.assertTrue(torch.allclose(loss_pos, -loss_neg))

    def test_rejects_shape_mismatch(self):
        from cosmos_rl.tools.gym_example.gym_algo import compute_simple_pg_loss

        with self.assertRaises(ValueError):
            compute_simple_pg_loss(
                torch.zeros((3, 2)),
                torch.tensor([0, 1]),  # length 2, T=3 -> mismatch
                torch.zeros(3),
            )
        with self.assertRaises(ValueError):
            compute_simple_pg_loss(
                torch.zeros((3, 2)),
                torch.tensor([0, 1, 0]),
                torch.zeros(2),  # returns length 2, T=3 -> mismatch
            )


# ---------------------------------------------------------------------------
# GymTrainer (composes TrajectoryExpansionMixin, rollout mode)
# ---------------------------------------------------------------------------


def _make_synthetic_rollout(ep_len: int, max_steps: int = 10, seed: int = 0):
    """Build a Rollout whose completion is a synthetic trajectory dict."""
    from cosmos_rl.dispatcher.data.schema import Rollout

    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((max_steps, 4)).astype(np.float32)
    actions = rng.integers(0, 2, size=(max_steps,)).astype(np.int64)
    rewards = np.ones((max_steps,), dtype=np.float32)
    return Rollout(
        prompt="{}",
        completion={
            OBSERVATIONS: obs,
            ACTIONS: actions,
            REWARDS: rewards,
            TERMINATED: np.zeros((max_steps,), dtype=np.bool_),
            TRUNCATED: np.zeros((max_steps,), dtype=np.bool_),
            EPISODE_LENGTH: np.array([ep_len], dtype=np.int64),
        },
    )


def _make_trainer(policy=None):
    """Construct a GymTrainer wired to CPU with a fresh GymPolicy."""
    from cosmos_rl.policy.config import Config as CosmosConfig
    from cosmos_rl.tools.gym_example.gym_trainer import GymTrainer
    from cosmos_rl.utils.parallelism import ParallelDims

    config = CosmosConfig()
    parallel_dims = ParallelDims(
        dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, world_size=1
    )
    if policy is None:
        policy = GymPolicy(GymMLPConfig(obs_dim=4, action_dim=2, discrete=True))
    return GymTrainer(
        config,
        parallel_dims,
        device=torch.device("cpu"),
        data_packer=GymDataPacker(),
        policy=policy,
    )


class TestGymTrainerComposition(unittest.TestCase):
    """The trainer correctly composes ``TrajectoryExpansionMixin``."""

    def test_chunk_size_is_none_rollout_mode(self):
        from cosmos_rl.tools.gym_example.gym_trainer import GymTrainer

        self.assertIsNone(GymTrainer.chunk_size)

    def test_inherits_trajectory_expansion_mixin(self):
        from cosmos_rl.policy.trainer.trajectory_mixin import (
            TrajectoryExpansionMixin,
        )
        from cosmos_rl.tools.gym_example.gym_trainer import GymTrainer

        self.assertTrue(issubclass(GymTrainer, TrajectoryExpansionMixin))

    def test_registered_under_gym_pg(self):
        from cosmos_rl.policy.trainer.base import TrainerRegistry
        from cosmos_rl.tools.gym_example.gym_trainer import GymTrainer

        self.assertIs(TrainerRegistry.get_trainer_cls("gym_pg"), GymTrainer)

    def test_packer_protocol_assertion_fires_on_non_trajectory_packer(self):
        """A misconfigured ``data_packer`` must trip the mixin's
        ``isinstance(self.data_packer, TrajectoryPacker)`` assertion."""
        from cosmos_rl.dispatcher.data.packer.tensor_data_packer import (
            TensorDataPacker,
        )
        from cosmos_rl.policy.config import Config as CosmosConfig
        from cosmos_rl.tools.gym_example.gym_trainer import GymTrainer
        from cosmos_rl.utils.parallelism import ParallelDims

        # Plain TensorDataPacker doesn't implement TrajectoryPacker.
        trainer = GymTrainer(
            CosmosConfig(),
            ParallelDims(dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, world_size=1),
            device=torch.device("cpu"),
            data_packer=TensorDataPacker(),
            policy=GymPolicy(GymMLPConfig()),
        )
        with self.assertRaises(AssertionError) as ctx:
            trainer.step_training([_make_synthetic_rollout(ep_len=2)])
        self.assertIn("TrajectoryPacker", str(ctx.exception))


class TestGymTrainerStepTraining(unittest.TestCase):
    """End-to-end: ``step_training`` on a fake batch applies real gradients."""

    def test_runs_and_applies_gradients(self):
        torch.manual_seed(0)
        trainer = _make_trainer()
        params_before = [p.detach().clone() for p in trainer.model.parameters()]

        rollouts = [
            _make_synthetic_rollout(ep_len=3, seed=1),
            _make_synthetic_rollout(ep_len=5, seed=2),
        ]
        metrics = trainer.step_training(rollouts)

        # Metrics shape: keys mirror cosmos-rl's per-step report contract
        # (``train/loss_avg``, ``train/loss_max``, ``train/learning_rate``,
        # ``train/iteration_time``, ``train_step`` are required by the
        # controller; the rest are gym-specific extras).
        for key in (
            "train/loss_avg",
            "train/loss_max",
            "train/learning_rate",
            "train/iteration_time",
            "train/grad_norm",
            "train/mean_return",
            "train/mean_episode_length",
            "train_step",
        ):
            self.assertIn(key, metrics)
            self.assertTrue(np.isfinite(metrics[key]), f"{key} not finite")
        self.assertEqual(metrics["train/num_rollouts"], 2)

        # At least one parameter changed.
        params_after = list(trainer.model.parameters())
        any_changed = any(
            not torch.allclose(b, a) for b, a in zip(params_before, params_after)
        )
        self.assertTrue(any_changed, "no policy parameter changed after step_training")

    def test_skips_zero_length_rollouts(self):
        torch.manual_seed(0)
        trainer = _make_trainer()
        rollouts = [
            _make_synthetic_rollout(ep_len=0),
            _make_synthetic_rollout(ep_len=4, seed=1),
        ]
        metrics = trainer.step_training(rollouts)
        self.assertEqual(metrics["train/num_rollouts"], 2)
        # Only one rollout actually contributed metrics; the average is
        # over the contributing rollout.
        self.assertEqual(metrics["train/mean_episode_length"], 4.0)

    def test_empty_rollouts_returns_metrics_without_crashing(self):
        trainer = _make_trainer()
        metrics = trainer.step_training([])
        # num_rollouts is floored at 1 to avoid divide-by-zero in the
        # backward scaling, but the metric reported reflects that floor;
        # the test asserts the call completes and returns a dict.
        self.assertIsInstance(metrics, dict)
        self.assertEqual(metrics["train/mean_episode_length"], 0.0)


class TestGymTrainerLauncherSurface(unittest.TestCase):
    """Tier-B surface: the trainer exposes the methods/attributes the
    ``rl_worker`` introspects beyond :class:`Trainer`'s abstract base.

    These tests cover the colocated-single-replica launch path; they
    don't bring up Redis or torch.distributed, just verify the surface
    is the right shape so a launch can get past
    ``prepare_shard_infos_for_weight_sync_insts`` and the per-step
    command handlers.

    The same structural shape is asserted generically (across all
    CPU-runnable RL trainers) by
    ``tests/contracts/test_rl_worker_trainer_surface_contract.py``;
    the tests here additionally verify gym-specific choices
    (``IdentityWeightMapper``, empty ``map_w_from_policy_to_rollout``,
    no-op ``weight_resume`` returning ``{}``, etc.).
    """

    def test_model_exposes_weight_mapper(self):
        from cosmos_rl.policy.model.base import IdentityWeightMapper

        trainer = _make_trainer()
        self.assertIsInstance(trainer.model.weight_mapper, IdentityWeightMapper)

    def test_model_trainable_params_lists_all_required_grad_params(self):
        trainer = _make_trainer()
        names = trainer.model.trainable_params
        self.assertIsInstance(names, list)
        # Every requires_grad=True param has a name in the list.
        expected = {n for n, p in trainer.model.named_parameters() if p.requires_grad}
        self.assertEqual(set(names), expected)
        # Empty trainable list would crash the worker; guard against
        # accidentally freezing everything.
        self.assertGreater(len(names), 0)

    def test_model_weight_sync_transforms_pairs_name_and_param(self):
        trainer = _make_trainer()
        transforms = trainer.model.weight_sync_transforms
        self.assertIsInstance(transforms, list)
        self.assertGreater(len(transforms), 0)
        for name, value in transforms:
            self.assertIsInstance(name, str)
            self.assertIsInstance(value, torch.Tensor)

    def test_trainer_weight_mapper_matches_model(self):
        # rl_worker reads both ``trainer.weight_mapper`` and
        # ``trainer.model.weight_mapper``; they should be the same object
        # so name-mapping decisions are consistent across the two read sites.
        trainer = _make_trainer()
        self.assertIs(trainer.weight_mapper, trainer.model.weight_mapper)

    def test_map_w_from_policy_to_rollout_is_empty_in_colocated(self):
        # Empty dict short-circuits ``pre_P2R_collect_parameters`` and the
        # P2R sync inner loop: in colocated mode the rollout backend
        # shares the policy's nn.Module reference, so no tensor needs
        # to be transmitted.
        trainer = _make_trainer()
        self.assertEqual(trainer.map_w_from_policy_to_rollout, {})

    def test_weight_resume_returns_empty_dict(self):
        trainer = _make_trainer()
        self.assertEqual(trainer.weight_resume(), {})

    def test_update_lr_schedulers_is_a_noop(self):
        trainer = _make_trainer()
        # Builds the scheduler so we can verify nothing changed across the call.
        trainer.build_lr_schedulers()
        before = trainer.lr_scheduler.state_dict()
        trainer.update_lr_schedulers(total_steps=42)
        self.assertEqual(trainer.lr_scheduler.state_dict(), before)

    def test_sync_all_states_returns_zero(self):
        trainer = _make_trainer()
        result = trainer.sync_all_states(
            is_send=True,
            send_hook=lambda *a, **kw: None,
            recv_hook=lambda *a, **kw: None,
        )
        self.assertEqual(result, 0)


class TestGymRolloutBackend(unittest.TestCase):
    """The :class:`RolloutBase` adapter runs an episode and packs a
    :class:`RolloutResult`, using a fake env factory so the test does
    not require gymnasium to be installed."""

    def _make_backend(self, policy=None, terminate_after=4):
        from cosmos_rl.policy.config import Config as CosmosConfig
        from cosmos_rl.tools.gym_example.gym_rollout_backend import (
            GymRolloutBackend,
        )

        if policy is None:
            policy = GymPolicy(GymMLPConfig(obs_dim=4, action_dim=2, discrete=True))

        return GymRolloutBackend(
            CosmosConfig(),
            parallel_dims=None,
            device=torch.device("cpu"),
            policy=policy,
            env_factory=lambda: _FakeDiscreteEnv(
                obs_dim=4, terminate_after=terminate_after
            ),
        )

    def test_registered_under_gym(self):
        from cosmos_rl.rollout.rollout_base import RolloutRegistry
        from cosmos_rl.tools.gym_example.gym_rollout_backend import (
            GymRolloutBackend,
        )

        self.assertIs(RolloutRegistry.get_rollout_cls("gym"), GymRolloutBackend)

    def test_rollout_generation_emits_one_result_per_payload(self):
        backend = self._make_backend(terminate_after=3)
        backend.init_engine()

        class _Payload:
            def __init__(self, prompt):
                self.prompt = prompt

        payloads = [_Payload('{"seed": 1}'), _Payload('{"seed": 2}')]
        results = backend.rollout_generation(payloads, data_packer=GymDataPacker())

        self.assertEqual(len(results), 2)
        for r in results:
            self.assertEqual(len(r.completions), 1)
            traj = r.completions[0]
            self.assertIn(OBSERVATIONS, traj)
            self.assertIn(ACTIONS, traj)
            self.assertIn(EPISODE_LENGTH, traj)
            self.assertEqual(int(traj[EPISODE_LENGTH][0]), 3)
        backend.shutdown()

    def test_get_and_set_underlying_model_share_policy(self):
        backend = self._make_backend()
        backend.init_engine()
        self.assertIs(backend.get_underlying_model(), backend._policy)

        new_policy = GymPolicy(GymMLPConfig())
        backend.set_underlying_model(new_policy)
        self.assertIs(backend.get_underlying_model(), new_policy)
        self.assertIs(backend._engine.policy, new_policy)
        backend.shutdown()

    def test_set_underlying_model_rejects_wrong_type(self):
        backend = self._make_backend()
        with self.assertRaises(TypeError):
            backend.set_underlying_model(torch.nn.Linear(4, 2))

    def test_init_engine_idempotent(self):
        backend = self._make_backend()
        backend.init_engine()
        engine_a = backend._engine
        backend.init_engine()
        self.assertIs(backend._engine, engine_a)
        backend.shutdown()

    def test_rollout_generation_before_init_engine_raises(self):
        """Calling rollout_generation before init_engine is a programming
        error and propagates loudly.  RolloutGenerationMixin's
        ``_assert_engine_initialized`` raises ``RuntimeError`` outside
        the template's ``try`` block, so it is not swallowed by
        ``_on_generation_error``."""
        backend = self._make_backend()

        class _Payload:
            prompt = "{}"

        with self.assertRaises(RuntimeError):
            backend.rollout_generation([_Payload()])


class TestGymBackendStructured(unittest.TestCase):
    """Post-migration: ``GymRolloutBackend`` composes
    :class:`RolloutGenerationMixin`.  Verifies the four hooks fire in
    the documented order and the returned ``RolloutResult`` shape is
    unchanged from the pre-mixin code path."""

    def _make_backend(self, terminate_after=3):
        from cosmos_rl.policy.config import Config as CosmosConfig
        from cosmos_rl.tools.gym_example.gym_rollout_backend import (
            GymRolloutBackend,
        )

        return GymRolloutBackend(
            CosmosConfig(),
            parallel_dims=None,
            device=torch.device("cpu"),
            policy=GymPolicy(GymMLPConfig(obs_dim=4, action_dim=2, discrete=True)),
            env_factory=lambda: _FakeDiscreteEnv(
                obs_dim=4, terminate_after=terminate_after
            ),
        )

    def test_inherits_from_rollout_generation_mixin(self):
        from cosmos_rl.rollout.generation_mixin import RolloutGenerationMixin
        from cosmos_rl.tools.gym_example.gym_rollout_backend import (
            GymRolloutBackend,
        )

        self.assertIn(RolloutGenerationMixin, GymRolloutBackend.__mro__)

    def test_hook_dispatch_order_recorded(self):
        from cosmos_rl.tools.gym_example.gym_rollout_backend import (
            GymRolloutBackend,
        )

        events: list[str] = []

        class _RecordingBackend(GymRolloutBackend):
            def _prepare_sample(self, *a, **kw):
                events.append("prepare")
                return super()._prepare_sample(*a, **kw)

            def _collate_batch(self, *a, **kw):
                events.append("collate")
                return super()._collate_batch(*a, **kw)

            def _generate(self, *a, **kw):
                events.append("generate")
                return super()._generate(*a, **kw)

            def _postprocess(self, *a, **kw):
                events.append("postprocess")
                return super()._postprocess(*a, **kw)

        from cosmos_rl.policy.config import Config as CosmosConfig

        backend = _RecordingBackend(
            CosmosConfig(),
            parallel_dims=None,
            device=torch.device("cpu"),
            policy=GymPolicy(GymMLPConfig()),
            env_factory=lambda: _FakeDiscreteEnv(terminate_after=2),
        )
        backend.init_engine()

        class _Payload:
            def __init__(self, prompt, prompt_idx):
                self.prompt = prompt
                self.prompt_idx = prompt_idx

        results = backend.rollout_generation(
            [
                _Payload('{"seed": 1}', prompt_idx=0),
                _Payload('{"seed": 2}', prompt_idx=1),
            ],
            data_packer=GymDataPacker(),
        )
        backend.shutdown()

        # 2 prepare + 1 collate + 1 generate + 1 postprocess.
        self.assertEqual(
            events,
            ["prepare", "prepare", "collate", "generate", "postprocess"],
        )
        self.assertEqual(len(results), 2)
        for r in results:
            traj = r.completions[0]
            self.assertIn(OBSERVATIONS, traj)
            self.assertIn(EPISODE_LENGTH, traj)

    def test_rollout_result_shape_unchanged_after_migration(self):
        backend = self._make_backend(terminate_after=3)
        backend.init_engine()

        class _Payload:
            def __init__(self, prompt, prompt_idx):
                self.prompt = prompt
                self.prompt_idx = prompt_idx

        results = backend.rollout_generation(
            [_Payload('{"seed": 1}', prompt_idx=0)],
            data_packer=GymDataPacker(),
        )
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(len(r.completions), 1)
        traj = r.completions[0]
        self.assertIn(OBSERVATIONS, traj)
        self.assertIn(ACTIONS, traj)
        self.assertIn(EPISODE_LENGTH, traj)
        self.assertEqual(int(traj[EPISODE_LENGTH][0]), 3)
        # Prompt is the parsed init dict (echoed-back via _postprocess).
        self.assertEqual(r.prompt, {"seed": 1})
        backend.shutdown()


class TestGymBackendPrefetch(unittest.TestCase):
    """``prefetch_rollout=True``: ``_prepare_sample`` runs on the bg
    setup worker.  When ``submit_setup`` is called before
    ``rollout_generation``, a deliberately slow ``_generate`` for batch
    B leaves the bg worker enough time to finish prepare for batch
    B+1 ahead of the next ``rollout_generation`` call."""

    def _make_backend_with_prefetch(self, terminate_after=2):
        from cosmos_rl.policy.config import Config as CosmosConfig
        from cosmos_rl.tools.gym_example.gym_rollout_backend import (
            GymRolloutBackend,
        )

        cfg = CosmosConfig()
        cfg.rollout.prefetch_rollout = True
        return GymRolloutBackend(
            cfg,
            parallel_dims=None,
            device=torch.device("cpu"),
            policy=GymPolicy(GymMLPConfig(obs_dim=4, action_dim=2, discrete=True)),
            env_factory=lambda: _FakeDiscreteEnv(
                obs_dim=4, terminate_after=terminate_after
            ),
        )

    def test_prepare_runs_on_bg_thread_when_prefetch_on(self):
        from cosmos_rl.tools.gym_example.gym_rollout_backend import (
            GymRolloutBackend,
        )

        captured_threads: list[str] = []

        class _Recording(GymRolloutBackend):
            def _prepare_sample(self, *a, **kw):
                import threading as _t

                captured_threads.append(_t.current_thread().name)
                return super()._prepare_sample(*a, **kw)

        from cosmos_rl.policy.config import Config as CosmosConfig

        cfg = CosmosConfig()
        cfg.rollout.prefetch_rollout = True
        backend = _Recording(
            cfg,
            parallel_dims=None,
            device=torch.device("cpu"),
            policy=GymPolicy(GymMLPConfig()),
            env_factory=lambda: _FakeDiscreteEnv(terminate_after=2),
        )
        # Bind the data_packer the bg worker should use.
        backend.bind_prefetch_context(data_packer=GymDataPacker())
        backend.init_engine()

        class _Payload:
            def __init__(self, prompt, prompt_idx):
                self.prompt = prompt
                self.prompt_idx = prompt_idx

        payloads = [
            _Payload('{"seed": 1}', prompt_idx=0),
            _Payload('{"seed": 2}', prompt_idx=1),
        ]
        backend.submit_setup(payloads)
        results = backend.rollout_generation(payloads, data_packer=GymDataPacker())
        backend.shutdown()

        self.assertEqual(len(results), 2)
        # All prepare invocations must have run on the bg setup worker
        # (not on the main test thread).
        import threading as _t

        main_name = _t.current_thread().name
        for n in captured_threads:
            self.assertNotEqual(n, main_name)
            self.assertEqual(n, "GymRolloutPrefetch")

    def test_setup_thread_started_iff_prefetch_enabled(self):
        # Off: no setup thread.
        from cosmos_rl.policy.config import Config as CosmosConfig
        from cosmos_rl.tools.gym_example.gym_rollout_backend import (
            GymRolloutBackend,
        )

        cfg_off = CosmosConfig()
        backend_off = GymRolloutBackend(
            cfg_off,
            parallel_dims=None,
            device=torch.device("cpu"),
            policy=GymPolicy(GymMLPConfig()),
            env_factory=lambda: _FakeDiscreteEnv(terminate_after=2),
        )
        self.assertIsNone(backend_off._setup_thread)
        backend_off.shutdown()

        # On: setup thread is alive.
        cfg_on = CosmosConfig()
        cfg_on.rollout.prefetch_rollout = True
        backend_on = GymRolloutBackend(
            cfg_on,
            parallel_dims=None,
            device=torch.device("cpu"),
            policy=GymPolicy(GymMLPConfig()),
            env_factory=lambda: _FakeDiscreteEnv(terminate_after=2),
        )
        self.assertIsNotNone(backend_on._setup_thread)
        self.assertTrue(backend_on._setup_thread.is_alive())
        backend_on.shutdown()
        self.assertFalse(backend_on._setup_thread.is_alive())


class TestGymEntry(unittest.TestCase):
    """Importing :mod:`cosmos_rl.tools.gym_example.gym_entry` registers
    the trainer + rollout backend and exposes a working seed dataset
    and reward function.
    """

    def test_import_registers_gym_pg_and_gym(self):
        # Import the entry module and verify both registries resolve.
        import cosmos_rl.tools.gym_example.gym_entry  # noqa: F401
        from cosmos_rl.policy.trainer.base import TrainerRegistry
        from cosmos_rl.rollout.rollout_base import RolloutRegistry

        self.assertTrue(TrainerRegistry.check_trainer_type_supported("gym_pg"))
        self.assertTrue(RolloutRegistry.check_rollout_type_supported("gym"))

    def test_seed_dataset_yields_json_encoded_seed(self):
        from cosmos_rl.tools.gym_example.gym_entry import GymSeedDataset

        ds = GymSeedDataset(size=5, seed_offset=100)
        self.assertEqual(len(ds), 5)
        item = ds[2]
        self.assertEqual(set(item.keys()), {"prompt"})
        # JSON-decodable, with the offset applied.
        import json as _json

        self.assertEqual(_json.loads(item["prompt"]), {"seed": 102})

    def test_episode_reward_sums_valid_prefix(self):
        from cosmos_rl.tools.gym_example.gym_entry import gym_episode_reward

        completion = {
            REWARDS: np.array([1.0, 2.0, 3.0, 99.0], dtype=np.float32),
            EPISODE_LENGTH: np.array([3], dtype=np.int64),
        }
        self.assertEqual(gym_episode_reward(completion), 6.0)

    def test_episode_reward_missing_episode_length_sums_full_array(self):
        from cosmos_rl.tools.gym_example.gym_entry import gym_episode_reward

        completion = {REWARDS: np.array([1.0, 1.0, 1.0], dtype=np.float32)}
        self.assertEqual(gym_episode_reward(completion), 3.0)

    def test_episode_reward_returns_zero_on_malformed_completion(self):
        from cosmos_rl.tools.gym_example.gym_entry import gym_episode_reward

        self.assertEqual(gym_episode_reward("not a dict"), 0.0)
        self.assertEqual(gym_episode_reward({}), 0.0)
        self.assertEqual(gym_episode_reward(None), 0.0)


class TestGymTrainerHookOrdering(unittest.TestCase):
    """The mixin invokes ``_begin_training_step``, then N x ``_train_one_rollout``,
    then ``_finalize_training_step``, in that order."""

    def test_phase_order_recorded_via_subclass(self):
        from cosmos_rl.policy.config import Config as CosmosConfig
        from cosmos_rl.tools.gym_example.gym_trainer import GymTrainer
        from cosmos_rl.utils.parallelism import ParallelDims

        events: list[str] = []

        class _RecordingGymTrainer(GymTrainer):
            def _begin_training_step(self, rollouts, *a, **kw):
                events.append("begin")
                super()._begin_training_step(rollouts, *a, **kw)

            def _train_one_rollout(self, rollout, *a, **kw):
                events.append("rollout")
                super()._train_one_rollout(rollout, *a, **kw)

            def _finalize_training_step(self, rollouts, *a, **kw):
                events.append("finalize")
                return super()._finalize_training_step(rollouts, *a, **kw)

        trainer = _RecordingGymTrainer(
            CosmosConfig(),
            ParallelDims(dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, world_size=1),
            device=torch.device("cpu"),
            data_packer=GymDataPacker(),
            policy=GymPolicy(GymMLPConfig()),
        )
        trainer.step_training(
            [_make_synthetic_rollout(ep_len=2, seed=i) for i in range(3)]
        )
        self.assertEqual(
            events,
            ["begin", "rollout", "rollout", "rollout", "finalize"],
        )


# ---------------------------------------------------------------------------
# Rollout engine
# ---------------------------------------------------------------------------


class TestRolloutEpisodeDiscrete(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        cfg = GymMLPConfig(obs_dim=4, action_dim=2, hidden_dim=8, discrete=True)
        self.policy = GymPolicy(cfg)

    def test_trajectory_shape_and_dtypes(self):
        env = _FakeDiscreteEnv(obs_dim=4, terminate_after=3)
        traj = rollout_episode(env, self.policy, max_steps=8, seed=1)
        self.assertEqual(traj[OBSERVATIONS].shape, (8, 4))
        self.assertEqual(traj[OBSERVATIONS].dtype, np.float32)
        self.assertEqual(traj[ACTIONS].shape, (8,))
        self.assertEqual(traj[ACTIONS].dtype, np.int64)
        self.assertEqual(traj[REWARDS].shape, (8,))
        self.assertEqual(traj[TERMINATED].shape, (8,))
        self.assertEqual(traj[TRUNCATED].shape, (8,))
        # Episode terminates after 3 steps -> ep_len == 3.
        self.assertEqual(int(traj[EPISODE_LENGTH][0]), 3)
        # Reward is +1 per valid step; padding bytes after ep_len remain 0.
        self.assertAlmostEqual(float(traj[REWARDS][:3].sum()), 3.0)
        self.assertEqual(float(traj[REWARDS][3:].sum()), 0.0)

    def test_runs_to_max_steps_when_env_does_not_terminate(self):
        # An env that never terminates fills the whole padded length.
        env = _FakeDiscreteEnv(obs_dim=4, terminate_after=10**6)
        traj = rollout_episode(env, self.policy, max_steps=5, seed=2)
        self.assertEqual(int(traj[EPISODE_LENGTH][0]), 5)
        self.assertTrue(np.all(traj[TERMINATED] == False))  # noqa: E712

    def test_deterministic_path_uses_argmax(self):
        env = _FakeDiscreteEnv(obs_dim=4, terminate_after=2)
        traj = rollout_episode(
            env, self.policy, max_steps=4, seed=3, deterministic=True
        )
        # Same env / policy with deterministic=True should be reproducible.
        env2 = _FakeDiscreteEnv(obs_dim=4, terminate_after=2)
        traj2 = rollout_episode(
            env2, self.policy, max_steps=4, seed=3, deterministic=True
        )
        np.testing.assert_array_equal(traj[ACTIONS], traj2[ACTIONS])


class TestRolloutEpisodeContinuous(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        cfg = GymMLPConfig(obs_dim=3, action_dim=1, hidden_dim=8, discrete=False)
        self.policy = GymPolicy(cfg)

    def test_actions_have_action_dim(self):
        env = _FakeContinuousEnv(action_dim=1, terminate_after=2)
        traj = rollout_episode(env, self.policy, max_steps=4, seed=1)
        self.assertEqual(traj[ACTIONS].shape, (4, 1))
        self.assertEqual(traj[ACTIONS].dtype, np.float32)


class TestGymRolloutEngine(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        cfg = GymMLPConfig(obs_dim=4, action_dim=2, hidden_dim=8, discrete=True)
        self.policy = GymPolicy(cfg)

    def test_engine_run_with_init(self):
        engine = GymRolloutEngine(
            env_factory=lambda: _FakeDiscreteEnv(obs_dim=4, terminate_after=3),
            policy=self.policy,
            max_steps=6,
        )
        traj = engine.run({"seed": 42})
        self.assertIn(OBSERVATIONS, traj)
        engine.close()

    def test_engine_init_from_prompt(self):
        self.assertEqual(GymRolloutEngine.init_from_prompt(""), {})
        self.assertEqual(GymRolloutEngine.init_from_prompt(None), {})
        self.assertEqual(GymRolloutEngine.init_from_prompt('{"seed": 5}'), {"seed": 5})
        self.assertEqual(GymRolloutEngine.init_from_prompt({"seed": 5}), {"seed": 5})
        # Invalid JSON shouldn't crash.
        self.assertEqual(GymRolloutEngine.init_from_prompt("not json"), {})

    def test_engine_logs_unknown_init_keys_without_failing(self):
        engine = GymRolloutEngine(
            env_factory=lambda: _FakeDiscreteEnv(obs_dim=4, terminate_after=2),
            policy=self.policy,
            max_steps=4,
        )
        # Unknown keys are ignored gracefully.
        traj = engine.run({"seed": 1, "unknown_field": 99})
        self.assertEqual(int(traj[EPISODE_LENGTH][0]), 2)
        engine.close()


# ---------------------------------------------------------------------------
# register_gym_policy() -- TOML loading + registry wiring
# ---------------------------------------------------------------------------


class TestRegisterGymPolicy(unittest.TestCase):
    def setUp(self):
        # Reset both registries to keep tests independent.
        from cosmos_rl.utils import model_config, util

        model_config.clear_local_model_configs()
        util.clear_tokenizer_loaders()
        util.setup_tokenizer.cache_clear()

        self.toml_path = os.path.join(
            tempfile.mkdtemp(prefix="cosmos_rl_gym_"), "cartpole.toml"
        )
        with open(self.toml_path, "w") as f:
            f.write(
                "[model]\n"
                "obs_dim = 4\n"
                "action_dim = 2\n"
                "hidden_dim = 16\n"
                "discrete = true\n"
            )

    def tearDown(self):
        from cosmos_rl.utils import model_config, util

        model_config.clear_local_model_configs()
        util.clear_tokenizer_loaders()
        util.setup_tokenizer.cache_clear()
        try:
            os.unlink(self.toml_path)
            os.rmdir(os.path.dirname(self.toml_path))
        except OSError:
            pass

    def test_local_model_config_loader_yields_gym_mlp_config(self):
        from cosmos_rl.utils.model_config import load_model_config

        register_gym_policy()
        cfg = load_model_config(self.toml_path)
        self.assertIsInstance(cfg, GymMLPConfig)
        self.assertEqual(cfg.obs_dim, 4)
        self.assertEqual(cfg.action_dim, 2)
        self.assertEqual(cfg.hidden_dim, 16)
        self.assertTrue(cfg.discrete)

    def test_tokenizer_loader_yields_no_op_tokenizer(self):
        from cosmos_rl.utils.no_op_tokenizer import NoOpTokenizer
        from cosmos_rl.utils.util import setup_tokenizer

        register_gym_policy()
        tok = setup_tokenizer(self.toml_path)
        self.assertIsInstance(tok, NoOpTokenizer)

    def test_unrelated_path_falls_through_to_default(self):
        from cosmos_rl.utils.model_config import load_model_config

        register_gym_policy()
        # Non-toml path should not match -- AutoConfig will fail because
        # the path doesn't exist, confirming we did not short-circuit.
        with self.assertRaises(Exception):
            load_model_config("/path/that/does/not/exist")


if __name__ == "__main__":
    unittest.main()
