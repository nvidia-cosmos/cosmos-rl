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
