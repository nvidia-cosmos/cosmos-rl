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

"""Tests for the Gym API extension surface (MR2)."""

import os
import tempfile
import unittest

import torch

from cosmos_rl.dispatcher.data.packer.tensor_data_packer import (
    ACTIONS,
    EPISODE_LENGTH,
    OBSERVATIONS,
    REWARDS,
    TensorDataPacker,
)
from cosmos_rl.utils.network_util import write_redis_config
from cosmos_rl.utils.no_op_tokenizer import NoOpTokenizer


class TestNoOpTokenizer(unittest.TestCase):
    def test_standard_attributes_present(self):
        tok = NoOpTokenizer()
        for attr in [
            "eos_token_id",
            "pad_token_id",
            "bos_token_id",
            "unk_token_id",
            "vocab_size",
            "model_max_length",
        ]:
            self.assertTrue(hasattr(tok, attr), f"missing {attr}")

    def test_encode_uses_episode_length(self):
        tok = NoOpTokenizer()
        ids = tok.encode({"episode_length": 7})
        self.assertEqual(len(ids), 7)
        self.assertTrue(all(i == 0 for i in ids))

    def test_encode_unknown_input_returns_singleton(self):
        tok = NoOpTokenizer()
        self.assertEqual(tok.encode("hello"), [0])
        self.assertEqual(tok.encode(123), [0])

    def test_call_returns_batch_encoding_shape(self):
        tok = NoOpTokenizer()
        out = tok({"episode_length": 5})
        self.assertEqual(out.input_ids.shape, (1, 5))
        self.assertEqual(out.attention_mask.shape, (1, 5))
        self.assertEqual(out["input_ids"].shape, (1, 5))
        self.assertIn("input_ids", out.keys())

    def test_decode_returns_empty(self):
        tok = NoOpTokenizer()
        self.assertEqual(tok.decode([0, 1, 2]), "")
        self.assertEqual(tok.batch_decode([[0], [1], [2]]), ["", "", ""])

    def test_episode_length_from_tensor(self):
        # Scalar tensors should also work -- common when the rollout
        # engine emits trajectories with torch.tensor(ep_len) fields.
        tok = NoOpTokenizer()
        out = tok({"episode_length": torch.tensor(4)})
        self.assertEqual(out.input_ids.shape, (1, 4))


class TestRegisterTokenizerLoader(unittest.TestCase):
    def setUp(self):
        # Lazy-import inside the test methods themselves to avoid pulling
        # the heavy cosmos_rl.utils.util module at collection time.
        from cosmos_rl.utils import util

        self.util = util
        util.clear_tokenizer_loaders()
        # ``setup_tokenizer`` is ``functools.lru_cache``-wrapped, so
        # cached results from previous tests would otherwise leak.
        util.setup_tokenizer.cache_clear()

    def tearDown(self):
        self.util.clear_tokenizer_loaders()
        self.util.setup_tokenizer.cache_clear()

    def test_predicate_match_short_circuits_hf(self):
        sentinel = NoOpTokenizer()

        self.util.register_tokenizer_loader(
            predicate=lambda p: p.endswith(".toml"),
            loader=lambda p: sentinel,
        )

        result = self.util.setup_tokenizer("/tmp/example.toml")
        self.assertIs(result, sentinel)

    def test_predicate_miss_falls_back(self):
        # No predicate matches -- the function should attempt the HF
        # path.  Use a path that AutoTokenizer cannot resolve and
        # expect a non-NoOp failure mode (i.e. an exception from
        # AutoTokenizer / network), confirming we did not short-circuit.
        self.util.register_tokenizer_loader(
            predicate=lambda p: p.startswith("never_matches://"),
            loader=lambda p: NoOpTokenizer(),
        )
        with self.assertRaises(Exception):
            self.util.setup_tokenizer("/this/path/definitely/does/not/exist")


class TestWriteRedisConfigTlsSkip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False
        )
        self.tmp.close()
        self._saved_env = os.environ.pop("COSMOS_REDIS_NO_TLS", None)

    def tearDown(self):
        os.unlink(self.tmp.name)
        if self._saved_env is not None:
            os.environ["COSMOS_REDIS_NO_TLS"] = self._saved_env
        else:
            os.environ.pop("COSMOS_REDIS_NO_TLS", None)

    def test_default_emits_tls_port(self):
        write_redis_config(12800, "/tmp/r.log", file_path=self.tmp.name)
        with open(self.tmp.name) as f:
            content = f.read()
        self.assertIn("tls-port 0", content)

    def test_skip_via_kwarg(self):
        write_redis_config(
            12800, "/tmp/r.log", file_path=self.tmp.name, skip_tls_port=True
        )
        with open(self.tmp.name) as f:
            content = f.read()
        # Header comment is fine, but the actual directive must be gone.
        self.assertNotIn("\ntls-port", content)

    def test_skip_via_env(self):
        os.environ["COSMOS_REDIS_NO_TLS"] = "1"
        write_redis_config(12800, "/tmp/r.log", file_path=self.tmp.name)
        with open(self.tmp.name) as f:
            content = f.read()
        self.assertNotIn("\ntls-port", content)

    def test_env_falsey_keeps_default(self):
        os.environ["COSMOS_REDIS_NO_TLS"] = "0"
        write_redis_config(12800, "/tmp/r.log", file_path=self.tmp.name)
        with open(self.tmp.name) as f:
            content = f.read()
        self.assertIn("tls-port 0", content)


class TestModelConfigRegistry(unittest.TestCase):
    def setUp(self):
        from cosmos_rl.utils import model_config

        self.model_config = model_config
        model_config.clear_local_model_configs()

    def tearDown(self):
        self.model_config.clear_local_model_configs()

    def test_matching_predicate_returns_factory_result(self):
        from transformers import PretrainedConfig

        class GymCfg(PretrainedConfig):
            model_type = "gym_mlp"

        self.model_config.register_local_model_config(
            predicate=lambda p: p.endswith(".toml"),
            factory=lambda p: GymCfg(),
        )
        cfg = self.model_config.load_model_config("/tmp/cartpole.toml")
        self.assertEqual(cfg.model_type, "gym_mlp")

    def test_non_matching_falls_back_to_auto_config(self):
        # No predicate registered; expect AutoConfig to attempt and
        # raise on a missing path.
        with self.assertRaises(Exception):
            self.model_config.load_model_config(
                "/this/path/definitely/does/not/exist"
            )


class _ConcreteTensorPacker(TensorDataPacker):
    """Minimal subclass to exercise the abstract surface.  No overrides
    needed since the base class's defaults are already concrete for
    tensor trajectories."""


def _make_traj(length: int) -> dict:
    return {
        OBSERVATIONS: torch.zeros((length, 4)),
        ACTIONS: torch.zeros((length,), dtype=torch.long),
        REWARDS: torch.ones((length,)),
        EPISODE_LENGTH: torch.tensor(length),
    }


class TestTensorDataPacker(unittest.TestCase):
    def test_get_rollout_input_passthrough(self):
        p = _ConcreteTensorPacker()
        sample = {"prompt": "{\"seed\": 1}"}
        self.assertEqual(p.get_rollout_input(sample), sample)

    def test_policy_compute_max_len_uses_episode_length(self):
        p = _ConcreteTensorPacker()
        traj = [_make_traj(3), _make_traj(8), _make_traj(5)]
        self.assertEqual(p.policy_compute_max_len(traj), 8)

    def test_policy_compute_max_len_falls_back_to_obs_shape(self):
        # Drop the episode_length field so the packer must look at the
        # observation shape.
        p = _ConcreteTensorPacker()
        traj = _make_traj(7)
        del traj[EPISODE_LENGTH]
        self.assertEqual(p.policy_compute_max_len([traj]), 7)

    def test_policy_collate_fn_keeps_trajectories(self):
        p = _ConcreteTensorPacker()
        traj = [_make_traj(3), _make_traj(8)]
        out = p.policy_collate_fn(traj, computed_max_len=8)
        self.assertEqual(out["trajectories"], traj)

    def test_get_rollout_output_passthrough(self):
        p = _ConcreteTensorPacker()
        completions = [_make_traj(2)]
        out = p.get_rollout_output(completions, [], [], [])
        self.assertIs(out[0], completions)

    def test_is_trajectory_detects_dict_shape(self):
        self.assertTrue(TensorDataPacker.is_trajectory(_make_traj(4)))
        self.assertFalse(TensorDataPacker.is_trajectory({"prompt": "x"}))
        self.assertFalse(TensorDataPacker.is_trajectory("not a dict"))


# IdentityWeightMapper lives under cosmos_rl/policy/model/, whose package
# __init__.py transitively imports diffusers.  Skip cleanly when the
# optional ``diffusers`` extra is not installed -- this is purely a test-
# harness concern; the mapper itself works fine in a real cosmos-rl
# environment with the standard install.
try:
    from cosmos_rl.policy.model.identity_weight_mapper import (
        IdentityWeightMapper,
    )
    _IDENTITY_WEIGHT_MAPPER_IMPORTABLE = True
    _IDENTITY_WEIGHT_MAPPER_SKIP_REASON = ""
except ImportError as _e:  # noqa: BLE001
    IdentityWeightMapper = None  # type: ignore[assignment]
    _IDENTITY_WEIGHT_MAPPER_IMPORTABLE = False
    _IDENTITY_WEIGHT_MAPPER_SKIP_REASON = (
        f"cosmos_rl.policy.model package not importable in this env "
        f"(missing optional dep: {_e}). The mapper itself does not depend "
        f"on diffusers; this is a harness-only limitation."
    )


@unittest.skipUnless(
    _IDENTITY_WEIGHT_MAPPER_IMPORTABLE, _IDENTITY_WEIGHT_MAPPER_SKIP_REASON
)
class TestIdentityWeightMapper(unittest.TestCase):
    def test_constructable_without_hf_config(self):
        mapper = IdentityWeightMapper()
        self.assertEqual(mapper.policy_map_local_key_to_hf_key("fc.weight"), "fc.weight")
        self.assertEqual(mapper.rollout_map_local_key_to_hf_key("fc.weight"), "fc.weight")

    def test_no_split(self):
        mapper = IdentityWeightMapper()
        t = torch.zeros((4, 4))
        out = mapper.rollout_split_local_key_n_param_to_hf_key_n_param("a.b", t)
        self.assertEqual(len(out), 1)
        name, tensor = out[0]
        self.assertEqual(name, "a.b")
        self.assertIs(tensor, t)


if __name__ == "__main__":
    unittest.main()
