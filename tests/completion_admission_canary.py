# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live companion fixture: generation failure, pre-reward rejection and replay.

Use as the cosmos-rl role entrypoint with the RL-Gym companion on PYTHONPATH.
Set train.train_policy.algo="admission_canary", n_generation=1, finite steps,
checkpointing disabled. Disaggregated runs additionally set
rollout.completion_admission=true. Colocated runs use the same quality masks
with ordinary local queue accounting. No identity/accounting implementation is
injected by this fixture: it exercises the standard producer implementation.
"""

import os

from cosmos_rl.dispatcher.algo.base import register_rule_based_algo
from cosmos_rl.dispatcher.algo.grpo import GRPO
from cosmos_rl.dispatcher.api.client import APIClient
from cosmos_rl.utils.model_config import register_local_model_config
from cosmos_rl.utils.no_op_tokenizer import NoOpTokenizer
from cosmos_rl.utils.util import register_tokenizer_loader
from rl_gym.simple_weight_mapper import SimpleRLConfig
from rl_gym.policy.trainer.simple_trainer import SimpleRLTrainer  # noqa: F401
from rl_gym.rollouts.modular_rollout_worker import ModularRolloutWorker
from rl_gym.rollouts.simple_rollout_worker import SimpleRolloutWorker
from rl_gym.reward_function import rl_gym_reward_fn
from rl_gym.simple_data_packer import SimpleRLDataPacker, TransportSimpleRLDataPacker


class CanaryScalarReward(GRPO):
    minimum_trainable_completions = 1

    def compute_advantage(self, rewards):
        return list(rewards)


register_rule_based_algo("admission_canary", CanaryScalarReward)
register_tokenizer_loader(
    predicate=lambda path: str(path).endswith(".toml"),
    loader=lambda path: NoOpTokenizer(),
)
register_local_model_config(
    predicate=lambda path: str(path).endswith(".toml"),
    factory=lambda path: SimpleRLConfig(),
)


def inject_producer_faults(cls):
    original = cls.rollout_generation

    def generate(self, **kwargs):
        if kwargs.get("is_validation"):
            return original(self, **kwargs)
        if self.config.mode == "disaggregated":
            assert all(
                p.completion_sequences is not None for p in kwargs["payloads"]
            ), "IDs must exist before generation"
        calls = getattr(self, "_canary_generation_calls", 0)
        self._canary_generation_calls = calls + 1
        if calls == 0:
            print("[ADMISSION-CANARY] injected generation failure", flush=True)
            return []
        results = original(self, **kwargs)
        for result in results:
            result.completion_trainable = [calls > 1] * len(result.completions)
            result.completion_drop_reasons = [
                None if calls > 1 else "canary_quality"
            ] * len(result.completions)
        if calls == 1:
            print(
                "[ADMISSION-CANARY] injected pre-reward quality rejection", flush=True
            )
        return results

    cls.rollout_generation = generate


inject_producer_faults(ModularRolloutWorker)
inject_producer_faults(SimpleRolloutWorker)

original_post = APIClient.post_rollout_completion


def replay_report(client, report):
    assert original_post(client, report), "first report failed"
    if not report.is_end and report.completion_identities is not None:
        assert original_post(client, report), "duplicate report failed"
        print(
            f"[ADMISSION-CANARY] replayed accepted={len(report.completion_identities)} rejected={len(report.completion_failures)}",
            flush=True,
        )
    return True


APIClient.post_rollout_completion = replay_report

role = os.environ["COSMOS_ROLE"].lower()
if role == "controller":
    from cosmos_rl.dispatcher import run_web_panel

    @run_web_panel.app.middleware("http")
    async def observe(request, call_next):
        response = await call_next(request)
        status = run_web_panel.controller.policy_status_manager
        rejected = status.filter_records.get("application_rejected", 0)
        if rejected:
            print(
                f"[ADMISSION-CANARY] step={status.current_step} rejected={rejected} in_flight={status.samples_on_the_fly}",
                flush=True,
            )
            assert status.samples_on_the_fly >= 0
        return response

    run_web_panel.main()
elif role == "policy":
    from cosmos_rl.policy.policy_entry import policy_entry

    # The live runner selects this fixture's mode explicitly for packer choice.
    packer = (
        SimpleRLDataPacker()
        if os.environ.get("ADMISSION_CANARY_MODE") == "colocated"
        else TransportSimpleRLDataPacker()
    )
    policy_entry(
        data_packer=packer, val_data_packer=packer, reward_fns=[rl_gym_reward_fn]
    )
elif role == "rollout":
    from cosmos_rl.rollout.rollout_entry import run_rollout

    packer = SimpleRLDataPacker()
    run_rollout(
        data_packer=packer, val_data_packer=packer, reward_fns=[rl_gym_reward_fn]
    )
else:
    raise ValueError(f"Unknown role {role}")
