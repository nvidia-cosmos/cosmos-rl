# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Application integration fixture for the RL-Gym companion launcher.

Use as its role entrypoint with disaggregated GRPO, zero allowed staleness,
checkpointing disabled, and a finite training horizon. Requires the RL-Gym
companion's simple trainer/rollout modules. The test-only wire wrapper assigns
stable identities at first send and replays every report once. Production
producers must allocate identities before generation, as documented.
"""

import itertools
import os
import threading

from cosmos_rl.dispatcher.data.admission import (
    CompletionDisposition,
    CompletionIdentity,
)
from cosmos_rl.utils.model_config import register_local_model_config
from cosmos_rl.utils.no_op_tokenizer import NoOpTokenizer
from cosmos_rl.utils.util import register_tokenizer_loader
from rl_gym.simple_weight_mapper import SimpleRLConfig
from rl_gym.policy.trainer.simple_trainer import SimpleRLTrainer  # noqa: F401
from rl_gym.rollouts.modular_rollout_worker import ModularRolloutWorker  # noqa: F401
from rl_gym.reward_function import rl_gym_reward_fn
from rl_gym.simple_data_packer import SimpleRLDataPacker, TransportSimpleRLDataPacker


class RejectInitialCompletions:
    def __init__(self):
        self.rejected = 0
        self.accepted = 0

    def admit(self, context, rollout):
        if self.rejected < 4:
            self.rejected += 1
            print(
                f"[ADMISSION-CANARY] reject sequence={context.identity.sequence} version={context.identity.weight_version} count={self.rejected}",
                flush=True,
            )
            return CompletionDisposition(outcome="rejected", reason="canary_quality")
        self.accepted += 1
        return CompletionDisposition(outcome="accepted")


register_tokenizer_loader(
    predicate=lambda path: str(path).endswith(".toml"),
    loader=lambda path: NoOpTokenizer(),
)
register_local_model_config(
    predicate=lambda path: str(path).endswith(".toml"),
    factory=lambda path: SimpleRLConfig(),
)

role = os.environ["COSMOS_ROLE"].lower()
if role == "controller":
    from cosmos_rl.dispatcher import run_web_panel

    adapter = RejectInitialCompletions()

    @run_web_panel.app.middleware("http")
    async def observe(request, call_next):
        response = await call_next(request)
        status = run_web_panel.controller.policy_status_manager
        if adapter.rejected:
            print(
                f"[ADMISSION-CANARY] step={status.current_step} rejected={adapter.rejected} accepted={adapter.accepted} in_flight={status.samples_on_the_fly}",
                flush=True,
            )
            assert status.samples_on_the_fly >= 0
        return response

    run_web_panel.main(completion_admission=adapter)
elif role == "policy":
    from cosmos_rl.policy.policy_entry import policy_entry

    packer = TransportSimpleRLDataPacker()
    policy_entry(
        data_packer=packer, val_data_packer=packer, reward_fns=[rl_gym_reward_fn]
    )
elif role == "rollout":
    from cosmos_rl.dispatcher.api.client import APIClient
    from cosmos_rl.reward.admission import COMPLETION_ADMISSION_METRIC_PREFIX
    from cosmos_rl.rollout.rollout_entry import run_rollout

    original_post = APIClient.post_rollout_completion
    sequences = itertools.count()
    report_lock = threading.Lock()

    def identified_post(client, report):
        if report.is_end:
            return original_post(client, report)
        with report_lock:
            if report.completion_identities is None:
                # This run injects controller rejection, not generation failure.
                assert not report.metrics.get("discarded_samples", 0)
                report.src_global_rank = int(os.environ.get("RANK", "0"))
                report.completion_identities = [
                    CompletionIdentity(
                        sequence=next(sequences), weight_version=payload.weight_version
                    )
                    for payload in report.payloads
                    for _ in payload.completions
                ]
                report.metrics = {
                    key: value
                    for key, value in report.metrics.items()
                    if not key.startswith(COMPLETION_ADMISSION_METRIC_PREFIX)
                }
        assert original_post(client, report), "first report failed"
        assert original_post(client, report), "duplicate report failed"
        print(
            f"[ADMISSION-CANARY] replayed {len(report.completion_identities)} identities",
            flush=True,
        )
        return True

    APIClient.post_rollout_completion = identified_post
    packer = SimpleRLDataPacker()
    run_rollout(
        data_packer=packer, val_data_packer=packer, reward_fns=[rl_gym_reward_fn]
    )
else:
    raise ValueError(f"Unknown role {role}")
