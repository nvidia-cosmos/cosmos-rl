# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Application integration fixture for the RL-Gym companion launcher.

Use as its role entrypoint with disaggregated GRPO, zero allowed staleness,
checkpointing disabled, a finite training horizon, and train.train_policy.algo
set to "admission_canary" (the companion emits one completion per group).
Requires the RL-Gym
companion's simple trainer/rollout modules. The test-only producer wrapper marks
quality exclusions before reward processing and replays every report once.
It allocates identities at generation return; production producers must allocate
before generation to cover generation failures as well.
"""

import itertools
import os
import threading

from cosmos_rl.dispatcher.algo.base import register_rule_based_algo
from cosmos_rl.dispatcher.algo.grpo import GRPO
from cosmos_rl.dispatcher.data.admission import (
    CompletionFailure,
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


class CanaryScalarReward(GRPO):
    """One-member algorithm for the companion's scalar-reward fixture only."""

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

    run_web_panel.main(completion_admission=True)
elif role == "policy":
    from cosmos_rl.policy.policy_entry import policy_entry

    packer = TransportSimpleRLDataPacker()
    policy_entry(
        data_packer=packer, val_data_packer=packer, reward_fns=[rl_gym_reward_fn]
    )
elif role == "rollout":
    from cosmos_rl.dispatcher.api.client import APIClient
    from cosmos_rl.dispatcher.algo.base import REGISTERED_ALGOs
    from cosmos_rl.dispatcher.data.schema import RLPayload, Rollout
    from cosmos_rl.reward.admission import COMPLETION_ADMISSION_METRIC_PREFIX
    from cosmos_rl.reward.admission import resolve_completion_admission
    from cosmos_rl.rollout.rollout_entry import run_rollout
    from cosmos_rl.rollout.worker.rollout_control import (
        DisaggregatedRolloutControlWorker,
    )

    original_post = APIClient.post_rollout_completion
    sequences = itertools.count()
    report_lock = threading.Lock()
    pending_failures = []
    original_filter = (
        DisaggregatedRolloutControlWorker._filter_valid_rollout_results_and_report
    )

    def mark_before_rewards(worker, results, payloads):
        if worker.config.train.train_policy.algo != "admission_canary":
            raise ValueError(
                "This fixture requires train.train_policy.algo='admission_canary'"
            )
        with report_lock:
            for result in results:
                ids = [next(sequences) for _ in result.completions]
                result.completion_trainable = [sequence >= 4 for sequence in ids]
                result.completion_drop_reasons = [
                    None if keep else "canary_quality"
                    for keep in result.completion_trainable
                ]
                result.extra_info = dict(result.extra_info or {})
                result.extra_info["canary_sequence"] = ids
                admission = resolve_completion_admission(
                    RLPayload(
                        completions=result.completions,
                        completion_trainable=result.completion_trainable,
                    ),
                    REGISTERED_ALGOs[
                        worker.config.train.train_policy.algo
                    ].minimum_trainable_completions,
                )
                rejected = (
                    range(len(ids))
                    if admission.group_excluded
                    else admission.excluded_indices
                )
                pending_failures.extend(
                    CompletionFailure(
                        identity=CompletionIdentity(
                            sequence=ids[index],
                            weight_version=worker.current_weight_version,
                        ),
                        reason="insufficient_group"
                        if admission.group_excluded
                        else "canary_quality",
                        payload=Rollout(
                            completion=result.completions[index],
                            weight_version=worker.current_weight_version,
                        ),
                    )
                    for index in rejected
                )
        return original_filter(worker, results, payloads)

    DisaggregatedRolloutControlWorker._filter_valid_rollout_results_and_report = (
        mark_before_rewards
    )

    def identified_post(client, report):
        if report.is_end:
            return original_post(client, report)
        with report_lock:
            if report.completion_identities is None:
                report.src_global_rank = int(os.environ.get("RANK", "0"))
                report.completion_identities = [
                    CompletionIdentity(
                        sequence=sequence, weight_version=payload.weight_version
                    )
                    for payload in report.payloads
                    for sequence in payload.extra_info["canary_sequence"]
                ]
                report.completion_failures = list(pending_failures)
                pending_failures.clear()
                report.metrics = {
                    key: value
                    for key, value in report.metrics.items()
                    if not key.startswith(COMPLETION_ADMISSION_METRIC_PREFIX)
                    and key
                    not in {
                        "discarded_samples",
                        "discard_report_id",
                        "discarded_weight_version",
                    }
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
