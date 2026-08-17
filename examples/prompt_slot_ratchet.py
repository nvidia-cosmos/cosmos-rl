# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Reproduce prompt-slot ratcheting with Cosmos-RL alone.

This drives the real controller dispatch and settlement methods over a tiny
in-memory prompt source. The "release disabled" case models the pre-fix
behavior; the normal case uses the production slot release.
"""

import asyncio
import logging
from types import SimpleNamespace

from cosmos_rl.dispatcher.controller import Controller
from cosmos_rl.dispatcher.data.schema import Rollout
from cosmos_rl.dispatcher.status import PolicyStatusManager

ALLOWED_OUTDATED_STEPS = 2
N_GENERATION = 2
PROMPTS_PER_FETCH = 2
TRAIN_BATCH_SIZE = 4
DEATH_INTERVAL = 3
MAX_ROUNDS = 100


class PromptSource:
    def __init__(self):
        self.next_prompt_idx = 0

    def get_batched_prompt(
        self,
        count,
        validation_step,
        rank_in_mesh,
        weight_version=None,
    ):
        payloads = [
            SimpleNamespace(
                prompt_idx=self.next_prompt_idx + offset,
                extra_info=None,
            )
            for offset in range(count)
        ]
        self.next_prompt_idx += count
        return payloads, False


def build_controller():
    config = SimpleNamespace(
        mode="disaggregated",
        train=SimpleNamespace(
            train_batch_per_replica=TRAIN_BATCH_SIZE,
            sync_weight_interval=1,
            train_policy=SimpleNamespace(
                type="grpo",
                variant="grpo",
                allowed_outdated_steps=ALLOWED_OUTDATED_STEPS,
                outdated_rollout_fetch_batch_size=0,
                max_inflight_steps=None,
                max_retry_for_on_policy=0,
                data_dispatch_as_rank_in_mesh=False,
            ),
        ),
        rollout=SimpleNamespace(n_generation=N_GENERATION),
        validation=SimpleNamespace(enable=False),
    )
    status = PolicyStatusManager()
    status.config = config
    status.remain_samples_num = 1_000_000
    status.policy_replicas = {
        "policy-0": SimpleNamespace(name="policy-0", all_atoms_arrived=True)
    }
    status._publish_payload_transport_cleanup = lambda *_args: None

    controller = object.__new__(Controller)
    controller.config = config
    controller.policy_status_manager = status
    controller.rollout_status_manager = SimpleNamespace(replica_scaling_log=[])
    controller.data_fetcher = PromptSource()
    controller.weight_version_to_prompt_num = status.weight_version_to_prompt_num
    controller._soft_throttle_engaged_since = None
    controller._soft_throttle_last_log_ts = 0.0
    return controller, status


def stuck_slots(status):
    held = sum(
        count
        for version, count in status.weight_version_to_prompt_num.items()
        if version >= status.current_step
    )
    live = status.samples_on_the_fly // N_GENERATION
    return held - live


def run(*, release_slots):
    controller, status = build_controller()
    if not release_slots:
        status.resolve_prompt_dispatches = lambda *args, **kwargs: 0

    dispatched = 0
    dead = 0
    max_stuck_slots = 0
    max_version_delta = 0

    for round_index in range(MAX_ROUNDS):
        payloads, _ = asyncio.run(
            controller._get_batched_prompt_impl(PROMPTS_PER_FETCH)
        )
        if not payloads:
            continue

        version_delta = payloads[0].weight_version - status.current_step
        max_version_delta = max(max_version_delta, version_delta)
        if version_delta > ALLOWED_OUTDATED_STEPS:
            return SimpleNamespace(
                outcome="WEDGED",
                rounds=round_index + 1,
                steps=status.current_step,
                dead=dead,
                max_stuck_slots=max_stuck_slots,
                max_version_delta=max_version_delta,
            )

        prompt_groups = []
        for payload in payloads:
            dispatched += 1
            if dispatched % DEATH_INTERVAL == 0:
                dead += 1
                status.settle_discarded_samples(
                    "rollout-0",
                    f"discard-{dispatched}",
                    N_GENERATION,
                    prompt_dispatch_ids=[payload.prompt_dispatch_id],
                )
                continue
            prompt_groups.append(
                [
                    Rollout(
                        prompt_idx=payload.prompt_idx,
                        prompt_dispatch_id=payload.prompt_dispatch_id,
                        completion=f"completion-{payload.prompt_idx}-{generation}",
                        weight_version=status.current_step,
                    )
                    for generation in range(N_GENERATION)
                ]
            )

        rollouts = [rollout for group in prompt_groups for rollout in group]
        accepted = status.filter_outdated_rollouts(
            rollouts,
            prompt_groups=prompt_groups,
        )
        for rollout in accepted:
            status.rollout_buffer.put(rollout)

        while status.rollout_buffer.qsize() >= TRAIN_BATCH_SIZE:
            for _ in range(TRAIN_BATCH_SIZE):
                status.rollout_buffer.get()
            status.samples_on_the_fly -= TRAIN_BATCH_SIZE
            status.current_step += 1
            status.prune_prompt_dispatches()

        max_stuck_slots = max(max_stuck_slots, stuck_slots(status))

    return SimpleNamespace(
        outcome="HEALTHY",
        rounds=MAX_ROUNDS,
        steps=status.current_step,
        dead=dead,
        max_stuck_slots=max_stuck_slots,
        max_version_delta=max_version_delta,
    )


def main():
    logging.disable(logging.CRITICAL)
    before = run(release_slots=False)
    after = run(release_slots=True)

    print("scenario          outcome  steps  dead  stuck slots  max version delta")
    print("-----------------------------------------------------------------------")
    for label, result in [
        ("release disabled", before),
        ("release enabled", after),
    ]:
        print(
            f"{label:<17} {result.outcome:<8} {result.steps:>5} "
            f"{result.dead:>5} {result.max_stuck_slots:>12} "
            f"{result.max_version_delta:>18}"
        )

    assert before.outcome == "WEDGED"
    assert after.outcome == "HEALTHY"
    assert after.max_stuck_slots == 0
    print("\nReproduction and fix assertions passed.")


if __name__ == "__main__":
    main()
