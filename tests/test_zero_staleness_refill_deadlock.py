# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Strict zero-staleness training must survive a terminally discarded group.

The reported deadlock: three version-N prompts fill the version's prompt
quota, one whole 8-rollout group is discarded, and the replacement prompt is
tagged N+1.  A worker at version N with ``allowed_outdated_steps=0`` rejects
it forever, while the policy cannot reach N+1 because it is still short the
rollouts that replacement was meant to supply -- a circular wait that only
ends at Slurm wall time.

The scenario below is the reported configuration: 3 policy replicas,
``train_batch_per_replica=8``, ``n_generation=8`` (24 rollouts from 3
prompts), ``on_policy=true``, ``allowed_outdated_steps=0``,
``max_inflight_steps=1``.
"""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

from cosmos_rl.dispatcher.controller import Controller
from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.dispatcher.status import PolicyStatusManager

N_GENERATION = 8
POLICY_REPLICAS = 3
TRAIN_BATCH_PER_REPLICA = 8
ROLLOUTS_PER_GLOBAL_BATCH = TRAIN_BATCH_PER_REPLICA * POLICY_REPLICAS  # 24
PROMPTS_PER_GLOBAL_BATCH = ROLLOUTS_PER_GLOBAL_BATCH // N_GENERATION  # 3


def _config(*, on_policy=True, allowed_outdated_steps=0, mode="disaggregated"):
    return SimpleNamespace(
        mode=mode,
        validation=SimpleNamespace(enable=False),
        rollout=SimpleNamespace(n_generation=N_GENERATION),
        train=SimpleNamespace(
            train_batch_per_replica=TRAIN_BATCH_PER_REPLICA,
            train_policy=SimpleNamespace(
                type="grpo",
                variant="grpo",
                on_policy=on_policy,
                allowed_outdated_steps=allowed_outdated_steps,
                outdated_rollout_fetch_batch_size=0,
                max_inflight_steps=1,
                max_retry_for_on_policy=0,
            ),
        ),
    )


def _controller(config, *, current_step=8, samples_on_the_fly, pending_rollouts):
    """Controller with the quota state a fully-fetched version-N batch leaves."""
    policy_status = MagicMock()
    policy_status.__len__.return_value = POLICY_REPLICAS
    policy_status.current_step = current_step
    policy_status.total_pending_rollouts.return_value = pending_rollouts
    policy_status.samples_on_the_fly = samples_on_the_fly
    policy_status.replica_scaling_log = []
    policy_status.training_finished.return_value = False

    controller = object.__new__(Controller)
    controller.config = config
    controller.policy_status_manager = policy_status
    controller.rollout_status_manager = SimpleNamespace(replica_scaling_log=[])
    controller.data_fetcher = SimpleNamespace(
        get_batched_prompt=MagicMock(
            return_value=([RLPayload(prompt_idx=99)], False),
        )
    )
    # Version N's prompt quota is already full: all three prompts were fetched
    # before any result came back.
    controller.weight_version_to_prompt_num = {current_step: PROMPTS_PER_GLOBAL_BATCH}
    controller.weight_version_to_replacement_prompt_num = {}
    controller._soft_throttle_engaged_since = None
    controller._soft_throttle_last_log_ts = 0.0
    return controller


def test_discarded_group_replacement_keeps_the_current_weight_version():
    """The reported scenario end to end: replacement must be version N."""
    config = _config()
    # One group of 8 is terminally discarded: 24 on the fly drops to 16.
    controller = _controller(
        config,
        current_step=8,
        samples_on_the_fly=ROLLOUTS_PER_GLOBAL_BATCH - N_GENERATION,
        pending_rollouts=ROLLOUTS_PER_GLOBAL_BATCH - N_GENERATION,
    )

    released = controller.register_discarded_samples_for_refill(8, N_GENERATION)
    assert released == 1, "one terminal group must release exactly one prompt slot"

    payloads, is_end = asyncio.run(controller._get_batched_prompt_impl(1))

    assert not is_end
    assert [p.weight_version for p in payloads] == [8], (
        "the replacement prompt must stay on version 8; tagging it 9 is the "
        "deadlock a strict worker cannot escape"
    )
    # And the refill credit is spent, not reusable.
    assert controller.weight_version_to_replacement_prompt_num == {8: 0}
    assert controller.weight_version_to_replacement_prompt_issued == {8: 1}


def test_one_group_releases_exactly_one_prompt_slot():
    """n_generation discarded completions == one prompt, not n_generation."""
    controller = _controller(_config(), samples_on_the_fly=16, pending_rollouts=16)
    assert controller.register_discarded_samples_for_refill(8, N_GENERATION) == 1
    assert controller.weight_version_to_replacement_prompt_num == {8: 1}


def test_repeated_discard_report_does_not_release_a_second_slot():
    """Settlement dedupes reports; refill must be idempotent on its own too."""
    controller = _controller(_config(), samples_on_the_fly=16, pending_rollouts=16)
    assert controller.register_discarded_samples_for_refill(8, N_GENERATION) == 1
    assert controller.register_discarded_samples_for_refill(8, N_GENERATION) == 1
    # Two distinct full-group discards, two slots -- but no slot per completion.
    assert controller.weight_version_to_replacement_prompt_num == {8: 2}


def test_strict_staleness_without_on_policy_flag_also_refills():
    """``allowed_outdated_steps=0`` is strict whether or not on_policy is set.

    The prompt-version quota is applied to every non-DAPO disaggregated run,
    so a run that leaves ``on_policy`` at its default still leaks a version-N
    slot on discard and deadlocks the same way.
    """
    config = _config(on_policy=False, allowed_outdated_steps=0)
    controller = _controller(
        config,
        samples_on_the_fly=ROLLOUTS_PER_GLOBAL_BATCH - N_GENERATION,
        pending_rollouts=ROLLOUTS_PER_GLOBAL_BATCH - N_GENERATION,
    )

    assert controller.register_discarded_samples_for_refill(8, N_GENERATION) == 1

    payloads, _ = asyncio.run(controller._get_batched_prompt_impl(1))
    assert [p.weight_version for p in payloads] == [8]


def test_lenient_staleness_does_not_refill():
    """With slack to spare, a next-version prompt is admissible; don't refill."""
    config = _config(on_policy=False, allowed_outdated_steps=1)
    controller = _controller(config, samples_on_the_fly=16, pending_rollouts=16)
    assert controller.register_discarded_samples_for_refill(8, N_GENERATION) == 0


def test_settlement_through_the_manager_also_reopens_the_slot():
    """A backend that settles directly must not lose the prompt slot.

    Backends report terminal drops through
    ``PolicyStatusManager.settle_discarded_samples``; if only the HTTP route
    reopens prompt capacity, such a caller settles the samples and deadlocks
    anyway.
    """
    controller = _controller(
        _config(),
        samples_on_the_fly=ROLLOUTS_PER_GLOBAL_BATCH,
        pending_rollouts=ROLLOUTS_PER_GLOBAL_BATCH,
    )
    manager = PolicyStatusManager()
    manager.samples_on_the_fly = ROLLOUTS_PER_GLOBAL_BATCH
    manager.set_discard_refill_hook(controller.register_discarded_samples_for_refill)

    settled = manager.settle_discarded_samples(
        source_replica="rollout-0",
        report_id="quality-gate-1",
        count=N_GENERATION,
        weight_version=8,
    )

    assert settled == N_GENERATION
    assert manager.samples_on_the_fly == ROLLOUTS_PER_GLOBAL_BATCH - N_GENERATION
    assert controller.weight_version_to_replacement_prompt_num == {8: 1}

    # Replaying the same report settles nothing and reopens nothing.
    assert (
        manager.settle_discarded_samples(
            source_replica="rollout-0",
            report_id="quality-gate-1",
            count=N_GENERATION,
            weight_version=8,
        )
        == 0
    )
    assert controller.weight_version_to_replacement_prompt_num == {8: 1}


def test_same_report_through_both_routes_releases_one_slot():
    """The settlement hook and the HTTP refill call must not stack.

    ``put_rollout_group`` settles (which fires the hook) and then calls the
    refill primitive itself; deduping by report id is what keeps one terminal
    group worth exactly one prompt slot.
    """
    controller = _controller(
        _config(),
        samples_on_the_fly=ROLLOUTS_PER_GLOBAL_BATCH,
        pending_rollouts=ROLLOUTS_PER_GLOBAL_BATCH,
    )
    manager = PolicyStatusManager()
    manager.samples_on_the_fly = ROLLOUTS_PER_GLOBAL_BATCH
    manager.set_discard_refill_hook(controller.register_discarded_samples_for_refill)

    settled = manager.settle_discarded_samples(
        source_replica="rollout-0",
        report_id="group-drop-1",
        count=N_GENERATION,
        weight_version=8,
    )
    controller.register_discarded_samples_for_refill(8, settled, "group-drop-1")

    assert controller.weight_version_to_replacement_prompt_num == {8: 1}


def test_backend_counter_settlement_can_reopen_the_slot():
    """The low-level counter seam a backend filter uses must reopen the slot.

    A quality gate that drops rollouts settles the reserved samples through
    ``_settle_samples_on_the_fly``; passing the version those samples were
    generated for is what keeps the strict step from stalling one group short.
    """
    controller = _controller(
        _config(),
        samples_on_the_fly=ROLLOUTS_PER_GLOBAL_BATCH,
        pending_rollouts=ROLLOUTS_PER_GLOBAL_BATCH,
    )
    manager = PolicyStatusManager()
    manager.samples_on_the_fly = ROLLOUTS_PER_GLOBAL_BATCH
    manager.set_discard_refill_hook(controller.register_discarded_samples_for_refill)

    manager._settle_samples_on_the_fly(
        N_GENERATION,
        "quality_ingest",
        weight_version=8,
        report_id="rot-drop-1",
    )

    assert manager.samples_on_the_fly == ROLLOUTS_PER_GLOBAL_BATCH - N_GENERATION
    assert controller.weight_version_to_replacement_prompt_num == {8: 1}
    # Replaying the same drop report reopens nothing further.
    manager._settle_samples_on_the_fly(
        N_GENERATION,
        "quality_ingest",
        weight_version=8,
        report_id="rot-drop-1",
    )
    assert controller.weight_version_to_replacement_prompt_num == {8: 1}


def test_versionless_backend_settlement_warns_once(caplog):
    """A backend that settles without a version is told why it may stall."""
    controller = _controller(
        _config(),
        samples_on_the_fly=ROLLOUTS_PER_GLOBAL_BATCH,
        pending_rollouts=ROLLOUTS_PER_GLOBAL_BATCH,
    )
    manager = PolicyStatusManager()
    manager.samples_on_the_fly = ROLLOUTS_PER_GLOBAL_BATCH
    manager.set_discard_refill_hook(controller.register_discarded_samples_for_refill)

    with caplog.at_level(logging.WARNING):
        manager._settle_samples_on_the_fly(N_GENERATION, "quality_ingest")
        manager._settle_samples_on_the_fly(N_GENERATION, "quality_ingest")

    warnings = [
        r for r in caplog.records if "without a weight version" in r.getMessage()
    ]
    assert len(warnings) == 1
    assert "quality_ingest" in warnings[0].getMessage()
    assert controller.weight_version_to_replacement_prompt_num == {}


def test_staleness_filtering_never_reopens_a_slot():
    """Outdated-rollout filtering drops samples whose slot must stay spent."""
    controller = _controller(
        _config(),
        samples_on_the_fly=ROLLOUTS_PER_GLOBAL_BATCH,
        pending_rollouts=ROLLOUTS_PER_GLOBAL_BATCH,
    )
    manager = PolicyStatusManager()
    manager.samples_on_the_fly = ROLLOUTS_PER_GLOBAL_BATCH
    manager.set_discard_refill_hook(controller.register_discarded_samples_for_refill)

    manager._settle_samples_on_the_fly(N_GENERATION, "filter_outdated")

    assert controller.weight_version_to_replacement_prompt_num == {}


def test_credits_for_completed_versions_are_pruned():
    """A credit for a finished batch must not linger for a later prompt."""
    controller = _controller(
        _config(), current_step=9, samples_on_the_fly=16, pending_rollouts=16
    )
    controller.weight_version_to_replacement_prompt_num = {7: 1, 8: 1}
    controller.weight_version_to_prompt_num = {9: PROMPTS_PER_GLOBAL_BATCH}
    controller.weight_version_to_replacement_prompt_issued = {7: 1}
    controller.weight_version_to_discarded_sample_num = {7: 8}

    asyncio.run(controller._get_batched_prompt_impl(1))

    assert 7 not in controller.weight_version_to_replacement_prompt_num
    assert 7 not in controller.weight_version_to_replacement_prompt_issued
    assert 7 not in controller.weight_version_to_discarded_sample_num
