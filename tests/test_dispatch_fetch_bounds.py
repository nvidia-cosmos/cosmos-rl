# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Real lazy fetches must stop at quota without advancing or draining epochs."""

from types import SimpleNamespace
import asyncio

import pytest

from cosmos_rl.dispatcher.data.data_fetcher import ControllerDataFetcher


def fetcher(*, rollout_size=2, policy_size=1, quota=1, indices=range(100)):
    item = ControllerDataFetcher.__new__(ControllerDataFetcher)
    item.config = SimpleNamespace(
        train=SimpleNamespace(
            train_batch_per_replica=quota,
            epoch=3,
            local_dataset=True,
            train_policy=SimpleNamespace(data_dispatch_as_rank_in_mesh=True),
        ),
        rollout=SimpleNamespace(
            n_generation=1, multi_turn_config=SimpleNamespace(enable=False)
        ),
    )
    item.data_fetched_for_each_policy_at_step = {}
    item.fetched_data_buffer = []
    item.fetched_data_buffer_for_validation = []
    item.rollout_global_mesh_size, item.policy_global_mesh_size = (
        rollout_size,
        policy_size,
    )
    item.rollout_batch_size = item.val_batch_size = 1
    item.train_sampler = item.batch_sampler = None
    item.epoch = 1
    item.pulled = []

    class Dataset:
        def __iter__(self):
            for index in indices:
                item.pulled.append(index)
                yield (
                    [index],
                    [
                        SimpleNamespace(
                            index=index,
                            prompt="p",
                            conversation=None,
                            reference_answer=None,
                        )
                    ],
                )

    item.train_dataloader = Dataset()
    item.train_dataloader_iter = iter(item.train_dataloader)
    return item


def test_full_shared_quota_does_not_pull_or_roll_epochs():
    item = fetcher()
    values, exhausted = item.get_batched_prompt(1, rank_in_mesh=0, weight_version=1)
    assert [value.index for value in values] == [0] and not exhausted
    for _ in range(3):
        assert item.get_batched_prompt(4, rank_in_mesh=1, weight_version=1) == (
            [],
            False,
        )
    assert item.pulled == [0]
    assert not item.fetched_data_buffer and item.epoch == 1
    values, exhausted = item.get_batched_prompt(1, rank_in_mesh=1, weight_version=2)
    assert [value.index for value in values] == [1] and not exhausted


@pytest.mark.parametrize(
    "rollout_size,policy_size,rank,reachable",
    [(4, 2, 2, [0]), (2, 4, 1, [1, 3]), (3, 2, 2, [0, 1])],
)
def test_only_reachable_policy_quotas_control_fetch(
    rollout_size, policy_size, rank, reachable
):
    item = fetcher(rollout_size=rollout_size, policy_size=policy_size)
    item.data_fetched_for_each_policy_at_step[7] = {peer: 1 for peer in reachable}
    assert item.get_batched_prompt(3, rank_in_mesh=rank, weight_version=7) == (
        [],
        False,
    )
    assert item.pulled == []


def test_partial_quota_caps_requested_work_without_dropping_buffer():
    item = fetcher(rollout_size=2, policy_size=2, quota=2)
    item.data_fetched_for_each_policy_at_step[1] = {0: 1}
    values, exhausted = item.get_batched_prompt(100, rank_in_mesh=0, weight_version=1)
    assert [value.index for value in values] == [0] and not exhausted
    assert len(item.pulled) <= 2
    before = list(item.fetched_data_buffer)
    assert item.get_batched_prompt(100, rank_in_mesh=0, weight_version=1) == ([], False)
    assert item.fetched_data_buffer == before


def test_one_unproductive_rank_scan_is_bounded_and_preserves_other_ranks():
    item = fetcher(indices=range(1, 10000, 2), quota=100)
    assert item.get_batched_prompt(1, rank_in_mesh=0, weight_version=1) == ([], False)
    assert item.pulled == [1, 3]
    assert [index for index, _ in item.fetched_data_buffer] == [1, 3]
    values, exhausted = item.get_batched_prompt(2, rank_in_mesh=1, weight_version=1)
    assert [value.index for value in values] == [1, 3] and not exhausted
    assert item.pulled == [1, 3]


def test_validation_ignores_training_quota_and_uses_separate_buffer():
    item = fetcher()
    item.data_fetched_for_each_policy_at_step[0] = {0: 1}
    item.validation_get_dataloader = lambda step: item.train_dataloader_iter
    values, exhausted = item.get_batched_prompt(
        1, validation_step=0, rank_in_mesh=0, weight_version=0
    )
    assert [value.index for value in values] == [0] and not exhausted
    assert item.data_fetched_for_each_policy_at_step[0] == {0: 1}


def test_zero_request_does_not_consume_or_claim_exhaustion():
    item = fetcher()
    assert item.get_batched_prompt(0, rank_in_mesh=0, weight_version=1) == ([], False)
    assert item.pulled == [] and item.epoch == 1


@pytest.mark.parametrize("requested,on_the_fly", [(0, 0), (1, 24)])
def test_empty_or_throttled_controller_fetch_has_no_missing_retry_counter(
    requested, on_the_fly
):
    from test_zero_staleness_refill_deadlock import _config, _controller

    config = _config()
    config.train.train_policy.max_retry_for_on_policy = 3
    config.train.train_policy.max_inflight_steps = None
    controller = _controller(config, samples_on_the_fly=on_the_fly, pending_rollouts=0)
    controller.weight_version_to_prompt_num = {}
    controller.data_fetcher.get_batched_prompt.return_value = ([], False)
    assert asyncio.run(controller._get_batched_prompt_impl(requested)) == ([], False)
    assert controller.weight_version_to_prompt_num == {}
    assert controller.policy_status_manager.samples_on_the_fly == on_the_fly
