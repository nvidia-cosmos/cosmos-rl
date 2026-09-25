# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Actual worker startup and queue-dispatch count regressions."""

from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from cosmos_rl.policy.worker.base import PolicyWorkerBase
from cosmos_rl.policy.worker.rl_worker import RLPolicyWorker
from cosmos_rl.utils.parallelism import ParallelDims


def configured_worker(*, count=3, shards=1, replicas=2, mini_batch=1):
    return SimpleNamespace(
        config=SimpleNamespace(
            train=SimpleNamespace(
                train_batch_per_replica=count,
                train_policy=SimpleNamespace(
                    type="grpo", mini_batch=mini_batch, trainer_type=None
                ),
            ),
            policy=SimpleNamespace(parallelism=SimpleNamespace(dp_shard_size=shards)),
        ),
        parallel_dims=ParallelDims(
            dp_shard=shards,
            dp_replicate=replicas,
            cp=1,
            tp=1,
            pp=1,
            world_size=shards * replicas,
        ),
    )


@pytest.mark.parametrize(
    "count,shards,replicas,mini_batch", [(3, 1, 2, 1), (4, 1, 3, 2), (6, 2, 2, 1)]
)
def test_startup_rejects_uneven_full_dp_collection(count, shards, replicas, mini_batch):
    worker = configured_worker(
        count=count, shards=shards, replicas=replicas, mini_batch=mini_batch
    )
    with pytest.raises((ValueError, AssertionError), match="divisible"):
        PolicyWorkerBase.check_config(worker)


@pytest.mark.parametrize("uncentralized", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_fixed_dispatch_rejects_before_dequeue_or_collective(uncentralized, rank):
    worker = SimpleNamespace(
        trainer=SimpleNamespace(),
        replica_batch_for_this_step=3,
        dp_world_size=2,
        world_size=2,
        global_rank=rank,
        parallel_dims=SimpleNamespace(get_rank_in_dim=lambda dim, r: r),
        config=SimpleNamespace(
            train=SimpleNamespace(
                local_dataset=False,
                train_policy=SimpleNamespace(uncentralized_training=uncentralized),
            )
        ),
        data_queue=Queue(),
        prepare_teacher_uuids_for_prefetch=Mock(return_value=0),
    )
    for index in range(3):
        worker.data_queue.put(
            SimpleNamespace(prompt_idx=index, teacher_result_uuid=None)
        )
    with patch("cosmos_rl.policy.worker.rl_worker.dist.scatter_object_list") as scatter:
        with pytest.raises(ValueError, match="refusing to silently round"):
            RLPolicyWorker.dispatch_rollouts(worker)
        scatter.assert_not_called()
    assert worker.data_queue.qsize() == 3
    worker.prepare_teacher_uuids_for_prefetch.assert_not_called()


@pytest.mark.parametrize(
    "count,shards,replicas,mini_batch", [(4, 1, 2, 1), (12, 1, 3, 2), (8, 2, 2, 1)]
)
def test_valid_full_dp_collection_remains_accepted(count, shards, replicas, mini_batch):
    PolicyWorkerBase.check_config(
        configured_worker(
            count=count, shards=shards, replicas=replicas, mini_batch=mini_batch
        )
    )
