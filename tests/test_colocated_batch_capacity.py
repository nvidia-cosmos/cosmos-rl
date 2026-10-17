# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Runs unchanged against the old revision as the uneven-capacity negative control."""

from queue import Queue
from types import SimpleNamespace as NS
from unittest.mock import patch

from cosmos_rl.colocated.controller import ColocatedController


def test_capacity_is_shared_even_when_local_queue_is_full():
    controller = object.__new__(ColocatedController)
    controller.config = NS(train=NS(train_policy=NS(uncentralized_training=True)))
    controller.policy = NS(world_size=2, data_queue=Queue())
    for _ in range(8):
        controller.policy.data_queue.put(object())
    with patch(
        "cosmos_rl.colocated.controller.dist_util.all_gather_object_cpu",
        return_value=[8, 4],
    ):
        assert controller.pending_policy_samples_all_replicas() == 8
