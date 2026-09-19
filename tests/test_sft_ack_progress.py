# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""SFT progress must not depend on interleaved train/validation status flags."""

from types import SimpleNamespace
from unittest.mock import Mock

from cosmos_rl.dispatcher.status import PolicyStatus, PolicyStatusManager


def manager():
    instance = object.__new__(PolicyStatusManager)
    instance.current_step = 2
    instance.total_steps = 10
    instance.config = SimpleNamespace(
        validation=SimpleNamespace(enable=True, freq=1),
        train=SimpleNamespace(train_batch_per_replica=4),
    )
    instance.data_fetcher = SimpleNamespace(validation_activate_dataloader=Mock())
    statuses = {"a": PolicyStatus.RUNNING, "b": PolicyStatus.RUNNING}
    instance.set_status = lambda name, value: statuses.__setitem__(name, value)
    instance.any_with_status = lambda values: any(
        v in values for v in statuses.values()
    )
    instance.all_with_status = lambda values: all(
        v in values for v in statuses.values()
    )
    instance.all_reduced = lambda: instance.all_with_status([PolicyStatus.REDUCED])
    instance.get_all_atoms_arrived_replicas = lambda: ["a", "b"]
    instance.remain_samples_num = 100
    instance.sft_report_summary = Mock()
    return instance


def test_validation_cannot_make_same_training_step_advance_twice():
    status = manager()
    status.sft_train_ack("a", {}, 3, 10)
    status.sft_train_ack("a", {"val/avg_loss": 1.0}, 3, 10)
    # No replica is REDUCED now, but b still acknowledges the SAME update.
    assert not status.any_with_status([PolicyStatus.REDUCED])
    status.sft_train_ack("b", {}, 3, 10)
    assert status.current_step == 3
    status.data_fetcher.validation_activate_dataloader.assert_called_once_with(3)


def test_next_step_advances_despite_previous_step_status_and_late_ack():
    status = manager()
    status.sft_train_ack("a", {}, 3, 10)
    status.sft_train_ack("a", {}, 4, 10)
    assert status.current_step == 4
    status.sft_train_ack("b", {}, 3, 10)
    assert status.current_step == 4
    assert status.data_fetcher.validation_activate_dataloader.call_count == 2


def test_duplicate_and_validation_acks_do_not_advance_progress():
    status = manager()
    status.sft_train_ack("a", {"val/avg_loss": 1.0}, 2, 10)
    status.sft_train_ack("a", {}, 2, 10)
    status.sft_train_ack("a", {}, 2, 10)
    assert status.current_step == 2
    status.data_fetcher.validation_activate_dataloader.assert_not_called()
