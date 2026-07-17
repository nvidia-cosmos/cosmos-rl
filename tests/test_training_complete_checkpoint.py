# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from cosmos_rl.dispatcher.command import TrainingCompleteCommand
from cosmos_rl.policy.worker.rl_worker import RLPolicyWorker


def _command(*, do_save: bool = True) -> TrainingCompleteCommand:
    return TrainingCompleteCommand(
        "policy-0",
        global_step=2,
        total_steps=2,
        final_step=1,
        checkpoint_total_steps=5,
        remain_samples_num=7,
        do_save=do_save,
    )


def _worker(save_checkpoint=None, invalidate_checkpoint=None):
    return SimpleNamespace(
        replica_name="policy-0",
        replica_batch_for_this_step=-1,
        is_master_replica=True,
        trainer=SimpleNamespace(
            save_checkpoint=save_checkpoint or MagicMock(),
            invalidate_checkpoint_completion=(invalidate_checkpoint or MagicMock()),
        ),
        profiler=SimpleNamespace(step=MagicMock(), check_finished=lambda: False),
        parallel_dims=object(),
        global_rank=0,
        api_client=SimpleNamespace(post_policy_train_ack=MagicMock()),
    )


class TestTrainingCompleteCheckpointAgreement(unittest.TestCase):
    def test_local_save_failure_participates_then_prevents_ack(self):
        local_error = RuntimeError("local shard failed")
        worker = _worker(MagicMock(side_effect=local_error))
        with patch(
            "cosmos_rl.policy.worker.rl_worker.dist_util.all_reduce_tensor_object_cpu",
            side_effect=[torch.tensor([1]), torch.tensor([0])],
        ) as reduce_ok:
            with self.assertRaisesRegex(RuntimeError, "local shard failed"):
                RLPolicyWorker.execute_training_complete(worker, _command())
        self.assertEqual(reduce_ok.call_count, 2)
        worker.api_client.post_policy_train_ack.assert_not_called()

    def test_peer_save_failure_prevents_master_ack(self):
        worker = _worker()
        with patch(
            "cosmos_rl.policy.worker.rl_worker.dist_util.all_reduce_tensor_object_cpu",
            side_effect=[torch.tensor([1]), torch.tensor([0])],
        ):
            with self.assertRaisesRegex(RuntimeError, "another rank"):
                RLPolicyWorker.execute_training_complete(worker, _command())
        worker.api_client.post_policy_train_ack.assert_not_called()

    def test_unanimous_save_acks_with_real_checkpoint_coordinates(self):
        save = MagicMock()
        worker = _worker(save)
        with (
            patch(
                "cosmos_rl.policy.worker.rl_worker.dist_util.all_reduce_tensor_object_cpu",
                return_value=torch.tensor([1]),
            ),
            patch(
                "cosmos_rl.policy.worker.rl_worker.is_master_rank",
                return_value=True,
            ),
        ):
            should_stop = RLPolicyWorker.execute_training_complete(worker, _command())
        save.assert_called_once_with(
            current_step=1,
            total_steps=5,
            remain_samples_num=7,
            is_final=True,
        )
        worker.trainer.invalidate_checkpoint_completion.assert_called_once_with(1)
        worker.api_client.post_policy_train_ack.assert_called_once()
        self.assertTrue(should_stop)

    def test_invalidation_failure_prevents_save_and_ack(self):
        error = RuntimeError("stale marker removal failed")
        worker = _worker(invalidate_checkpoint=MagicMock(side_effect=error))
        with patch(
            "cosmos_rl.policy.worker.rl_worker.dist_util.all_reduce_tensor_object_cpu",
            return_value=torch.tensor([0]),
        ):
            with self.assertRaisesRegex(RuntimeError, "stale marker removal failed"):
                RLPolicyWorker.execute_training_complete(worker, _command())
        worker.trainer.save_checkpoint.assert_not_called()
        worker.api_client.post_policy_train_ack.assert_not_called()

    def test_disabled_save_skips_collective_and_still_acks(self):
        worker = _worker()
        with (
            patch(
                "cosmos_rl.policy.worker.rl_worker.dist_util.all_reduce_tensor_object_cpu"
            ) as reduce_ok,
            patch(
                "cosmos_rl.policy.worker.rl_worker.is_master_rank",
                return_value=True,
            ),
        ):
            RLPolicyWorker.execute_training_complete(worker, _command(do_save=False))
        reduce_ok.assert_not_called()
        worker.trainer.save_checkpoint.assert_not_called()
        worker.api_client.post_policy_train_ack.assert_called_once()


if __name__ == "__main__":
    unittest.main()
