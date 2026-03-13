# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

from cosmos_rl.dispatcher.data.data_fetcher import ControllerDataFetcher
from cosmos_rl.policy.trainer.base import TrainerRegistry
from cosmos_rl.policy.worker.sft_worker import SFTPolicyWorker


class _IndexDataset(Dataset):
    def __init__(self, size: int):
        self._size = size

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        return {"idx": idx}


class _IdentityPacker:
    def setup(self, *_args, **_kwargs):
        return None

    def sft_process_sample(self, sample):
        return sample


class _WorkerHarness:
    build_runner = SFTPolicyWorker.build_runner
    get_batch_from_dataloader = SFTPolicyWorker.get_batch_from_dataloader

    def __init__(self, config):
        self.config = config
        self.parallel_dims = SimpleNamespace()
        self.train_stream = None
        self.dp_rank = 0
        self.dp_world_size = 1
        self.global_rank = 0
        self.enable_dp_load_balancing = False
        self.train_step = 0
        self.start_epoch = 0
        self.hook_fns = {}
        self.custom_logger_fns = []
        self._packer = _IdentityPacker()

    def setup(self, *_args, **_kwargs):
        self.data_packer = self._packer
        self.val_data_packer = self._packer


def _make_sft_config(resume: bool):
    return SimpleNamespace(
        train=SimpleNamespace(
            resume=resume,
            epoch=1,
            max_num_steps=100,
            train_batch_per_replica=2,
            sequence_packing=False,
            ckpt=SimpleNamespace(save_freq_in_epoch=0, save_freq=10),
            train_policy=SimpleNamespace(
                trainer_type="sft",
                type="sft",
                conversation_column_name="conversation",
                enable_dataset_cache=False,
                dataset=SimpleNamespace(name="unit_test_dataset"),
                enable_dp_load_balancing=False,
                dataloader_shuffle=False,
                dataloader_seed=42,
                dataloader_num_workers=1,
                dataloader_prefetch_factor=2,
                dataloader_drop_last=False,
                dataloader_broadcast=False,
                load_balanced_max_tokens_for_batch=None,
                load_balanced_batches_per_optimizer_step=1,
                load_balanced_pool_size=8,
                load_balanced_batching_strategy="greedy",
            ),
        ),
        policy=SimpleNamespace(is_diffusers=False, model_max_length=128),
        validation=SimpleNamespace(
            enable=False,
            batch_size=2,
            freq=1,
            val_before_train=False,
            dataset=SimpleNamespace(name="", subset="", revision="", split=[]),
        ),
        logging=SimpleNamespace(logger=[]),
    )


class TestSFTResumeDataloaderIndex(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.ckpt_dir = os.path.join(self.test_dir, "sft_resume_ckpt")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_state_file = os.path.join(self.ckpt_dir, "state.pt")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_resume_dataloader_index_matches_resume_point(self):
        ckpt_state_file = self.ckpt_state_file

        class _DummyTrainer:
            def __init__(
                self,
                config,
                parallel_dims,
                train_stream,
                data_packer,
                val_data_packer,
            ):
                self.config = config

            def load_model(self):
                if self.config.train.resume and os.path.exists(ckpt_state_file):
                    train_step = torch.load(ckpt_state_file)["train_step"]
                    return train_step + 10, train_step, None
                return 0, 0, None

        with patch.object(
            TrainerRegistry, "get_trainer_cls", lambda _name: _DummyTrainer
        ):
            dataset = _IndexDataset(size=20)

            # First run: consume several batches and persist a simple checkpoint state.
            worker_run1 = _WorkerHarness(_make_sft_config(resume=False))
            worker_run1.build_runner(
                data_packer=worker_run1._packer,
                val_data_packer=worker_run1._packer,
                dataset=dataset,
                val_dataset=dataset,
            )

            steps_before_resume = 3
            consumed_indices = []
            for batch in worker_run1.get_batch_from_dataloader(
                worker_run1.train_data_loader
            ):
                consumed_indices.extend(item["idx"] for item in batch)
                worker_run1.train_step += 1
                if worker_run1.train_step >= steps_before_resume:
                    break

            torch.save({"train_step": worker_run1.train_step}, ckpt_state_file)

            # Resume run: load step from checkpoint and verify dataloader is skipped correctly.
            worker_run2 = _WorkerHarness(_make_sft_config(resume=True))
            worker_run2.build_runner(
                data_packer=worker_run2._packer,
                val_data_packer=worker_run2._packer,
                dataset=dataset,
                val_dataset=dataset,
            )

            resumed_first_batch = next(iter(worker_run2.train_data_loader))
            resumed_batch_indices = [item["idx"] for item in resumed_first_batch]

            expected_resume_start = consumed_indices[-1] + 1
            self.assertEqual(worker_run2.train_step, steps_before_resume)
            self.assertEqual(resumed_batch_indices[0], expected_resume_start)
            self.assertEqual(
                resumed_batch_indices,
                [expected_resume_start, expected_resume_start + 1],
            )

    def test_resume_dataloader_index_across_epoch_boundary(self):
        ckpt_state_file = self.ckpt_state_file

        class _DummyTrainer:
            def __init__(
                self,
                config,
                parallel_dims,
                train_stream,
                data_packer,
                val_data_packer,
            ):
                self.config = config

            def load_model(self):
                if self.config.train.resume and os.path.exists(ckpt_state_file):
                    train_step = torch.load(ckpt_state_file)["train_step"]
                    return train_step + 10, train_step, None
                return 0, 0, None

        with patch.object(
            TrainerRegistry, "get_trainer_cls", lambda _name: _DummyTrainer
        ):
            dataset = _IndexDataset(size=20)
            batch_size = 2
            steps_per_epoch = len(dataset) // batch_size

            # Simulate a checkpoint saved after crossing epoch boundary.
            # For dataset size 20 and batch size 2, steps_per_epoch = 10.
            # train_step=12 means next resumed sample should start at index 4 in epoch 2.
            steps_before_resume = steps_per_epoch + 2
            torch.save({"train_step": steps_before_resume}, ckpt_state_file)

            worker_run = _WorkerHarness(_make_sft_config(resume=True))
            worker_run.build_runner(
                data_packer=worker_run._packer,
                val_data_packer=worker_run._packer,
                dataset=dataset,
                val_dataset=dataset,
            )

            resumed_first_batch = next(iter(worker_run.train_data_loader))
            resumed_batch_indices = [item["idx"] for item in resumed_first_batch]

            expected_start_epoch = steps_before_resume // steps_per_epoch
            expected_resume_start = (steps_before_resume % steps_per_epoch) * batch_size

            self.assertEqual(worker_run.train_step, steps_before_resume)
            self.assertEqual(worker_run.start_epoch, expected_start_epoch)
            self.assertEqual(
                resumed_batch_indices,
                [expected_resume_start, expected_resume_start + 1],
            )


class _RLPromptDataset(Dataset):
    def __init__(self, size: int):
        self._size = size

    def setup(self, config):
        return None

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        return f"prompt-{idx}"

    def get_reference_answer(self, idx):
        return f"answer-{idx}"


def _make_rl_config(resume: bool, epoch: int = 2):
    return SimpleNamespace(
        train=SimpleNamespace(
            resume=resume,
            epoch=epoch,
            local_dataset=False,
            train_batch_per_replica=2,
            train_policy=SimpleNamespace(
                type="grpo",
                dataloader_batch_size=2,
                dataloader_shuffle=False,
                dataloader_seed=42,
                dataloader_num_workers=1,
                dataloader_prefetch_factor=2,
                data_dispatch_as_rank_in_mesh=False,
            ),
        ),
        rollout=SimpleNamespace(
            batch_size=2,
            n_generation=1,
            multi_turn_config=SimpleNamespace(enable=False),
        ),
        validation=SimpleNamespace(enable=False, dataset=SimpleNamespace(name="")),
    )


class TestRLResumeDataFetcherIndex(unittest.TestCase):
    def _build_fetcher(self, remain_samples_num: int) -> ControllerDataFetcher:
        class _DummyCheckpointManager:
            def __init__(self, config):
                self.config = config

            def load_extra_info_from_checkpoint(self):
                return {"remain_samples_num": remain_samples_num}

        with patch(
            "cosmos_rl.dispatcher.data.data_fetcher.CheckpointMananger",
            _DummyCheckpointManager,
        ):
            return ControllerDataFetcher(
                config=_make_rl_config(resume=True, epoch=2),
                dataset=_RLPromptDataset(size=10),
                val_dataset=None,
                is_rl=True,
            )

    @staticmethod
    def _to_int_list(idxs):
        out = []
        for idx in idxs:
            out.append(int(idx.item()) if hasattr(idx, "item") else int(idx))
        return out

    def test_resume_skips_to_expected_index_same_epoch(self):
        # Dataset size 10, epochs 2 => total samples 20.
        # remain=14 means consumed=6 samples in epoch 1, so next index should be 6.
        fetcher = self._build_fetcher(remain_samples_num=14)
        self.assertEqual(fetcher.ckpt_extra_info, {"remain_samples_num": 14})

        idxs, _payloads = next(fetcher.train_dataloader_iter)
        idxs = self._to_int_list(idxs)

        self.assertEqual(fetcher.epoch, 1)
        self.assertEqual(idxs, [6, 7])

    def test_resume_skips_to_expected_index_across_epoch(self):
        # Dataset size 10, epochs 2 => total samples 20.
        # remain=8 means consumed=12 samples, i.e. epoch 2 with 2 samples consumed.
        # Next resumed index should be 2 in epoch 2.
        fetcher = self._build_fetcher(remain_samples_num=8)
        self.assertEqual(fetcher.ckpt_extra_info, {"remain_samples_num": 8})

        idxs, _payloads = next(fetcher.train_dataloader_iter)
        idxs = self._to_int_list(idxs)

        self.assertEqual(fetcher.epoch, 2)
        self.assertEqual(idxs, [2, 3])


if __name__ == "__main__":
    unittest.main()
