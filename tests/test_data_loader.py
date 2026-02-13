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
from cosmos_rl.utils.logging import logger
import unittest
from cosmos_rl.policy.worker.sft_worker import SFTPolicyWorker
from cosmos_rl.utils.distributed import init_distributed
from cosmos_rl.utils.parallelism import ParallelDims
from launch_test_worker import load_simple_sft_config

import cosmos_rl.utils.util as util
from cosmos_rl.policy.config import Config, ParallelismConfig
from torch.utils.data import DataLoader, DistributedSampler
from datasets import concatenate_datasets


class TestDataLoaderBroadcast(unittest.TestCase):
    class TestEntity(SFTPolicyWorker):
        def __init__(self, parallel_dims: ParallelDims, config: Config, **kwargs):
            self.parallel_dims = parallel_dims
            self.config = config

    def test_dataloader_broadcast(self):
        config_dict = load_simple_sft_config()
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        config_dict["train"]["train_policy"]["dataset"]["name"] = os.path.join(
            cur_dir, "data_fixtures", "test_dataset"
        )

        config = Config.from_dict(
            config_dict,
        )
        config.train.train_policy.dataloader_broadcast = True
        parallel_dims = ParallelDims.from_config(
            parallesim_config=ParallelismConfig(
                tp_size=2,
                cp_size=2,
                dp_shard_size=2,
            )
        )
        init_distributed()
        parallel_dims.build_mesh(device_type="cuda")

        dataset = util.load_data_from_disk_or_hf(
            config.train.train_policy.dataset.name,
            config.train.train_policy.dataset.subset,
            config.train.train_policy.dataset.revision or None,
        )
        dataset_list = []
        for split_name in config.train.train_policy.dataset.split:
            logger.info(
                f"Appending split {split_name}, dataset size = {len(dataset[split_name])}"
            )
            dataset_list.append(dataset[split_name])
        train_dataset = concatenate_datasets(dataset_list)
        # logger.info(f"Total dataset size after concatenation: {len(train_dataset)} {train_dataset[0]}")

        sampler = DistributedSampler(
            train_dataset,
            num_replicas=parallel_dims.dp_coord[1],
            rank=parallel_dims.dp_coord[0],
            shuffle=True,
            drop_last=False,
            seed=42,
        )

        data_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            prefetch_factor=2,
            sampler=sampler,
            drop_last=False,
            collate_fn=lambda x: x,  # Identity collate_fn to keep the batch as a list of samples for easier debugging
        )
        iterator = iter(data_loader)
        inst = TestDataLoaderBroadcast.TestEntity(parallel_dims, config)
        for batch in inst.get_batch_from_dataloader(data_loader):
            if parallel_dims.mesh["pp_cp_tp"].get_local_rank() != 0:
                ref = next(iterator)
                assert util.recursive_check_equal(
                    batch, ref
                ), f"Broadcast batch does not match dataloader batch for non-zero rank {batch} {ref}"


if __name__ == "__main__":
    unittest.main()
