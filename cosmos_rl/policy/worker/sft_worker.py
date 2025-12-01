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
import torch
from typing import Optional, Union, Callable, Dict
from torch.utils.data import Dataset
from tqdm import tqdm


from cosmos_rl.dispatcher.data.packer.base import BaseDataPacker
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.comm.base import CommMixin
from cosmos_rl.policy.trainer.base import TrainerRegistry
from cosmos_rl.comm.base import WorkerBase
from cosmos_rl.dispatcher.protocol import Role
from cosmos_rl.utils.profiler import CosmosProfiler
from cosmos_rl.utils import util
from cosmos_rl.utils.distributed import destroy_distributed
from cosmos_rl.utils.wandb_logger import (
    init_wandb,
    is_wandb_available,
)


from itertools import islice

from torch.utils.data import DataLoader, DistributedSampler, Sampler
from datasets import concatenate_datasets
from typing import Any


from cosmos_rl.policy.config import (
    SFTDataConfig,
    config_hash,
)
from cosmos_rl.policy.trainer.sampler import SkippingSampler
import cosmos_rl.utils.cache as cache

from cosmos_rl.policy.trainer.sft_trainer import SFTTrainer


class SFTDataset(Dataset):
    def __init__(
        self,
        config: SFTDataConfig,
        dataset: Dataset,
        data_packer: BaseDataPacker,
        is_user_dataset: bool = False,
    ):
        self.config = config
        self.column_name = config.conversation_column_name
        self.dataset = dataset
        self.data_packer = data_packer
        self.is_user_dataset = is_user_dataset
        self.cache = None
        if self.config.enable_dataset_cache:
            # TODO(zjx): can we reuse the cache between different training jobs?
            # It's not stable yet, we only checked if the config is the same
            # If there are any problems, it is recommended that the user clears the cache folder
            cache_folder = os.path.join(
                os.environ.get(
                    "COSMOS_CACHE",
                    os.path.join(os.path.expanduser("~"), ".cache/cosmos/"),
                ),
                "datasets_cache",
                f"{self.config.dataset.name}-{config_hash(config)}",
            )
            logger.info(f"SFTDataset Cache folder: {cache_folder}")
            self.cache = cache.DiskCache(cache_folder)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # we only cache on_the_fly result
        if self.cache is not None:
            cache_obj = self.cache.get(idx)
            if cache_obj is not None:
                return cache_obj

        raw_item = (
            self.dataset[idx][self.column_name]
            if not self.is_user_dataset and self.column_name
            else self.dataset[idx]
        )

        if isinstance(idx, list):  # a batch of items
            item = [self.data_packer.sft_process_sample(x) for x in raw_item]
        else:
            item: Dict[str, Any] = self.data_packer.sft_process_sample(raw_item)

        if self.cache is not None:
            # try cache obj
            self.cache.set(idx, item)
        return item


def construct_dataset(
    cosmos_config: CosmosConfig,
    data_packer: BaseDataPacker,
    user_provided_dataset: Optional[Dataset] = None,
    val_data_packer: Optional[BaseDataPacker] = None,
    user_provided_val_dataset: Optional[Dataset] = None,
):
    config = cosmos_config.train.train_policy
    if user_provided_dataset is not None:
        dataset = None
        train_dataset = user_provided_dataset
        logger.info("Using user-provided dataset, which will skip split processing.")
    else:
        dataset = util.load_data_from_disk_or_hf(
            config.dataset.name,
            config.dataset.subset,
            config.dataset.revision or None,
        )
        dataset_list = []
        for split_name in config.dataset.split:
            logger.info(
                f"Appending split {split_name}, dataset size = {len(dataset[split_name])}"
            )
            dataset_list.append(dataset[split_name])
        train_dataset = concatenate_datasets(dataset_list)
    logger.info(f"Final dataset size = {len(train_dataset)}")

    if cosmos_config.validation.enable:
        if user_provided_val_dataset is not None:
            test_dataset = user_provided_val_dataset
            logger.info(
                "Using user-provided validation dataset, which will skip split processing."
            )
        elif cosmos_config.validation.dataset.name:
            dataset = util.load_data_from_disk_or_hf(
                cosmos_config.validation.dataset.name,
                cosmos_config.validation.dataset.subset,
                cosmos_config.validation.dataset.revision or None,
            )
            dataset_list = []
            for split_name in cosmos_config.validation.dataset.split:
                logger.info(
                    f"Appending validation split {split_name}, validation dataset size = {len(dataset[split_name])}"
                )
                dataset_list.append(dataset[split_name])
            test_dataset = concatenate_datasets(dataset_list)
        else:
            logger.warning(
                "No validation dataset provided, using split of training dataset for validation."
            )
            if isinstance(train_dataset, torch.utils.data.Dataset):
                # Define the split ratio (e.g., 80% train, 20% test)
                if config.dataset.test_size is None:
                    logger.warning(
                        "No test size specified, using 10% of the training dataset for testing."
                    )
                    config.dataset.test_size = 0.1
                if isinstance(config.dataset.test_size, float):
                    n_test_samples = int(len(train_dataset) * config.dataset.test_size)
                else:
                    n_test_samples = config.dataset.test_size
                n_test_samples = max(min(n_test_samples, len(train_dataset) - 1), 1)

                # Generate deterministic indices
                indices = list(range(len(train_dataset)))
                test_indices = indices[:n_test_samples]
                train_indices = indices[n_test_samples:]

                test_dataset = torch.utils.data.Subset(train_dataset, test_indices)
                train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
            else:
                assert hasattr(
                    train_dataset, "train_test_split"
                ), "train_dataset must have train_test_split method"
                split = train_dataset.train_test_split(
                    test_size=config.dataset.test_size, shuffle=False
                )
                train_dataset = split["train"]
                test_dataset = split["test"]
    else:

        class EmptyDataset(Dataset):
            def __len__(self):
                return 0

            def __getitem__(self, idx):
                raise IndexError("EmptyDataset has no items")

        test_dataset = EmptyDataset()

    train_sft_dataset = SFTDataset(
        config,
        dataset=train_dataset,
        data_packer=data_packer,
        is_user_dataset=user_provided_dataset is not None,
    )
    test_sft_dataset = SFTDataset(
        config,
        dataset=test_dataset,
        data_packer=val_data_packer,
        is_user_dataset=user_provided_dataset is not None,
    )

    return train_sft_dataset, test_sft_dataset


def collate_fn(
    batch,
):
    return batch


class SFTPolicyWorker(WorkerBase, CommMixin):
    trainer: SFTTrainer

    def __init__(
        self,
        config: CosmosConfig,
        parallel_dims: ParallelDims,
        dataset: Optional[Union[Dataset, Callable[[CosmosConfig], Dataset]]] = None,
        data_packer: Optional[BaseDataPacker] = None,
        val_dataset: Optional[Union[Dataset, Callable[[CosmosConfig], Dataset]]] = None,
        val_data_packer: Optional[BaseDataPacker] = None,
        sampler: Optional[Callable] = None,
        batch_sampler: Optional[Callable] = None,
        val_sampler: Optional[Callable] = None,
        val_batch_sampler: Optional[Callable] = None,
    ):
        super(SFTPolicyWorker, self).__init__(
            config,
            parallel_dims=parallel_dims,
            dataset=dataset,
            data_packer=data_packer,
            val_dataset=val_dataset,
            val_data_packer=val_data_packer,
            sampler=sampler,
            batch_sampler=batch_sampler,
            val_sampler=val_sampler,
            val_batch_sampler=val_batch_sampler,
        )

    def worker_init(self, **kwargs):
        # Enlarge the compile cache size for validation
        if self.config.train.compile and self.config.validation.enable:
            torch._dynamo.config.cache_size_limit = 64

        parallel_dims: ParallelDims = kwargs.get("parallel_dims", None)
        if parallel_dims is None:
            raise ValueError("parallel_dims is required")
        self.parallel_dims = parallel_dims
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.global_rank = int(os.environ.get("RANK", 0))
        self.role = Role.POLICY
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)
        self.dp_rank, self.dp_world_size = 0, 1
        if self.parallel_dims.dp_enabled:
            self.dp_rank = self.parallel_dims.mesh["dp"].get_local_rank()
            self.dp_world_size = self.parallel_dims.mesh["dp"].size()

        # Prepare wandb
        if "wandb" in self.config.logging.logger and is_wandb_available():
            # Only initialize wandb on the first dp replicate coord and first rank for policy
            if self.parallel_dims.dp_replicate_coord[0] == 0 and self.global_rank == 0:
                init_wandb(self.config)
        else:
            logger.warning(
                "Wandb is not available. Please install it to use wandb logging features."
            )

        self.train_step = 0
        self.start_epoch = 0
        self.train_stream = torch.cuda.current_stream()

        self.init_comm()
        self.build_runner(**kwargs)

        self.profiler = CosmosProfiler(
            self.config,
            self.parallel_dims,
            replica_name=self.replica_name,
            api_client=self.api_client,
        )

    def setup(
        self,
        data_packer: Optional[BaseDataPacker] = None,
        val_data_packer: Optional[BaseDataPacker] = None,
    ):
        # setup data packer first
        self.init_data_packer(
            data_packer=data_packer,
            val_data_packer=val_data_packer,
        )

    def build_runner(self, **kwargs):
        self.setup(
            data_packer=kwargs.get("data_packer", None),
            val_data_packer=kwargs.get("val_data_packer", None),
        )

        self.trainer = TrainerRegistry.get_trainer_cls("sft")(
            config=self.config,
            parallel_dims=self.parallel_dims,
            train_stream=self.train_stream,
            data_packer=self.data_packer,
            val_data_packer=self.val_data_packer,
        )
        self.ckpt_total_steps, self.train_step = self.trainer.load_model()

        sampler = kwargs.get("sampler", None)
        batch_sampler = kwargs.get("batch_sampler", None)
        val_sampler = kwargs.get("val_sampler", None)
        val_batch_sampler = kwargs.get("val_batch_sampler", None)
        dataset = kwargs.get("dataset", None)
        val_dataset = kwargs.get("val_dataset", None)

        if isinstance(dataset, Callable):
            # Incase it is a factory function, we need to call it to get the dataset
            dataset = dataset(self.config)
            util.call_setup(dataset, self.config)

        if isinstance(val_dataset, Callable):
            val_dataset = val_dataset(self.config)
            util.call_setup(val_dataset, self.config)

        if not self.val_data_packer:
            self.val_data_packer = self.data_packer

        # Prepare dataset
        train_dataset, val_dataset = construct_dataset(
            self.config,
            data_packer=self.data_packer,
            user_provided_dataset=dataset,
            val_data_packer=self.val_data_packer,
            user_provided_val_dataset=val_dataset,
        )
        if sampler is not None:
            logger.info("Using user-provided sampler for training dataset.")
            if isinstance(sampler, Callable):
                train_sampler = sampler(
                    train_dataset,
                    num_replicas=self.dp_world_size,
                    rank=self.dp_rank,
                    shuffle=self.config.train.train_policy.dataloader_shuffle,
                    drop_last=False,
                )
            else:
                train_sampler = sampler
        else:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.dp_world_size,
                rank=self.dp_rank,
                shuffle=self.config.train.train_policy.dataloader_shuffle,
                drop_last=False,
            )

        if batch_sampler is not None and isinstance(batch_sampler, Callable):
            batch_sampler = batch_sampler(
                train_sampler,
                batch_size=self.config.train.train_batch_per_replica,
                drop_last=False,
            )

        def get_train_data_loader(
            sampler: Union[Sampler[int], Sampler[list[int]]],
            sampler_in_batch: Optional[Sampler[list[int]]] = None,
        ):
            if sampler_in_batch is not None:
                logger.info(
                    "Using custom batch Sampler that yields list of indices for training dataset."
                )
                data_loader = DataLoader(
                    train_dataset,
                    num_workers=self.config.train.train_policy.dataloader_num_workers,
                    prefetch_factor=self.config.train.train_policy.dataloader_prefetch_factor,
                    batch_sampler=sampler_in_batch,
                    collate_fn=collate_fn,
                )
            else:
                data_loader = DataLoader(
                    train_dataset,
                    batch_size=self.config.train.train_batch_per_replica,
                    shuffle=False,
                    num_workers=self.config.train.train_policy.dataloader_num_workers,
                    prefetch_factor=self.config.train.train_policy.dataloader_prefetch_factor,
                    sampler=sampler,
                    collate_fn=collate_fn,
                    drop_last=False,
                )
            return data_loader

        if self.config.train.resume and self.train_step > 0:
            """
            Note: Here we assume there is no data shuffling across epochs.
            Otherwise, we need to call `set_epoch` on the sampler after each epoch.
            """
            # Resume training from the last checkpoint if needed
            total_steps_per_epoch = len(
                get_train_data_loader(train_sampler, batch_sampler)
            )
            data_loader_bias = self.train_step % total_steps_per_epoch
            data_loader_bias *= self.config.train.train_batch_per_replica
            logger.info(
                f"Resuming training from step {self.train_step}/{self.ckpt_total_steps}"
            )
            train_sampler = SkippingSampler(
                train_sampler,
                skip_samples=data_loader_bias
                // (
                    len(list(islice(iter(train_sampler), 1))[0])
                    if isinstance(list(islice(iter(train_sampler), 1))[0], list)
                    else 1
                ),
            )
            if batch_sampler is not None:
                batch_sampler = SkippingSampler(
                    batch_sampler,
                    skip_samples=data_loader_bias
                    // (
                        len(list(islice(iter(batch_sampler), 1))[0])
                        if isinstance(list(islice(iter(batch_sampler), 1))[0], list)
                        else 1
                    ),
                )
            self.start_epoch = self.train_step // total_steps_per_epoch

        if val_sampler is not None:
            logger.info("Using user-provided sampler for validation dataset.")
            if isinstance(val_sampler, Callable):
                val_sampler = val_sampler(
                    val_dataset,
                    num_replicas=self.dp_world_size,
                    rank=self.dp_rank,
                    shuffle=False,
                    drop_last=False,
                )
        else:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.dp_world_size,
                rank=self.dp_rank,
                shuffle=False,
                drop_last=False,
            )
        self.epoch = self.config.train.epoch

        self.train_data_loader = get_train_data_loader(train_sampler, batch_sampler)
        if val_batch_sampler is not None:
            logger.info(
                "Using custom batch Sampler that yields list of indices for validation dataset."
            )
            if isinstance(val_batch_sampler, Callable):
                val_batch_sampler = val_batch_sampler(
                    val_sampler,
                    batch_size=self.config.validation.batch_size
                    or self.config.train.train_batch_per_replica,
                    drop_last=False,
                )
            self.val_data_loader = DataLoader(
                val_dataset,
                num_workers=self.config.train.train_policy.dataloader_num_workers,
                prefetch_factor=self.config.train.train_policy.dataloader_prefetch_factor,
                batch_sampler=val_batch_sampler,
                collate_fn=collate_fn,
            )
        else:
            self.val_data_loader = DataLoader(
                val_dataset,
                batch_size=self.config.validation.batch_size
                or self.config.train.train_batch_per_replica,
                num_workers=self.config.train.train_policy.dataloader_num_workers,
                prefetch_factor=self.config.train.train_policy.dataloader_prefetch_factor,
                sampler=val_sampler,
                collate_fn=collate_fn,
                drop_last=False,
            )

        steps_by_dataset = (
            self.ckpt_total_steps
            if self.ckpt_total_steps > 0
            else len(self.train_data_loader) * self.epoch
        )

        if self.config.train.max_num_steps is not None:
            self.total_steps = min(steps_by_dataset, self.config.train.max_num_steps)
        else:
            self.total_steps = steps_by_dataset

        # Calculate the step interval to save the checkpoint
        if self.config.train.ckpt.save_freq_in_epoch > 0:
            # Use save_freq_in_epoch to calculate the save frequency in priority
            self._save_freq = (
                self.config.train.ckpt.save_freq_in_epoch * len(self.train_data_loader)
            ) // self.dp_world_size
            logger.info(
                f"Checkpoint will be saved every {self._save_freq} steps, which is approximately every `train.ckpt.save_freq_in_epoch` {self.config.train.ckpt.save_freq_in_epoch} epochs. `train.ckpt.save_freq` will be ignored."
            )
        else:
            self._save_freq = self.config.train.ckpt.save_freq

    def execute(self):
        """
        Execute the training.
        """
        assert self.trainer is not None, "[Policy] Trainer has not been built."
        try:
            with torch.autocast(
                device_type="cuda",
                dtype=util.str2torch_dtype(self.config.train.param_dtype),
            ):
                self.main_loop()
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise e
        finally:
            self.destroy_worker()

    def validate(self, is_last_step: bool = False):
        if not self.config.validation.enable:
            return None
        if self.parallel_dims.dp_replicate_coord[0] != 0:
            return
        if self.train_step % self.config.validation.freq != 0 and not is_last_step:
            return None

        # validation
        logger.info(f"Validation at step {self.train_step}/{self.total_steps}...")
        val_total_loss = 0.0
        for val_global_batch in tqdm(self.val_data_loader, desc="Validation"):
            val_score = self.trainer.validate(
                val_global_batch, self.train_step, self.total_steps
            )
            val_total_loss += val_score
        val_avg_loss = val_total_loss / len(self.val_data_loader.dataset)
        logger.info(f"Validation loss: {val_avg_loss}")
        return val_avg_loss

    def main_loop(self):
        self.profiler.start()
        pp_last_stage = False

        if self.parallel_dims.pp_enabled:
            pp_last_stage = (
                self.parallel_dims.pp_coord[0] == self.parallel_dims.pp_coord[1] - 1
            )
        for cur_epoch in range(self.start_epoch, self.epoch):
            logger.info(f"Training epoch {cur_epoch + 1}/{self.epoch}")
            for global_batch in self.train_data_loader:
                # if [profiler.enable_nsys] is true, cudaProfilerStart() / cudaProfilerStop() are used to trigger nsys capture
                # settings from [profiler.sub_profiler_config] are reused
                if (
                    self.config.profiler.enable_nsys
                    and self.profiler.global_rank in self.profiler.rank_filter
                ):
                    if (
                        self.train_step
                        == self.profiler.wait_steps + self.profiler.warmup_steps
                    ):
                        torch.cuda.cudart().cudaProfilerStart()
                    elif (
                        self.train_step
                        == self.profiler.wait_steps
                        + self.profiler.warmup_steps
                        + self.profiler.active_steps
                    ):
                        torch.cuda.cudart().cudaProfilerStop()

                self.trainer.step_training(
                    global_batch=global_batch,
                    total_steps=self.total_steps,
                    train_step=self.train_step,
                    save_freq=self._save_freq,
                    pp_last_stage=pp_last_stage,
                )

                self.train_step += 1
                if (
                    self.config.train.max_num_steps is not None
                    and self.train_step >= self.total_steps
                ):
                    break  # break outer epoch loop

                val_avg_loss = self.validate(is_last_step=False)

                self.trainer.checkpointing(
                    total_steps=self.total_steps,
                    train_step=self.train_step,
                    save_freq=self._save_freq,
                    is_last_step=False,
                    pp_last_stage=pp_last_stage,
                    val_score=val_avg_loss,
                )

                self.profiler.step()

        # Finally: validation and save checkpoint
        val_avg_loss = self.validate(is_last_step=True)
        self.trainer.checkpointing(
            total_steps=self.total_steps,
            train_step=self.train_step,
            save_freq=self._save_freq,
            is_last_step=True,
            pp_last_stage=pp_last_stage,
            val_score=val_avg_loss,
        )

        self.unregister_from_controller()

    def destroy_worker(self):
        destroy_distributed()
        logger.info("[Policy] Process group destroyed.")
