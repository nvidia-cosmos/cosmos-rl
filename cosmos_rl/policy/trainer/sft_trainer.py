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
from functools import partial
from typing import Any, Dict, Optional

import cosmos_rl.utils.cache as cache
import cosmos_rl.utils.distributed as dist_util
import cosmos_rl.utils.util as util
import numpy as np
import torch
from cosmos_rl.dispatcher.data.packer import DataPacker
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.config import SFTDataConfig, config_hash
from cosmos_rl.policy.trainer import Trainer
from cosmos_rl.policy.trainer.optm import build_lr_schedulers
from cosmos_rl.policy.trainer.sampler import SkippingSampler
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.utils.ulysses import slice_inputs_for_ulysses
from cosmos_rl.utils.wandb_logger import (init_wandb, is_wandb_available,
                                          log_wandb)
from datasets import concatenate_datasets
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from tqdm import tqdm
from transformers import AutoTokenizer


def async_safe_ce(
    output: torch.Tensor,
    target: torch.LongTensor,
    ignore_index: int = -100,
    loss_scaling_factor: float = 1.0,
    output_packing_mask: Optional[torch.Tensor] = None,
    target_packing_mask: Optional[torch.Tensor] = None,
    dp_group: Optional[torch.distributed.ProcessGroup] = None,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    if output_packing_mask is not None:
        output = (
            output[output_packing_mask].contiguous().view(-1, output.size(-1)).float()
        )
    else:
        output = output[:, :-1].contiguous().view(-1, output.size(-1)).float()

    if target_packing_mask is not None:
        target = target[target_packing_mask].contiguous().view(-1)
    else:
        target = target[:, 1:].contiguous().view(-1)

    if cp_group is not None and cp_group.size() > 1:
        # Fallback to unbalance loss
        loss = (
            torch.nn.functional.cross_entropy(
                output,
                target,
                ignore_index=ignore_index,
                reduction="mean",
            )
            * loss_scaling_factor
        )
        # In case of all labels are ignored, loss will be nan.
        loss = torch.nan_to_num(loss, nan=0.0)
        return loss
    else:
        loss = torch.nn.functional.cross_entropy(
            output,
            target,
            ignore_index=ignore_index,
            reduction="none",
        )

        # Compute all token numbers across dp-world
        n_valid_tokens = (target != ignore_index).sum()
        num_dp_workers = 1
        if dp_group is not None:
            torch.distributed.all_reduce(n_valid_tokens, group=dp_group)
            num_dp_workers = torch.distributed.get_world_size(group=dp_group)

        loss = (
            loss.sum()
            / (n_valid_tokens + 1e-8)
            * (num_dp_workers * loss_scaling_factor)
        )
        return loss


def collate_fn(
    batch,
):
    return batch


def construct_dataset(
    cosmos_config: CosmosConfig,
    tokenizer: AutoTokenizer,
    data_packer: DataPacker,
    user_provided_dataset: Optional[Dataset] = None,
    val_data_packer: Optional[DataPacker] = None,
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
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_packer=data_packer,
        is_user_dataset=user_provided_dataset is not None,
    )
    test_sft_dataset = SFTDataset(
        config,
        tokenizer=tokenizer,
        dataset=test_dataset,
        data_packer=val_data_packer,
        is_user_dataset=user_provided_dataset is not None,
    )

    return train_sft_dataset, test_sft_dataset


class SFTDataset(Dataset):
    def __init__(
        self,
        config: SFTDataConfig,
        tokenizer: AutoTokenizer,
        dataset: Dataset,
        data_packer: DataPacker,
        is_user_dataset: bool = False,
    ):
        self.config = config
        self.tokenizer = tokenizer
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


class SFTTrainer(Trainer):
    def __init__(
        self,
        config: CosmosConfig,
        parallel_dims: ParallelDims,
        dataset: Optional[Dataset] = None,
        data_packer: Optional[DataPacker] = None,
        val_dataset: Optional[Dataset] = None,
        val_data_packer: Optional[DataPacker] = None,
        sampler: Optional[Callable] = None,
        batch_sampler: Optional[Callable] = None,
        val_sampler: Optional[Callable] = None,
        val_batch_sampler: Optional[Callable] = None,
    ):
        super(SFTTrainer, self).__init__(config, parallel_dims)

        # Enlarge the compile cache size for validation
        if config.train.compile and config.validation.enable:
            torch._dynamo.config.cache_size_limit = 64

        self.dp_rank, self.dp_world_size = 0, 1
        if parallel_dims.dp_enabled:
            self.dp_rank = parallel_dims.mesh["dp"].get_local_rank()
            self.dp_world_size = parallel_dims.mesh["dp"].size()

        # Prepare wandb
        if "wandb" in config.logging.logger and is_wandb_available():
            init_wandb(config, parallel_dims)
        else:
            logger.warning(
                "Wandb is not available. Please install it to use wandb logging features."
            )

        self.train_step = 0
        ckpt_total_steps = 0
        self.lr_schedulers = None
        self.start_epoch = 0
        # Load model
        if config.train.resume:
            try:
                # early init the lr_schedulers to avoid it is not initialized when loading the checkpoint
                ckpt_extra_vars, self.lr_schedulers = self.ckpt_manager.load_checkpoint(
                    model=self.model,
                    optimizer=self.optimizers,
                    scheduler=partial(
                        build_lr_schedulers, self.optimizers, self.config
                    ),
                )
                ckpt_total_steps = ckpt_extra_vars.get("total_steps", 0)
                self.train_step = ckpt_extra_vars.get("step", 0)
            except Exception as e:
                logger.error(
                    f"Cannot resume due to error: {e}. Trying to load from HuggingFace..."
                )
                self.model.load_hf_weights(
                    config.policy.model_name_or_path,
                    parallel_dims,
                    self.device,
                    revision=config.policy.model_revision,
                )
        else:
            self.model.load_hf_weights(
                config.policy.model_name_or_path,
                parallel_dims,
                self.device,
                revision=config.policy.model_revision,
            )
        self.model.train()

        if isinstance(dataset, Callable):
            # Incase it is a factory function, we need to call it to get the dataset
            dataset = dataset(self.config)
            dataset.setup(self.config, self.tokenizer)
        if data_packer:
            data_packer.setup(self.config, self.tokenizer)
            self.data_packer = data_packer

        if isinstance(val_dataset, Callable):
            val_dataset = val_dataset(self.config)
            val_dataset.setup(self.config, self.tokenizer)
        if val_data_packer:
            val_data_packer.setup(self.config, self.tokenizer)
            self.val_data_packer = val_data_packer
        else:
            self.val_data_packer = self.data_packer

        # Prepare dataset
        train_dataset, val_dataset = construct_dataset(
            config,
            tokenizer=self.tokenizer,
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
                    shuffle=config.train.train_policy.dataloader_shuffle,
                    drop_last=False,
                )
            else:
                train_sampler = sampler
        else:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.dp_world_size,
                rank=self.dp_rank,
                shuffle=config.train.train_policy.dataloader_shuffle,
                drop_last=False,
            )

        if batch_sampler is not None and isinstance(batch_sampler, Callable):
            batch_sampler = batch_sampler(
                train_sampler,
                batch_size=config.train.train_batch_per_replica,
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
                    num_workers=config.train.train_policy.dataloader_num_workers,
                    prefetch_factor=config.train.train_policy.dataloader_prefetch_factor,
                    batch_sampler=sampler_in_batch,
                    collate_fn=collate_fn,
                )
            else:
                data_loader = DataLoader(
                    train_dataset,
                    batch_size=config.train.train_batch_per_replica,
                    shuffle=False,
                    num_workers=config.train.train_policy.dataloader_num_workers,
                    prefetch_factor=config.train.train_policy.dataloader_prefetch_factor,
                    sampler=sampler,
                    collate_fn=collate_fn,
                    drop_last=False,
                )
            return data_loader

        if config.train.resume and self.train_step > 0:
            """
            Note: Here we assume there is no data shuffling across epochs.
            Otherwise, we need to call `set_epoch` on the sampler after each epoch.
            """
            # Resume training from the last checkpoint if needed
            total_steps_per_epoch = len(
                get_train_data_loader(train_sampler, batch_sampler)
            )
            data_loader_bias = self.train_step % total_steps_per_epoch
            data_loader_bias *= config.train.train_batch_per_replica
            logger.info(
                f"Resuming training from step {self.train_step}/{ckpt_total_steps}"
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
        self.epoch = config.train.epoch

        assert (
            self.tokenizer.pad_token_id is not None
        ), "Tokenizer must have a pad token id"
        self.train_data_loader = get_train_data_loader(train_sampler, batch_sampler)
        if val_batch_sampler is not None:
            logger.info(
                "Using custom batch Sampler that yields list of indices for validation dataset."
            )
            if isinstance(val_batch_sampler, Callable):
                val_batch_sampler = val_batch_sampler(
                    val_sampler,
                    batch_size=config.validation.batch_size
                    or config.train.train_batch_per_replica,
                    drop_last=False,
                )
            self.val_data_loader = DataLoader(
                val_dataset,
                num_workers=config.train.train_policy.dataloader_num_workers,
                prefetch_factor=config.train.train_policy.dataloader_prefetch_factor,
                batch_sampler=val_batch_sampler,
                collate_fn=collate_fn,
            )
        else:
            self.val_data_loader = DataLoader(
                val_dataset,
                batch_size=config.validation.batch_size
                or config.train.train_batch_per_replica,
                num_workers=config.train.train_policy.dataloader_num_workers,
                prefetch_factor=config.train.train_policy.dataloader_prefetch_factor,
                sampler=val_sampler,
                collate_fn=collate_fn,
                drop_last=False,
            )

        steps_by_dataset = (
            ckpt_total_steps
            if ckpt_total_steps > 0
            else len(self.train_data_loader) * self.epoch
        )
        if config.train.max_num_steps is not None:
            self.total_steps = min(steps_by_dataset, config.train.max_num_steps)
        else:
            self.total_steps = steps_by_dataset

        if self.lr_schedulers is None:
            assert (
                self.train_step == 0
            ), "`SFTTrainer.lr_schedulers` should be None if training is from scratch"
            self.lr_schedulers = build_lr_schedulers(
                self.optimizers, self.config, self.total_steps
            )

        if self.parallel_dims.dp_shard_enabled:
            dp_group = self.parallel_dims.mesh["dp_shard"].get_group()
        else:
            dp_group = None

        if self.parallel_dims.cp_enabled:
            cp_group = self.parallel_dims.mesh["cp"].get_group()
        else:
            cp_group = None

        self.loss_fn = partial(
            async_safe_ce,
            dp_group=dp_group,
            cp_group=cp_group,
        )

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

    def validate(self):
        if not self.config.validation.enable:
            return
        if self.parallel_dims.dp_replicate_coord[0] != 0:
            return

        logger.info(f"Validation at step {self.train_step}/{self.total_steps}...")
        self.model.eval()
        with torch.no_grad():
            val_total_loss = 0.0
            for val_global_batch in tqdm(self.val_data_loader, desc="Validation"):
                fixed_length = (
                    self.config.policy.model_max_length
                    if self.parallel_dims.pp_enabled
                    and not self.parallel_dims.pp_dynamic_shape
                    else None
                )
                if fixed_length is None:
                    max_len = min(
                        self.config.policy.model_max_length,
                        self.val_data_packer.sft_compute_max_len(val_global_batch),
                    )
                else:
                    max_len = fixed_length
                if self.seq_len_multiple > 1:
                    max_len = (
                        (max_len + self.seq_len_multiple - 1)
                        // self.seq_len_multiple
                        * self.seq_len_multiple
                    )

                val_batch = self.val_data_packer.sft_collate_fn(
                    val_global_batch,
                    computed_max_len=max_len,
                    pad_token_id=self.tokenizer.pad_token_id,
                    ignore_label_id=-100,
                )
                for k, v in val_batch.items():
                    val_batch[k] = (
                        v.to(self.device) if isinstance(v, torch.Tensor) else v
                    )
                val_inputs = val_batch["input_ids"]
                val_labels = val_batch.pop("label_ids")
                val_position_ids, _, val_pos_seq_dim = self.model.get_position_ids(
                    **val_batch
                )

                val_batch["position_ids"] = val_position_ids
                val_padding_mask = val_batch.get("padding_mask", None)

                delay_cp_slice_inputs = getattr(
                    self.model, "delay_cp_slice_inputs", False
                )
                if self.parallel_dims.cp_enabled and not delay_cp_slice_inputs:
                    [val_inputs, val_position_ids, val_padding_mask] = (
                        slice_inputs_for_ulysses(
                            [val_inputs, val_position_ids, val_padding_mask],
                            self.parallel_dims.mesh["cp"],
                            seq_dims=[1, val_pos_seq_dim, 1],
                        )
                    )

                    val_batch["input_ids"] = val_inputs
                    val_batch["position_ids"] = val_position_ids
                    if val_padding_mask is not None:
                        val_batch["padding_mask"] = val_padding_mask

                if self.parallel_dims.pp_enabled:
                    pp_last_stage = (
                        self.parallel_dims.pp_coord[0]
                        == self.parallel_dims.pp_coord[1] - 1
                    )
                    pp_first_stage = self.parallel_dims.pp_coord[0] == 0

                    if pp_first_stage:
                        self.pp_scheduler_val.step(
                            **val_batch,
                            pp_dynamic_shape_enabled=self.parallel_dims.pp_dynamic_shape_enabled,
                            seq_len_multiple=self.seq_len_multiple,
                        )
                    else:
                        pp_out = self.pp_scheduler_val.step(
                            position_ids=val_position_ids,
                            pp_dynamic_shape_enabled=self.parallel_dims.pp_dynamic_shape_enabled,
                            seq_len_multiple=self.seq_len_multiple,
                        )

                    if pp_last_stage:
                        val_loss = self.loss_fn(pp_out, val_labels)
                    else:
                        val_loss = torch.tensor([-1.0], device=self.device)
                else:
                    val_logits = self.model(**val_batch)

                    val_loss = self.loss_fn(val_logits, val_labels)
                val_total_loss += val_loss.item() * val_inputs.size(0)
            val_avg_loss = val_total_loss / len(self.val_data_loader.dataset)
            logger.info(f"Validation loss: {val_avg_loss}")
        return val_avg_loss

    def train(self):
        self.profiler.start()
        pp_last_stage = False

        for cur_epoch in range(self.start_epoch, self.epoch):
            logger.info(f"Training epoch {cur_epoch + 1}/{self.epoch}")
            for global_batch in self.train_data_loader:
                acc_loss = torch.zeros(1, device=self.device)
                self.optimizers.zero_grad()
                global_batch_size = len(global_batch)
                # split global_batch into mini_batches
                mini_batch_begin_idxs = list(
                    range(
                        0,
                        global_batch_size,
                        self.config.train.train_policy.mini_batch,
                    )
                )

                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

                for i in mini_batch_begin_idxs:
                    fixed_length = (
                        self.config.policy.model_max_length
                        if self.parallel_dims.pp_enabled
                        and not self.parallel_dims.pp_dynamic_shape
                        else None
                    )
                    raw_batch = global_batch[
                        i : i + self.config.train.train_policy.mini_batch
                    ]
                    if fixed_length is None:
                        max_len = min(
                            self.config.policy.model_max_length,
                            self.data_packer.sft_compute_max_len(raw_batch),
                        )
                    else:
                        max_len = fixed_length

                    if self.seq_len_multiple > 1:
                        max_len = (
                            (max_len + self.seq_len_multiple - 1)
                            // self.seq_len_multiple
                            * self.seq_len_multiple
                        )

                    packing_seq = self.config.train.sequence_packing
                    if packing_seq:
                        if self.parallel_dims.pp_enabled:
                            packing_seq = False
                            logger.debug(
                                "[Policy] Packing sequence is disabled due to incompatible dimensions."
                            )

                    batch = self.data_packer.sft_collate_fn(
                        raw_batch,
                        computed_max_len=max_len,
                        pad_token_id=self.tokenizer.pad_token_id,
                        ignore_label_id=-100,
                    )

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

                    self.model.train()
                    for k, v in batch.items():
                        batch[k] = (
                            v.to(self.device) if isinstance(v, torch.Tensor) else v
                        )

                    labels = batch.pop("label_ids")

                    position_ids, input_ids, pos_seq_dim = self.model.get_position_ids(
                        **batch
                    )

                    batch["position_ids"] = position_ids
                    padding_mask = batch.get("padding_mask", None)

                    if packing_seq:
                        # Prepare for the sequence packing information.
                        packed_args = pack_sequences_info_collect(
                            batch["input_ids"],
                            pad_token_id=self.tokenizer.pad_token_id,
                            label_ids=labels,
                            ignore_label_id=-100,
                            seq_len_multiple=self.seq_len_multiple,
                        )
                        batch.update(packed_args)
                        labels = pack_sequences_for_labels(
                            labels, batch["valid_input_len"]
                        )
                        packed_args = pack_sequences_for_masks(
                            batch["valid_input_len"], batch["valid_input_len"]
                        )
                        batch.update(packed_args)
                    # For VLMs, we need to delay the slice of inputs for CP until after the embedding generation in the model forward.
                    delay_cp_slice_inputs = getattr(
                        self.model, "delay_cp_slice_inputs", False
                    )
                    if (
                        self.parallel_dims.cp_enabled
                        and not packing_seq
                        and not delay_cp_slice_inputs
                    ):
                        [input_ids, position_ids, padding_mask] = (
                            slice_inputs_for_ulysses(
                                [input_ids, position_ids, padding_mask],
                                self.parallel_dims.mesh["cp"],
                                seq_dims=[1, pos_seq_dim, 1],
                            )
                        )

                        batch["input_ids"] = input_ids
                        batch["position_ids"] = position_ids
                        if padding_mask is not None:
                            batch["padding_mask"] = padding_mask

                    if self.parallel_dims.cp_enabled:
                        # Slice for cp after embedding generation and sequence packing in the model forward later.
                        batch["cp_mesh"] = self.parallel_dims.mesh["cp"]

                    if self.parallel_dims.pp_enabled:
                        pp_last_stage = (
                            self.parallel_dims.pp_coord[0]
                            == self.parallel_dims.pp_coord[1] - 1
                        )
                        pp_first_stage = self.parallel_dims.pp_coord[0] == 0

                        # Pipeline Parallel forward / backward inside step() call
                        targets, losses = (
                            (labels, []) if pp_last_stage else (None, None)
                        )
                        if pp_first_stage:
                            self.pp_scheduler.step(
                                **batch,
                                pp_dynamic_shape_enabled=self.parallel_dims.pp_dynamic_shape_enabled,
                                seq_len_multiple=self.seq_len_multiple,
                            )
                        else:
                            # FWD + BWD if it is 1F1B-like scheduler
                            self.pp_scheduler.step(
                                position_ids=batch["position_ids"],
                                target=targets,
                                losses=losses,
                                pp_dynamic_shape_enabled=self.parallel_dims.pp_dynamic_shape_enabled,
                                seq_len_multiple=self.seq_len_multiple,
                            )
                        loss = (
                            torch.mean(torch.stack(losses)).to(self.device)
                            if pp_last_stage
                            else torch.tensor([-1.0], device=self.device)
                        )
                    else:
                        # This code is just for debugging purposes, where we can test whether the model can generate tokens correctly
                        # last_token_ids = []
                        # with torch.no_grad():
                        #     N_NEW_TOKENS = 100
                        #     for _ in range(N_NEW_TOKENS):
                        #         if len(last_token_ids) > 0:
                        #             batch["input_ids"] = torch.cat(
                        #                 [batch["input_ids"], last_token_ids[-1]],
                        #                 dim=-1,
                        #             )
                        #             position_ids, _, _ = (
                        #                 self.model.get_position_ids(**batch)
                        #             )
                        #             batch["position_ids"] = position_ids

                        #         logits = self.model(**batch)
                        #         token_ids = torch.argmax(logits[:, -1:, :], dim=-1)
                        #         last_token_ids.append(token_ids)
                        #     if self.global_rank == 0:
                        #         text = ''
                        #         new_last_token_ids = torch.cat(last_token_ids, dim=-1).squeeze(0)
                        #         logger.info(f'{new_last_token_ids=}')
                        #         text = self.tokenizer.decode(new_last_token_ids)
                        #         logger.info(
                        #             f"generated tokens at sample : {text}"
                        #         )
                        # return
                        #########################################################################################

                        with self.act_offloading_ctx_manager:
                            logits = self.model(**batch)

                        loss = self.loss_fn(
                            logits,
                            labels,
                            output_packing_mask=batch.get("input_packing_mask", None),
                            target_packing_mask=batch.get("label_packing_mask", None),
                            loss_scaling_factor=1.0 / len(mini_batch_begin_idxs),
                        )
                        # # Hint FSDP to do all-reduce on the last backward pass
                        # if hasattr(self.model, "set_is_last_backward"):
                        #     print(f"set_is_last_backward: {i == mini_batch_begin_idxs[-1]}")
                        #     self.model.set_is_last_backward(i == mini_batch_begin_idxs[-1])
                        loss.backward()
                    acc_loss += loss.detach()

                """
                Compute the global grad norm on all parameters and then apply
                gradient clipping using the global grad norm.
                """
                all_params = [
                    p
                    for m in [model for model in self.model_parts if model is not None]
                    for p in m.parameters()
                ]
                grad_norm = dist_util.gradient_norm_clipping(
                    all_params,
                    self.config.train.optm_grad_norm_clip,
                    foreach=True,
                    pp_mesh=(
                        self.parallel_dims.mesh["pp"]
                        if self.parallel_dims.pp_enabled
                        else None
                    ),
                    return_norm_only=(self.config.train.optm_grad_norm_clip <= 0.0),
                )

                self.optimizers.step()
                self.lr_schedulers.step()

                self.train_step += 1

                # Early stop only when max_num_steps is specified
                if (
                    self.config.train.max_num_steps is not None
                    and self.train_step >= self.total_steps
                ):
                    break

                end_event.record()

                if (
                    self.parallel_dims.dp_replicate_enabled
                    or self.parallel_dims.dp_shard_enabled
                    or self.parallel_dims.cp_enabled
                ):
                    global_avg_loss, global_max_loss = (  # noqa: F841
                        dist_util.dist_mean(acc_loss, self.parallel_dims.mesh["dp_cp"]),
                        dist_util.dist_max(acc_loss, self.parallel_dims.mesh["dp_cp"]),
                    )
                else:
                    global_avg_loss = global_max_loss = acc_loss.item()  # noqa: F841

                if self.config.logging.logger:
                    if util.is_master_rank(self.parallel_dims, self.global_rank):
                        # Calculate last iteration time
                        assert end_event.query()
                        iter_time = (
                            start_event.elapsed_time(end_event) / 1000.0
                        )  # in seconds

                        report_data = {
                            "train/iteration_time": iter_time,
                            "train/loss_avg": global_avg_loss,
                            "train/loss_max": global_max_loss,
                            "train/learning_rate": self.lr_schedulers.get_last_lr()[0],
                            "train/grad_norm": (
                                grad_norm if grad_norm is not None else -1
                            ),
                        }

                        # FIXME(dinghaoy): only compute MFU of rank 0, if enable tp or pp,
                        # it will be inaccurate. Need a reduce for all the metrics.
                        if self.config.logging.report_mfu:
                            mfu = util.compute_mfu(
                                model=self.model,
                                n_tokens=np.prod(input_ids.shape),
                                iter_time=iter_time,
                                num_gpus=self.world_size,
                                dtype=self.config.train.param_dtype,
                            )
                            for k, v in mfu.items():
                                report_data[f"train/{k}"] = v
                        if (
                            "wandb" in self.config.logging.logger
                            and is_wandb_available()
                        ):
                            log_wandb(
                                data=report_data,
                                step=self.train_step,
                            )
                        if "console" in self.config.logging.logger:
                            logger.info(
                                f"Step: {self.train_step}/{self.total_steps}, Loss: {global_avg_loss:.5f}, Grad norm: {grad_norm:.5f}, Learning rate: {self.lr_schedulers.get_last_lr()[0]:.5e}, Iteration time: {iter_time:.2f}s."
                            )

                # For profiling
                self.profiler.step()

                val_score = None
                # validation
                if self.train_step % self.config.validation.freq == 0:
                    val_score = self.validate()

                # save checkpoint
                if (
                    self.config.train.ckpt.enable_checkpoint
                    and self.train_step % self._save_freq == 0
                    and self.train_step > 0
                    and self.parallel_dims.dp_replicate_coord[0] == 0
                ):
                    # TODO(dinghaoy): support export safetensors asynchronously.
                    if self.config.train.ckpt.export_safetensors:
                        logger.info(
                            f"Saving huggingface checkpoint at step {self.train_step} to {self.config.train.output_dir}..."
                        )
                        self.export_safetensors(
                            output_dir=self.config.train.output_dir,
                            rel_path=os.path.join(
                                "safetensors",
                                f"step_{self.train_step}",
                            ),
                            trainable_only=False,
                            dtype=util.str2torch_dtype(self.config.train.param_dtype),
                        )
                    logger.info(
                        f"Saving cosmos checkpoint at step {self.train_step}..."
                    )
                    self.ckpt_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizers,
                        scheduler=self.lr_schedulers,
                        step=self.train_step,
                        total_steps=self.total_steps,
                    )
                    self.ckpt_manager.save_check(
                        step=self.train_step,
                        val_score=val_score,
                        pp_enabled=self.parallel_dims.pp_enabled,
                        pp_last_stage=pp_last_stage,
                        pp_master_rank=self.parallel_dims.world_size
                        - self.parallel_dims.world_size / self.parallel_dims.pp,
                    )
            if (
                self.config.train.max_num_steps is not None
                and self.train_step >= self.total_steps
            ):
                break  # break outer epoch loop

        # process the final step
        val_score = self.validate()
        if (
            self.config.train.ckpt.export_safetensors
            and self.parallel_dims.dp_replicate_coord[0] == 0
        ):
            logger.info(
                f"Saving final huggingface checkpoint to {self.config.train.output_dir}..."
            )
            self.export_safetensors(
                output_dir=self.config.train.output_dir,
                rel_path=os.path.join(
                    "safetensors",
                    f"step_{self.train_step}",
                ),
                trainable_only=False,
                is_final=True,
                dtype=util.str2torch_dtype(self.config.train.param_dtype),
            )
        if self.config.train.ckpt.enable_checkpoint:
            logger.info(
                f"Training finished at step {self.train_step}/{self.total_steps}, saving final cosmos checkpoint..."
            )
            self.ckpt_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizers,
                scheduler=self.lr_schedulers,
                step=self.train_step,
                total_steps=self.total_steps,
                is_final=True,
            )
            self.ckpt_manager.save_check(
                step=self.train_step,
                val_score=val_score if self.config.validation.enable else -1,
                pp_enabled=self.parallel_dims.pp_enabled,
                pp_last_stage=pp_last_stage,
                pp_master_rank=self.parallel_dims.world_size
                - self.parallel_dims.world_size / self.parallel_dims.pp,
            )
        self.unregister_from_controller()

    @property
    def pp_loss_fn(self):
        # calculate the loss scaling factor
        # mini_batch_size = max(self.config.train.train_policy.mini_batch or 1, 1)
        mini_batch_size = self.config.policy.parallelism.pp_micro_batch_size
        mini_batch_size = min(
            mini_batch_size, self.config.train.train_batch_per_replica
        )
        loss_scaling_factor = (
            mini_batch_size / self.config.train.train_batch_per_replica
        )
        if self.parallel_dims.dp_shard_enabled:
            dp_group = self.parallel_dims.mesh["dp_shard"].get_group()
        else:
            dp_group = None

        if self.parallel_dims.cp_enabled:
            cp_group = self.parallel_dims.mesh["cp"].get_group()
        else:
            cp_group = None

        return torch.compile(
            partial(
                async_safe_ce,
                loss_scaling_factor=loss_scaling_factor,
                dp_group=dp_group,
                cp_group=cp_group,
            )
        )
        )
