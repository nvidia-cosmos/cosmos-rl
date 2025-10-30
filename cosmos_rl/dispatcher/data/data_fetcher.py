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

from typing import Optional, Callable, List, Dict, Iterator, Tuple
from itertools import islice
import math
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoTokenizer

from cosmos_rl.dispatcher.data.packer.base import DataPacker
from cosmos_rl.policy.config import Config
from cosmos_rl.dispatcher.data import (
    CosmosDataset,
    RLPayload,
    CosmosValidationDataset,
)
from cosmos_rl.dispatcher.data import IdxAndRLPayload
from cosmos_rl.dispatcher.command import PolicyToRolloutUnicastCommand
from cosmos_rl.utils.checkpoint import CheckpointMananger
from cosmos_rl.utils.logging import logger


class DataFetcher:
    def __init__(
        self,
        config: Config,
        dataset: Optional[Dataset] = None,
        data_packer: Optional[DataPacker] = None,
        val_dataset: Optional[Dataset] = None,
        val_data_packer: Optional[DataPacker] = None,
        sampler: Optional[Callable] = None,
        batch_sampler: Optional[Callable] = None,
        val_sampler: Optional[Callable] = None,
        val_batch_sampler: Optional[Callable] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        is_rl: bool = False,
    ):
        self.config = config
        if dataset is not None and isinstance(dataset, Callable):
            dataset = dataset(config)
        if val_dataset is not None and isinstance(val_dataset, Callable):
            val_dataset = val_dataset(config)

        self.user_data_packer = data_packer
        self.user_val_data_packer = val_data_packer
        self.dataset = None
        self.val_dataset = None
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.val_sampler = val_sampler
        self.val_batch_sampler = val_batch_sampler
        self.tokenizer = tokenizer

        self.ckpt_extra_info = {}

        remain_samples_num = 0

        if is_rl:
            self.rollout_batch_size = (
                config.train.train_policy.dataloader_batch_size
                or config.rollout.batch_size
            )
            if dataset is not None:
                assert isinstance(dataset, Dataset)
                self.dataset = CosmosDataset(
                    config=config, train_set=dataset, tokenizer=self.tokenizer
                )
                logger.info(
                    "[Controller] Using provided dataset for training, dataset specification in the toml config will be ignored"
                )
            else:
                self.dataset = CosmosDataset(config=config, tokenizer=self.tokenizer)

            remain_samples_num = (
                (
                    len(self.dataset.train_set)
                    * config.rollout.n_generation
                    * config.train.epoch
                )
                if self.dataset is not None
                else 0
            )  # Total number of samples of policy training will consume.

            if sampler is not None:
                logger.info("[DataFetcher] Using provided sampler for training")
                if isinstance(sampler, Callable):
                    train_sampler = sampler(
                        self.dataset.train_set,
                        num_replicas=1,
                        rank=0,
                        shuffle=config.train.train_policy.dataloader_shuffle,
                        drop_last=False,
                    )
                else:
                    train_sampler = sampler
            else:
                train_sampler = DistributedSampler(
                    self.dataset.train_set,
                    num_replicas=1,
                    rank=0,
                    shuffle=config.train.train_policy.dataloader_shuffle,
                    drop_last=False,
                )
            if batch_sampler is not None and isinstance(batch_sampler, Callable):
                batch_sampler = batch_sampler(
                    train_sampler,
                    batch_size=self.rollout_batch_size,
                    drop_last=False,
                )
            if config.train.resume:
                try:
                    # If resuming, disable the weight sync check flag for rollout to compare the received weight with the reference weight.
                    PolicyToRolloutUnicastCommand._do_weight_sync_check_flag = False
                    self.ckpt_manager = CheckpointMananger(config)
                    self.ckpt_extra_info = (
                        self.ckpt_manager.load_extra_info_from_checkpoint()
                    )
                    remain_samples_num = self.ckpt_extra_info.get(
                        "remain_samples_num", remain_samples_num
                    )
                    self.epoch = (
                        config.train.epoch
                        - (
                            math.ceil(
                                remain_samples_num
                                / (
                                    len(self.dataset.train_set)
                                    * config.rollout.n_generation
                                )
                            )
                        )
                        + 1
                    )
                    logger.info(
                        f"[DataFetcher] Resuming from checkpoint, current epoch: {self.epoch}, remaining samples: {remain_samples_num}"
                    )

                    train_dataloader_bias = max(
                        0,
                        len(self.dataset.train_set)
                        - (
                            (
                                math.ceil(
                                    remain_samples_num / config.rollout.n_generation
                                )
                            )
                            % len(self.dataset.train_set)
                        ),
                    )
                    logger.info(
                        f"[DataFetcher] Loaded extra info from checkpoint: {self.ckpt_extra_info}"
                    )
                    from cosmos_rl.policy.trainer.sampler import SkippingSampler

                    train_sampler = SkippingSampler(
                        base_sampler=train_sampler,
                        skip_samples=train_dataloader_bias
                        // (
                            len(list(islice(iter(train_sampler), 1))[0])
                            if isinstance(list(islice(iter(train_sampler), 1))[0], list)
                            else 1
                        ),
                    )
                    if batch_sampler is not None:
                        batch_sampler = SkippingSampler(
                            base_sampler=batch_sampler,
                            skip_samples=train_dataloader_bias
                            // (
                                len(list(islice(iter(batch_sampler), 1))[0])
                                if isinstance(
                                    list(islice(iter(batch_sampler), 1))[0], list
                                )
                                else 1
                            ),
                        )
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    logger.error(
                        f"[DataFetcher] Failed to load checkpoint extra info: {e}. Please check the checkpoint path and config."
                    )
            if batch_sampler is not None:
                logger.info(
                    "[DataFetcher] Using custom batch Sampler that yields list of indices for training dataset."
                )
                self.train_dataloader = DataLoader(
                    self.dataset.train_set,
                    num_workers=config.train.train_policy.dataloader_num_workers,
                    prefetch_factor=config.train.train_policy.dataloader_prefetch_factor,
                    collate_fn=RLPayload.collate_fn,
                    batch_sampler=batch_sampler,
                )
            else:
                self.train_dataloader = DataLoader(
                    self.dataset.train_set,
                    batch_size=self.rollout_batch_size,
                    shuffle=False,
                    num_workers=config.train.train_policy.dataloader_num_workers,
                    prefetch_factor=config.train.train_policy.dataloader_prefetch_factor,
                    collate_fn=RLPayload.collate_fn,
                    sampler=train_sampler,
                )
            self.train_dataloader_iter = iter(self.train_dataloader)

            if config.validation.enable:
                self.val_batch_size = (
                    config.train.train_policy.dataloader_batch_size
                    or config.validation.batch_size
                    or self.rollout_batch_size
                )
                assert (
                    self.val_batch_size > 0
                ), "[DataFetcher] val_batch_size should be greater than 0."
                if val_dataset is not None:
                    assert isinstance(val_dataset, Dataset)
                    self.val_dataset = CosmosValidationDataset(
                        config=config, val_set=val_dataset, tokenizer=self.tokenizer
                    )
                    logger.info(
                        "[DataFetcher] Using provided validation dataset for validation, dataset specification in the toml config will be ignored"
                    )
                else:
                    self.val_dataset = CosmosValidationDataset(
                        config=config, tokenizer=self.tokenizer
                    )
                if val_sampler is not None:
                    logger.info("[DataFetcher] Using provided sampler for validation")
                    if isinstance(val_sampler, Callable):
                        val_sampler = val_sampler(
                            self.val_dataset.val_set,
                            num_replicas=1,
                            rank=0,
                            shuffle=False,
                            drop_last=False,
                        )
                    self.val_dataloader = DataLoader(
                        self.val_dataset.val_set,
                        num_workers=config.train.train_policy.dataloader_num_workers,
                        prefetch_factor=config.train.train_policy.dataloader_prefetch_factor,
                        collate_fn=RLPayload.collate_fn,
                        batch_sampler=val_batch_sampler,
                    )
                else:
                    self.val_dataloader = DataLoader(
                        self.val_dataset.val_set,
                        batch_size=self.val_batch_size,
                        shuffle=False,
                        num_workers=config.train.train_policy.dataloader_num_workers,
                        prefetch_factor=config.train.train_policy.dataloader_prefetch_factor,
                        collate_fn=RLPayload.collate_fn,
                        sampler=val_sampler,
                    )
            else:
                self.val_dataset = None
                self.val_dataloader = None
        else:
            self.val_dataset = None
            self.val_dataloader = None

        # validation
        self.val_datasize: Optional[int] = (
            0 if self.val_dataset is None else len(self.val_dataset.val_set)
        )
        self.val_iters: Dict[int, Iterator] = {}
        self.activated_val_iter: Optional[Iterator] = None
        self.activated_val_tqdm: Optional[tqdm] = None

        self.remain_samples_num = remain_samples_num

    def get_batched_prompt(
        self, n: int, validation_step: Optional[int] = None
    ) -> Tuple[bool, List[IdxAndRLPayload]]:
        add_answer = (
            self.config.rollout.multi_turn_config.enable
            or not self.config.rollout.reference_answer_in_local
        )
        # query n prompts from the dataset [idx, payload]
        prompt_id_and_payload_list: List[IdxAndRLPayload] = []
        is_end = False

        is_validation = validation_step is not None

        if is_validation:
            iterator = self.validation_get_dataloader(validation_step)
            batch_size = self.val_batch_size
        else:
            iterator = self.train_dataloader_iter
            batch_size = self.rollout_batch_size

        def _next_payload(
            iterator, add_answer: bool
        ) -> tuple[List[int], List[RLPayload]]:
            idxs, payloads = next(iterator)
            assert len(idxs) <= batch_size
            assert len(payloads) <= batch_size
            assert len(idxs) == len(payloads)
            updated_payloads: List[RLPayload] = []
            for idx, payload in zip(idxs, payloads):
                if add_answer:
                    if is_validation:
                        payload.reference_answer = (
                            self.val_dataset.val_set.get_reference_answer(idx)
                        )
                    else:
                        payload.reference_answer = (
                            self.dataset.train_set.get_reference_answer(idx)
                        )
                updated_payloads.append(payload)
            return idxs, updated_payloads

        for _ in range(math.ceil(n / batch_size)):
            payload: RLPayload | None = None
            try:
                idxs, payloads = _next_payload(iterator, add_answer)
            except StopIteration:
                if not is_validation:
                    self.epoch += 1
                    if self.epoch <= self.config.train.epoch:
                        logger.info(f"[Controller] Epoch {self.epoch} start.")
                        iterator = iter(self.train_dataloader)
                        self.train_dataloader_iter = iterator

                        idxs, payloads = _next_payload(iterator, add_answer)
                    else:
                        if self.epoch == self.config.train.epoch + 1:
                            # We only log this all finished information once.
                            logger.info(
                                "[Controller] All epochs finished fetching rollout prompts, wait for rollouts generation and training to complete."
                            )
                        is_end = True
                        break
                else:
                    is_end = True
                    break
            assert len(idxs) == len(payloads)
            for idx, payload in zip(idxs, payloads):
                idx = idx.item() if isinstance(idx, torch.Tensor) else idx
                if self.config.train.local_dataset:
                    # If local dataset is enabled, we set prompt to None. And rollout worker will query
                    # the prompt from local dataset.
                    payload.prompt = None
                prompt_id_and_payload_list.append((idx, payload))

        return prompt_id_and_payload_list, is_end

    def validation_activate_dataloader(self, validation_step: int):
        if validation_step not in self.val_iters:
            logger.info(
                f"[DataFetcher] Activating validation dataloader for step {validation_step}, with length {(self.val_datasize or len(self.val_dataloader))}"
            )
            self.val_iters[validation_step] = iter(self.val_dataloader)
            self.activated_val_iter = self.val_iters[validation_step]
            self.activated_val_tqdm = tqdm(
                desc="validation",
                total=(self.val_datasize or len(self.val_dataloader)),
            )

    def validation_get_dataloader(
        self, validation_step: Optional[int] = None
    ) -> Iterator:
        if validation_step is None:
            return self.activated_val_iter
        else:
            return self.val_iters[validation_step]

    def get_payload_by_index(
        self, index: int, is_validation: bool = False
    ) -> RLPayload:
        if is_validation:
            return self.val_dataset.val_set[index][1].prompt
        else:
            return self.dataset.train_set[index][1].prompt

    def clear_validation_status(self):
        self.activated_val_iter = None
        self.activated_val_tqdm.clear()
        self.activated_val_tqdm = None
