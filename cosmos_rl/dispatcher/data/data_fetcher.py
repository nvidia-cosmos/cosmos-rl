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

from typing import Optional, Callable, List, Dict, Iterator, Tuple, Any
from itertools import islice
import math
from tqdm import tqdm
from abc import ABC

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from cosmos_rl.dispatcher.data.packer.base import BaseDataPacker
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


class DataFetcherBase(ABC):
    """
    DataFetcherBase is the base class for all data fetchers.
    """

    def __init__(
        self,
        config: Config,
        data_packer: BaseDataPacker,
        val_data_packer: BaseDataPacker,
        dataset: Optional[Callable[[Config], Dataset]] = None,
        val_dataset: Optional[Callable[[Config], Dataset]] = None,
        is_rl: bool = True,
    ):
        self.config = config
        self.data_packer = data_packer
        self.val_data_packer = val_data_packer
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.is_rl = is_rl

    def load_dataset(self):
        if self.dataset is not None and isinstance(self.dataset, Callable):
            self.dataset = self.dataset(self.config)
        if self.val_dataset is not None and isinstance(self.val_dataset, Callable):
            self.val_dataset = self.val_dataset(self.config)

    def query_reference_answer(
        self, prompt_idx: int, dataset_type: str = "train"
    ) -> Any:
        """
        Query the reference answer from the dataset based on the prompt index.
        Args:
            prompt_idx (int): The index of the prompt in the dataset.
            dataset_type (str): The type of the dataset, either "train" or "val".
        Returns:
            Any: The reference answer corresponding to the prompt index.
        """
        if self.dataset is None:
            raise ValueError("Dataset is not loaded")
        if self.config.validation.enable and self.val_dataset is None:
            raise ValueError("Validation dataset is not loaded")

        if dataset_type == "train":
            return self.dataset.train_set.get_reference_answer(prompt_idx)
        elif dataset_type == "val":
            return self.val_dataset.val_set.get_reference_answer(prompt_idx)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")


class ControllerDataFetcher(DataFetcherBase):
    """
    ControllerDataFetcher is responsible for fetching data from the dataset for policy and rollout.
    """

    def __init__(
        self,
        config: Config,
        dataset: Optional[Callable[[Config], Dataset]] = None,
        val_dataset: Optional[Callable[[Config], Dataset]] = None,
        sampler: Optional[Callable] = None,
        batch_sampler: Optional[Callable] = None,
        val_sampler: Optional[Callable] = None,
        val_batch_sampler: Optional[Callable] = None,
        is_rl: bool = True,
    ):
        # ControllerDataFetcher doesn't need data packer.
        super().__init__(
            config,
            None,
            None,
            dataset,
            val_dataset,
            is_rl,
        )

        self.ckpt_extra_info = {}
        self.epoch = 1
        self.remain_samples_num = -1
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.val_sampler = val_sampler
        self.val_batch_sampler = val_batch_sampler

        # Controller should always load the dataset and dataloader.
        self.load_dataset()

    def load_dataset(self):
        """
        Load the dataset and dataloader for epochs.
        """
        super().load_dataset()

        remain_samples_num = 0
        if self.is_rl:
            self.rollout_batch_size = (
                self.config.train.train_policy.dataloader_batch_size
                or self.config.rollout.batch_size
            )
            if self.dataset is not None:
                assert isinstance(self.dataset, Dataset)
                self.dataset = CosmosDataset(config=self.config, train_set=self.dataset)
                logger.info(
                    "[Controller] Using provided dataset for training, dataset specification in the toml config will be ignored"
                )
            else:
                self.dataset = CosmosDataset(config=self.config)

            remain_samples_num = (
                (
                    len(self.dataset.train_set)
                    * self.config.rollout.n_generation
                    * self.config.train.epoch
                )
                if self.dataset is not None
                else 0
            )  # Total number of samples of policy training will consume.

            if self.sampler is not None:
                logger.info("[DataFetcher] Using provided sampler for training")
                if isinstance(self.sampler, Callable):
                    train_sampler = self.sampler(
                        self.dataset.train_set,
                        num_replicas=1,
                        rank=0,
                        shuffle=self.config.train.train_policy.dataloader_shuffle,
                        drop_last=False,
                    )
                else:
                    train_sampler = self.sampler
            else:
                train_sampler = DistributedSampler(
                    self.dataset.train_set,
                    num_replicas=1,
                    rank=0,
                    shuffle=self.config.train.train_policy.dataloader_shuffle,
                    drop_last=False,
                )
            if self.batch_sampler is not None and isinstance(
                self.batch_sampler, Callable
            ):
                self.batch_sampler = self.batch_sampler(
                    train_sampler,
                    batch_size=self.rollout_batch_size,
                    drop_last=False,
                )
            if self.config.train.resume:
                try:
                    # If resuming, disable the weight sync check flag for rollout to compare the received weight with the reference weight.
                    PolicyToRolloutUnicastCommand._do_weight_sync_check_flag = False
                    self.ckpt_manager = CheckpointMananger(self.config)
                    self.ckpt_extra_info = (
                        self.ckpt_manager.load_extra_info_from_checkpoint()
                    )
                    remain_samples_num = self.ckpt_extra_info.get(
                        "remain_samples_num", remain_samples_num
                    )
                    self.epoch = (
                        self.config.train.epoch
                        - (
                            math.ceil(
                                remain_samples_num
                                / (
                                    len(self.dataset.train_set)
                                    * self.config.rollout.n_generation
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
                                    remain_samples_num
                                    / self.config.rollout.n_generation
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
                    if self.batch_sampler is not None:
                        self.batch_sampler = SkippingSampler(
                            base_sampler=self.batch_sampler,
                            skip_samples=train_dataloader_bias
                            // (
                                len(list(islice(iter(self.batch_sampler), 1))[0])
                                if isinstance(
                                    list(islice(iter(self.batch_sampler), 1))[0], list
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
            if self.batch_sampler is not None:
                logger.info(
                    "[DataFetcher] Using custom batch Sampler that yields list of indices for training dataset."
                )
                self.train_dataloader = DataLoader(
                    self.dataset.train_set,
                    num_workers=self.config.train.train_policy.dataloader_num_workers,
                    prefetch_factor=self.config.train.train_policy.dataloader_prefetch_factor,
                    collate_fn=RLPayload.collate_fn,
                    batch_sampler=self.batch_sampler,
                )
            else:
                self.train_dataloader = DataLoader(
                    self.dataset.train_set,
                    batch_size=self.rollout_batch_size,
                    shuffle=False,
                    num_workers=self.config.train.train_policy.dataloader_num_workers,
                    prefetch_factor=self.config.train.train_policy.dataloader_prefetch_factor,
                    collate_fn=RLPayload.collate_fn,
                    sampler=train_sampler,
                )
            self.train_dataloader_iter = iter(self.train_dataloader)

            if self.config.validation.enable:
                self.val_batch_size = (
                    self.config.train.train_policy.dataloader_batch_size
                    or self.config.validation.batch_size
                    or self.rollout_batch_size
                )
                assert (
                    self.val_batch_size > 0
                ), "[DataFetcher] val_batch_size should be greater than 0."
                if self.val_dataset is not None:
                    assert isinstance(self.val_dataset, Dataset)
                    self.val_dataset = CosmosValidationDataset(
                        config=self.config,
                        val_set=self.val_dataset,
                    )
                    logger.info(
                        "[DataFetcher] Using provided validation dataset for validation, dataset specification in the toml config will be ignored"
                    )
                else:
                    self.val_dataset = CosmosValidationDataset(config=self.config)
                if self.val_sampler is not None:
                    logger.info("[DataFetcher] Using provided sampler for validation")
                    if isinstance(self.val_sampler, Callable):
                        self.val_sampler = self.val_sampler(
                            self.val_dataset.val_set,
                            num_replicas=1,
                            rank=0,
                            shuffle=False,
                            drop_last=False,
                        )

                if self.val_batch_sampler is not None:
                    logger.info(
                        "[DataFetcher] Using custom batch Sampler that yields list of indices for validation dataset."
                    )
                    if isinstance(self.val_batch_sampler, Callable):
                        self.val_batch_sampler = self.val_batch_sampler(
                            self.val_sampler
                            if self.val_sampler is not None
                            else DistributedSampler(
                                self.val_dataset.val_set,
                                num_replicas=1,
                                rank=0,
                                shuffle=False,
                                drop_last=False,
                            ),
                            batch_size=self.val_batch_size,
                            drop_last=False,
                        )
                        self.val_dataloader = DataLoader(
                            self.val_dataset.val_set,
                            num_workers=self.config.train.train_policy.dataloader_num_workers,
                            prefetch_factor=self.config.train.train_policy.dataloader_prefetch_factor,
                            collate_fn=RLPayload.collate_fn,
                            batch_sampler=self.val_batch_sampler,
                        )
                else:
                    self.val_dataloader = DataLoader(
                        self.val_dataset.val_set,
                        batch_size=self.val_batch_size,
                        shuffle=False,
                        num_workers=self.config.train.train_policy.dataloader_num_workers,
                        prefetch_factor=self.config.train.train_policy.dataloader_prefetch_factor,
                        collate_fn=RLPayload.collate_fn,
                        sampler=self.val_sampler,
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
    ) -> Tuple[List[RLPayload], bool]:
        add_answer = (
            self.config.rollout.multi_turn_config.enable
            or not self.config.train.local_dataset
        )
        # query n prompts from the dataset [idx, payload]
        payloads_list: List[RLPayload] = []
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
                    payload.conversation = None
                    if not self.config.rollout.multi_turn_config.enable:
                        # For non-multi-turn rollout, we set reference answer to None.
                        payload.reference_answer = None

                payloads_list.append(payload)

        return payloads_list, is_end

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

    def clear_validation_status(self):
        self.activated_val_iter = None
        if self.activated_val_tqdm is not None:
            self.activated_val_tqdm.clear()
        self.activated_val_tqdm = None


class WorkerDataFetcher(DataFetcherBase):
    """
    WorkerDataFetcher is responsible for fetching data locally for policy and rollout, according to the index returned by the controller.
    WorkerDataFetcher is much more simpler than ControllerDataFetcher, because it only supports query data by index.
    """

    def __init__(
        self,
        config: Config,
        data_packer: BaseDataPacker,
        val_data_packer: BaseDataPacker,
        dataset: Optional[Callable[[Config], Dataset]] = None,
        val_dataset: Optional[Callable[[Config], Dataset]] = None,
        is_rl: bool = True,
    ):
        super().__init__(
            config,
            data_packer,
            val_data_packer,
            dataset,
            val_dataset,
            is_rl,
        )

        if self.config.train.local_dataset:
            self.load_dataset()

    def load_dataset(self):
        super().load_dataset()

        if self.dataset is not None:
            assert isinstance(self.dataset, Dataset)
            self.dataset = CosmosDataset(config=self.config, train_set=self.dataset)
            logger.info(
                "[DataFetcher] Using provided dataset for training, dataset specification in the toml config will be ignored"
            )
        else:
            self.dataset = CosmosDataset(config=self.config)

        if self.config.validation.enable:
            if self.val_dataset is not None:
                assert isinstance(self.val_dataset, Dataset)
                self.val_dataset = CosmosValidationDataset(
                    config=self.config,
                    val_set=self.val_dataset,
                )
                logger.info(
                    "[DataFetcher] Using provided validation dataset for validation, dataset specification in the toml config will be ignored"
                )
            else:
                self.val_dataset = CosmosValidationDataset(config=self.config)

    def get_payload_by_index(
        self, index: int, is_validation: bool = False, attr: str = "prompt"
    ) -> RLPayload:
        row: IdxAndRLPayload = None
        if is_validation:
            if self.val_dataset is None or not self.config.validation.enable:
                raise ValueError(
                    "[DataFetcher] Validation dataset is not loaded or validation is not enabled"
                )
            row = self.val_dataset.val_set[index]
        else:
            if self.dataset is None:
                raise ValueError("[DataFetcher] Local dataset is not loaded")
            row = self.dataset.train_set[index]
        return getattr(row[1], attr)
