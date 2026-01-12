# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

import os
import pickle
from collections.abc import Iterable
from functools import partial
from typing import Callable, Optional

import webdataset as wds
from webdataset import filters
from webdataset.handlers import reraise_exception

from cosmos_rl.utils.logging import logger
from cosmos_rl.tools.dataset.wfm.webdataset.augmentors import (
    AUGMENTORS_CLS_MAPPING,
)
from cosmos_rl.tools.dataset.wfm.webdataset.config.schema import (
    AugmentorConfig,
    DatasetConfig,
    DatasetInfo,
    TarSample,
    Wdinfo,
)
from cosmos_rl.tools.dataset.wfm.webdataset.utils.iterators import (
    WebDataset,
)
from cosmos_rl.tools.dataset.wfm.webdataset.utils.misc import (
    remove_extensions_from_keys,
    skip_keys,
    update_url,
)
from cosmos_rl.utils.wfm.distributed import get_world_size
from cosmos_rl.utils.wfm.io.object_store import ObjectStore


def get_augmentor(aug_conf: AugmentorConfig):
    aug_cls = AUGMENTORS_CLS_MAPPING.get(aug_conf["type"], None)
    if aug_cls is None:
        raise ValueError(f"Unknown augmentor type: {aug_conf['type']}")
    new_conf = {k: v for k, v in aug_conf.items() if k != "type"}
    return aug_cls(**new_conf)


def wrap_augmentor_func_as_generator(func: Callable, data: Iterable):
    for data_dict in data:
        data_dict_out = func(data_dict)
        if data_dict_out is None:
            # Skip "unhealthy" samples
            continue
        yield data_dict_out


class Dataset:
    def __init__(
        self,
        config: DatasetConfig,
        handler: Callable = reraise_exception,
    ):
        r"""Webdataloader class

        Args:
            config: Dataset config
            world_size: Total number of GPUs
        """
        super().__init__()

        self.config = config

        self.world_size = get_world_size()

        dataset_info = config.dataset_info
        self.streaming_download = config.streaming_download

        self.s3_client = dict()
        self.bucket = dict()
        self.data_keys = config.keys

        # Parse the metadata
        self.wdinfo = Wdinfo(tar_files=[], total_key_count=0, chunk_size=0)
        self.parse_dataset_info(dataset_info=dataset_info)
        self.handler = handler
        self.augmentors = dict()

    def parse_dataset_info(self, dataset_info: list[DatasetInfo]):
        r"""Parse metadata about the list of tar files.

        Args:
            dataset_info (list): List of dictionaries containing paths to metadata files.
        """

        for dset_num, dset_info in enumerate(dataset_info):
            # For each dataset, we parse the file paths and store them as a list of TarSample.
            # TarSample will then be used by each worker to load the data.

            use_object_store = dset_info.object_store_config.enabled
            self.use_object_store = use_object_store
            dset_id = "dset: {}".format(dset_num)
            if use_object_store:
                object_store_reader = ObjectStore(
                    config_object_storage=dset_info.object_store_config
                )

                # Create PBSS config if data is loaded from PBSS
                bucket_dset = dset_info.object_store_config.bucket
                s3_client_dset = object_store_reader.client
                self.s3_client[dset_id] = s3_client_dset
                self.bucket[dset_id] = bucket_dset

            # Read all wdinfo files and obtain the DataSample list
            for wdinfo_path in dset_info.wdinfo:
                if use_object_store:
                    if not object_store_reader.object_exists(wdinfo_path):
                        raise FileNotFoundError(f"{wdinfo_path} not found")
                    cur_dset_info = object_store_reader.load_object(
                        key=wdinfo_path, type="json"
                    )  # type: ignore
                else:
                    with open(wdinfo_path, "rb") as fp:
                        cur_dset_info = pickle.load(fp)

                if not hasattr(self.config, "sample_keys_full_list_path"):
                    # Remind the user to add the sample_keys_full_list_path to the config
                    logger.warning("sample_keys_full_list_path not found in config;")

                if getattr(self.config, "sample_keys_full_list_path", None):
                    logger.info(
                        f"Loading sample_keys_full_list_paths for {wdinfo_path}"
                    )

                    #  Exists the information about the sample keys for each tar file
                    sample_keys_full_list_per_tar = []
                    for sample_keys_full_list_path in cur_dset_info["data_list"]:
                        sample_keys_full_list_path = os.path.join(
                            cur_dset_info["root"],
                            self.config.sample_keys_full_list_path,
                            sample_keys_full_list_path.replace(".tar", ".parquet"),
                        )
                        sample_keys_full_list_per_tar.append(sample_keys_full_list_path)
                else:
                    logger.info(
                        f"Skip loading sample_keys_full_list_paths for {wdinfo_path}"
                    )
                    sample_keys_full_list_per_tar = [None] * len(
                        cur_dset_info["data_list"]
                    )

                data_root = cur_dset_info["root"]
                tar_files_list = cur_dset_info["data_list"]
                tar_files = [
                    TarSample(
                        path=tar_file,
                        root=data_root,
                        keys=(
                            dset_info.per_dataset_keys
                            if dset_info.per_dataset_keys
                            else self.data_keys
                        ),  # use per dataset keys if available
                        meta=dset_info,
                        dset_id=dset_id,
                        sample_keys_full_list=sample_keys_full_list,
                    )
                    for tar_file, sample_keys_full_list in zip(
                        tar_files_list, sample_keys_full_list_per_tar, strict=True
                    )
                ]

                # Update the master winfo
                self.wdinfo.tar_files.extend(tar_files)
                self.wdinfo.total_key_count += cur_dset_info["total_key_count"]
                self.wdinfo.chunk_size = cur_dset_info["chunk_size"]

    @staticmethod
    # This is the function that calls each augmentor in sequence.
    def augmentor_fn(data, augmentations):
        # Build augmentor chain
        for aug_fn in augmentations:
            # Use generator function as augmentor
            # (recommended, allows skipping or replicating samples inside the augmentor)
            if getattr(aug_fn, "is_generator", False):
                data = aug_fn(data)
            else:  # Use regular function as augmentor (backward compatibility)
                data = wrap_augmentor_func_as_generator(aug_fn, data)
        yield from data

    def build_data_augmentor(
        self, augmentor_cfg: dict[str, AugmentorConfig]
    ) -> Callable:
        r"""Function for building data augmentors from augmentor config."""
        augmentations = []
        for aug in augmentor_cfg.keys():
            augmentations.append(get_augmentor(augmentor_cfg[aug]))

        # This is the function that calls each augmentor in sequence.
        return partial(Dataset.augmentor_fn, augmentations=augmentations)

    def build_dataset(self, **kwargs) -> WebDataset:
        tar_list = self.wdinfo.tar_files
        num_tars = len(tar_list)
        assert num_tars > 0, "Did not find any data."

        shuffle_buffer_size = getattr(
            self.config, "buffer_size", self.wdinfo.chunk_size
        )

        # update distributor urls and chunk size
        distributor_fn = self.config.distributor

        distributor_fn.set_urls(tar_list)
        distributor_fn.set_chunk_size(self.wdinfo.chunk_size)

        dataset = WebDataset(
            distributor_fn,
            load_from_object_store=self.use_object_store,
            s3_client=self.s3_client,
            s3_bucket_name=self.bucket,
            streaming_download=self.streaming_download,
            handler=self.handler,
        )

        # Creating a shuffle buffer
        if shuffle_buffer_size > 0:
            dataset.append(wds.shuffle(shuffle_buffer_size))

        # Adding decoders
        # Decoders are functions that decode the input IO stream
        decoder_list = getattr(self.config, "decoders", [])
        decoder_functions = []
        for decoder in decoder_list:
            # If the specified decoder is a string, use the webdataset decoder
            # If its a callable function, use the defined function to decode data
            assert isinstance(decoder, str) or callable(
                decoder
            ), "Decoder should either be callable or a str"
            decoder_functions.append(decoder)
        dataset.append(wds.decode(*decoder_functions))

        # After the decoders are added, remove extension from the keys
        # Extensions in the data keys are needed for auto-detection of decoders in webdataset.
        if self.config.remove_extension_from_keys:
            dataset.append(remove_extensions_from_keys)

        # Function to skip keys
        dataset.append(skip_keys)
        # Building augmentors
        augmentor_cfg = getattr(self.config, "augmentation", None)
        augmentation_fn = self.build_data_augmentor(augmentor_cfg)
        dataset.append(augmentation_fn)

        # Updates URL names so that the collate function can handle
        dataset.append(update_url)

        dataset.total_images = self.wdinfo.total_key_count  # type: ignore
        logger.info("Total number of training shards: %d" % num_tars)
        logger.info("Total training key count: %d" % dataset.total_images)  # type: ignore

        return dataset


class WebDataLoaderDataset(Dataset):
    def __init__(
        self,
        config: DatasetConfig,
        handler: Callable = reraise_exception,
        decoder_handler: Optional[Callable] = None,
        detshuffle: bool = False,
    ):
        r"""Webdataloader class

        Args:
            config: Dataset config
            handler (Callable): Error handler for webdataset class
            decoder_handler (Callable): Error handler during decoding
        """
        super().__init__(config=config, handler=handler)
        self.decoder_handler = decoder_handler
        self.detshuffle = detshuffle

    def build_dataset(self, **kwargs) -> WebDataset:
        r"""
        Build the dataset object.
        The function only diffs from BaseDataset.build_dataset by only adding the decoder_handler to the WebDataset object.
        """
        tar_list = self.wdinfo.tar_files
        num_tars = len(tar_list)
        assert num_tars > 0, "Did not find any data."

        shuffle_buffer_size = getattr(
            self.config, "buffer_size", self.wdinfo.chunk_size
        )

        # update distributor urls and chunk size
        distributor_fn = self.config.distributor

        distributor_fn.set_urls(tar_list)
        distributor_fn.set_chunk_size(self.wdinfo.chunk_size)

        dataset = WebDataset(
            distributor_fn,
            load_from_object_store=self.use_object_store,
            s3_client=self.s3_client,
            s3_bucket_name=self.bucket,
            streaming_download=self.streaming_download,
            handler=self.handler,
        )

        # Creating a shuffle buffer
        if self.detshuffle:
            dataset.append(filters.detshuffle(shuffle_buffer_size))
        else:
            dataset.append(wds.shuffle(shuffle_buffer_size))

        # Adding decoders
        # Decoders are functions that decode the input IO stream
        decoder_list = getattr(self.config, "decoders", [])
        decoder_functions = []
        for decoder in decoder_list:
            # If the specified decoder is a string, use the webdataset decoder
            # If its a callable function, use the defined function to decode data
            assert isinstance(decoder, str) or callable(
                decoder
            ), "Decoder should either be callable or a str"
            decoder_functions.append(decoder)
        dataset.append(wds.decode(*decoder_functions, handler=self.decoder_handler))

        # After the decoders are added, remove extension from the keys
        # Extensions in the data keys are needed for auto-detection of decoders in webdataset.
        if self.config.remove_extension_from_keys:
            dataset.append(remove_extensions_from_keys)

        # Function to skip keys
        dataset.append(skip_keys)
        # Building augmentors
        augmentor_cfg = getattr(self.config, "augmentation", None)
        augmentation_fn = self.build_data_augmentor(augmentor_cfg)
        dataset.append(augmentation_fn)

        # Updates URL names so that the collate function can handle
        dataset.append(update_url)

        dataset.total_images = self.wdinfo.total_key_count  # type: ignore
        logger.info("Total number of training shards: %d" % num_tars)
        logger.info("Total training key count: %d" % dataset.total_images)  # type: ignore

        return dataset
