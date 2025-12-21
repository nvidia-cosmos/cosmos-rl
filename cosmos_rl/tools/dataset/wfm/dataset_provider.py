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


from typing import Callable, Optional
from webdataset.handlers import warn_and_continue

from cosmos_rl.utils.logging import logger
from cosmos_rl.tools.dataset.wfm.augmentor_provider import (
    AUGMENTOR_OPTIONS,
)
import cosmos_rl.tools.dataset.wfm.webdataset.decoders.image as image_decoders
import cosmos_rl.tools.dataset.wfm.webdataset.decoders.pickle as pickle_decoders
import cosmos_rl.tools.dataset.wfm.webdataset.distributors as distributors
import cosmos_rl.tools.dataset.wfm.webdataset.distributors.parallel_sync_multi_aspect_ratio as parallel_sync_multi_aspect_ratio
import cosmos_rl.tools.dataset.wfm.video_decoder as video_decoder
from cosmos_rl.tools.dataset.wfm.webdataset.webdataset import (
    WebDataLoaderDataset,
)
from cosmos_rl.tools.dataset.wfm.webdataset.config.schema import (
    DatasetConfig,
)
from cosmos_rl.tools.dataset.wfm.utils import (
    IMAGE_RES_SIZE_INFO,
    VIDEO_RES_SIZE_INFO,
)
from cosmos_rl.tools.dataset.wfm.data_sources.data_registration import (
    DATASET_OPTIONS,
)

from cosmos_rl.utils.wfm.distributed import get_global_parallel_dims


def get_video_dataset(
    dataset_name: str,
    video_decoder_name: str,
    resolution: str,
    is_train: bool = True,
    num_video_frames: int = 121,
    chunk_size: int = 0,
    min_fps_thres: int = 10,
    max_fps_thres: int = 60,
    dataset_resolution_type: str = "all",
    augmentor_name: str = "video_basic_augmentor_v1",
    object_store: Optional[str] = "s3",
    caption_type: str = "t2w_qwen2p5_7b",
    embedding_type: str = "t5_xxl",
    detshuffle: bool = False,
    long_caption_ratio: int = 7,
    medium_caption_ratio: int = 2,
    short_caption_ratio: int = 1,
    user_caption_ratio: int = 90,
    use_native_fps: bool = False,
    dataset_info_fn: Callable = None,
    **kwargs,
):
    assert (
        resolution in VIDEO_RES_SIZE_INFO.keys()
    ), "The provided resolution cannot be found in VIDEO_RES_SIZE_INFO."
    assert object_store in [
        "s3",
        "swiftstack",
        False,
    ], "We support s3 and swiftstack only, or False for local loading."
    basic_augmentor_names = [
        "video_basic_augmentor_v2",
        "video_basic_augmentor_v2_with_control",
    ]
    if video_decoder_name == "video_naive_bytes":
        assert (
            augmentor_name in basic_augmentor_names
        ), "We can only use video_basic_augmentor_v2 with video_naive_bytes decoder."
    if augmentor_name in basic_augmentor_names:
        assert (
            video_decoder_name == "video_naive_bytes"
        ), "We can only use video_naive_bytes decoder with video_basic_augmentor_v2."

    assert (
        dataset_resolution_type
        in [
            "all",
            "gt720p",
            "gt1080p",
        ]
    ), f"The provided dataset resolution type {dataset_resolution_type} is not supported."
    # dataset_resolution_type
    # -- all - uses all dataset resolutions
    # -- gt720p - Uses only resolutions >= 720p
    # -- gt1080p - Uses only resolutions >= 1080p
    if not object_store:
        assert (
            dataset_info_fn is not None
        ), "dataset_info_fn is required for local loading."
        dataset_info = dataset_info_fn()
    else:
        dataset_info_fn = DATASET_OPTIONS[dataset_name]
        dataset_info = dataset_info_fn(
            object_store, caption_type, embedding_type, dataset_resolution_type
        )  # type: ignore
    augmentor = AUGMENTOR_OPTIONS[augmentor_name](
        resolution=resolution,
        caption_type=caption_type,
        embedding_type=embedding_type,
        min_fps=min_fps_thres,
        max_fps=max_fps_thres,
        long_caption_ratio=long_caption_ratio,
        medium_caption_ratio=medium_caption_ratio,
        short_caption_ratio=short_caption_ratio,
        user_caption_ratio=user_caption_ratio,
        num_video_frames=num_video_frames,
        use_native_fps=use_native_fps,
    )

    global_parallelism = get_global_parallel_dims()
    assert global_parallelism is not None, "global_parallelism not initialized"
    _, cp_world_size = global_parallelism.cp_coord
    _, tp_world_size = global_parallelism.tp_coord
    if cp_world_size > 1 or tp_world_size > 1:
        logger.critical(
            f"Using parallelism size CP :{cp_world_size}, TP :{tp_world_size} for video dataset, switch to ShardlistMultiAspectRatioParallelSync distributor"
        )
        distributor = (
            parallel_sync_multi_aspect_ratio.ShardlistMultiAspectRatioParallelSync(
                shuffle=True,
                split_by_node=True,
                split_by_worker=True,
                resume_flag=True,
                verbose=True,
                is_infinite_loader=is_train,
            )
        )
        detshuffle = True  # overwrite detshuffle.
    else:
        distributor = distributors.ShardlistMultiAspectRatio(
            shuffle=True,
            split_by_node=True,
            split_by_worker=True,
            resume_flag=True,
            verbose=False,
            is_infinite_loader=is_train,
        )

    video_data_config = DatasetConfig(
        keys=[],  # use the per_dataset_keys in DatasetInfo instead
        buffer_size=100,
        streaming_download=True,
        dataset_info=dataset_info,
        distributor=distributor,
        decoders=[
            video_decoder.construct_video_decoder(
                video_decoder_name=video_decoder_name,
                sequence_length=num_video_frames,
                chunk_size=chunk_size,
                min_fps_thres=min_fps_thres,
                max_fps_thres=max_fps_thres,
            ),
            pickle_decoders.pkl_decoder,
        ],
        augmentation=augmentor,
        remove_extension_from_keys=True,
        sample_keys_full_list_path=None,
    )

    return WebDataLoaderDataset(
        config=video_data_config,
        decoder_handler=warn_and_continue,
        detshuffle=detshuffle,
    )


def get_image_dataset(
    dataset_name: str,
    resolution: str,
    dataset_resolution_type: str = "all",
    is_train: bool = True,
    augmentor_name: str = "image_basic_augmentor",
    object_store: str = "s3",
    detshuffle: bool = False,
    caption_type: str = "ai_v3p1",
    embedding_type: str = "t5_xxl",
    **kwargs,
):
    assert (
        resolution in IMAGE_RES_SIZE_INFO.keys()
    ), "The provided resolution cannot be found in IMAGE_RES_SIZE_INFO."
    assert object_store in ["s3", "swiftstack"], "We support s3 and swiftstack only."
    assert (
        dataset_resolution_type
        in [
            "all",
            "gt720p",
            "gt1080p",
        ]
    ), f"The provided dataset resolution type {dataset_resolution_type} is not supported."
    # dataset_resolution_type
    # -- all - uses all dataset resolutions
    # -- gt720p - Uses only resolutions >= 720p
    # -- gt1080p - Uses only resolutions >= 1080p
    dataset_info_fn = DATASET_OPTIONS[dataset_name]
    dataset_info = dataset_info_fn(
        object_store, caption_type, embedding_type, dataset_resolution_type
    )
    augmentation = AUGMENTOR_OPTIONS[augmentor_name](
        resolution=resolution,
        caption_type=caption_type,
        embedding_type=embedding_type,
    )

    global_parallelism = get_global_parallel_dims()
    assert global_parallelism is not None, "global_parallelism not initialized"
    _, cp_world_size = global_parallelism.cp_coord
    _, tp_world_size = global_parallelism.tp_coord

    if cp_world_size > 1 or tp_world_size > 1:
        logger.critical(
            f"Using parallelism size CP :{cp_world_size}, TP :{tp_world_size} for video dataset, switch to ShardlistMultiAspectRatioParallelSync distributor"
        )
        distributor = (
            parallel_sync_multi_aspect_ratio.ShardlistMultiAspectRatioParallelSync(
                shuffle=True,
                split_by_node=True,
                split_by_worker=True,
                resume_flag=True,
                verbose=True,
                is_infinite_loader=is_train,
            )
        )
        detshuffle = True  # overwrite detshuffle.
    else:
        distributor = distributors.ShardlistMultiAspectRatio(
            shuffle=True,
            split_by_node=True,
            split_by_worker=True,
            resume_flag=True,
            verbose=False,
            is_infinite_loader=is_train,
        )

    image_data_config = DatasetConfig(
        keys=[],
        # TODO: (qsh 2025-03-26) reduce buffer size from 100 to 25 to prevent oom
        buffer_size=25,
        streaming_download=True,
        dataset_info=dataset_info,
        distributor=distributor,
        decoders=[
            image_decoders.pil_loader,
            pickle_decoders.pkl_decoder,
        ],
        augmentation=augmentation,
    )

    return WebDataLoaderDataset(config=image_data_config, detshuffle=detshuffle)
