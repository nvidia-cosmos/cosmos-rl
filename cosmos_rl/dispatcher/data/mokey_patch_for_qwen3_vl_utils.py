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


import torch
import numpy as np
import qwen_vl_utils.vision_process as qwen_vl_vision_process

from typing import Dict, Any, Union, List
from PIL import Image
from qwen_vl_utils.vision_process import (
    SPATIAL_MERGE_SIZE,
    VIDEO_MIN_TOKEN_NUM,
    VIDEO_MAX_TOKEN_NUM,
    VIDEO_READER_BACKENDS,
    get_video_reader_backend,
    MAX_NUM_WORKERS_FETCH_VIDEO,
    ceil_by_factor,
    MODEL_SEQ_LEN,
    FRAME_FACTOR,
    fetch_image,
    smart_resize,  # original smart_resize from qwen_vl_utils
)
from concurrent.futures import ThreadPoolExecutor
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
    smart_resize as hf_smart_resize,
)
from cosmos_rl.utils.logging import logger


def fetch_video(
    ele: Dict[str, Any],
    image_patch_size: int = 14,
    return_video_sample_fps: bool = False,
    return_video_metadata: bool = False,
) -> Union[torch.Tensor, List[Image.Image]]:
    logger.info("LMS: in patched function.")
    image_factor = image_patch_size * SPATIAL_MERGE_SIZE
    VIDEO_FRAME_MIN_PIXELS = VIDEO_MIN_TOKEN_NUM * image_factor * image_factor
    VIDEO_FRAME_MAX_PIXELS = VIDEO_MAX_TOKEN_NUM * image_factor * image_factor
    if isinstance(ele["video"], str):
        video_reader_backend = get_video_reader_backend()
        try:
            video, video_metadata, sample_fps = VIDEO_READER_BACKENDS[
                video_reader_backend
            ](ele)
        except Exception as e:
            logger.warning(
                f"video_reader_backend {video_reader_backend} error, use torchvision as default, msg: {e}"
            )
            video, video_metadata, sample_fps = VIDEO_READER_BACKENDS["torchvision"](
                ele
            )
    else:
        # The input is a list of frames
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        # use ThreadPoolExecutor to parallel process frames
        max_workers = min(MAX_NUM_WORKERS_FETCH_VIDEO, len(ele["video"]))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    fetch_image, {"image": video_element, **process_info}, image_factor
                )
                for video_element in ele["video"]
            ]
            image_list = [future.result() for future in futures]

        nframes = ceil_by_factor(len(image_list), FRAME_FACTOR)
        if len(image_list) < nframes:
            image_list.extend([image_list[-1]] * (nframes - len(image_list)))

        sample_fps = ele.get("sample_fps", 2.0)
        video = torch.stack(
            [
                torch.from_numpy(np.array(image).transpose(2, 0, 1))
                for image in image_list
            ]
        )

        # fake video metadata
        raw_fps = process_info.pop("raw_fps", sample_fps)
        video_metadata = dict(
            fps=raw_fps,
            frames_indices=[i for i in range(len(video))],
            total_num_frames=(nframes / sample_fps) * raw_fps,
        )
    nframes, _, height, width = video.shape
    min_pixels = ele.get("min_pixels", VIDEO_FRAME_MIN_PIXELS)

    # logger.info(f"LMS: video_ele: {ele}")
    total_pixels = ele.get(
        "total_pixels", MODEL_SEQ_LEN * image_factor * image_factor * 0.9
    )
    max_pixels = max(
        min(VIDEO_FRAME_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR),
        int(min_pixels * 1.05),
    )
    max_pixels_supposed = ele.get("max_pixels", max_pixels)
    if max_pixels_supposed > max_pixels:
        logger.warning(
            f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}]."
        )
    max_pixels = min(max_pixels_supposed, max_pixels)
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=image_factor,
        )
    else:
        # Patch 2: Using smart_resize inside transformers instead.
        resized_height, resized_width = hf_smart_resize(
            nframes,
            height,
            width,
            temporal_factor=FRAME_FACTOR,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels_supposed,  # Patch 1: Using user specified max_pixels like `update_processor_pixels`
        )
    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()

    final_video = (video, video_metadata) if return_video_metadata else video
    if return_video_sample_fps:
        return final_video, sample_fps
    return final_video


def apply_patch_to_fetch_video():
    qwen_vl_vision_process.fetch_video = fetch_video
