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

from cosmos_rl.tools.dataset.wfm.webdataset.augmentors.image.cropping import (
    CenterCrop,
    RandomCrop,
)
from cosmos_rl.tools.dataset.wfm.webdataset.augmentors.image.flip import (
    HorizontalFlip,
)
from cosmos_rl.tools.dataset.wfm.webdataset.augmentors.image.normalize import (
    Normalize,
)
from cosmos_rl.tools.dataset.wfm.webdataset.augmentors.image.padding import (
    ReflectionPadding,
)
from cosmos_rl.tools.dataset.wfm.webdataset.augmentors.image.resize import (
    ResizeSmallestSide,
    ResizeLargestSide,
    ResizeSmallestSideAspectPreserving,
    ResizeLargestSideAspectPreserving,
)
from cosmos_rl.tools.dataset.wfm.webdataset.augmentors.image.text_transform_for_image import (
    TextTransformForImage,
    TextTransformForImageWithoutEmbeddings,
)
from cosmos_rl.tools.dataset.wfm.webdataset.augmentors.image.append_fps_frames_for_image import (
    AppendFPSFramesForImage,
)
from cosmos_rl.tools.dataset.wfm.webdataset.augmentors.merge_datadict import (
    DataDictMerger,
)
from cosmos_rl.tools.dataset.wfm.webdataset.augmentors.video.text_transforms_for_video import (
    TextTransformForVideo,
)
from cosmos_rl.tools.dataset.wfm.webdataset.augmentors.video.video_parsing import (
    VideoParsing,
)

AUGMENTORS_CLS_MAPPING = {
    "center_crop": CenterCrop,
    "random_crop": RandomCrop,
    "horizontal_flip": HorizontalFlip,
    "normalize": Normalize,
    "reflection_padding": ReflectionPadding,
    "resize_smallest_side": ResizeSmallestSide,
    "resize_largest_side": ResizeLargestSide,
    "resize_smallest_side_aspect_ratio_preserving": ResizeSmallestSideAspectPreserving,
    "resize_largest_side_aspect_ratio_preserving": ResizeLargestSideAspectPreserving,
    "append_fps_frames_for_image": AppendFPSFramesForImage,
    "text_transform_for_image": TextTransformForImage,
    "text_transform_for_image_without_embeddings": TextTransformForImageWithoutEmbeddings,
    "text_transforms_for_video": TextTransformForVideo,
    "merge_datadict": DataDictMerger,
    "video_parsing": VideoParsing,
}
