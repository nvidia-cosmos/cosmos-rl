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

from cosmos_rl.tools.dataset.wfm.webdataset.distributors.basic import (
    ShardlistBasic,
)
from cosmos_rl.tools.dataset.wfm.webdataset.distributors.multi_aspect_ratio import (
    ShardlistMultiAspectRatio,
)
from cosmos_rl.tools.dataset.wfm.webdataset.distributors.multi_aspect_ratio_v2 import (
    ShardlistMultiAspectRatioInfinite,
)

distributors_list = {
    "basic": ShardlistBasic,
    "multi_aspect_ratio": ShardlistMultiAspectRatio,
    "multi_aspect_ratio_infinite": ShardlistMultiAspectRatioInfinite,
}
