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

import pickle
import re
from typing import Optional


def pkl_decoder(key: str, data: bytes) -> Optional[dict]:
    r"""
    Function to decode a pkl file.
    Args:
        key: Data key.
        data: Data dict.
    """
    extension = re.sub(r".*[.]", "", key)
    if extension == "pkl" or extension == "pickle":
        data_dict = pickle.loads(data)
        return data_dict
    else:
        return None
