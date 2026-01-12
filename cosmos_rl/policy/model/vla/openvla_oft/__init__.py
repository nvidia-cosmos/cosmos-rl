# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenVLA-OFT model implementation for cosmos-rl"""

from .configuration_prismatic import OpenVLAConfig
from .modeling_prismatic import OpenVLAForActionPrediction
from .processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

__all__ = [
    "OpenVLAConfig",
    "OpenVLAForActionPrediction",
    "PrismaticImageProcessor",
    "PrismaticProcessor",
]
