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

from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import ASPECT_RATIO_1024_BIN


ASPECT_RATIO_480_BIN = {
    "0.5": [448.0, 896.0],
    "0.57": [480.0, 832.0],
    "0.68": [528.0, 768.0],
    "0.78": [560.0, 720.0],
    "1.0": [624.0, 624.0],
    "1.13": [672.0, 592.0],
    "1.29": [720.0, 560.0],
    "1.46": [768.0, 528.0],
    "1.67": [816.0, 496.0],
    "1.75": [832.0, 480.0],
    "2.0": [896.0, 448.0],
}

aspect_ratio_mapping = {
    "ASPECT_RATIO_1024_BIN": ASPECT_RATIO_1024_BIN,
    "ASPECT_RATIO_480_BIN": ASPECT_RATIO_480_BIN,
}


def get_ratio_bin(ratio_bin):
    if ratio_bin not in aspect_ratio_mapping.keys():
        raise ValueError(f"{ratio_bin} not in aspect_ratio_mapping, please add")
    else:
        return aspect_ratio_mapping[ratio_bin]
