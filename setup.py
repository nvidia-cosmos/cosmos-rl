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

import os
from setuptools import setup
import setuptools


with open(os.path.join(os.path.dirname(__file__), "requirements.txt")) as f:
    requirements = f.read().splitlines()

setup(
    name="cosmos_rl",
    version="0.1.1",
    packages=setuptools.find_packages(),
    package_data={
        "cosmos_rl": ["launcher/*.sh"]
    },
    entry_points={
        "console_scripts": [
            "cosmos-rl = cosmos_rl.launcher.launch_all:main",
        ],
    },
    install_requires=requirements,
    zip_safe=False,
)
