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

import asyncio
import unittest

from cosmos_rl.utils.async_utils import is_async_callable


class TestAsyncUtils(unittest.TestCase):
    """Test suite for AsyncUtils class."""

    def setUp(self):
        """Set up test fixtures."""

        # warmup the event loop
        async def async_dummy_func():
            pass

        asyncio.run(async_dummy_func())

    def test_is_async_callable(self):
        """Test is_async_callable function."""

        async def async_func():
            return 1

        self.assertTrue(is_async_callable(async_func))
        self.assertFalse(is_async_callable(lambda: 1))
