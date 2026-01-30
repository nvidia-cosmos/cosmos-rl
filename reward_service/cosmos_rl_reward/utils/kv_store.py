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

from collections import OrderedDict

from cosmos_rl.utils.logging import logger


class RecentDict:
    """
    A dictionary that keeps track of the most recent items up to a maximum size.
    When the maximum size is exceeded, the oldest items are removed.
    It acts like an LRU (Least Recently Used) cache based simple KV store.
    """

    def __init__(self, max_size=100000):
        self.max_size = max_size
        self.dict = OrderedDict()

    def __setitem__(self, key, value):
        # If key exists, delete it first (re-insertion = update)
        if key in self.dict:
            del self.dict[key]
        # Add new item
        self.dict[key] = value
        # Evict oldest if over capacity
        if len(self.dict) > self.max_size:
            self.dict.popitem(last=False)  # last=False => pop first (oldest)

    def __getitem__(self, key):
        # Accessing item: move to end (most recent)
        value = self.dict.pop(key)
        self.dict[key] = value
        return value

    def __delitem__(self, key):
        del self.dict[key]

    def __len__(self):
        return len(self.dict)

    def __contains__(self, key):
        return key in self.dict

    def get(self, key, default=None):
        if key in self.dict:
            return self.__getitem__(key)
        return default

    def keys(self):
        return self.dict.keys()

    def values(self):
        return self.dict.values()

    def items(self):
        return self.dict.items()

    def __repr__(self):
        return f"RecentDict({list(self.dict.items())})"


class SimpleKeyValueStore:
    """
    A simple key-value store that uses RecentDict to keep track of the most recent items.
    Replaces Redis when Redis is not configured and needed for testing or lightweight use cases.
    """

    def __init__(self):
        self.store = RecentDict()

    async def set_key_value(self, key: str, value: str):
        self.store[key] = value
        logger.info(f"[SimpleKeyValueStore] Set key {key} with value {value}")

    async def get_key_value(self, key: str):
        return self.store.get(key, None)
