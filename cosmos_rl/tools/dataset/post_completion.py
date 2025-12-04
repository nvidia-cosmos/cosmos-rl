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

from typing import Optional, Any, List, Union
from torch.utils.data import Dataset
from cosmos_rl.launcher.worker_entry import main as launch_worker
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.dispatcher.data.packer import DecoderOnlyLLMDataPacker
from cosmos_rl.dispatcher.data.packer.multi_turn import (
    ChatMessage,
)
from cosmos_rl.tools.dataset.gsm8k_grpo import GSM8kDataset

import sqlite3
import json
import uuid


class SQLiteStorage:
    """
    Example mechanism to handle the data content transfer between replicas using SQLite storage.
    Each item is stored as a key-value pair, where the key is a string and the value is a JSON-serialized object.
    Behave as a simple key-value store to reduce the data transfer among the replicas in the framework.
    Only used for demonstration of how to use the post-completion data packer.
    This is just a simple example for demonstration purpose.
    In real-world scenarios, you need more efficient solutions to handle the data transfer between rollout and policy replicas, to put the actual data transfer outside of the framework and use only UUIDs inside.
    """

    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS items (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)

    def add_item(self, key, value):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO items (key, value) VALUES (?, ?)",
                (key, json.dumps(value)),
            )

    def delete_item(self, key):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("DELETE FROM items WHERE key = ?", (key,))
            if cur.rowcount == 0:
                raise KeyError(f"Key '{key}' not found")

    def get_item(self, key):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT value FROM items WHERE key = ?", (key,))
            row = cur.fetchone()
            if row:
                return json.loads(row[0])
            raise KeyError(f"Key '{key}' not found")


class PostCompletionSampleDataPacker(DecoderOnlyLLMDataPacker):
    """
    This is a demo data packer that shows how to post-process the completions from rollout engine.
    Policy replica will receive the post-processed completions for further processing.
    This is meaningless for this example, but useful for explaining:
    1. How to reduce the data transfer overhead by storing the actual completion content in an external storage or an external transformer (SQLite in this case) and only transfer the UUIDs in the framework.
    2. How to post-process the completions from rollout engine before sending and pre-process the completions before feeding into policy model.
    """

    def __init__(self, *args, **kwargs):
        # Base on DecoderOnlyLLMDataPacker
        # DecoderOnlyLLMDataPacker as the underlying data packer.
        super().__init__(*args, **kwargs)

    def setup(
        self,
        config: CosmosConfig,
        *args,
        **kwargs,
    ):
        """
        This method is optional and get called by launcher after being mounted
        `config`: config;
        """
        super().setup(config, *args, **kwargs)
        self.storage = SQLiteStorage("sample_data_post_completion_test.db")

    def get_policy_input(
        self,
        item: Any,
        rollout_output: Union[str, List[int]],
        n_ignore_prefix_tokens: int = 0,
    ) -> Any:
        """
        Process samples & rollout output before collating them into a mini-batch
        """

        if not isinstance(rollout_output, str) or rollout_output == "":
            # If rollout_output is empty string or not str, return directly
            # Since it is not stored in SQLiteStorage
            return super().get_policy_input(
                item, rollout_output, n_ignore_prefix_tokens
            )
        # rollout_output is the id stored in SQLiteStorage
        # transform it back to actual completion content
        id = rollout_output
        rollout_output = self.storage.get_item(id)
        self.storage.delete_item(id)
        return super().get_policy_input(item, rollout_output, n_ignore_prefix_tokens)

    def get_rollout_output(
        self,
        completions: Optional[List[Any]],
        completed_conversations: Optional[List[Any]],
        logprobs: Optional[List[Any]],
        token_ids: Optional[List[Any]],
        **kwargs,
    ) -> Optional[List[Any]]:
        """
        Post-process to get the rollout outputs from the rollout engine
        For example, we can clean up the completions here.
        """
        # Handle None or empty cases for completions and completed_conversations
        if completions is None:
            completions_uuids = None
        assert isinstance(completions, list)
        if completed_conversations is None:
            completed_conversations_uuids = None
        assert isinstance(completed_conversations, list)

        completions_uuids = []
        # process completions for normal cases to store in SQLiteStorage and replace with UUIDs
        if len(completions) > 0 and isinstance(completions[0], str):
            # first case : items should be List[str]
            for item in completions:
                assert isinstance(item, str)
                if item == "":
                    completions_uuids.append(item)
                else:
                    id = uuid.uuid4()
                    self.storage.add_item(str(id), item)
                    completions_uuids.append(str(id))
        elif len(completions) > 0 and (
            isinstance(completions[0], ChatMessage) or isinstance(completions[0], dict)
        ):
            # second case : items should be List[ChatMessage]
            for item in completions:
                if isinstance(item, dict):
                    assert "role" in item and "content" in item
                    if item["content"] == "":
                        # If actual content is empty, no need to store in SQLite
                        completions_uuids.append(item)
                        continue
                elif isinstance(item, ChatMessage):
                    if item.content == "":
                        # If actual content is empty, no need to store in SQLite
                        completions_uuids.append(item)
                        continue
                else:
                    raise ValueError("Invalid item type")
                id = uuid.uuid4()
                self.storage.add_item(str(id), item)
                completions_uuids.append(str(id))
        else:
            raise ValueError("Invalid item type in completions")

        # process completed_conversations if multi-turn setting is enabled to store in SQLiteStorage and replace with UUIDs
        completed_conversations_uuids = []
        for item in completed_conversations:
            assert isinstance(item, list)
            if len(item) == 0:
                # If actual content is empty, no need to store in SQLite
                completed_conversations_uuids.append(item)
                continue
            if all(
                isinstance(sub_item, ChatMessage) and sub_item.content == ""
                for sub_item in item
            ) or all(
                isinstance(sub_item, dict) and sub_item["content"] == ""
                for sub_item in item
            ):
                # If actual content is empty, no need to store in SQLite
                completed_conversations_uuids.append(item)
                continue
            # Check all sub_item are ChatMessage or dict
            assert all(
                isinstance(sub_item, ChatMessage) or isinstance(sub_item, dict)
                for sub_item in item
            )
            # Store the actual content in SQLite and return the UUID
            id = uuid.uuid4()
            self.storage.add_item(str(id), item)
            completed_conversations_uuids.append(str(id))

        # logprobs and token_ids not handled in this example, just return directly since normally they are None or empty if we do not enable decoupled_loss and enable rollout_as_token_ids.
        # Or the logprobs and token ids can be combined with the completions as a whole for each item and stored them together in SQLite as well.
        return (
            completions_uuids,
            completed_conversations_uuids,
            logprobs,
            token_ids,
            kwargs,
        )


if __name__ == "__main__":
    # Use GSM8k dataset as an example
    def get_dataset(config: CosmosConfig) -> Dataset:
        return GSM8kDataset()

    # This is just a demo showing how to use PostCompletionSampleDataPacker
    # The PostCompletionSampleDataPacker is used to demonstrate how to using rollout results post-processing and policy inputs pre-processing to store intermediate results in an external storage (SQLite in this case).
    # This is important for reducing the data content transfer overhead in the framework by change the data content to be transferred as UUIDs and do the actual data transfer in the background in any more efficient way.
    launch_worker(dataset=get_dataset, data_packer=PostCompletionSampleDataPacker())

    """
    Sample commands:
        cosmos-rl --config configs/qwen2-5/qwen2-5-7b-p-fsdp4-r-tp2-pp1-grpo.toml tools/dataset/post_completion.py
    """
