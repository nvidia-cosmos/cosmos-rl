# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""A bounded-lived, idempotent publication of one logical dispatch.

One immutable plan may contain multiple rollout and command streams. A lost
reply retries the SAME plan, never regenerates a dispatch. A pending marker
after a script error is terminal: Redis scripts isolate execution but do not
roll back preceding writes. This is not durable queue/checkpoint recovery.
"""

from dataclasses import dataclass, field, replace
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import time
import uuid

import msgpack


PUBLICATION_LUA = """
local server_id = string.match(redis.call('INFO', 'server'), 'run_id:(%x+)')
if server_id ~= ARGV[5] then
    return redis.error_reply('Publication server restarted; refusing ambiguous replay')
end
local previous = redis.call('GET', KEYS[1])
local digest = ARGV[2]
if previous then
    if string.sub(previous, 1, 64) ~= digest then
        return redis.error_reply('Publication identity reused with different content')
    end
    if string.sub(previous, 66, 70) ~= 'done:' then
        return redis.error_reply('Publication incomplete; refusing ambiguous replay')
    end
    return string.sub(previous, 71)
end
local now = redis.call('TIME')
local remaining = tonumber(ARGV[1]) - (tonumber(now[1]) + tonumber(now[2]) / 1000000)
if remaining <= 0 then
    return redis.error_reply('Publication expired; refusing late replay')
end
for i = 2, #KEYS do
    local kind = redis.call('TYPE', KEYS[i]).ok
    if kind ~= 'none' and kind ~= 'stream' then
        return redis.error_reply('Publication target is not a stream')
    end
end
-- Reserve BEFORE appending. If any later command errors, the surviving pending
-- marker prevents a retry from duplicating the already-appended prefix.
redis.call('SET', KEYS[1], digest .. ':pending', 'PX', math.ceil((remaining + 60) * 1000))
local ids = {}
for i = 2, #KEYS do
    local arg = 6 + (i - 2) * 2
    ids[#ids + 1] = redis.call('XADD', KEYS[i], 'MAXLEN', ARGV[3], '*', ARGV[arg], ARGV[arg + 1], 'timestamp', ARGV[4])
end
local encoded = cjson.encode(ids)
redis.call('SET', KEYS[1], digest .. ':done:' .. encoded, 'KEEPTTL')
return encoded
"""


@dataclass(frozen=True)
class PublicationPlan:
    # Entries are (stream name, field name, already-serialized immutable bytes).
    entries: tuple[tuple[str, str, bytes], ...]
    deadline: float
    operation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    server_id: str = ""

    def __post_init__(self):
        if not math.isfinite(self.deadline) or not self.operation_id:
            raise ValueError("Publication requires a finite deadline and identity")
        if not isinstance(self.entries, tuple) or not self.entries:
            raise ValueError("Publication requires an immutable nonempty entry tuple")
        for entry in self.entries:
            if (
                not isinstance(entry, tuple)
                or len(entry) != 3
                or not isinstance(entry[0], str)
                or not entry[0]
                or not isinstance(entry[1], str)
                or not entry[1]
                or not isinstance(entry[2], bytes)
            ):
                raise ValueError("Invalid immutable publication entry")

    @classmethod
    def create(cls, entries, *, timeout_s=30):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Publication timeout must be finite and positive")
        return cls(tuple(entries), time.time() + timeout_s)

    @property
    def key(self):
        return f"cosmos:publication:{self.operation_id}"

    def for_server(self, server_id):
        if not server_id or (self.server_id and self.server_id != server_id):
            raise ValueError("Publication server identity cannot change")
        return replace(self, server_id=server_id)

    @property
    def digest(self):
        return hashlib.sha256(
            msgpack.packb((self.deadline, self.timestamp, self.entries, self.server_id))
        ).hexdigest()

    def publish(self, client, *, maxlen):
        """Attempt this plan; callers own bounded I/O/retry and terminal failure.

        Complete repeats return the original IDs. Incomplete, expired or changed
        plans raise Redis ResponseError and must not be treated as retryable.
        The marker outlives the publication deadline, so its eventual eviction
        cannot authorize a late duplicate. Ordinary delivery retains stream order.
        """
        if type(maxlen) is not int or maxlen <= 0:
            raise ValueError("Stream retention must be a positive integer")
        if not self.server_id:
            raise ValueError("Publication must be bound to a Redis server incarnation")
        if any(
            count > maxlen
            for count in Counter(entry[0] for entry in self.entries).values()
        ):
            raise ValueError(
                "Publication exceeds stream retention; refusing to trim its own payload"
            )
        keys = [self.key, *(entry[0] for entry in self.entries)]
        arguments = [self.deadline, self.digest, maxlen, self.timestamp, self.server_id]
        for _, name, data in self.entries:
            arguments.extend((name, data))
        return json.loads(client.eval(PUBLICATION_LUA, len(keys), *keys, *arguments))
