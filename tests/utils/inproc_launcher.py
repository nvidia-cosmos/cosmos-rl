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


"""
This tool launches multiple subprocesses within a single process and runs a user-provided function, it is useful for debugging distributed jobs.
The user function should initialize distributed communication, typically using torch.distributed.init_process_group with env://.
The function signature should be:
    fn(local_rank, world_size, *fn_args, **fn_kwargs)

Parameters:
- fn: User-provided function
- world_size: World size
- fn_args: Arguments for the user function
- fn_kwargs: Keyword arguments for the user function
- start_method: Launch method
- master_addr: Master address
- master_port: Master port
- extra_env: Additional environment variables
"""

from __future__ import annotations

import os
import socket
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import torch.multiprocessing as mp


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def _set_dist_env(
    *,
    rank: int,
    world_size: int,
    local_rank: int,
    master_addr: str,
    master_port: int,
    extra_env: Mapping[str, str] | None,
) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    if extra_env:
        os.environ.update(extra_env)


def _worker_entry(
    local_rank: int,
    world_size: int,
    fn: Callable[..., Any],
    fn_args: Sequence[Any],
    fn_kwargs: MutableMapping[str, Any],
    master_addr: str,
    master_port: int,
    extra_env: Mapping[str, str] | None,
) -> None:
    rank = local_rank
    _set_dist_env(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        master_addr=master_addr,
        master_port=master_port,
        extra_env=extra_env,
    )
    fn(local_rank, world_size, *fn_args, **fn_kwargs)


def launch_inproc(
    fn: Callable[..., Any],
    *,
    world_size: int,
    fn_args: Iterable[Any] = (),
    fn_kwargs: Mapping[str, Any] | None = None,
    start_method: str = "spawn",
    master_addr: str = "127.0.0.1",
    master_port: int | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> None:
    """Launch processes in-proc and run a user-provided function per rank.

    The user function should initialize distributed communication internally,
    typically via torch.distributed.init_process_group using env://.
    The function signature is expected to be:
        fn(local_rank, world_size, *fn_args, **fn_kwargs)

    Args:
        fn: User-provided function
        world_size: World size
        fn_args: Arguments for the user function
        fn_kwargs: Keyword arguments for the user function
        start_method: Launch method, available methods: "fork", "spawn", "forkserver"
        master_addr: Master address
        master_port: Master port
        extra_env: Additional environment variables
    """
    if world_size < 1:
        raise ValueError("world_size must be >= 1")

    resolved_kwargs: MutableMapping[str, Any] = dict(fn_kwargs or {})
    resolved_port = master_port if master_port is not None else _find_free_port()
    resolved_args = tuple(fn_args)

    mp.start_processes(
        _worker_entry,
        args=(
            world_size,
            fn,
            resolved_args,
            resolved_kwargs,
            master_addr,
            resolved_port,
            extra_env,
        ),
        nprocs=world_size,
        join=True,
        start_method=start_method,
    )


