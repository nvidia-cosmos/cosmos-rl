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
Utilities for handling async functions.
"""

import asyncio
import concurrent.futures
from typing import Callable, Any


def is_async_callable(func: Callable) -> bool:
    """
    Check if a function is async, including decorated functions.

    This function recursively checks the __wrapped__ attribute chain
    to find the original function, which is useful when the function
    is decorated by wrappers like @torch.no_grad() that use functools.wraps.

    Args:
        func: The function to check

    Returns:
        True if the function (or its wrapped original) is async, False otherwise

    Examples:
        >>> async def my_async_func():
        ...     pass
        >>> is_async_callable(my_async_func)
        True

        >>> @torch.no_grad()
        ... async def decorated_async_func():
        ...     pass
        >>> is_async_callable(decorated_async_func)
        True

        >>> def my_sync_func():
        ...     pass
        >>> is_async_callable(my_sync_func)
        False
    """
    # First check the function itself
    if asyncio.iscoroutinefunction(func):
        return True

    # Check if it has a __wrapped__ attribute (added by functools.wraps)
    # and recursively check the wrapped function
    if hasattr(func, "__wrapped__"):
        return is_async_callable(func.__wrapped__)

    return False


def nested_run_until_complete(func: Callable, *args, **kwargs) -> Any:
    """
    Run an async function to completion from synchronous code, with support for nested event loops.

    This function enables calling async functions from sync code, even when already inside
    a running event loop. It automatically detects the context and handles two scenarios:

    1. No running event loop: Creates a new loop and runs the coroutine directly
    2. Running event loop exists: Spawns a thread pool to run the coroutine in an isolated loop

    Args:
        func: The function to call (can be sync or async)
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The return value of the function

    Raises:
        Any exception raised by the called function

    Examples:
        >>> async def fetch_data():
        ...     await asyncio.sleep(0.1)
        ...     return "data"
        >>>
        >>> # From sync context
        >>> result = nested_run_until_complete(fetch_data)
        >>> print(result)  # "data"
        >>>
        >>> # From async context (nested call: async -> sync -> async)
        >>> async def outer():
        ...     def sync_wrapper():
        ...         return nested_run_until_complete(fetch_data)
        ...     return sync_wrapper()
        >>> asyncio.run(outer())  # "data"

    Note:
        When called from within a running event loop, this function automatically uses a
        thread pool to avoid "event loop already running" errors. This allows the
        async -> sync -> async call pattern to work correctly without deadlocks.
    """
    if not is_async_callable(func):
        return func(*args, **kwargs)

    try:
        # Check if there's a running event loop in the current thread
        asyncio.get_running_loop()
        # If we get here, there's a running loop - use a thread to avoid conflicts
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_in_new_loop, func, *args, **kwargs)
            return future.result()
    except RuntimeError:
        # No running event loop - safe to create one in the current thread
        return _run_in_new_loop(func, *args, **kwargs)


def _run_in_new_loop(func: Callable, *args, **kwargs) -> Any:
    """
    Helper function to run an async function in a new, isolated event loop.

    Uses asyncio.Runner() context manager to create a fresh event loop,
    run the coroutine to completion, and properly clean up resources.
    This approach is the recommended way in Python 3.11+ for managing
    event loop lifecycle.

    Args:
        func: The async function to call
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The return value of the async function
    """
    with asyncio.Runner() as runner:
        return runner.run(func(*args, **kwargs))
