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
import os
import warnings

from typing import Callable


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


def unsafe_enable_nest_asyncio():
    """
    Unsafely enable nest_asyncio. This should only be used in rare cases where you
    absolutely need to run async code from sync code.

    However, current python implementation does not support this, (maybe in the future it will support this).
    The `nest_asyncio` patches `asyncio` module, allow nested use of `asyncio.run` and `loop.run_until_complete`.

    If the blocking API is not needed, or `nest_asyncio` cause unexpected issues, you can disable it by setting
    the environment variable `DISABLE_NEST_ASYNCIO` to `True`.
    """

    flag = os.getenv("DISABLE_NEST_ASYNCIO")
    if flag == "1" or flag == "True":
        return

    warnings.warn(
        "nest_asyncio is enabled, this may cause unexpected issues, "
        "you can set the environment variable DISABLE_NEST_ASYNCIO to 1 or True to disable it.",
    )

    # skip if already enabled
    if hasattr(unsafe_enable_nest_asyncio, "done") and unsafe_enable_nest_asyncio.done:
        return

    import nest_asyncio

    nest_asyncio.apply()
    unsafe_enable_nest_asyncio.done = True


unsafe_enable_nest_asyncio.done = False
