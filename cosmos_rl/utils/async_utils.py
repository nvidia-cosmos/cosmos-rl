"""
Utilities for handling async functions.
"""

import asyncio
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
