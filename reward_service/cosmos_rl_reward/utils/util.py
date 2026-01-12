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
import random
from typing import List, Callable, Union, Any
from cosmos_rl_reward.utils.logging import logger
import requests
import time
import os
from pathlib import Path


class CosmosHttpRetryConfig:
    max_retries: int = 60
    retries_per_delay: int = 5
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0


COSMOS_HTTP_RETRY_CONFIG = CosmosHttpRetryConfig()


def download_file(url, filepath):
    """
    Download file from URL to specified filepath
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)  # Create directories if needed

    with open(filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    logger.info(f"Downloaded: {filepath}")


def get_cosmos_rl_reward_cache_dir():
    """
    Get the cache directory for Cosmos RL Reward.
    """
    cosmos_rl_reward_cache_dir = os.path.expanduser(
        os.getenv("COSMOS_RL_REWARD_CACHE_DIR", "~/.cache/cosmos_rl_reward")
    )
    return cosmos_rl_reward_cache_dir


def status_check_for_response(response):
    """
    Handle the status code for the response.
    Raises an exception if the status code is not 200.
    """
    response.raise_for_status()


async def make_request_with_retry(
    requests: Union[Callable, List[Callable]],
    urls: List[str] = None,
    response_parser: Callable = None,
    max_retries: int = COSMOS_HTTP_RETRY_CONFIG.max_retries,
    retries_per_delay: int = COSMOS_HTTP_RETRY_CONFIG.retries_per_delay,
    initial_delay: float = COSMOS_HTTP_RETRY_CONFIG.initial_delay,
    max_delay: float = COSMOS_HTTP_RETRY_CONFIG.max_delay,
    backoff_factor: float = COSMOS_HTTP_RETRY_CONFIG.backoff_factor,
) -> Any:
    """
    Make an HTTP GET request with exponential backoff retry logic.

    Args:
        requests (List[Callable]): The functions to make the request in an alternative way
        urls (List[str]): List of host URLs to try
        response_parser (Callable): Function to parse the response
        max_retries (int): Maximum number of retry attempts
        retries_per_delay (int): Number of retries to attempt at each delay level
        initial_delay (float): Initial delay between retries in seconds
        max_delay (float): Maximum delay between retries in seconds
        backoff_factor (float): Factor to increase delay between retries

    Returns:
        Any: The response object from the successful request or redis client request.

    Raises:
        Exception: If all retry attempts fail
    """
    delay = initial_delay
    last_exception = None
    total_attempts = 0
    url_index = 0
    request_idx = 0

    if isinstance(requests, Callable):
        requests = [requests]

    while total_attempts < max_retries:
        # Try multiple times at the current delay level
        total_retries_cur_delay = 0
        while total_retries_cur_delay < retries_per_delay:
            try:
                request = requests[request_idx]
                if urls is not None:
                    url = urls[url_index]
                    r = await request(url)
                else:
                    url = None
                    r = await request()
                if response_parser is not None:
                    response_parser(r)
                return r

            except Exception as e:
                last_exception = e
                url_index += 1
                if url_index >= (1 if urls is None else len(urls)):
                    url_index = 0
                    request_idx += 1
                    if request_idx >= len(requests):
                        request_idx = 0
                        total_retries_cur_delay += 1
                        total_attempts += 1
                logger.info(
                    f"Request failed: {e}. Attempt {total_attempts} of {max_retries} for {request} on {url}."
                )
                if total_attempts >= max_retries:
                    break

                if request_idx != 0 or url_index != 0:
                    jitter = (1.0 + random.random()) * initial_delay
                    await asyncio.sleep(jitter)
                    continue
                # Add some jitter to prevent thundering herd
                jitter = (1.0 + random.random()) * delay
                await asyncio.sleep(jitter)

        # Increase delay for next round of retries
        delay = min(delay * backoff_factor, max_delay)
    if last_exception is not None:
        raise last_exception
    else:
        raise Exception(f"All retry attempts failed for all urls: {urls}")


def make_request_with_retry_sync(
    requests: Union[Callable, List[Callable]],
    urls: List[str] = None,
    response_parser: Callable = status_check_for_response,
    max_retries: int = COSMOS_HTTP_RETRY_CONFIG.max_retries,
    retries_per_delay: int = COSMOS_HTTP_RETRY_CONFIG.retries_per_delay,
    initial_delay: float = COSMOS_HTTP_RETRY_CONFIG.initial_delay,
    max_delay: float = COSMOS_HTTP_RETRY_CONFIG.max_delay,
    backoff_factor: float = COSMOS_HTTP_RETRY_CONFIG.backoff_factor,
) -> Any:
    """
    Make an HTTP GET request with exponential backoff retry logic.

    Args:
        requests (List[Callable]): The functions to make the request in an alternative way
        urls (List[str]): List of host URLs to try
        response_parser (Callable): Function to parse the response
        max_retries (int): Maximum number of retry attempts
        retries_per_delay (int): Number of retries to attempt at each delay level
        initial_delay (float): Initial delay between retries in seconds
        max_delay (float): Maximum delay between retries in seconds
        backoff_factor (float): Factor to increase delay between retries

    Returns:
        Any: The response object from the successful request or redis client request.

    Raises:
        Exception: If all retry attempts fail
    """
    delay = initial_delay
    last_exception = None
    total_attempts = 0
    url_index = 0
    request_idx = 0

    if isinstance(requests, Callable):
        requests = [requests]

    while total_attempts < max_retries:
        # Try multiple times at the current delay level
        total_retries_cur_delay = 0
        while total_retries_cur_delay < retries_per_delay:
            try:
                request = requests[request_idx]
                if urls is not None:
                    url = urls[url_index]
                    # logger.info(f"Making request {request.keywords} to {url}")
                    r = request(url)
                    # logger.info(f"Finished request {request.keywords} to {url}")
                else:
                    url = None
                    # logger.info(f"Making request {request.keywords}")
                    r = request()
                    # logger.info(f"Finished request {request.keywords}")
                if response_parser is not None:
                    response_parser(r)
                return r

            except Exception as e:
                last_exception = e
                url_index += 1
                if url_index >= (1 if urls is None else len(urls)):
                    url_index = 0
                    request_idx += 1
                    if request_idx >= len(requests):
                        request_idx = 0
                        total_retries_cur_delay += 1
                        total_attempts += 1
                logger.info(
                    f"Request failed: {e}. Attempt {total_attempts} of {max_retries} for {request} on {url}."
                )
                if total_attempts >= max_retries:
                    break

                if request_idx != 0 or url_index != 0:
                    jitter = (1.0 + random.random()) * initial_delay
                    time.sleep(jitter)
                    continue
                # Add some jitter to prevent thundering herd
                jitter = (1.0 + random.random()) * delay
                time.sleep(jitter)

        # Increase delay for next round of retries
        delay = min(delay * backoff_factor, max_delay)
    if last_exception is not None:
        raise last_exception
    else:
        raise Exception(f"All retry attempts failed for all urls: {urls}")
