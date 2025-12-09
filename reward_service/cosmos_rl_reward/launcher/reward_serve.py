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
from concurrent.futures import ProcessPoolExecutor
from cosmos_rl_reward.utils.redis import start_redis_server
import os
from fastapi import Request
from cosmos_rl_reward.utils.logging import logger
import argparse
import multiprocessing as mp
import uuid
import json
from typing import Dict
from fastapi import FastAPI, HTTPException, status, Form
from cosmos_rl_reward.handler.decode import DecodeHandler
from cosmos_rl_reward.utils.protocol import (
    COSMOS_RL_REWARD_ENQUEUE_API_SUFFIX,
    COSMOS_RL_REWARD_LATENT_ATTR_API_SUFFIX,
    COSMOS_RL_REWARD_PING_API_SUFFIX,
    COSMOS_RL_REWARD_REDIS_PUSH_API_SUFFIX,
    COSMOS_RL_REWARD_REDIS_PULL_API_SUFFIX,
)
from cosmos_rl.utils.util import find_available_port
import uvicorn
from cosmos_rl_reward.handler.process import RewardProcessHandler
from cosmos_rl_reward.handler.score_kv import KeyValueHandler
from fastapi.responses import JSONResponse
from cosmos_rl_reward.launcher.config import Config
from cosmos_rl_reward.handler.registry import RewardRegistry
import toml
from cosmos_rl_reward.utils.util import get_cosmos_rl_reward_cache_dir, download_file

app = FastAPI()

server = None


# @app.middleware("http")
# async def log_requests(request: Request, call_next):
#     response = await call_next(request)
#     logger.info(f"{request.method} {request.url} Headers: {request.headers} Response status: {response.status_code}")
#     return response


@app.post(COSMOS_RL_REWARD_ENQUEUE_API_SUFFIX)
async def enqueue(request: Request):
    """
    Enqueue a video decoding task.
    This endpoint accepts raw video data in the request body, generates a unique UUID for the task,
    and places the task in the DecodeHandler's task queue for processing.
    If the decoding loop is not already running, it starts the loop in a separate process.
    Args:
        request (Request): The incoming HTTP request containing the decoded video latent data with the metadata.
    Returns:
        dict: A dictionary containing the UUID of the enqueued task.
    Raises:
        HTTPException: If there is an internal error while enqueuing the task.
    """

    logger.info("Enqueue...")
    raw_body = await request.body()
    try:
        controller = DecodeHandler.get_instance()
        uuis = str(uuid.uuid4())
        controller.task_queue.put((uuis, raw_body))
        if not controller.loop.is_set():
            controller.loop.set()
            controller.threading_pool.submit(
                DecodeHandler.extract_video_in_queue,
                controller.task_queue,
                controller.max_pending_tasks,
                controller.loop,
            )
        return {"uuid": uuis}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )


@app.post(COSMOS_RL_REWARD_LATENT_ATTR_API_SUFFIX)
async def set_latent_attr(
    fields: str = Form(...),  # This will be a JSON string
):
    """
    Set attributes for latent decoding.
    Args:
        fields (str): A JSON string containing the attributes to set.
    Returns:
        dict: A message indicating success of setting the attributes.
    Raises:
        HTTPException: If there is an internal error while setting attributes.
    """
    try:
        controller = DecodeHandler.get_instance()
        await asyncio.get_event_loop().run_in_executor(
            controller.set_pool, DecodeHandler.set_latent_attr, fields
        )
        return {"message": "Latent attributes set successfully"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )


@app.post(COSMOS_RL_REWARD_PING_API_SUFFIX)
async def ping(
    info_data: str = Form(...),  # This will be a JSON string
):
    """
    Handle a ping request.
    Args:
        info_data (str): A JSON string containing metadata for the ping.
    Returns:
        dict: A message indicating the ping request was received.
    """
    logger.info(f"Received metadata for ping: {info_data}")
    return {"message": "Ping request received"}


@app.post(COSMOS_RL_REWARD_REDIS_PUSH_API_SUFFIX)
async def push_scores(
    uuid: str = Form(...),
    scores: str = Form(...),  # This will be a JSON string
    type: str = Form(None),  # Type of the reward
):
    """
    Push scores to the key-value store.
    Args:
        uuid (str): The unique identifier for the scores calculation task.
        scores (str): A JSON string containing the scores with other related information.
        type (str, optional): The type of the reward.
    Returns:
        dict: A message indicating success of storing the scores.
    Raises:
        HTTPException: If there is an internal error while storing the scores.
    """
    try:
        logger.info(f"Start push for {uuid}")
        controller = KeyValueHandler.get_instance()
        if controller.pull_pool is None:
            await controller.set_score(uuid, scores, type)
        else:
            await asyncio.get_event_loop().run_in_executor(
                controller.push_pool,
                controller.set_score_sync,
                uuid,
                scores,
                type,
                controller.redis_port,
            )
        logger.info(f"Pushed scores for {uuid}.")
        return {"message": "Scores received and stored successfully"}
    except Exception as e:
        logger.error(f"Error storing scores for {uuid}: {e}")
        return JSONResponse(
            status_code=500, content={"message": f"Error storing scores: {e}"}
        )


@app.post(COSMOS_RL_REWARD_REDIS_PULL_API_SUFFIX)
async def pull_scores(
    uuid: str = Form(...),
    type: str = Form(None),  # Type of the reward
):
    """
    Pull scores from the key-value store using the provided UUID.
    To retrieve the calculated score results associated with a specific UUID for the enqueued reward calculation task.
    Args:
        uuid (str): The unique identifier for the scores calculation task.
        type (str, optional): The type of the reward.
    Returns:
        dict: The pulled score results or an error message.
        If the scores are found, returns them as a dictionary.
        If not found, returns a 400 error message indicating scores have not been calculated yet and need to be retried later.
    Raises:
        HTTPException: If there is an internal error while pulling the scores.
    """
    try:
        logger.info(f"Start pull for {uuid}")
        controller = KeyValueHandler.get_instance()
        if controller.pull_pool is None:
            value = await controller.get_score(uuid, type)
        else:
            value = await asyncio.get_event_loop().run_in_executor(
                controller.pull_pool,
                controller.get_score_sync,
                uuid,
                type,
                controller.redis_port,
            )
        if value is None or len(value) == 0 or value == "None":
            res = JSONResponse(
                status_code=400, content={"message": f"Not found scores for {uuid}"}
            )
        else:
            res = json.loads(
                value.decode("utf-8") if isinstance(value, bytes) else value
            )
        logger.info(f"Pulled scores for {uuid}: {res}")
        return res
    except Exception as e:
        logger.error(f"Error pulling scores for {uuid}: {e}")
        return JSONResponse(
            status_code=500, content={"message": f"Error pulling scores {e}"}
        )


def main():
    """
    Main function to start the reward service.
    This function sets up the FastAPI server, initializes the Redis server for key-value storage,
    and prepares the reward process handlers which start processes for different reward models calculation as specified in the configuration.
    It also initializes the DecodeHandler for latent decoding of video data.
    The server listens on the specified port and handles incoming requests for enqueuing tasks,
    setting latent attributes, pinging, and pushing/pulling scores to/from the key-value store.
    """
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(
        description="Run the web panel for the dispatcher."
    )

    parser.add_argument(
        "--config", type=str, default=None, help="Path to the configuration file."
    )

    args = parser.parse_args()

    if args.config is not None:
        logger.info(f"Attempting to load configuration from {args.config}")
        with open(args.config, "r") as f:
            config_dict = toml.load(f)
        loaded_config = Config.from_dict(config_dict)
    else:
        logger.info("No configuration file provided. Using default configuration file.")
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(cur_dir, "../configs/rewards.toml")
        if os.path.exists(config_file):
            logger.info(f"Loading default configuration from {config_file}")
            with open(config_file, "r") as f:
                config_dict = toml.load(f)
            loaded_config = Config.from_dict(config_dict)
        else:
            loaded_config = Config()
            logger.info(
                "No default configuration file found. Using default configuration."
            )

    redis_port = start_redis_server(
        loaded_config.redis_port
    )  # Start Redis server and get the port

    # Initialize the KeyValueHandler singleton
    # KeyValueHandler manages interactions with Redis for storing and retrieving scores
    kv_controller = KeyValueHandler.get_instance(redis_port)
    # Initialize the event loop for KeyValueHandler
    kv_controller.init_loop()

    # Prepare reward process handlers
    # Each reward model will have its own process handler
    port = find_available_port(loaded_config.port)
    redis_url = f"http://localhost:{port}"
    info = {"reward": {}}
    controllers: Dict[str, RewardProcessHandler] = {}
    for reward_arg in loaded_config.reward_args:
        logger.info(f"Setting up reward: {reward_arg.reward_type}")
        key = reward_arg.reward_type
        controllers[key] = RewardProcessHandler(
            key,
            reward_arg.model_path,
            reward_arg.dtype,
            device=reward_arg.device,
            download_path=reward_arg.download_path,
        )
        RewardRegistry.register_reward_venv(key, reward_arg.venv_python)
        controllers[key].init_process(envs={"REDIS_URL": redis_url})
        info["reward"][key] = controllers[key].process_handler.shm_cpu.name

    # Instantiate the DecodeHandler singleton
    decoder = DecodeHandler.get_instance()
    # Build the worker process for DecodeHandler
    decoder.threading_pool = ProcessPoolExecutor(max_workers=1)

    # Prepare model path for DecodeHandler
    if os.path.exists(loaded_config.decode_args.model_path):
        model_path = loaded_config.decode_args.model_path
    else:
        model_path = os.path.join(
            get_cosmos_rl_reward_cache_dir(),
            os.path.basename(loaded_config.decode_args.model_path),
        )
        if not os.path.exists(model_path):
            download_file(loaded_config.decode_args.model_path, model_path)

    # Initialize the DecodeHandler in the worker process
    initialized = decoder.threading_pool.submit(
        DecodeHandler.initialize,
        info=info,
        model_path=model_path,
        dtype=loaded_config.decode_args.dtype,
        device=loaded_config.decode_args.device,
        chunk_duration=loaded_config.decode_args.chunk_duration,
        temporal_window=loaded_config.decode_args.temporal_window,
        load_mean_std=loaded_config.decode_args.load_mean_std,
    )
    initialized.result()  # Wait for initialization to complete

    # Start the Uvicorn server
    config = uvicorn.Config(app, host="0.0.0.0", port=port, access_log=False)
    global server
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    main()
