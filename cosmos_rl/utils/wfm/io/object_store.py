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

from __future__ import annotations

import io
import json
import os
import pickle
import random
import time
from typing import Any, Callable

import boto3
import botocore
import numpy as np
import torch
import yaml
from botocore.config import Config
from botocore.exceptions import ClientError
from PIL import Image

from cosmos_rl.utils.logging import logger
import cosmos_rl.utils.wfm.io.easy_io.backends.auto_auth as auto

GLOBAL_S3_CONFIG = Config(
    retries={"max_attempts": 20, "mode": "adaptive"},
    connect_timeout=10,
    read_timeout=60,
)
Image.MAX_IMAGE_PIXELS = None


class ObjectStore:
    """This is the interface class for object store, used for interacting with PBSS/AWS (S3).

    Attributes:
        client (botocore.client.S3): Object store client object.
        bucket (str): Object store bucket name.
    """

    def __init__(self, config_object_storage):
        with auto.open_auth(config_object_storage.credentials, "r") as file:
            object_storage_config = auto.json_load_auth(file)
            self.client = Boto3Wrapper(
                "s3",
                **object_storage_config,
            )
        self.bucket = config_object_storage.bucket

    def load_object(
        self,
        key: str,
        type: str | None = None,
        load_func: Callable | None = None,
        encoding: str = "UTF-8",
        max_attempts: int = 10,
    ) -> Any:
        """Helper function for loading object from PBSS.

        Args:
            key (str): The key of the object.
            type (str): Specified for some common data types. If not provided, `load_func` should be specified.
                The predefined types currently supported are:
                - "torch": PyTorch model checkpoints, opened with torch.load().
                - "torch.jit": A JIT-compiled TorchScript model, loaded with torch.jit.load().
                - "image": Image objects, opened with PIL.Image.open().
                - "json": JSON files, opened with json.load().
                - "pickle": Picklable objects, opened with pickle.load().
                - "yaml": YAML files, opened with yaml.safe_load().
                - "text": Pure text files.
                - "numpy": Numpy arrays, opened with np.load().
                - "bytes": Raw bytes.
            load_func (Callable): a custom function for reading the buffer if `type` were not provided.
            encoding (str): Text encoding standard (default: "UTF-8").
            max_attempts (int): Max number of attempts to load the object if there is a failure.

        Returns:
            object (Any): The downloaded object.
        """

        for attempt in range(max_attempts):
            try:
                return self._load_object(
                    key,
                    type=type,
                    load_func=load_func,
                    encoding=encoding,
                )
            except botocore.exceptions.ClientError as e:
                retry_interval = min(0.1 * 2**attempt + random.uniform(0, 1), 30)
                logger.exception(
                    f"Failed to load ({self.bucket}) {key}, attempt {attempt}. {e}. Retrying in {retry_interval}s."
                )
                if attempt < max_attempts - 1:
                    time.sleep(retry_interval)
        raise ConnectionError(
            f"Unable to read ({self.bucket}) {key} after {max_attempts} attempts."
        )

    def _load_object(
        self,
        key: str,
        type: str | None = None,
        load_func: Callable | None = None,
        encoding: str = "UTF-8",
    ) -> Any:
        """Helper function for loading object from PBSS.

        Args:
            key (str): The key of the object.
            type (str): Specified for some common data types. If not provided, `load_func` should be specified.
            load_func (Callable): a custom function for reading the buffer if `type` were not provided.
            encoding (str): Text encoding standard (default: "UTF-8").

        Returns:
            object (Any): The downloaded object.
        """
        assert (
            type is not None or load_func is not None
        ), "Either type or load_func should be specified."
        with io.BytesIO() as buffer:
            # TODO(chenhsuanl): some may have more efficient implementations without loading the whole object.
            self.client.download_fileobj(Bucket=self.bucket, Key=key, Fileobj=buffer)
            buffer.seek(0)
            # Read from buffer for common data types.
            if type == "torch":
                object = torch.load(
                    buffer,
                    map_location=lambda storage, loc: storage,
                    weights_only=False,
                )
            elif type == "torch.jit":
                object = torch.jit.load(buffer)
            elif type == "image":
                object = Image.open(buffer)
                object.load()
            elif type == "json":
                object = json.load(buffer)
            elif type == "pickle":
                object = pickle.load(buffer)
            elif type == "yaml":
                object = yaml.safe_load(buffer)
            elif type == "text":
                object = buffer.read().decode(encoding)
            elif type == "numpy":
                object = np.load(buffer, allow_pickle=True)
            # Read from buffer as raw bytes.
            elif type == "bytes":
                object = buffer.read()
            # Customized load_func should be provided.
            else:
                object = load_func(buffer)
        return object

    def save_object(
        self,
        object: Any,
        key: str,
        type: str | None = None,
        save_func: Callable | None = None,
        encoding: str = "UTF-8",
    ) -> None:
        """Helper function for saving object to PBSS.

        Args:
            object (Any): The object to upload.
            key (str): The key of the object.
            type (str): Specified for some common data types. If not provided, `save_func` should be specified.
                The predefined types currently supported are:
                - "torch": PyTorch model checkpoints, saved with torch.save().
                - "torch.jit": A JIT-compiled TorchScript model, exported with torch.jit.save().
                - "image": Image objects, saved with PIL.Image.save().
                - "json": JSON files, saved with json.dumps().
                - "pickle": Picklable objects, saved with pickle.dump().
                - "yaml": YAML files, saved with yaml.safe_dump().
                - "text": Pure text files.
                - "numpy": Numpy arrays, saved with np.save().
                - "bytes": Raw bytes.
            save_func (Callable): a custom function for writing the buffer if `type` were not provided.
            encoding (str): Text encoding standard (default: "UTF-8").
        """
        assert type is not None or save_func is not None
        with io.BytesIO() as buffer:
            # TODO(chenhsuanl): some may have more efficient implementations without loading the whole object.
            # Write to buffer for common data types.
            if type == "torch":
                torch.save(object, buffer)
            elif type == "torch.jit":
                torch.jit.save(object, buffer)
            elif type == "image":
                type = os.path.basename(key).split(".")[-1]
                object.save(buffer, format=type)
            elif type == "json":
                buffer.write(json.dumps(object).encode(encoding))
            elif type == "pickle":
                pickle.dump(object, buffer)
            elif type == "yaml":
                buffer.write(yaml.safe_dump(object).encode(encoding))
            elif type == "text":
                buffer.write(object.encode(encoding))
            elif type == "numpy":
                np.save(buffer, object)
            # Write to buffer as raw bytes.
            elif type == "bytes":
                buffer.write(bytes(object))
            # Customized save_func should be provided.
            else:
                save_func(object, buffer)
            buffer.seek(0)
            self.client.upload_fileobj(Bucket=self.bucket, Key=key, Fileobj=buffer)

    def object_exists(
        self, key: str, max_retries: int = 10, retry_delay: float = 2.0
    ) -> bool:
        """
        Check whether an object exists in the storage, with retry logic for transient errors.

        Args:
            key (str): The key of the object.
            max_retries (int): The maximum number of retry attempts in case of errors.
            retry_delay (float): The delay (in seconds) between retry attempts.

        Returns:
            bool: True if the object exists, False if not or if an error persists after retries.
        """
        for attempt in range(max_retries):
            try:
                # Attempt to check if the object exists
                self.client.head_object(Bucket=self.bucket, Key=key)
                return True
            except ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    return False  # Object does not exist
                # Log or print the error for troubleshooting
                logger.error(f"Attempt {attempt + 1} failed: {e}")

                # If this is the last attempt, return False
                if attempt == max_retries - 1:
                    return False

                # Wait for the specified delay before retrying
                time.sleep(retry_delay)
            except Exception as e:
                # Handle other unexpected exceptions
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")

                # If this is the last attempt, return False
                if attempt == max_retries - 1:
                    return False

                # Wait for the specified delay before retrying
                time.sleep(retry_delay)

        # If all retries fail, return False
        return False


class Boto3Wrapper:
    """
    This class serves as a wrapper around boto3.client in order to make boto3.client serializable. It's required to use
    spawn method of creating DataLoader workers, which is in turn required to avoid segfaults when using Triton, e.g.
    for torch.compile or custom kernels.
    """

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self.client = None

    def __setstate__(self, state):
        self.__dict__ = state

    def __getattr__(self, item):
        is_worker = torch.utils.data.get_worker_info() is not None
        client = (
            boto3.client(*self._args, **self._kwargs, config=GLOBAL_S3_CONFIG)
            if self.client is None
            else self.client
        )
        if is_worker:
            self.client = client
        return getattr(client, item)
