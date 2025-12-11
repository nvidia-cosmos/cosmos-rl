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

import os
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config as BotoConfig

from cosmos_rl.utils.logging import logger

_S3_AVAILABLE = False


def set_s3_available(available: bool):
    global _S3_AVAILABLE
    _S3_AVAILABLE = available


def is_s3_available() -> bool:
    return _S3_AVAILABLE


def upload_file_to_s3(
    local_file_path: str,
    bucket_name: str,
    s3_file_path: str,
    max_retries: int = 3,
):
    config = BotoConfig(retries={"max_attempts": 10, "mode": "standard"})
    s3_client = boto3.client("s3", config=config)
    retry = 0
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except ClientError:
        logger.info(f"Bucket {bucket_name} does not exist, creating it now.")
        s3_client.create_bucket(Bucket=bucket_name)
    while retry < max_retries:
        try:
            s3_client.upload_file(local_file_path, bucket_name, s3_file_path)
            logger.info(
                f"Uploaded {local_file_path} to s3://{bucket_name}/{s3_file_path}"
            )
            return
        except ClientError as e:
            retry += 1
            logger.error(
                f"Failed to upload {local_file_path} to s3://{bucket_name}/{s3_file_path}. "
                f"Retry {retry}/{max_retries}. Error: {e}"
            )
    logger.error(
        f"Failed to upload {local_file_path} to s3://{bucket_name}/{s3_file_path} "
        f"after {max_retries} retries."
    )


def upload_folder_to_s3(
    local_folder: str,
    bucket_name: str,
    s3_folder: str,
    max_retries: int = 3,
):
    for root, _, files in os.walk(local_folder):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_folder)
            s3_file_path = os.path.join(s3_folder, relative_path)
            upload_file_to_s3(
                local_file_path, bucket_name, s3_file_path, max_retries=max_retries
            )
