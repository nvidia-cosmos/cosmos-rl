# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

import io

# PBSS
import time
from typing import Optional

import boto3
from botocore.exceptions import EndpointConnectionError
from urllib3.exceptions import ProtocolError as URLLib3ProtocolError
from urllib3.exceptions import ReadTimeoutError as URLLib3ReadTimeoutError
from urllib3.exceptions import SSLError as URLLib3SSLError

from cosmos_rl.utils.logging import logger


class RetryingStream:
    def __init__(self, client: boto3.client, bucket: str, key: str, retries: int = 10):  # type: ignore
        r"""Class for loading data in a streaming fashion.
        Args:
            client (boto3.client): Boto3 client
            bucket (str): Bucket where data is stored
            key (str): Key to read
            retries (int): Number of retries
        """
        self.client = client
        self.bucket = bucket
        self.key = key
        self.retries = retries
        self.content_size = self.get_length()
        self.stream, _ = self.get_stream()
        self._amount_read = 0

        self.name = f"{bucket}/{key}"

    def get_length(self) -> int:
        r"""Function for obtaining length of the bytestream"""
        head_obj = self.client.head_object(Bucket=self.bucket, Key=self.key)
        length = int(head_obj["ContentLength"])
        return length

    def get_stream(
        self, start_range: int = 0, end_range: Optional[int] = None
    ) -> tuple[io.BytesIO, int]:
        r"""Function for getting stream in a range
        Args:
            start_range (int): Start index for stream
            end_range (int): End index for stream
        Returns:
            stream (bytes): Stream of data being read
            content_size (int): Length of the bytestream read
        """
        extra_args = {}
        if start_range != 0 or end_range is not None:
            end_range = "" if end_range is None else end_range - 1  # type: ignore
            # Start and end are inclusive in HTTP, convert to Python convention
            range_param = f"bytes={start_range}-{end_range}"
            extra_args["Range"] = range_param
        response = self.client.get_object(
            Bucket=self.bucket, Key=self.key, **extra_args
        )
        content_size = response["ResponseMetadata"]["HTTPHeaders"]["content-length"]
        body = response["Body"]
        stream = body._raw_stream
        return stream, content_size

    def read(self, amt: Optional[int] = None) -> bytes:
        r"""Read function for reading the data stream.
        Args:
            amt (int): Amount of data to read
        Returns:
            chunk (bytes): Bytes read
        """

        chunk = b""
        for cur_retry_idx in range(self.retries):
            try:
                chunk = self.stream.read(amt)
                if len(chunk) == 0 and self._amount_read != self.content_size:
                    raise IOError
                break
            except URLLib3ReadTimeoutError as e:
                logger.warning(
                    f"[read] URLLib3ReadTimeoutError: {e} {self.name} retry: {cur_retry_idx} / {self.retries}",
                )
            except URLLib3ProtocolError as e:
                logger.warning(
                    f"[read] URLLib3ProtocolError: {e} {self.name} retry: {cur_retry_idx} / {self.retries}",
                )
            except URLLib3SSLError as e:
                logger.warning(
                    f"[read] URLLib3SSLError: {e} {self.name} retry: {cur_retry_idx} / {self.retries}",
                )
            except IOError as e:
                logger.warning(
                    f"[read] Premature end of stream. IOError {e}. Retrying...  {self.name} retry: {cur_retry_idx} / {self.retries}",
                )
            time.sleep(1)
            try:
                self.stream, _ = self.get_stream(self._amount_read)
            except EndpointConnectionError as e:
                logger.error(
                    f"[get_stream] EndpointConnectionError: {e} {self.name} retry: {cur_retry_idx} / {self.retries}",
                )

        if len(chunk) == 0 and self._amount_read != self.content_size:
            logger.warning(
                f"After {self.retries} retries, chunk is empty and self._amount_read != self.content_size {self._amount_read} != {self.content_size} {self.name}",
            )
            raise IOError

        self._amount_read += len(chunk)
        return chunk
