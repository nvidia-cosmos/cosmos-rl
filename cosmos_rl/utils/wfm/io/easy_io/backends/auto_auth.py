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

import contextlib
import json
from typing import Any, Optional

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.wfm.io.cred_env_parser import (
    CRED_ENVS,
    CRED_ENVS_DICT,
)

DEPLOYMENT_ENVS = ["prod", "dev", "stg"]


# context manger to open a file or read from env variable
@contextlib.contextmanager
def open_auth(s3_credential_path: Optional[Any], mode: str):
    if not s3_credential_path:
        logger.info(f"No credential file provided {s3_credential_path}.")
        yield None
        return

    name = s3_credential_path.split("/")[-1].split(".")[0]
    if not name:
        raise ValueError(f"Could not parse into env var: {s3_credential_path}")
    cred_env_name = f"PROD_{name.upper()}"

    if CRED_ENVS.APP_ENV in DEPLOYMENT_ENVS and cred_env_name in CRED_ENVS_DICT:
        object_storage_config = get_creds_from_env(cred_env_name)
        logger.info(f"using ENV vars for {cred_env_name}")
        yield object_storage_config
    else:
        logger.info(f"using credential file: {s3_credential_path}")
        with open(s3_credential_path, mode) as f:
            yield f


def get_creds_from_env(cred_env_name: str) -> dict[str, str]:
    try:
        object_storage_config = CRED_ENVS_DICT[cred_env_name]
    except KeyError:
        raise ValueError(f"Could not find {cred_env_name} in CRED_ENVS")
    empty_args = {
        key.upper() for key in object_storage_config if object_storage_config[key] == ""
    }
    if empty_args:
        raise ValueError(
            f"Some required environment variable(s) were not provided for {cred_env_name}",
            empty_args,
        )
    return object_storage_config


def json_load_auth(f):
    if CRED_ENVS.APP_ENV in DEPLOYMENT_ENVS:
        return f if f else {}
    else:
        return json.load(f)
