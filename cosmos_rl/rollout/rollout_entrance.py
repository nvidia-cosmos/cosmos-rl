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
import argparse
from typing import Optional
from cosmos_rl.dispatcher.data.packer.base import worker_entry_parser
from cosmos_rl.utils.logging import logger
from cosmos_rl.rollout.worker.llm_worker import LLMRolloutWorker


def run_rollout(args: Optional[argparse.Namespace] = None, **kwargs):
    # This means that args are not parsed in dataset entry script
    # So we need to parse the args manually
    if args is None:
        parser = worker_entry_parser()
        try:
            args = parser.parse_args()
        except SystemExit as e:
            logger.error(
                "Error when parsing args. Did you use custom arguments in your script? If so, please check your custom script and pass `args` to this main function."
            )
            raise e
    is_diffusion = os.environ.get("COSMOS_IS_DIFFUSION", "False").lower() == "true"

    if is_diffusion:
        raise NotImplementedError(
            "Diffusion rollout is not supported with this entrance script yet."
        )
    else:
        rollout_worker = LLMRolloutWorker(**kwargs)

    rollout_worker.execute()


if __name__ == "__main__":
    run_rollout()
