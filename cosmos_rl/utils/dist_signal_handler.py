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
import signal
from typing import List
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.distributed import all_gather_object_cpu


class DistributedSignalHandler:
    @classmethod
    def get_instance(
        cls, sig: List[str] = None, processes=None
    ) -> "DistributedSignalHandler":
        if not hasattr(DistributedSignalHandler, "_instance"):
            assert sig is not None, (
                "Signal list must be provided for the first time to initialize the DistributedSignalHandler instance."
            )
            DistributedSignalHandler._instance = DistributedSignalHandler(
                sig, processes
            )
        return DistributedSignalHandler._instance

    def __init__(self, sig: List[str], processes=None):
        self.sig = sig
        self._signal_received = False
        self.released = False
        self.original_handler = {
            signal.Signals[s.upper()]: signal.getsignal(signal.Signals[s.upper()])
            for s in self.sig
        }
        self.processes = processes

        def handler_default(signum, frame):
            logger.info(
                f"Signal {signum} received, setting signal_received to True in handler_default at {os.getpid()}"
            )
            self._signal_received = True

        def handler(signum, frame):
            import psutil

            logger.info(
                f"Signal {signum} received, forwarding to subprocesses at {os.getpid()}"
            )
            # forward to the entire subprocess group except for itself to avoid the risk of killing itself before forwarding the signal
            for p in self.processes:  # skip the controller process if it exists
                p = psutil.Process(p.pid)
                for c in p.children(recursive=False):
                    try:
                        cmd = c.cmdline()  # list[str]
                        cmd_str = " ".join(cmd).lower()
                        logger.info(f"Process name: cmdline: {cmd_str} {c.pid}")
                        if "torchrun" in cmd_str:
                            tp = psutil.Process(c.pid)
                            for tp_c in tp.children(recursive=False):
                                cmd = tp_c.cmdline()
                                cmd_str = " ".join(cmd).lower()
                                logger.info(
                                    f"Process name in torchrun: cmdline: {cmd_str} {tp_c.pid}"
                                )
                                if "python" in cmd[0] and "torchrun" not in cmd_str:
                                    logger.info(
                                        f"Sending signal {signum} to process in torchrun {tp_c.pid} with cmdline: {cmd_str}"
                                    )
                                    os.kill(tp_c.pid, signum)
                        elif "python" in cmd[0] and "torchrun" not in cmd_str:
                            logger.info(
                                f"Sending signal {signum} to process {c.pid} with cmdline: {cmd_str}"
                            )
                            os.kill(c.pid, signum)
                    except ProcessLookupError as e:
                        logger.warning(
                            f"Process {c.pid} does not exist anymore when sending signal: {e}"
                        )
            logger.info(
                f"Finished forwarding signal {signum} to subprocesses at {os.getpid()}"
            )

        for s in self.original_handler.keys():
            signal.signal(s, handler if self.processes is not None else handler_default)
            logger.info(
                f"Signal handler for signal {s} is set to {handler if self.processes is not None else handler_default}, pid {os.getpid()}"
            )

    def signals_received(self):
        all_received = all_gather_object_cpu(self._signal_received)
        return all_received

    def release(self):
        if self.released:
            return False

        for s, handler in self.original_handler.items():
            signal.signal(s, handler)
        self.released = True
        return True
