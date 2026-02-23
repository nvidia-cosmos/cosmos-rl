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

"""This script is launched by slurm job on each node.

It reads the node launch metadata from environment variables and launches
the appropriate cosmos-rl replicas (policy or rollout) on this node.

This version is designed to work with pip-installed cosmos-rl package,
using the installed console scripts instead of referencing the source tree.
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from argparse import REMAINDER
from typing import List

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from util import NodeLaunchMetadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_cosmos_rl_launcher() -> str:
    """Find the cosmos-rl replica launcher script.

    Returns the path to the launch_replica.sh script from the installed
    cosmos-rl package, or falls back to using the python module directly.
    """
    # Try to find the installed cosmos-rl package location
    try:
        import cosmos_rl

        cosmos_rl_dir = os.path.dirname(os.path.dirname(cosmos_rl.__file__))
        launcher_script = os.path.join(
            cosmos_rl_dir, "cosmos_rl", "launcher", "launch_replica.sh"
        )
        if os.path.exists(launcher_script):
            logger.info(f"Found cosmos-rl launcher at: {launcher_script}")
            return launcher_script
    except ImportError:
        pass

    # Fallback: check if launch_replica.sh is in PATH
    launch_replica = shutil.which("cosmos-rl-replica")
    if launch_replica:
        logger.info(f"Found cosmos-rl-replica in PATH: {launch_replica}")
        return launch_replica

    # Last resort: use python -m cosmos_rl.launcher
    logger.warning(
        "Could not find cosmos-rl launcher script, will use python module directly"
    )
    return None


def build_replica_command(
    launcher_path: str | None,
    replica_type: str,
    rdzv_endpoint: str,
    ngpus: int,
    nnodes: int,
    config: str | None,
    script: str | None,
    script_args: List[str],
) -> List[str]:
    """Build the command to launch a replica.

    Args:
        launcher_path: Path to launch_replica.sh or None to use python module
        replica_type: "policy" or "rollout"
        rdzv_endpoint: Rendezvous endpoint (host:port)
        ngpus: Number of GPUs for this replica
        nnodes: Number of nodes for this replica
        config: Path to config file
        script: Custom launcher script
        script_args: Additional arguments for the script

    Returns:
        Command as a list of strings
    """
    if launcher_path and os.path.exists(launcher_path):
        cmd = [
            launcher_path,
            "--type",
            replica_type,
            "--rdzv-endpoint",
            rdzv_endpoint,
            "--ngpus",
            str(ngpus),
            "--nnodes",
            str(nnodes),
        ]
        if config:
            cmd += ["--config", config]
        if script:
            cmd += ["--script", script]
        if script_args:
            cmd.extend(script_args)
    else:
        # Use python module directly
        cmd = [
            sys.executable,
            "-m",
            "cosmos_rl.launcher.launch_replica",
            "--type",
            replica_type,
            "--rdzv-endpoint",
            rdzv_endpoint,
            "--ngpus",
            str(ngpus),
            "--nnodes",
            str(nnodes),
        ]
        if config:
            cmd += ["--config", config]
        if script:
            cmd += ["--script", script]
        if script_args:
            cmd.extend(script_args)

    return cmd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch cosmos-rl replicas on a SLURM node"
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["policy", "rollout"],
        help="Type of replica to launch",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the config file for the policy or rollout script.",
    )
    parser.add_argument(
        "script",
        nargs="?",  # "?" means 0 or 1 occurrences
        default=None,
        help="A user script for custom dataset, reward functions, and model registration.",
    )
    parser.add_argument("script_args", nargs=REMAINDER)

    args = parser.parse_args()

    # Get node information from environment
    node_list_str = os.environ.get("LOCAL_NODE_LIST", "")
    if not node_list_str:
        logger.error("LOCAL_NODE_LIST environment variable not set")
        sys.exit(1)
    node_list = node_list_str.split(" ")

    cosmos_controller_host = os.environ.get("COSMOS_CONTROLLER_HOST", "")
    if not cosmos_controller_host:
        logger.error("COSMOS_CONTROLLER_HOST environment variable not set")
        sys.exit(1)

    self_node = os.environ.get("SLURMD_NODENAME", "")
    if not self_node:
        logger.error("SLURMD_NODENAME environment variable not set")
        sys.exit(1)

    # Get node launch metadata
    metadata_env_var = f"NODE_LAUNCH_METADATA_{args.type.upper()}"
    metadata_str = os.environ.get(metadata_env_var, "")
    if not metadata_str:
        logger.error(f"{metadata_env_var} environment variable not set")
        sys.exit(1)

    node_launch_metadata: List[NodeLaunchMetadata] = NodeLaunchMetadata.from_json_list(
        metadata_str
    )

    logger.info(f"COSMOS_CONTROLLER_HOST: {cosmos_controller_host}")
    logger.info(f"NODE LIST: {node_list}")
    logger.info(f"Self node: {self_node}")

    if self_node not in node_list:
        logger.error(f"self_node {self_node} not in node_list {node_list}")
        sys.exit(1)

    self_node_idx = node_list.index(self_node)
    self_node_launch_metadata = node_launch_metadata[self_node_idx]

    # Find the cosmos-rl launcher
    launcher_path = find_cosmos_rl_launcher()

    # Build and launch commands for each replica on this node
    cmds = []
    envs = []
    for replica_launch_metadata in self_node_launch_metadata.colocation:
        rendezvous_node = node_list[replica_launch_metadata.rendezvous_node]
        rendezvous_port = replica_launch_metadata.rendezvous_port
        visible_gpus = replica_launch_metadata.visible_gpus
        nnode = replica_launch_metadata.nnode

        logger.info(
            f"Rendezvous node: {rendezvous_node}, port: {rendezvous_port}, "
            f"visible GPUs: {visible_gpus}, nnodes: {nnode}"
        )

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in visible_gpus)
        env["VLLM_DISABLE_COMPILE_CACHE"] = "1"

        cmd = build_replica_command(
            launcher_path=launcher_path,
            replica_type=args.type,
            rdzv_endpoint=f"{rendezvous_node}:{rendezvous_port}",
            ngpus=len(visible_gpus),
            nnodes=nnode,
            config=args.config,
            script=args.script,
            script_args=args.script_args,
        )

        logger.info(f"Command: {' '.join(cmd)}")
        cmds.append(cmd)
        envs.append(env)

    # Launch all replicas
    procs = [subprocess.Popen(cmd, env=env) for cmd, env in zip(cmds, envs)]
    logger.info(f"Launched {len(procs)} replica(s)")

    # Block until every process finishes, and propagate any non-zero exit codes
    exit_code = 0
    while len(procs) > 0:
        for i, p in enumerate(procs):
            try:
                # Check if process has finished without blocking
                if p.poll() is not None:
                    returncode = p.returncode
                    if returncode != 0:
                        logger.error(
                            f"Process {i} failed with return code {returncode}"
                        )
                        # Kill all remaining processes and exit
                        for remaining_proc in procs:
                            try:
                                remaining_proc.kill()
                            except Exception:
                                pass
                        sys.exit(returncode)
                    # Remove completed process from list
                    procs.remove(p)
                    logger.info(f"Process {i} completed successfully")
            except Exception as e:
                logger.exception(f"Error checking process {i}: {e}")
                # Terminate all remaining processes
                for remaining_proc in procs:
                    try:
                        remaining_proc.kill()
                    except Exception:
                        pass
                sys.exit(1)
        # Small sleep to prevent busy waiting
        time.sleep(0.1)

    logger.info("All replicas completed successfully")
    sys.exit(exit_code)
