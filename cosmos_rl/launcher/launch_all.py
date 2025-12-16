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

#!/usr/bin/env python3

import socket
import subprocess
import sys
import time
import os
import shutil
import re
import argparse
from argparse import REMAINDER
from typing import List, Dict, Optional, Any, Callable
import toml
from cosmos_rl.policy.config.wfm import CosmosVisionGenConfig
from cosmos_rl.launcher.launch_vision import launch_vision_gen
from cosmos_rl.launcher.utility import (
    get_available_gpus,
    get_non_lepton_args,
    NUM_PRIORITY_MAPPING,
    set_lepton_job,
    get_worker_ip,
    resolve_host_blocking,
    launch_lepton_job,
    launch_processes,
    dump_config_with_literal_patterns_to_tmpfile,
)
from cosmos_rl.utils.logging import logger


def wait_for_url_ready(url: str, process: Optional[subprocess.Popen] = None):
    """
    Wait for a URL to be ready by sending a GET request.

    Args:
        url: The URL to check

    Returns:
        None
    """
    while True:
        # create TCP socket
        try:
            if process is not None:
                if process.poll() is not None:
                    if process.returncode != 0:
                        logger.error(
                            f"Process {process.pid} exited with code {process.returncode}. Exiting."
                        )
                        sys.exit(1)
                    else:
                        logger.error(
                            f"Process {process.pid} exited as soon as launched. Exiting."
                        )
                        sys.exit(1)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            host, port = url.split(":")
            sock.connect((host, int(port)))
            sock.close()
            break
        except socket.error:
            # If the connection fails, wait and retry
            time.sleep(1)


def read_config(config_file: str) -> Dict[str, Any]:
    """
    Read configuration from a TOML file.

    Args:
        config_file: Path to the TOML configuration file

    Returns:
        Dictionary containing the configuration
    """
    try:
        with open(config_file, "r") as f:
            config = toml.load(f)
        return config
    except Exception as e:
        logger.error(f"Error reading config file {config_file}: {e}")
        sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch multiple processes with GPU assignments"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to TOML configuration file, which specifies the detailed configuration for the whole training process including algorithm, model, data, parallelism, etc.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="URL of the controller for the policy and rollout replicas to connect to, consisting of IP and port in the format ip:port. If not provided, the controller will be launched on the local machine. If provided and the IP is the local IP, the controller will be launched on the local machine. If provided and the IP is not the local IP, the controller will be launched on the remote machine.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port of the controller to connect to, default is 8000. This is only used when --url is not provided to launch the controller on the local machine.",
    )
    parser.add_argument(
        "--policy",
        type=int,
        default=None,
        help="Total number of policy replicas to launch in the whole system. If not provided, the number of policy replicas will be obtained from TOML configuration file.",
    )
    parser.add_argument(
        "--rollout",
        type=int,
        default=None,
        help="Total number of rollout replicas to launch in the whole system. If not provided, the number of rollout replicas will be obtained from TOML configuration file.",
    )
    parser.add_argument(
        "--reference",
        type=int,
        default=None,
        help="Total number of reference replicas to launch in the whole system. If not provided, the number of reference replicas will be obtained from TOML configuration file.",
    )
    parser.add_argument(
        "--p2r-ratio",
        type=str,
        default=None,
        help="Ratio of policy replicas to rollout replicas. This is used to determine the number of rollout replicas and the number of policy replicas based on the number of workers.",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory to save logs. If not provided, logs will be printed to stdout.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of workers to use for the job, default is 1. This is used when multi-node training are used for the job.",
    )
    parser.add_argument(
        "--worker-idx",
        type=int,
        default=0,
        help="Worker index for local execution. In Lepton mode, this is ignored as worker indices are automatically assigned by Lepton.",
    )

    parser.add_argument(
        "--node-ip-list",
        type=str,
        default=None,
        help="list of ips for all the workers, separated by ';'. This is used when multi-node training are used for one replica.",
    )

    parser.add_argument(
        "--rdzv-port",
        type=int,
        default=29345,
        help="Rendezvous endpoint port for the job, default is 29345. This is used when multi-node training are used for one replica.",
    )

    parser.add_argument(
        "--lepton-mode",
        action="store_true",
        default=False,
        help="Enable Lepton mode for remote execution",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable log level to debug",
    )

    # Lepton specific options
    lepton_group = parser.add_argument_group("Lepton mode options")
    lepton_group.add_argument("--lepton-job-name", "-n", type=str, help="Job name")
    lepton_group.add_argument(
        "--lepton-container-image", type=str, help="Container image for the job"
    )
    lepton_group.add_argument(
        "--lepton-container-port",
        type=str,
        help="Ports to expose for the job, in the format portnumber[:protocol]",
        action="append",
    )
    lepton_group.add_argument(
        "--lepton-resource-shape", type=str, help="Resource shape for the pod"
    )
    lepton_group.add_argument(
        "--lepton-node-group",
        "-ng",
        type=str,
        help="Node group for the job",
        action="append",
    )
    lepton_group.add_argument(
        "--lepton-max-failure-retry",
        type=int,
        help="Maximum number of failures to retry per worker",
    )
    lepton_group.add_argument(
        "--lepton-max-job-failure-retry",
        type=int,
        help="Maximum number of failures to retry per whole job",
    )
    lepton_group.add_argument(
        "--lepton-env",
        "-e",
        type=str,
        help="Environment variables to pass to the job, in the format `NAME=VALUE`",
        action="append",
    )
    lepton_group.add_argument(
        "--lepton-secret",
        "-s",
        type=str,
        help="Secrets to pass to the job",
        action="append",
    )
    lepton_group.add_argument(
        "--lepton-mount",
        type=str,
        help="Persistent storage to be mounted to the job",
        action="append",
    )
    lepton_group.add_argument(
        "--lepton-image-pull-secrets",
        type=str,
        help="Secrets to use for pulling images",
        action="append",
    )
    lepton_group.add_argument(
        "--lepton-intra-job-communication",
        type=bool,
        help="Enable intra-job communication",
    )
    lepton_group.add_argument(
        "--lepton-privileged",
        action="store_true",
        help="Run the job in privileged mode",
    )
    lepton_group.add_argument(
        "--lepton-ttl-seconds-after-finished",
        type=int,
        help="TTL for finished jobs in seconds",
        default=259200,
    )
    lepton_group.add_argument(
        "--lepton-log-collection",
        "-lg",
        type=bool,
        help="Enable or disable log collection",
    )
    lepton_group.add_argument(
        "--lepton-node-id", "-ni", type=str, help="Node for the job", action="append"
    )
    lepton_group.add_argument(
        "--lepton-queue-priority",
        "-qp",
        type=int,
        choices=list(NUM_PRIORITY_MAPPING.keys()),
        help=(
            "Queue priority for dedicated node groups. Provide a number 1-9 which"
            " will be mapped to priority classes low-1000 … high-9000."
        ),
    )
    # Whether the job can be preempted by higher-priority jobs (only valid for
    # dedicated node groups). Tri-state: flag present → True; absent → None.
    lepton_group.add_argument(
        "--lepton-can-be-preempted",
        "-cbp",
        action="store_true",
        default=None,
        help=(
            "Allow this job to be preempted by higher priority jobs (only for"
            " dedicated node groups)."
        ),
    )

    # Whether the job itself is allowed to preempt lower-priority jobs.
    lepton_group.add_argument(
        "--lepton-can-preempt",
        "-cp",
        action="store_true",
        default=None,
        help=(
            "Allow this job to preempt lower priority jobs (only for dedicated"
            " node groups)."
        ),
    )
    lepton_group.add_argument(
        "--lepton-visibility", type=str, help="Visibility of the job (public/private)"
    )
    lepton_group.add_argument(
        "--lepton-shared-memory-size", type=int, help="Shared memory size in MiB"
    )
    lepton_group.add_argument(
        "--lepton-with-reservation",
        type=str,
        help="Reservation ID for dedicated node groups",
    )

    wfm_group = parser.add_argument_group("World foundational model mode options")
    wfm_group.add_argument(
        "--wfm-mode",
        "-dfg",
        action="store_true",
        default=False,
        help="In World foundational model mode.",
    )

    # Positional arguments
    parser.add_argument(
        "script",
        nargs="?",  # “?” means 0 or 1 occurrences
        default=None,
        help="A user script which can be provided for custom dataset, reward functions, and model registration.",
    )

    parser.add_argument("script_args", nargs=REMAINDER)

    args = parser.parse_args()

    # Validate Lepton mode arguments
    if args.lepton_mode:
        required_args = [("lepton_job_name", "--lepton-job-name")]

        for arg_name, arg_flag in required_args:
            if not getattr(args, arg_name):
                parser.error(f"{arg_flag} is required when --lepton-mode is enabled")

    return args, parser


def get_local_ip():
    """
    Get the local IP address of the machine.

    Returns:
        Local IP address as a string
    """
    try:
        import socket

        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        return [local_ip, hostname]
    except Exception as e:
        logger.error(f"Error getting local IP address: {e}")
        return None


def replica_placement(
    available_gpus: List[int],
    n_policy: int,
    n_rollouts: int,
    n_reference: int,
    min_n_gpus_policy: int,
    min_n_gpus_rollout: int,
    min_n_gpus_reference: int,
    replica_script: str,
    control_url: str,
    output_dir: Optional[str],
    args: argparse.Namespace,
    get_worker_ip: Optional[Callable] = None,
    rdzv_port: Optional[int] = None,
    script: Optional[str] = None,
    backend: str = "vllm",
    config_path: Optional[str] = None,
    script_args: Optional[List[Any]] = None,
) -> List[List[str]]:
    commands = []
    gpu_devices = []
    control_urls = []
    output_files = []
    assert len(available_gpus) in [
        1,
        2,
        4,
        8,
    ], "Number of GPUs per worker must be 1, 2, 4, or 8"
    # Prepare the command to launch the controller for all workers
    global_available_gpus = [available_gpus]
    # Create commands for policy and rollout replicas
    gpu_idx = 0
    global_worker_idx = 0
    global_launch_settings = []
    # Assign launch settings for each worker
    for i in range(n_policy):
        if min_n_gpus_policy > len(global_available_gpus[global_worker_idx]):
            assert (
                min_n_gpus_policy % len(global_available_gpus[global_worker_idx]) == 0
            ), f"min_n_gpus_policy {min_n_gpus_policy} is not divisible by {len(global_available_gpus[global_worker_idx])}"
            nodes_needed = min_n_gpus_policy // len(
                global_available_gpus[global_worker_idx]
            )
            rdzv_ip = "localhost"
            for node_in_replica in range(nodes_needed):
                gpu_devices.append(
                    ",".join([str(g) for g in global_available_gpus[global_worker_idx]])
                )
                commands.append(
                    f"{replica_script} --type policy --ngpus {len(global_available_gpus[global_worker_idx])} --nnodes {nodes_needed} --backend {backend} --config {config_path}"
                )
                if script is not None:
                    commands[-1] += f" --script {script}"
                if node_in_replica == 0:
                    commands[-1] += f" --rdzv-endpoint {rdzv_ip}:{rdzv_port}"
                    if get_worker_ip is not None:
                        rdzv_ip = get_worker_ip(global_worker_idx, args)
                else:
                    commands[-1] += f" --rdzv-endpoint {rdzv_ip}:{rdzv_port}"

                if script_args is not None:
                    commands[-1] += f" {' '.join(script_args)}"

                control_urls.append(control_url)
                output_files.append(
                    os.path.join(output_dir, f"policy_{i}.log")
                    if output_dir is not None
                    else None
                )
                global_launch_settings.append(
                    [commands, gpu_devices, control_urls, output_files]
                )
                commands = []
                gpu_devices = []
                control_urls = []
                output_files = []
                global_worker_idx += 1
                global_available_gpus.append(available_gpus)
        else:
            if gpu_idx + min_n_gpus_policy > len(
                global_available_gpus[global_worker_idx]
            ):
                global_launch_settings.append(
                    [commands, gpu_devices, control_urls, output_files]
                )
                commands = []
                gpu_devices = []
                control_urls = []
                output_files = []
                gpu_idx = 0
                global_worker_idx += 1
                global_available_gpus.append(available_gpus)

            gpu_devices.append(
                ",".join(
                    [
                        str(g)
                        for g in global_available_gpus[global_worker_idx][
                            gpu_idx : gpu_idx + min_n_gpus_policy
                        ]
                    ]
                )
            )
            commands.append(
                f"{replica_script} --type policy --ngpus {min_n_gpus_policy} --backend {backend} --config {config_path}"
            )
            if script is not None:
                commands[-1] += f" --script {script}"
            if script_args is not None:
                commands[-1] += f" {' '.join(script_args)}"
            control_urls.append(control_url)
            output_files.append(
                os.path.join(output_dir, f"policy_{i}.log")
                if output_dir is not None
                else None
            )
            gpu_idx += min_n_gpus_policy

    if min_n_gpus_rollout > len(global_available_gpus[global_worker_idx]):
        # If the number of GPUs needed for rollout is more than available GPUs, we need to allocate a new worker
        if gpu_idx > 0:
            global_launch_settings.append(
                [commands, gpu_devices, control_urls, output_files]
            )
            commands = []
            gpu_devices = []
            control_urls = []
            output_files = []
            gpu_idx = 0
            global_worker_idx += 1
            global_available_gpus.append(available_gpus)

    for i in range(n_rollouts):
        if min_n_gpus_rollout > len(global_available_gpus[global_worker_idx]):
            assert (
                min_n_gpus_rollout % len(global_available_gpus[global_worker_idx]) == 0
            ), f"min_n_gpus_rollout {min_n_gpus_rollout} is not divisible by {len(global_available_gpus[global_worker_idx])}"
            nodes_needed = min_n_gpus_rollout // len(
                global_available_gpus[global_worker_idx]
            )
            rdzv_ip = "localhost"
            for node_in_replica in range(nodes_needed):
                gpu_devices.append(
                    ",".join([str(g) for g in global_available_gpus[global_worker_idx]])
                )
                commands.append(
                    f"{replica_script} --type rollout --ngpus {len(global_available_gpus[global_worker_idx])} --nnodes {nodes_needed} --backend {backend} --config {config_path}"
                )
                if script is not None:
                    commands[-1] += f" --script {script}"
                if node_in_replica == 0:
                    commands[-1] += f" --rdzv-endpoint {rdzv_ip}:{rdzv_port}"
                    if get_worker_ip is not None:
                        rdzv_ip = get_worker_ip(global_worker_idx, args)
                else:
                    commands[-1] += f" --rdzv-endpoint {rdzv_ip}:{rdzv_port}"

                if script_args is not None:
                    commands[-1] += f" {' '.join(script_args)}"

                control_urls.append(control_url)
                output_files.append(
                    os.path.join(output_dir, f"rollout_{i}.log")
                    if output_dir is not None
                    else None
                )
                global_launch_settings.append(
                    [commands, gpu_devices, control_urls, output_files]
                )
                commands = []
                gpu_devices = []
                control_urls = []
                output_files = []
                global_worker_idx += 1
                global_available_gpus.append(available_gpus)
        else:
            if gpu_idx + min_n_gpus_rollout > len(
                global_available_gpus[global_worker_idx]
            ):
                global_launch_settings.append(
                    [commands, gpu_devices, control_urls, output_files]
                )
                commands = []
                gpu_devices = []
                control_urls = []
                output_files = []
                gpu_idx = 0
                global_worker_idx += 1
                global_available_gpus.append(available_gpus)

            gpu_devices.append(
                ",".join(
                    [
                        str(g)
                        for g in available_gpus[gpu_idx : gpu_idx + min_n_gpus_rollout]
                    ]
                )
            )
            commands.append(
                f"{replica_script} --type rollout --ngpus {min_n_gpus_rollout} --backend {backend} --config {config_path}"
            )
            if script is not None:
                commands[-1] += f" --script {script}"

            if script_args is not None:
                commands[-1] += f" {' '.join(script_args)}"

            control_urls.append(control_url)
            output_files.append(
                os.path.join(output_dir, f"rollout_{i}.log")
                if output_dir is not None
                else None
            )
            gpu_idx += min_n_gpus_rollout

    if min_n_gpus_reference > len(global_available_gpus[global_worker_idx]):
        # If the number of GPUs needed for reference is more than available GPUs, we need to allocate a new worker
        if gpu_idx > 0:
            global_launch_settings.append(
                [commands, gpu_devices, control_urls, output_files]
            )
            commands = []
            gpu_devices = []
            control_urls = []
            output_files = []
            gpu_idx = 0
            global_worker_idx += 1
            global_available_gpus.append(available_gpus)

    for i in range(n_reference):
        if min_n_gpus_reference > len(global_available_gpus[global_worker_idx]):
            assert (
                min_n_gpus_reference % len(global_available_gpus[global_worker_idx])
                == 0
            ), f"min_n_gpus_reference {min_n_gpus_reference} is not divisible by {len(global_available_gpus[global_worker_idx])}"
            nodes_needed = min_n_gpus_reference // len(
                global_available_gpus[global_worker_idx]
            )
            rdzv_ip = "localhost"
            for node_in_replica in range(nodes_needed):
                gpu_devices.append(
                    ",".join([str(g) for g in global_available_gpus[global_worker_idx]])
                )
                commands.append(
                    f"{replica_script} --type reference --ngpus {len(global_available_gpus[global_worker_idx])} --nnodes {nodes_needed} --backend {backend} --config {config_path}"
                )
                if script is not None:
                    commands[-1] += f" --script {script}"
                if node_in_replica == 0:
                    commands[-1] += f" --rdzv-endpoint {rdzv_ip}:{rdzv_port}"
                    if get_worker_ip is not None:
                        rdzv_ip = get_worker_ip(global_worker_idx, args)
                else:
                    commands[-1] += f" --rdzv-endpoint {rdzv_ip}:{rdzv_port}"

                if script_args is not None:
                    commands[-1] += f" {' '.join(script_args)}"

                control_urls.append(control_url)
                output_files.append(
                    os.path.join(output_dir, f"reference_{i}.log")
                    if output_dir is not None
                    else None
                )
                global_launch_settings.append(
                    [commands, gpu_devices, control_urls, output_files]
                )
                commands = []
                gpu_devices = []
                control_urls = []
                output_files = []
                global_worker_idx += 1
                global_available_gpus.append(available_gpus)
        else:
            if gpu_idx + min_n_gpus_reference > len(
                global_available_gpus[global_worker_idx]
            ):
                global_launch_settings.append(
                    [commands, gpu_devices, control_urls, output_files]
                )
                commands = []
                gpu_devices = []
                control_urls = []
                output_files = []
                gpu_idx = 0
                global_worker_idx += 1
                global_available_gpus.append(available_gpus)

            gpu_devices.append(
                ",".join(
                    [
                        str(g)
                        for g in available_gpus[
                            gpu_idx : gpu_idx + min_n_gpus_reference
                        ]
                    ]
                )
            )
            commands.append(
                f"{replica_script} --type reference --ngpus {min_n_gpus_reference} --backend {backend} --config {config_path}"
            )
            if script is not None:
                commands[-1] += f" --script {script}"

            if script_args is not None:
                commands[-1] += f" {' '.join(script_args)}"

            control_urls.append(control_url)
            output_files.append(
                os.path.join(output_dir, f"reference_{i}.log")
                if output_dir is not None
                else None
            )
            gpu_idx += min_n_gpus_reference

    if len(commands) > 0:
        global_launch_settings.append(
            [commands, gpu_devices, control_urls, output_files]
        )
    return global_launch_settings


def get_hostname_from_host(ip):
    try:
        # Run 'host' command
        result = subprocess.run(
            ["host", ip], capture_output=True, text=True, check=True
        )
        # Parse output
        output = result.stdout.strip()
        if "has address" in output:
            # Extract part after "has address "
            hostname = output.rsplit("has address ", 2)[-1].strip()
            return hostname
        else:
            return None
    except Exception as e:
        logger.error(f"Error: {e}")
        return None


def main():
    args, parser = parse_args()
    if args.debug:
        os.environ["COSMOS_LOG_LEVEL"] = "DEBUG"

    # Check if the config file is provided
    cosmos_config = read_config(args.config)

    if args.script is not None and args.script.endswith(".py"):
        # If the script is a Python file, we need to make sure it is absolute path
        # so that it can be found by the launched processes
        script = os.path.abspath(args.script)
    else:
        script = args.script if args.script is not None else None

    if args.wfm_mode:
        # launcher for vision gen task
        cosmos_config = CosmosVisionGenConfig.from_dict(read_config(args.config))
        logger.info(
            f"Launching vision gen task with config: {cosmos_config.model_dump()}"
        )
        return launch_vision_gen(cosmos_config, args, parser, script=script)

    # Get the number of GPUs required for policy and rollout
    # and the number of replicas for each
    policy_parallelism = cosmos_config.get("policy", {}).get("parallelism", {})
    rollout_parallelism = cosmos_config.get("rollout", {}).get("parallelism", {})
    reference_parallelism = cosmos_config.get("distillation", {}).get("parallelism", {})
    # Calculate the minimum number of GPUs required for policy and rollout
    # based on the parallelism settings in the configuration
    # Treat dp_shard_size as 1 if it is not set
    min_n_gpus_policy = (
        policy_parallelism.get("tp_size", 1)
        * policy_parallelism.get("dp_replicate_size", 1)
        * policy_parallelism.get("pp_size", 1)
        * policy_parallelism.get("cp_size", 1)
    )
    min_n_gpus_rollout = (
        rollout_parallelism.get("tp_size", 1)
        * rollout_parallelism.get("dp_replicate_size", 1)
        * rollout_parallelism.get("pp_size", 1)
        * rollout_parallelism.get("cp_size", 1)
    )
    min_n_gpus_reference = (
        reference_parallelism.get("tp_size", 1)
        * reference_parallelism.get("dp_replicate_size", 1)
        * reference_parallelism.get("pp_size", 1)
        * reference_parallelism.get("cp_size", 1)
    )
    if policy_parallelism.get("dp_shard_size", 1) >= 1:
        min_n_gpus_policy = min_n_gpus_policy * policy_parallelism.get(
            "dp_shard_size", 1
        )
    if rollout_parallelism.get("dp_shard_size", 1) >= 1:
        min_n_gpus_rollout = min_n_gpus_rollout * rollout_parallelism.get(
            "dp_shard_size", 1
        )
    if reference_parallelism.get("dp_shard_size", 1) >= 1:
        min_n_gpus_reference = min_n_gpus_reference * reference_parallelism.get(
            "dp_shard_size", 1
        )
    backend = cosmos_config.get("rollout", {}).get("backend", "vllm")
    logger.info(f"Using rollout backend: {backend}")

    if args.p2r_ratio is not None:
        assert (
            args.num_workers is not None
        ), "When using --p2r-ratio, --num-workers must be specified"
        p2r_ratio = args.p2r_ratio.split(":")
        assert (
            len(p2r_ratio) == 2
        ), "Invalid --p2r-ratio format. Use 'policy:rollout' format."
        p_ratio = int(p2r_ratio[0])
        r_ratio = int(p2r_ratio[1])

        if args.lepton_mode:
            match = re.search(r"(8|4|2)x", args.lepton_resource_shape)
            if match:
                num_gpus_per_node = int(match.group(1))
            else:
                num_gpus_per_node = 1
        else:
            num_gpus_per_node = len(get_available_gpus())

        num_per_ratio = (
            args.num_workers
            * num_gpus_per_node
            / (p_ratio * min_n_gpus_policy + r_ratio * min_n_gpus_rollout)
        )
        args.policy = int(num_per_ratio * p_ratio)
        args.rollout = int(num_per_ratio * r_ratio)
        args.reference = 0
        logger.warning(
            "Reference training is not supported yet in P2R ratio mode, set number of reference replicas to 0"
        )
        assert args.policy >= 1, "Number of policy replicas must be at least 1"
        assert (
            args.policy * min_n_gpus_policy + args.rollout * min_n_gpus_rollout
            <= args.num_workers * num_gpus_per_node
        )

    if args.policy is None:
        n_policy = policy_parallelism.get("n_init_replicas", 1)
    else:
        n_policy = args.policy
    if args.rollout is None:
        n_rollouts = rollout_parallelism.get("n_init_replicas", 1)
    else:
        n_rollouts = args.rollout
    if args.reference is None:
        n_reference = reference_parallelism.get("n_init_replicas", 1)
    else:
        n_reference = args.reference

    # If the training type is SFT, set n_rollouts to 0
    if (
        cosmos_config.get("train", {}).get("train_policy", {}).get("type", "grpo")
        == "sft"
    ):
        n_rollouts = 0
        n_reference = 0
        if n_policy > 1:
            logger.warning(
                "Warning: n_init_replicas for rollout is set to 0 for SFT training, but n_init_replicas for policy is more than 1."
            )
            pre_dp_replicate_size = policy_parallelism.get("dp_replicate_size", 1)
            cosmos_config["policy"]["parallelism"]["dp_replicate_size"] = (
                pre_dp_replicate_size * n_policy
            )
            logger.info(
                f"[Config ]SFT type job does not support n_init_replicas > 1, automatically set n_init_replicas from {n_policy} to 1 and scale up dp_replicate_size from {pre_dp_replicate_size} to {cosmos_config['policy']['parallelism']['dp_replicate_size']}."
            )
            min_n_gpus_policy = min_n_gpus_policy * n_policy
            n_policy = 1
    if not cosmos_config.get("distillation", {}).get("enable", False):
        n_reference = 0

    is_colocated = cosmos_config.get("mode", "disaggregated") == "colocated"
    if is_colocated:
        assert (
            n_policy == n_rollouts
        ), "Colocated mode only supports equal number of policy and rollout replicas"
        assert (
            min_n_gpus_policy == min_n_gpus_rollout
        ), "Colocated mode requires policy and rollout to have the same GPU requirements"

    # Handle Lepton mode
    if args.lepton_mode:
        from leptonai.api.v1.types.job import LeptonJobUserSpec

        # Create job specification
        job_spec = LeptonJobUserSpec()

        # Construct the original launch_processes command
        # Update policy and rollout numbers in the lepton config
        if "policy" in cosmos_config and "parallelism" in cosmos_config["policy"]:
            cosmos_config["policy"]["parallelism"]["n_init_replicas"] = n_policy
        if "rollout" in cosmos_config and "parallelism" in cosmos_config["rollout"]:
            cosmos_config["rollout"]["parallelism"]["n_init_replicas"] = n_rollouts
        if (
            "distillation" in cosmos_config
            and "parallelism" in cosmos_config["distillation"]
        ):
            cosmos_config["distillation"]["parallelism"]["n_init_replicas"] = (
                n_reference
            )
        config_content = toml.dumps(cosmos_config)
        launch_cmd = f"""\
cat >config.toml <<EOF
{config_content}
EOF

cosmos-rl --config config.toml"""
        non_lepton_args = get_non_lepton_args(args, parser)
        # Add all non-Lepton arguments to the command
        launch_cmd += " " + " ".join(non_lepton_args)
        if script is not None:
            launch_cmd += f" {script}"

        set_lepton_job(args, job_spec)

        global_launch_settings = replica_placement(
            list(range(num_gpus_per_node)),
            n_policy,
            0 if is_colocated else n_rollouts,
            n_reference,
            min_n_gpus_policy,
            min_n_gpus_rollout,
            min_n_gpus_reference,
            replica_script="",
            control_url="",
            output_dir=None,
            script=script,
            backend=backend,
            args=args,
        )
        if args.num_workers is not None:
            assert args.num_workers >= len(global_launch_settings)
        num_workers = len(global_launch_settings)
        logger.info(f"Number of workers required: {num_workers}")

        return launch_lepton_job(job_spec, num_workers, args, launch_cmd)

    import cosmos_rl.utils.util as util

    logger.info(
        f"Number of policy replicas: {n_policy} with {min_n_gpus_policy} gpus each"
    )
    logger.info(
        f"Number of rollout replicas: {n_rollouts} with {min_n_gpus_rollout} gpus each"
    )
    logger.info(
        f"Number of reference replicas: {n_reference} with {min_n_gpus_reference} gpus each"
    )

    # Get available GPUs
    available_gpus = get_available_gpus()
    if not available_gpus:
        raise RuntimeError("No GPUs available. Please check your GPU configuration.")

    # List of bash scripts to run (these should exist in the same directory)
    script_names = ["launch_controller.sh", "launch_replica.sh"]

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Verify scripts exist and are executable
    for script_name in script_names:
        script_path = os.path.join(script_dir, script_name)
        if not os.path.exists(script_path):
            logger.error(f"Error: Script {script_path} does not exist")
            sys.exit(1)
        if not os.access(script_path, os.X_OK):
            logger.error(f"Error: Script {script_path} is not executable")
            sys.exit(1)

    controller_script = os.path.join(script_dir, "launch_controller.sh")
    replica_script = os.path.join(script_dir, "launch_replica.sh")

    # Create commands for controller
    if args.log_dir is not None:
        output_dir = args.log_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        latest_dir = os.path.join(output_dir, "logs_latest")
        output_dir = os.path.join(output_dir, f"logs_{timestamp}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # create a symlink to the output_dir in the latest_dir
        if os.path.exists(latest_dir):
            os.remove(latest_dir)
        os.symlink(os.path.basename(output_dir), latest_dir)
    else:
        output_dir = None

    if (
        "LEPTON_JOB_WORKER_INDEX" in os.environ
        and int(os.environ.get("LEPTON_JOB_WORKER_INDEX")) >= 0
    ):
        cur_work_idx = int(os.environ.get("LEPTON_JOB_WORKER_INDEX"))
    else:
        cur_work_idx = args.worker_idx

    control_url = None
    if args.url is not None:
        ip, port = args.url.split(":")
        if ip in get_local_ip():
            # If the IP is the local IP, launch the controller on the local machine
            port = util.find_available_port(int(port))
            logger.info(f"Using local IP: {ip} so launching controller on port {port}")
        else:
            control_url = args.url
    else:
        if (
            "LEPTON_JOB_WORKER_INDEX" in os.environ
            and int(os.environ.get("LEPTON_JOB_WORKER_INDEX")) != 0
        ):
            # For non-primary workers, connect to the primary worker (index 0) using its hostname
            prefix = os.environ.get(
                "LEPTON_JOB_SERVICE_PREFIX", os.environ.get("LEPTON_JOB_NAME")
            )
            subdomain = os.environ.get("LEPTON_SUBDOMAIN", "")
            primary_hostname = f"{prefix}-0.{subdomain}"
            primary_hostname = resolve_host_blocking(primary_hostname)
            control_url = f"{primary_hostname}:{args.port}"
        elif "LEPTON_JOB_WORKER_INDEX" in os.environ:
            # If we're in a Lepton job prime node, check if the port is available
            if not util.is_port_free(args.port):
                raise RuntimeError(f"Port {args.port} is not available")
            else:
                port = args.port
        else:
            port = util.find_available_port(args.port)

    if control_url is None:
        logger.info(f"Controller will be launched locally on port {port}.")
    else:
        logger.info(
            f"Controller will be launched on another node. This node will connect to {control_url} for control."
        )

    controller_cmd = None
    tmpfile_toml = None
    if n_policy > 0 or n_rollouts > 0 or n_reference > 0:
        # Do not update the config if no replicas are needed which means launch controller only.
        if "policy" in cosmos_config and "parallelism" in cosmos_config["policy"]:
            cosmos_config["policy"]["parallelism"]["n_init_replicas"] = n_policy
        if "rollout" in cosmos_config and "parallelism" in cosmos_config["rollout"]:
            # Only available for RL.
            cosmos_config["rollout"]["parallelism"]["n_init_replicas"] = n_rollouts
        if (
            "distillation" in cosmos_config
            and "parallelism" in cosmos_config["distillation"]
        ):
            cosmos_config["distillation"]["parallelism"]["n_init_replicas"] = (
                n_reference
            )
    # Create a temporary file and write to it
    tmpfile_toml = dump_config_with_literal_patterns_to_tmpfile(cosmos_config)

    if control_url is None:
        logger.info(f"Temporary configuration file created at {tmpfile_toml}")
        controller_cmd = f"{controller_script} --config {tmpfile_toml}"
        controller_cmd += f" --port {port}"
        if script:
            controller_cmd += f" --script {script}"
        if args.script_args is not None:
            controller_cmd += f" {' '.join(args.script_args)}"
        control_url = f"localhost:{port}"

    global_launch_settings = replica_placement(
        available_gpus,
        n_policy,
        0 if is_colocated else n_rollouts,
        n_reference,
        min_n_gpus_policy,
        min_n_gpus_rollout,
        min_n_gpus_reference,
        replica_script,
        control_url,
        output_dir,
        get_worker_ip=get_worker_ip,
        rdzv_port=args.rdzv_port,
        script=script,
        backend=backend,
        config_path=tmpfile_toml,
        script_args=args.script_args,
        args=args,
    )

    num_workers = len(global_launch_settings)
    logger.info(f"Number of workers required: {num_workers}")
    if num_workers > 1:
        logger.info(
            "Multiple worker nodes will be used. Ensure that the launch script is excuted on all worker nodes."
        )
    assert (
        len(available_gpus) * num_workers
        >= min_n_gpus_policy * n_policy
        + min_n_gpus_rollout * (0 if is_colocated else n_rollouts)
        + min_n_gpus_reference * n_reference
    ), f"Not enough GPUs available. Required: {min_n_gpus_policy * n_policy + min_n_gpus_rollout * (0 if is_colocated else n_rollouts) + min_n_gpus_reference * n_reference}, Available: {len(available_gpus)}"

    if "LEPTON_JOB_WORKER_INDEX" in os.environ:
        prefix = os.environ.get(
            "LEPTON_JOB_SERVICE_PREFIX", os.environ.get("LEPTON_JOB_NAME")
        )
        subdomain = os.environ.get("LEPTON_SUBDOMAIN", "")
        hostname = f"{prefix}-{cur_work_idx}.{subdomain}"
        import cosmos_rl.utils.network_util as network_util

        ips = network_util.get_eth_ips()
        assert len(ips) > 0, "No IPs found for the current machine"
        logger.info(
            f"Setting hostname to {hostname} {ips[0]} for worker index {cur_work_idx}"
        )
        os.system(f"hostname {ips[0]}")
        if shutil.which("host") is not None:
            # Do a blocking wait until the hostname is properly resolved
            # Only do this if 'host' command is available
            idx = 0
            while idx < num_workers:
                remote_host = f"{prefix}-{idx}.{subdomain}"
                remote_hostname = get_hostname_from_host(remote_host)
                pattern = r"^\d+\.\d+\.\d+\.\d+$"
                if (
                    remote_hostname is not None
                    and re.match(pattern, remote_hostname) is not None
                ):
                    idx = idx + 1
                    continue
                else:
                    logger.info(
                        f"Waiting for hostname {remote_host} to be changed as ready, current: {remote_hostname}"
                    )
                    time.sleep(1)
    if (
        len(global_launch_settings) <= cur_work_idx
        or len(global_launch_settings[cur_work_idx]) == 0
    ):
        if controller_cmd is None:
            logger.info(
                f"No launch settings found for worker index {cur_work_idx}, no need launch"
            )
            sys.exit(0)

    processes = []

    controller_id = -1

    if controller_cmd is not None:
        controller_process = launch_processes(
            [controller_cmd],
            [""],
            [""],
            [
                os.path.join(output_dir, "controller.log")
                if output_dir is not None
                else None
            ],
        )
        controller_id = len(processes)
        processes.append(controller_process[0])

    logger.info(f"Waiting for controller to be ready at {control_url}")
    wait_for_url_ready(
        control_url, controller_process[0] if controller_cmd is not None else None
    )
    logger.info(f"Controller is ready at {control_url}")

    if (
        len(global_launch_settings) > cur_work_idx
        and len(global_launch_settings[cur_work_idx]) != 0
    ):
        commands = global_launch_settings[cur_work_idx][0]
        gpu_devices = global_launch_settings[cur_work_idx][1]
        control_urls = global_launch_settings[cur_work_idx][2]
        output_files = global_launch_settings[cur_work_idx][3]

        # Combine all commands
        logger.info(f"Commands to be executed: {commands}")
        logger.info(f"GPU devices to be used: {gpu_devices}")
        logger.info(f"Control URLs to be used: {control_urls}")
        logger.info(f"Output files: {output_files}")

        # Check if the number of GPU devices matches the number of commands
        assert (
            len(gpu_devices) == len(commands)
        ), f"Number of GPU devices ({len(gpu_devices)}) does not match number of commands ({len(commands)})"

        # Launch all processes
        processes.extend(
            launch_processes(commands, gpu_devices, control_urls, output_files)
        )

    # Wait for all processes to complete without blocking
    while len(processes) > 0:
        for i, process in enumerate(processes):
            try:
                # Check if process has finished without blocking
                if process.poll() is not None:
                    returncode = process.returncode
                    if returncode == 0:
                        logger.info(f"Process {i} completed successfully")
                    else:
                        logger.error(
                            f"Process {i} failed with return code {returncode}"
                        )
                        # Terminate all remaining processes
                        if controller_id == -1 or i == controller_id:
                            for p in processes:
                                try:
                                    p.kill()
                                except Exception as e:
                                    logger.error(f"Error kill process {p}: {e}")
                            logger.error("Terminated all processes due to failure")
                            sys.exit(1)  # Exit with error code 1 if any process failed
                    # Remove completed process from list
                    processes.remove(process)
            except Exception as e:
                logger.error(f"Error monitoring process {i}: {e}")
                # Terminate all remaining processes
                if controller_id == -1 or i == controller_id:
                    for p in processes:
                        try:
                            p.kill()
                        except Exception as e:
                            logger.error(f"Error kill process {p}: {e}")
                    logger.error("Terminated all processes due to error")
                    sys.exit(1)
        # Small sleep to prevent busy waiting
        time.sleep(0.1)

    if tmpfile_toml is not None and os.path.exists(tmpfile_toml):
        # Clean up the temporary file
        try:
            os.unlink(tmpfile_toml)
            tmpfile_toml = None
        except Exception as e:
            logger.error(f"Error deleting temporary file {tmpfile_toml}: {e}")


if __name__ == "__main__":
    main()
