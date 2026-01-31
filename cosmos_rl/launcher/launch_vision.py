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

from cosmos_rl.policy.config.wfm import CosmosVisionGenConfig
from cosmos_rl.launcher.utility import (
    get_available_gpus,
    get_non_lepton_args,
    set_lepton_job,
    get_worker_ip,
    launch_lepton_job,
    launch_processes,
    resolve_host_blocking,
    SingleWorkerCommands,
)
import tempfile
import time
import toml
import argparse
import re
import os
import sys
from typing import List, Optional, Callable, Dict
from cosmos_rl.utils.logging import logger


def replica_placement(
    available_gpus_per_node: List[int],
    num_replicas: int,
    min_n_gpus: int,
    args: argparse.Namespace,
    output_dir: Optional[str],
    get_worker_ip: Optional[Callable] = None,
    rdzv_port: Optional[int] = None,
    config_path: Optional[str] = None,
    control_url: Optional[str] = None,
    script: Optional[str] = None,
) -> List[List[str]]:
    commands: List[str] = []
    gpu_devices: List[Optional[str]] = []
    control_urls: List[Optional[str]] = []
    output_files: List[Optional[str]] = []
    envs: List[Optional[Dict[str, str]]] = []
    assert len(available_gpus_per_node) in [
        1,
        2,
        4,
        8,
    ], "Number of GPUs per worker must be 1, 2, 4, or 8"

    global_available_gpus = [available_gpus_per_node]
    gpu_idx = 0
    global_worker_idx = 0
    global_launch_settings = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    replica_script = os.path.join(script_dir, "launch_replica.sh")

    for i in range(num_replicas):
        if min_n_gpus > len(global_available_gpus[global_worker_idx]):
            # it needs multiple nodes
            assert (
                min_n_gpus % len(global_available_gpus[global_worker_idx]) == 0
            ), f"min_n_gpus {min_n_gpus} is not divisible by {len(global_available_gpus[global_worker_idx])}"
            nodes_needed = min_n_gpus // len(global_available_gpus[global_worker_idx])
            rdzv_ip = "localhost"
            for node_in_replica in range(nodes_needed):
                gpu_devices_for_node = ",".join(
                    [str(g) for g in global_available_gpus[global_worker_idx]]
                )
                gpu_devices.append(gpu_devices_for_node)

                command_for_node = f"{replica_script} --type policy --wfm-mode True --ngpus {len(global_available_gpus[global_worker_idx])} --nnodes {nodes_needed} --config {config_path}"
                if node_in_replica == 0:
                    command_for_node += f" --rdzv-endpoint {rdzv_ip}:{rdzv_port}"
                    if get_worker_ip is not None:
                        rdzv_ip = get_worker_ip(global_worker_idx, args)
                else:
                    command_for_node += f" --rdzv-endpoint {rdzv_ip}:{rdzv_port}"
                if script is not None:
                    command_for_node += f" --script {script}"

                commands.append(command_for_node)

                control_urls.append(control_url)
                output_files.append(
                    os.path.join(output_dir, f"wfm_gen_{i}.log")
                    if output_dir is not None
                    else None
                )
                envs.append(None)

                worker_commands = SingleWorkerCommands(global_worker_idx)
                worker_commands.extend_commands(
                    commands, gpu_devices, control_urls, output_files, envs
                )
                global_launch_settings.append(worker_commands)

                commands = []
                gpu_devices = []
                control_urls = []
                output_files = []
                envs = []
                global_worker_idx += 1
                global_available_gpus.append(available_gpus_per_node)
        else:
            if gpu_idx + min_n_gpus > len(global_available_gpus[global_worker_idx]):
                global_launch_settings.append(
                    [commands, gpu_devices, control_urls, output_files, envs]
                )
                commands = []
                gpu_devices = []
                control_urls = []
                output_files = []
                envs = []
                gpu_idx = 0
                global_worker_idx += 1
                global_available_gpus.append(available_gpus_per_node)

            gpu_devices.append(
                ",".join(
                    [
                        str(g)
                        for g in global_available_gpus[global_worker_idx][
                            gpu_idx : gpu_idx + min_n_gpus
                        ]
                    ]
                )
            )
            commands.append(
                f"{replica_script} --ngpus {min_n_gpus} --type policy --wfm-mode True --config {config_path} "
            )
            if script is not None:
                commands[-1] += f" --script {script}"

            control_urls.append(control_url)
            output_files.append(
                os.path.join(output_dir, f"wfm_gen_{i}.log")
                if output_dir is not None
                else None
            )
            envs.append(None)
            gpu_idx += min_n_gpus

    if len(commands) > 0:
        worker_commands = SingleWorkerCommands(global_worker_idx)
        worker_commands.extend_commands(
            commands,
            gpu_devices,
            control_urls,
            output_files,
            envs,
        )
        global_launch_settings.append(worker_commands)

        commands = []
        gpu_devices = []
        control_urls = []
        output_files = []
        envs = []

    return global_launch_settings


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


def launch_vision_gen(
    config: CosmosVisionGenConfig,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    script: Optional[str] = None,
):
    # calculate the minimum number of GPUs required for vision gen
    # TP + CP + PP + DP
    model_parallel_config = config.model_parallel
    data_parallel_size = model_parallel_config.data_parallel_size
    if data_parallel_size == -1:
        data_parallel_size = 1
    pipeline_model_parallel_size = model_parallel_config.pipeline_model_parallel_size
    context_parallel_size = model_parallel_config.context_parallel_size
    tensor_model_parallel_size = model_parallel_config.tensor_model_parallel_size
    fsdp_shard_size = config.model.fsdp_shard_size
    dp_replicate_size = config.model.dp_replicate_size
    min_n_gpus = (
        data_parallel_size
        * pipeline_model_parallel_size
        * tensor_model_parallel_size
        * context_parallel_size
        * fsdp_shard_size
        * dp_replicate_size
    )

    # We use HSDP (dp_replicate_size) for the multi-node case, so the number of replicas is 1
    num_replicas = 1
    logger.info(f"Number of replicas: {num_replicas}")

    if args.lepton_mode:
        match = re.search(r"(8|4|2)x", args.lepton_resource_shape)
        if match:
            num_gpus_per_node = int(match.group(1))
        else:
            num_gpus_per_node = 1
    else:
        num_gpus_per_node = len(get_available_gpus())

    if args.lepton_mode:
        # Create job specification
        from leptonai.api.v1.types.job import LeptonJobUserSpec

        job_spec = LeptonJobUserSpec()

        config_content = toml.dumps(config.model_dump())
        launch_cmd = f"""\
cat >config.toml <<EOF
{config_content}
EOF

cosmos-rl --config config.toml --wfm-mode"""

        # get all non-lepton arguments
        non_lepton_args = get_non_lepton_args(args, parser)

        launch_cmd += " " + " ".join(non_lepton_args)

        set_lepton_job(args, job_spec)

        global_launch_settings = replica_placement(
            list(range(num_gpus_per_node)),
            num_replicas,
            min_n_gpus,
            output_dir=None,
            get_worker_ip=None,
            rdzv_port=args.rdzv_port,
            config_path=args.config,
            args=args,
            script=script,
        )

        if args.num_workers is not None:
            assert args.num_workers >= num_replicas
        num_workers = len(global_launch_settings)
        logger.info(f"Number of workers required: {num_workers}")

        return launch_lepton_job(job_spec, num_workers, args, launch_cmd)

    logger.info(f"Number of World foundational model replicas: {num_replicas}")

    from cosmos_rl.utils import network_util

    # Get available GPUs
    available_gpus = get_available_gpus()

    if not available_gpus:
        raise RuntimeError("No GPUs available. Please check your GPU configuration.")

    # List of bash scripts to run (these should exist in the same directory)
    script_name = "launch_replica.sh"

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    script_path = os.path.join(script_dir, script_name)

    if not os.path.exists(script_path):
        logger.error(f"Error: Script {script_path} does not exist")
        sys.exit(1)
    if not os.access(script_path, os.X_OK):
        logger.error(f"Error: Script {script_path} is not executable")
        sys.exit(1)

    if args.log_dir is not None:
        output_dir = args.log_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(output_dir, f"logs_{timestamp}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        output_dir = None

    if (
        "LEPTON_JOB_WORKER_INDEX" in os.environ
        and int(os.environ.get("LEPTON_JOB_WORKER_INDEX")) >= 0
    ):
        cur_work_idx = int(os.environ.get("LEPTON_JOB_WORKER_INDEX"))
    else:
        cur_work_idx = args.worker_idx

    if "LEPTON_JOB_WORKER_INDEX" in os.environ:
        prefix = os.environ.get(
            "LEPTON_JOB_SERVICE_PREFIX", os.environ.get("LEPTON_JOB_NAME")
        )
        subdomain = os.environ.get("LEPTON_SUBDOMAIN", "")
        hostname = f"{prefix}-{cur_work_idx}.{subdomain}"
        ips = network_util.get_eth_ips()
        assert len(ips) > 0, "No IPs found for the current machine"
        logger.info(
            f"Setting hostname to {hostname} {ips[0]} for worker index {cur_work_idx}"
        )
        os.system(f"hostname {ips[0]}")

    # Create a temporary file and write to it
    # Use shutil to copy the original config file to avoid toml serialization issues
    import shutil

    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".toml", delete=False
    ) as tmpfile:
        tmpfile_toml = tmpfile.name

    # Copy the original config file
    shutil.copy(args.config, tmpfile_toml)

    control_url = None
    if args.url is not None:
        ip, port = args.url.split(":")
        if ip in get_local_ip():
            # If the IP is the local IP, launch the controller on the local machine
            port = network_util.find_available_port(int(port))
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
            if not network_util.is_port_free(args.port):
                raise RuntimeError(f"Port {args.port} is not available")
            else:
                port = args.port
        else:
            port = network_util.find_available_port(args.port)

    if control_url is None:
        logger.info(f"Controller will be launched locally on port {port}.")
    else:
        logger.info(
            f"Controller will be launched on another node. This node will connect to {control_url} for control."
        )

    if control_url is None:
        logger.info(f"Temporary configuration file created at {tmpfile_toml}")
        control_url = f"localhost:{port}"

    global_launch_settings = replica_placement(
        available_gpus,
        num_replicas,
        min_n_gpus,
        control_url=control_url,
        output_dir=output_dir,
        get_worker_ip=get_worker_ip,
        rdzv_port=args.rdzv_port,
        config_path=tmpfile_toml,
        args=args,
        script=script,
    )

    processes = []

    if (
        len(global_launch_settings) > cur_work_idx
        and len(global_launch_settings[cur_work_idx]) != 0
    ):
        command_collections = global_launch_settings[cur_work_idx]

        assert (
            command_collections.global_worker_idx == cur_work_idx
        ), f"Global worker index {command_collections.global_worker_idx} does not match current worker index {cur_work_idx}"

        logger.info(
            f"Command items to be executed for worker {command_collections.global_worker_idx}:\n{command_collections}"
        )

        # Launch all processes
        processes.extend(launch_processes(command_collections))

    while len(processes) > 0:
        for i, process in enumerate(processes):
            try:
                if process.poll() is not None:
                    returncode = process.returncode
                    if returncode == 0:
                        logger.info(f"Process {i} completed successfully")
                    else:
                        logger.error(
                            f"Process {i} failed with return code {returncode}"
                        )
                        for p in processes:
                            try:
                                p.kill()
                            except Exception as e:
                                logger.error(f"Error kill process {p}: {e}")
                        logger.error("Terminated all processes due to failure")
                        sys.exit(1)
                    # Remove completed process from list
                    processes.remove(process)
            except Exception as e:
                logger.error(f"Error monitoring process {i}: {e}")
        # Small sleep to prevent busy waiting
        time.sleep(0.1)

    if tmpfile_toml is not None and os.path.exists(tmpfile_toml):
        # Clean up the temporary file
        try:
            os.unlink(tmpfile_toml)
            tmpfile_toml = None
        except Exception as e:
            logger.error(f"Error deleting temporary file {tmpfile_toml}: {e}")

    return 0
