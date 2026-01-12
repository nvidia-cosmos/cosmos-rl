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
import subprocess
from typing import List, Optional, Dict, Any
import time
import argparse
import sys
import tempfile
import copy
import toml
from cosmos_rl.utils.logging import logger


# ---------------------------------------------------------------------------
# Queue priority helper
# ---------------------------------------------------------------------------
# Map numeric queue-priority (1-9) to the corresponding Lepton priority_class
# string expected by the backend API.
#
# 1-3  → low-1000 / 2000 / 3000
# 4-6  → mid-4000 / 5000 / 6000
# 7-9  → high-7000 / 8000 / 9000
#
# Note: keep in sync with lepton-cli definitions.
NUM_PRIORITY_MAPPING = {
    1: "low-1000",
    2: "low-2000",
    3: "low-3000",
    4: "mid-4000",
    5: "mid-5000",
    6: "mid-6000",
    7: "high-7000",
    8: "high-8000",
    9: "high-9000",
}


def get_available_gpus() -> List[str]:
    """
    Detect available GPUs using nvidia-smi and return their IDs.

    Returns:
        List of GPU IDs as strings
    """
    try:
        cmd = ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]
        cvd = os.getenv("CUDA_VISIBLE_DEVICES", None)
        if cvd is not None:
            # Add the GPU IDs to the command
            cmd += ["--id=" + cvd]
        # Run nvidia-smi to get GPU information
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Parse the output to get GPU IDs
        gpu_ids = [line.strip() for line in result.stdout.splitlines()]

        if not gpu_ids:
            logger.error("Warning: No GPUs detected")
            return []

        logger.info(f"Detected {len(gpu_ids)} GPUs: {', '.join(gpu_ids)}")
        return gpu_ids

    except subprocess.CalledProcessError as e:
        logger.error(f"Error running nvidia-smi: {e}")
        return []
    except Exception as e:
        logger.error(f"Error detecting GPUs: {e}")
        return []


def get_non_lepton_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> List[str]:
    # Get all non-Lepton arguments
    non_lepton_args = []
    for action in parser._actions:
        if hasattr(action, "option_strings") and action.option_strings:
            # Skip help action, lepton related arguments, and worker-idx
            if (
                action.dest == "help"
                or any(
                    opt.startswith("--lepton-") or opt == "--lepton-mode"
                    for opt in action.option_strings
                )
                or action.dest == "worker_idx"
                or action.dest == "config"
            ):  # skip worker-idx
                continue

            value = getattr(args, action.dest)
            if value is not None:
                if isinstance(value, bool):
                    if value:
                        non_lepton_args.append(action.option_strings[0])
                else:
                    non_lepton_args.append(f"{action.option_strings[0]} {value}")

    return non_lepton_args


def set_lepton_job(args: argparse.Namespace, job_spec):
    from leptonai.api.v1.types.job import LeptonJobUserSpec, LeptonResourceAffinity
    from leptonai.api.v1.types.deployment import QueueConfig
    from leptonai.config import VALID_SHAPES
    from leptonai.cli.util import (
        _get_valid_nodegroup_ids,
        _get_valid_node_ids,
    )

    assert isinstance(
        job_spec, LeptonJobUserSpec
    ), "job_spec must be a LeptonJobUserSpec object"
    # Handle node groups, queue priority and preemption flags
    if (
        args.lepton_node_group
        or args.lepton_queue_priority is not None
        or args.lepton_can_be_preempted is not None
        or args.lepton_can_preempt is not None
    ):
        if (
            args.lepton_queue_priority is not None
            or args.lepton_can_be_preempted is not None
            or args.lepton_can_preempt is not None
        ) and not args.lepton_node_group:
            logger.error(
                "Error: Queue priority is only available for dedicated node groups"
            )
            logger.error("Please use --lepton-queue-priority with --lepton-node-group")
            sys.exit(1)

        node_group_ids = _get_valid_nodegroup_ids(
            args.lepton_node_group,
            need_queue_priority=(
                args.lepton_queue_priority is not None
                or args.lepton_can_be_preempted is not None
                or args.lepton_can_preempt is not None
            ),
        )
        valid_node_ids = (
            _get_valid_node_ids(node_group_ids, args.lepton_node_id)
            if args.lepton_node_id
            else None
        )

        job_spec.affinity = LeptonResourceAffinity(
            allowed_dedicated_node_groups=node_group_ids,
            allowed_nodes_in_node_group=valid_node_ids,
        )

        if (
            args.lepton_queue_priority is not None
            or args.lepton_can_be_preempted is not None
            or args.lepton_can_preempt is not None
        ):
            # Ensure queue_config exists
            if job_spec.queue_config is None:
                job_spec.queue_config = QueueConfig()

            priority_class = None
            if args.lepton_queue_priority is not None:
                # Convert numeric priority to the Lepton priority_class string.
                priority_class = NUM_PRIORITY_MAPPING[args.lepton_queue_priority]

            job_spec.queue_config.priority_class = priority_class or "mid-4000"

            if args.lepton_can_be_preempted is not None:
                job_spec.queue_config.can_be_preempted = bool(
                    args.lepton_can_be_preempted
                )

            if args.lepton_can_preempt is not None:
                job_spec.queue_config.can_preempt = bool(args.lepton_can_preempt)

    # Set resource shape
    if args.lepton_resource_shape:
        job_spec.resource_shape = args.lepton_resource_shape
    else:
        available_types = "\n      ".join(VALID_SHAPES)
        logger.error(
            "Error: Missing option '--lepton-resource-shape'.\n"
            f"Available types are:\n      {available_types}.\n"
        )
        sys.exit(1)


def resolve_host(host):
    try:
        result = subprocess.run(
            ["getent", "hosts", "--", host],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode == 0:
            if len(result.stdout.strip().split()) > 0:
                return result.stdout.strip().split()[0]
            else:
                return None
        else:
            raise RuntimeError(f"Resolution failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        raise TimeoutError("DNS resolution timed out")


def resolve_host_blocking(hostname):
    try:
        while True:
            new_hostname = resolve_host(hostname)
            if new_hostname is not None:
                hostname = new_hostname
                logger.info(f"Resolved hostname: {hostname}")
                break
            time.sleep(1)
    except Exception:
        pass
    return hostname


def get_lepton_ip(worker_idx: int) -> str:
    if "LEPTON_JOB_WORKER_INDEX" in os.environ:
        # For non-primary workers, connect to the primary worker (index 0) using its hostname
        prefix = os.environ.get(
            "LEPTON_JOB_SERVICE_PREFIX", os.environ.get("LEPTON_JOB_NAME")
        )
        subdomain = os.environ.get("LEPTON_SUBDOMAIN", "")
        hostname = f"{prefix}-{worker_idx}.{subdomain}"
        hostname = resolve_host_blocking(hostname)
    else:
        raise RuntimeError("Lepton job worker index not found in environment variables")
    return hostname


def get_ip_from_list(worker_idx: int, args: argparse.Namespace) -> str:
    if args.node_ip_list is not None:
        logger.info(f"Node IP list provided: {args.node_ip_list}")
        ip_list = args.node_ip_list.split(";")
        logger.info(f"Node IP list: {ip_list}")
        if worker_idx < len(ip_list):
            return ip_list[worker_idx]
        else:
            raise RuntimeError(
                f"Worker index {worker_idx} exceeds the length of the IP list"
            )
    else:
        raise RuntimeError("Node IP list not provided")


def get_worker_ip(worker_idx: int, args: argparse.Namespace) -> str:
    if "LEPTON_JOB_WORKER_INDEX" in os.environ:
        return get_lepton_ip(worker_idx)
    elif args.node_ip_list is not None:
        return get_ip_from_list(worker_idx)
    else:
        raise RuntimeError(
            "Replica with GPUs larger than 8 occurs but not on Lepton job, please specify --node-ip-list to provide the IPs of all nodes to enable conenctions to each Rendezvous head node."
        )


def launch_lepton_job(
    job_spec,
    num_workers: int,
    args: argparse.Namespace,
    launch_cmd: str,
):
    from leptonai.api.v1.types.job import LeptonJob, LeptonJobUserSpec
    from leptonai.api.v1.types.deployment import LeptonLog
    from leptonai.config import BASE_IMAGE
    from leptonai.api.v1.types.common import Metadata, LeptonVisibility
    from leptonai.api.v1.photon import (
        make_env_vars_from_strings,
        make_mounts_from_strings,
    )
    from leptonai.cli.util import make_container_port_from_string
    from leptonai.api.v1.types.deployment import ReservationConfig
    from leptonai.api.v2.client import APIClient

    assert isinstance(
        job_spec, LeptonJobUserSpec
    ), "job_spec must be a LeptonJobUserSpec object"
    # Handle workers and communication
    if num_workers > 0:
        job_spec.completions = num_workers
        job_spec.parallelism = num_workers
        job_spec.intra_job_communication = True
    elif args.lepton_intra_job_communication is not None:
        job_spec.intra_job_communication = args.lepton_intra_job_communication

    # Set failure retry settings
    if args.lepton_max_failure_retry:
        job_spec.max_failure_retry = args.lepton_max_failure_retry
    if args.lepton_max_job_failure_retry:
        job_spec.max_job_failure_retry = args.lepton_max_job_failure_retry

    # Handle command
    job_spec.container.command = ["/bin/bash", "-c", launch_cmd]

    # Set container image
    if args.lepton_container_image:
        job_spec.container.image = args.lepton_container_image
    else:
        job_spec.container.image = BASE_IMAGE

    # Handle ports
    if args.lepton_container_port:
        job_spec.container.ports = [
            make_container_port_from_string(p) for p in args.lepton_container_port
        ]

    # Handle environment variables and secrets
    if args.lepton_env or args.lepton_secret:
        job_spec.envs = make_env_vars_from_strings(args.lepton_env, args.lepton_secret)

    # Handle mounts
    if args.lepton_mount:
        job_spec.mounts = make_mounts_from_strings(args.lepton_mount)

    # Set other configurations
    if args.lepton_image_pull_secrets:
        job_spec.image_pull_secrets = args.lepton_image_pull_secrets
    if args.lepton_privileged:
        job_spec.privileged = args.lepton_privileged
    if args.lepton_ttl_seconds_after_finished:
        job_spec.ttl_seconds_after_finished = args.lepton_ttl_seconds_after_finished
    if args.lepton_log_collection is not None:
        job_spec.log = LeptonLog(enable_collection=args.lepton_log_collection)
    if args.lepton_shared_memory_size is not None:
        job_spec.shared_memory_size = args.lepton_shared_memory_size

    # Handle reservation
    if args.lepton_with_reservation:
        if not args.lepton_node_group:
            logger.error(
                "Error: --lepton-with-reservation is only supported for dedicated node groups"
            )
            sys.exit(1)
        job_spec.reservation_config = ReservationConfig(
            reservation_id=args.lepton_with_reservation
        )

    # Create job
    job = LeptonJob(
        spec=job_spec,
        metadata=Metadata(
            id=args.lepton_job_name,
            visibility=LeptonVisibility(args.lepton_visibility)
            if args.lepton_visibility
            else None,
        ),
    )

    # Initialize Lepton client
    client = APIClient()
    # Create the job
    created_job = client.job.create(job)
    new_job_id = created_job.metadata.id_
    logger.info("🎉 Job Created Successfully!")
    logger.info(f"Name: {args.lepton_job_name}")
    logger.info(f"ID: {new_job_id}")


def launch_processes(
    commands: List[str],
    gpu_devices: Optional[List[str]],
    control_urls: Optional[List[str]],
    output_files: Optional[List[str]],
    extra_env: Optional[Dict[str, str]] = None,
) -> List[subprocess.Popen]:
    """
    Launch multiple subprocesses and return their process objects.

    Args:
        commands: List of command strings to execute
        gpu_devices: List of GPU device IDs to assign to each process (e.g., ["0", "1", "2"])
        control_urls: List of controller URLs to assign to each process (e.g., ["localhost:8000"])
        output_files: List of output files to redirect process output to (e.g., ["output1.log", "output2.log"])

    Returns:
        List of Popen objects for the launched processes
    """
    processes = []

    if gpu_devices is None:
        gpu_devices = [None] * len(commands)
    elif len(gpu_devices) != len(commands):
        raise ValueError("Number of GPU devices must match number of commands")

    for cmd, gpu_id, url, ofile in zip(
        commands, gpu_devices, control_urls, output_files
    ):
        try:
            # Prepare environment variables
            env = dict(os.environ)
            if gpu_id is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpu_id
            if url is not None:
                env["COSMOS_CONTROLLER_HOST"] = url
            if extra_env is not None:
                env.update(extra_env)
            if ofile is not None:
                f = open(ofile, "wb")
                cout = f
                cerr = f
            else:
                cout = sys.stdout
                cerr = sys.stderr

            # Launch process and capture output
            logger.info(f"Launching process with command: {cmd}")
            process = subprocess.Popen(
                cmd, shell=True, stdout=cout, stderr=cerr, env=env
            )
            processes.append(process)
            if ofile is not None:
                f.close()
        except Exception as e:
            logger.error(f"Error launching process for command '{cmd}': {e}")

    return processes


def dump_config_with_literal_patterns_to_tmpfile(config: Dict[str, Any]) -> str:
    """
    Write config to TOML, while emitting legacy dict-based
    policy.lora.{alpha_pattern,r_pattern} as literal sections to avoid
    backslash-escaping of regex keys.
    """
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".toml", delete=False
    ) as tmp_file:
        config_for_dump = copy.deepcopy(config)
        lora_config = (config_for_dump.get("policy", {})).get("lora", {})
        alpha_pattern_table = lora_config.pop("alpha_pattern", None)
        r_pattern_table = lora_config.pop("r_pattern", None)

        toml.dump(config_for_dump, tmp_file)

        if isinstance(alpha_pattern_table, dict) and alpha_pattern_table:
            tmp_file.write("\n[policy.lora.alpha_pattern]\n")
            for key, value in alpha_pattern_table.items():
                tmp_file.write(f"'{key}' = {value}\n")

        if isinstance(r_pattern_table, dict) and r_pattern_table:
            tmp_file.write("\n[policy.lora.r_pattern]\n")
            for key, value in r_pattern_table.items():
                tmp_file.write(f"'{key}' = {value}\n")

        return tmp_file.name
