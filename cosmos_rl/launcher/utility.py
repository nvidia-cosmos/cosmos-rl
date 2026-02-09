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
from typing import List, Optional, Dict, Any, NamedTuple, Iterator, Callable
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
        return get_ip_from_list(worker_idx, args)
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


class CommandItem(NamedTuple):
    command: str
    # gpu device indexes used for this command
    gpu_devices: Optional[str]
    control_url: Optional[str]
    output_file: Optional[str]
    env: Optional[Dict[str, str]]

    def __repr__(self) -> str:
        return f"CommandItem(command={self.command}, gpu_devices={self.gpu_devices}, control_url={self.control_url}, output_file={self.output_file}, env={self.env})"


class SingleWorkerCommands:
    def __init__(self, global_worker_idx: int):
        # Commands to be executed on this worker
        self.global_worker_idx = global_worker_idx
        self.command_items: List[CommandItem] = []

    def append_command(
        self,
        command: str,
        gpu_devices: Optional[str],
        control_url: Optional[str],
        output_file: Optional[str],
        env: Optional[Dict[str, str]] = None,
    ):
        logger.info(
            f"Appending command: {command} with gpu devices: {gpu_devices}, control URL: {control_url}, output files: {output_file}, env: {env}"
        )
        self.command_items.append(
            CommandItem(command, gpu_devices, control_url, output_file, env)
        )

    def extend_commands(
        self,
        commands: List[str],
        gpu_devices: List[str],
        control_urls: List[str],
        output_files: List[str],
        envs: Optional[List[Dict[str, str]]] = None,
    ):
        logger.info(
            f"Extending commands for worker {self.global_worker_idx} with {len(commands)} commands"
        )
        assert (
            len(commands) == len(gpu_devices) == len(control_urls) == len(output_files)
        ), "The number of commands, gpu devices, control URLs, and output files must be the same"
        envs = envs or [None] * len(commands)
        for command, gpu_device, control_url, output_file, env in zip(
            commands, gpu_devices, control_urls, output_files, envs
        ):
            self.append_command(command, gpu_device, control_url, output_file, env)

    def __iter__(self) -> Iterator[CommandItem]:
        yield from self.command_items

    def __len__(self) -> int:
        return len(self.command_items)

    def __repr__(self) -> str:
        return "\n".join([repr(command_item) for command_item in self.command_items])


class Node:
    def __init__(self, worker_idx: int, available_gpus: List[int]):
        self.worker_idx = worker_idx
        self.available_gpus = available_gpus

        self.gpu_idx = 0
        self.launch_commands = SingleWorkerCommands(worker_idx)


class NodesManager:
    def __init__(
        self,
        replica_sh: str,
        available_gpus: List[int],
        controller_url: Optional[str] = None,
        output_dir: Optional[str] = None,
        config_path: Optional[str] = None,
        rdzv_port: Optional[int] = None,
        rl_mode: str = "disaggregated",
        backend: str = "vllm",
        get_worker_ip: Optional[Callable] = None,
    ):
        self.nodes: List[Node] = []  # Global nodes list.
        self.global_worker_idx = 0  # Current node index for replica assignment.
        self.gpu_idx = 0  # unused gpu start index for current node index.

        self.controller_url = controller_url
        self.output_dir = output_dir
        self.config_path = config_path
        self.rdzv_port = rdzv_port
        self.available_gpus = available_gpus
        self.rl_mode = rl_mode
        self.rollout_backend = backend
        self.get_worker_ip = get_worker_ip

        # Checks
        assert len(available_gpus) in [
            1,
            2,
            4,
            8,
        ], "Number of GPUs per node must be 1, 2, 4, or 8"

        # Commands, command-associated GPU devices, control URLs, output files, and environment variables.
        self.commands: List[str] = []
        self.gpu_devices: List[str] = []
        self.control_urls: List[str] = []
        self.output_files: List[str] = []
        self.envs: List[Dict[str, str]] = []

        # Backup states for rollback
        self.backuped_states = None

    def replica_placement(
        self,
        args: argparse.Namespace,
        n_policy: int,  # Number of policy replicas.
        n_rollouts: int,  # Number of rollout replicas.
        n_reference: int,  # Number of reference replicas.
        min_n_gpus_policy: int,  # Minimum number of GPUs per policy replica.
        min_n_gpus_rollout: int,  # Minimum number of GPUs per rollout replica.
        min_n_gpus_reference: int,  # Minimum number of GPUs per reference replica.
        replica_sh: str,
        script: Optional[str] = None,  # Entrypoint script for cosmos-rl.
        script_args: Optional[List[Any]] = None,
    ):
        if self.rl_mode in ["colocated"]:
            if n_rollouts > 0:
                n_rollouts = 0
                logger.warning(
                    f"Launching Cosmos-RL in colocated mode, rollout replicas will share the same devices as policy replicas, reset n_rollouts from {n_rollouts} to 0"
                )

        if self.rl_mode in ["colocated_separated"]:
            if n_rollouts <= 0:
                # If n_rollouts is not specified, set it equal to n_policy.
                n_rollouts = n_policy
                logger.warning(
                    f"Launching Cosmos-RL in colocated-separated mode, rollout replicas will share the same devices as policy replicas, reset n_rollouts from {n_rollouts} to {n_policy}"
                )

        # launch policy replicas, put it at first, fixed order.
        self.replica_placement_for_role(
            args=args,
            role="policy",
            n_replicas=n_policy,
            min_n_gpus_replica=min_n_gpus_policy,
            replica_sh=replica_sh,
            script=script,
            script_args=script_args,
        )

        # launch rollout rollout replicas, put it at last, fixed order.
        self.replica_placement_for_role(
            args=args,
            role="rollout",
            n_replicas=n_rollouts,
            min_n_gpus_replica=min_n_gpus_rollout,
            replica_sh=replica_sh,
            script=script,
            script_args=script_args,
        )

        # launch reference replicas if needed
        self.replica_placement_for_role(
            args=args,
            role="reference",
            n_replicas=n_reference,
            min_n_gpus_replica=min_n_gpus_reference,
            replica_sh=replica_sh,
            script=script,
            script_args=script_args,
        )

        # If commands list is not empty, we need to append the commands to the last node.
        if self.commands:
            node = self.creating_or_using_node()
            logger.info(
                f"Appending commands to node {self.global_worker_idx} with {len(self.commands)} commands"
            )
            node.launch_commands.extend_commands(
                self.commands,
                self.gpu_devices,
                self.control_urls,
                self.output_files,
                self.envs,
            )
            self.global_worker_idx += 1
            self.clear_list()

    def creating_or_using_node(self):
        if self.global_worker_idx + 1 <= len(self.nodes):
            node = self.nodes[self.global_worker_idx]
        else:
            node = Node(self.global_worker_idx, self.available_gpus)
            self.nodes.append(node)
        return node

    def replica_placement_for_role(
        self,
        args: argparse.Namespace,
        role: str,
        n_replicas: int,
        min_n_gpus_replica: int,
        replica_sh: str,
        script: Optional[str] = None,
        script_args: Optional[List[Any]] = None,
    ):
        if n_replicas == 0:
            return

        if (self.nodes and (min_n_gpus_replica > len(self.available_gpus))) or (
            self.rl_mode in ["colocated_separated"] and role == "rollout"
        ):
            # If:
            # 1. not in the first replica placement
            # 2. the number of GPUs available for the current node is not enough for the minimum number of GPUs per replica
            # 3. some cards has already been assigned to other replicas
            # 4. or in colocated-separated mode and this is a rollout role placement, we have to interrupt the current node's placement
            # then we need to increase the global worker index and allocate a new node.
            if self.gpu_idx > 0 or (
                self.rl_mode in ["colocated_separated"] and role == "rollout"
            ):
                node = self.creating_or_using_node()
                node.launch_commands.extend_commands(
                    self.commands,
                    self.gpu_devices,
                    self.control_urls,
                    self.output_files,
                    self.envs,
                )

                self.gpu_idx = 0
                self.global_worker_idx += 1

                # clear the commands, gpu devices, control URLs, output files, and environment variables for the current node.
                self.clear_list()

        # If in colocated-separated mode, we reset the global worker index and gpu index to 0.
        # To let rollout replicas use the same devices as policy replicas.
        self.backuped_states = (
            self.commands,
            self.gpu_devices,
            self.control_urls,
            self.output_files,
            self.envs,
            self.global_worker_idx,
            self.gpu_idx,
        )
        if self.rl_mode in ["colocated_separated"] and role == "rollout":
            logger.info(
                "Reset global worker index and gpu index to 0 for rollout replicas in colocated-separated mode"
            )
            self.clear_all(including_nodes=False)

        for i in range(n_replicas):
            if min_n_gpus_replica > len(self.available_gpus):
                # A single node is not enough for one replica
                nodes_needed = min_n_gpus_replica // len(self.available_gpus)
                rdzv_ip = "localhost"
                for node_in_replica in range(nodes_needed):
                    gpu_devices_for_node = ",".join(
                        [str(d) for d in self.available_gpus]
                    )
                    self.gpu_devices.append(gpu_devices_for_node)

                    command_for_node = f"{replica_sh} --type {role} --ngpus {len(self.available_gpus)} --nnodes {nodes_needed} --backend {self.rollout_backend} --config {self.config_path}"
                    if node_in_replica == 0:
                        command_for_node += (
                            f" --rdzv-endpoint {rdzv_ip}:{self.rdzv_port}"
                        )
                        if self.get_worker_ip is not None:
                            rdzv_ip = self.get_worker_ip(self.global_worker_idx, args)
                    else:
                        command_for_node += (
                            f" --rdzv-endpoint {rdzv_ip}:{self.rdzv_port}"
                        )

                    if script is not None:
                        command_for_node += f" --script {script}"
                    if script_args is not None:
                        command_for_node += f" {' '.join(script_args)}"

                    self.commands.append(command_for_node)

                    control_url_for_node = self.controller_url
                    output_file_for_node = (
                        os.path.join(self.output_dir, f"{role}_{i}.log")
                        if self.output_dir is not None
                        else None
                    )
                    self.control_urls.append(control_url_for_node)
                    self.output_files.append(output_file_for_node)
                    env_for_node = None
                    if self.rl_mode != "colocated_separated":
                        env_for_node = {
                            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
                        }
                    self.envs.append(env_for_node)

                    # Add a node or use existing node
                    node = self.creating_or_using_node()
                    node.launch_commands.extend_commands(
                        self.commands,
                        self.gpu_devices,
                        self.control_urls,
                        self.output_files,
                        self.envs,
                    )

                    self.global_worker_idx += 1
                    self.clear_list()
            else:
                # if the current node has enough GPUs for the minimum number of GPUs per replica, we can continue to place the next replica.
                if self.gpu_idx + min_n_gpus_replica > len(self.available_gpus):
                    # if the remaining GPUs are not enough for the minimum number of GPUs per replica, we need to move to a new node.
                    node = self.creating_or_using_node()
                    node.launch_commands.extend_commands(
                        self.commands,
                        self.gpu_devices,
                        self.control_urls,
                        self.output_files,
                        self.envs,
                    )
                    self.global_worker_idx += 1

                    self.clear_list()
                    self.gpu_idx = 0

                gpu_devices_for_replica = ",".join(
                    [
                        str(d)
                        for d in self.available_gpus[
                            self.gpu_idx : self.gpu_idx + min_n_gpus_replica
                        ]
                    ]
                )
                commands_for_replica = f"{replica_sh} --type {role} --ngpus {min_n_gpus_replica} --backend {self.rollout_backend} --config {self.config_path}"
                if script is not None:
                    commands_for_replica += f" --script {script}"
                if script_args is not None:
                    commands_for_replica += f" {' '.join(script_args)}"
                self.commands.append(commands_for_replica)
                self.gpu_devices.append(gpu_devices_for_replica)
                self.control_urls.append(self.controller_url)
                self.output_files.append(
                    os.path.join(self.output_dir, f"{role}_{i}.log")
                    if self.output_dir is not None
                    else None
                )
                env_for_replica = None
                if self.rl_mode != "colocated_separated":
                    env_for_replica = {
                        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
                    }
                self.envs.append(env_for_replica)

                self.gpu_idx += min_n_gpus_replica

        if self.rl_mode in ["colocated_separated"] and role == "rollout":
            (
                original_commands,
                original_gpu_devices,
                original_control_urls,
                original_output_files,
                original_envs,
                original_global_worker_idx,
                original_gpu_idx,
            ) = self.backuped_states
            if original_global_worker_idx < self.global_worker_idx:
                # We added new nodes to the previous nodes list, keep current worker index
                pass
            else:
                # We have not used all the previous nodes of policy.
                # First finalize existing commands
                if len(self.commands) > 0:
                    node = self.nodes[self.global_worker_idx]
                    node.launch_commands.extend_commands(
                        self.commands,
                        self.gpu_devices,
                        self.control_urls,
                        self.output_files,
                        self.envs,
                    )
                # recover the backuped states
                self.commands = original_commands
                self.gpu_devices = original_gpu_devices
                self.control_urls = original_control_urls
                self.output_files = original_output_files
                self.envs = original_envs
                self.global_worker_idx = original_global_worker_idx
                self.gpu_idx = original_gpu_idx

    def clear_list(self):
        # clear the commands, gpu devices, control URLs, output files, and environment variables for the current node.
        # Note: gpu_idx may not be 0 after clearing the list
        self.commands = []
        self.gpu_devices = []
        self.control_urls = []
        self.output_files = []
        self.envs = []

    def clear_all(self, including_nodes: bool = False):
        self.clear_list()
        self.global_worker_idx = 0
        self.gpu_idx = 0
        if including_nodes:
            self.nodes = []

    def finalize(self) -> List[SingleWorkerCommands]:
        return [node.launch_commands for node in self.nodes]


def launch_processes(
    command_collections: SingleWorkerCommands,
) -> List[subprocess.Popen]:
    """
    Launch multiple subprocesses and return their process objects.
    Returns:
        List of Popen objects for the launched processes
    """
    processes = []
    for command_item in command_collections.command_items:
        cmd = command_item.command
        gpu_id = command_item.gpu_devices
        url = command_item.control_url
        ofile = command_item.output_file
        command_env = command_item.env
        try:
            # Prepare environment variables
            env = dict(os.environ)
            if gpu_id is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpu_id
            if url is not None:
                env["COSMOS_CONTROLLER_HOST"] = url
            if command_env is not None:
                env.update(command_env)
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
            import traceback

            logger.error(traceback.format_exc())
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
