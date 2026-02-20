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

"""Dispatch script for launching Cosmos RL jobs on SLURM clusters.

This script handles:
- Code sandbox: copies source code to output directory for reproducibility
- Auto-resume: automatically requeues jobs on preemption/timeout
- Signal handling: catches SIGUSR1 before job timeout
- Retry logic: handles transient failures with configurable retries
"""

import argparse
import copy
import datetime
import json
import logging
import math
import os
import subprocess
import sys
from argparse import REMAINDER
from shutil import copytree, ignore_patterns
from typing import Any, Callable, Dict, List, Literal

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import toml
from util import NodeLaunchMetadata, ReplicaLaunchMetadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger(__name__)


def get_git_revision_hash() -> str:
    """Returns the hash of the current git revision."""
    if os.path.exists(".git"):
        try:
            return (
                subprocess.check_output(["git", "rev-parse", "HEAD"])
                .decode("ascii")
                .strip()
            )
        except subprocess.CalledProcessError:
            return ""
    return ""


def get_git_remote_url() -> str:
    """Returns the URL of the remote Git repository."""
    if os.path.exists(".git"):
        try:
            return (
                subprocess.check_output(["git", "config", "--get", "remote.origin.url"])
                .decode("ascii")
                .strip()
            )
        except subprocess.CalledProcessError:
            return ""
    return ""


# Patterns to ignore when copying source code
IGNORE_PATTERNS_LIST = [
    "*.pyc",
    "*.tar",
    "*.egg-info",
    "__pycache__",
    "*.npz",
    "*.pt",
    "outputs",
    "build",
    "*.so",
    "*.safetensors",
    "*.npy",
    "*.pkl",
    "*.parquet",
    "*.mp4",
    "*.sqsh",
]

# Directories to ignore only at the top level
IGNORE_TOP_LEVEL_DIRS = {
    "wandb",
    ".venv",
    ".git",
    ".vscode",
    ".ruff_cache",
    ".pytest_cache",
    ".mypy_cache",
    "figures",
    "tests",
}


def make_ignore_function(src_root: str) -> Callable[[str, list[str]], set[str]]:
    """Create an ignore function for copytree that handles top-level-only ignores."""
    pattern_ignore = ignore_patterns(*IGNORE_PATTERNS_LIST)

    def ignore_func(directory: str, files: list[str]) -> set[str]:
        ignored = set(pattern_ignore(directory, files))
        if os.path.abspath(directory) == os.path.abspath(src_root):
            ignored.update(name for name in IGNORE_TOP_LEVEL_DIRS if name in files)
        return ignored

    return ignore_func


def dump_config_to_file(config: Dict[str, Any], output_path: str) -> None:
    """Write config to a TOML file, handling lora patterns specially.

    This function handles legacy dict-based policy.lora.{alpha_pattern,r_pattern}
    by appending them as literal sections to avoid regex key backslash escaping.
    """
    with open(output_path, "w") as f:
        config_for_dump = copy.deepcopy(config)
        lora_config = (config_for_dump.get("policy", {})).get("lora", {})
        alpha_pattern_table = lora_config.pop("alpha_pattern", None)
        r_pattern_table = lora_config.pop("r_pattern", None)

        toml.dump(config_for_dump, f)

        if isinstance(alpha_pattern_table, dict) and alpha_pattern_table:
            f.write("\n[policy.lora.alpha_pattern]\n")
            for key, value in alpha_pattern_table.items():
                f.write(f"'{key}' = {value}\n")

        if isinstance(r_pattern_table, dict) and r_pattern_table:
            f.write("\n[policy.lora.r_pattern]\n")
            for key, value in r_pattern_table.items():
                f.write(f"'{key}' = {value}\n")


def compute_nodes(
    n_gpu_per_node: int,
    n_gpu_per_replica: int,
    n_replicas: int,
    role: Literal["policy", "rollout"],
) -> List[NodeLaunchMetadata]:
    """Compute the number of nodes required for the given GPU and replica configuration.

    If multiple replicas are colocated on the same node, the visible GPUs for each
    replica are computed.

    Returns:
        A list of NodeLaunchMetadata, one for each node.
    """
    n_nodes = 0
    rendezvous_port = 29345

    node_launch_metadata = []
    if n_gpu_per_replica >= n_gpu_per_node:
        assert (
            n_gpu_per_replica % n_gpu_per_node == 0
        ), f"Number of GPUs per policy must be a multiple of {n_gpu_per_node}"
        n_policy_nodes = n_replicas * (n_gpu_per_replica // n_gpu_per_node)

        rendezvous_node = 0
        for i_node in range(n_policy_nodes):
            if i_node % (n_gpu_per_replica // n_gpu_per_node) == 0:
                rendezvous_node = i_node

            replica_launch_meta = [
                # Only one replica per node, no colocation or rendezvous conflicts
                ReplicaLaunchMetadata(
                    nnode=n_gpu_per_replica // n_gpu_per_node,
                    role=role,
                    rendezvous_node=rendezvous_node,
                    rendezvous_port=rendezvous_port,
                    visible_gpus=list(range(0, n_gpu_per_node)),
                )
            ]
            node_launch_metadata.append(
                NodeLaunchMetadata(colocation=replica_launch_meta)
            )
    else:
        possible_gpu_per_replica = []
        for divisor in range(1, n_gpu_per_node):
            if n_gpu_per_node % divisor == 0:
                possible_gpu_per_replica.append(divisor)

        assert (
            n_gpu_per_replica in possible_gpu_per_replica
        ), f"GPUs per policy must be one of {possible_gpu_per_replica}, got {n_gpu_per_replica}."
        n_policy_nodes = math.ceil(n_replicas * n_gpu_per_replica / n_gpu_per_node)

        replica_counter = 0
        for i_node in range(n_policy_nodes):
            replica_launch_meta = []
            local_replica_counter = 0
            while replica_counter < n_replicas:
                replica_launch_meta.append(
                    ReplicaLaunchMetadata(
                        nnode=1,
                        role=role,
                        rendezvous_node=i_node,  # Always on the same node
                        # Avoid port conflicts with other replicas on the same node
                        rendezvous_port=rendezvous_port + replica_counter,
                        visible_gpus=list(
                            range(
                                local_replica_counter * n_gpu_per_replica,
                                (local_replica_counter + 1) * n_gpu_per_replica,
                            )
                        ),
                    )
                )
                replica_counter += 1
                local_replica_counter += 1
                if replica_counter == n_replicas:
                    break
                elif local_replica_counter * n_gpu_per_replica >= n_gpu_per_node:
                    # Dispatch left to next node
                    break
            node_launch_metadata.append(
                NodeLaunchMetadata(colocation=replica_launch_meta)
            )
    n_nodes += n_policy_nodes

    return node_launch_metadata


def main():
    """Parse arguments and dispatch a Cosmos RL slurm job."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--job-name", type=str, default="cosmos_job")
    parser.add_argument(
        "--ngpu-per-node", type=int, default=8, help="Number of GPUs per compute node."
    )
    parser.add_argument(
        "--n-policy-replicas",
        type=int,
        default=None,
        help="Number of policy replicas to launch (default: use config value)",
    )
    parser.add_argument(
        "--n-rollout-replicas",
        type=int,
        default=None,
        help="Number of rollout replicas to launch (default: use config value)",
    )
    parser.add_argument(
        "--slurm-partition", type=str, default="pool0_av", help="SLURM partition to use"
    )
    parser.add_argument(
        "--slurm-account",
        type=str,
        default="av_alpamayo_reasoning",
        help="SLURM account to use",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to the controller config file",
    )
    parser.add_argument(
        "--output-root-path",
        type=str,
        required=True,
        help="Path to the output root directory",
    )
    parser.add_argument(
        "--container",
        type=str,
        required=True,
        help="Path to the container (Docker URI or .sqsh file)",
    )
    parser.add_argument(
        "--extra-sbatch-args",
        type=str,
        nargs="*",
        default=[],
        help="Extra #SBATCH arguments",
    )
    # Arguments for sandbox, autoresume, etc.
    parser.add_argument(
        "--no-copycode",
        dest="copycode",
        action="store_false",
        default=True,
        help="Disable code copying (default: enabled)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=4,
        help="Job duration in hours, supports decimals e.g. 1.5 (default: 4)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries on failure (default: 3)",
    )
    parser.add_argument(
        "--pre-timeout-signal",
        type=int,
        default=1200,
        help="Seconds before timeout to send SIGUSR1 (default: 1200)",
    )
    parser.add_argument(
        "--no-autoresume",
        dest="autoresume",
        action="store_false",
        default=True,
        help="Disable auto-resume (default: enabled)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sbatch script without submitting",
    )
    parser.add_argument(
        "launcher",
        nargs="?",  # "?" means 0 or 1 occurrences
        default="cosmos_rl.dispatcher.run_web_panel",
        help="The launcher module (default: cosmos_rl.dispatcher.run_web_panel). "
        "A custom launcher can be provided for custom dataset and reward functions.",
    )

    parser.add_argument("launcher_args", nargs=REMAINDER)

    args = parser.parse_args()

    # Get current user for proper $USER handling in container
    current_user = os.environ.get("USER", "unknown")

    # Generate timestamp-based run name
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    run_name = f"{args.job_name}_{timestamp}"
    output_dir = os.path.join(args.output_root_path, run_name)

    # Create output directory structure
    if not args.dry_run:
        os.makedirs(output_dir, exist_ok=True)
    slurm_dir = os.path.join(output_dir, "slurm")
    if not args.dry_run:
        os.makedirs(slurm_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Copy source code if enabled
    workdir = os.path.abspath("./")
    if args.copycode:
        code_dir = os.path.join(output_dir, "code")
        logger.info(f"Copying source code to {code_dir}")
        if not args.dry_run:
            copytree(
                "./",
                code_dir,
                ignore=make_ignore_function("./"),
                dirs_exist_ok=True,
            )
        workdir = code_dir
        logger.info(f"Will use workdir: {workdir}")

    with open(args.config_path) as f:
        config = toml.load(f)
    min_n_gpus_policy = (
        config["policy"]["parallelism"]["tp_size"]
        * config["policy"]["parallelism"]["pp_size"]
        * config["policy"]["parallelism"]["cp_size"]
        * config["policy"]["parallelism"]["dp_replicate_size"]
    )
    train_type = config["train"]["train_policy"]["type"]

    min_n_gpus_rollout = 0
    if "rollout" in config:
        # sft case may not have rollout config
        min_n_gpus_rollout = (
            config["rollout"]["parallelism"]["tp_size"]
            * config["rollout"]["parallelism"]["pp_size"]
        )
    if config["policy"]["parallelism"]["dp_shard_size"] >= 1:
        min_n_gpus_policy = (
            min_n_gpus_policy * config["policy"]["parallelism"]["dp_shard_size"]
        )

    # Determine n_policy_replicas and n_rollout_replicas from args or config
    n_policy_replicas = args.n_policy_replicas
    if n_policy_replicas is None:
        n_policy_replicas = config["policy"]["parallelism"].get("n_init_replicas", 1)
    n_rollout_replicas = args.n_rollout_replicas
    if n_rollout_replicas is None:
        if "rollout" in config and "parallelism" in config["rollout"]:
            n_rollout_replicas = config["rollout"]["parallelism"].get(
                "n_init_replicas", 1
            )
        else:
            n_rollout_replicas = 0  # SFT case

    # Update the n_init_replicas in the config
    if (
        "policy" in config
        and "parallelism" in config["policy"]
        and args.n_policy_replicas is not None
    ):
        config["policy"]["parallelism"]["n_init_replicas"] = n_policy_replicas
    if "rollout" in config and "parallelism" in config["rollout"]:
        # Only available for RL.
        config["rollout"]["parallelism"]["n_init_replicas"] = n_rollout_replicas

    # update output dir and timestamps
    config["train"]["output_dir"] = os.path.join(output_dir, "outputs", timestamp)
    config["train"]["timestamp"] = timestamp

    # Copy config to output directory
    config_output_dir = os.path.join(output_dir, "config")
    if not args.dry_run:
        os.makedirs(config_output_dir, exist_ok=True)
    config_path = os.path.join(config_output_dir, "config.toml")
    if not args.dry_run:
        dump_config_to_file(config, config_path)
    logger.info(f"Config written to {config_path}")

    policy_node_launch_metadata: List[NodeLaunchMetadata] = compute_nodes(
        args.ngpu_per_node, min_n_gpus_policy, n_policy_replicas, "policy"
    )
    n_policy_nodes = len(policy_node_launch_metadata)

    if train_type == "sft":
        rollout_node_launch_metadata = []
    else:
        rollout_node_launch_metadata: List[NodeLaunchMetadata] = compute_nodes(
            args.ngpu_per_node, min_n_gpus_rollout, n_rollout_replicas, "rollout"
        )
    n_rollout_nodes = len(rollout_node_launch_metadata)

    if args.launcher_args is not None:
        launcher_args = " ".join(args.launcher_args)
    else:
        launcher_args = ""

    # Git info for tracking
    git_commit = get_git_revision_hash()
    git_remote = get_git_remote_url()

    # Template for the slurm script
    template_vars = {
        "TOTAL_NODES": f"{n_policy_nodes + n_rollout_nodes}",
        "OUTPUT_DIR": output_dir,
        "SLURM_DIR": slurm_dir,
        "WORKDIR": workdir,
        "CONTAINER": args.container,
        "SLURM_PARTITION": args.slurm_partition,
        "SLURM_ACCOUNT": args.slurm_account,
        "SLURM_JOB_NAME": run_name,
        "CONFIG_PATH": config_path,
        "LAUNCHER": args.launcher,
        "LAUNCHER_ARGS": launcher_args,
        "EXTRA_SBATCH_ARGS": "\n".join(
            f"#SBATCH {arg}" for arg in args.extra_sbatch_args
        ),
        "DURATION": f"{int(args.duration)}:{int((args.duration % 1) * 60):02d}:00",
        "RETRIES": str(args.retries),
        "PRE_TIMEOUT_SIGNAL": str(args.pre_timeout_signal),
        "AUTORESUME": "1" if args.autoresume else "0",
        "SUBMIT_USER": current_user,
        "WANDB_GIT_COMMIT": git_commit,
        "WANDB_GIT_REMOTE_URL": git_remote,
        # Node configuration (explicit in template for cluster compatibility)
        "NUM_POLICY_NODES": str(n_policy_nodes),
        "NUM_ROLLOUT_NODES": str(n_rollout_nodes),
        "NODE_LAUNCH_METADATA_POLICY": json.dumps(
            [x.to_json() for x in policy_node_launch_metadata]
        ),
        "NODE_LAUNCH_METADATA_ROLLOUT": json.dumps(
            [x.to_json() for x in rollout_node_launch_metadata]
        ),
    }

    # Read the template relative to the current file
    with open(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "cosmos_rl_job_multi_node.sh"
        ),
    ) as f:
        template = f.read()

    # Replace the template variables
    for key, value in template_vars.items():
        template = template.replace(f"[[{key}]]", value)

    # Write the sbatch script to output directory
    sbatch_script_path = os.path.join(slurm_dir, "sbatch_script.sh")
    if not args.dry_run:
        with open(sbatch_script_path, "w") as f:
            f.write(template)
        os.chmod(sbatch_script_path, 0o755)
    logger.info(f"Sbatch script written to {sbatch_script_path}")

    if args.dry_run:
        logger.info("Dry run mode - sbatch script content:")
        print(template)
        return

    # All critical variables are now embedded in the template itself,
    # so we don't need to pass them via environment variables
    proc = subprocess.Popen(["sbatch", sbatch_script_path])
    proc.wait()
    if proc.returncode != 0:
        logger.error(f"Failed to submit job: {proc.returncode}")
        sys.exit(1)
    logger.info(f"Job submitted successfully. Output dir: {output_dir}")


if __name__ == "__main__":
    main()
