> [!IMPORTANT]
> ## 🚀 [Cosmos 3 Has Arrived](https://github.com/nvidia/cosmos)
>
> Cosmos 3 is NVIDIA's next-generation foundation model platform for Physical AI. Compared with Cosmos-RL, Cosmos 3 unifies reasoning, world prediction, simulation, transfer, and action generation within a single model family and ecosystem.
>
> Rather than relying on separate models for reasoning, prediction, transfer, and policy learning, a single Cosmos 3 model can understand the world, reason about physical interactions, predict future outcomes, transform observations across domains, and generate actions for embodied agents. This unified architecture enables stronger performance across a broad range of Physical AI applications, including robotics, autonomous vehicles, and smart spaces.
>
> This repository is no longer under active development and will receive only limited maintenance updates. Future model releases, features, documentation, and community support will be focused on Cosmos 3.
>
> 👉 Visit the new Cosmos home: https://github.com/nvidia/cosmos
>
> There you will find the latest Cosmos 3 models, technical reports, tutorials, benchmarks, and ecosystem updates.
>
> Thank you for your support of Cosmos-RL. We encourage all users to migrate to Cosmos 3 for the latest state-of-the-art Physical AI capabilities.

# Cosmos RL SLURM Launch Scripts

Launch Cosmos RL training jobs on SLURM clusters with support for:

- **Code sandbox**: Copies source code to output directory for reproducibility
- **Auto-resume**: Automatically requeues jobs on timeout
- **Signal handling**: Catches SIGUSR1 before job timeout for graceful shutdown
- **Retry logic**: Handles transient failures with configurable retries

## Prerequisites

- Have `${HOME}/.cache/huggingface` linked to your huggingface cache directory.
- Have `${CLUSTER_NAME}` environment variable defined in your `.bashrc`.
- Have ran `wandb login` to store your wandb api key in `${HOME}/.netrc`.

## Basic Usage

Run this on the login node of the slurm cluster:

```bash
python tools/slurm/dispatch_job.py \
    --job-name <job-name> \
    --config-path <cosmos-rl toml config file> \
    --output-root-path <output-root-path> \
    --container <container-image>
```

Note that the `python` env needs to have `toml` installed. You can use the python bin from

```bash
/lustre/fsw/portfolios/av/projects/av_alpamayo_reasoning/infra/infra_py/bin/python
```

## Full Example with All Options

```bash
python tools/slurm/dispatch_job.py \
    --job-name my_cosmos_job \
    --ngpu-per-node 8 \
    --n-policy-replicas 1 \
    --n-rollout-replicas 2 \
    --slurm-partition <slurm-partition> \
    --slurm-account <slurm-account> \
    --output-root-path <output-root-path> \
    --container <container-image> \
    --duration 4 \
    --retries 3 \
    --pre-timeout-signal 1200 \
    --config-path <config-path> \
    <launcher> -- <launcher-args>
```

## Command Line Arguments

| Argument                 | Type  | Req   | Default                     | Description                               |
| ------------------------ | ----- | ----- | --------------------------- | ----------------------------------------- |
| `--job-name`             | str   |       | `cosmos_job`                | Base name for the SLURM job               |
| `--ngpu-per-node`        | int   |       | 8                           | GPUs per node (usually no need to change) |
| `--n-policy-replicas`    | int   |       | config                      | Policy replicas (overrides config)        |
| `--n-rollout-replicas`   | int   |       | config                      | Rollout replicas (overrides config)       |
| `--slurm-partition`      | str   |       | `pool0_av`                  | SLURM partition                           |
| `--slurm-account`        | str   |       | `av_alpamayo_reasoning`     | SLURM account                             |
| `--config-path`          | str   | **Y** | -                           | Controller config file (TOML)             |
| `--output-root-path`     | str   | **Y** | -                           | Output root directory                     |
| `--repo-root-path`       | str   |       | None                        | Cosmos-rl repo root directory (to specify cosmos-rl code for execution otherwise executing the installed cosmos-rl in container) |
| `--container (--cosmos-container)` | str   | **Y** | -                 | Container (Docker URI or `.sqsh`)         |
| `--extra-sbatch-args`    | str[] |       | `[]`                        | Extra #SBATCH args                        |
| `--copycode`             | flag  |       | disabled                    | Enable code copying                      |
| `--duration`             | float |       | 4                           | Duration in hours (e.g., 1.5)             |
| `--slurm-job-time`       | str   |       | None                        | Duration in H:M:S format (e.g., 4:0:0), prior to `--duration`    |
| `--retries`              | int   |       | 3                           | Retries on failure                        |
| `--pre-timeout-signal`   | int   |       | 1200                        | Seconds before auto-resume                |
| `--no-autoresume`        | flag  |       | enabled                     | Disable auto-resume                       |
| `--dry-run`              | flag  |       | False                       | Print script only                         |
| `launcher`               | str   |       | `cosmos_rl...run_web_panel` | Custom launcher module                    |
| `launcher_args`          | str[] |       | -                           | Launcher args (after `--`)                |

## Output Directory Structure

When a job is submitted, the following directory structure is created:

```
{output-root-path}/{job-name}_{timestamp}/
├── code/                    # Copied source code (if copycode enabled)
├── config/
│   └── config.toml          # Processed config file
├── outputs/                  # Training outputs (checkpoints, logs, etc.)
├── slurm/
│   ├── sbatch_script.sh     # Generated sbatch script
│   └── {job_id}/
│       ├── part_0/          # First job execution segment
│       │   ├── run_0/       # First run attempt
│       │   │   ├── controller.out
│       │   │   ├── controller.err
│       │   │   ├── policy_0.out
│       │   │   ├── policy_0.err
│       │   │   ├── rollout_0.out
│       │   │   └── rollout_0.err
│       │   ├── run_1/       # Retry after transient failure
│       │   │   └── ...
│       │   └── remaining-retries  # Retry counter
│       ├── part_1/          # After autoresume (timeout)
│       │   └── ...
│       ├── latest_run -> part_X/run_Y  # Symlink to latest run
│       ├── latest_part -> part_X       # Symlink to latest part
│       └── slurm.log        # SLURM output log
└── ...
```

## Auto-Resume Behavior

The auto-resume feature handles two scenarios:

### 1. Timeout

When the job receives SIGUSR1 (sent `--pre-timeout-signal` seconds before timeout):

- All processes are gracefully terminated (with up to 130s grace period)
- Job is automatically requeued
- A new `part_N` directory is created
- Retry counter is reset

### 2. Transient Failures

When the job fails with non-zero exit code:

- Retry counter is decremented
- If retries remain, job is requeued in same `part_N`
- Creates a new `run_N` directory
- If no retries remain, job fails permanently

### Aborting Auto-Resume

To prevent a job from auto-resuming, create an abort file:

```bash
touch {output_dir}/ABORT-AUTORESUME
```

## Custom Launchers

For custom datasets and reward functions, provide a custom launcher:

```bash
python tools/slurm/dispatch_job.py \
    --config-path configs/my_config.toml \
    --output-root-path /path/to/output/ \
    --container /path/to/container.sqsh \
    tools/dataset/gsm8k_grpo.py -- --custom-flag
```

The launcher should be a Python module that registers custom datasets and reward functions with cosmos-rl.
