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

## Running full CI on Slurm

To run the full CI test suite (`tests/run_test.sh`, including the multi-GPU
`torchrun` tests) on a single 8-GPU Slurm node, use the helpers below. This
mirrors what GitHub Actions runs in `.github/workflows/build-and-test.yaml`, but
on a pyxis/enroot cluster.

Slurm compute/login nodes usually don't have a Docker daemon, so the typical
flow is: **build the `.sqsh` locally** (on a workstation that has `docker` +
`enroot`), **upload it to the login node**, then **submit the job** from the
login node.

### 1. Build the CI container image (locally)

`build_ci_image.sh` runs `docker build` (the same base image as GitHub CI), then
bakes `tests/` + `configs/` into a thin layer at `/workspace/cosmos-rl` (the base
Dockerfile removes the source tree, so this is what lets the job run without a
repo mount), and finally `enroot import`s the result into a `.sqsh`. Run it on a
host that has **both** `docker` and `enroot`:

```bash
# from the cosmos-rl repo root, on your local machine
bash tools/slurm/build_ci_image.sh --sqsh-out cosmos_rl_ci.sqsh
```

Useful flags: `--image-tag` (docker tag, default `cosmos_rl_ci:latest`),
`--no-clobber` (refuse to overwrite an existing `.sqsh`; by default it is
overwritten), `--no-build` (import an already-built local image),
`--no-bake-tests` (skip baking tests, then `--repo-root-path` is required at
launch), `--build-mode efa|no-efa` (default `no-efa`),
`--torch-variant 2.8|2.10`.

> Write the `.sqsh` **outside** the repo (e.g. `--sqsh-out ~/cosmos_rl_ci.sqsh`).
> The repo root is the Docker build context, so a multi-GB `.sqsh` left there gets
> pulled in by `COPY . /workspace/cosmos_rl` — busting the layer cache and making
> "exporting layers" crawl even with no source change.

A few CUDA extensions (flash-attn/FA3, apex, transformer_engine, grouped_gemm,
DeepEP) have no prebuilt wheels for this torch/CUDA/arch combo and are compiled
from source. These `nvcc` jobs are memory-hungry, so the build caps parallelism
via `--max-jobs N` (default: auto from RAM/cores). If the build OOMs, lower it
(e.g. `--max-jobs 4`); raise it to build faster on a big-RAM host. This maps to
the `MAX_JOBS` build-arg added to the root `Dockerfile`.

#### Reuse an existing container (skip the source build)

The from-source build is only needed to *produce* an image. If you already have
a cosmos-rl container, skip it entirely:

- **You already have a `.sqsh`** (e.g. the one your training jobs use): don't run
  the build at all. Launch directly and provide tests via `--repo-root-path`:

  ```bash
  ./cosmos_rl_ci_job.sh \
      --container /lustre/.../cosmos_rl.sqsh \
      --repo-root-path /lustre/.../cosmos-rl \
      --output-root-path /lustre/.../ci-runs
  ```

- **You have a published/registry image but not a `.sqsh`**: convert it with
  `--from-uri` (needs only `enroot`, no `docker`, no source build):

  ```bash
  bash tools/slurm/build_ci_image.sh \
      --from-uri docker://nvcr.io#nvidia/cosmos-rl:<tag> \
      --sqsh-out cosmos_rl_ci.sqsh
  ```

  Images built this way don't have `tests/` baked in, so `--repo-root-path` is
  required when launching.

### 2. Upload the launcher + `.sqsh` to the Slurm login node

`cosmos_rl_ci_job.sh` is a **standalone, self-submitting** launcher: copy just
this one file plus the `.sqsh` to the login node. No python, no template, and the
launcher itself does not need to live on shared storage (`sbatch` captures the
script into its spool). Put the `.sqsh` (and a repo checkout, see below) on
shared storage reachable by the compute nodes:

```bash
scp tools/slurm/cosmos_rl_ci_job.sh <login-node>:~/
scp cosmos_rl_ci.sqsh               <login-node>:/lustre/.../cosmos_rl_ci.sqsh
```

### 3. Submit the CI job (on the login node)

To reproduce GitHub's installed-package layout with a reused dependency image,
build a wheel from the exact candidate and pass `--package-wheel` together with
the matching checkout in `--repo-root-path`. This mode mounts only `tests/`,
installs the wheel into the ephemeral container's interpreter, and runs the
complete `tests/run_test.sh`. It does not overlay checkout source or replace
the suite with selected tests. Subprocesses use the installed candidate even
when they replace `PYTHONPATH`.

```bash
bash tools/slurm/cosmos_rl_ci_job.sh \
    --container /shared/cosmos_rl_ci.sqsh \
    --repo-root-path /shared/pinned-candidate \
    --package-wheel /shared/wheels/cosmos_rl-0.4.6-py3-none-any.whl \
    --test-deps-dir /shared/pytest-wheelhouse \
    --output-root-path /shared/ci-results \
    --slurm-account ACCOUNT --slurm-partition PARTITION
```

`--test-deps-dir` is optional; when provided it must contain pytest 8, GitHub's
`ucxx-cu12>=0.40.0` extra, and their dependencies for offline installation.
Otherwise the runner installs those requirements from the configured package
index. Setup verifies native UCXX imports and at least eight visible CUDA GPUs
before executing the suite. Without
`--package-wheel`, the existing source-overlay/baked-test modes remain available.
Per-suite logs and the environment manifest are retained under the run directory.
A passing suite against a reused dependency image does not validate the GitHub
Docker build; check dependency/image differences and required native-test skips.

With a pre-populated `--hf-cache`, installed-wheel runs can add
`--cached-model-paths`. This exposes cached main-revision model snapshots under
their usual `org/repo` paths inside the disposable test workspace. It avoids
repeated remote metadata requests for those models without modifying tests or
production code. The selected revision is logged, and existing workspace data
is never overwritten. Warm the required assets first: config-only snapshots,
empty checkpoints and missing indexed shards fail cache setup before the suite
starts. This is a presence check, not weight-content validation. This is not a
guarantee of network-free execution; uncached models
and datasets can still require the Hub, and remote-download behavior is not
exercised for locally aliased models. Use serialized cache preparation rather than
launching many cold model downloads against the same external rate limit.

To avoid copying large completed models for each run, also pass
`--readonly-model-cache /path/to/completed/cache/hub`. That directory is mounted
read-only and supplies the model aliases; `--hf-cache` remains a separate writable
runtime cache for datasets/downloads. The paths must not overlap. Use a completed,
stable cache: the read-only mount prevents this test container from changing it,
not another process on the host. Revision and complete-checkpoint checks still
apply, and the full suite is unchanged.

Run the launcher directly. With no `SLURM_JOB_ID` in the environment it parses
its args and `sbatch`es itself; Slurm then re-invokes the same file on the
compute node, where it runs `bash tests/run_test.sh` inside the container on a
single-node, 8-GPU, `--exclusive` allocation. The job's exit code reflects the CI
result (any failed test fails the job).

```bash
./cosmos_rl_ci_job.sh \
    --container /lustre/.../cosmos_rl_ci.sqsh \
    --output-root-path /lustre/.../ci-runs \
    --slurm-partition <partition> \
    --slurm-account <account>
```

`--repo-root-path` is **optional**. By default the job runs the `tests/` baked
into the image. Pass `--repo-root-path /lustre/.../cosmos-rl` to **override** the
baked-in code/tests with a working-tree checkout (mounted at `/opt/cosmos-rl`
and put on `PYTHONPATH`), so you can iterate on code/tests and re-run CI without
rebuilding the image. Logs land under `<output-root>/cosmos_ci_<timestamp>/slurm/`
(`slurm_<jobid>.log` plus a `run_<jobid>/run_test.log` with the per-test
PASS/FAIL summary). Because `cosmos_rl/_version.py` is generated and ignored,
mount mode copies that module from the installed package when the checkout does
not already contain it, before the checkout is added to `PYTHONPATH`.

Use `--dry-run` to print the exact `sbatch` command without submitting.

| Argument               | Req   | Default                 | Description                                  |
| ---------------------- | ----- | ----------------------- | -------------------------------------------- |
| `--container`          | **Y** | -                       | CI container `.sqsh` (from step 1) or URI    |
| `--output-root-path`   | **Y** | -                       | Output root directory for logs               |
| `--slurm-partition`    | **Y** | -                       | SLURM partition                              |
| `--slurm-account`      | **Y** | -                       | SLURM account                                |
| `--scratch-path`       |       | node-local `$SLURM_TMPDIR`/`$TMPDIR` | Writable scratch backing the container `/tmp` + caches; avoids `$HOME`/Lustre per-user quota (EDQUOT) on HF downloads |
| `--hf-cache`           |       | fresh dir under scratch | Pre-populated HuggingFace cache to reuse (mounted at `/root/.cache/huggingface`) |
| `--repo-root-path`     |       | None                    | Repo to mount and test, overriding baked-in  |
| `--package-wheel`      |       | None                    | Install the pinned wheel in the disposable interpreter, mounting only matching tests |
| `--test-deps-dir`      |       | None                    | Offline pytest/UCXX dependency wheelhouse |
| `--cached-model-paths` |       | False                   | Local aliases for cached model revisions; requires wheel mode and an explicit HF cache |
| `--readonly-model-cache` |     | None                    | Completed Hub model-cache directory mounted read-only for aliases; requires `--cached-model-paths` and a distinct runtime HF cache |
| `--job-name`           |       | `cosmos_ci`             | Base name for the SLURM job                  |
| `--ngpu-per-node`      |       | 8                       | GPUs to request                              |
| `--test-timeout`       |       | `2h`                    | `timeout` applied to `run_test.sh`           |
| `--duration`           |       | test-timeout + 30m      | Override SLURM `--time` (hours)              |
| `--slurm-job-time`     |       | None                    | Override SLURM `--time` in H:M:S             |
| `--extra-sbatch-arg`   |       | -                       | Extra `sbatch` arg (repeatable)              |
| `--dry-run`            |       | False                   | Print the sbatch command without submitting  |

**Troubleshooting**

- `OSError: ... Disk quota exceeded (os error 122)` (EDQUOT) during HF model
  downloads or `/tmp` writes: the container's scratch is landing on a
  quota-limited filesystem (often `$HOME`). By default scratch is node-local
  (`$SLURM_TMPDIR`/`$TMPDIR`); if your node lacks roomy local storage, point
  `--scratch-path` at a project space with quota headroom, and optionally
  `--hf-cache` at a pre-populated cache to avoid re-downloading models.
- `RuntimeError: FP8 is only supported for device that has compute capability
  8.9 or higher` (`test_fp8.py`): the FP8 path needs Ada/Hopper GPUs (L40S/H100,
  cc ≥ 8.9). It will fail on A100 (cc 8.0) regardless of this tooling — run the
  job on a cc ≥ 8.9 partition to exercise it.
