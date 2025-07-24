DGXC-Lepton Job
===============

If you already have access to the **DGX Cloud Lepton Platform**, you can launch your training job with the following steps:

1. Ensure the LeptonAI CLI is installed and you’re logged in to your workspace.
   ::

     pip install -U leptonai>=0.25.0

   Go to your workspace dashboard and generate the token, login with:
   ::

     lep login -c <workspace_id>:<token>

2. (OPTIONAL) In your training configuration file, set the `output_dir` to the mounted storage path.

3. Build your Docker image.

4. Push it to your container registry (e.g., NVIDIA NGC, Docker Hub).

5. Go to **DGX Lepton workspace dashboard → Settings** to configure your container **registries** and **secrets**.
   
   You can also quickly create a secret in your workspace using the CLI:
   ::

     lep secret create --name MY_HF_TOKEN --value hf_xxxxxxxxx

   In the examples below, we’ll reference this secret using the name `MY_HF_TOKEN`.

6. Include the image, registries, and secrets in the launch command as shown below.

7. (OPTIONAL) Ensure your job mounts the available **Local File System** as shown below.

8. Check available **node group** and **nodes** with:
   ::

     lep node list
     lep node list --detail

To launch your job, use the following command as a template. Make sure to **replace the placeholders** (e.g., image, config, secrets) with your own values.
::

  cosmos-rl \
  --config configs/qwen3/qwen3-8b-p-tp4-r-tp2-pp1-grpo.toml \
  --lepton-mode \
  --lepton-job-name <job_name> \
  --lepton-container-image <image> \
  --lepton-resource-shape <resource-type> \  # e.g., gpu.8xa100-80gb
  --lepton-node-group <node_group> \
  --lepton-image-pull-secrets <registry_name> \
  --lepton-secret HUGGING_FACE_HUB_TOKEN=MY_HF_TOKEN \  # Example usage of a secret. Make sure to setup 'MY_HF_TOKEN' in your workspace under Settings → Secrets.
  --lepton-env <ENVIRONMENT_VARIABLE_NAME>=<VALUE> \
  --lepton-mount /:<mount_path>:local-path-for-local:<local_disc_volume_name> \
  cosmos_rl.tools.dataset.gsm8k_grpo

.. warning::
   `--mount` currently only works for node groups that have **Local Disk Enabled**.

Example
-------

If your node group has **Local Disk enabled**, and you have a volume named `volume_A`, inside your volume there is a `dataset` folder you can mount that `dataset` on a specific path `job_dataset` in your job using `--lepton-mount`.
::

  --lepton-mount /dataset:/job_dataset:local-path-for-local:volume_A

For more available options, scroll down to [**Option Reference for `cosmos-rl` command**]


Valid Resource Shapes
----------------------

Please use the command `lep node list --detail` to view the available GPUs in the node group. 
Note: It is allowed to use CPU resource shapes within a GPU node group.


CPU Instances
-------------

- `cpu.small`
- `cpu.medium`
- `cpu.large`

GPU Instances
-------------

**NVIDIA A10:**
- `gpu.a10`
- `gpu.a10.6xlarge`

**NVIDIA A100 (40GB):**
- `gpu.a100-40gb`
- `gpu.2xa100-40gb`
- `gpu.4xa100-40gb`
- `gpu.8xa100-40gb`

**NVIDIA A100 (80GB):**
- `gpu.a100-80gb`
- `gpu.2xa100-80gb`
- `gpu.4xa100-80gb`
- `gpu.8xa100-80gb`

**NVIDIA H100 SXM:**
- `gpu.h100-sxm`
- `gpu.2xh100-sxm`
- `gpu.4xh100-sxm`
- `gpu.8xh100-sxm`


Option Reference for `cosmos-rl` command
--------------------------------------

.. list-table:: 
   :header-rows: 1

   * - Option (long)
     - Short
     - Type / Action
     - Default
     - Description
   * - `--config`
     - —
     - `str` (required)
     - —
     - Path to TOML configuration file (algorithm, model, data, parallelism…).
   * - `--url`
     - —
     - `str`
     - `None`
     - Controller URL `ip:port`; local controller if absent or IP is local.
   * - `--port`
     - —
     - `int`
     - `8000`
     - Controller port when `--url` is **not** provided.
   * - `--policy`
     - —
     - `int`
     - `None`
     - Total policy replicas (else read from TOML).
   * - `--rollout`
     - —
     - `int`
     - `None`
     - Total rollout replicas (else read from TOML).
   * - `--log-dir`
     - —
     - `str`
     - `None`
     - Directory for logs (stdout if not set).
   * - `--weight-sync-check`
     - `-wc`
     - `store_true`
     - `False`
     - Debug: check weight-sync correctness between policy and rollout.
   * - `--num-workers`
     - —
     - `int`
     - `1`
     - Workers used for multi-node training.
   * - `--worker-idx`
     - —
     - `int`
     - `0`
     - Local worker index.(**ignored in Lepton mode, which automatically assigns each replica’s index**.)
   * - `--lepton-mode`
     - —
     - `store_true`
     - `False`
     - Enable Lepton remote-execution mode.
   * - **Lepton-specific options**
     - 
     - 
     - 
     - 
   * - `--lepton-job-name`
     - `-n`
     - `str` (required)
     - `None`
     - Job name.(required in lepton mode)
   * - `--lepton-container-port`
     - —
     - `str`, `append`
     - `None`
     - Exposed ports `port[:protocol]` (repeatable).
   * - `--lepton-resource-shape`
     - —
     - `str`
     - `None`
     - Pod resource shape.
   * - `--lepton-node-group`
     - `-ng`
     - `str`, `append`
     - `None`
     - Target node group(s).
   * - `--lepton-max-failure-retry`
     - —
     - `int`
     - `None`
     - Max per-worker retries.
   * - `--lepton-max-job-failure-retry`
     - —
     - `int`
     - `None`
     - Max job-level retries.
   * - `--lepton-env`
     - `-e`
     - `str`, `append`
     - `None`
     - Env vars `NAME=VALUE` (repeatable).
   * - `--lepton-secret`
     - `-s`
     - `str`, `append`
     - `None`
     - Secrets (repeatable).
   * - `--lepton-mount`
     - —
     - `str`, `append`
     - `None`
     - Persistent storage mounts.
   * - `--lepton-image-pull-secrets`
     - —
     - `str`, `append`
     - `None`
     - Image-pull secrets.
   * - `--lepton-intra-job-communication`
     - —
     - `bool`
     - `None`
     - Enable intra-job communication.
   * - `--lepton-privileged`
     - —
     - `store_true`
     - `False`
     - Run in privileged mode.
   * - `--lepton-ttl-seconds-after-finished`
     - —
     - `int`
     - `259200`
     - TTL (s) for finished jobs.
   * - `--lepton-log-collection`
     - `-lg`
     - `bool`
     - `None`
     - Enable/disable log collection.
   * - `--lepton-node-id`
     - `-ni`
     - `str`, `append`
     - `None`
     - Specific node(s) to run on.
   * - `--lepton-queue-priority`
     - `-qp`
     - `int`
     - `None`
     - Queue priority for dedicated node groups (1-9, mapped to low-1000…high-9000).
   * - `--lepton-can-be-preempted`
     - `-cbp`
     - `store_true`
     - `False`
     - Allow this job to be preempted by higher priority jobs (dedicated node groups only).
   * - `--lepton-can-preempt`
     - `-cp`
     - `store_true`
     - `False`
     - Allow this job to preempt lower priority jobs (dedicated node groups only).
   * - `--lepton-visibility`
     - —
     - `str`
     - `None`
     - Job visibility (public/private).
   * - `--lepton-shared-memory-size`
     - —
     - `int`
     - `None`
     - Shared memory size (MiB).
   * - `--lepton-with-reservation`
     - —
     - `str`
     - `None`
     - Reservation ID for dedicated node groups.
