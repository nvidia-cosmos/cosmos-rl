.. _slurm-launch-job:

Slurm Job
====================

We provide a script to easily launch training jobs via Slurm on a cluster. For example, to launch a job with 1 policy replica, 2 rollout replicas, and 8 GPUs per node, use the following command:

.. code-block:: bash

    python ./tools/slurm/dispatch_job.py \
        --ngpu-per-node 8 \
        --n-policy-replicas 1 \
        --n-rollout-replicas 2  \
        --config-path configs/qwen3/qwen3-32b-p-fsdp1-tp8-r-tp4-pp1-grpo.toml \
        --repo-root-path ${COSMOS_PATH} \
        --output-root-path ${COSMOS_OUTPUT_PATH} \
        --cosmos-container ${COSMOS_SQSQ_PATH}

**Output:**

::

    Submitted batch job 2878971

The total number of GPU nodes is computed and dispatched automatically by the Slurm platform. The number of GPUs required per policy or rollout replica is determined from the configuration TOML file.

Custom Launcher
---------------

If you need a custom launcher to inject custom datasets and reward functions, you can pass a positional argument specifying the custom launcher file. Example launcher files are located in the ``tools/dataset`` folder.

For instance, to run a GRPO job with a custom launcher for the ``gsm8k`` dataset:

.. code-block:: bash

    python ./tools/slurm/dispatch_job.py \
        --ngpu-per-node 8 \
        --n-policy-replicas 1 \
        --n-rollout-replicas 2  \
        --config-path configs/qwen2-5/qwen2-5-32b-p-fsdp2-tp4-r-tp4-pp1-grpo-gsm8k.toml \
        --repo-root-path ${COSMOS_PATH} \
        --output-root-path ${COSMOS_OUTPUT_PATH} \
        --cosmos-container ${COSMOS_SQSQ_PATH} \
        --slurm-partition ${SLURM_PARTITION} \
        --slurm-account ${SLURM_ACCOUNT} \
        cosmos_rl.tools.dataset.gsm8k_grpo

Full Argument List
------------------

The full list of arguments accepted by ``dispatch_job.py`` is shown below:

.. list-table::
   :widths: 25 15 10 25 35
   :header-rows: 1

   * - Argument
     - Type
     - Required
     - Default Value
     - Description
   * - ``--job-name``
     - str
     - No
     - ``cosmos_job``
     - Name of the SLURM job.
   * - ``--ngpu-per-node``
     - int
     - No
     - 8
     - Number of GPUs per compute node.
   * - ``--n-policy-replicas``
     - int
     - No
     - 1
     - Number of policy replicas to launch.
   * - ``--n-rollout-replicas``
     - int
     - No
     - 1
     - Number of rollout replicas to launch.
   * - ``--slurm-partition``
     - str
     - No
     - ``batch``
     - SLURM partition to use.
   * - ``--slurm-account``
     - str
     - No
     - ``sw_aidot``
     - SLURM account to use.
   * - ``--config-path``
     - str
     - Yes
     - *(required)*
     - Path to the controller config file.
   * - ``--repo-root-path``
     - str
     - Yes
     - *(required)*
     - Path to the repository root.
   * - ``--output-root-path``
     - str
     - Yes
     - *(required)*
     - Path to the output root.
   * - ``--cosmos-container``
     - str
     - Yes
     - *(required)*
     - Path to the Cosmos container.
   * - ``--extra-sbatch-args``
     - str
     - No
     - ``["--gres=gpu:8"]``
     - Extra #SBATCH arguments.
   * - ``launcher``
     - str (positional)
     - No
     - ``cosmos_rl.dispatcher.run_web_panel``
     - Launcher to use for dataset-related operations. A custom launcher can be provided for custom dataset and reward function injection. See ``tools/dataset`` for examples.
