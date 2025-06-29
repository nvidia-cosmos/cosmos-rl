# Launch Job by Slurm

We create a script for easily launching training job by slurm on the cluster. For example, if you want to launch a job with `1` replica for policy, `2` replica for rollout, 8 GPUs per node, you can run the following commands:
```bash
python ./tools/slurm/dispatch_job.py \
    --ngpu-per-node 8 \
    --n-policy-replicas 1 \
    --n-rollout-replicas 2  \
    --config-path configs/qwen3/qwen3-32b-p-fsdp1-tp8-r-tp4-pp1-grpo.toml \
    --repo-root-path ${COSMOS_PATH} \
    --output-root-path ${COSMOS_OUTPUT_PATH} \
    --cosmos-container ${COSMOS_SQSQ_PATH}


### Output:
Submitted batch job 2878971
```
Total number of GPU node is computed and dispatched via slurm platform automatically. Number of GPUs each policy or rollout need is automatically computed from the config toml file.

If we need a custom launcher to inject the custom dataset and reward functions, a positional argument can be added to specify the custom launcher file. Some example launcher files can be found in folder `tools/dataset`. As an example, the command to run a slurm grpo job with a custom launcher for handling `gsm8k` dataset is as follows:

```bash
python ./tools/slurm/dispatch_job.py \
    --ngpu-per-node 8 \
    --n-policy-replicas 1 \
    --n-rollout-replicas 2  \
    --config-path configs/qwen2-5/qwen2-5-32b-p-fsdp2-tp4-r-tp4-pp1-grpo-gsm8k.toml \ \
    --repo-root-path ${COSMOS_PATH} \
    --output-root-path ${COSMOS_OUTPUT_PATH} \
    --cosmos-container ${COSMOS_SQSQ_PATH} \
    --slurm-partition ${SLURM_PARTITION} \
    --slurm-account ${SLURM_ACCOUNT} \
    tools/dataset/gsm8k_grpo.py
```


The full arguments for `dispatch_job.py` are as follows:

| Argument               | Type                     | Required | Default Value | Description                                                                 |
|------------------------|--------------------------|----------|---------------|-----------------------------------------------------------------------------|
| --job-name             | str                      | No       | cosmos_job    | Name of the SLURM job.                                                      |
| --ngpu-per-node        | int                      | No       | 8             | Number of GPUs per compute node.                                            |
| --n-policy-replicas    | int                      | No       | 1             | Number of policy replicas to launch.                                        |
| --n-rollout-replicas   | int                      | No       | 1             | Number of rollout replicas to launch.                                       |
| --slurm-partition      | str                      | No       | batch         | SLURM partition to use.                                                     |
| --slurm-account        | str                      | No       | sw_aidot      | SLURM account to use.                                                       |
| --config-path          | str                      | Yes      |   -            | Path to the controller config file.                                         |
| --repo-root-path       | str                      | Yes      |   -            | Path to the repository root.                                                |
| --output-root-path     | str                      | Yes      |   -            | Path to the output root.                                                    |
| --cosmos-container     | str                      | Yes      |   -            | Path to the cosmos container.                                               |
| --extra-sbatch-args    | str                      | No       | ["--gres=gpu:8"] | Extra #SBATCH arguments.                                              |
| launcher               | str (positional)         | No       | cosmos_rl.dispatcher.run_web_panel | Launcher to use for dataset related operations, a custom launcher can be provided for custom dataset and reward functions injection. (Check in tools/dataset)                                              |