Overview
=================

Cosmos-RL provides native support for SFT and RL of world foundational models.

SFT
----
(Coming soon)

RL
----

Cosmos-RL supports `FlowGRPO <https://arxiv.org/pdf/2505.05470>`_ and `DDRL <https://arxiv.org/pdf/2512.04332>`_ algorithms for world foundational model reinforcement learning.

**Quick start**: A quick start guide for world foundational model's RL:

1. Configure the training recipe by editing toml files under ``configs/wfm/``.

2. Launch the reward service, you can refer docs here: `Reward Service <https://github.com/nvidia-cosmos/cosmos-rl/tree/main/reward_service>`_.

3. Launch the training script with the configured recipe::

      cosmos-rl --config ./configs/wfm/cosmos-predict2-5_2b_480_grpo_mock_data.toml --wfm-mode ./cosmos_rl/tools/dataset/wfm_rl.py

4. Monitor training progress via Wandb.

5. Evaluate the trained world foundational model using the evaluation script. 
   For Cosmos-Predict2.5, you can refer this repo: `cosmos-predict2.5 <https://github.com/nvidia-cosmos/cosmos-predict2.5>`_.

.. note::
    1. You can find detailed tutorials for DDRL here: `DDRL Tutorials <https://github.com/nvidia-cosmos/cosmos-rl/blob/main/examples/ddrl.md>`_.
    2. For a quick rollout of the training pipeline, we recommend you use the mock_data config file, i.e., ./configs/wfm/cosmos-predict2-5_2b_480_grpo_mock_data.toml

**Reward services**: Considering the computation overhead, it's necessary to use a seperated async service for reward computing.

- You can launch a reward service by following the instructions here: `Reward Service <https://github.com/nvidia-cosmos/cosmos-rl/tree/main/reward_service>`_.

- Configure the environment variable ``WFM_REWARD_TOKEN``, ``WFM_REWARD_ENQUEUE_URL``, and ``WFM_REWARD_FETCH_URL`` to make the trainer communicate with the reward service.

**Models**:

- Cosmos-Predict2.5-2B/14B

- Wan2.1 (coming soon)

**Parallelism**: Support HSDP/FSDP, and context parallel (CP) for world foundational model training. You can edit the related configurations in the toml file to enable these parallelism techniques.::

    [model]
    fsdp_shard_size = 8
    dp_replicate_size = 4

    [model_parallel]
    context_parallel_size = 2

**Datasets**:

- Local dataset: you can use local dataset for training. We follows the local dataset structure as `Cosmos-Predict2.5 <https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/post-training_video2world_cosmos_nemo_assets.md>`_. The dataset folder format should be::

    datasets/<your_local_dataset>/
    ├── metas/
    │   └── *.txt
    ├── videos/
    │   └── *.mp4
    └── text_embedding <optional> /
        └── *.pickle

- Webdataset: you need to configure the s3 access via environment variables, then you can use webdataset for training.

    - PROD_S3_CHECKPOINT_ACCESS_KEY_ID: Your S3 access key ID.

    - PROD_S3_CHECKPOINT_SECRET_ACCESS_KEY: Your S3 secret access key.

    - PROD_S3_CHECKPOINT_ENDPOINT_URL: Your S3 endpoint url.

    - PROD_S3_CHECKPOINT_REGION_NAME: Your S3 region name.

**Storage**:

- Local storage: you can use local disk for storing checkpoints and logs.

- S3 storage: you need to configure the s3 access via environment variables, then you can use s3 storage for storing checkpoints and logs.