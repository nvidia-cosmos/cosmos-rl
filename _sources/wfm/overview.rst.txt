Diffusion Models
=================

Cosmos-RL provides native support for SFT and RL of diffusion models.

SFT
----
(Coming soon)

RL
----

Cosmos-RL supports `FlowGRPO <https://arxiv.org/pdf/2505.05470>`_ and `DDRL <https://arxiv.org/pdf/2512.04332>`_ algorithms for diffusion model reinforcement learning.

**Quick start**: A quick start guide for diffusion model RL:
1. Configure the training recipe by editing toml files under ``configs/diffusion/``.
2. Launch the reward service, you can refer docs here: `Reward Service <../../reward_service/README.md>`_.
3. Launch the training script with the configured recipe.
    ::
    
      cosmos-rl --config ./configs/diffusion/cosmos_predict2-5_2b_480_grpo_mock_data.toml --diffusion-mode ./cosmos_rl/tools/dataset/diffusion_grpo.py 
4. Monitor training progress via Wandb.
5. Evaluate the trained diffusion model using the evaluation script. For Cosmos-Predict2.5, you can refer this repo: `cosmos-predict2.5 <https://github.com/nvidia-cosmos/cosmos-predict2.5>`_.

.. note::
    1. You can find detailed tutorials for DDRL here: `DDRL Tutorials <../../examples/ddrl.md>`_.
    2. For a quick rollout of the training pipeline, we recommend you use the mock_data config file, i.e., ./configs/diffusion/cosmos_predict2-5_2b_480_grpo_mock_data.toml

**Reward services**: Considering the computation overhead, it's necessary to use a seperated async service for reward computing.
- You can launch a reward service by following the instructions here: `Reward Service <../../reward_service/README.md>`_.
- Configure the environment variable ``DIFFUSION_REWARD_TOKEN``, ``DIFFUSION_REWARD_ENQUEUE_URL``, and ``DIFFUSION_REWARD_FETCH_URL`` to make the trainer communicate with the reward service.

**Models**:
- Cosmos-Predict2.5-2B/14B
- Wan2.1 (coming soon)

**Datasets**:
- Local dataset: you can use local dataset for training. We follows the local dataset structure as `Cosmos-Predict2.5 <https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/post-training_video2world_cosmos_nemo_assets.md>`_. The dataset folder format should be:
    ```
        datasets/<your_local_dataset>/
        ├── metas/
        │   └── *.txt
        ├── videos/
        │   └── *.mp4
        └── text_embedding <optional> /
            └── *.pickle
    ```
- Webdataset: you need to configure the s3 access via environment variables, then you can use webdataset for training.
    - PROD_S3_CHECKPOINT_ACCESS_KEY_ID: Your S3 access key ID.
    - PROD_S3_CHECKPOINT_SECRET_ACCESS_KEY: Your S3 secret access key.
    - PROD_S3_CHECKPOINT_ENDPOINT_URL: Your S3 endpoint url.
    - PROD_S3_CHECKPOINT_REGION_NAME: Your S3 region name.

**Storage**:
- Local storage: you can use local disk for storing checkpoints and logs.
- S3 storage: you need to configure the s3 access via environment variables, then you can use s3 storage for storing checkpoints and logs.