# Data-regularized Reinforcement Learning (DDRL) for Cosmos Diffusion Models

<div align="center">
<a href="https://www.arxiv.org/abs/2512.04332">
    <img src="https://img.shields.io/badge/Paper-Arxiv-red?style=for-the-badge&logo=arxiv" alt="Paper">
  </a>
  &nbsp; <a href="https://github.com/nvidia-cosmos/cosmos-rl/blob/main/examples/ddrl.md">
    <img src="https://img.shields.io/badge/Code-Github-black?style=for-the-badge&logo=github" alt="Code">
  </a>
  &nbsp;
  <a href="https://research.nvidia.com/labs/dir/ddrl/">
    <img src="https://img.shields.io/badge/Website-Demo-blue?style=for-the-badge&logo=google-chrome" alt="Website">
  </a>
</div>


This document specifies how to post-train a diffusion model with DDRL, using the cosmos-predict model series as an example. 

The overall idea is to combine reward maximization (such as the GRPO objective) with standard diffusion training objective to replace the unreliable KL divergence regularization. For more details about the algorithm, please check the [paper](https://www.arxiv.org/pdf/2512.04332).

## News

- [Dec. 24, 2025] We release our initial version of DDRL codebase under NVIDIA [Cosmos-RL](https://github.com/nvidia-cosmos/cosmos-rl). Give it a try!



## Model Zoo

| Model                               |                           Download                           |
| ----------------------------------- | :----------------------------------------------------------: |
| Cosmos-Predict2.5-2B [merged] + RL  | [huggingface](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B/blob/main/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt) |
| Cosmos-Predict2.5-14B [merged] + RL | [huggingface](https://huggingface.co/nvidia/Cosmos-Predict2.5-14B/blob/main/base/post-trained/e21d2a49-4747-44c8-ba44-9f6f9243715f_ema_bf16.pt) |

For inference, you can see the Cosmos-Predict2.5 [document](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/inference.md).

## Tutorial

For fully document about diffusion RL, you can find it at the [offical document](https://nvidia-cosmos.github.io/cosmos-rl/wfm/overview.html) of Cosmos-RL.

### Configuration

**Experiment**: DDRL configurations can be found in `configs/wfm`. 

- The 2B experiment from the pre-trained checkpoint is `cosmos-predict2-5_2b_720_reason_embedding_ddrl.toml`.
- The 14B experiment from the pre-trained checkpoint is `cosmos-predict2-5_14b_720_reason_embedding_ddrl.toml`.

> **Important Note**: Since the merged SFT checkpoints are not released in the huggingface, we use the pre-trained checkpoint instead.

The DDRL parameters and explanations are listed as below. Detailed usage can be found in `cosmos_rl/policy/config/wfm/__init__.py`.

```python
class RLConfig:
    enabled: bool = False   

    ## Rollout parameters ##
    
    # Number of rollout group size. The total batch size is
    # World_size / num_rollout / model_parallel.context_parallel_size 
    num_rollout: int = 8    

    on_policy: bool = True  # Whether to use training policy to rollout
    sample_steps: int = 20  # How many sample steps in rollout
    # Control sample data type (0-t2v, 1-i2v, 2-v2v)
    min_num_conditional_frames: int = 0 
    max_num_conditional_frames: int = 1
    
    # Whether to use same initial seed within a rollout group
    use_same_seed: bool = False

    ## Standard sampler parameters ##
    solver_option: str = "2ab"  # Solver method (multistep or runge-kutta)
    s_churn: float = 1.0
    s_t_max: float = float("inf")
    s_t_min: float = 0.0
    s_noise: float = 1.0
    guidance: float = 0.0       # We do not enable CFG during DDRL
    
    # Training parameters
    train_on: list[int] = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    update_ref_every_iter: int = 0  # Frequency to update ref model
    clip_ratio: float = 0.0001
    kl_beta: float = 0.0           # Coefficient of reverse KL
    data_beta: float = 0.01          # Coefficient of diffusion loss
    
    # Diffusion loss paramteres. Required When data_beta > 0
    use_rl_sigma_and_noise: bool = True # whether to use rollout noise
    data_on_first_only: bool = False    # Whether to comput diffusion loss once
    
    # Reward configuration
    reward_config: RewardConfig = RewardConfig()
    exp_reward: bool = False            # Whether to rescale reward
```

### Reward service

Considering the computation overhead, it's necessary to use a seperated async service for reward computing.

- You can launch a reward service by following this [document](https://github.com/nvidia-cosmos/cosmos-rl/tree/main/reward_service/README.md).
- Configure the trainer to make it communicate with the reward service. Set environment variable ``WFM_REWARD_TOKEN``, ``WFM_REWARD_ENQUEUE_URL``, and ``WFM_REWARD_FETCH_URL`` 

### Dataset

We provide a data preparation example based on [Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/post-training_video2world_cosmos_nemo_assets.md), you can also construct your own dataset with the same format. 

#### Downloading Dataset

The first step is downloading a dataset with videos.

You must provide a folder containing a collection of videos in **MP4 format**, preferably 720p. These videos should focus on the subject throughout the entire video so that each video chunk contains the subject.

Dataset folder format:

```
datasets/example/
├── videos/
│   ├── *.mp4
```

#### Preprocessing the Data

To generate text prompt files for each video in the dataset, use the provided preprocessing script:

> **Note:** If your dataset have prompt files, you can skip this step. You only need to align the prompt files with video files by name.

Run the following command to generate metas for the video caption used for post-training:

```bash
# Create prompt files for all videos with a custom prompt
cd cosmos-predict2.5
python -m scripts.create_prompts_for_nemo_assets \
    --dataset_path datasets/example \
    --prompt "A video of sks teal robot."
```

Dataset folder format:

```
datasets/example/
├── metas/
│   └── *.txt
└── videos/
    └── *.mp4
```

### Launch Training Job

```bash
cosmos-rl --config ./configs/wfm/cosmos-predict2-5_2b_720_reason_embedding_ddrl.toml --wfm-mode ./cosmos_rl/tools/dataset/wfm_rl.py 
```

## Citation

```
@article{ye2025data,
  title={Data-regularized Reinforcement Learning for Diffusion Models at Scale},
  author={Ye, Haotian and Zheng, Kaiwen and Xu, Jiashu and Li, Puheng and Chen, Huayu and Han, Jiaqi and Liu, Sheng and Zhang, Qinsheng and Mao, Hanzi and Hao, Zekun and others},
  journal={arXiv preprint arXiv:2512.04332},
  year={2025}
}
```