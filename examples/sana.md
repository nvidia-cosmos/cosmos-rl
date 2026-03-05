# SANA: Efficient High-Resolution Image & Video Generation

<div align="center">
<a href="https://nvlabs.github.io/Sana/docs/">
    <img src="https://img.shields.io/badge/SANA-OfficialDoc-blue?style=for-the-badge&logo=google-chrome" alt="Official Documents">
  </a>
  &nbsp; <a href="https://github.com/NVlabs/Sana">
    <img src="https://img.shields.io/badge/SANA-Github-black?style=for-the-badge&logo=github" alt="Code">
  </a>
</div>

SANA is an efficiency-oriented codebase for high-resolution image and video generation, providing complete training and inference pipelines.

This document specifies how to post-train (SFT/RL) a SANA-Image or SANA-Video model on Cosmos-RL.

## Tutorial

For a full document about the post-training of diffusion models, you can find it in the [official document](https://nvidia-cosmos.github.io/cosmos-rl/wfm/overview.html) of Cosmos-RL.

### Configuration

**Experiment**: configurations of SANA can be found in [`configs/sana`](https://github.com/nvidia-cosmos/cosmos-rl/tree/main/configs/sana). We provided several preset config files:

- SFT
  - Image: [`sana-image-sft`](https://github.com/nvidia-cosmos/cosmos-rl/blob/main/configs/sana/sana-image-sft.toml), [`sana-image-sft-lora`](https://github.com/nvidia-cosmos/cosmos-rl/blob/main/configs/sana/sana-image-sft-lora.toml)
  - Video: [`sana-vidoe-sft`](https://github.com/nvidia-cosmos/cosmos-rl/blob/main/configs/sana/sana-video-sft.toml), [`sana-video-sft-lora`](https://github.com/nvidia-cosmos/cosmos-rl/blob/main/configs/sana/sana-video-sft-lora.toml)
- RL
  - Image: [`sana-image-nft`](https://github.com/nvidia-cosmos/cosmos-rl/blob/main/configs/sana/sana-image-nft.toml)
  - Video:  [`sana-video-nft`](https://github.com/nvidia-cosmos/cosmos-rl/blob/main/configs/sana/sana-video-nft.toml)

For a detailed explanation of the arguments, you can see the [Configuration Page](https://nvidia-cosmos.github.io/cosmos-rl/quickstart/configuration.html) of Cosmos-RL

### Reward service

Considering the computation overhead, it's necessary to use a separate async service for reward computing.

- You can launch a reward service by following this [document](https://github.com/nvidia-cosmos/cosmos-rl/tree/main/reward_service/README.md).
- Configure the trainer to make it communicate with the reward service. Set environment variable ``REMOTE_REWARD_TOKEN``, ``REMOTE_REWARD_ENQUEUE_URL``, and ``REMOTE_REWARD_FETCH_URL``

### Dataset

#### SFT

We support loading the dataset from a local directory. You should prepare your paired prompts and multimodal datas with the following format:

```
local_image_dataset_dir/
├── *.json
├── *.jpg
│...
local_video_dataset_dir/
├── *.json
├── *.mp4
│ ...
```

#### RL

We support some popular datasets for RL training.

- Image: pickscore, ocr, geneval
- Video: filtered VidProM from DanceGRPO

> **Note:** The Cosmos-RL is very flexible for the user-customized dataset and input/output format. You can edit the [`./cosmos_rl/tools/dataset/diffusion_nft.py`](https://github.com/nvidia-cosmos/cosmos-rl/blob/main/cosmos_rl/tools/dataset/diffusion_nft.py) launcher to customize your own dataset for training. For more details about the customization of the dataset, please refer to [Customization](https://nvidia-cosmos.github.io/cosmos-rl/quickstart/customization.html).

### Training

#### SFT

```bash
cosmos-rl --config ./configs/stable-diffusion-3-5/stable-diffusion-3-5-image-sft-lora.toml cosmos_rl.tools.dataset.diffusers_dataset
```

#### RL

```bash
cosmos-rl --config ./configs/sana/sana-image-nft.toml cosmos_rl.tools.dataset.diffusion_nft
```

## Citation

```
@misc{xie2024sana,
      title={Sana: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformer},
      author={Enze Xie and Junsong Chen and Junyu Chen and Han Cai and Haotian Tang and Yujun Lin and Zhekai Zhang and Muyang Li and Ligeng Zhu and Yao Lu and Song Han},
      year={2024},
      eprint={2410.10629},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.10629},
    }
@misc{xie2025sana,
      title={SANA 1.5: Efficient Scaling of Training-Time and Inference-Time Compute in Linear Diffusion Transformer},
      author={Xie, Enze and Chen, Junsong and Zhao, Yuyang and Yu, Jincheng and Zhu, Ligeng and Lin, Yujun and Zhang, Zhekai and Li, Muyang and Chen, Junyu and Cai, Han and others},
      year={2025},
      eprint={2501.18427},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.18427},
    }
@misc{chen2025sanasprint,
      title={SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation},
      author={Junsong Chen and Shuchen Xue and Yuyang Zhao and Jincheng Yu and Sayak Paul and Junyu Chen and Han Cai and Song Han and Enze Xie},
      year={2025},
      eprint={2503.09641},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.09641},
    }
@misc{chen2025sana,
      title={SANA-Video: Efficient Video Generation with Block Linear Diffusion Transformer},
      author={Chen, Junsong and Zhao, Yuyang and Yu, Jincheng and Chu, Ruihang and Chen, Junyu and Yang, Shuai and Wang, Xianbang and Pan, Yicheng and Zhou, Daquan and Ling, Huan and others},
      year={2025},
      eprint={2509.24695},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.24695},
    }
```
