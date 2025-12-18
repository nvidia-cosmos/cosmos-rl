# Cosmos RL Reward

A service for calculating rewards of generated videos for world foundational model reinforcement learning.


## Overview

`cosmos_rl_reward` is a Python package that provides a service architecture for computing rewards in world foundational model RL applications. It handles reward calculation requests from client, enabling efficient and scalable reward computation.

### Supported Reward Types

**Video Rewards:**
- `cosmos_reason1`: Uses [`nvidia/Cosmos-Reason1-7B-Reward`](https://huggingface.co/nvidia/Cosmos-Reason1-7B-Reward) as the reward model
- `dance_grpo`: Adopts the [`VideoAlign`](https://huggingface.co/KwaiVGI/VideoReward) based `VQ`, `MQ` and `TA` scores from [`DanceGRPO`](https://github.com/XueZeyue/DanceGRPO)

**Image Rewards:**
- `hpsv2`: [HPSv2](https://github.com/tgxs002/HPSv2) - Human Preference Score v2 for text-to-image alignment
- `image_reward`: [ImageReward](https://github.com/THUDM/ImageReward) - Image quality and prompt alignment scoring
- `ocr`: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) based reward for text rendering accuracy
- `gen_eval`: [GenEval](https://github.com/djghosh13/geneval) - Object detection based compositional generation evaluation

## Installation

Install the package from source:

```bash
python -m pip install .
```

If in-place edition is needed, install with:

```bash
python -m pip install -e .
```

## Quick Start

### 1. Prepare the Environment

Set up the required environment and dependencies:

```bash
cosmos-rl-reward-prepare --config cosmos_rl_reward/configs/rewards.toml
```

Since different reward calculations may have different dependencies, separate environment for each type of rewards might be needed if necessary. Virtual python environment is used to offer separate environments for different types of rewards. 

This command will:
- Initialize necessary directories
- Download required models
- Prepare the runtime environment
- Install the needed dependencies

The directories where to build the virtual environment and where to download the models are specified in the `toml` configuration file. For each reward type, its `venv_python` decides the virtual environment directory, while `download_path` decides the folder to download related models to.

### 2. Start the Reward Service

Launch the reward calculation service:

```bash
cosmos-rl-reward --config cosmos_rl_reward/configs/rewards.toml
```

The service will start listening for reward calculation requests.

The configuration `toml` file specifies the basic setting of the service including host, port, redis configurations, etc. Various types of rewards are specified in `reward_args` field. Each `reward_args` includes its detailed specifications and stands for one reward type. For example, inside each `reward_args`, `venv_python` defines the virtual environment used to run this reward calculation, `download_path` defines the folder storing the downloaded related models for this reward, `model_path` defines the path of the main model this reward needs.

By modifying the settings in configuration `toml`, the properties of the service can be adjust.


### 3. Test with Example Client

Run the example client to interact with the service:

```bash
python cosmos_rl_reward/example/client.py
```

This example demonstrates how to:
- Connect to the reward service
- Submit reward calculation requests
- Receive and process results
- The host url in the example can be adjust accordingly.

## Configuration

The service is configured through `cosmos_rl_reward/configs/rewards.toml`. Key configuration options include:

### Server Configuration
The cosmos-rl-reward service is launched at the following `host:port` url. Clients will request to this url:

| Field | Type | Description |
|-------|------|-------------|
| `host` | string | The hostname or IP address where the service will listen |
| `port` | integer | The port number for the service endpoint |

### Redis Configuration
A Redis server is launched for recording the calculated scores inside the cosmos-rl-reward service. The following specifies the properties of the Redis service:

| Field | Type | Description |
|-------|------|-------------|
| `redis_host` | string | The hostname or IP address of the Redis server |
| `redis_port` | integer | The port number of the Redis server |

### Reward Arguments (`[[reward_args]]`)

Each `[[reward_args]]` block defines a reward calculation type. Multiple blocks can be specified.

| Field | Type | Description |
|-------|------|-------------|
| `reward_type` | string | Unique string identifier for the reward type such as `cosmos_reason1` and `dance_grpo` |
| `venv_python` | string | Path to the Python interpreter used for this reward type to ensure the environment separation |
| `model_path` | string | Path or identifier of the reward model, specific to each reward type |
| `download_path` | string | (Optional) Directory where model files should be downloaded if explicit downloading is needed |
| `dtype` | string | Data type for model inference (e.g., "float16", "float32") |
| `enable` | boolean | Whether this reward calculator is enabled (true/false) |


## Service Interface

### POST /api/reward/enqueue

Submit a reward calculation request to the service queue.

**Request Body**:

The request body consists of JSON metadata followed by video bytes, they are combined as a whole chunk of bytes: `<JSON>\n<VIDEO_BYTES>`
The sent video bytes are bytes of the latents encoded from the original videos to reduce the size. Inside the service the received video bytes are decoded back to normal video format first.

**JSON Metadata**:
```json
{
  "prompts": ["prompt text"],
  "reward_fn": { 
    "cosmos_reason1": 1.0, 
    "dance_grpo": 1.0
  },
  "video_infos": [
    {
      "fps": 30
    }
  ]
}
```
| Field | Type | Description |
|-------|------|-------------|
| `prompts` | List[str] | List of text prompts corresponding to the videos |
| `reward_fn` | Dict | Reward types with their weights, can specify one or multiple reward types required for the videos |
| `video_infos` | List[Dict] | Video metadata including fps |

**Response**:
```json
{
  "uuid": "1e29d4bd-51a9-4ee5-84ba-7e26b81ac79c"
}
```
**Response Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `uuid` | string | Unique identifier for tracking this request |

**Status Codes**:
- `200 OK`: Request successfully queued


### POST /api/reward/pull

Retrieve completed reward calculation results.

**Request Body**:
```json
{
  "uuid": "1e29d4bd-51a9-4ee5-84ba-7e26b81ac79c",
  "type": "cosmos_reason1"
}
```
| Field | Type | Description |
|-------|------|-------------|
| `uuid` | str | Request UUID to retrieve returned by enqueue|
| `type` | str | Reward type: the key string in `reward_fn` when enqueue |

**Response (Cosmos Reason1)**:
```json
{
  "scores": {
    "prediction": ["Good"],
    "no_score": [0.9997965693473816],
    "yes_logit": [17.5],
    "no_logit": [26.0]
  },
  "input_info": {
    "shape": [1, 16, 24, 54, 96],
    "dtype": "torch.bfloat16",
    "min": "0.000",
    "max": "1.000",
    "video_infos": [
      {
        "video_fps": 16.0
      }
    ]
  },
  "duration": "2.23",
  "decoded_duration": "2.02",
  "type": "cosmos_reason1"
}
```

**Response (Dance GRPO)**:
```json
{
  "scores": {
    "vq_reward": [-0.5091875791549683],
    "mq_reward": [-1.1062785387039185],
    "ta_reward": [-2.6613192558288574],
    "overall_reward": [-4.276785373687744]
  },
  "input_info": {
    "shape": [1, 16, 24, 54, 96],
    "dtype": "torch.bfloat16",
    "min": "0.000",
    "max": "1.000",
    "video_infos": [
      {
        "video_fps": 16.0
      }
    ]
  },
  "duration": "0.77",
  "decoded_duration": "2.02",
  "type": "dance_grpo"
}
```

**Response Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `scores` | Dict | Calculated scores details, specific to reward type |
| `input_info.shape` | List[int] | Shape of the received video latent when enqueue |
| `input_info.dtype` | str | Data type of the received latent |
| `input_info.min(max)` | str | Minimum(Maximum) value in the received latent |
| `input_info.video_infos` | List[Dict] | Video metadata received when enqueue like `fps` |
| `duration` | str | Reward calculation duration in seconds |
| `decoded_duration` | str | Video decoding from latent duration in seconds |
| `type` | str | Reward type identifier for this response |


**Status Codes**:
- `200 OK`: Results retrieved successfully
- `400 Not Ready`: Scores calculation not completed and please retry later

For more detailed usage of the above interface apis, please refer to `cosmos_rl_reward/example/client.py`.

## Image Rewards

For image-based rewards, the request format differs from video rewards. Images are sent as numpy arrays and the service will pass them directly to the reward handler.

**Note:** The current four image rewards (`hpsv2`, `image_reward`, `ocr`, `gen_eval`) expect NHWC format `[B, H, W, C]` with `uint8` dtype. If you implement a custom image reward, you can define your own expected image format in your reward handler's `calculate_reward` method.

### Request Format

**Payload Structure**: `<JSON_METADATA>\n<NPY_BYTES>`

The metadata must include `"media_type": "image"` to indicate this is an image request.

```python
import numpy as np
import json
import io
import requests

# Prepare image as numpy array [B, H, W, C] uint8
images = [np.array(img.convert("RGB"), dtype=np.uint8) for img in pil_images]
arr = np.stack(images, axis=0)  # Shape: [B, H, W, 3]

# Save to bytes
buf = io.BytesIO()
np.save(buf, arr, allow_pickle=False)
npy_bytes = buf.getvalue()

# Build metadata (must include media_type: image)
metadata = {
    "media_type": "image",
    "reward_fn": {"hpsv2": 1.0},
    "prompts": ["prompt for image 1", "prompt for image 2"],
}
# Combine metadata + npy bytes
payload = json.dumps(metadata).encode("utf-8") + b"\n" + npy_bytes
response = requests.post(
    "http://localhost:8080/api/reward/enqueue",
    data=payload,
    headers={"Content-Type": "application/octet-stream"},
)
uuid = response.json()["uuid"]
```

### Metadata Examples for Each Image Reward

#### HPSv2 (Human Preference Score v2)
```json
{
  "media_type": "image",
  "reward_fn": {"hpsv2": 1.0},
  "prompts": ["a photo of a cat", "a beautiful sunset"]
}
```

#### Image Reward
```json
{
  "media_type": "image",
  "reward_fn": {"image_reward": 1.0},
  "prompts": ["a photo of a cat", "a beautiful sunset"]
}
```

#### OCR Reward
```json
{
  "media_type": "image",
  "encoding": "npy_uint8",
  "reward_fn": {"ocr": 1.0},
  "prompts": ["New York Skyline with 'Hello World' written with fireworks"],
  "ocr_use_gpu": false
}
```

#### GenEval (Object Detection Evaluation)
```json
{
  "media_type": "image",
  "reward_fn": {"gen_eval": 1.0},
  "tag": "single_object",
  "include": [
    {"class": "giraffe", "count": 1, "color": "brown"},
    {"class": "stop sign", "count": 1, "color": "black"}
  ],
  "prompt": "a photo of a brown giraffe and a white stop sign"
}
```

## Building Distribution

```bash
python -m build
ls dist/
```

Built package wheel and tar files can be found in `dist`. 

## Add New Reward Support

To add a new reward support, a reward handler class based on `BaseRewardHandler` should be added like the classes in `cosmos_rl_reward/model/`. The following three interfaces are needed for each reward handler class and a registration is needed with the `reward_name` attribute. Detailed interface implementations and explanations can be found in existing reward examples in `cosmos_rl_reward/model/` and the base class `BaseRewardHandler` in `cosmos_rl_reward/handler/reward_base.py`.

```python
@RewardRegistry.register()
class NewReward(BaseRewardHandler):
    reward_name = "new_reward_type"
  
    def __init__(
        self,
        model_path,
        dtype=torch.float16,
        device="cuda",
        download_path="",
        **kwargs,
    ):
        """
        Initialize the BaseRewardHandler with necessary attributes.
        Override this method in subclasses to add more attributes if needed.
        Usually the added attributes in subclasses include:
            model_path: the path to the main model used for rewarding.
            dtype: the data type for model.
            device: the device to run the model on, e.g., "cuda" or "cpu".
            download_path: the path to download or load related files including models.
        """
        pass

    def set_up(self):
        """
        Set up the inference engine and models for the reward handler.
        This method should be overridden in subclasses to set up specific inference engine.
        """
        pass

    def calculate_reward(self, images, metadata):
        """
        Calculate the reward given bunches of images as videos with metadata for additional information.

        Args:
            images (Tensor): bunches of images in tensor format (batch_size, C, frame, H, W) to present videos.
            metadata (Dict): additional information for the reward calculation such as prompts, fps, etc.

        Returns:
            A dictionary including the calculated reward scores:
                "scores": the calculated reward scores in dict of list format, reward specific.
                "input_info": the information about the input video latents including value range, shape, etc.
                "duration": the duration in seconds taken for the reward calculation.
                "decoded_duration": the duration in seconds taken for decoding the input latents to videos.
                "type": the type of the reward calculated.
        """
        calculation_results = {
          "scores": {
            "vq_reward": [-0.5091875791549683],
            "mq_reward": [-1.1062785387039185],
            "ta_reward": [-2.6613192558288574],
            "overall_reward": [-4.276785373687744]
          },
          "input_info": {
            "shape": [1, 16, 24, 54, 96],
            "dtype": "torch.bfloat16",
            "min": "0.000",
            "max": "1.000",
            "video_infos": [
              {
                "video_fps": 16.0
              }
            ]
          },
          "duration": "0.77",
          "decoded_duration": "2.02",
          "type": "dance_grpo"
        }
        return calculation_results
```

Besides the reward handler class, a script for each reward type to download its required files and prepare its virtual environment with needed dependencies should be added as `new_reward_type.sh` in `cosmos_rl_reward/setup/`. The first and second arguments to this script is the folder storing the downloaded files and the virtual python environment path, respectively. The required operations should be executed in this script to make sure all the files including models are downloaded if needed, and the dependencies are installed in the virtual environment. Detailed example for such scrips can refer to `cosmos_rl_reward/setup/*.sh`.