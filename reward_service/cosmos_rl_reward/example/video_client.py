# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import requests
import numpy as np
import io
import json
import torch
import time


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

"""
Cosmos-Reason1, DanceGRPO VQ and MQ reward functions are supported currently.
"""

# Example prompts.
prompts = [
    "The camera follows a young explorer through an abandoned urban building at night, exploring hidden corridors and forgotten spaces, with a mix of light and shadow creating a mysterious atmosphere.",
    "The camera remains still, a girl with braided hair and wearing a pink dress approached the chair in the room and sat on it, the background is a cozy bedroom, warm indoor lighting.",
]

# Url host to access.
host = (
    "http://localhost:8080"  # For local testing, change to your local server address.
)

# Different API endpoints.
api_ping = "/api/reward/ping"  # For check the server availability.
api_enqueue = "/api/reward/enqueue"  # For enqueueing reward calculation tasks.
api_reward = "/api/reward/pull"  # For pulling reward calculation results.
token = None  # If authentication is needed, set the token here.


def make_headers(replica_id: str | None = None, extra: dict | None = None) -> dict:
    headers: dict = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if replica_id:
        headers["X-Lepton-Replica-Target"] = replica_id
    if extra:
        headers.update(extra)
    return headers

# The folllowing code is an example of how to generate encoded latents from video using the Wan2pt1TokenizerHelper.
# The generated latents can be sent to the "/api/reward/enqueue" endpoint for calculating rewards.
# "api/reward/enqueue" endpoint expects only encoded latent tensor in the request body.
use_fake = True  # Set to True to use a fake video tensor for demonstration purposes.
if use_fake:
    # Example fake video info data.
    # This is a dictionary containing video information such as FPS.
    # The video info is used to provide additional context about the video.
    # Example: {'video_fps': 16.0}
    video_infos = [
        {
            "video_fps": 16.0,  # Example FPS for the fake video
        },
        {
            "video_fps": 16.0,  # Example FPS for the fake video
        },
    ]

    # Fake a latent tensor for demonstration purposes.
    # Example fake latent tensor shape [2, 16, 24, 54, 96]
    latents = torch.from_numpy(np.random.rand(2, 16, 24, 54, 96).astype(np.float16)).to(
        torch.bfloat16
    )
else:
    from cosmos_rl.policy.model.wfm.tokenizer.wan2pt1 import Wan2pt1TokenizerHelper
    import torchvision
    import numpy as np

    # Initialize the tokenizer for latent encoding.
    # The Wan2pt1TokenizerHelper is used to encode video tensors into latents.
    # The tokenizer requires a VAE checkpoint path and other parameters.
    demo = Wan2pt1TokenizerHelper(
        chunk_duration=81,
        load_mean_std=False,
        vae_pth="/workspace/tokenizer_ckpt.pth",
        temporal_window=16,
        device="cuda",
    )

    # Read a real video tensor using torchvision.io.read_video.
    # This is an example of how to read a video file and convert it to a tensor.
    # The video tensor will be in the shape [B, T, C, H, W] after reading.
    # The range of the sample video tensor is [-1, 1] after normalization.
    with torch.no_grad():
        video, audio, info = torchvision.io.read_video(
            "sample.mp4",  # Path to video file
            start_pts=0.0,
            end_pts=None,  # Read the entire video
            pts_unit="sec",  # Time unit is seconds
            output_format="TCHW",  # Output format: Time, Channels, Height, Width
        )
    video = video.detach().cpu()
    video = video.permute(1, 0, 2, 3).contiguous()  # Change to CTHW format
    video = video.unsqueeze(0)  # Add batch dimension
    # The video tensor is now in the shape [B, C, T, H, W]

    # Record the sample video tensor without normalization for later use.
    # The sample video tensor is in the range of [0, 255] and in uint8 format.
    sample_video = video.clone().clamp(0, 255).to(torch.uint8)

    # Normalize the video tensor to [-1, 1] range.
    # This is required for the Wan2pt1TokenizerHelper to work correctly.
    # The video tensor should be in the range of [0, 255] before normalization.
    video = video.float()  # Ensure the video tensor is in float format
    video = (video - 127.5) / 127.5  # Normalize to [-1, 1]
    log(f"Sample video shape: {video.shape}")
    log(f"Sample video range: [{video.min():.3f}, {video.max():.3f}]")

    # Create video info dictionary to record the video info data from the read_video function.
    # The info usually contains fps about the video, Example: {'video_fps': 30.0}
    video_infos = [info, info]

    # Encode the normalized [-1, 1] video tensor to latents.
    latents = demo.encode_video(video)
    log(
        f"Latent range: [{latents.min():.3f}, {latents.max():.3f}] shape: {latents.shape} dtype: {latents.dtype}"
    )

    # Duplicate the latents for batch size of 2 to demonstrate batch processing.
    latents = torch.cat([latents, latents], dim=0)  # Duplicate for batch size of 2


# Convert the latents to a numpy array for sending to the API.
# The latents should be in the shape [B, T, C, H, W] and in bfloat16 format.
# Here we convert the latents to uint8 format view for sending to the API since no bfloat16 support in numpy.
# This is an example of how to convert the latents to a numpy array and send it to the API.
tensor = latents[0:1].cpu()


# Example json format to be sent together with the encoded latents to the "/api/reward/enqueue" endpoint for calculating rewards.
data = {
    "prompts": prompts[0:1],  # List of prompts corresponding to the batch size.
    "reward_fn": {
        "cosmos_reason1": 1.0,  # Cosmos-Reason1 function and all Cosmos-Reason1 related reward will be calculated.
        "dance_grpo": 1.0,  # DanceGRPO function and all DanceGRPO related reward will be calculated including VQ and MQ, TA and Overall.
    },
    "video_infos": video_infos[0:1],  # Required for video info including fps.
}


# The following code is an example of how to check the server availability using the "/api/reward/ping" endpoint.
# This endpoint is used to check if the server is running and can be accessed.
log("Checking server availability...")
url = host + api_ping
response = requests.post(
    url,
    data={"info_data": json.dumps(data)},
    headers=make_headers(),
)
log(f"Status Code: {response.status_code}")
log(f"Response: {response.json()}")


# How to make the reward calculation request to the "/api/reward/enqueue" endpoint.
# This endpoint expects a binary file containing the encoded latents and a JSON string with metadata.
# The metadata includes prompts and reward function weights.
# Please follow the example below to send the request to calculate rewards.
# Currently, only DanceGRPO related reward and Cosmos-Reason1 reward are supported.
pending_requests = []
for i in range(1):
    url = host + api_enqueue
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    # Combine JSON (UTF-8) + newline + binary to form the payload.
    # The JSON string is encoded in UTF-8 and followed by a newline character. Then the binary data for the latent tensor is appended.
    payload = json.dumps(data).encode("utf-8") + b"\n" + buffer.getvalue()
    # Send the combined data as a POST request to the "/api/reward/enqueue" endpoint for reward calculation.
    response = requests.post(
        url,
        data=payload,
        headers=make_headers(extra={"Content-Type": "application/octet-stream"}),
    )
    log(f"Status Code: {response.status_code}")
    log(f"Response: {response.json()}")
    response_json = response.json()
    pending_requests.append(
        (
            response_json["uuid"],
            response_json.get("replica_id"),
            list(data["reward_fn"].keys()),
        )
    )

for uuid, replica_id, reward_types in pending_requests:
    url = host + api_reward
    # Send the recorded uuid with reward type as a POST request to the "/api/reward/pull" endpoint to get the reward calculation results.
    for reward_type in reward_types:
        log(f"Pulling reward results for UUID: {uuid}, reward type: {reward_type}...")
        while True:
            response = requests.post(
                url,
                data={
                    "uuid": uuid,
                    "type": reward_type,
                },
                headers=make_headers(replica_id=replica_id),
            )
            log(f"Status Code: {response.status_code}")
            log(f"Response: {response.json()}")
            if response.status_code == 200:
                # Successfully got the reward results, exit the loop.
                # The response contains the reward scores for the requested reward type.
                break
            time.sleep(1)


"""
    Final response example:  
    # For Cosmos-Reason1 type score: 
    {
        'scores': {
            'prediction': ['Good'], 
            'no_score': [0.9997965693473816], 
            'yes_logit': [17.5], 
            'no_logit': [26.0]
        }, 
        'input_info': {
            'shape': [1, 16, 24, 54, 96], 
            'dtype': 'torch.bfloat16', 
            'min': '0.000', 
            'max': '1.000', 
            'video_infos': [
                {'video_fps': 16.0}
            ]
        }, 
        'duration': '2.23', 
        'decoded_duration': '2.02', 
        'type': 'cosmos_reason1'
    }
    # For DanceGRPO type score:
    {
        'scores': {
            'vq_reward': [-0.5091875791549683], 
            'mq_reward': [-1.1062785387039185], 
            'ta_reward': [-2.6613192558288574], 
            'overall_reward': [-4.276785373687744]
        }, 
        'input_info': {
            'shape': [1, 16, 24, 54, 96], 
            'dtype': 'torch.bfloat16', 
            'min': '0.000', 
            'max': '1.000', 
            'video_infos': [
                {'video_fps': 16.0}
            ]
        }, 
        'duration': '0.77', 
        'decoded_duration': '2.02', 
        'type': 'dance_grpo'
    }
    Inside each field of the response, the values are lists corresponding to the batch size.    
"""
