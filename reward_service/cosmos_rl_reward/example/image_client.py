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
import time
import os


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

"""
HPSv2, ImageReward, OCR, and GenEval reward functions are supported currently.
"""

# Example prompts.
prompts = [
    "The camera follows a young explorer through an abandoned urban building at night, exploring hidden corridors and forgotten spaces, with a mix of light and shadow creating a mysterious atmosphere.",
    "The camera remains still, a girl with braided hair and wearing a pink dress approached the chair in the room and sat on it, the background is a cozy bedroom, warm indoor lighting.",
]
images = []

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
    # Create a sample fake video tensor in the shape [B, H, W, C].
    # The sample video tensor is generated with random values in the range of [0, 255].
    fake_image = (
        (np.random.rand(2, 512, 512, 3) * 255.0).clip(0, 255).astype(np.uint8)
    )  # Example tensor shape [2, 512, 512, 3]
    images.extend(fake_image)
    arr = np.array(images, dtype=np.uint8)  # Shape: [B, H, W, 3]
else:
    import os
    from PIL import Image
    img_folder_path = "/path/to/your/image/folder"  # Set your image folder path here.
    for img_name in os.listdir(img_folder_path):
        img_path = os.path.join(img_folder_path, img_name)
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        images.append(img)
    arr = np.stack(images, axis=0)  # Shape: [B, H, W, 3]

# Save to bytes
buf = io.BytesIO()
np.save(buf, arr, allow_pickle=False)
npy_bytes = buf.getvalue()

# Example json format to be sent together with the encoded latents to the "/api/reward/enqueue" endpoint for calculating rewards.
data = {
    "media_type": "image",
    "prompts": prompts[0:2],  # List of prompts corresponding to the batch size.
    "reward_fn": {
        "hpsv2": 1.0,  # HPSv2 function and all HPSv2 related reward will be calculated.
    },
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
    # Combine JSON (UTF-8) + newline + binary to form the payload.
    # The JSON string is encoded in UTF-8 and followed by a newline character. Then the binary data for the latent tensor is appended.
    payload = json.dumps(data).encode("utf-8") + b"\n" + npy_bytes
    # Send the combined data as a POST request to the "/api/reward/enqueue" endpoint for reward calculation.
    response = requests.post(
        url,
        data=payload,
        headers=make_headers(extra={"Content-Type": "application/octet-stream"}),
        timeout=30,
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
                timeout=10,
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
    # For HPSv2 type score: 
    {
        'scores': {
            'hpsv2': [0.08438428491353989, 0.047684457153081894]
        },
        'input_info': {
            'shape': [2, 512, 512, 3],
            'dtype': 'torch.uint8',
            'min': '0.000',
            'max': '254.000'
        },
        'duration': '0.35',
        'decoded_duration': '0.00',
        'type': 'hpsv2'
    }
    # For ImageReward type score:
    {
        'scores':{
            'image_reward': [-2.2803564071655273, -2.2761073112487793]
        },
        'input_info': {
            'shape': [2, 512, 512, 3],
            'dtype': 'torch.uint8',
            'min': '0.000',
            'max': '254.000'
        },
        'duration': '0.30',
        'decoded_duration': '0.00',
        'type': 'image_reward'
    }
    # For OCR type score:
    {
        'scores': {
            'ocr_reward': [0.09090909090909094, 0.09999999999999998]
        },
        'input_info': {
            'shape': [2, 512, 512, 3],
            'dtype': 'torch.uint8',
            'min': '0.000',
            'max': '254.000'
        },
        'duration': '0.12',
        'decoded_duration': '0.00',
        'type': 'ocr'
    }
    # For GenEval type score:
    {
        'scores': {
            'gen_eval_score': [0.0],
            'gen_eval_reward': [0.0],
            'gen_eval_strict': [0.0],
            'gen_eval_group': {
                'single_object': [0.0],
                'two_object': [-10.0],
                'counting': [-10.0],
                'colors': [-10.0],
                'position': [-10.0],
                'color_attr': [-10.0]
        },
        'input_info': {
            'shape': [1, 512, 512, 3],
            'dtype': 'torch.uint8',
            'min': '0.000',
            'max': '254.000'
        },
        'duration': '1.07',
        'decoded_duration': '0.00',
        'type': 'gen_eval'
    }
    Inside each field of the response, the values are lists corresponding to the batch size.    
"""
