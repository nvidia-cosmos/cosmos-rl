import transformers
from typing import Union, List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoConfig, AutoProcessor,AutoModel, AutoModelForCausalLM
import torch
import json
import importlib
from safetensors import safe_open
from safetensors.torch import save_file
from collections import OrderedDict
import json
import os
from glob import glob
import numpy as np
from safetensors.torch import save_file
from accelerate import init_on_device
import PIL
from PIL import Image

from qwen_vl_utils import process_vision_info

model_path = './nemotron_siglip2'
safetensor_file = '/workspace/jiaxinc/jobs/fast_s15_siglip2-nemotron-moe/20260315105228/safetensors/step_10000'


def set_model_with_weight(model, safetensor_dir):
    safetensors_files = sorted(
        [f for f in os.listdir(safetensor_dir) if f.endswith(".safetensors")]
    )
    safetensor_weight = {}
    for safetensor_file in safetensors_files:
        safetensor_weight[safetensor_file] = safe_open(
            os.path.join(safetensor_dir, safetensor_file),
            framework="pt",
            device='cpu'
        )
    index_dict = json.load(open(os.path.join(safetensor_dir, 'model.safetensors.index.json')))['weight_map']
    expert_weight_dict = {}
    # get expert weights if expert save if wrong (only useful for early version of modeling)
    for key in index_dict.keys():
        if ".mixer.experts.gate_and_up_projs" in key:
            weight = safetensor_weight[index_dict[key]].get_tensor(key)
            weight_name = '.'.join(key.split('.')[:-1])
            for i in range(128):
                expert_weight_dict[f"{weight_name}.{i}.up_proj.weight"] = weight[i]
        elif ".mixer.experts.down_projs" in key:
            weight = safetensor_weight[index_dict[key]].get_tensor(key)
            weight_name = '.'.join(key.split('.')[:-1])
            for i in range(128):
                expert_weight_dict[f"{weight_name}.{i}.down_proj.weight"] = weight[i]


    for name, parameters in model.named_parameters():
        if name in index_dict:
            parameters.data = torch.zeros(parameters.shape, dtype=torch.bfloat16)
            parameters.data.copy_(safetensor_weight[index_dict[name]].get_tensor(name))
        elif "mixer.experts." in name:
            parameters.data = torch.zeros(parameters.shape, dtype=torch.bfloat16)
            parameters.data.copy_(expert_weight_dict[name])
        else:
            raise ValueError(f"parameter {name} is missing")
    
    for name, parameters in model.named_buffers():
        if "rotary_pos_emb" in name or 'rotary_emb' in name:
            # Ignore rope, using init value
            continue
        if name in index_dict:
            # this should be e_correction_bias since it is buffer instead of paramters
            tensor = safetensor_weight[index_dict[name]].get_tensor(name)
            parameters.data = torch.zeros(parameters.shape, dtype=torch.bfloat16)
            parameters.data.copy_(safetensor_weight[index_dict[name]].get_tensor(name))
        else:
            raise ValueError(f"parameter {name} is missing")


def compare_model_with_safetensor(model, safetensor_dir):
    safetensors_files = sorted(
        [f for f in os.listdir(safetensor_dir) if f.endswith(".safetensors")]
    )
    safetensor_weight = {}
    for safetensor_file in safetensors_files:
        safetensor_weight[safetensor_file] = safe_open(
            os.path.join(safetensor_dir, safetensor_file),
            framework="pt",
            device='cpu'
        )

    index_dict = json.load(open(os.path.join(safetensor_dir, 'model.safetensors.index.json')))['weight_map']
    cnt = 0
    for name, parameters in model.named_parameters():
        ori_weight = safetensor_weight[index_dict[name]].get_tensor(name)
        if torch.sum(parameters.data - ori_weight) != 0.0:
            print(name) # Only trainable parameters should be changed
    
    # for name, parameters in model.named_buffers():
    #     if name in index_dict:
    #         ori_weight = safetensor_weight[index_dict[name]].get_tensor(name)
    #         if torch.sum(parameters.data - ori_weight) != 0.0:
    #             print(f"{name}, diff: {torch.sum(parameters.data - ori_weight)}") # Only trainable parameters should be changed

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

with init_on_device("meta", include_buffers=False):
    model = AutoModel.from_config(config, trust_remote_code=True)

model._apply(
    lambda t: torch.empty_like(t, device='cpu')
    if t.device.type == "meta"
    else t.to("cpu"),
    recurse=True,
)

set_model_with_weight(model, safetensor_file)

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model.set_attn_implementation("flash_attention_2")

processor.tokenizer.pad_token = tokenizer.eos_token
processor.tokenizer.padding_side = 'left'
# NOTE:since we use do_resize=False in processor, we don't need to set the size
# processor.video_processor.size['longest_edge'] = 32 * 32 * 196 * 8

video_messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "fps": 2,
                "video": "/workspace/simonz/share/test_samples/night_driving.mp4"
            },
            {"type": "text", "text": "Describe this video in detail."},
        ],
    }
]

image_messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/workspace/simonz/share/test_samples/tennis_girl.png",
            },
            {"type": "text", "text": "Describe this image in detail."},
        ],
    }
]

messages_list = [video_messages, image_messages]

model.to('cuda').to(torch.bfloat16)
model.eval()

for messages in messages_list:
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, image_patch_size=16, return_video_metadata=True, return_video_kwargs=True
    )

    video_metadatas = None
    if video_inputs is not None:
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        for i, v in enumerate(video_inputs):
            print(f"[DEBUG] video[{i}] raw shape: {v.shape}  (frames, C, H, W)")

    if image_inputs is not None:
        for i, img in enumerate(image_inputs):
            print(f"[DEBUG] image[{i}] raw size: {img.size}  (W, H)")

    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_metadatas, 
        return_tensors="pt", 
        do_resize=False, 
        **video_kwargs
    )
    inputs = {k: v.cuda() for k, v in inputs.items()}

    if "video_grid_thw" in inputs:
        for i, thw in enumerate(inputs["video_grid_thw"]):
            print(f"[DEBUG] Video {i}: grid_thw={thw.tolist()} (t={thw[0].item()} frames)")
    if "image_grid_thw" in inputs:
        for i, thw in enumerate(inputs["image_grid_thw"]):
            print(f"[DEBUG] Image {i}: grid_thw={thw.tolist()} (h={thw[1].item()}, w={thw[2].item()})")

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(response)