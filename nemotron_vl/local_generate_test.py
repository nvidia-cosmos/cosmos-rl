import transformers
from transformers import AutoTokenizer, AutoConfig, AutoProcessor,AutoModel, AutoModelForCausalLM
import torch
import json
import importlib
from safetensors import safe_open
from safetensors.torch import save_file
from collections import OrderedDict
import json
import os
from safetensors.torch import save_file
from accelerate import init_on_device

model_path = './nemotron_vl_overwrite'
original_path = '/data/nemotron-vlm/NVIDIA-Nemotron-3-Nano-VL-30B-A3B-BF16'
safetensor_file = '/data/yangyangt/nemotron_bridge/nemotron_vl/outputs/Nemotron-v3-stage1/20260123160819/safetensors/step_36317'

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
        if "rotary_pos_emb" in name:
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
compare_model_with_safetensor(model, original_path)

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model.set_attn_implementation("flash_attention_2")
messages = [
[{
    "role": "user",
    "content": [
        {"type": "image", "image": "/data/haoyuan/images/unsplash/DhTWLaAPFCI.jpg"},
        {"type": "text", "text": "Describe this image concisely"}
        ]
    }
]
,
[{
    "role": "user",
    "content": [
        {"type": "image", "image": "/workspace/multi_framework/a6e28d8944e8ede5bb43666b17e9b3b6.png"},
        {"type": "text", "text": "Describe this image concisely"}
        ]
    }
],
[{
    "role": "user",
    "content": [
        {"type": "image", "image": "/workspace/multi_framework/a6ffe84dd4d241143c87277ede7af2ae.png"},
        {"type": "text", "text": "Describe this image concisely"}
        ]
    }
]
]

processor.tokenizer.pad_token = tokenizer.eos_token
processor.tokenizer.padding_side='left'
result = processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True, enable_thinking=False, padding=True)
response = tokenizer.decode(result['input_ids'][0])

model.to('cuda').to(torch.bfloat16)
model.eval()
result.to('cuda')
with torch.no_grad():
    output = model.generate(
        **result,
        max_new_tokens=64,
        do_sample=False,
        temperature=0.0,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id
    )

for i in range(3):
    response = tokenizer.decode(output[i], skip_special_tokens=True)
    print(response)
