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

model_path = '/data/nemotron-vlm/nemotron_v3_vl'
safetensor_file = '/data/yangyangt/nemotron_bridge/nemotron_vl/outputs/Nemotron-12b-vlm-fsdp8-sft/20260123043614/safetensors/step_2700/'

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
    # get expert weightsåå
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


def compare_model_with_safetensor(model, safetensor_dir):
    found_param_name = []
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
            
    return found_param_name

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

with torch.device('meta'):
    model = AutoModel.from_config(config, trust_remote_code=True)
model.to_empty(device='cpu')

set_model_with_weight(model, safetensor_file)
compare_model_with_safetensor(model, model_path)

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

messages = [{
    "role": "user", 
    "content": [
        {"type": "image", "image": "/workspace/multi_framework/single-fresh-red-strawberry-on-table-green-background-food-fruit-sweet-macro-juicy-plant-image-photo.jpg"}, 
        {"type": "text", "text": "Caption the photo"}
        ]
    }
]
result = processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True, enable_thinking=False)
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

response = tokenizer.decode(output[0])
print(response)
