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
import PIL

model_path = '/workspace/ruipul/cosmos-rl-private/nemotron_vl/nemotron_siglip2'
safetensor_file = '/workspace/ruipul/cosmos-rl-private/nemotron_vl/outputs/siglip2_official_nemotron_vl_stage1_match_video_rerun/20260306143627/safetensors/step_12398'

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
messages = [
[{
    "role": "user",
    "content": [
        {"type": "video", "video": "/workspace/yangyangt/cosmos-rl-private/nemotron_vl/2dbb31d4-b92a-4399-a35d-ddbdba49072f_2761808000796.camera_front_wide_120fov.mp4"},
        {"type": "text", "text": "describe the video in detail."}
        ]
    }
]
]

processor.tokenizer.pad_token = tokenizer.eos_token
processor.tokenizer.padding_side='left'
processor.video_processor.size['longest_edge'] = 32 * 32 * 196 * 8
processor.video_processor.fps = 2
# 传预采样帧（PIL 列表）时：content 里的 "fps"/"num_frames" 不会被 apply_chat_template 读取。
# 若不设 do_sample_frames=False，会按 fps 再采样；metadata.fps 为 None 会被当成 24，导致 8 帧被采成 4 帧。
# 因此显式关闭帧采样，保留全部 8 帧；或改为传入 num_frames=8。
result = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
    enable_thinking=False,
    padding=True,
    max_num_patches=196
)
response = tokenizer.decode(result['input_ids'][0])
model.to('cuda').to(torch.bfloat16)
model.eval()
result.to('cuda')
with torch.no_grad():
    output = model.generate(
        **result,
        max_new_tokens=512,
        do_sample=False,
        temperature=0.0,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

for i in range(1):
    response = tokenizer.decode(output[i], skip_special_tokens=True)
    print(response)
