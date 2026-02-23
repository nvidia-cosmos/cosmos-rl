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
import orjson

model_path = './nemotron_vl_overwrite'
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

def read_jsonl(jsonl_file):
    data_list = []
    cnt = 0
    with open(jsonl_file) as f:
        for line in f:
            line = line.strip()
            if line:
                data_list.append(orjson.loads(line)['messages'])
            if cnt > 100:
                break
            cnt += 1
    return data_list

def extract_image_mask(input_ids, processor):
    image_mask = input_ids == processor.image_token_id
    video_mask = input_ids == processor.video_token_id
    image_mask = None if (~image_mask).all() else image_mask
    video_mask = None if (~video_mask).all() else video_mask
    return image_mask, video_mask

def modify_messages(messages):
    assistant_contents = []
    for message in messages:
        if isinstance(message['content'], str):
            message['content'] = [{'type': 'text', 'text': message['content']}]
        for content in message['content']:
            if content['type'] == 'image':
                content['image'] = content['image'].replace('workspace', 'data')
            elif content['type'] == 'video':
                content['video'] = content['video'].replace('workspace', 'data')
        if message['role'] == 'assistant':
            for content in message['content']:
                if content['type'] == 'text':
                    assistant_contents.append(content['text'])
    return messages, assistant_contents

def extract_assistant_mask(input_ids, assistant_contents, tokenizer):
    """
    Extracts a binary mask marking assistant response tokens in input_ids.
    
    Args:
        input_ids: Tensor of shape (seq_len,) or (batch_size, seq_len)
        assistant_contents: List of assistant response strings (in conversation order)
        tokenizer: Hugging Face tokenizer instance
    
    Returns:
        Tensor of same shape as input_ids with 1s at assistant token positions
    """
    # Handle batch dimension (only batch_size=1 supported with single assistant_contents list)
    original_shape = input_ids.shape
    if input_ids.ndim == 2:
        if input_ids.shape[0] != 1:
            raise ValueError(
                "Batch size > 1 not supported with single assistant_contents list. "
                "Provide per-example assistant_contents or process batch elements separately."
            )
        input_ids = input_ids.squeeze(0)
    
    assistant_mask = torch.zeros_like(input_ids)
    start_search = 0
    
    for assistant_content in assistant_contents:
        # Critical: Encode WITHOUT special tokens and WITH proper spacing context
        # Use add_special_tokens=False to avoid BOS/EOS tokens
        # Prepend space if not at conversation start (heuristic for space-sensitive tokenizers)
        assistant_tokens = tokenizer.encode(
            assistant_content, 
            add_special_tokens=False,
            truncation=False
        )
        
        if not assistant_tokens:
            continue
            
        token_tensor = torch.tensor(assistant_tokens, device=input_ids.device)
        seq_len = len(token_tensor)
        
        # Search forward from last match position (respects conversation order)
        found = False
        for i in range(start_search, len(input_ids) - seq_len + 1):
            if torch.equal(input_ids[i:i+seq_len], token_tensor):
                assistant_mask[i:i+seq_len] = 1
                start_search = i + seq_len  # Continue search after this match
                found = True
                break
        
        # Fallback: Full sequence search if not found in forward pass
        if not found:
            for i in range(0, len(input_ids) - seq_len + 1):
                if torch.equal(input_ids[i:i+seq_len], token_tensor):
                    assistant_mask[i:i+seq_len] = 1
                    start_search = i + seq_len
                    found = True
                    break
        
        # Optional: Debug warning for missing content (remove in production)
        if not found:
            print(f"Warning: Assistant content not found in input_ids: '{assistant_content[:30]}...'")
    
    # Restore original shape if batch dimension was present
    if len(original_shape) == 2:
        assistant_mask = assistant_mask.unsqueeze(0)
    
    return assistant_mask

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

# Update image_processor's max_pixel if you want
processor.image_processor.max_pixel = 8192 * 0.9 * 32 * 32

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model.set_attn_implementation("flash_attention_2")

processor.tokenizer.pad_token = tokenizer.eos_token
data_list = read_jsonl('./infinitmm_stage1_qwenvl_filter_clip_id_shuf_5m.jsonl')
model.to('cuda').to(torch.bfloat16)
model.eval()

for messages in data_list[:1]:
    messages, assistant_contents = modify_messages(messages)
    tokenizer_result = processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True, enable_thinking=False, padding=True)
    image_mask, video_mask = extract_image_mask(tokenizer_result['input_ids'], processor)
    if image_mask is None and video_mask is None:
        raise ValueError("no image token found")
    visual_mask = image_mask if image_mask is not None else video_mask
    assistant_mask = extract_assistant_mask(tokenizer_result['input_ids'], assistant_contents, tokenizer).bool()

    tokenizer_result.to('cuda')
    with torch.no_grad():
        output = model.forward(
            **tokenizer_result,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
    layer_type_list = model.config.text_config.layers_block_type
    all_hidden_states = output.hidden_states
    for layer_type, hidden_state in zip(layer_type_list, all_hidden_states[1:]): # first element in all_hidden_states is embedding
        hiddens_states_dim = hidden_state.shape[-1]
        visual_hidden_states = hidden_state[visual_mask.unsqueeze(-1).repeat(1, 1, hiddens_states_dim)].view(-1, hiddens_states_dim)
        assistant_hidden_states = hidden_state[assistant_mask.unsqueeze(-1).repeat(1, 1, hiddens_states_dim)].view(-1,hiddens_states_dim)
        print(f"layer_type: {layer_type}, visual_hidden_states: {visual_hidden_states}, assistant_hidden_states: {assistant_hidden_states}")