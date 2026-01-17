# check_weights.py
import os
from safetensors import safe_open

ckpt_path = "/workspace/comet_weights_pytorch_2/pi05-b1kpt50-cs32/model.safetensors"

print(f"Checking keys in {ckpt_path}...")
with safe_open(ckpt_path, framework="pt", device="cpu") as f:
    keys = list(f.keys())
    print(f"Total keys: {len(keys)}")
    
    print("\n--- Sample Keys ---")
    for k in keys[:5]:
        print(f"  {k}")
    
    print("\n--- Key Analysis ---")
    has_cosmos_prefix = any(k.startswith("paligemma_with_expert.") for k in keys)
    has_module_prefix = any(k.startswith("module.") for k in keys)
    
    print(f"Has 'paligemma_with_expert.' prefix: {has_cosmos_prefix}")
    print(f"Has 'module.' prefix: {has_module_prefix}")
    
    # 检查 Vision Tower
    vision_keys = [k for k in keys if "vision_tower" in k]
    if vision_keys:
        print(f"\nExample Vision Key: {vision_keys[0]}")
    else:
        print("\nWARNING: No vision_tower keys found!")