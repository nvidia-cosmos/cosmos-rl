#!/usr/bin/env python3
import numpy as np
from safetensors import safe_open

# 尝试用简单的方式读取 zarr
import json
import os

def read_zarr_simple(base_path):
    """简单读取 zarr 格式的数组"""
    # 读取 .zarray 元数据
    zarray_path = os.path.join(base_path, ".zarray")
    with open(zarray_path, 'r') as f:
        meta = json.load(f)
    
    shape = tuple(meta['shape'])
    dtype = np.dtype(meta['dtype'])
    chunks = tuple(meta['chunks'])
    
    # 读取所有 chunks
    data = np.zeros(shape, dtype=dtype)
    
    # 遍历所有 chunk 文件
    for fname in os.listdir(base_path):
        if fname.startswith('.'):
            continue
        
        # 解析 chunk 索引
        indices = fname.split('.')
        if len(indices) != len(shape):
            continue
        
        chunk_indices = tuple(int(i) for i in indices)
        
        # 读取 chunk 数据
        chunk_path = os.path.join(base_path, fname)
        chunk_data = np.fromfile(chunk_path, dtype=dtype)
        
        # 计算 chunk 的位置
        slices = []
        for i, (idx, chunk_size, dim_size) in enumerate(zip(chunk_indices, chunks, shape)):
            start = idx * chunk_size
            end = min(start + chunk_size, dim_size)
            slices.append(slice(start, end))
        
        # 将 chunk 数据放入正确位置
        chunk_shape = tuple(s.stop - s.start for s in slices)
        data[tuple(slices)] = chunk_data.reshape(chunk_shape)
    
    return data

# 读取 JAX 权重
print("=== Reading JAX action_in_proj.kernel ===")
jax_path = "/workspace/new_weights/pi05-b1kpt50-cs32/params/params.action_in_proj.kernel.value"
jax_kernel = read_zarr_simple(jax_path)
print(f"shape: {jax_kernel.shape}")
print(f"mean: {jax_kernel.mean():.6f}, std: {jax_kernel.std():.6f}")
print(f"first 3x3:\n{jax_kernel[:3, :3]}")

# 读取 PyTorch 权重
print("\n=== Reading PyTorch action_in_proj.weight ===")
pytorch_path = "/workspace/comet_weights_pytorch_2/pi05-b1kpt50-cs32/model.safetensors"
with safe_open(pytorch_path, framework="pt", device="cpu") as f:
    pytorch_weight = f.get_tensor("action_in_proj.weight").numpy()
print(f"shape: {pytorch_weight.shape}")
print(f"mean: {pytorch_weight.mean():.6f}, std: {pytorch_weight.std():.6f}")
print(f"first 3x3:\n{pytorch_weight[:3, :3]}")

# 比较 (JAX kernel 是 [in, out]，PyTorch weight 是 [out, in])
print("\n=== Comparison (JAX.T vs PyTorch) ===")
diff = np.abs(jax_kernel.T - pytorch_weight)
print(f"Max diff: {diff.max():.10f}")
print(f"Mean diff: {diff.mean():.10f}")

if diff.max() < 1e-4:
    print("\n✓ action_in_proj weights are essentially identical!")
else:
    print("\n✗ action_in_proj weights are DIFFERENT!")
    print(f"JAX kernel.T[:3,:3]:\n{jax_kernel.T[:3,:3]}")
    print(f"PyTorch weight[:3,:3]:\n{pytorch_weight[:3,:3]}")

# 检查 action_out_proj
print("\n=== action_out_proj comparison ===")
jax_path = "/workspace/new_weights/pi05-b1kpt50-cs32/params/params.action_out_proj.kernel.value"
jax_kernel = read_zarr_simple(jax_path)
with safe_open(pytorch_path, framework="pt", device="cpu") as f:
    pytorch_weight = f.get_tensor("action_out_proj.weight").numpy()

print(f"JAX kernel shape: {jax_kernel.shape}")
print(f"PyTorch weight shape: {pytorch_weight.shape}")
diff = np.abs(jax_kernel.T - pytorch_weight)
print(f"Max diff: {diff.max():.10f}")
if diff.max() < 1e-4:
    print("✓ action_out_proj weights are essentially identical!")
else:
    print("✗ action_out_proj weights are DIFFERENT!")

