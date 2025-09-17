import torch
import transformer_engine as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
import pandas as pd
import argparse

# --- NVFP4 Benchmarking Function (Quantized) ---

def benchmark_nvfp4_gemm(m, n, k, iterations, warmup):
    """
    Runs the NVFP4 GEMM benchmark, creating buffers only once.
    """
    # Setup
    te_dtype = tex.DType.kFloat4E2M1
    device = "cuda"
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    x_shape = (m, k)
    w_shape = (n, k)
    x_dtype = w_dtype = torch.bfloat16
    
    x = torch.randn(x_shape, dtype=x_dtype, device=device)
    w = torch.randn(w_shape, dtype=w_dtype, device=device)

    x_quantizer = NVFP4Quantizer(fp4_dtype=te_dtype, rowwise=True)
    w_quantizer = NVFP4Quantizer(fp4_dtype=te_dtype, rowwise=True)
    
    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
    workspace_size = workspace.numel()
    transa, transb = True, False
    
    # All other GEMM args
    out_quantizer, bias, gelu_input, D_preallocated = None, None, None, None
    bias_dtype = TE_DType[torch.bfloat16]
    use_gelu, use_grad, accumulate, use_split_accumulator = False, False, False, False

    x_nvfp4 = x_quantizer.make_empty(x_shape, dtype=x_dtype, device=device)
    w_nvfp4 = w_quantizer.make_empty(w_shape, dtype=w_dtype, device=device)

    # Warm-up (reusing the empty tensors)
    for _ in range(warmup):
        x_nvfp4 = x_quantizer.update_quantized(x, x_nvfp4)
        w_nvfp4 = w_quantizer.update_quantized(w, w_nvfp4)
        
        _ = tex.generic_gemm(
            w_nvfp4, transa, x_nvfp4, transb, D_preallocated, out_quantizer,
            TE_DType[torch.bfloat16], bias, bias_dtype, use_gelu, gelu_input,
            use_grad, workspace, workspace_size, accumulate, use_split_accumulator
        )[0]
    torch.cuda.synchronize()

    # Timed Benchmarking
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    timings = { "update_quantized_ms": 0.0, "nvfp4_gemm_ms": 0.0 }

    for _ in range(iterations):
        start_event.record()
        x_nvfp4 = x_quantizer.update_quantized(x, x_nvfp4)
        w_nvfp4 = w_quantizer.update_quantized(w, w_nvfp4)
        end_event.record()
        torch.cuda.synchronize()
        timings["update_quantized_ms"] += start_event.elapsed_time(end_event)

        start_event.record()
        _ = tex.generic_gemm(
            w_nvfp4, transa, x_nvfp4, transb, D_preallocated, out_quantizer,
            TE_DType[torch.bfloat16], bias, bias_dtype, use_gelu, gelu_input,
            use_grad, workspace, workspace_size, accumulate, use_split_accumulator
        )[0]
        end_event.record()
        torch.cuda.synchronize()
        timings["nvfp4_gemm_ms"] += start_event.elapsed_time(end_event)

    # Return Averages
    return {
        "M": m, "N": n, "K": k,
        "Update Quantized (ms)": timings["update_quantized_ms"] / iterations,
        "NVFP4 GEMM (ms)": timings["nvfp4_gemm_ms"] / iterations,
    }

# --- BF16 Benchmarking Function (Native) ---

def benchmark_bf16_gemm(m, n, k, iterations, warmup):
    device = "cuda"
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    x = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    w = torch.randn((n, k), dtype=torch.bfloat16, device=device)

    for _ in range(warmup):
        _ = torch.matmul(x, w.T)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    total_time_ms = 0.0

    for _ in range(iterations):
        start_event.record()
        _ = torch.matmul(x, w.T)
        end_event.record()
        torch.cuda.synchronize()
        total_time_ms += start_event.elapsed_time(end_event)
        
    return {"BF16 GEMM (ms)": total_time_ms / iterations}

# --- Quantization Configuration Benchmark ---

def benchmark_quantization_configs(iterations, warmup, mem_bandwidth):
    print("\n" + "="*60)
    print("🚀 Starting Quantization Configuration Benchmark...")
    print("="*60)

    device = "cuda"
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    tensor_dimensions = [
        (4096, 4096),
        (8192, 8192),
        (16384, 16384),
    ]
    
    configs = [
        {'name': 'Row-wise', 'rowwise': True, 'columnwise': False},
        {'name': 'Column-wise', 'rowwise': False, 'columnwise': True},
        {'name': 'Row & Column-wise', 'rowwise': True, 'columnwise': True},
    ]

    all_results = []
    for dims in tensor_dimensions:
        print(f"Benchmarking Tensor Dimension: {dims}...")
        tensor = torch.randn(dims, dtype=torch.bfloat16, device=device)
        
        bytes_read = dims[0] * dims[1] * 2
        theo_time_ms = (bytes_read / mem_bandwidth) * 1000
        
        result_row = {'Dim 1': dims[0], 'Dim 2': dims[1], 'Theoretical Time (ms)': theo_time_ms}

        for config in configs:
            quantizer = NVFP4Quantizer(
                fp4_dtype=tex.DType.kFloat4E2M1,
                rowwise=config['rowwise'],
                columnwise=config['columnwise']
            )

            for _ in range(warmup):
                q_tensor = quantizer.make_empty(dims, dtype=torch.bfloat16, device=device)
                q_tensor = quantizer.update_quantized(tensor, q_tensor)
            torch.cuda.synchronize()

            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            total_time_ms = 0.0
            for _ in range(iterations):
                start_event.record()
                q_tensor = quantizer.make_empty(dims, dtype=torch.bfloat16, device=device)
                q_tensor = quantizer.update_quantized(tensor, q_tensor)
                end_event.record()
                torch.cuda.synchronize()
                total_time_ms += start_event.elapsed_time(end_event)
            
            avg_time = total_time_ms / iterations
            result_row[f"{config['name']} (ms)"] = avg_time

        all_results.append(result_row)

    df_quant = pd.DataFrame(all_results)
    df_quant.set_index(['Dim 1', 'Dim 2'], inplace=True)

    for config in configs:
        col_name = f"{config['name']} (ms)"
        eff_col_name = f"{config['name']} Efficiency (%)"
        df_quant[eff_col_name] = (df_quant['Theoretical Time (ms)'] / df_quant[col_name]) * 100

    print("\n📊 Quantization Configuration Results:\n")
    print(df_quant)
    print("\n" + "="*60)


# --- `make_empty()` Cost Benchmark ---

def benchmark_make_empty_cost(iterations, warmup):
    print("\n" + "="*60)
    print("🚀 Starting make_empty() Cost Benchmark...")
    print("="*60)

    device = "cuda"
    tensor_shape = (8192, 8192)
    tensor = torch.randn(tensor_shape, dtype=torch.bfloat16, device=device)
    quantizer = NVFP4Quantizer(fp4_dtype=tex.DType.kFloat4E2M1, rowwise=True)
    results = []
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    print("Benchmarking Scenario A: Reuse pre-allocated tensor...")
    quantized_tensor = quantizer.make_empty(tensor_shape, dtype=torch.bfloat16, device=device)
    for _ in range(warmup):
        quantizer.update_quantized(tensor, quantized_tensor)
    torch.cuda.synchronize()
    
    total_time_a = 0.0
    for _ in range(iterations):
        start_event.record()
        quantizer.update_quantized(tensor, quantized_tensor)
        end_event.record()
        torch.cuda.synchronize()
        total_time_a += start_event.elapsed_time(end_event)
    avg_time_a = total_time_a / iterations
    results.append({'Scenario': 'A: Reuse Tensor (update_quantized only)', 'Avg Time (ms)': avg_time_a})

    print("Benchmarking Scenario B: Recreate tensor in each step...")
    for _ in range(warmup):
        q_tensor = quantizer.make_empty(tensor_shape, dtype=torch.bfloat16, device=device)
        q_tensor = quantizer.update_quantized(tensor, q_tensor)
    torch.cuda.synchronize()

    total_time_b = 0.0
    for _ in range(iterations):
        start_event.record()
        q_tensor = quantizer.make_empty(tensor_shape, dtype=torch.bfloat16, device=device)
        q_tensor = quantizer.update_quantized(tensor, q_tensor)
        end_event.record()
        torch.cuda.synchronize()
        total_time_b += start_event.elapsed_time(end_event)
    avg_time_b = total_time_b / iterations
    results.append({'Scenario': 'B: Recreate Tensor (make_empty + update)', 'Avg Time (ms)': avg_time_b})
    
    df = pd.DataFrame(results).set_index('Scenario')
    print("\n📊 `make_empty()` Cost Analysis:\n")
    print(df)

    cost_per_call = avg_time_b - avg_time_a
    overhead_percent = (cost_per_call / avg_time_b) * 100
    print(f"\nDerived cost of make_empty() per call: {cost_per_call:.4f} ms")
    print(f"Overhead of recreating vs. reusing: {overhead_percent:.2f}%")
    print("\n" + "="*60)

# --- NEW: End-to-End Linear Layer Benchmark ---

def benchmark_linear_layer_e2e(m, n, k, iterations, warmup):
    """
    Benchmarks an end-to-end linear layer pass (fwd, w_grad, d_grad).
    """
    device = "cuda"
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    # Define shapes and tensors
    activation = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    weight = torch.randn((n, k), dtype=torch.bfloat16, device=device)
    grad = torch.randn((m, n), dtype=torch.bfloat16, device=device)

    # --- 1. BF16 Scenario ---
    for _ in range(warmup):
        _ = torch.matmul(activation, weight.T) # Fwd
        _ = torch.matmul(grad.T, activation)   # w_grad
        _ = torch.matmul(grad, weight)         # d_grad
    torch.cuda.synchronize()
    total_time_bf16 = 0.0
    for _ in range(iterations):
        start_event.record()
        _ = torch.matmul(activation, weight.T)
        _ = torch.matmul(grad.T, activation)
        _ = torch.matmul(grad, weight)
        end_event.record()
        torch.cuda.synchronize()
        total_time_bf16 += start_event.elapsed_time(end_event)
    avg_time_bf16 = total_time_bf16 / iterations

    # --- 2. NVFP4 Scenario ---
    quantizer = NVFP4Quantizer(fp4_dtype=tex.DType.kFloat4E2M1, rowwise=True, columnwise=True)
    
    q_activation = quantizer.make_empty(activation.shape, dtype=torch.bfloat16, device=device)
    q_weight = quantizer.make_empty(weight.shape, dtype=torch.bfloat16, device=device)
    q_grad = quantizer.make_empty(grad.shape, dtype=torch.bfloat16, device=device)
    
    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
    workspace_size = workspace.numel()
    out_quantizer, bias, gelu_input, D_preallocated = None, None, None, None
    bias_dtype = TE_DType[torch.bfloat16]
    use_gelu, use_grad, accumulate, use_split_accumulator = False, False, False, False

    # Warm-up
    for _ in range(warmup):
        q_activation = quantizer.update_quantized(activation, q_activation)
        q_weight = quantizer.update_quantized(weight, q_weight)
        q_grad = quantizer.update_quantized(grad, q_grad)
        # Fwd: activation @ weight.T
        _ = tex.generic_gemm(
            q_weight, True, q_activation, False, D_preallocated, out_quantizer, TE_DType[torch.bfloat16],
            bias, bias_dtype, use_gelu, gelu_input, use_grad, workspace, workspace_size,
            accumulate, use_split_accumulator)[0]
        # w_grad: grad.T @ activation
        _ = tex.generic_gemm(
            q_activation, False, q_grad, True, D_preallocated, out_quantizer, TE_DType[torch.bfloat16],
            bias, bias_dtype, use_gelu, gelu_input, use_grad, workspace, workspace_size,
            accumulate, use_split_accumulator)[0]
        # d_grad: grad @ weight
        _ = tex.generic_gemm(
            q_weight, False, q_grad, False, D_preallocated, out_quantizer, TE_DType[torch.bfloat16],
            bias, bias_dtype, use_gelu, gelu_input, use_grad, workspace, workspace_size,
            accumulate, use_split_accumulator)[0]
    torch.cuda.synchronize()

    total_time_nvfp4 = 0.0
    for _ in range(iterations):
        start_event.record()
        q_activation = quantizer.update_quantized(activation, q_activation)
        q_weight = quantizer.update_quantized(weight, q_weight)
        q_grad = quantizer.update_quantized(grad, q_grad)
        # Fwd: activation @ weight.T
        _ = tex.generic_gemm(
            q_weight, True, q_activation, False, D_preallocated, out_quantizer, TE_DType[torch.bfloat16],
            bias, bias_dtype, use_gelu, gelu_input, use_grad, workspace, workspace_size,
            accumulate, use_split_accumulator)[0]
        # w_grad: grad.T @ activation
        _ = tex.generic_gemm(
            q_activation, False, q_grad, True, D_preallocated, out_quantizer, TE_DType[torch.bfloat16],
            bias, bias_dtype, use_gelu, gelu_input, use_grad, workspace, workspace_size,
            accumulate, use_split_accumulator)[0]
        # d_grad: grad @ weight
        _ = tex.generic_gemm(
            q_weight, False, q_grad, False, D_preallocated, out_quantizer, TE_DType[torch.bfloat16],
            bias, bias_dtype, use_gelu, gelu_input, use_grad, workspace, workspace_size,
            accumulate, use_split_accumulator)[0]
        end_event.record()
        torch.cuda.synchronize()
        total_time_nvfp4 += start_event.elapsed_time(end_event)
    avg_time_nvfp4 = total_time_nvfp4 / iterations

    return {
        'M': m, 'N': n, 'K': k,
        'BF16 E2E Time (ms)': avg_time_bf16,
        'NVFP4 E2E Time (ms)': avg_time_nvfp4
    }

# --- Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(description="Full suite of Transformer Engine benchmarks.")
    parser.add_argument("--iterations", type=int, default=50, help="Number of timed iterations.")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warm-up iterations.")
    args = parser.parse_args()

    # B200 GPU Theoretical Specifications
    B200_MEM_BANDWIDTH = 8 * 1e12
    B200_FP4_PFLOPS = 10 * 1e15
    B200_BF16_PFLOPS = 5 * 1e15

    if not torch.cuda.is_available():
        print("❌ CUDA is not available.")
        return

    # --- Run GEMM Benchmarks ---
    print(f"🚀 Starting GEMM Benchmark on {torch.cuda.get_device_name(0)}...")
    print("-" * 60)
    TEST_CONFIGURATIONS_GEMM = [
        (1024, 1024, 1024), (2048, 2048, 2048),
        (4096, 4096, 4096), (4096, 4096, 8192),
        (8192, 8192, 8192),
    ]
    all_results = []
    for m, n, k in TEST_CONFIGURATIONS_GEMM:
        print(f"Benchmarking GEMM: M={m}, N={n}, K={k}...")
        try:
            results_nvfp4 = benchmark_nvfp4_gemm(m, n, k, args.iterations, args.warmup)
            results_bf16 = benchmark_bf16_gemm(m, n, k, args.iterations, args.warmup)
            all_results.append({**results_nvfp4, **results_bf16})
        except Exception as e:
            print(f"   -> ❌ An error occurred: {e}. Skipping.")
        print("-" * 60)

    if all_results:
        df = pd.DataFrame(all_results)
        df.set_index(["M", "N", "K"], inplace=True)
        df['Speedup (BF16/NVFP4)'] = df['BF16 GEMM (ms)'] / df['NVFP4 GEMM (ms)']
        for index, row in df.iterrows():
            m, n, k = index
            bytes_read = (m * k * 2) + (n * k * 2)
            theo_quant_time_ms = (bytes_read / B200_MEM_BANDWIDTH) * 1000
            df.loc[index, 'Quant Update Efficiency (%)'] = (theo_quant_time_ms / row['Update Quantized (ms)']) * 100
            flops = 2 * m * n * k
            df.loc[index, 'NVFP4 Efficiency (%)'] = ((flops / B200_FP4_PFLOPS * 1000) / row['NVFP4 GEMM (ms)']) * 100
            df.loc[index, 'BF16 Efficiency (%)'] = ((flops / B200_BF16_PFLOPS * 1000) / row['BF16 GEMM (ms)']) * 100
        pd.options.display.float_format = '{:.4f}'.format
        pd.options.display.width = 160
        print("\n📊 GEMM Benchmark Results vs. Theoretical Maximums (B200):\n")
        print(df)

    # --- Run Quantization Config Benchmark ---
    benchmark_quantization_configs(args.iterations, args.warmup, B200_MEM_BANDWIDTH)

    # --- Run make_empty() Cost Benchmark ---
    benchmark_make_empty_cost(args.iterations, args.warmup)

    # --- Run End-to-End Linear Layer Benchmark ---
    print("\n" + "="*60)
    print("🚀 Starting End-to-End Linear Layer Benchmark...")
    print("="*60)
    
    TEST_CONFIGS_E2E = [
        # M,    N,     K
        (4096, 4096, 4096),
        (4096, 4096, 11008), # Llama2-7B MLP FFN1
        (4096, 11008, 4096), # Llama2-7B MLP FFN2
        (4096, 8192, 8192), 
    ]
    
    e2e_results = []
    for m, n, k in TEST_CONFIGS_E2E:
        print(f"Benchmarking E2E Layer: M={m}, N={n}, K={k}...")
        try:
            result = benchmark_linear_layer_e2e(m, n, k, args.iterations, args.warmup)
            e2e_results.append(result)
        except Exception as e:
            print(f"   -> ❌ An error occurred: {e}. Skipping.")
    
    if e2e_results:
        df_e2e = pd.DataFrame(e2e_results)
        df_e2e.set_index(['M', 'N', 'K'], inplace=True)
        df_e2e['Speedup (E2E)'] = df_e2e['BF16 E2E Time (ms)'] / df_e2e['NVFP4 E2E Time (ms)']
        pd.options.display.float_format = '{:.4f}'.format
        pd.options.display.width = 160
        print("\n📊 End-to-End Linear Layer Results:\n")
        print(df_e2e)
    
    print("\nAll benchmarks complete. ✨")


if __name__ == "__main__":
    main()

