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

import os
import time
import torch
import deep_ep
import unittest
import inspect
import argparse
import numpy as np
import torch.distributed as dist


# ============================================================================
# Utility Functions (inlined from https://github.com/deepseek-ai/DeepEP/blob/main/tests/utils.py)
# =======================================================================


def init_dist(local_rank: int, num_local_ranks: int):
    """Initialize distributed environment"""
    # NOTES: you may rewrite this function with your own cluster settings
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8361"))
    num_nodes = int(os.getenv("WORLD_SIZE", 1))
    node_rank = int(os.getenv("RANK", 0))

    sig = inspect.signature(dist.init_process_group)
    params = {
        "backend": "nccl",
        "init_method": f"tcp://{ip}:{port}",
        "world_size": num_nodes * num_local_ranks,
        "rank": node_rank * num_local_ranks + local_rank,
    }
    if "device_id" in sig.parameters:
        # noinspection PyTypeChecker
        params["device_id"] = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(**params)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(local_rank)

    return (
        dist.get_rank(),
        dist.get_world_size(),
        dist.new_group(list(range(num_local_ranks * num_nodes))),
    )


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    """Calculate difference between two tensors"""
    x, y = x.double() + 1, y.double() + 1
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


def align_up(x, y):
    """Align x up to nearest multiple of y"""
    return (x + y - 1) // y * y


def per_token_cast_to_fp8(x: torch.Tensor):
    """Cast tensor to FP8 format per token"""
    assert x.dim() == 2
    m, n = x.shape
    aligned_n = align_up(n, 128)
    x_padded = torch.nn.functional.pad(x, (0, aligned_n - n), mode="constant", value=0)
    x_padded_view = x_padded.view(m, -1, 128)
    x_amax = x_padded_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_padded_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(
        m, aligned_n
    )[:, :n].contiguous(), (x_amax / 448.0).view(m, -1)


def per_token_cast_back(x_fp8: torch.Tensor, x_scales: torch.Tensor):
    """Cast tensor back from FP8 format"""
    if x_fp8.numel() == 0:
        return x_fp8.to(torch.bfloat16)

    assert x_fp8.dim() == 2
    m, n = x_fp8.shape
    aligned_n = align_up(n, 128)
    x_fp8_padded = torch.nn.functional.pad(
        x_fp8, (0, aligned_n - n), mode="constant", value=0
    )
    if x_scales.dtype == torch.int:
        x_scales = x_scales.view(dtype=torch.uint8).to(torch.int) << 23
        x_scales = x_scales.view(dtype=torch.float)
    x_fp32_padded = x_fp8_padded.to(torch.float32).view(x_fp8.size(0), -1, 128)
    x_scales = x_scales.view(x_fp8.size(0), -1, 1)
    return (
        (x_fp32_padded * x_scales)
        .view(x_fp8_padded.shape)
        .to(torch.bfloat16)[:, :n]
        .contiguous()
    )


def inplace_unique(x: torch.Tensor, num_slots: int):
    """In-place unique operation on tensor"""
    assert x.dim() == 2
    mask = x < 0
    x_padded = x.masked_fill(mask, num_slots)
    bin_count = torch.zeros((x.size(0), num_slots + 1), dtype=x.dtype, device=x.device)
    bin_count.scatter_add_(1, x_padded, torch.ones_like(x_padded))
    bin_count = bin_count[:, :num_slots]
    sorted_bin_count, sorted_bin_idx = torch.sort(bin_count, dim=-1, descending=True)
    sorted_bin_idx.masked_fill_(sorted_bin_count == 0, -1)
    sorted_bin_idx = torch.sort(sorted_bin_idx, descending=True, dim=-1).values
    x[:, :].fill_(-1)
    valid_len = min(num_slots, x.size(1))
    x[:, :valid_len] = sorted_bin_idx[:, :valid_len]


def bench(fn, num_warmups: int = 50, num_tests: int = 50, post_fn=None):
    """Benchmark a function"""
    # Flush L2 cache with 256 MB data
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")

    # Warmup
    for _ in range(num_warmups):
        fn()

    # Flush L2
    cache.zero_()

    # Testing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    for i in range(num_tests):
        # Record
        start_events[i].record()
        fn()
        end_events[i].record()
        if post_fn is not None:
            post_fn()
    torch.cuda.synchronize()

    times = np.array(
        [s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)]
    )[1:]
    return np.average(times), np.min(times), np.max(times)


class TestGetHiddenBytes(unittest.TestCase):
    """Test cases for get_hidden_bytes function."""

    def setUp(self):
        """Import the module for testing."""
        # Import here to ensure we can mock dependencies
        from cosmos_rl.policy.kernel.megatron_moe.fused_a2a import get_hidden_bytes

        self.get_hidden_bytes = get_hidden_bytes

    def test_float32_tensor(self):
        """Test with float32 tensor (4 bytes per element)."""
        x = torch.randn(10, 20, dtype=torch.float32)
        # hidden_bytes = size(1) * max(element_size(), 2)
        # = 20 * max(4, 2) = 20 * 4 = 80
        self.assertEqual(self.get_hidden_bytes(x), 80)

    def test_float16_tensor(self):
        """Test with float16 tensor (2 bytes per element)."""
        x = torch.randn(10, 30, dtype=torch.float16)
        # hidden_bytes = 30 * max(2, 2) = 30 * 2 = 60
        self.assertEqual(self.get_hidden_bytes(x), 60)

    def test_int8_tensor(self):
        """Test with int8 tensor (1 byte per element)."""
        x = torch.randint(0, 10, (10, 40), dtype=torch.int8)
        # hidden_bytes = 40 * max(1, 2) = 40 * 2 = 80
        self.assertEqual(self.get_hidden_bytes(x), 80)

    def test_different_shapes(self):
        """Test with different tensor shapes."""
        x1 = torch.randn(100, 768, dtype=torch.float32)
        self.assertEqual(self.get_hidden_bytes(x1), 768 * 4)

        x2 = torch.randn(50, 1024, dtype=torch.float32)
        self.assertEqual(self.get_hidden_bytes(x2), 1024 * 4)


# ============================================================================
# Worker Functions (must be at module level for pickling)
# ============================================================================


def _run_test_main(
    args: argparse.Namespace,
    num_sms: int,
    local_rank: int,
    num_ranks: int,
    rank: int,
    buffer: deep_ep.Buffer,
    group: dist.ProcessGroup,
):
    """Main test logic (extracted from original test_main function)"""
    # Settings
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts

    assert num_experts % num_ranks == 0, "num_experts must be divisible by num_ranks"

    if local_rank == 0:
        print(
            f"[config] num_tokens={num_tokens}, hidden={hidden}, num_topk={num_topk}",
            flush=True,
        )

    # Random data
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * rank
    x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    x_e4m3 = per_token_cast_to_fp8(x) if deep_ep.Buffer.is_sm90_compiled() else None
    x_e4m3 = (x_e4m3[0], x_e4m3[1].T.contiguous().T) if x_e4m3 is not None else None
    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs()
        + 1
    )
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)
    topk_weights = (
        torch.ones((num_tokens, num_topk), dtype=torch.float32, device="cuda") * rank
    )
    topk_weights_pure_rand = torch.randn(
        (num_tokens, num_topk), dtype=torch.float32, device="cuda"
    )
    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx = rank_idx.to(torch.int64)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    # Expert meta
    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device="cuda")
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()
    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
    dist.all_reduce(gbl_num_tokens_per_expert, group=group)

    # Rank layout meta
    num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device="cuda")
    token_idx_in_rank = torch.full(
        (num_ranks, num_tokens), -1, dtype=torch.long, device="cuda"
    )
    for i in range(num_ranks):
        num_tokens_per_rank[i] = (rank_idx == i).sum()
        token_sel = (rank_idx == i).max(dim=-1)[0]
        count = token_sel.sum().item()
        tokens = torch.sort(token_sel.to(torch.int), descending=True)[1]
        tokens[:count] = torch.sort(tokens[:count])[0]
        token_idx_in_rank[i][tokens[:count]] = torch.arange(
            count, dtype=torch.long, device="cuda"
        )
    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
    is_token_in_rank = token_idx_in_rank >= 0
    gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
    dist.all_reduce(gbl_num_tokens_per_rank, group=group)

    ref_num_tokens_per_rank, _, ref_num_tokens_per_expert, ref_is_token_in_rank, _ = (
        buffer.get_dispatch_layout(topk_idx, num_experts)
    )

    assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
    assert torch.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)
    assert torch.allclose(ref_is_token_in_rank, is_token_in_rank)

    t = bench(lambda: buffer.get_dispatch_layout(topk_idx, num_experts))[0]
    if local_rank == 0:
        print(f"[layout] Kernel performance: {t * 1000:.3f} ms", flush=True)
        print("", flush=True)
    group.barrier()
    time.sleep(1)

    # Config
    nvl_buffer_size = 256
    config = deep_ep.Config(num_sms, 8, nvl_buffer_size)

    # Test dispatch
    def check_data(check_x, rank_prefix_matrix):
        assert torch.allclose(check_x.amin(dim=1), check_x.amax(dim=1))
        check_start = 0
        for i in range(num_ranks):
            check_end = rank_prefix_matrix[i][rank].item()
            assert (check_x[check_start:check_end, :].int() - i).sum().item() == 0
            check_start = check_end

    for previous_mode in (False, True):
        for async_mode in (False, True):
            for current_x in filter(
                lambda elem: elem is not None, (x_pure_rand, x, x_e4m3)
            ):
                for with_topk in (False, True):
                    if local_rank == 0:
                        print(
                            f'[testing] Running with {"FP8" if isinstance(current_x, tuple) else "BF16"}, '
                            f'{"with" if with_topk else "without"} top-k (async={async_mode}, previous={previous_mode}) ...',
                            flush=True,
                            end="",
                        )
                    dispatch_args = {
                        "x": current_x,
                        "num_tokens_per_rank": num_tokens_per_rank,
                        "is_token_in_rank": is_token_in_rank,
                        "num_tokens_per_expert": num_tokens_per_expert,
                        "config": config,
                        "async_finish": async_mode,
                    }
                    if with_topk:
                        dispatch_args.update(
                            {
                                "topk_idx": topk_idx,
                                "topk_weights": topk_weights_pure_rand
                                if current_x is x_pure_rand
                                else topk_weights,
                            }
                        )
                    if previous_mode:
                        dispatch_args.update({"previous_event": buffer.capture()})

                    (
                        recv_x,
                        recv_topk_idx,
                        recv_topk_weights,
                        recv_num_tokens_per_expert_list,
                        handle,
                        event,
                    ) = buffer.dispatch(**dispatch_args)
                    event.current_stream_wait() if async_mode else ()
                    recv_x = (
                        per_token_cast_back(*recv_x)
                        if isinstance(recv_x, tuple)
                        else recv_x
                    )

                    # Checks
                    rank_prefix_matrix = handle[0]
                    assert gbl_num_tokens_per_rank[rank].item() == recv_x.size(
                        0
                    ), f"{gbl_num_tokens_per_rank[rank].item()} != {recv_x.size(0)}"
                    assert (
                        gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist()
                        == recv_num_tokens_per_expert_list
                    )

                    if current_x is not x_pure_rand:
                        check_data(recv_x, rank_prefix_matrix)

                    recv_topk_weights_clone = None
                    if with_topk:
                        # Check `topk_idx`
                        assert (
                            recv_topk_idx.eq(-1)
                            | (
                                (recv_topk_idx >= 0)
                                & (recv_topk_idx < (num_experts // num_ranks))
                            )
                        ).sum().item() == recv_topk_idx.numel()
                        for i, count in enumerate(recv_num_tokens_per_expert_list):
                            assert recv_topk_idx.eq(i).sum().item() == count

                        # Check `topk_weights`
                        recv_topk_weights_clone = recv_topk_weights.clone()
                        if current_x is not x_pure_rand:
                            recv_topk_weights[recv_topk_idx.eq(-1)] = (
                                recv_topk_weights.amax(
                                    dim=1, keepdim=True
                                ).expand_as(recv_topk_weights)[recv_topk_idx.eq(-1)]
                            )
                            check_data(recv_topk_weights, rank_prefix_matrix)

                    # Test `num_worst_tokens != 0`
                    if with_topk:
                        num_worst_tokens = num_tokens * num_ranks
                        dispatch_args.update({"num_worst_tokens": num_worst_tokens})
                        (
                            recv_worst_x,
                            recv_worst_topk_idx,
                            recv_worst_topk_weights,
                            empty_list,
                            _,
                            event,
                        ) = buffer.dispatch(**dispatch_args)
                        event.current_stream_wait() if async_mode else ()
                        recv_worst_x = (
                            per_token_cast_back(*recv_worst_x)
                            if isinstance(recv_worst_x, tuple)
                            else recv_worst_x
                        )

                        assert len(empty_list) == 0
                        assert num_worst_tokens == recv_worst_x.size(0)
                        assert num_worst_tokens == recv_worst_topk_idx.size(0)
                        assert num_worst_tokens == recv_worst_topk_weights.size(0)
                        assert torch.equal(recv_x, recv_worst_x[: recv_x.size(0)])
                        assert torch.equal(
                            recv_topk_idx, recv_worst_topk_idx[: recv_x.size(0)]
                        )
                        assert torch.equal(
                            recv_topk_weights_clone,
                            recv_worst_topk_weights[: recv_x.size(0)],
                        )
                        assert torch.all(
                            recv_worst_topk_idx[recv_x.size(0) :] == -1
                        ).item()

                    # Test cached dispatch (must without top-k staffs)
                    if not with_topk:
                        dispatch_args = {
                            "x": current_x,
                            "handle": handle,
                            "config": config,
                            "async_finish": async_mode,
                        }
                        if previous_mode:
                            dispatch_args.update({"previous_event": buffer.capture()})
                        recv_x, _, _, _, _, event = buffer.dispatch(**dispatch_args)
                        event.current_stream_wait() if async_mode else ()
                        recv_x = (
                            per_token_cast_back(*recv_x)
                            if isinstance(recv_x, tuple)
                            else recv_x
                        )
                        if current_x is not x_pure_rand:
                            check_data(recv_x, rank_prefix_matrix)

                    # Test combine
                    combine_args = {
                        "x": recv_x,
                        "handle": handle,
                        "config": config,
                        "async_finish": async_mode,
                    }
                    if with_topk:
                        combine_args.update({"topk_weights": recv_topk_weights})
                    if previous_mode:
                        combine_args.update({"previous_event": buffer.capture()})

                    combined_x, combined_topk_weights, event = buffer.combine(
                        **combine_args
                    )
                    event.current_stream_wait() if async_mode else ()
                    check_x = combined_x.float() / is_token_in_rank.sum(
                        dim=1
                    ).unsqueeze(1)
                    ref_x = x_pure_rand if current_x is x_pure_rand else x
                    assert calc_diff(check_x, ref_x) < 5e-6

                    if with_topk:
                        check_topk_weights = (
                            combined_topk_weights
                            if (current_x is x_pure_rand)
                            else (
                                combined_topk_weights
                                / is_token_in_rank.sum(dim=1).unsqueeze(1)
                            )
                        )
                        ref_topk_weights = (
                            topk_weights_pure_rand
                            if current_x is x_pure_rand
                            else topk_weights
                        )
                        assert calc_diff(check_topk_weights, ref_topk_weights) < 1e-9

                    # For later tuning
                    dispatch_bf16_nvl_recv_bytes = recv_x.numel() * 2
                    combine_bf16_nvl_send_bytes = dispatch_bf16_nvl_recv_bytes

                    if local_rank == 0:
                        print(" passed", flush=True)

    if local_rank == 0:
        print("", flush=True)

    # Tune dispatch performance
    best_dispatch_results = None
    fp8_factor = (1 + 4 / 128) / 2
    for current_x in filter(lambda elem: elem is not None, (x_e4m3, x)):
        best_time, best_results = 1e10, None
        nvl_recv_bytes = (
            (dispatch_bf16_nvl_recv_bytes * fp8_factor)
            if isinstance(current_x, tuple)
            else dispatch_bf16_nvl_recv_bytes
        )
        for nvl_chunk_size in tuple(range(4, 33, 2)) + (0,):
            if nvl_chunk_size > 0:
                config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size)
            else:
                # Test default config as well
                deep_ep.Buffer.set_num_sms(num_sms)
                config = deep_ep.Buffer.get_dispatch_config(num_ranks)
            tune_args = {"x": current_x, "handle": handle, "config": config}
            t = bench(lambda: buffer.dispatch(**tune_args))[0]  # noqa: B023
            if t < best_time and nvl_chunk_size > 0:
                best_time, best_results = t, (num_sms, nvl_chunk_size)
            if local_rank == 0:
                print(
                    f'[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size if nvl_chunk_size else "default"}: '
                    f'{nvl_recv_bytes / 1e9 / t:.2f} GB/s (NVL), {t * 1e6:.2f} us',
                    flush=True,
                )
        if local_rank == 0:
            print(
                f'[tuning] Best dispatch ({"FP8" if isinstance(current_x, tuple) else "BF16"}): '
                f'SMs {best_results[0]}, NVL chunk {best_results[1]}, '
                f'{nvl_recv_bytes / 1e9 / best_time:.2f} GB/s (NVL), t: {best_time * 1e6:.2f} us',
                flush=True,
            )
            print("", flush=True)

        # Gather the best config from rank 0 and the first test setting
        if best_dispatch_results is None:
            best_dispatch_results = torch.tensor(
                [best_results[0], best_results[1]], dtype=torch.int32, device="cuda"
            )
            all_best_fp8_results_list = [
                torch.zeros_like(best_dispatch_results)
                for _ in range(torch.distributed.get_world_size())
            ]
            dist.all_gather(
                all_best_fp8_results_list, best_dispatch_results, group=group
            )
            best_dispatch_results = all_best_fp8_results_list[0].tolist()

    dispatch_config = deep_ep.Config(
        best_dispatch_results[0], best_dispatch_results[1], nvl_buffer_size
    )

    dispatch_args = {
        "x": x,
        "num_tokens_per_rank": num_tokens_per_rank,
        "is_token_in_rank": is_token_in_rank,
        "num_tokens_per_expert": num_tokens_per_expert,
        "config": dispatch_config if dispatch_config is not None else config,
    }
    recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)

    # Tune combine performance
    best_time, best_results = 1e10, None
    for nvl_chunk_size in tuple(range(1, 17, 1)) + (0,):
        if nvl_chunk_size > 0:
            config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size)
        else:
            # Test default config as well
            deep_ep.Buffer.set_num_sms(num_sms)
            config = deep_ep.Buffer.get_combine_config(num_ranks)
        tune_args = {"x": recv_x, "handle": handle, "config": config}
        t = bench(lambda: buffer.combine(**tune_args))[0]  # noqa: B023
        if local_rank == 0:
            print(
                f'[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size if nvl_chunk_size else "default"}: '
                f'{combine_bf16_nvl_send_bytes / 1e9 / t:.2f} GB/s (NVL), {t * 1e6:.2f} us',
                flush=True,
            )
            if t < best_time and nvl_chunk_size > 0:
                best_time, best_results = t, (num_sms, nvl_chunk_size)

    if local_rank == 0:
        print(
            f"[tuning] Best combine: SMs {best_results[0]}, NVL chunk {best_results[1]}: "
            f"{combine_bf16_nvl_send_bytes / 1e9 / best_time:.2f} GB/s (NVL), t: {best_time * 1e6:.2f} us",
            flush=True,
        )
        print("", flush=True)


def _worker_test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    """Worker function for multiprocessing (must be at module level for pickling)"""
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    # Note: test_ll_compatibility is disabled to avoid dependency on test_low_latency
    buffer = deep_ep.Buffer(
        group,
        int(2e9),
        0,  # num_rdma_bytes
        low_latency_mode=False,
        num_qps_per_rank=1,
        explicitly_destroy=True,
        allow_mnnvl=args.allow_mnnvl,
        use_fabric=args.use_fabric,
    )
    torch.manual_seed(rank)

    for i in (24,):
        _run_test_main(args, i, local_rank, num_ranks, rank, buffer, group)
        if local_rank == 0:
            print("", flush=True)

    # Destroy the buffer runtime and communication group
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


# ============================================================================
# Test Class
# ============================================================================


class TestDeepEP(unittest.TestCase):
    """Test suite for DeepEP intranode kernels (self-contained version)"""

    # Class variables for test configuration
    num_processes = 8
    num_tokens = 4096
    hidden = 7168
    num_topk = 8
    num_experts = 256
    allow_mnnvl = False
    use_fabric = False

    @classmethod
    def setUpClass(cls):
        """Parse command line arguments or environment variables"""
        # Try to get configuration from environment variables first
        # For num_processes: use env var if set, otherwise auto-detect GPU count
        if "NUM_PROCESSES" in os.environ:
            cls.num_processes = int(os.environ["NUM_PROCESSES"])
        else:
            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            cls.num_processes = gpu_count if gpu_count > 0 else cls.num_processes

        cls.num_tokens = int(os.getenv("NUM_TOKENS", cls.num_tokens))
        cls.hidden = int(os.getenv("HIDDEN", cls.hidden))
        cls.num_topk = int(os.getenv("NUM_TOPK", cls.num_topk))
        cls.num_experts = int(os.getenv("NUM_EXPERTS", cls.num_experts))
        cls.allow_mnnvl = os.getenv("ALLOW_MNNVL", "").lower() in ("true", "1", "yes")
        cls.use_fabric = os.getenv("USE_FABRIC", "").lower() in ("true", "1", "yes")

    def test_intranode_kernels(self):
        """Main test for intranode EP kernels"""
        # Create a simple namespace object to hold configuration
        args = argparse.Namespace(
            num_processes=self.num_processes,
            num_tokens=self.num_tokens,
            hidden=self.hidden,
            num_topk=self.num_topk,
            num_experts=self.num_experts,
            allow_mnnvl=self.allow_mnnvl,
            use_fabric=self.use_fabric,
        )

        # Run the test with multiprocessing
        torch.multiprocessing.spawn(
            _worker_test_loop,
            args=(self.num_processes, args),
            nprocs=self.num_processes,
        )


def suite():
    """Create test suite"""
    test_suite = unittest.TestSuite()
    test_suite.addTest(TestDeepEP("test_intranode_kernels"))
    return test_suite


if __name__ == "__main__":
    # Support command line arguments
    parser = argparse.ArgumentParser(
        description="Test intranode EP kernels (unittest version)"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Number of processes to spawn (default: auto-detect from GPU count)",
    )
    parser.add_argument(
        "--num-tokens", type=int, default=4096, help="Number of tokens (default: 4096)"
    )
    parser.add_argument(
        "--hidden", type=int, default=7168, help="Hidden dimension size (default: 7168)"
    )
    parser.add_argument(
        "--num-topk", type=int, default=8, help="Number of top-k experts (default: 8)"
    )
    parser.add_argument(
        "--num-experts", type=int, default=256, help="Number of experts (default: 256)"
    )
    parser.add_argument(
        "--allow-mnnvl", action="store_true", help="Enable MNNVL support"
    )
    parser.add_argument("--use-fabric", action="store_true", help="Enable fabric mode")
    parser.add_argument(
        "--unittest-args", nargs="*", help="Additional unittest arguments"
    )
    args, unknown = parser.parse_known_args()

    # Set class variables from command line arguments
    # Automatically use GPU count if num_processes is not specified
    if args.num_processes is None:
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        TestDeepEP.num_processes = gpu_count if gpu_count > 0 else 8
        print(
            f"Auto-detected {gpu_count} GPUs, setting num_processes={TestDeepEP.num_processes}",
            flush=True,
        )
    else:
        TestDeepEP.num_processes = args.num_processes
    TestDeepEP.num_tokens = args.num_tokens
    TestDeepEP.hidden = args.hidden
    TestDeepEP.num_topk = args.num_topk
    TestDeepEP.num_experts = args.num_experts
    TestDeepEP.allow_mnnvl = args.allow_mnnvl
    TestDeepEP.use_fabric = args.use_fabric

    # Prepare unittest arguments
    unittest_argv = ["test_deepep.py"] + (args.unittest_args or []) + unknown

    # Run tests
    unittest.main(argv=unittest_argv, verbosity=1)
