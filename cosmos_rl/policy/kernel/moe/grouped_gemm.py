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

import torch


class FakeGroupMMBackwardCheck(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, m_offsets, out_dtype):
        ctx.save_for_backward(x, w, m_offsets)
        ctx.out_dtype = out_dtype
        return torch._grouped_mm(x, w, m_offsets, out_dtype=out_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, w, m_offsets = ctx.saved_tensors
        out_dtype = ctx.out_dtype

        grad_x = torch._grouped_mm(
            grad_output, w.transpose(-2, -1), m_offsets, out_dtype=out_dtype
        )
        grad_w = torch._grouped_mm(
            x.transpose(-2, -1), grad_output, m_offsets, out_dtype=out_dtype
        )
        return grad_x, grad_w, None, None


def run_group_gemm_hopper(
    contig_tokens, m_sizes, m_offsets, gate_weight, up_weight, down_weight, act_fn
):
    gate_proj = FakeGroupMMBackwardCheck.apply(
        contig_tokens,
        gate_weight,
        m_offsets,
        torch.bfloat16,
    )

    up_proj = FakeGroupMMBackwardCheck.apply(
        contig_tokens,
        up_weight,
        m_offsets,
        torch.bfloat16,
    )

    # Apply activation
    hidden_outputs = act_fn(gate_proj) * up_proj

    # Run the third GEMM (down projection)
    hidden_outputs = FakeGroupMMBackwardCheck.apply(
        hidden_outputs,
        down_weight,
        m_offsets,
        torch.bfloat16,
    )
    return hidden_outputs


def run_group_gemm_3rd_party(
    contig_tokens, m_sizes, m_offsets, gate_weight, up_weight, down_weight, act_fn
):
    try:
        from grouped_gemm import ops
    except ImportError:
        print(
            "grouped_gemm is not available. Please run:"
            "pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4"
        )
        raise

    sizes_cpu = m_sizes.cpu().to(torch.int64)

    gate_proj = ops.gmm(
        contig_tokens,
        gate_weight,
        sizes_cpu,
        trans_b=False,
    )

    up_proj = ops.gmm(
        contig_tokens,
        up_weight,
        sizes_cpu,
        trans_b=False,
    )

    # Apply activation
    hidden_outputs = act_fn(gate_proj) * up_proj

    # Run the third GEMM (down projection)
    hidden_outputs = ops.gmm(
        hidden_outputs,
        down_weight,
        sizes_cpu,
        trans_b=False,
    )
    return hidden_outputs


use_torch_group_gemm_impl = None


def group_gemm_imp():
    global use_torch_group_gemm_impl
    if use_torch_group_gemm_impl is None:
        cuda_device_props = torch.cuda.get_device_properties()
        # currently torch._grouped_mm is only available on nightly build
        use_torch_group_gemm_impl = (
            cuda_device_props.major == 9
            and cuda_device_props.minor == 0
            and hasattr(torch, "_grouped_mm")
        )

    # Use torch implemetation if torch._grouped_mm is available, otherwise use 3rd party implementation
    return (
        run_group_gemm_hopper if use_torch_group_gemm_impl else run_group_gemm_3rd_party
    )
