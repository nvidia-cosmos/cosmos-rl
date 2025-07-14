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

try:
    from grouped_gemm import backend
except ImportError:
    print(
        "grouped_gemm is not available. To enable grouped_gemm, please run:"
        "pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4"
    )
    backend = None


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
        # Fault due to grouped_gemm, where aligned memory is not used which may contains nan
        # Check https://github.com/pytorch/pytorch/issues/154557
        grad_x = torch.nan_to_num(grad_x, nan=0.0)
        grad_w = torch.nan_to_num(grad_w, nan=0.0)
        return grad_x, grad_w, None, None


class FallbackGroupedGemmImpl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, batch_sizes, trans_b):
        assert backend is not None, "grouped_gemm is not available."
        assert (
            torch.count_nonzero(batch_sizes) != 0
        ), "Input batch_sizes should not be all zeros!"
        ctx.save_for_backward(a, b, batch_sizes)
        ctx.trans_b = trans_b
        return backend.gmm(a, b, batch_sizes, trans_a=False, trans_b=trans_b)

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        a, b, batch_sizes = ctx.saved_tensors
        trans_b = ctx.trans_b

        agrad = None
        if ctx.needs_input_grad[0]:
            agrad = backend.gmm(
                grad, b, batch_sizes, trans_a=False, trans_b=not trans_b
            )

        bgrad = None
        if ctx.needs_input_grad[1]:
            lhs, rhs = (grad, a) if trans_b else (a, grad)
            bgrad = backend.gmm(lhs, rhs, batch_sizes, trans_a=True, trans_b=False)
        # Fault due to grouped_gemm, where aligned memory is not used which may contains nan
        agrad = torch.nan_to_num(agrad, nan=0.0)
        bgrad = torch.nan_to_num(bgrad, nan=0.0)
        return agrad, bgrad, None, None


def run_group_gemm_hopper(
    contig_tokens, m_sizes, m_offsets, gate_weight, up_weight, down_weight, act_fn
):
    gate_proj = FakeGroupMMBackwardCheck.apply(
        contig_tokens,
        gate_weight.transpose(-2, -1).contiguous(),
        m_offsets,
        torch.bfloat16,
    )

    up_proj = FakeGroupMMBackwardCheck.apply(
        contig_tokens,
        up_weight.transpose(-2, -1).contiguous(),
        m_offsets,
        torch.bfloat16,
    )

    # Apply activation
    hidden_outputs = act_fn(gate_proj) * up_proj

    # Run the third GEMM (down projection)
    hidden_outputs = FakeGroupMMBackwardCheck.apply(
        hidden_outputs,
        down_weight.transpose(-2, -1).contiguous(),
        m_offsets,
        torch.bfloat16,
    )
    return hidden_outputs


def run_group_gemm_3rd_party(
    contig_tokens, m_sizes, m_offsets, gate_weight, up_weight, down_weight, act_fn
):
    sizes_cpu = m_sizes.cpu().to(torch.int64)

    gate_proj = FallbackGroupedGemmImpl.apply(
        contig_tokens,
        gate_weight.transpose(-2, -1).contiguous(),
        sizes_cpu,
        False,
    )

    up_proj = FallbackGroupedGemmImpl.apply(
        contig_tokens,
        up_weight.transpose(-2, -1).contiguous(),
        sizes_cpu,
        False,
    )

    # Apply activation
    hidden_outputs = act_fn(gate_proj) * up_proj

    # Run the third GEMM (down projection)
    hidden_outputs = FallbackGroupedGemmImpl.apply(
        hidden_outputs,
        down_weight.transpose(-2, -1).contiguous(),
        sizes_cpu,
        False,
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
