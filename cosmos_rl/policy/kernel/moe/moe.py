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

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Shard
import torch.distributed._symmetric_memory as symm_mem

try:
    from grouped_gemm import ops
except ImportError:
    print(
        "grouped_gemm is not available. Please run:"
        "pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4"
    )

from cosmos_rl.policy.kernel.moe.indices import generate_permute_indices
from cosmos_rl.policy.kernel.moe.grouped_gemm import group_gemm_imp
from cosmos_rl.policy.kernel.megatron_moe.moe_utils import WeightedSwiGLUFunction
from cosmos_rl.policy.kernel.megatron_moe.token_dispatcher import (
    MoEConfig,
    MoEFlexTokenDispatcher,
)
from cosmos_rl.policy.kernel.symm_mem_recipes import OnDeviceAllToAllV


class GroupedExperts(nn.Module):
    """
    Sparse MoE implementation using all-gather/reduce-scatter primitives.

    Once the experts for a particular token have been identified, this module
    is invoked to compute and average the output of the activated experts.

    Attributes:
        n_routed_experts (int): Total number of experts in the model.
        gate_projs (nn.Parameter): Linear layer for input-to-gate transformation.
        up_projs (nn.Parameter): Linear layer for input-to-hidden transformation.
        down_projs (nn.Parameter): Linear layer for hidden-to-output transformation.
    """

    def __init__(
        self,
        dim: int,
        inter_dim: int,
        n_routed_experts: int,
    ):
        """
        Initializes the GroupedExperts module.

        Args:
            dim (int): Dimension of the model.
            inter_dim (int): Intermediate dimension of the feed-forward layers.
            n_routed_experts (int): Total number of experts in the MoE.
        """
        super().__init__()
        self.n_routed_experts = n_routed_experts
        self.gate_projs = nn.Parameter(
            torch.empty(n_routed_experts, inter_dim, dim)
        )
        self.up_projs = nn.Parameter(
            torch.empty(n_routed_experts, inter_dim, dim)
        )
        self.down_projs = nn.Parameter(
            torch.empty(n_routed_experts, dim, inter_dim)
        )

    def setup_mesh(self, ep_mesh: DeviceMesh) -> None:
        assert ep_mesh is not None:
        assert ep_mesh.ndim == 1, "We only support 1D mesh for MoE"
        self.ep_mesh = ep_mesh
        self.ep_size = ep_mesh.size()
        self.ep_rank = ep_mesh.get_local_rank()
        assert (
            self.n_routed_experts % self.ep_size == 0
        ), f"Number of experts must be divisible by ep_size (ep_size={self.ep_size})"


    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the grouped experts.

        Args:
            x (torch.Tensor): Input tensor. Shape is [num_tokens, model_dim].
            token_mask (torch.Tensor): Boolean mask indicating valid tokens.
                Shape is [num_tokens].
            weights (torch.Tensor): Routing weights for the selected experts.
                Shape is [num_tokens, num_activated_experts].
            indices (torch.Tensor): Indices of the selected experts.
                Shape is [num_tokens, num_activated_experts].

        Returns:
            torch.Tensor: Output tensor after expert computation.
                Shape is [num_tokens, model_dim]
        """
        if not hasattr(self, "ep_mesh"):
            # Default expert parallelism parameters if EP is not enabled.
            self.ep_mesh = None
            self.ep_size = 1
            self.ep_rank = 0

        # Replicate the tensor to all experts. This is sub-optimal but is
        # used by this implementation for correctness.
        if self.ep_size > 1:
            x = DTensor.from_local(
                x, device_mesh=self.ep_mesh, placements=[Shard(0)]
            ).full_tensor()
            weights = DTensor.from_local(
                weights, device_mesh=self.ep_mesh, placements=[Shard(0)]
            ).full_tensor()
            indices = DTensor.from_local(
                indices, device_mesh=self.ep_mesh, placements=[Shard(0)]
            ).full_tensor()
            token_mask = DTensor.from_local(
                token_mask, device_mesh=self.ep_mesh, placements=[Shard(0)]
            ).full_tensor()

        n_local_experts = self.n_routed_experts // self.ep_size
        experts_start_idx = self.ep_rank * n_local_experts
        experts_end_idx = experts_start_idx + n_local_experts

        def get_local_proj(proj, expert_id):
            local_proj = proj.to_local() if isinstance(proj, DTensor) else proj
            return local_proj[expert_id - experts_start_idx]

        y = torch.zeros_like(x)

        active_local_experts = 0
        for i in range(experts_start_idx, experts_end_idx):
            indices_mask = torch.logical_and(indices == i, token_mask.unsqueeze(-1))
            idx, top = torch.where(indices_mask)

            if idx.numel() == 0:
                continue
            active_local_experts += 1

            gate_proj = get_local_proj(self.gate_projs, i)
            down_proj = get_local_proj(self.down_projs, i)
            up_proj = get_local_proj(self.up_projs, i)

            idx_b = idx[:, None].expand(-1, x.size(1))
            x_idx = x.gather(dim=0, index=idx_b)

            expert_out = (
                swiglu(x_idx, gate_proj, down_proj, up_proj) * weights[idx, top, None]
            )

            y.scatter_add_(dim=0, index=idx_b, src=expert_out)

        if active_local_experts == 0:
            # We need to handle the case where no token selects the experts on this device.
            gate_proj = get_local_proj(self.gate_projs, experts_start_idx)
            down_proj = get_local_proj(self.down_projs, experts_start_idx)
            up_proj = get_local_proj(self.up_projs, experts_start_idx)
            expert_out = (
                swiglu(torch.zeros_like(x[0]), gate_proj, down_proj, up_proj)
                * weights[0, 0, None]
            )
            y[0] += expert_out

        if self.ep_size > 1:
            y = DTensor.from_local(y, device_mesh=self.ep_mesh, placements=[Partial()])
            y = y.redistribute(placements=[Shard(0)]).to_local()

        return y


class GroupedExpertsDeepEP(nn.Module):
    """
    Sparse MoE implementation using DeepEP.

    Once the experts for a particular token have been identified, this module
    is invoked to compute and average the output of the activated experts.

    Attributes:
        n_routed_experts (int): Total number of experts in the model.
        gate_and_up_projs part1 / gate_projs (nn.Parameter): Linear layer for input-to-gate transformation.
        gate_and_up_projs part2 / up_projs (nn.Parameter): Linear layer for input-to-hidden transformation.
        down_projs (nn.Parameter): Linear layer for hidden-to-output transformation.
    """

    def __init__(
        self,
        dim: int,
        inter_dim: int,
        n_routed_experts: int,
        n_activated_experts: int,
    ):
        """
        Initializes the GroupedExperts module.

        Args:
            dim (int): Dimension of the model.
            inter_dim (int): Intermediate dimension of the feed-forward layers.
            n_routed_experts (int): Total number of experts in the MoE.
            n_activated_experts (int): Number of activated experts in the MoE.
        """
        super().__init__()
        self.n_activated_experts = n_activated_experts
        self.n_routed_experts = n_routed_experts

        self.gate_and_up_projs = nn.Parameter(
            torch.empty(n_routed_experts, dim, inter_dim * 2)
        )
        self.down_projs = nn.Parameter(
            torch.empty(n_routed_experts, inter_dim, dim)
        )

    def init_token_dispatcher(self, ep_mesh: DeviceMesh):
        assert ep_mesh is not None
        self.ep_size = ep_mesh.size()
        self.ep_rank = ep_mesh.get_local_rank()

        assert (
            self.n_routed_experts % self.ep_size == 0
        ), f"Number of experts must be divisible by ep_size (ep_size={self.ep_size})"

        config = MoEConfig(
            moe_router_topk=self.n_activated_experts,
            num_moe_experts=self.n_routed_experts,
            moe_permute_fusion=True,
        )

        num_local_experts = self.n_routed_experts // self.ep_size

        local_expert_indices_offset = self.ep_rank * num_local_experts
        local_expert_indices = [
            local_expert_indices_offset + i for i in range(num_local_experts)
        ]

        self.token_dispatcher = MoEFlexTokenDispatcher(
            num_local_experts=num_local_experts,
            local_expert_indices=local_expert_indices,
            config=config,
            ep_group=ep_mesh.get_group(),
        )


    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the grouped experts.

        Args:
            x (torch.Tensor): Input tensor. Shape is [num_tokens, model_dim].
            token_mask (torch.Tensor): Boolean mask indicating valid tokens.
                Shape is [num_tokens].
            weights (torch.Tensor): Routing weights for the selected experts.
                Shape is [num_tokens, num_activated_experts].
            indices (torch.Tensor): Indices of the selected experts.
                Shape is [num_tokens, num_activated_experts].

        Returns:
            torch.Tensor: Output tensor after expert computation.
                Shape is [num_tokens, model_dim]
        """
        assert not isinstance(x, DTensor)

        indices = indices.masked_fill(~token_mask.unsqueeze(-1), -1)

        (permuted_local_hidden_states, tokens_per_expert, permuted_probs) = (
            self.token_dispatcher.token_permutation2(
                hidden_states=x,
                num_local_tokens=x.size(0),
                token_probs=weights,
                token_indices=indices,
            )
        )
        permuted_probs = permuted_probs.unsqueeze(-1)

        if torch.count_nonzero(tokens_per_expert) > 0:
            output1 = ops.gmm(
                permuted_local_hidden_states,
                self.gate_and_up_projs.to_local(),
                tokens_per_expert,
                trans_b=False,
            )
            output1_ = WeightedSwiGLUFunction.apply(output1, permuted_probs, False)
            output2 = ops.gmm(
                output1_, self.down_projs.to_local(), tokens_per_expert, trans_b=False
            )
        else:
            output1 = torch.matmul(x[0] * 0, self.gate_and_up_projs.to_local()[0])
            output1_ = WeightedSwiGLUFunction.apply(output1, permuted_probs, False)
            output2 = torch.matmul(output1_, self.down_projs.to_local()[0])

        y = self.token_dispatcher.token_unpermutation(output2)

        return y


def setup_symm_mem(
    max_batch_tokens: int,
    model_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    # Basically, max_seq_len * 2 is enough for all-to-all-v communication.
    overflow = 2

    OnDeviceAllToAllV.max_output_len = max_batch_tokens * overflow

    # Init MoE kernel related buffers
    if GroupedExpertsSymmMem.token_send_buf is None:
        # Input buffer for DP-to-EP shuffle
        GroupedExpertsSymmMem.token_send_buf = symm_mem.empty(
            max_batch_tokens,
            model_dim,  # hidden dim
            dtype=dtype,
            device=device,
        )
        GroupedExpertsSymmMem.token_send_buf.zero_()

        # Input buffer for EP-to-DP shuffle
        GroupedExpertsSymmMem.token_gather_buf = symm_mem.empty(
            max_batch_tokens * overflow,
            model_dim,  # hidden dim
            dtype=dtype,
            device=device,
        )
        GroupedExpertsSymmMem.token_gather_buf.zero_()


class GroupedExpertsSymmMem(nn.Module):
    """
    FeedForward module, support hybrid parallelism including:
    - TP: Shard the experts row/col-wisely across TP groups
    - EP: split the experts into groups, located on EP groups
    - FSDP: Shard the weights across FSDP groups

    Args:
        dim (int): Input dimension.
        inter_dim (int): Intermediate dimension.
        model_args (Qwen3MoeArgs): Model configuration arguments.
    """

    token_send_buf: Optional[torch.Tensor] = None
    token_gather_buf: Optional[torch.Tensor] = None

    def __init__(
        self,
        dim: int,
        inter_dim: int,
        n_routed_experts: int,
    ):
        super().__init__()
        self.total_experts = n_routed_experts
        self.local_experts = n_routed_experts
        self.up_proj = FakeLinear(dim, inter_dim, self.local_experts)
        self.gate_proj = FakeLinear(dim, inter_dim, self.local_experts)
        self.down_proj = FakeLinear(inter_dim, dim, self.local_experts)
        self.act_fn = F.silu

        self.group_gemm_imp = group_gemm_imp()

    def sort_tokens(self, x, topk_ids, topk_weights):
        # This part sorts the token indices so that tokens routed to the
        # same expert reside consecutively. An implication is that tokens
        # to the same "expert group" (i.e., device) are also consecutive.
        # Since this is an "artificial" index creation (final outcome being
        # `idxs`), we don't need gradients here.

        del topk_weights

        with torch.no_grad():
            # [seq_len, n_routed_experts]
            expert_counts = topk_ids.new_zeros((topk_ids.shape[0], self.total_experts))
            # Fill 1 to the selected experts
            expert_counts.scatter_(1, topk_ids, 1)
            tokens_per_expert = expert_counts.sum(dim=0)
            # Token indices for each expert
            token_indices = topk_ids.view(-1).argsort()

        sorted_tokens = x[token_indices // topk_ids.shape[1]]

        return (sorted_tokens, token_indices, tokens_per_expert)

    def get_send_buf(self):
        # [Why detach?] During a first forward-backward step, the buffer would
        # be included in a computational graph. In a second step, autograd will
        # return an error saying "Trying to backward through the graph a second
        # time (or directly access saved tensors more than once)". This is
        # because the buffer is still in the graph, and autograd is trying to
        # backward through the graph a second time. To avoid this, we detach the
        # buffer from the graph. `detach()` returns a new tensor, which shares
        # the same storage with the original one.
        self.token_send_buf.grad = None
        return self.token_send_buf.detach()

    def get_gather_buf(self):
        # See [Why detach?] in `get_send_buf`
        self.token_gather_buf.grad = None
        return self.token_gather_buf.detach()

    def moe_on_device(self, x, topk_ids, topk_weight):
        """
        x: [batch * local_seq_len, dim]
        topk_ids: [batch * local_seq_len, topk]
        topk_weight: [batch * local_seq_len, topk]

        sorted_tokens: [batch * local_seq_len * topk, dim]
        token_indices: [batch * local_seq_len * topk]
        tokens_per_expert: [n_experts]
        """
        (
            sorted_tokens,
            token_indices,
            tokens_per_expert,
        ) = self.sort_tokens(x, topk_ids, topk_weight)
        # keep the seqlen dimension for later use without holding onto the sorted tokens
        seqlen_sorted_tokens = sorted_tokens.shape[0]

        # Sum the tokens over local experts, then we get tokens per EP rank,
        # which is the input splits
        with torch.no_grad():
            # tokens_per_expert: [n_experts, 1]
            # tokens_per_expert_group: [n_experts, 1]
            tokens_per_expert_group = tokens_per_expert.new_empty(
                tokens_per_expert.shape[0]
            )
            # For TP/EP mode, the input is sequencely parallelized
            # So each EP group will have distinct, but the same number of tokens
            # After this collective, tokens_per_expert_group is still of shape [n_experts, 1]

            # Let's say we are on EP rank 0:
            # recv: [(e0, e1, e2 ...), (e0, e1, e2 ...), ...], totally `n_experts` elements
            #        ----------------: tokens from EP group 0 to EP group 0
            #                          ----------------: tokens from EP group 1 to EP group 0
            #                          ...
            # So we can just concat
            dist.all_to_all_single(
                tokens_per_expert_group,
                tokens_per_expert,
                group=self.ep_group,
                async_op=False,
            )
            input_splits = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)
        # Move input to the `token_send_buf` symm mem
        token_send_buf = self.get_send_buf()
        token_send_buf[: token_indices.shape[0]].copy_(sorted_tokens)
        # Note: `out=` avoids copy, but it is not differentiable
        # torch.index_select(x, 0, idxs // topk_ids.shape[1], out=token_send_buf[: idxs.shape[0]])

        # Reference:
        #   1. [TorchTitan](https://github.com/pytorch/torchtitan/blob/main/torchtitan/experiments/deepseek_v3/symm_mem_recipes/triton_on_device_all_to_all_v.py)
        #   2. [Symm-mem-recipes](https://github.com/yifuwang/symm-mem-recipes)
        token_gather_buf, output_splits = OnDeviceAllToAllV.apply(
            token_send_buf,
            input_splits,
            self.ep_group,
        )

        # We need to permute the received tokens so that tokens for the same expert are contiguous.
        # This part prepares a 1D tensor `permuted_indices` for such permutation.
        # This part doesn't need gradient.
        with torch.no_grad():
            ALIGN_SIZE_M = 128
            permuted_indices, m_sizes, m_offsets = generate_permute_indices(
                tokens_per_expert_group,
                self.local_experts,
                self.ep_size,
                ALIGN_SIZE_M,
            )
        # Permute the received tokens so that tokens for the same expert are contiguous.
        contig_tokens = token_gather_buf[permuted_indices]
        # group gemm - handle all three group gemms (up, gate, down for all experts)
        # print(f"m_sizes: {m_sizes}, m_offsets: {m_offsets}")
        hidden_outputs = self.group_gemm_imp(
            contig_tokens,
            m_sizes,
            m_offsets,
            self.gate_proj.weight.to_local(),
            self.up_proj.weight.to_local(),
            self.down_proj.weight.to_local(),
            self.act_fn,
        )

        # Prepare buffer for tokens processed by experts
        processed_tokens = self.get_gather_buf()

        # Move into Symmetric Memory for the return shuffle
        processed_tokens[permuted_indices] = hidden_outputs

        # Now shuffle the tokens back to their original owner, i.e. EP to DP shuffle.
        # The input/output splits are just a reverse of the previous shuffle.
        token_return_buf, _ = OnDeviceAllToAllV.apply(
            processed_tokens,
            output_splits,
            self.ep_group,
        )

        returned_tokens = token_return_buf[:seqlen_sorted_tokens]
        output_tokens = torch.empty_like(returned_tokens)
        output_tokens[token_indices] = returned_tokens

        final_out = (
            output_tokens.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(returned_tokens.dtype)
        )

        return final_out

    def forward(self, hidden_states: torch.Tensor):
        """
        hidden_states: [bsz, seqlen // ep_size, dim]
        """
        assert hasattr(self, "ep_group"), "EP group is not set."
        assert hasattr(self, "ep_size"), "EP size is not set."

        orig_shape = hidden_states.shape
        # topk_idx: [batch * local_seq_len, topk]
        # topk_weight: [batch * local_seq_len, topk]
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        y = self.moe_on_device(hidden_states, topk_idx, topk_weight)
        y = y.view(*orig_shape)
        return self.reshard_helper_layer(y)


class FakeLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_experts: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, out_features, in_features))


def swiglu(x, gate_proj, down_proj, up_proj):
    inter = F.silu(F.linear(x, gate_proj)) * F.linear(x, up_proj)
    return F.linear(inter, down_proj)


class FakeBalancedGate(nn.Module):
    """
    Load balanced gate implementation, spreads tokens uniformly across all experts.
    The rationale for this class is to do performance experiments to understand
    how the load imbalance with real data is impacting end-to-end performance.
    """

    def __init__(self, n_routed_experts: int, n_activated_experts: int):
        super().__init__()
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated_experts

    def forward(
        self,
        x: torch.Tensor,
        *args, **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            weights (torch.Tensor): Routing weights for the selected experts.
            indices (torch.Tensor): Indices of the selected experts.
        """
        del args, kwargs

        n_exp = self.n_routed_experts
        a_exp = self.n_activated_experts
        weights = torch.ones(x.size(0), a_exp, device=x.device) / a_exp
        indices = (
            torch.arange(x.size(0) * a_exp, device=x.device).view(-1, a_exp) % n_exp
        )

        return weights.type_as(x), indices




