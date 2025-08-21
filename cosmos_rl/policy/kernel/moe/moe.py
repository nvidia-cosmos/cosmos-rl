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

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard

try:
    from grouped_gemm import ops
except ImportError:
    print(
        "grouped_gemm is not available. Please run:"
        "pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4"
    )

from cosmos_rl.policy.kernel.megatron_moe.moe_utils import (
    WeightedSwiGLUFunction,
)
from cosmos_rl.policy.kernel.megatron_moe.token_dispatcher import (
    MoEConfig,
    MoEFlexTokenDispatcher,
)
from cosmos_rl.policy.kernel.mlp import MLP


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
        self.n_routed_experts = args.n_routed_experts
        self.gate_projs = nn.Parameter(
            torch.empty(n_routed_experts, moe_inter_dim, dim)
        )
        self.up_projs = nn.Parameter(
            torch.empty(n_routed_experts, moe_inter_dim, dim)
        )
        self.down_projs = nn.Parameter(
            torch.empty(n_routed_experts, dim, moe_inter_dim)
        )

        # Default expert parallelism parameters if EP is not enabled.
        self.ep_mesh = None
        self.ep_size = 1
        self.ep_rank = 0

    def setup_mesh(self, ep_mesh: DeviceMesh) -> None:
        assert ep_mesh is not None:
        assert ep_mesh.ndim == 1, "We only support 1D mesh for MoE"
        self.ep_mesh = ep_mesh
        self.ep_size = ep_mesh.size()
        self.ep_rank = ep_mesh.get_local_rank()
        assert (
            self.n_routed_experts % self.ep_size == 0
        ), f"Number of experts must be divisible by ep_size (ep_size={ep_size})"


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
            num_moe_experts=self..n_routed_experts,
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


def swiglu(x, gate_proj, down_proj, up_proj):
    inter = F.silu(F.linear(x, gate_proj)) * F.linear(x, up_proj)
    return F.linear(inter, down_proj)


