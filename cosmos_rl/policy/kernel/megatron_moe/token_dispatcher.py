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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from .fused_a2a import (
    fused_combine,
    fused_dispatch,
)
from .fused_indices_converter import (
    fused_indices_to_multihot,
)
from .moe_utils import permute, unpermute, sort_chunks_by_idxs

SHARING_DEEPEP_MANAGER = True

""" We use the following notation throughout this file:
     H: hidden size
     B: micro batch size
     S: sequence length
     TP: tensor model parallel size
     EP: expert model parallel size
     num_local_tokens: S/TP*B
     num_global_tokens: num_local_tokens*TP*EP
"""


class _DispatchManager(ABC):
    """
    A manager class to handle dispatch and combine processes for MoE models.

    DispatcherManager handles token dispatching according to the routing_map of format
    [num_local_tokens, world_size, num_instances]. The routing_map is a 3D tensor where each
    element indicates whether a token should be sent to a specific rank.

    num_instances is the maximum number of tokens instances dispatched into a target rank, it
    can be the number of local experts, or the size of sub_group.
    """

    @abstractmethod
    def setup_metadata(self, routing_map: torch.Tensor, probs: torch.Tensor):
        """Set up metadata of routing_map and probs."""
        pass

    @abstractmethod
    def dispatch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Dispatch the hidden_states according to the routing_map."""
        pass

    @abstractmethod
    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Combine the hidden_states after expert processing."""
        pass

    @abstractmethod
    def get_dispached_metadata(self) -> torch.Tensor:
        """Get the metadata of the dispatched hidden_states."""
        pass

    @abstractmethod
    def get_permuted_hidden_states_by_experts(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Get the permuted hidden states by instances."""
        pass

    @abstractmethod
    def get_restored_hidden_states_by_experts(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Get the restored hidden states by instances."""
        pass


class _DeepepManager(_DispatchManager):
    """
    A manager class to handle fused all-to-all communication processes for MoE models using
    DeepEP backend. See https://github.com/deepseek-ai/deepep for more details.

    The workflow of the DeepEP dispatcher is:
    (1) setup_metadata(): Process routing map and probabilities to prepare dispatch metadata
    (2) dispatch():
        - Use fused kernel to permute tokens and perform all-to-all communication in single step
    (3) get_permuted_hidden_states_by_instances():
        - Convert routing map and probabilities to multihot format
        - Permute tokens using fused kernel
    (4) get_restored_hidden_states_by_instances():
        - Reverse permutation using fused kernel
    (5) combine():
        - Reverse process using fused kernel to unpermute and perform all-to-all in single step

    This implementation uses fused communication kernels (fused_dispatch/fused_combine) that
    combine permutation and communication operations for improved efficiency compared to
    separate permute+alltoall steps.
    """

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        capacity_factor: Optional[float] = None,
        num_experts: Optional[int] = None,
        num_local_experts: Optional[int] = None,
        router_dtype: Optional[str] = None,
        moe_router_expert_pad_multiple: Optional[int] = None,
    ):
        self.group = group
        self.router_topk = router_topk
        self.capacity_factor = capacity_factor
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.router_dtype = router_dtype
        self.moe_router_expert_pad_multiple = moe_router_expert_pad_multiple

        # Metadata
        self.token_indices: Optional[torch.Tensor] = None
        self.token_probs: Optional[torch.Tensor] = None
        # Handle used for combine operation
        self.handle = None

        if fused_dispatch is None:
            raise ImportError(
                "DeepEP is not installed. Please install DeepEP package from "
                "https://github.com/deepseek-ai/deepep."
            )

    def setup_metadata(self, num_local_tokens: int, probs: torch.Tensor):
        """
        Process routing map and probabilities to prepare dispatch metadata
        """
        probs = probs.reshape(num_local_tokens, self.num_experts)
        # Convert the format of routing map from multihot to indices.
        self.token_probs, self.token_indices = torch.topk(
            probs, self.router_topk, dim=-1
        )
        # Mask the indices of dropped tokens with -1
        if self.capacity_factor is not None:
            mask = self.token_probs == 0
            self.token_indices = self.token_indices.masked_fill(mask, -1)

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ) -> torch.Tensor:
        """
        Dispatch the hidden_states
        """
        # DeepEP only supports float32 probs
        if self.token_probs.dtype != torch.float32:
            if self.token_probs.dtype in [torch.bfloat16, torch.float16]:
                # print("DeepEP only supports float32 probs, please set --moe-router-dtype=fp32")
                # TODO: remove this
                pass
            self.token_probs = self.token_probs.float()  # downcast or upcast
        (
            hidden_states,
            dispatched_indices,
            dispatched_probs,
            num_tokens_per_expert,
            handle,
        ) = fused_dispatch(
            hidden_states,
            self.token_indices,
            self.token_probs,
            self.num_experts,
            self.group,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        self.handle = handle
        self.tokens_per_expert = num_tokens_per_expert
        self.dispatched_indices = dispatched_indices
        self.dispatched_probs = dispatched_probs

        return hidden_states

    def _indices_to_multihot(self, indices, probs):
        """
        Converts a tensor of indices to a multihot vector.

        Args:
            indices (torch.Tensor): [num_tokens, topk] token indices, where -1 means masked out.
            probs (torch.Tensor): [num_tokens, topk] token probabilities.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - routing_map: Multihot vector.
                - probs: Multihot probabilities.
        """
        batch_size = indices.shape[0]
        multihot_routing_map = torch.zeros(
            (batch_size, self.num_local_experts),
            dtype=torch.long,
            device=indices.device,
        )

        multihot_probs = torch.zeros(
            (batch_size, self.num_local_experts),
            dtype=torch.float,
            device=indices.device,
        )

        mask = indices != -1
        valid_indices = indices[mask]
        row_indices = torch.arange(batch_size, device=indices.device).repeat_interleave(
            mask.sum(dim=1)
        )
        multihot_routing_map[row_indices, valid_indices] = 1
        multihot_probs[row_indices, valid_indices] = probs[mask]
        return multihot_routing_map.bool(), multihot_probs

    def get_dispached_metadata(self) -> torch.Tensor:
        return self.dispatched_indices, self.dispatched_probs

    def get_number_of_tokens_per_expert(self) -> torch.Tensor:
        """
        Get the number of tokens per expert.
        """
        return self.tokens_per_expert

    def combine(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ) -> torch.Tensor:
        """
        Reverse process using fused kernel to unpermute and perform all-to-all in single step
        """
        hidden_states, _ = fused_combine(
            hidden_states,
            self.group,
            self.handle,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        # Release the handle after combine operation
        self.handle = None
        return hidden_states

    def get_permuted_hidden_states_by_experts(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        - Convert routing map and probabilities to multihot format
        - Permute tokens using fused kernel
        """
        if self.permute_fusion:
            self.dispatched_routing_map, self.dispatched_probs = (
                fused_indices_to_multihot(
                    self.dispatched_indices,
                    self.dispatched_probs,
                    self.num_local_experts,
                )
            )
        else:
            self.dispatched_routing_map, self.dispatched_probs = (
                self._indices_to_multihot(
                    self.dispatched_indices, self.dispatched_probs
                )
            )
        if self.moe_router_expert_pad_multiple:
            with torch.cuda.nvtx.range("pad_routing_map"):
                from megatron.core.transformer.moe.moe_utils import pad_routing_map

                self.dispatched_routing_map = pad_routing_map(
                    self.dispatched_routing_map, self.moe_router_expert_pad_multiple
                )
            # self.tokens_per_expert = self.dispatched_routing_map.sum(dim=0)
            self.tokens_per_expert = (
                torch.ceil(self.tokens_per_expert / self.moe_router_expert_pad_multiple)
                * self.moe_router_expert_pad_multiple
            )
            self.tokens_per_expert = self.tokens_per_expert.long()

        self.hidden_shape_before_permute = hidden_states.shape
        assert self.dispatched_probs.dtype == torch.float32, (
            "DeepEP only supports float32 probs"
        )
        hidden_states, permuted_probs, self.reversed_mapping_for_combine = permute(
            hidden_states,
            self.dispatched_routing_map,
            probs=self.dispatched_probs,
            num_out_tokens=self.tokens_per_expert.sum().item(),
            fused=self.permute_fusion,
        )
        if self.router_dtype == "fp64":
            permuted_probs = permuted_probs.to(torch.float64)
        return hidden_states, permuted_probs

    def get_restored_hidden_states_by_experts(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Restore the hidden states to their original ordering before expert processing
        """
        hidden_states = unpermute(
            hidden_states,
            self.reversed_mapping_for_combine,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.dispatched_routing_map,
            fused=self.permute_fusion,
        )
        return hidden_states


@dataclass
class MoEConfig:
    moe_enable_deepep: bool = True
    """Enable DeepEP for efficient token dispatching and combine in MoE models."""

    moe_permute_fusion: bool = False
    """Fuse token rearrangement ops during token dispatching."""

    moe_expert_capacity_factor: Optional[float] = None
    """moe_expert_capacity_factor (float): The capacity factor for each expert, None means no token
    will be dropped. The default is None."""

    moe_router_topk: int = 2
    """Number of experts to route to for each token."""

    moe_router_expert_pad_multiple: Optional[int] = None
    """Number of tokens to pad to a multiple of for each expert."""

    num_moe_experts: int = 64
    """Number of experts to use for MoE layer. When set, it replaces MLP with MoE layer. Set to None
    for no MoE."""

    moe_router_dtype: str = "fp32"
    """Data type for routing and expert output weighted averaging. Using fp32 or fp64 can
    improve stability especially when the number of experts is large (e.g. finegrained-moe).
    None means no changes for dtype."""


class MoEFlexTokenDispatcher:
    """
    Flex token dispatcher using DeepEP.
    """

    shared_comm_manager: _DeepepManager = (
        None  # shared by all instances of MoEFlexTokenDispatcher
    )

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: MoEConfig,
        ep_group: torch.distributed.ProcessGroup,
    ):
        """
        Initialize the Flex token dispatcher.

        Args:
            num_local_experts (int): Number of local experts on the current device.
            local_expert_indices (List[int]): Indices of local experts on the current device.
            config (MoEConfig): Configuration for the transformer model.
            group (torch.distributed.ProcessGroup): Process group for MoE operations.
        """
        self.config = config
        self.shared_experts = None

        self.group = ep_group
        self.ep_size = ep_group.size()
        # use model_comm_pgs.expt_tp_group as tensor parallel group in this module.
        # self.tp_group = tp_group
        # self.tp_group = tp_ep_group

        self.tp_size = 1  # TP is not used in Cosmos-R1
        # self.tp_rank = self.tp_group.rank()

        self.num_local_experts = num_local_experts
        self.local_expert_indices = local_expert_indices
        assert self.tp_size * self.ep_size > 1, (
            "Flex token dispatcher requires TPxEP > 1"
        )
        assert self.config.moe_enable_deepep, (
            "DeepEP is not enabled. Please set --moe-enable-deepep to use DeepEP backend."
        )
        if SHARING_DEEPEP_MANAGER:
            if MoEFlexTokenDispatcher.shared_comm_manager is None:
                MoEFlexTokenDispatcher.shared_comm_manager = _DeepepManager(
                    group=ep_group,
                    router_topk=self.tp_size * self.config.moe_router_topk,
                    permute_fusion=self.config.moe_permute_fusion,
                    capacity_factor=self.config.moe_expert_capacity_factor,
                    num_experts=self.tp_size * self.config.num_moe_experts,
                    num_local_experts=self.num_local_experts,
                    router_dtype=self.config.moe_router_dtype,
                    moe_router_expert_pad_multiple=self.config.moe_router_expert_pad_multiple,
                )
            self._comm_manager = MoEFlexTokenDispatcher.shared_comm_manager
        else:
            self._comm_manager = _DeepepManager(
                group=ep_group,
                router_topk=self.tp_size * self.config.moe_router_topk,
                permute_fusion=self.config.moe_permute_fusion,
                capacity_factor=self.config.moe_expert_capacity_factor,
                num_experts=self.tp_size * self.config.num_moe_experts,
                num_local_experts=self.num_local_experts,
                router_dtype=self.config.moe_router_dtype,
                moe_router_expert_pad_multiple=self.config.moe_router_expert_pad_multiple,
            )

    def _initialize_metadata(
        self, num_local_tokens: int, probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Initialize the routing map and probs to a unified format covering the TPxEP group.
        This design decouples the communication group from underlying model parallelism groups,
        such that the communication strategy of tokens can be agnostic of TP size and EP size.
        """
        world_size = self.tp_size * self.ep_size
        probs = (
            probs.reshape(num_local_tokens, self.ep_size, 1, self.num_local_experts)
            .expand(-1, -1, self.tp_size, -1)
            .reshape(num_local_tokens, world_size, self.num_local_experts)
        ).contiguous()
        return probs

    def dispatch_preprocess2(
        self,
        hidden_states: torch.Tensor,
        num_local_tokens: int,
        token_probs: torch.Tensor,
        token_indices: torch.Tensor,
    ):
        """
        Preprocesses the hidden states and routing information before dispatching tokens to experts.
        Args:
            hidden_states (torch.Tensor): Input hidden states to be processed, shape: [num_tokens, hidden_size].
            num_local_tokens (int): Number of tokens to be processed.
            token_probs (torch.Tensor): Routing probabilities for each token-expert pair, shape: [num_tokens, topk].
            token_indices (torch.Tensor): Indices of the selected experts for each token, shape: [num_tokens, topk].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - hidden_states: Reshaped hidden states, shape: [num_tokens, hidden_size].
                - token_probs: Token probabilities from the communication manager, shape: [num_tokens, topk].
        """
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        self._comm_manager.token_probs = token_probs
        self._comm_manager.token_indices = token_indices
        return hidden_states, self._comm_manager.token_probs

    def dispatch_preprocess(
        self, hidden_states: torch.Tensor, num_local_tokens: int, probs: torch.Tensor
    ):
        """
        Preprocesses the hidden states and routing information before dispatching tokens to experts.
        Args:
            hidden_states (torch.Tensor): Input hidden states to be processed
            num_local_tokens (int): Number of tokens to be processed
            probs (torch.Tensor): Routing probabilities for each token-expert pair

        Returns:
            Tuple containing:
            - torch.Tensor: Reshaped hidden states
            - torch.Tensor: Token probabilities from the communication manager
            - None: Placeholder for compatibility
        """
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        # Initialize metadata
        probs = self._initialize_metadata(
            num_local_tokens=num_local_tokens, probs=probs
        )

        self._comm_manager.setup_metadata(
            num_local_tokens=num_local_tokens, probs=probs
        )
        return hidden_states, self._comm_manager.token_probs

    def dispatch_all_to_all(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor = None,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ):
        """
        Performs all-to-all communication to dispatch tokens across expert parallel ranks.
        """
        return (
            self._comm_manager.dispatch(
                hidden_states, async_finish, allocate_on_comm_stream
            ),
            self._comm_manager.dispatched_probs,
        )

    def dispatch_postprocess(self, hidden_states: torch.Tensor):
        """
        Post-processes the dispatched hidden states after all-to-all communication.

        This method retrieves the permuted hidden states by experts, calculates the number of tokens
        per expert, and returns the processed data ready for expert processing.
        """
        global_input_tokens, permuted_probs = (
            self._comm_manager.get_permuted_hidden_states_by_experts(hidden_states)
        )
        tokens_per_expert = self._comm_manager.get_number_of_tokens_per_expert()
        return global_input_tokens, tokens_per_expert, permuted_probs

    def token_permutation(
        self, hidden_states: torch.Tensor, num_local_tokens: int, probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Permutes tokens according to probs and dispatches them to experts.

        This method implements the token permutation process in three steps:
        1. Preprocess the hidden states
        2. Perform all-to-all communication to dispatch tokens
        3. Post-process the dispatched tokens for expert processing
        """
        hidden_states, _ = self.dispatch_preprocess(
            hidden_states=hidden_states, num_local_tokens=num_local_tokens, probs=probs
        )
        hidden_states, _ = self.dispatch_all_to_all(
            hidden_states, async_finish=True, allocate_on_comm_stream=True
        )
        global_input_tokens, tokens_per_expert, permuted_probs = (
            self.dispatch_postprocess(hidden_states)
        )

        return global_input_tokens, tokens_per_expert, permuted_probs

    def token_permutation2(
        self,
        hidden_states: torch.Tensor,
        num_local_tokens: int,
        token_probs: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Permutes tokens according to probs and dispatches them to experts.

        This method implements the token permutation process in three steps:
        1. Preprocess the hidden states
        2. Perform all-to-all communication to dispatch tokens
        3. Post-process the dispatched tokens for expert processing

        Args:
            hidden_states (torch.Tensor): Input hidden states to be processed, shape: [num_tokens, hidden_size].
            num_local_tokens (int): Number of tokens to be processed.
            token_probs (torch.Tensor): Routing probabilities for each token-expert pair, shape: [num_tokens, topk].
            token_indices (torch.Tensor): Indices of the selected experts for each token, shape: [num_tokens, topk].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - global_input_tokens: Permuted hidden states, shape: [num_tokens, hidden_size].
                - tokens_per_expert: Number of tokens per expert, shape: [num_experts].
                - permuted_probs: Permuted probabilities, shape: [num_tokens, topk].
        """
        hidden_states, _ = self.dispatch_preprocess2(
            hidden_states=hidden_states,
            num_local_tokens=num_local_tokens,
            token_probs=token_probs,
            token_indices=token_indices,
        )
        hidden_states, _ = self.dispatch_all_to_all(
            hidden_states, async_finish=True, allocate_on_comm_stream=True
        )
        global_input_tokens, tokens_per_expert, permuted_probs = (
            self.dispatch_postprocess(hidden_states)
        )

        return global_input_tokens, tokens_per_expert, permuted_probs

    def combine_preprocess(self, hidden_states: torch.Tensor):
        """
        Pre-processes the hidden states before combining them after expert processing.

        This method restores the hidden states to their original ordering before expert processing
        by using the communication manager's restoration function.
        """
        hidden_states = self._comm_manager.get_restored_hidden_states_by_experts(
            hidden_states
        )
        return hidden_states

    def combine_all_to_all(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ):
        """
        Performs all-to-all communication to combine tokens after expert processing.
        """
        return self._comm_manager.combine(
            hidden_states, async_finish, allocate_on_comm_stream
        )

    def combine_postprocess(self, hidden_states: torch.Tensor):
        """
        Post-processes the combined hidden states after all-to-all communication.

        This method reshapes the combined hidden states to match the original input shape.
        """
        return hidden_states.view(self.hidden_shape)

    def token_unpermutation(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Reverses the token permutation process to restore the original token order.

        This method implements the token unpermutation process in three steps:
        1. Pre-process the hidden states to restore their original ordering
        2. Perform all-to-all communication to combine tokens
        3. Post-process the combined tokens to match the original input shape
        """
        hidden_states = self.combine_preprocess(hidden_states)
        hidden_states = self.combine_all_to_all(hidden_states, True, True)
        hidden_states = self.combine_postprocess(hidden_states)

        return hidden_states


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes):
        """Forward function."""
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        world_size = group.size()
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input

        input = input.contiguous()
        if output_split_sizes is None:
            # Equal split (all2all)
            output = torch.empty_like(input)
        else:
            # Unequal split (all2all-v)
            output = input.new_empty(
                size=[sum(output_split_sizes)] + list(input.size()[1:]),
                dtype=input.dtype,
                device=torch.cuda.current_device(),
            )
        torch.distributed.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        """Backward function."""
        return (
            None,
            _AllToAll.apply(
                ctx.group, *grad_output, ctx.input_split_sizes, ctx.output_split_sizes
            ),
            None,
            None,
        )


def all_to_all(group, input_, output_split_sizes_=None, input_split_sizes=None):
    assert group is not None, "group should not be None"
    return _AllToAll.apply(group, input_, output_split_sizes_, input_split_sizes)


class MoETokenDispatcher:
    """
    MoE Token Dispatcher
    """

    def __init__(
        self,
        config: MoEConfig,
        ep_group: torch.distributed.ProcessGroup,
    ) -> None:
        self.config = config
        self.shared_experts = None

        self.ep_group = ep_group
        self.ep_size = self.ep_group.size()
        self.ep_rank = torch.distributed.get_rank(self.ep_group)

    @abstractmethod
    def dispatch_preprocess(
        self, tokens: torch.Tensor, routing_map: torch.Tensor, probs: torch.Tensor
    ):
        """Prepares tokens for dispatch without inter-device communication.

        This method should handle all local computations like tensor rearrangement and
        metadata extraction before the main communication step.

        Note:
            Try to avoid any communication here to enable optimal computation-communication
            overlapping when enabling communication overlap, since communications in the
            same stream runs sequentially and may get exposed.

        Args:
            tokens (torch.Tensor): Input tokens.
            routing_map (torch.Tensor): Token to expert mapping tensor.
            probs (torch.Tensor): The routing probability tensor, [num_tokens, num_experts].

        Returns:
            A tuple of preprocessed tokens and probabilities.
        """
        raise NotImplementedError("dispatch_preprocess function not implemented.")

    @abstractmethod
    def token_dispatch(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Dispatches tokens to expert devices using communication.

        This method performs the main communication (e.g., All-to-All) to send
        tokens to the devices where their assigned experts reside.

        Args:
            hidden_states (torch.Tensor): Preprocessed hidden states to be dispatched.
            probs (torch.Tensor): Preprocessed probabilities for each token-expert pair.

        Returns:
            A tuple of dispatched tokens and probabilities.
        """
        raise NotImplementedError("token_dispatch function not implemented.")

    @abstractmethod
    def dispatch_postprocess(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Performs local processing after token dispatch communication.

        This method handles post-communication tasks like token reordering and
        preparing metadata for the expert forward pass.

        Note:
            Try to avoid any communication here to enable optimal computation-communication
            overlapping when enabling communication overlap, since communications in the
            same stream runs sequentially and may get exposed.

        Args:
            hidden_states (torch.Tensor): Dispatched hidden states.
            probs (torch.Tensor): Dispatched probabilities.

        Returns:
            A tuple containing the permuted tokens for experts, the number of
            tokens per expert, and the permuted probabilities.
        """
        raise NotImplementedError("dispatch_postprocess function not implemented.")

    @abstractmethod
    def combine_preprocess(self, hidden_states):
        """Prepares expert outputs for the combine step.

        This method performs local computations on expert outputs before the
        communication step for combining them.

        Note:
            Try to avoid any communication here to enable optimal computation-communication
            overlapping when enabling communication overlap, since communications in the
            same stream runs sequentially and may get exposed.

        Args:
            hidden_states (torch.Tensor): The output tensor from the experts.

        Returns:
            The preprocessed expert output.
        """
        raise NotImplementedError("combine_preprocess function not implemented.")

    @abstractmethod
    def token_combine(self, hidden_states):
        """Combines expert outputs across devices using communication.

        This method aggregates expert outputs from different devices via
        communication (e.g., All-to-All or Reduce-Scatter).

        Args:
            hidden_states (torch.Tensor): Preprocessed output from experts.

        Returns:
            The combined expert outputs.
        """
        raise NotImplementedError("token_combine function not implemented.")

    @abstractmethod
    def combine_postprocess(self, hidden_states):
        """Performs local processing after token combine.

        This method handles post-communication tasks like unpermuting and
        reshaping to restore the original tensor structure.

        Note:
            Try to avoid any communication here to enable optimal computation-communication
            overlapping when enabling communication overlap, since communications in the
            same stream runs sequentially and may get exposed.

        Args:
            hidden_states (torch.Tensor): Combined hidden states from token combination

        Returns:
            The final output tensor.
        """
        raise NotImplementedError("combine_postprocess function not implemented.")

    def set_shared_experts(self, shared_experts):
        """Set shared expert to the dispatcher."""
        self.shared_experts = shared_experts


class MoEAlltoAllTokenDispatcher(MoETokenDispatcher):
    """
    AlltoAll-based token dispatcher.

    The workflow of AlltoAll token dispatcher is as follows:
    (1) preprocess: calculate necessary metadata for communication and permute
    (2) dispatch process: permute tokens
    (3) token dispatch: A2A(EP)
    (4) dispatch postprocess: sort_chunk(if num_local_experts>1)
    (5) combine preprocess: sort_chunk(if num_local_experts>1)
    (6) token combine: A2A(EP)
    (7) combine postprocess: unpermute tokens
    """

    # DtoH copies are performed on this stream for overlapping with the main stream.
    cuda_dtoh_stream = None

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: MoEConfig,
        ep_group: torch.distributed.ProcessGroup,
    ) -> None:
        """
        Initialize the AlltoAll token dispatcher.

        Args:
            num_local_experts (int): Number of local experts on the current device.
            local_expert_indices (List[int]): Indices of local experts on the current device.
            config (MoEConfig): Configuration for the transformer model.
            ep_group (ProcessGroupCollection, optional): Process groups for MoE operations.
        """
        super().__init__(config=config, ep_group=ep_group)
        self.num_local_experts = num_local_experts
        assert config.num_moe_experts is not None
        self.num_experts = config.num_moe_experts
        assert self.num_local_experts > 0, "Expected at least one expert"
        self.local_expert_indices = local_expert_indices
        assert len(self.local_expert_indices) == self.num_local_experts, (
            "Invalid local expert indices"
        )
        for i in range(len(self.local_expert_indices) - 1):
            assert (
                self.local_expert_indices[i] == self.local_expert_indices[i + 1] - 1
            ), "local_expert_indices must be continuous"

        # [ep_size]. Represents the number of tokens sent by the current rank to other
        # EP ranks.
        self.input_splits = None
        # [ep_size]. Represents the number of tokens received by the current rank from
        # other EP ranks.
        self.output_splits = None
        self.permute_idx_device = (
            torch.device("cuda") if self.config.moe_permute_fusion else "cpu"
        )
        input_chunk_idxs = torch.arange(
            self.num_experts, device=self.permute_idx_device
        )
        # [num_local_experts, ep_size]. Sort the input chunks by local experts.
        self.sort_input_by_local_experts = input_chunk_idxs.reshape(
            -1, self.num_local_experts
        ).T.ravel()
        # [ep_size, num_local_experts]. Restore the output chunks by local experts.
        self.restore_output_by_local_experts = input_chunk_idxs.reshape(
            self.num_local_experts, -1
        ).T.ravel()

        # Token drop and padding.
        # Drop and pad the input to capacity.
        self.capacity = None
        self.shared_experts = None

    def set_shared_experts(self, shared_experts):
        """Set shared expert to the dispatcher."""
        super().set_shared_experts(shared_experts)

    def preprocess(self, routing_map: torch.Tensor) -> torch.Tensor:
        """
        Preprocesses the token routing map for All-to-All communication and token permutation.

        This method computes the number of tokens assigned to each expert based on the routing_map.
        It also initializes necessary data structures for All-to-All communication, such as input
        and output splits, and the mapping between global tokens and local experts. This method
        should not call any DtoH data copying due to performance consideration. The necessary DtoH
        copies are made on the `self.cuda_dtoh_stream` at `self.cuda_dtoh_point`.

        Args:
            routing_map (torch.Tensor): The mapping of tokens to experts.

        Returns:
            A tensor with the number of tokens for each local expert.
        """
        # [num_experts], number of tokens assigned to each expert from the current rank's input.
        num_local_tokens_per_expert = routing_map.sum(dim=0).long()
        self.num_out_tokens = routing_map.size(0) * self.config.moe_router_topk
        if self.ep_size > 1:
            # ===================================================
            # Calculate input_splits, output_splits for alltoall/allgather in variable size.
            # ===================================================
            # [ep_size]. Represents the number of tokens sent by the current rank to other
            # EP ranks.
            self.input_splits = (
                num_local_tokens_per_expert.reshape(
                    self.ep_size, self.num_local_experts
                )
                .sum(axis=1)
                .cpu()
                .tolist()
            )
            # Gather the global distribution of tokens across ranks.
            # num_global_tokens_per_expert represents the number of tokens sent to each
            # expert by all ranks.
            # [ep_size, num_experts]
            dim_size = list(num_local_tokens_per_expert.size())
            dim_size[0] = dim_size[0] * self.ep_size
            output = torch.empty(
                dim_size,
                dtype=num_local_tokens_per_expert.dtype,
                device=torch.cuda.current_device(),
            )
            torch.distributed.all_gather_into_tensor(
                output, num_local_tokens_per_expert.contiguous(), group=self.ep_group
            )
            num_global_tokens_per_expert = output.reshape(
                self.ep_size, self.num_experts
            )
            # [ep_size, num_experts] -> [ep_size, num_local_experts]
            num_global_tokens_per_local_expert = num_global_tokens_per_expert[
                :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
            ].contiguous()
            # [ep_size, num_local_experts] -> [ep_size]
            num_global_tokens_per_rank = (
                num_global_tokens_per_local_expert.sum(axis=1).cpu().tolist()
            )
            # [ep_size] -> [ep_size]
            # self.output_splits represents the number of tokens received by the current rank
            # from other rank.
            self.output_splits = num_global_tokens_per_rank
            # [ep_size, num_local_experts] -> [num_local_experts]
            num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=0)
        else:
            num_global_tokens_per_local_expert = num_local_tokens_per_expert.reshape(
                self.num_experts
            )
            num_tokens_per_local_expert = num_local_tokens_per_expert

            # A synchronization is needed before the returns
            # to get the `num_tokens_per_local_expert` CPU value.

        if self.num_local_experts > 1:
            # [ep_size, num_local_experts]. Represents the number of tokens sent
            # to each local expert by all ranks.
            self.num_global_tokens_per_local_expert = num_global_tokens_per_local_expert
        return num_tokens_per_local_expert

    def dispatch_preprocess(
        self,
        hidden_states: torch.Tensor,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
    ):
        """Prepares hidden states and probabilities for dispatch.

        This method reshapes the hidden states, computes communication metadata,
        and permutes the tokens and probabilities before the All-to-All communication.

        Args:
            hidden_states (torch.Tensor): Input token embeddings.
            routing_map (torch.Tensor): The mapping of tokens to experts.
            probs (torch.Tensor): Routing probabilities.

        Returns:
            A tuple of permuted hidden states and probabilities.
        """
        # Preprocess: Get the metadata for communication, permutation and computation operations.
        self.hidden_shape = hidden_states.shape
        self.probs = probs
        self.routing_map = routing_map
        assert probs.dim() == 2, "Expected 2D tensor for probs"
        assert routing_map.dim() == 2, "Expected 2D tensor for token2expert mask"
        assert routing_map.dtype == torch.bool, "Expected bool tensor for mask"
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        self.tokens_per_expert = self.preprocess(self.routing_map)

        self.hidden_shape_before_permute = hidden_states.shape
        (
            permutated_local_input_tokens,
            permuted_probs,
            self.reversed_local_input_permutation_mapping,
        ) = permute(
            hidden_states,
            self.routing_map,
            probs=probs,
            num_out_tokens=self.num_out_tokens,
            fused=self.config.moe_permute_fusion,
            drop_and_pad=False,
        )
        return permutated_local_input_tokens, permuted_probs

    def token_dispatch(self, permutated_local_input_tokens, permuted_probs):
        """
        Perform all-to-all communication for dispatching tokens.

        This method performs the all-to-all communication step to dispatch tokens across
        expert parallel ranks. It synchronizes metadata at the appropriate point before
        performing the communication.

        Args:
            permutated_local_input_tokens (torch.Tensor): Pre-permuted input tokens.
            permuted_probs (torch.Tensor): Pre-permuted probabilities.

        Returns:
            A tuple of tokens and probabilities after All-to-All.
        """
        # Perform expert parallel AlltoAll communication

        global_input_tokens = all_to_all(
            self.ep_group,
            permutated_local_input_tokens,
            self.output_splits,
            self.input_splits,
        )
        global_probs = all_to_all(
            self.ep_group, permuted_probs, self.output_splits, self.input_splits
        )

        return global_input_tokens, global_probs

    def dispatch_postprocess(self, global_input_tokens, global_probs):
        """Post-processes tokens after All-to-All communication.

        This involves an All-Gather in the tensor parallel dimension and sorting
        tokens by expert if there are multiple local experts.

        Args:
            global_input_tokens (torch.Tensor): Tokens after All-to-All.
            global_probs (torch.Tensor): Probabilities after All-to-All.

        Returns:
            A tuple of processed tokens, token counts per expert, and processed probabilities.
        """
        # Permutation 2: Sort tokens by local expert.
        if self.num_local_experts > 1:
            global_input_tokens, global_probs = sort_chunks_by_idxs(
                global_input_tokens,
                self.num_global_tokens_per_local_expert.ravel(),
                self.sort_input_by_local_experts,
                probs=global_probs,
                fused=self.config.moe_permute_fusion,
            )

        tokens_per_expert = self.tokens_per_expert
        return global_input_tokens, tokens_per_expert, global_probs

    def token_permutation2(
        self,
        hidden_states: torch.Tensor,
        num_local_tokens: int,
        token_probs: torch.Tensor,
        token_indices: torch.Tensor,
    ):
        """
        Using TE's permute here
            Turn token_indices: int tensor (num_token, topk) to routing_map bool tensor (num_token, num_experts)
                with routing_map[i,k] = True if this token is routed to expert k
            Trun token_probs: tensor (num_token, topk) to probs_full (num_token, num_expert)
        """
        routing_map = (
            torch.zeros(hidden_states.shape[0], self.num_experts)
            .int()
            .to(hidden_states.device)
            .scatter(1, token_indices, 1)
            .bool()
        )
        probs_full = (
            torch.zeros(
                hidden_states.shape[0], self.num_experts, dtype=token_probs.dtype
            )
            .to(hidden_states.device)
            .scatter_(dim=1, index=token_indices, src=token_probs)
        )

        permutated_local_input_tokens, permuted_probs = self.dispatch_preprocess(
            hidden_states=hidden_states,
            routing_map=routing_map,
            probs=probs_full,
        )
        dispatched_hidden_states, dispatched_probs = self.token_dispatch(
            permutated_local_input_tokens, permuted_probs
        )
        global_input_tokens, tokens_per_expert, global_probs = (
            self.dispatch_postprocess(dispatched_hidden_states, dispatched_probs)
        )
        return global_input_tokens, tokens_per_expert.cpu(), global_probs

    def combine_preprocess(self, hidden_states):
        """Prepares hidden states for token combination after expert computations.

        This may involve un-sorting tokens and a Reduce-Scatter in the tensor
        parallel dimension.
        """
        # Unpermutation 2: Unsort tokens by local expert.
        if self.num_local_experts > 1:
            hidden_states, _ = sort_chunks_by_idxs(
                hidden_states,
                self.num_global_tokens_per_local_expert.T.ravel(),
                self.restore_output_by_local_experts,
                fused=self.config.moe_permute_fusion,
            )

        return hidden_states

    def token_combine(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ):
        """Executes fused un-permutation and communication using DeepEP kernels.

        This method performs the inverse AlltoAll communication operation to collect expert
        outputs from their processing ranks and redistribute them back to the ranks that
        originally held the corresponding tokens. This completes the expert processing
        communication pattern and prepares tokens for final unpermutation.

        Args:
            hidden_states (torch.Tensor): Expert outputs ready for combination
            async_finish (bool): Whether to use asynchronous communication completion
            allocate_on_comm_stream (bool): Whether to allocate buffers on communication stream

        Returns:
            Tokens after the All-to-All communication for combining.
        """
        # Perform expert parallel AlltoAll communication
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        permutated_local_input_tokens = all_to_all(
            self.ep_group, hidden_states, self.input_splits, self.output_splits
        )
        return permutated_local_input_tokens

    def combine_postprocess(self, permutated_local_input_tokens):
        """Finalizes token reconstruction with un-permutation and reshaping.

        This method un-permutes the tokens back to their original order,
        reshapes the tensor to its original shape, and adds the shared
        expert output if enabled.

        Args:
            permutated_local_input_tokens (torch.Tensor): Permuted hidden states from token combine.

        Returns:
            The final MoE layer output reshaped to its original dimensions.
        """
        if self.shared_experts is not None:
            self.shared_experts.linear_fc2_forward(permutated_local_input_tokens)
            self.shared_experts.post_forward_comm()

        # Unpermutation 1: AlltoAll output to output
        output = unpermute(
            permutated_local_input_tokens,
            self.reversed_local_input_permutation_mapping,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.routing_map,
            fused=self.config.moe_permute_fusion,
            drop_and_pad=False,
        )

        # Reshape the output tensor
        output = output.view(self.hidden_shape)

        # Add shared experts output
        if self.shared_experts is not None:
            shared_expert_output = self.shared_experts.get_output()
            output += shared_expert_output
        return output

    def token_unpermutation(self, hidden_states):
        hidden_states = self.combine_preprocess(hidden_states)
        hidden_states = self.token_combine(hidden_states)
        hidden_states = self.combine_postprocess(hidden_states)
        return hidden_states

    def _maybe_update_cuda_sync_point(self, point: str):
        """
        Update the CUDA sync point if the priority of the new point is higher than the current
        sync point, which means the new point is reached earlier than the current sync point.
        """
        if (
            self.cuda_sync_point_priority[point]
            < self.cuda_sync_point_priority[self.cuda_sync_point]
        ):
            self.cuda_sync_point = point
