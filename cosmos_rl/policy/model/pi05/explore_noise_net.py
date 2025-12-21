# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
ExploreNoiseNet: Learnable exploration noise network for flow_noise method.
Ported from RLinf: rlinf/models/embodiment/modules/explore_noise_net.py
"""

from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn


activation_dict = nn.ModuleDict(
    {
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "mish": nn.Mish(),
        "identity": nn.Identity(),
        "softplus": nn.Softplus(),
        "silu": nn.SiLU(),
    }
)


class ExploreNoiseNet(nn.Module):
    """
    Neural network to generate learnable exploration noise,
    conditioned on time embeddings and/or state embeddings.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: List[int],
        activation_type: str,
        noise_logvar_range: List[float],  # [min_std, max_std]
        noise_scheduler_type: str,
    ):
        super().__init__()
        self.mlp_logvar = MLP(
            [in_dim] + hidden_dims + [out_dim],
            activation_type=activation_type,
            out_activation_type="identity",
        )
        self.noise_scheduler_type = noise_scheduler_type
        self.set_noise_range(noise_logvar_range)

    def set_noise_range(self, noise_logvar_range: List[float]):
        self.noise_logvar_range = noise_logvar_range
        noise_logvar_min = self.noise_logvar_range[0]
        noise_logvar_max = self.noise_logvar_range[1]
        self.register_buffer(
            "logvar_min",
            torch.log(torch.tensor(noise_logvar_min**2, dtype=torch.float32)).unsqueeze(0),
        )
        self.register_buffer(
            "logvar_max",
            torch.log(torch.tensor(noise_logvar_max**2, dtype=torch.float32)).unsqueeze(0),
        )

    def forward(self, noise_feature: torch.Tensor) -> torch.Tensor:
        if "const" in self.noise_scheduler_type:
            # Use the lowest noise level when using constant noise schedulers.
            noise_std = torch.exp(0.5 * self.logvar_min)
        else:
            # Use learnable noise level.
            noise_logvar = self.mlp_logvar(noise_feature)
            noise_std = self.post_process(noise_logvar)
        return noise_std

    def post_process(self, noise_logvar: torch.Tensor) -> torch.Tensor:
        """
        Post-process raw logvar output to bounded std.
        
        Input: torch.Tensor([B, Ta, Da])
        Output: torch.Tensor([B, Ta, Da])
        """
        noise_logvar = torch.tanh(noise_logvar)
        noise_logvar = (
            self.logvar_min
            + (self.logvar_max - self.logvar_min) * (noise_logvar + 1) / 2.0
        )
        noise_std = torch.exp(0.5 * noise_logvar)
        return noise_std


class MLP(nn.Module):
    """Simple MLP with configurable activation and layer normalization."""

    def __init__(
        self,
        dim_list: List[int],
        append_dim: int = 0,
        append_layers: List[int] = None,
        activation_type: str = "tanh",
        out_activation_type: str = "identity",
        use_layernorm: bool = False,
        use_layernorm_final: bool = False,
        dropout: float = 0,
        use_drop_final: bool = False,
        out_bias_init=None,
    ):
        super().__init__()
        self.append_layers = append_layers if append_layers is not None else []

        self.moduleList = nn.ModuleList()
        num_layer = len(dim_list) - 1
        for idx in range(num_layer):
            i_dim = dim_list[idx]
            o_dim = dim_list[idx + 1]
            if append_dim > 0 and idx in self.append_layers:
                i_dim += append_dim
            linear_layer = nn.Linear(i_dim, o_dim)

            layers = [("linear_1", linear_layer)]
            if use_layernorm and (idx < num_layer - 1 or use_layernorm_final):
                layers.append(("norm_1", nn.LayerNorm(o_dim)))
            if dropout > 0 and (idx < num_layer - 1 or use_drop_final):
                layers.append(("dropout_1", nn.Dropout(dropout)))

            act = (
                activation_dict[activation_type.lower()]
                if idx != num_layer - 1
                else activation_dict[out_activation_type.lower()]
            )
            layers.append(("act_1", act))

            module = nn.Sequential(OrderedDict(layers))
            self.moduleList.append(module)
        # Initialize the bias of the final linear layer if specified
        if out_bias_init is not None:
            final_linear = self.moduleList[-1][
                0
            ]  # Linear layer is first in the last Sequential # type: ignore
            nn.init.constant_(final_linear.bias, out_bias_init)
    def forward(self, x: torch.Tensor, append: torch.Tensor = None) -> torch.Tensor:
        for layer_ind, m in enumerate(self.moduleList):
            if append is not None and layer_ind in self.append_layers:
                x = torch.cat((x, append), dim=-1)
            x = m(x)
        return x

