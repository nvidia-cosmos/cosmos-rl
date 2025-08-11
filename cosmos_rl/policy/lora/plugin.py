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

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from cosmos_rl.policy.config import LoraConfig


class WeightWrapper(nn.Module):
    def __init__(self, weight: nn.Parameter):
        super().__init__()
        self.weight = weight


class LoraInjectedLinear(nn.Linear):
    """
    nn.Linear with optional LoRA adapters:
      y = x W^T + b + scale * ( (dropout(x) A^T) B^T )
      where ΔW = B @ A and scale = lora_alpha / r
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        use_rslora: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(in_features, out_features, bias=bias, **factory_kwargs)

        self.r = int(r)
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout_p = float(lora_dropout)
        self.merged = False

        assert self.r > 0, "LoRA rank must be greater than 0"
        # LoRA parameters
        self.lora_A = WeightWrapper(
            nn.Parameter(torch.empty(self.r, in_features, **factory_kwargs))
        )
        self.lora_B = WeightWrapper(
            nn.Parameter(torch.empty(out_features, self.r, **factory_kwargs))
        )
        # Init as in the LoRA paper: A ~ N(0, 0.02), B = 0 so initial ΔW=0
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B.weight)
        self.lora_dropout = (
            nn.Dropout(self.lora_dropout_p)
            if self.lora_dropout_p > 0.0
            else nn.Identity()
        )
        if use_rslora:
            self.scaling = self.lora_alpha / math.sqrt(self.r)
        else:
            self.scaling = self.lora_alpha / self.r

    @torch.no_grad()
    def reinitialize_lora_params(self):
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B.weight)

    @classmethod
    def from_linear(
        cls,
        base: nn.Linear,
        r: int,
        lora_alpha: float,
        lora_dropout: float,
        use_rslora: bool = False,
    ) -> "LoraInjectedLinear":
        # Create same-shaped Linear and copy weights/bias
        new = cls(
            base.in_features,
            base.out_features,
            bias=base.bias is not None,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            device=base.weight.device,
            dtype=base.weight.dtype,
            use_rslora=use_rslora,
        )
        with torch.no_grad():
            new.weight.copy_(base.weight)
            if base.bias is not None:
                new.bias.copy_(base.bias)
        return new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.r > 0 and not self.merged:
            # (x A^T) B^T = x @ A^T @ B^T
            after_A = self.lora_dropout(x) @ self.lora_A.weight.t()  # [*, r]
            lora_out = after_A @ self.lora_B.weight.t()  # [*, out]
            out = out + self.scaling * lora_out
        return out

    @torch.no_grad()
    def merge_adapters_(self) -> None:
        if self.r == 0 or self.merged:
            return
        delta_w = self.lora_B.weight @ self.lora_A.weight  # [out, in]
        self.weight.add_(self.scaling * delta_w)
        self.merged = True

    @torch.no_grad()
    def unmerge_adapters_(self) -> None:
        if self.r == 0 or not self.merged:
            return
        delta_w = self.lora_B.weight @ self.lora_A.weight
        self.weight.sub_(self.scaling * delta_w)
        self.merged = False

    def lora_parameters(self) -> Iterable[nn.Parameter]:
        if self.r == 0:
            return []
        return [self.lora_A.weight, self.lora_B.weight]


def _name_matches(name: str, needles: Iterable[str]) -> bool:
    for n in needles:
        if n and n in name:
            return True
    return False


def inject_lora_adapters(
    model: nn.Module, config: LoraConfig
) -> Tuple[nn.Module, List[str]]:
    """
    Replace matching nn.Linear modules with LoraInjectedLinear.
    Returns (model, replaced_module_names).
    """
    if config.target_modules is None:
        raise ValueError(
            "LoraConfig.target_modules must be set (list of substrings to match)."
        )

    replaced: List[str] = []

    for module_name, module in list(model.named_modules()):
        # Only consider leaves that are nn.Linear
        # We need the parent to set the attribute
        parent: Optional[nn.Module] = _get_parent_by_qualified_name(model, module_name)
        if parent is None:
            continue

        child_name = module_name.split(".")[-1]
        if isinstance(module, nn.Linear) and _name_matches(
            module_name, config.target_modules
        ):
            lora_linear = LoraInjectedLinear.from_linear(
                base=module,
                r=config.r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
            )
            setattr(parent, child_name, lora_linear)
            replaced.append(module_name)

    if not replaced:
        raise RuntimeError(
            "inject_lora_adapters found no matching nn.Linear modules. "
            f"target_modules={config.target_modules}"
        )
    return model, replaced


def _get_parent_by_qualified_name(
    root: nn.Module, qualified_name: str
) -> Optional[nn.Module]:
    """
    For a qualified child name like 'transformer.h.0.attn.q_proj',
    return the parent module object.
    """
    if not qualified_name:
        return None
    parts = qualified_name.split(".")
    if len(parts) == 1:
        return root
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p, None)
        if parent is None:
            return None
    return parent


def mark_only_lora_as_trainable(model: nn.Module, config: LoraConfig) -> None:
    """
    Freeze all non-LoRA parameters; enable grads only for LoRA parameters.
    """
    for p in model.parameters():
        p.requires_grad = False

    # Enable grads only for the modules in config.modules_to_save
    if config.modules_to_save is not None:
        for module_name, module in model.named_modules():
            if _name_matches(module_name, config.modules_to_save):
                for p in module.parameters():
                    p.requires_grad = True

    # Enable grads only for the LoRA parameters
    for m in model.modules():
        if isinstance(m, LoraInjectedLinear) and m.r > 0:
            m.lora_A.requires_grad_(True)
            m.lora_B.requires_grad_(True)


def reinitialize_lora_params(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, LoraInjectedLinear):
            m.reinitialize_lora_params()


@torch.no_grad()
def merge_lora_weights_(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, LoraInjectedLinear):
            m.merge_adapters_()


@torch.no_grad()
def unmerge_lora_weights_(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, LoraInjectedLinear):
            m.unmerge_adapters_()


def lora_state_dict(model: nn.Module, prefix: str = "lora") -> dict:
    """
    Return only LoRA parameters in a flat dict:
      f"{prefix}.{module_qualified_name}.lora_A" / ".lora_B"
    """
    out = {}
    for name, module in model.named_modules():
        if isinstance(module, LoraInjectedLinear) and module.r > 0:
            out[f"{prefix}.{name}.lora_A.weight"] = (
                module.lora_A.weight.detach().clone()
            )
            out[f"{prefix}.{name}.lora_B.weight"] = (
                module.lora_B.weight.detach().clone()
            )
    return out


@torch.no_grad()
def load_lora_state_dict(
    model: nn.Module, state: dict, prefix: str = "lora", strict: bool = True
) -> None:
    """
    Load only LoRA params from a state dict produced by lora_state_dict().
    Accepts keys like: f"{prefix}.{qualified.module.name}.lora_A" / ".lora_B"
    """
    missing = []

    for key, tensor in state.items():
        if not key.startswith(prefix + "."):
            continue

        # Strip "prefix."
        tail = key[len(prefix) + 1 :]  # "<qual.name>.lora_A"
        try:
            qual_name, leaf = tail.rsplit(".", 1)  # ("qual.name", "lora_A" or "lora_B")
        except ValueError:
            if strict:
                missing.append(key)
            continue

        module = _get_module_by_qualified_name(model, qual_name)
        if not isinstance(module, WeightWrapper) or not hasattr(module, leaf):
            if strict:
                missing.append(key)
            continue

        param = getattr(module, leaf)
        param.copy_(tensor.to(dtype=param.dtype, device=param.device))

    if strict and missing:
        raise KeyError(f"Missing LoRA destinations for keys: {missing}")


def _get_module_by_qualified_name(
    root: nn.Module, qualified_name: str
) -> Optional[nn.Module]:
    m = root
    for p in qualified_name.split("."):
        m = getattr(m, p, None)
        if m is None:
            return None
    return m
