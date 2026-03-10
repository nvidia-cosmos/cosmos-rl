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

"""Utilities for orthonormal optimizers (Muon, NorMuon, Dion, Dion2): param grouping and mesh extraction."""

from __future__ import annotations

import inspect
import math
from typing import Any, Optional

import torch.nn as nn

from cosmos_rl.utils.logging import logger

# Orthonormal optimizer names.
_ORTHONORMAL_OPTIMIZER_NAMES = frozenset({"Muon", "Dion", "Dion2", "NorMuon"})


def is_orthonormal_optimizer(name: str) -> bool:
    """Return True if the given optimizer name is an orthonormal optimizer.

    True for name in {"Muon", "Dion", "Dion2", "NorMuon"}.
    """
    return name in _ORTHONORMAL_OPTIMIZER_NAMES


def separate_param_groups_for_orthonormal_optim(
    model: nn.Module,
    base_lr: float,
    scalar_opt: str,
    weight_decay: float,
    scalar_betas: Optional[tuple[float, float]] = None,
    scalar_eps: Optional[float] = None,
    scalar_lr: Optional[float] = None,
    embed_lr: Optional[float] = None,
    lm_head_lr: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Separate model parameters into groups for orthonormal optimizers (Muon, NorMuon, Dion, Dion2).

    Groups:
        - Group 0: matrix params (ndim == 2), no "algorithm" key.
        - Group 1: vector params (bias, etc.), algorithm=scalar_opt, lr, weight_decay, betas, eps.
        - Group 2: embed params, algorithm=scalar_opt, embed_lr, weight_decay=0.
        - Group 3 (optional): lm_head params, algorithm=scalar_opt, lm_head_lr or base_lr/sqrt(d_in), weight_decay=0.
    """
    matrix_params: list[nn.Parameter] = []
    vector_params: list[nn.Parameter] = []
    embed_params: list[nn.Parameter] = []
    lm_head_params: list[nn.Parameter] = []

    name_to_module = dict(model.named_modules())

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        module = None
        try:
            module_name = name.rsplit(".", 1)[0]
            module = name_to_module.get(module_name, None)
        except Exception:
            module = None

        if isinstance(module, nn.Embedding):
            embed_params.append(param)
            continue

        if "lm_head" in name:
            lm_head_params.append(param)
            continue

        if param.ndim == 2:
            matrix_params.append(param)
        else:
            vector_params.append(param)

    scalar_kwargs: dict[str, Any] = {}
    if scalar_betas is not None and len(scalar_betas) >= 2:
        scalar_kwargs["beta1"] = scalar_betas[0]
        scalar_kwargs["beta2"] = scalar_betas[1]
    if scalar_eps is not None:
        scalar_kwargs["epsilon"] = scalar_eps

    effective_scalar_lr = scalar_lr if scalar_lr is not None else base_lr
    effective_embed_lr = embed_lr if embed_lr is not None else effective_scalar_lr

    param_groups: list[dict[str, Any]] = [
        {"params": matrix_params},
        {
            "params": vector_params,
            "algorithm": scalar_opt,
            "lr": effective_scalar_lr,
            "weight_decay": weight_decay,
            **scalar_kwargs,
        },
        {
            "params": embed_params,
            "algorithm": scalar_opt,
            "lr": effective_embed_lr,
            "weight_decay": 0.0,
            **scalar_kwargs,
        },
    ]

    if lm_head_params:
        if lm_head_lr is not None:
            effective_lm_head_lr = lm_head_lr
        else:
            first = lm_head_params[0]
            d_in = first.shape[-1] if first.ndim >= 2 else max(1, first.numel())
            effective_lm_head_lr = base_lr / math.sqrt(float(d_in))
        param_groups.append(
            {
                "params": lm_head_params,
                "algorithm": scalar_opt,
                "lr": effective_lm_head_lr,
                "weight_decay": 0.0,
                **scalar_kwargs,
            }
        )

    return param_groups


def get_orthonormal_optimizer_mesh(mesh_or_parallel_dims: Any) -> Any:
    """Return the 1D mesh to pass to orthonormal optimizer for distributed training.

    - None -> None.
    - If already 1D mesh (ndim == 1) -> return as is.
    - Cosmos RL: if mesh has submesh "dp_shard_cp", return mesh[("dp_shard_cp",)].
    - Automodel-style 2D: try mesh[("dp_replicate", "dp_shard_cp")]["dp_shard_cp"]; on failure return original.
    """
    if mesh_or_parallel_dims is None:
        return None

    mesh = mesh_or_parallel_dims
    # If caller passed ParallelDims, use its built mesh.
    if hasattr(mesh_or_parallel_dims, "mesh"):
        mesh = mesh_or_parallel_dims.mesh

    if not hasattr(mesh, "ndim"):
        return mesh

    if getattr(mesh, "ndim", None) == 1:
        return mesh

    # Cosmos RL: 1D submesh named "dp_shard_cp" (flattened dp_shard × cp).
    try:
        key = ("dp_shard_cp",)
        if hasattr(mesh, "__getitem__"):
            sub = mesh[key]
            if getattr(sub, "ndim", None) == 1:
                return sub
    except (KeyError, TypeError, AttributeError) as e:
        logger.debug("get_orthonormal_optimizer_mesh: no dp_shard_cp submesh: %s", e)

    # Automodel-style 2D: ("dp_replicate", "dp_shard_cp") -> ["dp_shard_cp"].
    try:
        dp_2d = mesh[("dp_replicate", "dp_shard_cp")]
        submesh = dp_2d["dp_shard_cp"]
        if getattr(submesh, "ndim", None) == 1:
            return submesh
    except (KeyError, TypeError, AttributeError, RuntimeError) as e:
        logger.debug(
            "get_orthonormal_optimizer_mesh: Automodel-style lookup failed: %s", e
        )

    return mesh


def build_orthonormal_optimizer(
    optimizer_name: str,
    model: nn.Module,
    config: Any,
    distributed_mesh: Any = None,
) -> Any:
    """Build a single orthonormal optimizer (Muon, NorMuon, Dion, Dion2) with param groups."""
    from cosmos_rl.policy.trainer.optm import orthonormal_optimizers

    if getattr(orthonormal_optimizers, "_import_error", None) is not None:
        raise RuntimeError(
            "Failed to import dion. Please install dion to use orthonormal optimizers."
        ) from orthonormal_optimizers._import_error

    name_to_cls = {
        "Muon": orthonormal_optimizers.Muon,
        "NorMuon": orthonormal_optimizers.NorMuon,
        "Dion": orthonormal_optimizers.Dion,
        "Dion2": orthonormal_optimizers.Dion2,
    }
    if optimizer_name not in name_to_cls:
        raise ValueError(
            f"Unknown orthonormal optimizer: {optimizer_name}. Use one of Muon, NorMuon, Dion, or Dion2."
        )
    opt_cls = name_to_cls[optimizer_name]

    train = getattr(config, "train", config)
    base_lr = getattr(train, "optm_lr", 1e-4)
    if isinstance(base_lr, (list, tuple)):
        base_lr = float(base_lr[0]) if base_lr else 1e-4
    base_lr = float(base_lr)
    weight_decay = float(getattr(train, "optm_weight_decay", 0.01))
    _betas = getattr(train, "optm_betas", (0.9, 0.999))
    try:
        betas = (
            (float(_betas[0]), float(_betas[1])) if len(_betas) >= 2 else (0.9, 0.999)
        )
    except (TypeError, IndexError):
        betas = (0.9, 0.999)
    scalar_opt = getattr(train, "optm_scalar_opt", "adamw")
    _scalar_betas = getattr(train, "optm_scalar_betas", None)
    if _scalar_betas is not None:
        try:
            scalar_betas = (
                (float(_scalar_betas[0]), float(_scalar_betas[1]))
                if len(_scalar_betas) >= 2
                else betas
            )
        except (TypeError, IndexError):
            scalar_betas = betas
    else:
        scalar_betas = betas
    scalar_eps = getattr(train, "optm_scalar_eps", None) or getattr(
        train, "epsilon", 1e-8
    )
    scalar_lr = getattr(train, "optm_scalar_lr", None)
    embed_lr = getattr(train, "optm_embed_lr", None)
    lm_head_lr = getattr(train, "optm_lm_head_lr", None)

    param_groups = separate_param_groups_for_orthonormal_optim(
        model,
        base_lr=base_lr,
        scalar_opt=scalar_opt,
        weight_decay=weight_decay,
        scalar_betas=scalar_betas,
        scalar_eps=scalar_eps,
        scalar_lr=scalar_lr,
        embed_lr=embed_lr,
        lm_head_lr=lm_head_lr,
    )

    mesh = get_orthonormal_optimizer_mesh(distributed_mesh)

    kwargs: dict[str, Any] = {
        "lr": base_lr,
        "weight_decay": weight_decay,
        "betas": tuple(betas)[:2],
        "epsilon": getattr(train, "epsilon", 1e-8),
    }

    def _opt(val: Any, default: Any) -> Any:
        """Use default when config value is None so orthonormal optimizers never receive None."""
        return val if val is not None else default

    init_params = set(inspect.signature(opt_cls.__init__).parameters.keys())
    if mesh is not None and "distributed_mesh" in init_params:
        kwargs["distributed_mesh"] = mesh
    if optimizer_name == "Muon":
        kwargs.setdefault("mu", _opt(getattr(train, "optm_mu", None), 0.95))
        kwargs.setdefault(
            "adjust_lr", _opt(getattr(train, "optm_adjust_lr", None), "spectral_norm")
        )
        for key, attr in (
            ("nesterov", "optm_nesterov"),
            ("flatten", "optm_flatten"),
            ("cautious_wd", "optm_cautious_wd"),
            ("use_triton", "optm_use_triton"),
        ):
            if key in init_params:
                val = getattr(train, attr, None)
                if val is not None:
                    kwargs[key] = val
    elif optimizer_name == "NorMuon":
        kwargs.setdefault("mu", _opt(getattr(train, "optm_mu", None), 0.95))
        kwargs.setdefault(
            "muon_beta2", _opt(getattr(train, "optm_muon_beta2", None), 0.95)
        )
        kwargs.setdefault(
            "adjust_lr", _opt(getattr(train, "optm_adjust_lr", None), "rms_norm")
        )
        for key, attr in (
            ("nesterov", "optm_nesterov"),
            ("flatten", "optm_flatten"),
            ("cautious_wd", "optm_cautious_wd"),
            ("use_triton", "optm_use_triton"),
        ):
            if key in init_params:
                val = getattr(train, attr, None)
                if val is not None:
                    kwargs[key] = val
    elif optimizer_name == "Dion":
        kwargs.setdefault("mu", _opt(getattr(train, "optm_mu", None), 0.95))
        kwargs.setdefault("rank", _opt(getattr(train, "optm_rank", None), 768))
        for key, attr in (
            ("rank_fraction", "optm_rank_fraction"),
            ("rank_multiple_of", "optm_rank_multiple_of"),
            ("power_iters", "optm_power_iters"),
            ("qr_method", "optm_qr_method"),
            ("rcqr_oversample", "optm_rcqr_oversample"),
        ):
            if key in init_params:
                val = getattr(train, attr, None)
                if val is not None:
                    kwargs[key] = val
    elif optimizer_name == "Dion2":
        kwargs.setdefault("fraction", _opt(getattr(train, "optm_fraction", None), 0.25))
        kwargs.setdefault("ef_decay", _opt(getattr(train, "optm_ef_decay", None), 0.95))
        kwargs.setdefault(
            "adjust_lr", _opt(getattr(train, "optm_adjust_lr", None), "spectral_norm")
        )
        for key, attr in (
            ("flatten", "optm_flatten"),
            ("use_triton", "optm_use_triton"),
            ("verbose", "optm_verbose"),
        ):
            if key in init_params:
                val = getattr(train, attr, None)
                if val is not None:
                    kwargs[key] = val

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in init_params}
    return opt_cls(param_groups, **filtered_kwargs)
