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

from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.distributed._composable.replicate import replicate
from torch.distributed.tensor.parallel import parallelize_module
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    fully_shard,
    MixedPrecisionPolicy,
)

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.util import str2torch_dtype
from cosmos_rl.utils.parallelism import ParallelDims, pre_parallelize_sanity_check
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.model.hf_models.tp_plans import get_tp_plans


@pre_parallelize_sanity_check
def parallelize(
    model: nn.Module,
    parallel_dims: ParallelDims,
    config: CosmosConfig,
    pp_loss_fn: Optional[Callable] = None,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    world_mesh = parallel_dims.mesh
    _, pp_size = parallel_dims.pp_coord

    assert not parallel_dims.cp_enabled, (
        "Context parallelism is not supported for HFModel"
    )
    assert pp_size == 1, "Pipeline parallelism is not supported for HFModel"
    assert not config.train.compile, "Compile is not supported for HFModel"

    if parallel_dims.tp_enabled:
        apply_tp(
            model,
            world_mesh["tp"],
            enable_float8_tensorwise_tp=config.train.fp8.enable_fp8
            and config.train.fp8.quant_recipe == "tensorwise",
            enable_async_tp=config.train.async_tp_enabled,
        )

    # apply FSDP or HSDP
    if parallel_dims.dp_shard_enabled:
        apply_fsdp(
            model,
            parallel_dims=parallel_dims,
            param_dtype=str2torch_dtype(config.train.param_dtype),
            reduce_dtype=str2torch_dtype(config.train.fsdp_reduce_dtype),
            pp_enabled=False,
            cpu_offload=config.train.fsdp_offload,
            reshard_after_forward_policy=config.train.fsdp_reshard_after_forward,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if config.train.fsdp_offload:
            logger.info("Applied CPU Offloading to the model")
    elif parallel_dims.dp_replicate_enabled:
        if world_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        apply_ddp(
            model,
            world_mesh,
            enable_compile=False,
            enable_compiled_autograd=False,
        )

    return None, None


def apply_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    enable_float8_tensorwise_tp: bool,
    enable_async_tp: bool,
):
    """Apply tensor parallelism."""
    tp_plans = get_tp_plans(
        model, enable_float8_tensorwise_tp=enable_float8_tensorwise_tp
    )

    parallelize_module(
        module=model.model,
        device_mesh=tp_mesh,
        parallelize_plan=tp_plans,
    )

    if enable_async_tp:
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group

        torch._inductor.config._micro_pipeline_tp = True
        enable_symm_mem_for_group(tp_mesh.get_group().group_name)

    logger.info(
        f"Applied {'Float8 tensorwise ' if enable_float8_tensorwise_tp else ''}{'Async ' if enable_async_tp else ''}"
        "Tensor Parallelism to the model"
    )


def apply_fsdp(
    model: nn.Module,
    parallel_dims: ParallelDims,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
):
    """
    Apply data parallelism (via FSDP2) to the model.

    Args:
        model (nn.Module): The model to apply data parallelism to.
        parallel_dims (ParallelDims): device mesh to use for data parallelism/tp fused into dp_shard.
        param_dtype (torch.dtype): The data type to use for model parameters.
        reduce_dtype (torch.dtype): The data type to use for reduction operations.
        pp_enabled (bool): Whether pipeline parallelism is enabled.
        cpu_offload (bool, optional): Whether to offload model parameters to CPU. Defaults to False.
        reshard_after_forward_policy (str, optional): The policy to use for resharding after forward pass. Defaults to "default".
            Other options: "never", "always".
            - "default" applies default resharding behavior, implementing "smart defaults" for known optimal scenarios.
            - "always" will enable `reshard_after_forward` for all forward passes.
            - "never" will disable `reshard_after_forward` for all forward passes.

    """
    world_mesh = parallel_dims.mesh
    if parallel_dims.dp_replicate_enabled:
        dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
    else:
        dp_mesh_dim_names = ("dp_shard_cp",)
    dp_mesh = world_mesh[tuple(dp_mesh_dim_names)]

    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    fsdp_config_dp_shard_tp = None
    if parallel_dims.tp_enabled:
        dp_group_names = ["dp_shard"]
        # Flatten dp_shard × tp into one mesh dim: FSDP shards submodules that do *not* use TP
        # (GatedDeltaNet linear_attn, MoE experts) across both ranks — distinct from mesh "dp_cp_tp".
        mesh_dim_name_dp_shard_tp = "fsdp_dp_shard_tp"
        world_mesh[tuple(dp_group_names + ["tp"])]._flatten(
            mesh_dim_name=mesh_dim_name_dp_shard_tp
        )
        # For transformer_block (TP-sharded attention / rest): reuse top-level ``fsdp_config`` (``dp_mesh``).
        # HFModel asserts ``not cp_enabled``, so ``dp_shard_cp`` is only the ``dp_shard`` axis — same
        # process group as the old separate ``fsdp_dp_shard_only`` flatten; no second mesh needed.
        if parallel_dims.dp_replicate_enabled:
            mesh_dim_names_dp_shard_tp = (
                "dp_replicate",
                mesh_dim_name_dp_shard_tp,
            )
        else:
            mesh_dim_names_dp_shard_tp = (mesh_dim_name_dp_shard_tp,)

        fsdp_config_dp_shard_tp = {
            "mesh": world_mesh[tuple(mesh_dim_names_dp_shard_tp)],
            "mp_policy": mp_policy,
        }
        if cpu_offload:
            fsdp_config_dp_shard_tp["offload_policy"] = CPUOffloadPolicy()

    # 1. First shard the multi-modal projector
    # Since it could be a submodule of the vision model, we shard it before sharding the vision model to avoid redundant FSDP wrapping and potential FSDP assertion error.
    if model.multi_modal_projector is not None:
        fully_shard(
            model.multi_modal_projector,
            **fsdp_config,
            reshard_after_forward=True,
        )

    # 2. Shard the vision model
    if model.vision_model is not None:
        logger.info("Applying FSDP to the visual model")
        for layer_id, transformer_block in enumerate(model.vision_layers):
            if reshard_after_forward_policy == "always":
                reshard_after_forward = True
            elif reshard_after_forward_policy == "never":
                reshard_after_forward = False
            elif reshard_after_forward_policy == "default":
                reshard_after_forward = int(layer_id) < model.n_vision_layers - 1
            else:
                raise ValueError(
                    f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
                )
            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )

        fully_shard(
            model.vision_model,
            **fsdp_config,
            reshard_after_forward=True,
        )

    # 3. Shard the language model
    for layer_id, transformer_block in enumerate(model.lm_layers):
        if reshard_after_forward_policy == "always":
            reshard_after_forward = True
        elif reshard_after_forward_policy == "never":
            reshard_after_forward = False
        elif reshard_after_forward_policy == "default":
            reshard_after_forward = int(layer_id) < model.n_lm_layers - 1
        else:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )

        ignored_params = set()
        linear_attn = getattr(transformer_block, "linear_attn", None)
        mlp = getattr(transformer_block, "mlp", None)
        moe_mlp = (
            mlp
            if mlp is not None and getattr(mlp, "experts", None) is not None
            else None
        )

        if linear_attn is not None:
            # With TP: dp_shard×tp mesh; without TP: same as rest of LM (dp_shard_cp only).
            config = (
                fsdp_config_dp_shard_tp
                if fsdp_config_dp_shard_tp is not None
                else fsdp_config
            )
            fully_shard(
                linear_attn,
                **config,
                reshard_after_forward=reshard_after_forward,
            )
            ignored_params |= set(linear_attn.parameters())

        if moe_mlp is not None:
            # HF may leave gate / shared_expert_gate in fp32; align with param_dtype before
            # FSDP (ReplicatedLinear-style: not managed by MixedPrecisionPolicy).
            for name in ("gate", "shared_expert_gate"):
                mod = getattr(moe_mlp, name, None)
                if mod is not None:
                    mod.to(dtype=param_dtype)
            # Match vLLM ReplicatedLinear: gate + shared_expert_gate are full replicas on every
            # rank (no TP slice, no FSDP shard on dp). Only list them in ignored_params so
            # transformer_block's fully_shard leaves them as unsharded parameters.
            # experts + shared_expert use fsdp_config_dp_shard_tp (FSDP over flattened dp_shard×tp).
            config = (
                fsdp_config_dp_shard_tp
                if fsdp_config_dp_shard_tp is not None
                else fsdp_config
            )
            fully_shard(
                moe_mlp.experts,
                **config,
                reshard_after_forward=reshard_after_forward,
            )
            fully_shard(
                moe_mlp.shared_expert,
                **config,
                reshard_after_forward=reshard_after_forward,
            )
            ignored_params |= set(moe_mlp.parameters())
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
            ignored_params=ignored_params,
        )

    if model.embed_tokens is not None:
        logger.info("Applying FSDP to the language model embed_tokens")
        fully_shard(model.embed_tokens, **fsdp_config, reshard_after_forward=True)
    fully_shard(model.language_model, **fsdp_config, reshard_after_forward=True)
    if model.model is not model.language_model:
        # model.model might be the same with model.language_model which is already shard above,
        # so we only shard it when not the same to avoid redundant sharding assertion error.
        fully_shard(model.model, **fsdp_config, reshard_after_forward=True)
    # No need to shard the whole model wrapper since the whole model only has model.model
    # The whole model is already included in the above shards


def apply_ddp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    enable_compile: bool,
    enable_compiled_autograd: bool,
):
    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)

    logger.info("Applied DDP to the model")
