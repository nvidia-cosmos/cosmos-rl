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
from typing import Optional, List, Tuple, Union, Callable, Dict, Type, Any
from functools import cached_property
from cosmos_rl.utils.parallelism import ParallelDims, ParallelismConfig
from cosmos_rl.utils.logging import logger
from cosmos_rl.policy.config import Config as CosmosConfig
import cosmos_rl.utils.util as util
import torch
from transformers import AutoConfig
from cosmos_rl.dispatcher.data.packer import DataPacker
from vllm.model_executor.layers.linear import (
    RowParallelLinear,
    ColumnParallelLinear,
    QKVParallelLinear,
    MergedColumnParallelLinear,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

from torch.nn.parameter import Parameter


class BaseModel(torch.nn.Module, ABC):
    def __init__(self, hf_config: AutoConfig):
        super().__init__()
        self.weight_mapper = WeightMapper.get_weight_mapper(
            self.supported_model_types()[0]
        )(hf_config)

    def current_device(self):
        """
        Get the current device of the model
        """
        return next(self.parameters()).device

    @cached_property
    def sorted_hf_key_n_rank(self) -> List[Tuple[str, int]]:
        """
        Return sorted parameter tensor name and their rank of local view.
        """
        sorted_key_n_rank = []
        for k, v in self.named_parameters():
            k = self.weight_mapper.policy_map_local_key_to_hf_key(k)
            is_dist_tensor = isinstance(v, torch.distributed.tensor.DTensor)
            local_view = v.to_local() if is_dist_tensor else v
            sorted_key_n_rank.append((k, local_view.ndim))
        sorted_key_n_rank.sort(key=lambda x: x[0])
        return sorted_key_n_rank

    """
    Abstract methods
    """

    @staticmethod
    @abstractmethod
    def supported_model_types():
        raise NotImplementedError

    @property
    @abstractmethod
    def parallelize_fn(self):
        raise NotImplementedError

    @abstractmethod
    def apply_pipeline_split(self, pp_rank, pp_size):
        raise NotImplementedError

    def post_to_empty_hook(self, cosmos_config: CosmosConfig):
        """
        Hook to be called when the model is moved to CUDA device.
        This is used to re-initialize buffers like `inv_freq` for rotary embeddings.
        """
        raise NotImplementedError

    @abstractmethod
    def get_position_ids(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Method to get the position ids of the model.
        This function is declared due to that `Context Parallelism`
        requires the shuffle of both `input_ids` and `position_ids`.

        Args:
            **kwargs: Keyword arguments.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, int]:
                - Tensor of position ids
                - Tensor of input ids
                - Sequence dimension index of position ids.
        """
        raise NotImplementedError

    @abstractmethod
    def load_hf_weights(
        self,
        model_name_or_path: str,
        parallel_dims: ParallelDims,
        device: torch.device,
    ):
        """
        Load weights from a HuggingFace model.

        Args:
            model_name_or_path (str): The name or path of the model.
            parallel_dims (ParallelDims): The parallel dimensions.
            device (torch.device): The device to load the weights.
        """
        raise NotImplementedError

    @abstractmethod
    def separate_model_parts(self) -> List[torch.nn.Module]:
        """
        Model parts that should be trained in separate optimizers. (i.e. Multi-model training)
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        hf_config: AutoConfig,
        model_name_or_path: str,
        max_position_embeddings: Optional[int] = None,
    ) -> "BaseModel":
        raise NotImplementedError

    @cached_property
    def weight_sync_transforms(self) -> List[Tuple[str, Union[torch.Tensor, Callable]]]:
        """
        Get the local view of the tensors from the state dict.
        This method retrieves the state dict of the model, clears the weight names,
        and returns a list of tuples containing the destination name, and either a tensor or a callable returning a tensor.
        Returns:
            List[Tuple[str, Union[torch.Tensor, Callable]]]: A list of tuples containing the destination name,
            and either a tensor or a callable returning a tensor.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_nparams_and_flops(cls, seq_len: int) -> tuple[int, int]:
        """
        Get the number of parameters and flops of the model.
        Args:
            seq_len (int): The sequence length of the model.
        Returns:
            tuple[int, int]: The number of parameters and flops of the model.
        """
        raise NotImplementedError

    def check_cp_compatible(self, cp_size: int, tp_size: int):
        raise NotImplementedError(
            "This func should not be called in BaseModel instance."
        )


class ModelRegistry:
    _MODEL_REGISTRY: Dict[str, Type] = {}

    @classmethod
    def register_model(
        cls, model_cls: Type, data_packer_cls: Type, weight_mapper_cls: Type
    ):
        model_types = model_cls.supported_model_types()
        if isinstance(model_types, str):
            model_types = [model_types]
        for model_type in model_types:
            ModelRegistry._MODEL_REGISTRY[model_type] = model_cls
            WeightMapper.register_class(model_type, weight_mapper_cls)
            DataPacker.register(model_type, data_packer_cls)
            setattr(cls, "__cosmos_data_packer_cls", data_packer_cls)
            setattr(cls, "__cosmos_weight_mapper_cls", weight_mapper_cls)

    @classmethod
    def register(
        x,
        default_data_packer_cls,
        default_weight_mapper_cls,
        *,
        allow_override: bool = False,
    ):
        def decorator(cls: Type) -> Type:
            model_types = cls.supported_model_types()
            if isinstance(model_types, str):
                model_types = [model_types]

            for model_type in model_types:
                if (
                    not allow_override
                    and model_type in ModelRegistry._MODEL_REGISTRY
                    and ModelRegistry._MODEL_REGISTRY[model_type] != cls
                ):
                    raise ValueError(f"Model {model_type} is already registered.")
                ModelRegistry.register_model(
                    cls, default_data_packer_cls, default_weight_mapper_cls
                )
            return cls

        return decorator

    @classmethod
    def check_model_type_supported(cls, model_type: str) -> bool:
        return model_type in ModelRegistry._MODEL_REGISTRY

    @classmethod
    def build_model(cls, config: CosmosConfig):
        model_name_or_path = config.policy.model_name_or_path
        model = None
        hf_config = util.retry(AutoConfig.from_pretrained)(
            model_name_or_path, trust_remote_code=True
        )
        if hf_config.model_type not in ModelRegistry._MODEL_REGISTRY:
            raise ValueError(f"Model {hf_config.model_type} not supported.")
        model_cls = ModelRegistry._MODEL_REGISTRY[hf_config.model_type]

        with torch.device("meta"):
            with util.cosmos_default_dtype(
                util.str2torch_dtype(config.train.param_dtype)
            ):
                try:
                    model = model_cls.from_pretrained(
                        hf_config,
                        model_name_or_path,
                        max_position_embeddings=config.policy.model_max_length,
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to load model {model_name_or_path} with error: {e}"
                    )
                    raise e
        if model is None:
            raise ValueError(f"Model {model_name_or_path} not supported.")
        return model


class WeightMapper(ABC):
    _MODEL_WEIGHT_MAPPER_REGISTRY: Dict[str, Tuple[Type["WeightMapper"], int]] = {}

    def __init__(self, hf_config: AutoConfig):
        logger.info(f"WeightMapper: {type(self).__name__} is being initialized.")
        self.config = hf_config

    @torch.no_grad()
    def policy_maybe_decompose_weights_to_hf_naming(self, name, param):
        """
        Decompose the weights of the model parameters into fine-grained weights
        This is especially useful for models with non-symmetric parameter layout than the original HuggingFace one
        For example, MoE experts' weights are stacked in the 0th dimension,
        while they are stored in different keys in the original HuggingFace naming convention
        """
        yield name, param

    def policy_pre_P2R_gather_required_for_sync(self, name: str) -> bool:
        """
        For P->R weight sync, some weights need to be pre-collected before first `nccl_send/recv` instruction.
        To not be messed up with the following `nccl_send/recv` instructions,
        pre-collect those weights before first `nccl_send/recv` instruction.

        Args:
            name (str): The name of the tensor.
        Returns:
            bool: True if the tensor sync precollect is required, False otherwise.
        """
        return False

    @abstractmethod
    def rollout_prepare_recv(
        self, vllm_model: Any
    ) -> Tuple[Dict[str, torch.Tensor], List[Tuple[str, int]]]:
        """
        Rollout prepare recv list for P2R weight sync:
            - vllm_weight_inplace_view_map: Dict[str, torch.Tensor]: the map of vllm weight inplace view to be written by P2R weight sync
            - recv_key_n_rank_list: List[Tuple[str, int]]: the list of recv key and its tensor rank
        """
        pass

    def name_to_model_part_index(self, dest_name: str) -> int:
        return 0

    @abstractmethod
    def policy_map_local_key_to_hf_key(self, name: str) -> str:
        pass

    @abstractmethod
    def get_rollout_parallelism(self, replica_parallelism: ParallelismConfig):
        pass

    @abstractmethod
    def get_policy_parallelism(self, replica_parallelism: ParallelismConfig):
        pass

    def get_policy_parallelism_strategy(self):
        return []

    def get_rollout_parallelism_strategy(self):
        return []

    @classmethod
    def register_class(
        x,
        reg_key: Union[str, List[str]],
        default_weight_mapper_cls: Type["WeightMapper"],
        *,
        allow_override: bool = False,
    ):
        if isinstance(reg_key, str):
            reg_key = [reg_key]

        for model_type in reg_key:
            if (
                not allow_override
                and model_type in WeightMapper._MODEL_WEIGHT_MAPPER_REGISTRY
                and WeightMapper._MODEL_WEIGHT_MAPPER_REGISTRY[model_type]
                != default_weight_mapper_cls
            ):
                raise ValueError(
                    f"WeightMapper for '{model_type}' is already registered."
                )
            WeightMapper._MODEL_WEIGHT_MAPPER_REGISTRY[model_type] = (
                default_weight_mapper_cls
            )

    @classmethod
    def get_weight_mapper(cls, model_type: str) -> Type["WeightMapper"]:
        if model_type not in WeightMapper._MODEL_WEIGHT_MAPPER_REGISTRY:
            raise ValueError(f"ModelType '{model_type}' is not supported now.")

        return WeightMapper._MODEL_WEIGHT_MAPPER_REGISTRY[model_type]

    def parallelism_info_for_param(
        self,
        param_name: str,
    ):
        """
        Get the parallelism info for a specific parameter.
        This method returns a dictionary with parameter names as keys and their parallel dimensions as values.
        """
        if hasattr(self, "parallelism_info_for_params"):
            return self.parallelism_info_for_params[param_name]
        else:
            raise ValueError("No parallelism info found for the given parameter name.")

    def insert_to_parallelism_info(
        self,
        param_name: str,
        dims_map: Dict[str, int],
        parallelism: ParallelDims,
        name_to_hf: Callable,
        packed_modules_mapping: Dict[str, Any] = {},
    ):
        """
        Insert the parallelism info for the policy model parameters.
        This method updates the `policy_parallelism_info_for_params` dictionary with the parameter name,
        its dimensions map, tensor dimension to parallel map, and the pipeline rank.
        """
        tensor_dim_to_parallel_map = {}
        for k, v in dims_map.items():
            if v not in tensor_dim_to_parallel_map:
                tensor_dim_to_parallel_map[v] = []
            tensor_dim_to_parallel_map[v].append(k)
        p_rank = parallelism.pp_coord[0]
        for k, v in packed_modules_mapping.items():
            if k in param_name:
                for rename in v:
                    name = name_to_hf(param_name).replace(k, rename)
                    self.parallelism_info_for_params[name] = (
                        dims_map,
                        tensor_dim_to_parallel_map,
                        p_rank,
                    )
                return
        name = name_to_hf(param_name)
        self.parallelism_info_for_params[name] = (
            dims_map,
            tensor_dim_to_parallel_map,
            p_rank,
        )

    def parallelism_info_for_policy_params(
        self, model: BaseModel, parallelism: ParallelDims
    ):
        """
        Get the parallelism info for the policy model parameters.
        This method returns a dictionary with parameter names as keys and their parallel dimensions as values.
        """
        if hasattr(self, "parallelism_info_for_params"):
            assert hasattr(self, "is_policy") and not hasattr(self, "is_rollout"), (
                "parallelism_info_for_params already exists, "
                "but is_policy or is_rollout flag is not set correctly."
            )
            return self.parallelism_info_for_params
        self.parallelism_info_for_params = {}
        self.is_policy = True
        for name, param in model.named_parameters():
            is_dist_tensor = isinstance(param, torch.distributed.tensor.DTensor)
            if not is_dist_tensor:
                dims_map = {}
            else:
                dims_map = {}
                global_shape = tuple(param.shape)
                mesh = param.device_mesh
                placements = param.placements
                assert (
                    len(placements) == len(mesh.mesh_dim_names)
                ), f"Number of placements {placements} does not match number of mesh dimensions {mesh}."
                for dim, placement in zip(mesh.mesh_dim_names, placements):
                    if placement.is_shard():
                        dims_map[dim] = placement.dim
                    elif placement.is_replicate():
                        pass
                    else:
                        raise ValueError(f"Unsupported placement type: {placement}")
                chunk_meta_list = param.__create_chunk_list__()
                local = param.to_local()
                for meta in chunk_meta_list:
                    assert (
                        len(meta.offsets)
                        == len(meta.sizes)
                        == len(global_shape)
                        == len(tuple(local.shape))
                    ), f"Offsets {meta.offsets} and sizes {meta.sizes} must match global shape {global_shape} and local shape {tuple(local.shape)}."

            self.insert_to_parallelism_info(
                name, dims_map, parallelism, self.policy_map_local_key_to_hf_key
            )

    def parallelism_info_for_rollout_params(
        self, model: Any, parallelism: ParallelDims
    ):
        """
        Get the parallelism info for the rollout model parameters.
        This method returns a dictionary with parameter names as keys and their parallel dimensions as values.
        """
        if hasattr(self, "parallelism_info_for_params"):
            assert hasattr(self, "is_rollout") and not hasattr(self, "is_policy"), (
                "parallelism_info_for_params already exists, "
                "but is_rollout or is_policy flag is not set correctly."
            )
            return self.parallelism_info_for_params
        self.parallelism_info_for_params = {}
        self.is_rollout = True
        self.packed_modules_mapping = {}
        if hasattr(model, "packed_modules_mapping") and model.packed_modules_mapping:
            for k, v in model.packed_modules_mapping.items():
                if (
                    "qkv_proj" in k
                    and all(["_proj" in x for x in v])
                    and "qkv" not in model.packed_modules_mapping
                ):
                    # If the packed modules mapping does not contain "qkv", but contains "qkv_proj",
                    # we can assume that it is a QKVParallelLinear module.
                    self.packed_modules_mapping["qkv"] = [
                        x.replace("_proj", "") for x in v
                    ]
                self.packed_modules_mapping[k] = v

        for param_name, param in model.named_parameters():
            name_parts = param_name.split(".")
            part = model
            is_bias = False
            for part_name in name_parts:
                if hasattr(part, part_name):
                    if isinstance(getattr(part, part_name), Parameter):
                        if part_name == "bias":
                            is_bias = True
                        elif part_name == "weight":
                            is_bias = False
                        else:
                            raise ValueError(
                                f"Part {part_name} is not a Parameter. Skipping."
                            )
                        break
                    part = getattr(part, part_name)
                elif str.isdigit(part_name):
                    part = part[int(part_name)]
                else:
                    raise ValueError(f"Part {part_name} not found in {part}. Skipping.")
            dims_map = {}
            if isinstance(part, (QKVParallelLinear)):
                output_dim = getattr(param, "output_dim", None)
                assert (
                    output_dim is not None
                ), f"QKVParallelLinear {param_name} has no output_dim attribute."
                dims_map["tp"] = output_dim
                assert any(
                    [k in param_name for k in self.packed_modules_mapping.keys()]
                ), f"QKVParallelLinear {param_name} is not in packed_modules_mapping {self.packed_modules_mapping}."
            elif isinstance(part, (MergedColumnParallelLinear)):
                output_dim = getattr(param, "output_dim", None)
                assert (
                    output_dim is not None
                ), f"MergedColumnParallelLinear {param_name} has no output_dim attribute."
                dims_map["tp"] = output_dim
                assert any(
                    [k in param_name for k in self.packed_modules_mapping.keys()]
                ), f"MergedColumnParallelLinear {param_name} is not in packed_modules_mapping {self.packed_modules_mapping}."
            elif isinstance(part, (RowParallelLinear)):
                input_dim = getattr(param, "input_dim", None)
                if not is_bias:
                    assert (
                        input_dim is not None
                    ), f"RowParallelLinear {param_name} has no input_dim attribute."
                    dims_map["tp"] = input_dim
            elif isinstance(part, (ColumnParallelLinear)):
                output_dim = getattr(param, "output_dim", None)
                assert (
                    output_dim is not None
                ), f"ColumnParallelLinear {param_name} has no output_dim attribute."
                dims_map["tp"] = output_dim
            elif isinstance(part, VocabParallelEmbedding):
                output_dim = getattr(param, "output_dim", None)
                assert (
                    not is_bias
                ), f"VocabParallelEmbedding {param_name} should not have bias."
                assert (
                    output_dim is not None
                ), f"VocabParallelEmbedding {param_name} has no output_dim attribute."
                dims_map["tp"] = output_dim
            else:
                assert (
                    "Parallel" not in part.__class__.__name__
                ), f"Part {part.__class__.__name__} is not a parallel layer. Skipping."

            self.insert_to_parallelism_info(
                param_name,
                dims_map,
                parallelism,
                self._rollout_vllm_name_to_hf,
                self.packed_modules_mapping
                if hasattr(self, "packed_modules_mapping")
                else {},
            )
