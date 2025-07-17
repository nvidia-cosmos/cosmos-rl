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
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.utils.logging import logger
from cosmos_rl.policy.config import Config as CosmosConfig
import cosmos_rl.utils.util as util
from cosmos_rl.utils.constant import COSMOS_HF_MODEL_TYPES
import torch
from transformers import AutoConfig
from cosmos_rl.dispatcher.data.packer import DataPacker
import collections
from functools import partial


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
    def weight_sync_transforms(
        self,
    ) -> List[Tuple[str, Union[torch.Tensor, Callable]]]:
        from cosmos_rl.utils.parallelism_map import DimSliceInfo, ParallelTopoMapper

        # 1. get all parameters, but not buffers
        named_parameters = {name: param for name, param in self.named_parameters()}
        keys = list(named_parameters.keys())
        keys = sorted(keys, key=lambda x: x[0])
        transforms = collections.OrderedDict()
        for k in keys:
            v = named_parameters[k]
            is_dist_tensor = isinstance(v, torch.distributed.tensor.DTensor)
            local_view = v.to_local() if is_dist_tensor else v
            transforms[
                self.weight_mapper.policy_map_local_key_to_hf_key(
                    util.clear_weight_name(k)
                )
            ] = local_view

        # 2. do 1->n decomposition on weights like qkv_proj.weight -> q.weight, k.weight, v.weight
        for name, param in self.named_parameters():
            is_dist_tensor = isinstance(param, torch.distributed.tensor.DTensor)
            dims_rank_info = {}
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
                assert (
                    len(chunk_meta_list) == 1
                ), f"Expected only one chunk meta, but got {len(chunk_meta_list)} for {name}."
                meta = chunk_meta_list[0]
                assert (
                    len(meta.offsets)
                    == len(meta.sizes)
                    == len(global_shape)
                    == len(tuple(local.shape))
                ), f"Offsets {meta.offsets} and sizes {meta.sizes} must match global shape {global_shape} and local shape {tuple(local.shape)}."

                for idx, g_size in enumerate(global_shape):
                    offset = int(meta.offsets[idx])
                    total_size = int(g_size)
                    length = int(meta.sizes[idx])
                    if total_size == length:
                        assert (
                            offset == 0
                        ), f"Expected rank 0 for full size dimension {idx}, but got {offset}."
                    else:
                        dims_rank_info[idx] = DimSliceInfo(
                            offset=offset,
                            total_size=total_size,
                            length=length,
                        ).__dict__
                decomposed_key_and_slices = (
                    self.weight_mapper.policy_decompose_param_1_to_n_for_sync(
                        self.weight_mapper.policy_map_local_key_to_hf_key(name)
                    )
                )
                if decomposed_key_and_slices:
                    for part_name, part_slice in decomposed_key_and_slices:
                        splitted_dim_rank_info = {}
                        part_in_local = {}
                        part_slice = {
                            len(global_shape) + k if k < 0 else k: v
                            for k, v in part_slice.items()
                        }
                        all_dims = part_slice.keys() | dims_rank_info.keys()
                        for dim in all_dims:
                            if dim not in part_slice:
                                dim_slice = DimSliceInfo(
                                    offset=0,
                                    total_size=1,
                                )
                            else:
                                dim_slice = DimSliceInfo.from_dict(part_slice[dim])
                            if dim not in dims_rank_info:
                                assert (
                                    len(global_shape) > dim
                                ), f"Dimension {dim} is out of bounds for global shape {global_shape}."
                                local_part = DimSliceInfo(offset=0, total_size=1)
                            else:
                                local_part = DimSliceInfo.from_dict(dims_rank_info[dim])
                            slice_in_splited, overlap_in_local = (
                                ParallelTopoMapper.tensor_overlap_info_at_dim(
                                    {dim: dim_slice}, {dim: local_part}, dim
                                )
                            )
                            if slice_in_splited is None:
                                splitted_dim_rank_info = None
                                break

                            splitted_dim_rank_info[dim] = slice_in_splited.__dict__
                            part_in_local[dim] = overlap_in_local
                        if splitted_dim_rank_info is not None:

                            def slice_tensor_with_part(
                                local: torch.Tensor,
                                part_in_local: Dict[int, DimSliceInfo],
                            ) -> torch.Tensor:
                                """
                                Slice the local tensor with the part in local information.
                                :param local: The local tensor to be sliced.
                                :param part_in_local: The part in local information for slicing.
                                :return: The sliced tensor.
                                """
                                return local.cosmos_slice(part_in_local)

                            self.weight_mapper.set_transform_func_from_local_param_for_sync(
                                self.weight_mapper.policy_map_local_key_to_hf_key(
                                    part_name
                                ),
                                partial(
                                    slice_tensor_with_part,
                                    part_in_local=part_in_local,
                                ),
                            )

        weight_sync_transforms = []
        for name, _ in transforms.items():
            decomposed_key_and_ranks: List[Tuple[str, int]] = (
                self.weight_mapper.policy_decompose_param_1_to_n_for_sync(name)
            )

            if decomposed_key_and_ranks:
                # The current parameter is decomposed into multiple parameters, so we need to transform each of them.
                # (This does not happen for most cases, i.e. `qkv_proj.weight` to be decomposed into `q.weight`, `k.weight`, and `v.weight`)
                for decomposed_name, _ in decomposed_key_and_ranks:
                    # There are three cases:
                    # 1. The transformation logic of the decomposed parameter is already in the `weight_sync_transforms_per_model`,
                    #    so we can directly use it.
                    # 2. The transformation logic of the decomposed parameter is specified in weight mapper for 1 to n decomposition,
                    #    so we can use it.
                    # 3. The decomposed parameter does not reside in the current rank, skip it.
                    if decomposed_name in transforms:
                        weight_sync_transforms.append(
                            (
                                decomposed_name,
                                transforms[decomposed_name],
                            )
                        )
                    elif (
                        self.weight_mapper.get_transform_func_from_local_param_for_sync(
                            decomposed_name
                        )
                        is not None
                    ):
                        transform = self.weight_mapper.get_transform_func_from_local_param_for_sync(
                            decomposed_name
                        )
                        direct_view = transforms[name]
                        if isinstance(direct_view, torch.Tensor):
                            weight_sync_transforms.append(
                                (decomposed_name, transform(direct_view))
                            )
                        else:
                            assert isinstance(direct_view, Callable)

                            def wrapper(transform, direct_view):
                                return transform(direct_view())

                            weight_sync_transforms.append(
                                (
                                    decomposed_name,
                                    partial(wrapper, transform, direct_view),
                                )
                            )
                    else:
                        # If no transform function is set, means the current parameter is not transformed and synchronized at this rank.
                        pass
            else:
                weight_sync_transforms.append((name, transforms[name]))
        return weight_sync_transforms

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
        cls, model_cls: Type, weight_mapper_cls: Type, data_packer_cls: Type = None
    ):
        model_types = model_cls.supported_model_types()
        if isinstance(model_types, str):
            model_types = [model_types]
        for model_type in model_types:
            ModelRegistry._MODEL_REGISTRY[model_type] = model_cls
            WeightMapper.register_class(model_type, weight_mapper_cls)
            if data_packer_cls is not None:
                DataPacker.register(model_type, data_packer_cls)

    @classmethod
    def register(
        x,
        default_weight_mapper_cls,
        *,
        allow_override: bool = False,
        default_data_packer_cls=None,
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
                    cls,
                    default_weight_mapper_cls,
                    data_packer_cls=default_data_packer_cls,
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
        model_type = hf_config.model_type
        is_supported_model_type = model_type in ModelRegistry._MODEL_REGISTRY
        if not is_supported_model_type:
            logger.info(
                f"Model type {hf_config.model_type} not registered, using {COSMOS_HF_MODEL_TYPES} instead."
            )
            model_type = COSMOS_HF_MODEL_TYPES

        model_cls = ModelRegistry._MODEL_REGISTRY[model_type]

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

    @abstractmethod
    def rollout_prepare_recv(
        self,
        vllm_model: Any,
    ) -> Tuple[Dict[str, torch.Tensor], List[List[Tuple[str, int]]]]:
        """
        Rollout prepare recv list for P2R weight sync:
            - vllm_weight_inplace_view_map: Dict[str, torch.Tensor]: the map of vllm weight inplace view to be written by P2R weight sync
            - recv_key_n_rank_list: List[List[Tuple[str, int]]]: the list of grouped recv key and its tensor rank
        """
        pass

    def name_to_model_part_index(self, dest_name: str) -> int:
        return 0

    @abstractmethod
    def policy_map_local_key_to_hf_key(self, name: str) -> str:
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

    def set_transform_func_from_local_param_for_sync(
        self, name: str, transform: Callable
    ):
        """
        Set the mapping of a parameter to be synced to a transform function to get the sent view of the parameter.
        The function is Callable(local_param: torch.Tensor) -> torch.Tensor
        """
        if not hasattr(self, "policy_map_param_to_transform_func_for_sync"):
            self.policy_map_param_to_transform_func_for_sync = {}
        self.policy_map_param_to_transform_func_for_sync[name] = transform

    def get_transform_func_from_local_param_for_sync(
        self, name: str
    ) -> Optional[Callable]:
        """
        Get the transform function for a parameter to be synced.
        This function returns the transform function that is used to get the sent view of the parameter if specified,
        The function is Callable(local_param: torch.Tensor) -> torch.Tensor
        otherwise returns None.
        """
        if not hasattr(self, "policy_map_param_to_transform_func_for_sync"):
            return None
        return self.policy_map_param_to_transform_func_for_sync.get(name, None)

    @classmethod
    def get_weight_mapper(cls, model_type: str) -> Type["WeightMapper"]:
        if model_type not in WeightMapper._MODEL_WEIGHT_MAPPER_REGISTRY:
            raise ValueError(f"ModelType '{model_type}' is not supported now.")

        return WeightMapper._MODEL_WEIGHT_MAPPER_REGISTRY[model_type]

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

    @cached_property
    def packed_modules_mapping(self):
        """
        Return the packed modules mapping for the model.
        This method defines a mapping of packed modules to their corresponding components.
        This is used to handle packed modules like QKVParallelLinear and MergedColumnParallelLinear.
        """
        # This mapping is used to handle packed modules like QKVParallelLinear and MergedColumnParallelLinear
        # where multiple components are packed into a single parameter.
        # The keys are the names of the packed modules, and the values are lists of component
        # The following mapping is general for most cases.
        return {
            "qkv": [
                "q",
                "k",
                "v",
            ],
            "gate_up_proj": [
                "gate_proj",
                "up_proj",
            ],
            "qkv_proj": [
                "q_proj",
                "k_proj",
                "v_proj",
            ],
        }

    def policy_decompose_param_1_to_n_for_sync(self, name):
        """
        Map a parameter of the policy model to set of transformed parameters that need to be synchronized.
        This method returns a list containing tuples of the new parameter name and the corresponding new tensor transformed from the original tensor of the given name.
        Each tuple element includes a transformed tensor and its corresponding slice strategy to derive from the original tensor.
        """
        return []
