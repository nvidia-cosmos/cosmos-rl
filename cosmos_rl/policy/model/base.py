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
        self,
        vllm_model: Any,
        quantization: bool = False,
        promotion_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[
        Dict[str, torch.Tensor], List[Tuple[str, torch.Size]], Dict[str, torch.Tensor]
    ]:
        """
        Rollout prepare recv list for P2R weight sync:
            - vllm_weight_inplace_view_map: Dict[str, torch.Tensor]: the map of vllm weight inplace view to be written by P2R weight sync
            - recv_key_n_rank_list: List[Tuple[str, int]]: the list of recv key and its tensor rank
            - vllm_full_weight_map: Dict[str, torch.Tensor]: the map of vllm full weight that needed for fp8 quantization, if fp8 available in rollout.
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

    @abstractmethod
    def get_policy_parallelism_strategy(self):
        pass

    @abstractmethod
    def get_rollout_parallelism_strategy(self):
        pass

    @abstractmethod
    def quantized_weight_partial_keys(self):
        pass

    @abstractmethod
    def get_unsplited_weight_name(self, weight_key: str) -> str:
        pass

    @abstractmethod
    def is_fp8_quantized_weight(self, weight_key: str) -> bool:
        pass

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
