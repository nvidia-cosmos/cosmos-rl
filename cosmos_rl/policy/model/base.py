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
from typing import Optional, List, Tuple, Union, Callable, Dict, Type
import torch
from functools import cached_property
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config import Config as CosmosConfig
from transformers import AutoConfig
from cosmos_rl.utils.logging import logger
import cosmos_rl.utils.util as util


class BaseModel(torch.nn.Module, ABC):
    def __init__(self, hf_config: AutoConfig):
        super().__init__()
        from cosmos_rl.rollout.weight_mapper import WeightMapper

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

    _MODEL_REGISTRY: Dict[str, Type] = {}

    @classmethod
    def register(cls, *, allow_override: bool = False):
        def decorator(cls: Type) -> Type:
            model_types = cls.supported_model_types()
            if isinstance(model_types, str):
                model_types = [model_types]

            for model_type in model_types:
                if not allow_override and model_type in BaseModel._MODEL_REGISTRY:
                    raise ValueError(f"Model {model_type} is already registered.")
                BaseModel._MODEL_REGISTRY[model_type] = cls
            return cls

        return decorator

    @classmethod
    def build_model(cls, config: CosmosConfig):
        model_name_or_path = config.policy.model_name_or_path
        model = None
        hf_config = util.retry(AutoConfig.from_pretrained)(
            model_name_or_path, trust_remote_code=True
        )
        if hf_config.model_type not in BaseModel._MODEL_REGISTRY:
            raise ValueError(f"Model {hf_config.model_type} not supported.")
        model_cls = BaseModel._MODEL_REGISTRY[hf_config.model_type]

        with torch.device("meta"):
            with util.cosmos_default_dtype(config.train.param_torch_dtype):
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
