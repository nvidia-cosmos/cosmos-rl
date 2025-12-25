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
from typing import List, Callable, Dict, Tuple, Type
from cosmos_rl.dispatcher.data.schema import RLPayload

from cosmos_rl.rollout.schema import RolloutResult

from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.model import WeightMapper
from cosmos_rl.utils.parallelism import ParallelDims
import torch
from cosmos_rl.dispatcher.data.packer.base import BaseDataPacker
from cosmos_rl.dispatcher.data.data_fetcher import DataFetcherBase


class RolloutBase(ABC):
    def __init__(
        self,
        config: CosmosConfig,
        parallel_dims: ParallelDims = None,
        device: torch.device = None,
        **kwargs,
    ):
        """
        Initialize the RolloutBase class.
        Args:
            config: The configuration for the whole project.
            parallel_dims: The parallel dimensions for the rollout engine.
            device: The device on which the rollout engine will run.
        """
        self.config = config
        self.parallel_dims = parallel_dims
        self.device = device
        self._engine_initialized = False
        self._model_param_map = None  # key: compatible name, value: param

        self.post_init_hook(**kwargs)

    @abstractmethod
    def post_init_hook(self, **kwargs):
        """
        Post initialization hook for the rollout, which will be called at the end of __init__.
        """
        raise NotImplementedError("post_init_hook is not implemented yet.")

    @abstractmethod
    def rollout_generation(
        self,
        payloads: List[RLPayload] | List[Dict[str, torch.Tensor]],
        stream: torch.cuda.Stream,
        data_packer: BaseDataPacker,
        data_fetcher: DataFetcherBase,
        is_validation: bool = False,
        *args,
        **kwargs,
    ) -> List[RolloutResult] | List[Dict[str, torch.Tensor]]:
        """
        Generate sequences given the payloads.
        Args:
            payloads: The list of payloads for generation or the list of dictionaries of tensors as input in tensor native mode.
            stream: The CUDA stream for generation.
            data_packer: The data packer for packing the data.
            data_fetcher: The data fetcher for fetching the data, can access dataset and validation dataset from it.
            is_validation: Whether the rollout is for validation.

        Returns:
            A list of RolloutResult or a list of dictionaries of tensors as output in tensor native mode.
        """
        raise NotImplementedError("rollout_generation is not implemented yet.")

    @abstractmethod
    def init_engine(self, quantization: str, seed: int, load_format: str, **kwargs):
        """
        Initialize the engine for rollout.
        Args:
            quantization: The quantization type of the underlying model, one of "fp8", "mxfp4".
            seed: The random seed for initialization.
            load_format: The format to load the model, one of "dummy", "auto".
        """
        self._engine_initialized = True  # Set the engine initialized flag to True
        raise NotImplementedError("init_engine is not implemented yet.")

    @abstractmethod
    def get_underlying_model(self) -> torch.nn.Module:
        """
        Get the underlying model
        Returns:
            model: The underlying model instance.
        """
        raise NotImplementedError("get_underlying_model is not implemented yet.")

    def set_underlying_model(self, model: torch.nn.Module):
        """
        Set the underlying model
        Only used in where model instance is shared between rollout and policy.
        For instance sharing colocated rollout and policy case.
        Args:
            model: The underlying model instance to set.
        """
        raise NotImplementedError("set_underlying_model is not implemented yet.")

    def post_init_engine_hook(
        self,
        consume_command_hook: Callable,
        report_rollouts_hook: Callable,
        validation_flag: bool,
        **kwargs,
    ):
        """
        Post initialization hook for the engine, which will be called after the engine is initialized.
        Args:
            consume_command_hook: The hook function to consume the command from the controller.
            report_rollouts_hook: The hook function to report the rollouts to the controller.
            validation_flag: Whether the rollout is for validation.
        """
        pass

    def pre_get_params_for_sync_hook(
        self,
        quantization_type: str,
        weight_mapper: WeightMapper,
        parallel_dims: ParallelDims,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Pre get sync param hook for the engine, which will be called before getting the sync params.
        Args:
            quantization_type: The quantization type of the underlying model, one of "fp8", "mxfp4".
            weight_mapper: The weight mapper of the underlying model.
            parallel_dims: The parallel dimensions of the underlying model.
        Returns:
            hp_weight_map: The original high-precision weight map from name to original tensor of the underlying model for the quantized parameters.
            quantized_weight_map: The quantized weight map from name to quantized tensor of the underlying model for the quantized parameters.
        """
        # For quantization, we need to keep the quantized weight map and the high-precision weight map.
        # This is related with the quantization strategy of the underlying rollout engine type.
        # Also, related operations to the underlying model before weight sync are performed in this hook.
        # Defaultly, we just return the empty high-precision weight map and an empty quantized weight map.
        hp_weight_map, quantized_weight_map = {}, {}
        return hp_weight_map, quantized_weight_map

    def post_get_params_for_sync_hook(
        self,
        quantization_type: str,
        weight_mapper: WeightMapper,
        weight_view_map: Dict[str, torch.Tensor],
        quantized_weight_map: Dict[str, torch.Tensor],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Post get sync param hook for the engine, which will be called after getting the sync params.
        Args:
            quantization_type: The quantization type of the underlying model.
            weight_mapper: The weight mapper of the underlying model.
            weight_view_map: The weight view map of the underlying model used for weight sync.
            quantized_weight_map: The quantized weight map from name to tensor of the underlying model for the quantized parameters.
        Returns:
            weight_view_map: The updated weight view map of the underlying model used for weight sync.
        """
        # For quantization, we update the weight view map after getting the sync params.
        # Defaultly, we just return the weight view map without any update.
        # For the quantization strategy of the underlying rollout engine type, we need to update the weight view map.
        # For example, for vLLM, we need to update the weight view map using the quantized weight map information.
        return weight_view_map

    def model_param_map(self, weight_mapper: WeightMapper) -> Dict[str, torch.Tensor]:
        """
        All the parameters of the rollout model:
            - All the parameters of the model.
            - All the scales of quantized weights.
        """
        if not self._engine_initialized:
            raise RuntimeError(
                "[Rollout] Engine is not initialized, please call init_engine first."
            )

        if self._model_param_map:
            return self._model_param_map
        model = self.get_underlying_model()
        param_map = {}
        for name, param in model.state_dict().items():
            compatible_name = weight_mapper.rollout_map_local_key_to_hf_key(name)
            param_map[compatible_name] = param

        quantized_tensors = self.get_quantized_tensors(weight_mapper)
        param_map.update(quantized_tensors)

        self._model_param_map = param_map
        return self._model_param_map

    def is_engine_initialized(self):
        return self._engine_initialized

    def get_quantized_tensors(
        self, weight_mapper: WeightMapper
    ) -> Dict[str, torch.Tensor]:
        """
        Get the quantized tensors of the rollout model.
        Defaultly, we return an empty dictionary.
        Args:
            weight_mapper: The weight mapper of the underlying model.
        Returns:
            A dictionary of quantized tensors from name to tensor of the underlying model for the quantized parameters.
        """
        if not self._engine_initialized:
            raise RuntimeError(
                "[Rollout] Engine is not initialized, please call init_engine first."
            )
        quantized_tensors = {}
        return quantized_tensors


class RolloutRegistry:
    _ROLLOUT_REGISTRY: Dict[str, Type] = {}

    @classmethod
    def register_rollout_backend(cls, rollout_cls: Type, rollout_type: str):
        RolloutRegistry._ROLLOUT_REGISTRY[rollout_type] = rollout_cls

    @classmethod
    def register(
        x,
        rollout_type: str,
        *,
        allow_override: bool = False,
    ):
        def decorator(cls: Type) -> Type:
            assert issubclass(
                cls, RolloutBase
            ), "Registered rollout must be a subclass of RolloutBase."
            assert isinstance(rollout_type, str), "rollout_type must be a string."
            if (
                not allow_override
                and rollout_type in RolloutRegistry._ROLLOUT_REGISTRY
                and RolloutRegistry._ROLLOUT_REGISTRY[rollout_type] != cls
            ):
                raise ValueError(f"Rollout {rollout_type} is already registered.")
            RolloutRegistry.register_rollout_backend(
                cls,
                rollout_type,
            )
            return cls

        return decorator

    @classmethod
    def check_rollout_type_supported(cls, rollout_type: str) -> bool:
        return rollout_type in RolloutRegistry._ROLLOUT_REGISTRY

    @classmethod
    def get_rollout_cls(cls, rollout_type: str) -> Type:
        if rollout_type not in RolloutRegistry._ROLLOUT_REGISTRY:
            raise ValueError(f"Rollout {rollout_type} is not supported.")
        return RolloutRegistry._ROLLOUT_REGISTRY[rollout_type]
