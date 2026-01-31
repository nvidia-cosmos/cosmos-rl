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
from cosmos_rl.utils.diffusers_utils import diffusers_config_fn

from cosmos_rl.dispatcher.data.packer import BaseDataPacker
import collections
from functools import partial
from typing import Mapping
from accelerate import init_on_device
from contextlib import nullcontext
from cosmos_rl.policy.lora.plugin import LoraInjectedLinear
from cosmos_rl.utils.dim_slice_info import (
    DimSliceInfo,
    extract_infomation_from_dtensor,
    tensor_overlap_info_at_dim,
)


class BaseModel(torch.nn.Module, ABC):
    _gradient_checkpointing_enabled = False

    def __init__(self, hf_config: Optional[AutoConfig] = None):
        super().__init__()
        if hf_config is not None:
            self.weight_mapper = WeightMapper.get_weight_mapper(
                self.supported_model_types()[0]
            )(hf_config)

    def current_device(self):
        """
        Get the current device of the model
        """
        return next(self.parameters()).device

    def set_gradient_checkpointing_enabled(self, enabled: bool):
        """
        Set the gradient checkpointing enabled flag.
        This is used to enable or disable the gradient checkpointing for the model.
        """
        self._gradient_checkpointing_enabled = enabled
        for module in self.modules():
            if isinstance(module, torch.nn.Module):
                if not hasattr(module, "_gradient_checkpointing_enabled"):
                    setattr(module, "_gradient_checkpointing_enabled", enabled)

    def post_transform_of_local_view(
        self, local_view: torch.Tensor, name: str
    ) -> torch.Tensor:
        """
        Post-transform the local view of the tensor. In some cases, we need to transform the local view of the tensor before sending it to the rollout model.
        """
        return local_view

    @cached_property
    def trainable_params(self) -> List[str]:
        """
        Get the list of trainable parameters.
        This method returns a list of parameter names that are marked as trainable (i.e., `requires_grad` is True).
        Maybe customized modification of this function is needed for some later special models in order to get the correct trainable parameters used for weight synchronization.
        """
        trainable_params = []
        # Get all parameters.
        named_parameters = {name: param for name, param in self.named_parameters()}
        for k, v in named_parameters.items():
            # Clear and get the correct format of the param names.
            name = self.weight_mapper.policy_map_local_key_to_hf_key(
                util.clear_weight_name(k)
            )
            is_trainable = v.requires_grad
            decomposed_key_and_ranks: List[Tuple[str, int]] = (
                self.weight_mapper.policy_decompose_param_1_to_n_for_sync(name)
            )
            if decomposed_key_and_ranks:
                # The current parameter is decomposed into multiple parameters, so we need to record each of them if trainable.
                # (This does not happen for most cases, i.e. `qkv_proj.weight` to be decomposed into `q.weight`, `k.weight`, and `v.weight`)
                for decomposed_name, _ in decomposed_key_and_ranks:
                    if is_trainable:
                        trainable_params.append(decomposed_name)
            else:
                if is_trainable:
                    trainable_params.append(name)
        # Handle the lora case.
        for k, m in self.named_modules():
            if isinstance(m, LoraInjectedLinear):
                # Clear and get the correct format of the param names.
                name = self.weight_mapper.policy_map_local_key_to_hf_key(
                    util.clear_weight_name(k + ".weight")
                )
                decomposed_key_and_ranks: List[Tuple[str, int]] = (
                    self.weight_mapper.policy_decompose_param_1_to_n_for_sync(name)
                )
                if decomposed_key_and_ranks:
                    # The current parameter is decomposed into multiple parameters, so we need to record each of them if trainable.
                    # (This does not happen for most cases, i.e. `qkv_proj.weight` to be decomposed into `q.weight`, `k.weight`, and `v.weight`)
                    for decomposed_name, _ in decomposed_key_and_ranks:
                        trainable_params.append(decomposed_name)
                else:
                    trainable_params.append(name)
        return trainable_params

    def gen_local_view_transforms(self) -> Dict[str, Union[torch.Tensor, Callable]]:
        """
        Generate the local view or transform function for a P2R weight sync inst.
        """
        # 1. get all parameters, but not buffers
        named_parameters = {name: param for name, param in self.named_parameters()}
        keys = list(named_parameters.keys())
        keys = sorted(keys, key=lambda x: x[0])
        transforms = collections.OrderedDict()
        for k in keys:
            v = named_parameters[k]
            is_dist_tensor = isinstance(v, torch.distributed.tensor.DTensor)
            local_view = v.to_local() if is_dist_tensor else v
            local_view = self.post_transform_of_local_view(local_view, k)
            transforms[
                self.weight_mapper.policy_map_local_key_to_hf_key(
                    util.clear_weight_name(k)
                )
            ] = local_view
        return transforms

    @cached_property
    def weight_sync_transforms(self) -> List[Tuple[str, Union[torch.Tensor, Callable]]]:
        # 1. get all parameters, but not buffers
        transforms = self.gen_local_view_transforms()

        # 2. do 1->n decomposition on weights like qkv_proj.weight -> q.weight, k.weight, v.weight
        for name, param in self.named_parameters():
            dims_rank_info, dims_map = extract_infomation_from_dtensor(param, name)
            global_shape = tuple(param.shape)

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
                        slice_in_splited, overlap_in_local = tensor_overlap_info_at_dim(
                            {dim: dim_slice}, {dim: local_part}, dim
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
            # `name` from transforms is alread in HF naming convention
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

    def apply_trainable(self, trainable_map: Mapping[str, bool]) -> dict:
        """
        Apply trainable flags to modules and parameters.

        Args:
            trainable_map: mapping of name -> bool.
                    Keys may be:
                    - exact parameter names (from model.named_parameters())
                    - exact module paths (from model.named_modules())

        Returns:
            A dict with lists of which params/modules were touched.

        Raises:
            TypeError: for non-bool values.
            TrainablePathError: if a key matches neither a param nor a module.
        """
        if not isinstance(trainable_map, Mapping):
            raise TypeError("trainable_map must be a mapping[str, bool]")

        # Build lookup tables
        param_map = dict(self.named_parameters())
        module_map = dict(self.named_modules())
        module_map.pop("", None)  # drop root entry to avoid confusion

        touched_params, touched_modules = [], []

        # Process in the insertion order of `config`.
        for name, flag in trainable_map.items():
            if not isinstance(flag, bool):
                raise TypeError(
                    f"value for '{name}' must be bool, got {type(flag).__name__}"
                )

            if name in param_map:
                param_map[name].requires_grad = flag
                touched_params.append(name)
                continue

            if name in module_map:
                for p in module_map[name].parameters(recurse=True):
                    p.requires_grad = flag
                touched_modules.append(name)
                continue

            # Not found: raise with fuzzy suggestions
            raise KeyError(f"Path '{name}' not found among parameters or modules.")
        return {"touched_params": touched_params, "touched_modules": touched_modules}

    def apply_freeze_pattern(self, freeze_pattern: List[str]) -> dict:
        """
        Apply pattern-based freezing to parameters using regex matching.

        Args:
            freeze_pattern: List of regex patterns to match against parameter names.
                    Matched parameters will be frozen (requires_grad=False).

        Returns:
            A dict with pattern match counts.
        """
        import re

        compiled_patterns = [(p, re.compile(p)) for p in freeze_pattern if p]

        pattern_counts: Dict[str, int] = {p: 0 for p in freeze_pattern if p}
        total_params = 0
        frozen_params = 0

        for param_name, param in self.named_parameters():
            total_params += param.numel()

            for pattern_str, pattern_re in compiled_patterns:
                if pattern_re.search(param_name):
                    param.requires_grad = False
                    pattern_counts[pattern_str] += 1
                    util.rank0_print(
                        f"[freeze_pattern] freeze '{param_name}' (matched '{pattern_str}')"
                    )
                    break

            if not param.requires_grad:
                frozen_params += param.numel()

        # Log summary
        for pattern, count in pattern_counts.items():
            if count > 0:
                util.rank0_print(f"[freeze_pattern] '{pattern}' matched {count} params")

        util.rank0_print(
            f"[freeze_pattern] Total={total_params / 1e9:.2f}B, "
            f"Frozen={frozen_params:,}, Trainable={total_params - frozen_params:,}"
        )

        return {"pattern_counts": pattern_counts}

    def apply_trainable_pattern(self, trainable_pattern: List[str]) -> dict:
        """
        Apply pattern-based training to parameters using regex matching.

        Args:
            trainable_pattern: List of regex patterns to match against parameter names.
                    Matched parameters will be set to require_grad=True, others will be set to require_grad=False.

        Returns:
            A dict with pattern match counts.
        """
        import re

        compiled_patterns = [(p, re.compile(p)) for p in trainable_pattern if p]

        pattern_counts: Dict[str, int] = {p: 0 for p in trainable_pattern if p}
        total_params = 0
        trainable_params = 0

        for param_name, param in self.named_parameters():
            total_params += param.numel()
            param.requires_grad = False  # Default to frozen
            for pattern_str, pattern_re in compiled_patterns:
                if pattern_re.search(param_name):
                    param.requires_grad = True
                    pattern_counts[pattern_str] += 1
                    util.rank0_print(
                        f"[trainable_pattern] train '{param_name}' (matched '{pattern_str}')"
                    )
                    break

            if param.requires_grad:
                trainable_params += param.numel()

        # Log summary
        for pattern, count in pattern_counts.items():
            if count > 0:
                util.rank0_print(
                    f"[trainable_pattern] '{pattern}' matched {count} params"
                )

        util.rank0_print(
            f"[trainable_pattern] Total={total_params / 1e9:.2f}B, "
            f"Trainable={trainable_params:,}, Frozen={total_params - trainable_params:,}"
        )

        return {"pattern_counts": pattern_counts}

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

    def step_hook(self):
        """
        Hook to be called after each step update.
        """
        pass

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
        revision: Optional[str] = None,
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
        """
        Check if the model is compatible with context parallelism. By default, it does nothing.
        This is a model-specific check, so it should be overridden in the derived class of different models.
        """
        pass

    def check_tp_compatible(self, tp_size: int):
        """
        Check if the model is compatible with tensor parallelism. By default, it does nothing.
        This is a model-specific check, so it should be overridden in the derived class of different models.
        """
        pass


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
                BaseDataPacker.register(model_type, data_packer_cls)

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
    def build_hf_model(cls, config: CosmosConfig, hf_config_args=None):
        model_name_or_path = config.policy.model_name_or_path
        model = None
        hf_config_args = hf_config_args if hf_config_args is not None else {}
        for k, v in hf_config_args.items():
            logger.info(f"Set hf config args {k} to {v}")
        hf_config = util.retry(AutoConfig.from_pretrained)(
            model_name_or_path, trust_remote_code=True, **hf_config_args
        )
        model_type = hf_config.model_type
        is_supported_model_type = model_type in ModelRegistry._MODEL_REGISTRY
        if not is_supported_model_type or config.train.force_use_hf:
            logger.info(
                f"Model type {hf_config.model_type} not registered or force using HF, using {COSMOS_HF_MODEL_TYPES} instead."
            )
            model_type = COSMOS_HF_MODEL_TYPES

        model_cls = ModelRegistry._MODEL_REGISTRY[model_type]

        cosmos_default_dtype = util.str2torch_dtype(
            config.train.master_dtype
            if config.train.master_dtype is not None
            else config.train.param_dtype
        )
        hf_config.torch_dtype = cosmos_default_dtype

        def _apply_model_post_processing(model, config):
            """Apply LoRA, liger kernel, and trainable map configurations to the model."""
            # Apply LoRA to the model
            if config.policy.lora is not None:
                logger.info(f"Applying LoRA to the model: {config.policy.lora}")
                from cosmos_rl.policy.lora.plugin import (
                    inject_lora_adapters,
                    mark_only_lora_as_trainable,
                )

                model, _ = inject_lora_adapters(model, config.policy.lora)
                mark_only_lora_as_trainable(model, config.policy.lora)

            if config.policy.enable_liger_kernel:
                util.replace_with_liger_equivalents(model)

            # Apply pattern-based trainable configuration
            trainable_pattern = getattr(config.policy, "trainable_pattern", None)
            if trainable_pattern is not None:
                model.apply_trainable_pattern(trainable_pattern)

            # If we further need finer-grained control over trainable parameters, we need to apply trainable flags after LoRA is applied
            if config.policy.trainable_map is not None:
                if config.policy.lora is not None:
                    # Only setting `requires_grad` to `False` can be combined with LoRA
                    # This can be useful for:
                    #  lora_config.target_modules is set to ['q_proj', 'k_proj', 'v_proj']
                    # But there are both `q_proj` and `k_proj` in LLM and vision encoder,
                    # If we only want to train lora on vision, we can disable grad on LLM by setting `config.policy.trainable_map` to `{"model.llm": False}`
                    if any(v for v in config.policy.trainable_map.values()):
                        raise RuntimeError(
                            "If LoRA is applied, only setting `requires_grad` to `False` inside `config.policy.trainable_map` can be combined with LoRA."
                            "Otherwise, please instead include the trainable modules in `config.policy.lora.modules_to_save`."
                        )
                model.apply_trainable(config.policy.trainable_map)

            # Apply pattern-based freeze configuration
            freeze_pattern = getattr(config.policy, "freeze_pattern", None)
            if freeze_pattern is not None:
                model.apply_freeze_pattern(freeze_pattern)

            return model

        def _load_model_with_config(model_cls, hf_config, model_name_or_path, config):
            """Load model and apply post-processing configurations."""
            model = model_cls.from_pretrained(
                hf_config,
                model_name_or_path,
                max_position_embeddings=config.policy.model_max_length,
            )
            return _apply_model_post_processing(model, config)

        def _get_init_context_for_model_build(hf_config):
            # Workaround for OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview
            if (
                hf_config.model_type == "internvl_chat"
                and hasattr(hf_config, "llm_config")
                and hf_config.llm_config.model_type == "gpt_oss"
            ):
                logger.info(f"Using cuda for model build of {model_name_or_path}.")
                return torch.device("cuda")
            elif hf_config.model_type == "openvla" and hasattr(
                hf_config, "timm_model_ids"
            ):
                # VLA models with TIMM vision backbone, use CUDA for initialization
                return torch.device("cuda")
            else:
                return init_on_device("meta", include_buffers=False)

        if hasattr(model_cls, "preprocess_hf_config"):
            hf_config = model_cls.preprocess_hf_config(config)

        init_context = _get_init_context_for_model_build(hf_config)
        with init_context:
            with util.cosmos_default_dtype(cosmos_default_dtype):
                try:
                    model = _load_model_with_config(
                        model_cls, hf_config, model_name_or_path, config
                    )

                except Exception as e:
                    if model_type == COSMOS_HF_MODEL_TYPES:
                        raise e
                    else:
                        logger.warning(
                            f"Failed to load model {model_name_or_path} with error: {e}. Trying to load with {COSMOS_HF_MODEL_TYPES} instead."
                        )
                        model_type = COSMOS_HF_MODEL_TYPES
                        model_cls = ModelRegistry._MODEL_REGISTRY[model_type]

                        try:
                            model = _load_model_with_config(
                                model_cls, hf_config, model_name_or_path, config
                            )
                        except Exception as fallback_e:
                            raise RuntimeError(
                                f"Both primary and fallback model loading strategies failed. "
                                f"Primary: {e}, Fallback: {fallback_e}"
                            ) from e
        if model is None:
            raise ValueError(f"Model {model_name_or_path} not supported.")
        return model

    @classmethod
    def build_diffusers_model(cls, config, diffusers_config_args=None):
        # TODO (yy): Find a similar function like AutoConfig from transformers for diffusers or write one
        model_name_or_path = config.policy.model_name_or_path
        model_revision = config.policy.model_revision or "main"
        model = None
        model_type = util.retry(diffusers_config_fn)(
            model_name_or_path, revision=model_revision
        )["_class_name"]

        model_cls = ModelRegistry._MODEL_REGISTRY[model_type]
        cosmos_default_dtype = util.str2torch_dtype(
            config.train.master_dtype
            if config.train.master_dtype is not None
            else config.train.param_dtype
        )

        def _load_model_with_config(
            model_cls, config, model_name_or_path, model_revision
        ):
            """Load model and apply post-processing configurations."""
            model = model_cls.from_pretrained(
                config, model_name_or_path, model_revision
            )
            return model

        def _get_init_context_for_model_build(device):
            # TODO(yy): support meta init for diffusers model
            # Cannot use torch.device('cuda') here, conflict with scheduler's initialization
            # Control device inside model
            return nullcontext()

        init_context = _get_init_context_for_model_build("cuda")
        with init_context:
            with util.cosmos_default_dtype(cosmos_default_dtype):
                try:
                    model = _load_model_with_config(
                        model_cls, config, model_name_or_path, model_revision
                    )

                except Exception as e:
                    # TODO (yy): Add exception handle
                    raise e

        if model is None:
            raise ValueError(f"Model {model_name_or_path} not supported.")
        return model

    @classmethod
    def build_model(cls, config: CosmosConfig, hf_config_args=None):
        if not config.policy.is_diffusers:
            return cls.build_hf_model(config, hf_config_args)
        else:
            return cls.build_diffusers_model(config, hf_config_args)


class WeightMapper(ABC):
    _WEIGHT_MAPPER_BACKEND_SUPPORTED = ["vllm", "trtllm"]
    _MODEL_WEIGHT_MAPPER_REGISTRY: Dict[str, Tuple[Type["WeightMapper"], int]] = {}

    def __init__(self, hf_config: AutoConfig):
        logger.info(f"WeightMapper: {type(self).__name__} is being initialized.")
        self.config = hf_config
        self.backend = "vllm"  # default rollout backend is vllm.

    @torch.no_grad()
    def policy_map_local_key_for_export_tensor(self, name, param):
        """
        Transform the weight to the Huggingface weight store and naming convention.
        For example, Qwen3 MoE experts' weight of `gate_and_up_proj` are stacked in the 0th dimension(expert dimension) in cosmos-rl,
        with shape [experts, 2 * ffn_dim, hidden_dim], while they are splited into `experts` single-expert weights to store in Huggingface,
        each of the single-expert weights has shape [ffn_dim, hidden_dim] of `gate_proj`, and [ffn_dim, hidden_dim] of `up_proj`.
        """
        yield name, param

    def rollout_prepare_recv(
        self,
        rollout_model: torch.nn.Module,
    ) -> Tuple[Dict[str, torch.Tensor], List[List[Tuple[str, int]]]]:
        """
        Prepare the rollout receive list for P2R weight synchronization.
        It does the splitting of weights if needed, maps the weight names to consistent naming convention with policy side,
        and create the inplace view tensors for vllm model weights to be written by P2R weight sync.
        The final mapped name from this function should be consistent with the name from `policy_map_local_key_to_hf_key` for the same parameter.
        Rollout prepare recv list for P2R weight sync:
            - rollout_weight_inplace_view_map: Dict[str, torch.Tensor]: the map of vllm weight inplace view to be written by P2R weight sync
            - recv_key_n_rank_list: List[List[Tuple[str, int]]]: the list of grouped recv key and its tensor rank
        It call `rollout_split_local_key_n_param_to_hf_key_n_param` to do the mapping and splitting of weights specifically.
        """
        recv_key_n_shape_list = []
        rollout_weight_inplace_view_map = {}
        self.map_to_unsplited_weight_name = {}
        for param_name, param in rollout_model.named_parameters():
            unsplited_weight_name = self.rollout_map_local_key_to_hf_key(param_name)
            group_keys_n_params = (
                self.rollout_split_local_key_n_param_to_hf_key_n_param(
                    param_name, param
                )
            )
            recv_key_n_shape_list.append([(k, w.ndim) for k, w in group_keys_n_params])
            rollout_weight_inplace_view_map.update(
                {k: w for k, w in group_keys_n_params}
            )
            if len(group_keys_n_params) > 1:
                self.map_to_unsplited_weight_name.update(
                    {k: unsplited_weight_name for k, w in group_keys_n_params}
                )
        return rollout_weight_inplace_view_map, recv_key_n_shape_list

    def rollout_split_local_key_n_param_to_hf_key_n_param(
        self, param_name: str, param: torch.Tensor
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Map the local parameter name and param to the Huggingface parameter name and param at rollout side with splitting if needed.
        It does the splitting of weights if needed.
        The returned names should be consistent with the final names in `policy_map_local_key_to_hf_key` for the same parameters.
        This is to make sure the mapped names are consistent between policy and rollout side.
        It can call `rollout_map_local_key_to_hf_key` to transform the base name format alongside the splitting logic.
        Returns the list of splitted weight names and params.
        """
        raise NotImplementedError

    def policy_map_local_key_to_hf_key(self, name: str) -> str:
        """
        Map the local parameter name to the Huggingface parameter name at policy side.
        The name should be consistent with the final name in `rollout_prepare_recv` and `rollout_split_local_key_n_param_to_hf_key_n_param` for the same parameter.
        This is to make sure the mapped name is consistent between policy and rollout side.
        """
        return name

    def rollout_map_local_key_to_hf_key(self, name: str) -> str:
        """
        Map the local parameter name to the Huggingface parameter name format at rollout side.
        It only transforms the name format without splitting. `rollout_split_local_key_n_param_to_hf_key_n_param` will do the splitting if needed.
        This can be called by `rollout_split_local_key_n_param_to_hf_key_n_param` to transform the base name format alongside the splitting logic.
        The name format should be consistent with the final name in `policy_map_local_key_to_hf_key`.
        This is to make sure the mapped name format is consistent between policy and rollout side.
        """
        return name

    def get_unsplited_weight_name(self, weight_key: str) -> str:
        """
        Get the unsplited weight name for a given weight key.
        This method is used to map the splitted weight names back to their original unsplitted names.
        It is inverse of the split operations in function `rollout_prepare_recv` and only do for name tranferring.
        If no split in the weight key, return the original weight key.
        """
        assert hasattr(
            self, "map_to_unsplited_weight_name"
        ), "map_to_unsplited_weight_name is not set. Please call rollout_prepare_recv first."
        if (
            hasattr(self, "map_to_unsplited_weight_name")
            and weight_key in self.map_to_unsplited_weight_name
        ):
            return self.map_to_unsplited_weight_name[weight_key]
        else:
            return weight_key

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
        `name` is in HF naming convention
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
    def packed_modules_mapping(self) -> Dict[str, List[str]]:
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

    def setup_rollout_backend(self, backend: str):
        """
        Setup the rollout backend for the weight mapper.
        """
        self.backend = backend
        if backend not in WeightMapper._WEIGHT_MAPPER_BACKEND_SUPPORTED:
            raise ValueError(f"Backend {backend} is not supported by weight mapper.")

    def rollout_prepare_recv_filter(self, key: str) -> bool:
        """ "
        Filter the weights that are not needed to be synced when generating recv key and shape list
        """
        if "_scale" in key:
            # Filter weight scale
            return True
        return False

    def cosmos_rollout_prepare_recv(
        self,
        rollout_model: Any,
    ) -> Tuple[Dict[str, torch.Tensor], List[List[Tuple[str, int]]]]:
        rollout_weight_inplace_view_map, recv_key_n_shape_list = (
            self.rollout_prepare_recv(rollout_model)
        )
        final_rollout_weight_inplace_view_map = {}
        final_recv_key_n_shape_list = []
        for key, value in rollout_weight_inplace_view_map.items():
            if self.rollout_prepare_recv_filter(key):
                continue
            final_rollout_weight_inplace_view_map[key] = value
        total_count = 0
        for group_keys in recv_key_n_shape_list:
            group_key = group_keys[0][0]
            if self.rollout_prepare_recv_filter(group_key):
                continue
            final_recv_key_n_shape_list.append(group_keys)
            total_count += len(group_keys)
        assert (
            len(final_rollout_weight_inplace_view_map) == total_count
        ), f"{len(final_rollout_weight_inplace_view_map)} != {total_count} in rollout recv instructions generation"
        return final_rollout_weight_inplace_view_map, final_recv_key_n_shape_list

    def update_tensor_view(
        self,
        tensor_view: torch.Tensor,
        recv_tensor: torch.Tensor,
        inst_dest_name: str,
        **kwargs,
    ):
        """
        Update the tensor view with the recv tensor. This is called when weight sync is done, we want to update the
        original tensor in rollout model.

        @param tensor_view: the tensor view to be updated, this is from rollout model
        @param recv_tensor: the recv tensor from policy model, data filled by NCCL recv of P2R.
        @param inst_dest_name: the name of the tensor to be updated, compatible name.
        """
        tmp_recv_tensor = recv_tensor.to(tensor_view.dtype)
        tensor_view.copy_(tmp_recv_tensor)
