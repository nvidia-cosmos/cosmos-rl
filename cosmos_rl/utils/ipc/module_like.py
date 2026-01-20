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

import warnings

import torch
from typing import Dict, Iterator, Tuple, Set
from torch.nn import Parameter


class ModuleLike:
    """
    The adaptor to handle the state_dict works like a nn.module.

    Attention, this class is just a mirror of the nn.Module in another process. Can only ensure the weight's memory is shared.
    """

    def __init__(
        self, state_dict: Dict[str, torch.Tensor], not_parameter_names: Set[str]
    ):
        """
        Initialize the ModuleLike with the given state_dict.

        Args:
            state_dict: The state dict of the module.
            not_parameter_names: The names of the tensors that are not parameters.
        """
        self._parameters: Dict[str, Parameter] = {}
        # it will not appear in named_parameters(), for example, lm_head.weight.
        self._not_parameter_tensors: Dict[str, torch.Tensor] = {}
        self._modules: Dict[str, "ModuleLike"] = {}

        self.__recurse_init_module(state_dict, not_parameter_names)

    def __recurse_init_module(
        self, state_dict: Dict[str, torch.Tensor], not_parameter_names: Set[str]
    ) -> "ModuleLike":
        """
        Recurse init the module.
        """
        # Convert the state_dict to a nested dictionary.
        nested_state_dict = {}
        for name, value in state_dict.items():
            if "." in name:
                parent_name, child_name = name.split(".", 1)
                if parent_name not in nested_state_dict:
                    nested_state_dict[parent_name] = {}
                nested_state_dict[parent_name][child_name] = value
            else:
                nested_state_dict[name] = value

        nested_not_parameter_names = {}
        for name in not_parameter_names:
            if "." in name:
                parent_name, child_name = name.split(".", 1)
                if parent_name not in nested_not_parameter_names:
                    nested_not_parameter_names[parent_name] = set()
                nested_not_parameter_names[parent_name].add(child_name)
            else:
                nested_not_parameter_names[name] = True

        # update the parameters and modules.
        for name, value in nested_state_dict.items():
            if isinstance(value, dict):
                chile_not_parameter_names = nested_not_parameter_names.get(name, set())
                self._modules[name] = ModuleLike(value, chile_not_parameter_names)
            else:
                if name in nested_not_parameter_names:
                    self._not_parameter_tensors[name] = value
                else:
                    # wrap the value to Parameter if it is not a Parameter, this keep same behavior with nn.Module
                    if not isinstance(value, Parameter):
                        value = Parameter(value)
                    # not trainable by default for inference engine.
                    value.requires_grad = False
                    self._parameters[name] = value

    def __getitem__(self, idx: int | str) -> "ModuleLike":
        """
        Simulate the behavior like ModuleList. Return the sub-module.
        """
        assert isinstance(
            idx, (int, str)
        ), "For ModuleList, ModuleDict, the index must be an integer or string, but got {type(idx)}"
        idx_str = str(idx)
        if idx_str not in self._modules:
            raise IndexError(f"ModuleList index {idx} is out of range")
        return self._modules[idx_str]

    def __getattr__(self, name: str) -> torch.Tensor | None:
        """
        Get the parameter or sub-module of the module.
        """
        if "_parameters" in self.__dict__ and name in self._parameters:
            return self._parameters[name]
        if "_modules" in self.__dict__ and name in self._modules:
            return self._modules[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value) -> None:
        """
        Set the parameter or sub-module of the module.
        """
        if "_parameters" in self.__dict__ and name in self._parameters:
            raise ValueError("Cannot set parameter of ModuleLike")
        if "_modules" in self.__dict__ and name in self._modules:
            raise ValueError("Cannot set sub-module of ModuleLike")

        # TODO(zjx): support set tensor to compatible with `replace_weight_of_quantized_module`
        if isinstance(value, torch.Tensor):
            warnings.warn(
                f"Setting tensor {name} to ModuleLike will not sync to other processes"
            )
            self._parameters[name] = value
        elif isinstance(value, (ModuleLike, torch.nn.Module)):
            warnings.warn(
                f"Setting Module {name} to ModuleLike will not sync to other processes"
            )
            self._modules[name] = value

        # Set common attribute to the module.
        super().__setattr__(name, value)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Get the state dict of the module.
        """
        state_dict = {}
        for name, param in self._parameters.items():
            state_dict[name] = param
        for name, tensor in self._not_parameter_tensors.items():
            state_dict[name] = tensor
        for name, sub_module in self._modules.items():
            for sub_name, sub_value in sub_module.state_dict().items():
                state_dict[f"{name}.{sub_name}"] = sub_value
        return state_dict

    def named_parameters(self, *args, **kwargs) -> Iterator[Tuple[str, Parameter]]:
        """
        Get the named parameters of the module.
        """
        for name, param in self._parameters.items():
            yield name, param
        for name, sub_module in self._modules.items():
            for sub_name, sub_value in sub_module.named_parameters():
                yield f"{name}.{sub_name}", sub_value

    def named_modules(
        self, prefix: str = "", *args, **kwargs
    ) -> Iterator[Tuple[str, torch.nn.Module]]:
        """
        Get the named modules of the module.
        """
        yield prefix, self
        for name, module in self._modules.items():
            if module is None:
                continue
            submodule_prefix = prefix + ("." if prefix else "") + name
            yield from module.named_modules(submodule_prefix, *args, **kwargs)

    def modules(self) -> Iterator[torch.nn.Module]:
        """
        Get the modules of the module.
        """
        yield self
        for module in self._modules.values():
            if module is None:
                continue
            yield from module.modules()
