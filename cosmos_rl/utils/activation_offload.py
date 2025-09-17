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

import torch
import functools
from abc import abstractmethod

from typing import Any, Callable, Tuple
from torch.distributed.fsdp import FSDPModule as FSDP

from cosmos_rl.utils.logging import logger


def _gen_unique_tensor_key(tensor):
    key = (tensor.untyped_storage().data_ptr() + tensor.storage_offset(), tensor.dtype)
    return key


class TrainableParameterFilter:
    def __init__(self):
        self.model_parameters_storage = set()

    def __call__(self, tensor):
        return tensor.untyped_storage().data_ptr() not in self.model_parameters_storage

    def update_model_parameters(self, model):
        new_storage = set()
        for p in model.parameters():
            if p.requires_grad:
                new_storage.add(p.data.untyped_storage().data_ptr())
        self.model_parameters_storage = new_storage


class LayerSyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, cpu_offload_handler):
        cpu_offload_handler.on_group_commit_forward()
        ctx.cpu_offload_handler = cpu_offload_handler
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        cpu_offload_handler = ctx.cpu_offload_handler
        cpu_offload_handler.on_group_commit_backward()
        return grad_output, None


layer_prefetch_offload_sync = LayerSyncFunction.apply


class ActivationOffloadHandler:
    def __init__(self):
        pass

    @abstractmethod
    def push(self, tensor: torch.Tensor, **kwargs):
        pass

    @abstractmethod
    def pop(self, tensor_tag: Any, **kwargs):
        pass


class SynchronizedActivationOffloadHandler(ActivationOffloadHandler):
    def __init__(
        self,
        layers_to_offload: int,
        tensor_offload_filter: Callable[[torch.Tensor], bool] = lambda _: True,
    ):
        super().__init__()

        self.layers_to_offload = layers_to_offload
        self.tensor_offload_filter = tensor_offload_filter
        self.tensor_tag_to_state = {}
        self.current_layer_index = 0
        self.tensor_count_current_layer = 0

    @staticmethod
    def offload(cls, tensor: torch.Tensor):
        cpu_tensor = torch.empty(
            tensor.size(),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device="cpu",
            pin_memory=True,
        )
        cpu_tensor.copy_(tensor, non_blocking=True)
        return (tensor.device, cpu_tensor)

    @staticmethod
    def reload(cls, state: Tuple[torch.device, torch.Tensor]):
        dev, cpu_tensor = state
        non_blocking = cpu_tensor.is_pinned()
        return cpu_tensor.to(dev, non_blocking=non_blocking)

        return cpu_tensor.to(dev, non_blocking=True)

    def push(self, tensor: torch.Tensor, **kwargs):
        tensor_key = (self.current_layer_index, self.tensor_count_current_layer)
        self.tensor_count_current_layer += 1
        assert tensor_key not in self.tensor_tag_to_state
        if (
            self.current_layer_index < self.layers_to_offload
            and self.tensor_offload_filter(tensor)
        ):
            self.tensor_tag_to_state[tensor_key] = self.offload(tensor)
        else:
            self.tensor_tag_to_state[tensor_key] = tensor
        return tensor_key

    def pop(self, tensor_tag: Any, **kwargs):
        assert tensor_tag in self.tensor_tag_to_state
        state = self.tensor_tag_to_state.pop(tensor_tag)
        if isinstance(state, tuple):
            tensor = self.reload(state)
        else:
            tensor = state
        return tensor


class AsynchronousActivationOffloadHandler(SynchronizedActivationOffloadHandler):
    def __init__(
        self,
        layers_to_offload: int,
        num_total_layers: int,
        tensor_offload_filter: Callable[[torch.Tensor], bool] = lambda _: True,
    ):
        super().__init__(layers_to_offload, tensor_offload_filter)

        self.num_total_layers = num_total_layers
        self.tensor_tag_to_buf = {}
        self.offloaded_layers_count = 0
        self.layer_window_map = {}
        self.layer_offload_mapping = {}

        constant = 0
        for i in range(self.layers_to_offload):
            # lms: change here to group
            self.layer_window_map[i] = (
                self.num_total_layers // self.layers_to_offload
            ) * (i + 1) - 1
            if i < (self.num_total_layers % self.layers_to_offload):
                self.layer_window_map[i] += i + 1
                constant = i + 1
            else:
                self.layer_window_map[i] += constant

        self.d2h_stream = torch.cuda.Stream()
        self.h2d_stream = torch.cuda.Stream()

    def push(self, tensor: torch.Tensor, **kwargs):
        if self.tensor_offload_filter(tensor):
            tensor_tag = (self.current_layer_index, self.tensor_count_current_layer)
            self.tensor_count_current_layer += 1
            assert tensor_tag not in self.tensor_tag_to_state
            self.tensor_tag_to_state[tensor_tag] = tensor
            if self.current_layer_index < self.layers_to_offload:
                self.tensor_tag_to_buf[tensor_tag] = tensor

        else:
            tensor_tag = tensor

        return tensor_tag

    def pop(self, tensor_tag: Any, **kwargs):
        if isinstance(tensor_tag, torch.Tensor):
            return tensor_tag
        assert tensor_tag in self.tensor_tag_to_state
        tensor = self.tensor_tag_to_state.pop(tensor_tag)
        self.tensor_tag_to_buf.pop(tensor_tag, None)

        assert not isinstance(tensor, tuple)
        return tensor

    def bulk_offload_layer(self, layer_to_offload):
        offload_mapping = {}
        with torch.cuda.stream(self.d2h_stream):
            for tensor_tag, state in self.tensor_tag_to_state.items():
                layer_id, _ = tensor_tag
                if layer_id == layer_to_offload:
                    assert not isinstance(state, tuple)
                    key = _gen_unique_tensor_key(state)
                    if key not in offload_mapping:
                        offload_mapping[key] = state

                    self.tensor_tag_to_state[tensor_tag] = (key, state.shape)

            for key, tensor in offload_mapping.items():
                state = self.offload(tensor)
                offload_mapping[key] = state

            self.layer_offload_mapping[layer_to_offload] = offload_mapping

    def synchronize_on_layer_commit_forward(self, current_layer):
        if self.current_layer_index == 0:
            self.d2h_stream.wait_stream(torch.cuda.current_stream())
            self.bulk_offload_layer(self.current_layer_index)

        if (
            self.layer_window_map[self.offloaded_layers_count]
            == self.current_layer_index
        ):
            self.d2h_stream.wait_stream(torch.cuda.current_stream())
            torch.cuda.current_stream().wait_stream(self.d2h_stream)

            for tensor_tag, _ in self.tensor_tag_to_buf.items():
                if tensor_tag[0] == self.offloaded_layers_count:
                    self.tensor_tag_to_buf[tensor_tag] = None

            if self.offloaded_layers_count < (self.layers_to_offload - 1):
                self.bulk_offload_layer(self.offloaded_layers_count + 1)

            self.offloaded_layers_count += 1

    def on_layer_commit_forward(self):
        self.synchronize_on_layer_commit_forward(self.current_layer_index)
        super().on_layer_commit_forward()

    @torch.no_grad()
    def bulk_reload_layer(self, layer_to_reload):
        assert layer_to_reload < self.layers_to_offload
        with torch.cuda.stream(self.h2d_stream):
            offload_mapping = self.layer_offload_mapping.pop(layer_to_reload)
            assert offload_mapping is not None
            for key, state in offload_mapping.items():
                offload_mapping[key] = self.reload(state)
            for tensor_tag, state in self.tensor_tag_to_state.items():
                layer_id, _ = tensor_tag
                if layer_id == layer_to_reload and not isinstance(state, torch.Tensor):
                    assert isinstance(state, tuple), f"{layer_id} {state}"
                    key, shape = state
                    recovered_tensor = offload_mapping[key].view(shape)
                    self.tensor_tag_to_state[tensor_tag] = recovered_tensor

    def on_layer_commit_backward(self):
        self.current_layer_index -= 1
        assert self.current_layer_index >= 0

        if (
            self.layer_window_map[self.offloaded_layers_count - 1]
            == self.current_layer_index
        ):
            self.h2d_stream.wait_stream(torch.cuda.current_stream())
            torch.cuda.current_stream().wait_stream(self.h2d_stream)

            self.bulk_reload_layer(self.offloaded_layers_count - 1)

            self.offloaded_layers_count -= 1 if self.offloaded_layers_count > 1 else 0

        if self.current_layer_index == 0:
            torch.cuda.current_stream().wait_stream(self.h2d_stream)
            self.offloaded_layers_count = 0


class CPUOffloadHookWithActivationOffloadHandler:
    def __init__(self, activation_offload_handler: ActivationOffloadHandler):
        self.activation_offload_handler = activation_offload_handler

    def __enter__(self):
        torch._C._autograd._push_saved_tensors_default_hooks(
            self.on_save_for_backward, self.on_get_saved_tensor
        )

    def __exit__(self, *args, **kwargs):
        torch._C._autograd._pop_saved_tensors_default_hooks()

    def on_save_for_backward(self, tensor: torch.Tensor):
        return self.activation_offload_handler.push(tensor)

    def on_get_saved_tensor(self, saved_state):
        return self.activation_offload_handler.pop(saved_state)


def get_activation_offload_handler(
    layers_to_offload: int,
    num_total_layers: int,
    tensor_offload_filter: Callable[[torch.Tensor], bool] = lambda _: True,
):
    cpu_offload_handler = AsynchronousActivationOffloadHandler(
        layers_to_offload, num_total_layers, tensor_offload_filter
    )

    def layer_prefetch_offload_async(tensor):
        return layer_prefetch_offload_sync(tensor, cpu_offload_handler)

    return cpu_offload_handler, layer_prefetch_offload_async


class ActivationOffloader:
    def __init__(
        self,
        cpu_offload_context: CPUOffloadHookWithActivationOffloadHandler,
        sync_func: Callable[[torch.Tensor], torch.Tensor],
        tensor_offload_filter: Callable[[torch.Tensor], bool] = lambda _: True,
    ):
        self._cpu_offload_context = cpu_offload_context
        self._sync_func = sync_func
        self._tensor_offload_filter = tensor_offload_filter

    def pre_forward(self, module):
        if module.training:
            self._cpu_offload_context.__enter__()
            self._tensor_offload_filter.update_model_parameters(module)

    def post_forward(self, module):
        if module.training:
            self._cpu_offload_context.__exit__(None, None, None)

    def forward(self, module, forward_method, *args, **kwargs):
        if not module.training:
            return forward_method(*args, **kwargs)

        ret = forward_method(*args, **kwargs)
        binded_tensor = ret
        if isinstance(ret, tuple):
            binded_tensor = ret[0]
        binded_tensor = self._sync_func(binded_tensor)
        final_ret = binded_tensor
        if isinstance(ret, tuple):
            final_ret = (final_ret,) + ret[1:]
        return final_ret

    def wrap_module_forward_method(self, module):
        orig_method = module.forward
        handler = self

        @functools.wraps(orig_method)
        def wrapped_method(model_self, *args, **kwargs):
            nonlocal handler
            handler.pre_forward(model_self)
            out = handler.forward(model_self, orig_method, *args, **kwargs)
            handler.post_forward(model_self)
            return out

        module.forward = wrapped_method.__get__(module, type(module))


def enable_activation_offload(model):
    model_layers = []

    def get_layers(module):
        nonlocal model_layers
        for name, child in module.named_children():
            if not isinstance(child, FSDP):
                get_layers(child)
            else:
                wrapped_module = child._fsdp_wrapped_module
                if not isinstance(wrapped_module, torch.nn.Embedding):
                    model_layers.append(child)

    get_layers(model)

    if len(model_layers) < 3:
        logger.warning(
            f"Find only {len(model_layers)} layers, not neccessary to enable activation offload"
        )
        return
    layers_to_offload = len(model_layers) - 1
    num_total_layers = len(model_layers)
    tensor_offload_filter = TrainableParameterFilter()
    cpu_offload_handler, sync_func = get_activation_offload_handler(
        layers_to_offload, num_total_layers, tensor_offload_filter
    )
    activation_offloader = ActivationOffloader(
        cpu_offload_handler, sync_func, tensor_offload_filter
    )
    for layer in model_layers:
        if isinstance(layer, FSDP):
            layer = layer._fsdp_wrapped_module
        activation_offloader.wrap_module_forward_method(layer)
