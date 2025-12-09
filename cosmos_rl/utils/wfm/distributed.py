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

import ctypes
import functools
import itertools
import math
import os
import pynvml
from contextlib import contextmanager
from datetime import timedelta

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed._tensor.api import DTensor
from torch.distributed.tensor import Replicate, distribute_tensor

if dist.is_available():
    from torch.distributed.distributed_c10d import _get_default_group
    from torch.distributed.utils import (
        _sync_module_states,
        _verify_param_shape_across_processes,
    )

from typing import Any, Callable, Container, Optional

from cosmos_rl.utils.logging import logger
from cosmos_rl.policy.config.wfm import DDPConfig

from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config.wfm import CosmosVisionGenConfig

_COSMOS_GLOBAL_PARALLEL_DIMS = None


class Device:
    # TODO: fill in docstring.

    _nvml_affinity_elements = math.ceil(os.cpu_count() / 64)  # type: ignore

    def __init__(self, device_idx: int):
        # TODO: fill in docstring.
        super().__init__()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)

    def get_name(self) -> str:
        # TODO: fill in docstring.
        return pynvml.nvmlDeviceGetName(self.handle)

    def get_cpu_affinity(self) -> list[int]:
        # TODO: fill in docstring.
        affinity_string = ""
        for j in pynvml.nvmlDeviceGetCpuAffinity(
            self.handle, Device._nvml_affinity_elements
        ):
            # assume nvml returns list of 64 bit ints
            affinity_string = "{:064b}".format(j) + affinity_string
        affinity_list = [int(x) for x in affinity_string]
        affinity_list.reverse()  # so core 0 is in 0th element of list
        return [i for i, e in enumerate(affinity_list) if e != 0]


class DistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    """This extends torch.nn.parallel.DistributedDataParallel with .training_step().

    This borrows the concept of `forward-redirection` from Pytorch lightning. It wraps an CosmosVisionGenModel such that
    model.training_step() would be executed when calling self.training_step(), while preserving the behavior of calling
    model() for Pytorch modules. Internally, this is a double rerouting mechanism (training_step -> forward ->
    training_step), allowing us to preserve the function names and signatures.
    """

    def __init__(self, model: torch.nn.Module, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.show_sync_grad_static_graph_warning = True

    def training_step(self, *args, **kwargs) -> Any:
        # Cache the original model.forward() method.
        original_forward = self.module.forward

        def wrapped_training_step(*_args, **_kwargs):  # noqa: ANN202
            # Unpatch immediately before calling training_step() because itself may want to call the real forward.
            self.module.forward = original_forward
            # The actual .training_step().
            return self.module.training_step(*_args, **_kwargs)

        # Patch the original_module's forward so we can redirect the arguments back to the real method.
        self.module.forward = wrapped_training_step
        # Call self, which implicitly calls self.forward() --> model.forward(), which is now model.training_step().
        # Without calling self.forward() or model.forward() explciitly, implicit hooks are also executed.
        return self(*args, **kwargs)


def init() -> int | None:
    """Initialize distributed training."""
    if dist.is_initialized():
        return torch.cuda.current_device()

    # Set GPU affinity.
    pynvml.nvmlInit()
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    try:
        device = Device(local_rank)
        os.sched_setaffinity(0, device.get_cpu_affinity())
    except pynvml.NVMLError as e:
        logger.warning(f"Failed to set device affinity: {e}")
    # Set up NCCL communication.
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    if dist.is_available():
        torch.cuda.set_device(local_rank)
        # Get the timeout value from environment variable
        timeout_seconds = os.getenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", 1800)
        # Convert the timeout to an integer (if it isn't already) and then to a timedelta
        timeout_timedelta = timedelta(seconds=int(timeout_seconds))
        dist.init_process_group(
            backend="nccl", init_method="env://", timeout=timeout_timedelta
        )
        logger.critical(
            f"Initialized distributed training with local rank {local_rank} with timeout {timeout_seconds}",
        )
    # Increase the L2 fetch granularity for faster speed.
    _libcudart = ctypes.CDLL("libcudart.so")
    # Set device limit on the current device.
    p_value = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    _libcudart.cudaDeviceGetLimit(p_value, ctypes.c_int(0x05))
    logger.info(f"Training with {get_world_size()} GPUs.")


def barrier() -> None:
    """Barrier for all GPUs."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def get_rank(group: Optional[dist.ProcessGroup] = None) -> int:
    """Get the rank (GPU device) of the worker.

    Returns:
        rank (int): The rank of the worker.
    """
    rank = 0
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank(group)
    return rank


def is_rank0() -> bool:
    """Check if current process is the master GPU.

    Returns:
        (bool): True if this function is called from the master GPU, else False.
    """
    return get_rank() == 0


def get_world_size(group: Optional[dist.ProcessGroup] = None) -> int:
    """Get world size. How many GPUs are available in this job.

    Returns:
        world_size (int): The total number of GPUs available in this job.
    """
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size(group)
    return world_size


@torch.no_grad()
def all_gather_tensor(tensor: torch.Tensor) -> list[torch.Tensor]:
    """Gather the corresponding tensor from all GPU devices to a list.

    Args:
        tensor (torch.Tensor): Pytorch tensor.

    Returns:
        tensor_list (list[torch.Tensor]): A list of Pytorch tensors gathered from all GPU devices.
    """
    tensor_list = [torch.zeros_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(tensor_list, tensor)
    return tensor_list


def broadcast(tensor, src, group=None, async_op=False):
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    dist.broadcast(tensor, src=src, group=group, async_op=async_op)


def hsdp_device_mesh(replica_group_size=None, sharding_group_size=None, device=None):
    """
     Initializes a device mesh for use with Hybrid Sharding strategy in FSDP (HSDP) training.

    This function requires explicit sizes for replica and sharding groups to accommodate models
    whose GPU fit is unknown, providing flexibility in distributed training setups.

    Args:
        replica_group_size (int): The size of each replica group. Must be provided to ensure
            the model fits within the available resources.
        sharding_group_size (int): The size of each sharding group that the model can fit. Must be provided to
            ensure the correct distribution of model parameters.
        device (str, optional): The device to use (e.g., "cuda:0"). If None, defaults to "cuda"
            with the local rank as the device index.

    Returns:
        A device mesh object compatible with FSDP.

    Raises:
        ValueError: If replica_group_size or sharding_group_size are not provided, or if the
            world size is not evenly divisible by the sharding group size.
        RuntimeError: If a valid device mesh cannot be created.

    Usage:
        If your model fits on 4 GPUS, and you have 3 nodes of 8 GPUs, then:
        Sharding_Group_Size = 4
        Replica_Groups_Size = (24 total gpus, 4 per sharding group) = 6 Replica Groups
        >>> device_mesh = initialize_device_mesh(replica_group_size, sharding_group_size)
        >>> sharded_model = FSDP(model, device_mesh=device_mesh, ...)
    """

    # world_size = int(os.getenv("WORLD_SIZE", "1"))
    world_size = get_world_size()
    if sharding_group_size is None:
        sharding_group_size = min(world_size, 8)
    sharding_group_size = min(sharding_group_size, world_size)
    if replica_group_size is None:
        replica_group_size = world_size // sharding_group_size

    device = device or "cuda"

    if world_size % sharding_group_size != 0:
        raise ValueError(
            f"World size {world_size} is not evenly divisible by "
            f"sharding group size {sharding_group_size}."
        )

    if (world_size // sharding_group_size) % replica_group_size != 0:
        raise ValueError(
            f"The calculated number of replica groups is not evenly divisible by "
            f"replica_group_size {replica_group_size}."
        )

    device_mesh = init_device_mesh(
        device,
        (replica_group_size, sharding_group_size),
        mesh_dim_names=("replicate", "shard"),
    )
    if device_mesh is None:
        raise RuntimeError("Failed to create a valid device mesh.")

    logger.critical(
        f"Device mesh initialized with replica group size {replica_group_size} and sharding group size {sharding_group_size}"
    )

    return device_mesh


def common_broadcast(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    ndims1 = x.ndim
    ndims2 = y.ndim

    common_ndims = min(ndims1, ndims2)
    for axis in range(common_ndims):
        assert x.shape[axis] == y.shape[axis], "Dimensions not equal at axis {}".format(
            axis
        )

    if ndims1 < ndims2:
        x = x.reshape(x.shape + (1,) * (ndims2 - ndims1))
    elif ndims2 < ndims1:
        y = y.reshape(y.shape + (1,) * (ndims1 - ndims2))

    return x, y


def batch_mul(x: Tensor, y: Tensor) -> Tensor:
    x, y = common_broadcast(x, y)
    return x * y


def get_local_tensor_if_DTensor(tensor: Tensor | DTensor) -> torch.tensor:
    if isinstance(tensor, DTensor):
        local = tensor.to_local()
        # As per PyTorch documentation, if the communication is not finished yet, we need to wait for it to finish
        # https://pytorch.org/docs/stable/distributed.tensor.html#torch.distributed.tensor.DTensor.to_local
        if isinstance(local, AsyncCollectiveTensor):
            return local.wait()
        else:
            return local
    return tensor


def broadcast_dtensor(
    tensor: torch.Tensor, cp_or_tp_mesh: dist.DeviceMesh
) -> torch.Tensor:
    tensor = tensor.to("cuda")
    if cp_or_tp_mesh.size() > 1:
        tensor = distribute_tensor(tensor, cp_or_tp_mesh, [Replicate()]).to_local()
    return tensor


def broadcast_dtensor_with_shape_check(
    tensor: torch.Tensor, cp_or_tp_mesh: dist.DeviceMesh
) -> torch.Tensor:
    """Broadcast a tensor and check if the shape is the same across CP/TP ranks.
    If not, create a new tensor matching rank 0 and broadcast it.

    Args:
        tensor (torch.Tensor): The tensor to broadcast.
        cp_or_tp_mesh (DeviceMesh): The device mesh used to broadcast.

    Returns:
        torch.Tensor: The broadcasted tensor.
    """
    # create a tensor with the original value of the shape
    original_shape = torch.tensor(tensor.shape).cuda()

    # create a tensor that tracks the shape from rank 0.
    final_shape = torch.tensor(tensor.shape).cuda()
    final_shape = broadcast_dtensor(final_shape, cp_or_tp_mesh)

    # if final shape is different from current shape, create a new tensor
    if final_shape.ne(original_shape).any():
        tensor = torch.zeros(
            final_shape.tolist(), dtype=tensor.dtype, device=tensor.device
        )

    tensor = broadcast_dtensor(tensor, cp_or_tp_mesh)
    return tensor


def broadcast_dtensor_model_states(model: nn.Module, mesh: dist.DeviceMesh):
    """Broadcast model states from replicate mesh's rank 0."""
    replicate_group = mesh.get_group("replicate")
    all_ranks = dist.get_process_group_ranks(replicate_group)
    if len(all_ranks) == 1:
        return

    for _, tensor in itertools.chain(model.named_parameters(), model.named_buffers()):
        # Get src rank which is the first rank in each replication group
        src_rank = all_ranks[0]
        # Broadcast the local tensor
        local_tensor = get_local_tensor_if_DTensor(tensor)
        dist.broadcast(
            local_tensor,
            src=src_rank,
            group=replicate_group,
        )


def parallel_model_wrapper(
    config_ddp: DDPConfig, model: torch.nn.Module
) -> torch.nn.Module | DistributedDataParallel:
    """Wraps the model to enable data parallalism for training across multiple GPU devices.

    Args:
        config_ddp (DDPConfig): The data parallel config.
        model (torch.nn.Module): The PyTorch module.

    Returns:
        model (torch.nn.Module | DistributedDataParallel): The data parallel model wrapper
            if distributed environment is available, otherwise return the original model.
    """
    if dist.is_available() and dist.is_initialized():
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        try:
            # So we use dp_shard_cp group for DDP, include cp as megatron does: with_context_parallel
            ddp_group = _COSMOS_GLOBAL_PARALLEL_DIMS.mesh["dp_shard_cp"].get_group()
        except Exception as e:
            logger.info(e)
            logger.info(
                "global_parallelism not initialized, treating all GPUs equally for DDP"
            )
            ddp_group = None

        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=config_ddp.find_unused_parameters,
            static_graph=config_ddp.static_graph,
            broadcast_buffers=config_ddp.broadcast_buffers,
            process_group=ddp_group,
        )
    return model


@contextmanager
def ddp_sync_grad(model, enabled):
    r"""
    Context manager to enable/disable gradient synchronizations across DDP processes for DDP model.
    Modified from:
    https://pytorch.org/docs/stable/_modules/torch/nn/parallel/distributed.html#DistributedDataParallel.no_sync
    Note that this is incompatible with static_graph=True and will be an no-op if static_graph=True.

    Within this context, gradients will be accumulated on module
    variables, which will later be synchronized in the first
    forward-backward pass exiting the context.

    .. warning::
        The forward pass should be included inside the context manager, or
        else gradients will still be synchronized.
    """
    assert isinstance(model, torch.nn.Module)
    if isinstance(model, DistributedDataParallel):
        old_require_backward_grad_sync = model.require_backward_grad_sync
        if model.static_graph and model.require_backward_grad_sync != enabled:
            if model.show_sync_grad_static_graph_warning:
                logger.warning(
                    "DDP static_graph=True is incompatible with sync_grad(). Performance will be reduced."
                )
                model.show_sync_grad_static_graph_warning = False
        else:
            model.require_backward_grad_sync = enabled
    try:
        yield
    finally:
        if isinstance(model, DistributedDataParallel):
            model.require_backward_grad_sync = old_require_backward_grad_sync


def rank0_only(func: Callable) -> Callable:
    """Apply this function only to the master GPU.

    Example usage:
        @rank0_only
        def func(x):
            return x + 3

    Args:
        func (Callable): a function.

    Returns:
        (Callable): A function wrapper executing the function only on the master GPU.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # noqa: ANN202
        if is_rank0():
            return func(*args, **kwargs)
        else:
            return None

    return wrapper


def is_tp_cp_pp_rank0():
    assert (
        _COSMOS_GLOBAL_PARALLEL_DIMS is not None
    ), "global_parallelism not initialized"
    mp_rank, _ = _COSMOS_GLOBAL_PARALLEL_DIMS.mp_coord
    cp_rank, _ = _COSMOS_GLOBAL_PARALLEL_DIMS.cp_coord
    pp_rank, _ = _COSMOS_GLOBAL_PARALLEL_DIMS.pp_coord
    return mp_rank == 0 and pp_rank == 0 and cp_rank == 0


def initialize_global_parallelism(config: CosmosVisionGenConfig):
    # Set up the distributed computing environment. make sure distributed.init() is called before this function is called.
    global _COSMOS_GLOBAL_PARALLEL_DIMS
    tp_size = config.model_parallel.tensor_model_parallel_size
    pp_size = config.model_parallel.pipeline_model_parallel_size
    cp_size = config.model_parallel.context_parallel_size
    dp_shard_size = config.model.fsdp_shard_size
    model_size = tp_size * pp_size * cp_size * dp_shard_size
    world_size = dist.get_world_size()
    assert (
        world_size % model_size == 0
    ), f"world_size must be divisible by model_size, got: {world_size} % {model_size} != 0"
    dp_size = world_size // model_size
    _COSMOS_GLOBAL_PARALLEL_DIMS = ParallelDims(
        dp_replicate=dp_size,
        dp_shard=dp_shard_size,
        cp=cp_size,
        tp=tp_size,
        pp=pp_size,
        world_size=world_size,
        pp_dynamic_shape=False,
    )
    _COSMOS_GLOBAL_PARALLEL_DIMS.build_mesh(device_type="cuda")


def get_global_parallel_dims():
    assert (
        _COSMOS_GLOBAL_PARALLEL_DIMS is not None
    ), "Global parallelism is not initialized"
    return _COSMOS_GLOBAL_PARALLEL_DIMS


def sync_model_states(
    model: torch.nn.Module,
    process_group: Optional[dist.ProcessGroup] = None,
    src: int = 0,
    params_and_buffers_to_ignore: Optional[Container[str]] = None,
    broadcast_buffers: bool = True,
):
    """
    Modify based on DDP source code
    Synchronizes the parameters and buffers of a model across different processes in a distributed setting.

    This function ensures that all processes in the specified process group have the same initial parameters and
    buffers from the source rank, typically rank 0. It is useful when different processes start with different model
    states and a synchronization is required to ensure consistency across all ranks.

    Args:
        model (nn.Module): The model whose parameters and buffers are to be synchronized.
        process_group (dist.ProcessGroup, optional): The process group for communication. If None,
            the default group is used. Defaults to None.
        src (int, optional): The source rank from which parameters and buffers will be broadcasted.
            Defaults to 0.
        params_and_buffers_to_ignore (Optional[Container[str]], optional): A container of parameter and buffer
            names to exclude from synchronization. Defaults to None, which means all parameters and buffers are
            included.
        broadcast_buffers (bool, optional): Whether to broadcast buffers or not. Defaults to True.

    Side Effects:
        This function modifies the state of the model in-place to synchronize it with the source rank's model state.

    Raises:
        RuntimeError: If the shapes of parameters across processes do not match, a runtime error will be raised.

    Examples:
        >>> # downloading duplicated model weights from s3 in each rank and save network bandwidth
        >>> # useful and save our time when model weights are huge
        >>> if dist.get_rank == 0:
        >>>     model.load_state_dict(network_bound_weights_download_fn(s3_weights_path))
        >>> dist.barrir()
        >>> sync_model_states(model) # sync rank0 weights to other ranks
    """
    if not dist.is_available() or not dist.is_initialized():
        return
    if process_group is None:
        process_group = _get_default_group()
    if not params_and_buffers_to_ignore:
        params_and_buffers_to_ignore = set()

    logger.info(
        f"Synchronizing model states from rank {src} to all ranks in process group {dist.get_process_group_ranks(process_group)}."
    )

    # Build tuple of (module, parameter) for all parameters that require grads.
    modules_and_parameters = [
        (module, parameter)
        for module_name, module in model.named_modules()
        for parameter in [
            param
            # Note that we access module.named_parameters instead of
            # parameters(module). parameters(module) is only needed in the
            # single-process multi device case, where it accesses replicated
            # parameters through _former_parameters.
            for param_name, param in module.named_parameters(recurse=False)
            if f"{module_name}.{param_name}" not in params_and_buffers_to_ignore
            # if param.requires_grad
            # and f"{module_name}.{param_name}" not in params_and_buffers_to_ignore
        ]
    ]

    # Deduplicate any parameters that might be shared across child modules.
    memo = set()
    modules_and_parameters = [
        # "p not in memo" is the deduplication check.
        # "not memo.add(p)" is always True, and it's only there to cause "add(p)" if needed.
        (m, p)
        for m, p in modules_and_parameters
        if p not in memo and not memo.add(p)  # type: ignore[func-returns-value]
    ]

    # Build list of parameters.
    parameters = [parameter for _, parameter in modules_and_parameters]
    if len(parameters) == 0:
        return

    _verify_param_shape_across_processes(process_group, parameters)

    _sync_module_states(
        module=model,
        process_group=process_group,
        broadcast_bucket_size=int(250 * 1024 * 1024),
        src=src,
        params_and_buffers_to_ignore=params_and_buffers_to_ignore,
        broadcast_buffers=broadcast_buffers,
    )
