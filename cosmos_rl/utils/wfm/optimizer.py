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

import collections
import functools
import itertools
import math
from apex.multi_tensor_apply import multi_tensor_applier
from copy import deepcopy
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim.lr_scheduler import LambdaLR

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.wfm import distributed
from cosmos_rl.policy.config.wfm.reason1_model_config import (
    FSDP2ModelConfig,
)


class FusedAdam(torch.optim.Optimizer):
    """Implements Adam algorithm.

    Currently GPU-only.  Requires Apex to be installed via
    ``pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./``.

    This version of fused Adam implements 2 fusions.

      * Fusion of the Adam update's elementwise operations
      * A multi-tensor apply launch that batches the elementwise updates applied to all the model's parameters
        into one or a few kernel launches.

    :class:`apex.optimizers.FusedAdam` may be used as a drop-in replacement for ``torch.optim.AdamW``,
    or ``torch.optim.Adam`` with ``adam_w_mode=False``::

        opt = apex.optimizers.FusedAdam(model.parameters(), lr = ....)
        ...
        opt.step()

    :class:`apex.optimizers.FusedAdam` may be used with or without Amp.  If you wish to use :class:`FusedAdam` with Amp,
    you may choose any ``opt_level``::

        opt = apex.optimizers.FusedAdam(model.parameters(), lr = ....)
        model, opt = amp.initialize(model, opt, opt_level="O0" or "O1 or "O2")
        ...
        opt.step()

    In general, ``opt_level="O1"`` is recommended.


    .. warning::
        A previous version of :class:`FusedAdam` allowed a number of additional arguments to ``step``.
        These additional arguments are now deprecated and unnecessary.

    Adam was been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in FusedAdam!
        adam_w_mode (boolean, optional): Apply L2 regularization or weight decay
            True for decoupled weight decay(also known as AdamW) (default: True)
        capturable (bool, optional): whether to use the version of the optimizer
            that can be used with CUDA Graphs. (default: False)
        master_weights (bool, optional): whether to maintain FP32 master weights
           in the optimizer with FP16 mixed precision training, currently can
           only be used with capturable set to True. (default: False)

    .. _Adam - A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        adam_w_mode=True,
        weight_decay=0.0,
        amsgrad=False,
        capturable=False,
        master_weights=False,
    ):
        if amsgrad:
            raise RuntimeError("FusedAdam does not support the AMSGrad variant.")
        if master_weights and not capturable:
            raise RuntimeError(
                "Master weights is currently only supported with the capturable version."
            )
        # If the optimizer is capturable then LR should be a tensor (on GPU)
        logger.warning(
            f"FusedAdam master_weights: {master_weights} capturable: {capturable}"
        )
        lr = torch.tensor(lr, dtype=torch.float32) if capturable else lr
        defaults = dict(
            lr=lr,
            bias_correction=bias_correction,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super(FusedAdam, self).__init__(params, defaults)
        self.adam_w_mode = 1 if adam_w_mode else 0

        self.capturable = capturable
        self.master_weights = master_weights

        self.param_groups_master = None

        if capturable:
            for idx, group in enumerate(self.param_groups):
                if len(group["params"]) == 0:
                    continue
                device = group["params"][0].device
                for item in ["lr"]:
                    if isinstance(group[item], float):
                        group[item] = torch.tensor(group[item], dtype=torch.float32)
                    self.param_groups[idx][item] = group[item].to(device=device)

            self._step_supports_amp_scaling = True

        if multi_tensor_applier.available:
            import amp_C

            # Skip buffer
            self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device="cuda")
            self.multi_tensor_adam = amp_C.multi_tensor_adam
            self.multi_tensor_adam_capturable = amp_C.multi_tensor_adam_capturable
            self.multi_tensor_adam_capturable_master = (
                amp_C.multi_tensor_adam_capturable_master
            )
        else:
            raise RuntimeError("apex.optimizers.FusedAdam requires cuda extensions")

    def step(
        self,
        closure=None,
        grads=None,
        output_params=None,
        scale=None,
        grad_norms=None,
        grad_scaler=None,
    ):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        The remaining arguments are deprecated, and are only retained (for the moment) for error-checking purposes.
        """
        if any(p is not None for p in [grads, output_params, scale, grad_norms]):
            raise RuntimeError(
                "FusedAdam has been updated. "
                "Simply initialize it identically to torch.optim.Adam, and call step() with no arguments."
            )
        loss = None
        if closure is not None:
            loss = closure()

        if self.param_groups_master is None:
            # Create full precision master weights
            self.param_groups_master = []
            for i, pg in enumerate(self.param_groups):
                param_list = pg["params"]
                self.param_groups_master.append(
                    {
                        # Change related to master weights
                        "params": [
                            distributed.get_local_tensor_if_DTensor(p)
                            .clone()
                            .detach()
                            .float()
                            if self.master_weights
                            else None
                            for p in param_list
                        ],
                    }
                )

        for group, group_master in zip(self.param_groups, self.param_groups_master):
            if len(group["params"]) == 0:
                continue
            device = group["params"][0].device
            bias_correction = (
                1 if "bias_correction" in group and group["bias_correction"] else 0
            )
            beta1, beta2 = group["betas"]

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if "step" in group:
                if self.capturable:
                    group["step"] = (
                        group["step"].to(device=device)
                        if isinstance(group["step"], torch.Tensor)
                        else torch.tensor(
                            group["step"], dtype=torch.int32, device=device
                        )
                    )
                    group["step"] += (self._dummy_overflow_buf != 1).to(torch.int)
                else:
                    group["step"] += 1
            else:
                group["step"] = (
                    1
                    if not self.capturable
                    else torch.tensor([1], dtype=torch.int, device=device)
                )

            if self.capturable:
                group["lr"] = (
                    group["lr"].to(device=device)
                    if isinstance(group["lr"], torch.Tensor)
                    else torch.tensor(group["lr"], dtype=torch.float32, device=device)
                )

            # create lists for multi-tensor apply
            g_16, p_16, m_16, v_16 = [], [], [], []
            g_bf, p_bf, m_bf, v_bf = [], [], [], []
            g_32, p_32, m_32, v_32 = [], [], [], []
            p_16_master = []
            p_32_master = []
            bf16_master = []

            for p, p_master in zip(group["params"], group_master["params"]):
                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError(
                        "FusedAdam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    # Change that makes .step() not crash
                    state["exp_avg"] = torch.zeros_like(
                        distributed.get_local_tensor_if_DTensor(p).data
                    ).float()
                    # Exponential moving average of squared gradient values
                    # Change that makes .step() not crash
                    state["exp_avg_sq"] = torch.zeros_like(
                        distributed.get_local_tensor_if_DTensor(p).data
                    ).float()

                if p.dtype == torch.float16:
                    if self.master_weights:
                        p_16_master.append(
                            distributed.get_local_tensor_if_DTensor(p_master).data
                        )
                    g_16.append(distributed.get_local_tensor_if_DTensor(p.grad))
                    p_16.append(distributed.get_local_tensor_if_DTensor(p))
                    m_16.append(state["exp_avg"])
                    v_16.append(state["exp_avg_sq"])
                elif p.dtype == torch.bfloat16:
                    if self.master_weights:
                        # Change that makes .step() not crash
                        bf16_master.append(
                            distributed.get_local_tensor_if_DTensor(p_master).data
                        )
                    # Change that makes .step() not crash
                    g_bf.append(distributed.get_local_tensor_if_DTensor(p.grad))
                    # Change that makes .step() not crash
                    p_bf.append(distributed.get_local_tensor_if_DTensor(p))
                    m_bf.append(state["exp_avg"])
                    v_bf.append(state["exp_avg_sq"])
                elif p.dtype == torch.float32:
                    if self.master_weights:
                        p_32_master.append(p_master.data)
                    g_32.append(p.grad.data)
                    p_32.append(p.data)
                    m_32.append(state["exp_avg"])
                    v_32.append(state["exp_avg_sq"])
                else:
                    raise RuntimeError("FusedAdam only support fp16 and fp32.")

            # If the optimizer is capturable, then if there's a grad scaler it works
            # on the GPU + a different multi_tensor_applier should be called
            if self.capturable:
                # overflow check of gradients
                found_inf = (
                    grad_scaler._check_inf_per_device(self)[device]
                    if grad_scaler is not None
                    else torch.zeros((1,), device=device)
                )
                self._dummy_overflow_buf.copy_(found_inf)

                # get unscale scale factor
                scale, inv_scale = None, None
                if grad_scaler:
                    scale = grad_scaler._get_scale_async()
                    inv_scale = scale.double().reciprocal().float()
                else:
                    scale = torch.ones((1,), device=device, dtype=torch.float32)
                    inv_scale = torch.ones((1,), device=device, dtype=torch.float32)

                if len(g_16) > 0:
                    multi_tensor_applier(
                        (
                            self.multi_tensor_adam_capturable_master
                            if self.master_weights
                            else self.multi_tensor_adam_capturable
                        ),
                        self._dummy_overflow_buf,
                        [g_16, p_16, m_16, v_16, p_16_master]
                        if self.master_weights
                        else [g_16, p_16, m_16, v_16],
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                        inv_scale,
                    )

                if len(g_bf) > 0:
                    multi_tensor_applier(
                        (
                            self.multi_tensor_adam_capturable_master
                            if self.master_weights
                            else self.multi_tensor_adam_capturable
                        ),
                        self._dummy_overflow_buf,
                        [g_bf, p_bf, m_bf, v_bf, bf16_master]
                        if self.master_weights
                        else [g_bf, p_bf, m_bf, v_bf],
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                        inv_scale,
                    )

                if len(g_32) > 0:
                    multi_tensor_applier(
                        (
                            self.multi_tensor_adam_capturable_master
                            if self.master_weights
                            else self.multi_tensor_adam_capturable
                        ),
                        self._dummy_overflow_buf,
                        [g_32, p_32, m_32, v_32, p_32_master]
                        if self.master_weights
                        else [g_32, p_32, m_32, v_32],
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                        inv_scale,
                    )
            else:
                if len(g_16) > 0:
                    multi_tensor_applier(
                        self.multi_tensor_adam,
                        self._dummy_overflow_buf,
                        [g_16, p_16, m_16, v_16],
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                    )

                if len(g_bf) > 0:
                    multi_tensor_applier(
                        self.multi_tensor_adam,
                        self._dummy_overflow_buf,
                        [g_bf, p_bf, m_bf, v_bf],
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                    )

                if len(g_32) > 0:
                    multi_tensor_applier(
                        self.multi_tensor_adam,
                        self._dummy_overflow_buf,
                        [g_32, p_32, m_32, v_32],
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                    )

        return loss

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            if self.capturable:
                group["lr"] = (
                    group["lr"].cuda()
                    if isinstance(group["lr"], torch.Tensor)
                    else torch.tensor(group["lr"], dtype=torch.float32).cuda()
                )

            if "step" in group:
                if self.capturable:
                    if distributed.get_rank() == 0:
                        step = (
                            group["step"].cuda()
                            if isinstance(group["step"], torch.Tensor)
                            else torch.tensor([group["step"]], dtype=torch.int32).cuda()
                        )
                    else:
                        step = torch.zeros(1, dtype=torch.int32).cuda()
                    # make it compatible with FSDP optimizer
                    distributed.broadcast(step, 0)
                    group["step"] = step
                elif isinstance(group["step"], torch.Tensor):
                    group["step"] = group["step"].item()
            for p in group["params"]:
                state = self.state[p]
                if "exp_avg" in state:
                    state["exp_avg"] = state["exp_avg"].float()
                    state["exp_avg_sq"] = state["exp_avg_sq"].float()


def get_regular_param_group(net: nn.Module):
    """
    seperate the parameters of the network into two groups: decay and no_decay.
    based on nano_gpt codebase.
    """
    param_dict = {pn: p for pn, p in net.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    return decay_params, nodecay_params


def get_base_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    optim_type: str = "adamw",
    **kwargs,
) -> torch.optim.Optimizer:
    net_decay_param, net_nodecay_param = get_regular_param_group(model)

    num_decay_params = sum(p.numel() for p in net_decay_param)
    num_nodecay_params = sum(p.numel() for p in net_nodecay_param)
    net_param_total = num_decay_params + num_nodecay_params
    logger.critical(f"total num parameters : {net_param_total:,}")

    param_group = [
        {
            "params": net_decay_param + net_nodecay_param,
            "lr": lr,
            "weight_decay": weight_decay,
        },
    ]

    if optim_type == "adamw":
        opt_cls = torch.optim.AdamW
        del kwargs["master_weights"]
        del kwargs["capturable"]
    elif optim_type == "fusedadam":
        opt_cls = FusedAdam
    else:
        raise ValueError(f"Unknown optimizer type: {optim_type}")

    return opt_cls(param_group, **kwargs)


### Optimizer and Scheduler of Cosmos-Reason1 ###


def _optimizer_cls(
    params: List[nn.Parameter], optimizer_kwargs: Dict[str, Any], name: str
):
    if name == "Adam":
        # TODO: make the optimizer options configurable by toml/cmd args
        optimizer = torch.optim.Adam(params, **optimizer_kwargs)
    elif name == "AdamW":
        optimizer = torch.optim.AdamW(params, **optimizer_kwargs)
    elif name == "FusedAdam":
        optimizer = FusedAdam(
            params,
            lr=optimizer_kwargs["lr"],
            weight_decay=optimizer_kwargs["weight_decay"],
            betas=optimizer_kwargs["betas"],
            capturable=True,
            master_weights=True,
        )
    else:
        raise NotImplementedError(f"Optimizer {name} not added.")
    return optimizer


class OptimizersContainer(Stateful):
    """Util for calling step/zero_grad on multiple optimizers needed for virtual pipeline stages
    and saving/loading optimizer state_dict at checkpoint.
    """

    def __init__(
        self,
        model_parts: List[nn.Module],
        optimizer_kwargs: Dict[str, Any],
        name: str,
        lr_multiplier: list[float],
        model_part_names: list[str],
    ) -> None:
        assert len(model_parts) == len(
            lr_multiplier
        ), "lr_multiplier must have the same length as model_parts"
        self.model_parts = model_parts
        self.optimizers = [[] for _ in self.model_parts]
        self.model_part_names = model_part_names
        for model_id, model in enumerate(self.model_parts):
            optimizer_kwargs_copy = deepcopy(optimizer_kwargs)
            optimizer_kwargs_copy["lr"] *= lr_multiplier[model_id]

            if optimizer_kwargs_copy["fused"]:
                # Group the parameters by device mesh to do optimizer fusion.
                parameters_by_mesh = collections.defaultdict(list)
                for p in model.parameters():
                    if p.requires_grad:
                        device_mesh = (
                            p.device_mesh if hasattr(p, "device_mesh") else "default"
                        )
                        parameters_by_mesh[device_mesh].append(p)
                for params in parameters_by_mesh.values():
                    optimizer = _optimizer_cls(params, optimizer_kwargs_copy, name)
                    self.optimizers[model_id].append(optimizer)
            else:
                for p in model.parameters():
                    if p.requires_grad:
                        optimizer = _optimizer_cls([p], optimizer_kwargs_copy, name)
                        self.optimizers[model_id].append(optimizer)

    def __iter__(self) -> torch.optim.Optimizer:
        return iter(itertools.chain(*self.optimizers))

    def step(self) -> None:
        for optimizer in itertools.chain(*self.optimizers):
            optimizer.step()

    def zero_grad(self, set_to_none: bool = False) -> None:
        for optimizer in itertools.chain(*self.optimizers):
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        sd = {}
        for model, optimizers in zip(self.model_parts, self.optimizers):
            sd.update(
                get_optimizer_state_dict(
                    model=model,
                    optimizers=optimizers,
                    options=StateDictOptions(flatten_optimizer_state_dict=True),
                )
            )
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        for model, optimizers in zip(self.model_parts, self.optimizers):
            set_optimizer_state_dict(
                model=model,
                optimizers=optimizers,
                optim_state_dict=state_dict,
                options=StateDictOptions(flatten_optimizer_state_dict=True),
            )


class OptimizersInBackwardContainer(OptimizersContainer):
    """Optimiers in backward to skip .step() and .zero_grad()"""

    def __init__(
        self,
        model_parts: List[nn.Module],
        optimizer_kwargs: Dict[str, Any],
        name: str,
        lr_multiplier: list[float] = [1.0, 1.0, 1.0],
        model_part_names: list[str] = [],
    ) -> None:
        self.model_parts = model_parts
        self.optimizers = [None for _ in self.model_parts]
        self.model_part_names = model_part_names
        optim_dict = {}
        for model_id, model in enumerate(self.model_parts):
            optimizer_kwargs_copy = deepcopy(optimizer_kwargs)
            optimizer_kwargs_copy["lr"] *= lr_multiplier[model_id]

            for param in model.parameters():
                optim_dict[param] = _optimizer_cls([param], optimizer_kwargs_copy, name)

        def optim_hook(param) -> None:
            optim_dict[param].step()
            optim_dict[param].zero_grad()

        for model_id, model in enumerate(self.model_parts):
            for param in model.parameters():
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(optim_hook)

            self.optimizers[model_id] = [
                optim_dict[param] for param in model.parameters()
            ]

    def step(self) -> None:
        pass

    def zero_grad(self) -> None:
        pass


# consider split between PP and non-PP
def build_optimizers(
    model_parts: List[nn.Module],
    job_config: FSDP2ModelConfig,
    lr_multiplier: list[float],
    model_part_names: list[str],
) -> OptimizersContainer:
    """Wrap one optimizer per model part in an OptimizersContainer which provides a single
    step() and zero_grad() method for all the child optimizers.
    """
    assert (
        len(model_parts) == len(lr_multiplier) == len(model_part_names)
    ), "lr_multiplier and model_part_names must have the same length as model_parts"
    optim_in_bwd = job_config.optimizer.early_step_in_backward
    if optim_in_bwd and job_config.experimental.pipeline_parallel_degree > 1:
        raise NotImplementedError(
            "Optimizers in backward is not supported with pipeline parallelism."
        )
    name = job_config.optimizer.name
    lr = job_config.optimizer.lr
    fused = job_config.optimizer.fused
    optimizer_kwargs = {
        "lr": lr,
        "betas": (0.9, 0.95),
        "weight_decay": 0.1,
        "fused": fused,
        "foreach": not fused,
    }

    return (
        OptimizersContainer(
            model_parts, optimizer_kwargs, name, lr_multiplier, model_part_names
        )
        if not optim_in_bwd
        else OptimizersInBackwardContainer(
            model_parts, optimizer_kwargs, name, lr_multiplier, model_part_names
        )
    )


class SchedulersContainer(Stateful):
    """Util for calling step on multiple learning rate schedulers needed for virtual pipeline stages"""

    def __init__(self, optimizers: OptimizersContainer, lr_lambda) -> None:
        self.schedulers = []
        for optimizer in optimizers:
            self.schedulers.append(LambdaLR(optimizer, lr_lambda=lr_lambda))

    def step(self) -> None:
        for id, scheduler in enumerate(self.schedulers):
            scheduler.step()

    def state_dict(self) -> Dict[str, Any]:
        # Currently, we have one scheduler per optimizer. However, when using MultiSchedule PP or optimizer-in-backward,
        # there are multiple optimizers and schedulers, but the scheduler state_dict remains the same for all.
        # Therefore, we only save the first one and later load it for all.
        assert (
            len(self.schedulers) > 0
        ), "Must have at least one scheduler to save state_dict"
        return self.schedulers[0].state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # Load the same state_dict for all schedulers. The key value we're concerned with in scheduler.state_dict() is `last_epoch`,
        # which is an integer that will be automatically copied. As long as `training.steps` and `training.warmup_steps` remain
        # unchanged when resuming from a checkpoint, this approach is safe. We call `.copy()` here to ensure extra safety.
        last_epoch = state_dict["last_epoch"]  # Extract last known epoch
        _step_count = state_dict["_step_count"]
        logger.info(
            f"Resuming schedulers by stepping them to last_epoch: {last_epoch}; _step_count: {_step_count}"
        )

        # Manually step all schedulers to match the saved state -- this is a workaround for the inherited issue in the state dict saving (only saved the first scheduler)
        # But we have different learning rate for each scheduler, so we need to step them separately instead of loading the state dict
        # The benefit of this approach is that we can resume from a checkpoint even if the learning rate is changed
        for idx, scheduler in enumerate(self.schedulers):
            for step in range(_step_count):
                scheduler.step()  # Step forward to match previous training state
            logger.info(
                f"Scheduler {idx+1}/{len(self.schedulers)} stepped {_step_count} times."
            )
            logger.info(f"Updated learning rate: {scheduler.get_last_lr()}")

    def get_last_lr(self) -> List[float]:
        return [scheduler.get_last_lr() for scheduler in self.schedulers]


def linear_warmup_linear_decay(
    warmup_steps: int, decay_steps: int, current_step: int
) -> float:
    """Computes linear warmup followed by linear decay.
    Per LambdaLR requirement, this is accomplished by returning
    a multiplicative factor to adjust the learning rate to
    create the desired schedule.
    """
    if current_step < warmup_steps:
        # linear warmup
        # 0-indexed step, hence + 1 adjustments
        current_step += 1
        curr_adjustment = float(current_step / (warmup_steps + 1))

    else:
        # linear decay
        normalized_step = decay_steps - (current_step - warmup_steps)
        curr_adjustment = 1 - (decay_steps - normalized_step) / decay_steps

    return curr_adjustment


def linear_warmup(warmup_steps: int, current_step: int) -> float:
    """Computes linear warmup only
    Per LambdaLR requirement, this is accomplished by returning
    a multiplicative factor to adjust the learning rate to
    create the desired schedule.
    """
    if current_step < warmup_steps:
        # linear warmup
        # 0-indexed step, hence + 1 adjustments
        current_step += 1
        curr_adjustment = float(current_step / (warmup_steps + 1))
    else:
        curr_adjustment = 1

    return curr_adjustment


def linear_warmup_cosine_cooldown(
    warmup_steps: int,
    cooldown_steps: int,
    current_step: int,
    base_lr: float,
    init_lr: float,
    end_lr: float,
) -> float:
    """This scheduler will warmup the learning rate from init_lr to base_lr for warmup_steps,
    then decay the learning rate from base_lr to end_lr for cooldown_steps. After cooldown_steps + warmup_steps,
    the learning rate will be set to end_lr.
    Per LambdaLR requirement, this is accomplished by returning
    a multiplicative factor to adjust the learning rate to
    create the desired schedule.

    Args:
        warmup_steps (int): The number of steps to warmup the learning rate.
        cooldown_steps (int): The number of steps to decay the learning rate.
        current_step (int): The current step.
        base_lr (float): The base learning rate.
        init_lr (float): The initial learning rate before warmup.
        end_lr (float): The final learning rate after cooldown.

    Returns:
        float: The multiplicative factor to adjust the learning rate.
    """
    total_steps = warmup_steps + cooldown_steps

    # Normalize
    init_multiplier = init_lr / base_lr
    end_multiplier = end_lr / base_lr
    if current_step <= warmup_steps:
        progress = float(current_step / warmup_steps)
        return init_multiplier + (1.0 - init_multiplier) * progress
    elif current_step <= total_steps:
        progress = (current_step - warmup_steps) / cooldown_steps
        return end_multiplier + 0.5 * (1.0 - end_multiplier) * (
            1 + math.cos(math.pi * progress)
        )
    else:
        return end_multiplier


def build_lr_schedulers(
    optimizers: OptimizersContainer, job_config: FSDP2ModelConfig
) -> SchedulersContainer:
    warmup_steps = int(job_config.training.warmup_steps)
    decay_steps = float(max(1, job_config.training.steps - warmup_steps))
    if job_config.training.use_cosine_decay:
        lr_lambda = functools.partial(
            linear_warmup_cosine_cooldown,
            warmup_steps,
            decay_steps,
            base_lr=job_config.optimizer.lr,
            init_lr=job_config.optimizer.init_lr,  # TODO (maxzhaoshuol): This should probably be defined in scheduler instead of bundled with optimizer.
            end_lr=job_config.optimizer.end_lr,
        )
    elif job_config.training.use_linear_decay:
        lr_lambda = functools.partial(
            linear_warmup_linear_decay, warmup_steps, decay_steps
        )
    else:
        lr_lambda = functools.partial(linear_warmup, warmup_steps)

    return SchedulersContainer(optimizers, lr_lambda)
