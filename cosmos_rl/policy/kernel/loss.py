from typing import Optional
from enum import Enum

import torch
from torch.nn import functional as F

from cosmos_rl.policy.config import Config as CosmosConfig


def liger_cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction

    loss, _, _ = LigerCrossEntropyFunction.apply(
        input,
        target,
        None,
        ignore_index,
        0.0,  # lse_square_scale
        label_smoothing,
        reduction,
        None,  # softcap
        False,  # return_z_loss
        False,  # return_token_accuracy
    )
    return loss


def fused_linear_cross_entropy(
    lin_weight: torch.Tensor,
    input: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    from liger_kernel.ops.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyFunction,
    )

    loss, _, _ = LigerFusedLinearCrossEntropyFunction.apply(
        input,
        lin_weight,  # linear weight
        target,
        None,  # bias
        None,  # ce_weight
        ignore_index,
        0.0,  # lse_square_scale
        label_smoothing,
        reduction,
        None,  # softcap
        False,  # return_z_loss
        torch.float32,  # accum_dtype
        False,  # use_token_scaling
        False,  # return_token_accuracy
    )

    return loss


class CrossEntropyType(Enum):
    LIGER_KERNEL = 1
    LIGER_FUSED_CROSS_ENTROPY = 2
    TORCH_CROSS_ENTROPY = 3


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, config: Optional[CosmosConfig] = None):
        super().__init__()

        # check if liger installed
        try:
            import liger_kernel  # noqa: F401

            liger_installed = True
        except ImportError:
            liger_installed = False

        if (
            config is None
            or not config.policy.enable_liger_kernel
            or not liger_installed
        ):
            self.ce_type = CrossEntropyType.TORCH_CROSS_ENTROPY
            self.ce_impl = F.cross_entropy
        else:
            if config.policy.enable_liger_cross_entropy:
                self.ce_type = CrossEntropyType.LIGER_KERNEL
                self.ce_impl = liger_cross_entropy
            elif config.policy.enable_liger_fused_cross_entropy:
                self.ce_type = CrossEntropyType.LIGER_FUSED_CROSS_ENTROPY
                self.ce_impl = fused_linear_cross_entropy
            else:
                raise ValueError(
                    "Either enable_liger_cross_entropy or enable_liger_fused_cross_entropy must be True."
                )

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        lin_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if lin_weight is not None:
            return self.ce_impl(
                lin_weight,
                input,
                target,
                ignore_index=ignore_index,
                reduction=reduction,
                label_smoothing=label_smoothing,
            )
        else:
            if self.ce_type == CrossEntropyType.TORCH_CROSS_ENTROPY:
                # passed in input if in bfloat16, cast to float32 to avoid precision issues
                # only for F.cross_entropy
                input = input.float()

            return self.ce_impl(
                input,
                target,
                ignore_index=ignore_index,
                reduction=reduction,
                label_smoothing=label_smoothing,
            )
