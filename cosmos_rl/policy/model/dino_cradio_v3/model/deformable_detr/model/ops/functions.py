"""MSDeformAttnFunction modules."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import os


def load_ops(ops_dir, lib_name):
    """Load C++ Ops to PyTorch.

    Args:
        ops_dir (str): Path to the C++ src code directory.
        lib_name (str): Name of the library to load.
    """
    module_path = os.path.join(ops_dir, lib_name)
    torch.ops.load_library(module_path)


class MSDeformAttnFunction(Function):
    """MSDeformAttnFunction"""

    @staticmethod
    def forward(
        ctx,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        """Forward function.

        Args:
            value (torch.Tensor): The value has shape
                (bs, num_keys, mum_heads, embed_dims//num_heads)
            value_spatial_shapes (torch.Tensor): Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            sampling_locations (torch.Tensor): The location of sampling points,
                has shape
                (bs ,num_queries, num_heads, num_levels, num_points, 2),
                the last dimension 2 represent (x, y).
            attention_weights (torch.Tensor): The weight of sampling points
                used when calculate the attention, has shape
                (bs ,num_queries, num_heads, num_levels, num_points),
            im2col_step (torch.Tensor): The step used in image to column.

        Returns:
            torch.Tensor: has shape (bs, num_queries, embed_dims)

        """
        ctx.im2col_step = im2col_step
        output = torch.ops.nvidia.MultiscaleDeformableAttnPlugin_TRT(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        )
        ctx.save_for_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """Backward function.

        Args:
            grad_output (torch.Tensor): Gradient of output tensor of forward.

        Returns:
            tuple[Tensor]: Gradient of input tensors in forward.
        """
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        ) = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = (
            torch.ops.nvidia.DMHA_backward(
                value,
                value_spatial_shapes,
                value_level_start_index,
                sampling_locations,
                attention_weights,
                grad_output,
                ctx.im2col_step,
            )
        )

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None
