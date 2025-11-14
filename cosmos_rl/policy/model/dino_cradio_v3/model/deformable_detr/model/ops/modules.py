"""MSDeformAttn modules."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math
import sys
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, constant_
import os

from .functions import MSDeformAttnFunction, load_ops


def _is_power_of_2(n):
    """Check if n is power of 2.

    Args:
        n (int): input

    Returns:
        Boolean on if n is power of 2 or not.
    """
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(f"invalid input for _is_power_of_2: {n} (type: {type(n)})")
    return (n & (n - 1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    """Multi-Scale Deformable Attention Module."""

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, ratio=1.0):
        """Multi-Scale Deformable Attention Constructor.

        Args:
            d_model (int): hidden dimension
            n_levels (int): number of feature levels
            n_heads (int): number of attention heads
            n_points (int): number of sampling points per attention head per feature level
            ratio (float): deformable ratio
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads, but got {} and {}".format(
                    d_model, n_heads
                )
            )
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.im2col_step = 64
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.ratio = ratio

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, int(d_model * ratio))
        self.output_proj = nn.Linear(int(d_model * ratio), d_model)

        self._reset_parameters()
        # load custom ops
        ops_dir = os.path.dirname(os.path.abspath(__file__))
        lib_name = f"MultiScaleDeformableAttention.cpython-{sys.version_info.major}{sys.version_info.minor}-{os.uname().machine}-linux-gnu.so"
        load_ops(ops_dir, lib_name)

    def _reset_parameters(self):
        """Reset parameters."""
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.n_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
        export=False,
    ):
        """Forward function.

        Args:
            query (torch.Tensor): (N, Length_{query}, C)
            reference_points (torch.Tensor): (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                             or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
            input_flatten (torch.Tensor): (N, sum_{l=0}^{L-1} H_l cdot W_l, C)
            input_spatial_shapes (torch.Tensor): (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            input_level_start_index (torch.Tensor): (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
            input_padding_mask (torch.Tensor): (N, sum_{l=0}^{L-1} H_l cdot W_l), True for padding elements, False for non-padding elements

        Returns:
            output (torch.Tensor): (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape

        # assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in, \
        #     f"{(input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum()} {Len_in}"

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(
            N, Len_in, self.n_heads, int(self.ratio * self.d_model) // self.n_heads
        )
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points
        )
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.n_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(
                    reference_points.shape[-1]
                )
            )

        input_spatial_shapes = input_spatial_shapes.long()
        input_level_start_index = input_level_start_index.long()

        if export:
            if torch.cuda.is_available() and value.is_cuda:
                output = torch.ops.nvidia.MultiscaleDeformableAttnPlugin_TRT(
                    value,
                    input_spatial_shapes,
                    input_level_start_index,
                    sampling_locations,
                    attention_weights,
                )
            else:
                # CPU implementation of multi-scale deformable attention
                # Note that this implementation uses GridSample operator which requires
                # opset version >= 16 and is much slower in TensorRT
                # warnings.warn("PyTorch native implementation of multi-scale deformable attention is being used. "
                #               "Expect slower inference performance until TensorRT further optimizes GridSample.")
                output = multi_scale_deformable_attn_pytorch(
                    value, input_spatial_shapes, sampling_locations, attention_weights
                )
        else:
            if torch.cuda.is_available() and value.is_cuda:
                # For mixed precision training
                half_float = False
                if value.dtype in [torch.float16, torch.bfloat16]:
                    half_float = value.dtype
                    value = value.float()
                    sampling_locations = sampling_locations.float()
                    attention_weights = attention_weights.float()

                output = MSDeformAttnFunction.apply(
                    value,
                    input_spatial_shapes,
                    input_level_start_index,
                    sampling_locations,
                    attention_weights,
                    self.im2col_step,
                )

                if half_float:
                    output = output.to(half_float)

            else:
                # CPU implementation of multi-scale deformable attention
                output = multi_scale_deformable_attn_pytorch(
                    value, input_spatial_shapes, sampling_locations, attention_weights
                )

        output = output.view(N, Len_q, int(self.d_model * self.ratio))
        output = self.output_proj(output)
        return output


def multi_scale_deformable_attn_pytorch(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor|List): [n_levels, 2]
        value_level_start_index (Tensor|List): [n_levels]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels, n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels, n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    bs, _, n_head, c = value.shape
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.split(split_shape, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = (
            value_list[level].flatten(2).permute(0, 2, 1).reshape(bs * n_head, c, h, w)
        )
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = (
            sampling_grids[:, :, :, level].permute(0, 2, 1, 3, 4).flatten(0, 1)
        )
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        bs * n_head, 1, Len_q, n_levels * n_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .reshape(bs, n_head * c, Len_q)
    )

    return output.permute(0, 2, 1)
