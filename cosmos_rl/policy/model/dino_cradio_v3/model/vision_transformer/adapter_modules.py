"""Adapter Modules."""

from functools import partial
import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath

from ..deformable_detr.model.ops.modules import MSDeformAttn
from .transformer_modules import AdaptivePadding, LayerScale


def get_reference_points(spatial_shapes, device):
    """Create reference points for Injector's and Extractor's MultiScaleDeformableAttention."""
    reference_points_list = []
    for H_, W_ in spatial_shapes:
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
        )
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(x, patch_size=16):
    """Create deform inputs for InteractionBlock."""
    h, w = x.shape[2:]
    spatial_shapes = torch.as_tensor(
        [
            (int(math.ceil(h / patch_size) * i), int(math.ceil(w / patch_size) * i))
            for i in [2, 1, 0.5]
        ],
        dtype=torch.long,
        device=x.device,
    )
    level_start_index = torch.cat(
        (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
    )
    reference_points = get_reference_points(
        [(math.ceil(h / patch_size), math.ceil(w / patch_size))], x.device
    )
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

    spatial_shapes = torch.as_tensor(
        [(math.ceil(h / patch_size), math.ceil(w / patch_size))],
        dtype=torch.long,
        device=x.device,
    )
    level_start_index = torch.cat(
        (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
    )
    reference_points = get_reference_points(
        [
            (int(math.ceil(h // patch_size) * i), int(math.ceil(w / patch_size) * i))
            for i in [2, 1, 0.5]
        ],
        x.device,
    )
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]

    return deform_inputs1, deform_inputs2


class ConvFFN(nn.Module):
    """An implementation of ConvFFN in ViTAdapter.

    The differences between ConvFFN & FFN:
        1. ConvFFN introduces VitAdapterDWConv to encode positional
           information.
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        """ConvFFN constructor.

        Args:
            in_features (int): The feature dimension. Same as
                `MultiheadAttention`.
            hidden_features (int): The hidden dimension of FFNs.
            out_features (int): The feature dimension. Same as
                `MultiheadAttention`.
            act_layer (nn.Module): activation layer.
            drop (float): dropout probability.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        """Forward function."""
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    """An implementation of DWConv in VitAdapter.

    The differences between DWConv & regular DWConv:
        1. Split multi stage features then apply DWConv.
    """

    def __init__(self, dim=768, kernel_size=3, stride=1, padding=1):
        """DWConv constructor.

        Args:
            dim (int): The feature dimension.
            kernel_size (int): kernel size in Conv2d.
            stride (int): stride in Conv2d
            padding (int)L padding in Conv2d
        """
        super().__init__()
        self.dwconv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
            groups=dim,
        )

    def forward(self, x, H, W):
        """Forward function."""
        B, _, C = x.shape
        split_position = [H * 2 * W * 2, H * 2 * W * 2 + H * W]
        x1 = (
            x[:, 0 : split_position[0], :]
            .transpose(1, 2)
            .view(B, C, H * 2, W * 2)
            .contiguous()
        )
        x2 = (
            x[:, split_position[0] : split_position[1], :]
            .transpose(1, 2)
            .view(B, C, H, W)
            .contiguous()
        )
        x3 = (
            x[:, split_position[1] :, :]
            .transpose(1, 2)
            .view(B, C, H // 2, W // 2)
            .contiguous()
        )
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class Extractor(nn.Module):
    """Multi Scale Feature Extractor in ViT-Adapter."""

    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        n_levels=1,
        deform_ratio=1.0,
        with_cffn=True,
        cffn_ratio=0.25,
        drop=0.0,
        drop_path=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        with_cp=False,
    ):
        """Extractor Constructor.

        Args:
            dims (int): The feature dimension.
            num_heads (int): Parallel attention heads. Defaults to 6.
            n_points (int): The number of sampling points for each query in each
                head of MultiScaleDeformableAttention. Defaults to 4.
            n_levels (int): The number of feature map used in
                Attention. Defaults to 1.
            deform_ratio (float): The expansion ratio of value_proj in DMHA.
                Defaults to 1.0.
            with_ffn (bool): The option to use ffn. If True, it use ffn.
                Default to True.
            cffn_ratio (float): The number of expansion ratio of feedforward
                network hidden layer channels. Default to 0.25.
            drop (float): Probability of an element to be zeroed
                after the feed forward layer. Defaults to 0.
            drop_path (float): stochastic depth rate. Defaults to 0.
            norm_layer (nn.Module): norm layer.
            with_cp (bool): Use checkpoint or not. Using checkpoint will save some
                memory while slowing down the training speed. Defaults to False.
        """
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(
            d_model=dim,
            n_levels=n_levels,
            n_heads=num_heads,
            n_points=n_points,
            ratio=deform_ratio,
        )
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(
                in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop
            )
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self, query, reference_points, feat, spatial_shapes, level_start_index, H, W
    ):
        """Forward function."""

        def _inner_forward(query, feat):
            """Inner forward function."""
            attn = self.attn(
                self.query_norm(query),
                reference_points,
                self.feat_norm(feat),
                spatial_shapes,
                level_start_index,
                None,
            )
            query = query + attn

            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
            return query

        if self.with_cp and query.requires_grad and not torch.onnx.is_in_onnx_export():
            query = checkpoint.checkpoint(
                _inner_forward, query, feat, use_reentrant=True
            )
        else:
            query = _inner_forward(query, feat)

        return query


class Injector(nn.Module):
    """Injector in ViT-Adapter."""

    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        n_levels=1,
        deform_ratio=1.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.0,
        with_cp=False,
    ):
        """Injector Constructor

        Args:
            dims (int): The feature dimension.
            num_heads (int): Parallel attention heads. Defaults to 6.
            n_points (int): The number of sampling points for each query in each
                head of MultiScaleDeformableAttention. Defaults to 4.
            n_levels (int): The number of feature map used in
                Attention. Defaults to 1.
            deform_ratio (float): The expansion ratio of value_proj in DMHA.
                Defaults to 1.0.
            norm_layer (nn.Module): norm layer.
            init_values (float): initial value in LayerScale. If set to 0, LayerScale
                is not applied. Defaults to 0.0.
            with_cp (bool): Use checkpoint or not. Using checkpoint will save some
                memory while slowing down the training speed. Defaults to False.
        """
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(
            d_model=dim,
            n_levels=n_levels,
            n_heads=num_heads,
            n_points=n_points,
            ratio=deform_ratio,
        )

        self.ls = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):
        """Forward function."""

        def _inner_forward(query, feat):
            """Inner forward function."""
            attn = self.attn(
                self.query_norm(query),
                reference_points,
                self.feat_norm(feat),
                spatial_shapes,
                level_start_index,
                None,
            )
            return query + self.ls(attn)

        if self.with_cp and query.requires_grad and not torch.onnx.is_in_onnx_export():
            query = checkpoint.checkpoint(
                _inner_forward, query, feat, use_reentrant=True
            )
        else:
            query = _inner_forward(query, feat)

        return query


class InteractionBlock(nn.Module):
    """InteractionBlock in ViT-Adapter."""

    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop=0.0,
        drop_path=0.0,
        with_cffn=True,
        cffn_ratio=0.25,
        init_values=0.0,
        deform_ratio=1.0,
        extra_extractor=False,
        with_cp=False,
    ):
        """InteractionBlock Constructor

        Args:
            dims (int): The feature dimension.
            num_heads (int): Parallel attention heads. Defaults to 6.
            n_points (int): The number of sampling points for each query in each
                head of MultiScaleDeformableAttention. Defaults to 4.
            norm_layer (nn.Module): norm layer.
            drop (float): Probability of an element to be zeroed
                after the feed forward layer. Defaults to 0.
            drop_path (float): stochastic depth rate. Defaults to 0.
            with_ffn (bool): The option to use ffn. If True, it use ffn.
                Default to True.
            cffn_ratio (float): The number of expansion ratio of feedforward
                network hidden layer channels. Default to 0.25.
            init_values (float): initial value in LayerScale. If set to 0, LayerScale
                is not applied. Defaults to 0.0.
            deform_ratio (float): The expansion ratio of value_proj in DMHA.
                Defaults to 1.0.
            extra_extractor (bool): The option to use extra Extractor in
                InteractionBlock. If True, it use extra Extractor.
                Default to False.
            with_cp (bool): Use checkpoint or not. Using checkpoint will save some
                memory while slowing down the training speed. Defaults to False.
        """
        super().__init__()

        self.injector = Injector(
            dim=dim,
            n_levels=3,
            num_heads=num_heads,
            init_values=init_values,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
            with_cp=with_cp,
        )
        self.extractor = Extractor(
            dim=dim,
            n_levels=1,
            num_heads=num_heads,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            drop=drop,
            drop_path=drop_path,
            with_cp=with_cp,
        )
        if extra_extractor:
            self.extra_extractors = nn.Sequential(
                *[
                    Extractor(
                        dim=dim,
                        num_heads=num_heads,
                        n_points=n_points,
                        norm_layer=norm_layer,
                        with_cffn=with_cffn,
                        cffn_ratio=cffn_ratio,
                        deform_ratio=deform_ratio,
                        drop=drop,
                        drop_path=drop_path,
                        with_cp=with_cp,
                    )
                    for _ in range(2)
                ]
            )
        else:
            self.extra_extractors = None

    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W):
        """Forward function."""
        x = self.injector(
            query=x,
            reference_points=deform_inputs1[0],
            feat=c,
            spatial_shapes=deform_inputs1[1],
            level_start_index=deform_inputs1[2],
        )

        for blk in blocks:
            x = blk(x)

        c = self.extractor(
            query=c,
            reference_points=deform_inputs2[0],
            feat=x,
            spatial_shapes=deform_inputs2[1],
            level_start_index=deform_inputs2[2],
            H=H,
            W=W,
        )
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(
                    query=c,
                    reference_points=deform_inputs2[0],
                    feat=x,
                    spatial_shapes=deform_inputs2[1],
                    level_start_index=deform_inputs2[2],
                    H=H,
                    W=W,
                )
        return x, c


class SpatialPriorModule(nn.Module):
    """SpatialPriorModule in ViT-Adapter."""

    def __init__(
        self,
        in_channel,
        patch_size,
        embed_dim,
        inplanes=64,
        out_indices=[0, 1, 2, 3],
        padding="corner",
    ):
        """SpatialPriorModule Constructor.

        Args:
            in_channel (int): channel size of input.
            patch_size (int): The patch size in patch embedding.
            embed_dims (int): The feature dimension.
            inplanes (int): Hidden dimension. Defaults to 64.
            out_indices (list): List of block indices to return as feature.
            padding (str): Support "same" and "corner", "corner" mode
                would pad zero to bottom right, and "same" mode would
                pad zero around input. Default to "corner".
        """
        super().__init__()
        self.out_indices = out_indices
        self.adaptive_padding = AdaptivePadding(
            kernel_size=patch_size, stride=patch_size, padding=padding
        )

        self.stem = nn.Sequential(
            *[
                nn.Conv2d(
                    in_channel, inplanes, kernel_size=3, stride=2, padding=1, bias=False
                ),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False
                ),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False
                ),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )
        self.conv2 = nn.Sequential(
            *[
                nn.Conv2d(
                    inplanes,
                    2 * inplanes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.SyncBatchNorm(2 * inplanes),
                nn.ReLU(inplace=True),
            ]
        )
        self.conv3 = nn.Sequential(
            *[
                nn.Conv2d(
                    2 * inplanes,
                    4 * inplanes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.SyncBatchNorm(4 * inplanes),
                nn.ReLU(inplace=True),
            ]
        )
        self.conv4 = nn.Sequential(
            *[
                nn.Conv2d(
                    4 * inplanes,
                    4 * inplanes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.SyncBatchNorm(4 * inplanes),
                nn.ReLU(inplace=True),
            ]
        )
        if len(out_indices) == 4:
            self.fc1 = nn.Conv2d(
                inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True
            )
            self.fc2 = nn.Conv2d(
                2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True
            )
            self.fc3 = nn.Conv2d(
                4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True
            )
            self.fc4 = nn.Conv2d(
                4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True
            )
        else:
            self.fc2 = nn.Conv2d(
                2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True
            )
            self.fc3 = nn.Conv2d(
                4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True
            )
            self.fc4 = nn.Conv2d(
                4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True
            )

    def forward(self, x):
        """Forward function."""
        x = self.adaptive_padding(x)
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)

        if len(self.out_indices) == 4:
            c1 = self.fc1(c1)

        c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        c4 = self.fc4(c4)
        bs, dim, _, _ = c2.shape

        c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
        c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
        c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

        return c1, c2, c3, c4
