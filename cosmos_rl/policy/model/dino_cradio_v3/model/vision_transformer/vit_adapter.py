"""ViT Adatper backbone."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_

from timm.layers import trunc_normal_

from ..backbone_v2.radio import RADIOBase

from ..deformable_detr.model.ops.modules import MSDeformAttn
from .adapter_modules import SpatialPriorModule, InteractionBlock, deform_inputs


class CRADIOAdapter(RADIOBase):
    """ViT-Adapter from https://arxiv.org/abs/2205.08534."""

    def __init__(
        self,
        model_name=None,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=6,
        init_values=0.0,
        interaction_indexes=None,
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        add_vit_feature=True,
        use_extra_extractor=True,
        out_indices=[0, 1, 2, 3],
        activation_checkpoint=False,
        drop_path_rate=0.4,
        add_summary=True,
        **kwargs,
    ):
        """ViT-Adapter Constructor.

        Args:
            model_name (str): CRADIO model name.
            conv_inplane (int): The hidden dimension of Conv2D in SPM.
            n_points (int): The number of sampling points for
                each query in each head of MultiScaleDeformableAttention.
            deform_num_heads (int): Parallel attention heads of MultiScaleDeformableAttention.
            init_values (float): Init value of LayerScale.
            interaction_indexes (list): The indexes of each interaction block.
            with_cffn (bool): The option to use ffn for adapter. If True, it use ffn.
            cffn_ratio (float): The number of expansion ratio of feedforward
                network hidden layer channels of adapter.
            deform_ratio (float): The expansion ratio of value_proj.
            add_vit_feature (bool): The option to add vit feature to adapter
                feature. If True, it add vit feature.
            use_extra_extractor (bool): The option to use extra Extractor in
                InteractionBlock. If True, it use extra Extractor.
            out_indices (list): List of block indices to return as feature.
            activation_checkpoint (bool): Use activation checkpoint or not.
            add_summary (bool): Use summary token of backbone or not.
        """
        super().__init__(backbone=model_name)

        self.num_block = len(self.model.blocks)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        self.add_summary = add_summary
        self.num_summary = self.num_summary_tokens
        self.drop_path_rate = drop_path_rate
        self.embed_dim = self.model.embed_dim
        self.depth = self.model_cfg.get("depth", 12)

        self.level_embed = nn.Parameter(torch.zeros(3, self.embed_dim))
        self.spm = SpatialPriorModule(
            in_channel=3,
            patch_size=self.model.patch_generator.patch_size,
            inplanes=conv_inplane,
            embed_dim=self.embed_dim,
            out_indices=out_indices,
        )
        self.interactions = nn.Sequential(
            *[
                InteractionBlock(
                    dim=self.embed_dim,
                    num_heads=deform_num_heads,
                    n_points=n_points,
                    init_values=init_values,
                    drop_path=self.drop_path_rate,  # drop_path is 0.4
                    norm_layer=nn.LayerNorm,
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    deform_ratio=deform_ratio,
                    extra_extractor=(
                        (i == len(interaction_indexes) - 1) and use_extra_extractor
                    ),
                    with_cp=activation_checkpoint,
                )
                for i in range(len(interaction_indexes))
            ]
        )
        if 0 in out_indices:
            self.up = nn.ConvTranspose2d(self.embed_dim, self.embed_dim, 2, 2)
            self.up.apply(self._init_weights)
        else:
            self.up = None

        self.out_indices = out_indices

        for i_layer in self.out_indices:
            layer = nn.SyncBatchNorm(self.embed_dim)
            layer_name = f"out_norm{i_layer}"
            self.add_module(layer_name, layer)

        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

        if self.add_summary:
            self.fc_summary = nn.Linear(
                self.num_summary * self.embed_dim, self.embed_dim
            )
            if len(self.out_indices) == 4:
                self.conv1 = nn.Conv2d(2 * self.embed_dim, self.embed_dim, 1)
                self.conv1.apply(self._init_weights)
            else:
                self.conv1 = None
            self.conv2 = nn.Conv2d(2 * self.embed_dim, self.embed_dim, 1)
            self.conv2.apply(self._init_weights)
            self.conv3 = nn.Conv2d(2 * self.embed_dim, self.embed_dim, 1)
            self.conv3.apply(self._init_weights)
            self.conv4 = nn.Conv2d(2 * self.embed_dim, self.embed_dim, 1)
            self.conv4.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        """Forward function."""
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        patch_size = self.model.patch_generator.patch_size
        H, W = tuple(d // patch_size for d in x.shape[-2:])

        # Obtain patch + summary tokens from CPE patch generator
        x = self.model.patch_generator(x)
        bs = x.shape[0]

        # Separate summary tokens (class + register) from patch tokens to
        # ensure the deformable attention only processes spatial tokens.
        if self.add_summary:
            summary_tokens = x[:, : self.num_summary, :]
            x = x[:, self.num_summary :, :]

        dim = self.model.embed_dim

        # Interaction
        outs = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(
                x,
                c,
                self.model.blocks[indexes[0] : indexes[-1] + 1],
                deform_inputs1,
                deform_inputs2,
                H,
                W,
            )

        # Split & Reshape
        c2 = c[:, 0 : c2.size(1), :]
        c3 = c[:, c2.size(1) : c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1) :, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        if self.up is not None:
            c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x3 = x.transpose(1, 2).view(bs, dim, H, W).contiguous()
            x1 = F.interpolate(x3, scale_factor=4, mode="bilinear", align_corners=False)
            x2 = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
            x4 = F.interpolate(
                x3, scale_factor=0.5, mode="bilinear", align_corners=False
            )
            if len(self.out_indices) == 4:
                c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4
            else:
                c2, c3, c4 = c2 + x2, c3 + x3, c4 + x4

        if self.add_summary:
            summary = summary_tokens.view(bs, -1)
            summary = self.fc_summary(summary)
            summary = summary.unsqueeze(2).unsqueeze(3)
            if self.conv1 is not None:
                c1 = torch.cat(
                    [summary.expand(-1, -1, c1.shape[2], c1.shape[3]), c1], dim=1
                )
            c2 = torch.cat(
                [summary.expand(-1, -1, c2.shape[2], c2.shape[3]), c2], dim=1
            )
            c3 = torch.cat(
                [summary.expand(-1, -1, c3.shape[2], c3.shape[3]), c3], dim=1
            )
            c4 = torch.cat(
                [summary.expand(-1, -1, c4.shape[2], c4.shape[3]), c4], dim=1
            )
            if self.conv1 is not None:
                c1 = self.conv1(c1)
            c2 = self.conv2(c2)
            c3 = self.conv3(c3)
            c4 = self.conv4(c4)

        outs = {}
        # Final Norm
        out_features = [c1, c2, c3, c4]
        for idx in self.out_indices:
            level = out_features[idx]
            norm_layer = getattr(self, f"out_norm{idx}")
            outs[f"p{idx}"] = norm_layer(level)
        return outs


def vit_base_cradiov3(
    out_indices=[0, 1, 2, 3],
    activation_checkpoint=False,
    use_summary_token=True,
    **kwargs,
):
    """ViT-Base C-RADIO-v3 model.

    Args:
        out_indices (list): List of block indices to return as feature.
        activation_checkpoint (bool): flag to indicate if activation checkpoint is used.
        use_summary_token (bool): Whether to use summary_token of backbone.

    Return:
        model: CRADIOAdapter model.
    """
    model = CRADIOAdapter(
        model_name="vit_base_patch14_reg4_dinov2",
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        drop_path_rate=0.4,
        init_values=1e-5,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        interaction_indexes=[[0, 3], [4, 7], [8, 11], [12, 15]],
        out_indices=out_indices,
        activation_checkpoint=activation_checkpoint,
        add_summary=use_summary_token,
        **kwargs,
    )

    return model


def vit_large_cradiov3(
    out_indices=[0, 1, 2, 3],
    activation_checkpoint=False,
    use_summary_token=True,
    **kwargs,
):
    """ViT-Large C-RADIO-v3 model.

    Args:
        out_indices (list): List of block indices to return as feature.
        activation_checkpoint (bool): flag to indicate if activation checkpoint is used.
        use_summary_token (bool): Whether to use summary_token of backbone.

    Return:
        model: CRADIOAdapter model.
    """
    model = CRADIOAdapter(
        model_name="vit_large_patch14_reg4_dinov2",
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        drop_path_rate=0.3,
        init_values=1e-5,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        out_indices=out_indices,
        activation_checkpoint=activation_checkpoint,
        add_summary=use_summary_token,
        **kwargs,
    )

    return model


def vit_huge_cradiov3(
    out_indices=[0, 1, 2, 3],
    activation_checkpoint=False,
    use_summary_token=True,
    **kwargs,
):
    """ViT-Huge C-RADIO-v3 model.

    Args:
        out_indices (list): List of block indices to return as feature.
        activation_checkpoint (bool): flag to indicate if activation checkpoint is used.
        use_summary_token (bool): Whether to use summary_token of backbone.

    Return:
        model: CRADIOAdapter model.
    """
    model = CRADIOAdapter(
        model_name="vit_huge_patch14_reg4_dinov2",
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        drop_path_rate=0.3,
        init_values=1e-5,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        interaction_indexes=[[0, 7], [8, 15], [16, 23], [24, 31]],
        out_indices=out_indices,
        activation_checkpoint=activation_checkpoint,
        add_summary=use_summary_token,
        **kwargs,
    )

    return model


def vit_giant_cradiov3(
    out_indices=[0, 1, 2, 3],
    activation_checkpoint=False,
    use_summary_token=True,
    **kwargs,
):
    """ViT-Giant C-RADIO-v3 model.

    Args:
        out_indices (list): List of block indices to return as feature.
        activation_checkpoint (bool): flag to indicate if activation checkpoint is used.
        use_summary_token (bool): Whether to use summary_token of backbone.

    Return:
        model: CRADIOAdapter model.
    """
    model = CRADIOAdapter(
        model_name="vit_giant_patch14_reg4_dinov2",
        patch_size=16,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        drop_path_rate=0.3,
        init_values=1e-5,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        interaction_indexes=[[0, 9], [10, 19], [20, 29], [30, 39]],
        out_indices=out_indices,
        activation_checkpoint=activation_checkpoint,
        add_summary=use_summary_token,
        **kwargs,
    )

    return model


vit_model_dict = {
    "vit_base_cradiov3": vit_base_cradiov3,
    "vit_large_cradiov3": vit_large_cradiov3,
    "vit_huge_cradiov3": vit_huge_cradiov3,
    "vit_giant_cradiov3": vit_giant_cradiov3,
}
