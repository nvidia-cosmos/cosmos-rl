# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""RADIO backbone."""

import math
import types
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models import checkpoint_seq
from timm.models.vision_transformer import VisionTransformer


radio_model_cfg = {
    # CRADIOV3.
    "vit_base_patch14_reg4_dinov2": {
        "img_size": 518 * 16 // 14,
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "init_values": 1e-5,
        "reg_tokens": 4,
        "no_embed_class": True,
    },
    "vit_large_patch14_reg4_dinov2": {
        "img_size": 518 * 16 // 14,
        "patch_size": 16,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "init_values": 1e-5,
        "reg_tokens": 4,
        "no_embed_class": True,
    },
    "vit_huge_patch14_reg4_dinov2": {
        "img_size": 518 * 16 // 14,
        "patch_size": 16,
        "embed_dim": 1280,
        "depth": 32,
        "num_heads": 16,
        "init_values": 1e-5,
        "reg_tokens": 4,
        "no_embed_class": True,
    },
    "vit_giant_patch14_reg4_dinov2": {
        "img_size": 518 * 16 // 14,
        "patch_size": 16,
        "embed_dim": 1536,
        "depth": 40,
        "num_heads": 24,
        "init_values": 1e-5,
        "reg_tokens": 4,
        "no_embed_class": True,
    },
}


class Im2Patches(nn.Module):
    """Image patches module."""

    def __init__(self, patch_size: int):
        """Image patches module."""
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function to transform input image to patches."""
        if self.patch_size == 1:
            patches = x.flatten(2)
            patches = patches.permute(0, 2, 1)
            return patches

        py = x.shape[-2] // self.patch_size
        px = x.shape[-1] // self.patch_size
        patches = rearrange(
            x,
            "b c (py yy) (px xx) -> b (py px) (c yy xx)",
            py=py,
            yy=self.patch_size,
            px=px,
            xx=self.patch_size,
        )
        return patches


class ViTPatchLinear(nn.Linear):
    """Patch embedding module."""

    def __init__(self, patch_size: int, embed_dim: int, **kwargs):
        """Patch embedding module.

        Args:
            patch_size (int): patch size.
            embed_dim (int): patch embedding dimension.
        """
        super().__init__(3 * (patch_size**2), embed_dim, bias=False, **kwargs)
        self.patch_size = patch_size

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if self.bias is not None:
            self.bias.data.copy_(state_dict[f"{prefix}bias"])
        chk_weight = state_dict[f"{prefix}weight"]
        if chk_weight.shape != self.weight.shape:
            src_patch_size = int(math.sqrt(chk_weight.shape[1] // 3))
            assert (src_patch_size**2) * 3 == chk_weight.shape[
                1
            ], "Unable to interpolate non-square patch size"
            chk_weight = rearrange(
                chk_weight,
                "b (c h w) -> b c h w",
                c=3,
                h=src_patch_size,
                w=src_patch_size,
            )
            chk_weight = F.interpolate(
                chk_weight,
                size=(self.patch_size, self.patch_size),
                mode="bicubic",
                align_corners=True,
                antialias=False,
            )
            chk_weight = rearrange(chk_weight, "b c h w -> b (c h w)")
        self.weight.data.copy_(chk_weight)


class ClsToken(nn.Module):
    """Class token class for RADIO ViT backbone."""

    def __init__(
        self,
        ndim: int,
        num_tokens: int = 1,
        enabled: bool = True,
        register_multiple: int = 0,
    ):
        """Class token class for RADIO ViT backbone.

        Args:
            ndim (int): class token dimension.
            num_tokens (int): number of class tokens. Default: `1`.
            enabled (bool): whether to enable class token. Default: `True`.
            register_multiple (int): number of extra tokens. Default: `0`.
        """
        super().__init__()
        self.ndim = ndim
        self.enabled = enabled
        self.num_registers = 0
        self.num_tokens = num_tokens
        if enabled:
            if register_multiple > 0:
                self.num_registers = register_multiple - (
                    num_tokens % register_multiple
                )

            scale = ndim**-0.5
            self.token = nn.Parameter(
                torch.randn(num_tokens + self.num_registers, ndim) * scale
            )
        else:
            self.token = None
        self.num_patches = self.num_tokens + self.num_registers

    def no_weight_decay(self):
        """No weight decay."""
        return ["token"]

    def disable(self):
        """Disable ClsToken."""
        self.token = None
        self.enabled = False

    def forward(self, x: torch.Tensor):
        """Forward function and return the input with class tokens concatenated"""
        if self.token is None:
            return x
        token = self.token.unsqueeze(0).expand(x.shape[0], -1, -1)
        return torch.cat([token, x], dim=1)


class ViTPatchGenerator(nn.Module):
    """ViT patch generator class for CPE (Cropped Position Embedding) purpose"""

    def __init__(
        self,
        patch_size: int,
        embed_dim: int,
        input_dims: Union[int, Tuple[int, int]],
        abs_pos: bool = True,
        normalize_patches: bool = False,
        cls_token: bool = False,
        max_input_dims: Optional[Union[int, Tuple[int, int]]] = None,
        pos_dropout: float = 0.0,
        return_pos_enc: bool = False,
        num_cls_tokens: int = 1,
        register_multiple: int = 0,
        device=None,
        dtype=None,
    ):
        """ViT patch generator class for CPE (Cropped Position Embedding) purpose

        Args:
            patch_size (int): patch size
            embed_dim (int): patch embedding dimension
            input_dims (Union[int, Tuple[int, int]]): input dimension
            abs_pos (bool): absolute position embedding. Default: `True`.
            normalize_patches (bool): whether to normalize patch. Default: `False`.
            cls_token (bool): whether to enable class token. Default: `False`.
            max_input_dims (Optional[Union[int, Tuple[int, int]]]): maximum input dimension. Default: `None`.
            pos_dropout (float): position embedding dropout rate. Default: `0.0`.
            return_pos_enc (bool): whether to return position embedding. Default: `False`.
            num_cls_tokens (int): number of class tokens. Default: `1`.
            register_multiple (int): number of extra tokens. Default: `0`.
            device (str): device. Default: `None`.
            dtype (str): data type. Default: `None`.
        """
        super().__init__()

        if isinstance(input_dims, int):
            input_dims = (input_dims, input_dims)
        if max_input_dims is None:
            max_input_dims = input_dims
        if isinstance(max_input_dims, int):
            max_input_dims = (max_input_dims, max_input_dims)
        max_input_dims = tuple(
            int(math.ceil(d / patch_size) * patch_size) for d in max_input_dims
        )

        self.cpe_mode = max_input_dims != input_dims
        self.pos_dropout = pos_dropout
        self.return_pos_enc = return_pos_enc
        self.patch_size = patch_size
        self.abs_pos = abs_pos
        self.embed_dim = embed_dim
        self.num_rows = max_input_dims[0] // patch_size
        self.num_cols = max_input_dims[1] // patch_size
        self.input_dims = tuple(d // patch_size for d in input_dims)
        self.num_patches = self.num_rows * self.num_cols
        self.max_input_dims = max_input_dims

        self.im_to_patches = Im2Patches(patch_size)
        self.embedder = ViTPatchLinear(
            patch_size, embed_dim, device=device, dtype=dtype
        )
        if abs_pos:
            scale = embed_dim**-0.5
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.num_patches, embed_dim, device=device, dtype=dtype)
                * scale
            )
        self.cls_token = ClsToken(
            embed_dim,
            num_tokens=num_cls_tokens,
            enabled=cls_token,
            register_multiple=register_multiple,
        )
        self.patch_normalizer = (
            nn.LayerNorm(embed_dim) if normalize_patches else nn.Identity()
        )

    def forward(self, x: torch.Tensor):
        """Forward function to return the patch embeddings with position embedding applied."""
        patches = self.embed_patches(x)
        patches, pos_enc = self.apply_pos_enc(patches, input_size=x.shape[2:])
        patches = self.cls_token(patches)
        patches = self.patch_normalizer(patches)
        if self.return_pos_enc:
            return patches, pos_enc
        return patches

    @property
    def num_cls_tokens(self):
        """Number of class tokens"""
        return self.cls_token.num_tokens

    @property
    def num_registers(self):
        """Number of register tokens"""
        return self.cls_token.num_registers

    @property
    def num_skip(self):
        """Total number of extra tokens (class tokens + register tokens)"""
        return self.num_cls_tokens + self.num_registers

    def no_weight_decay(self):
        """No weight decay."""
        return ["pos_embed"]

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if self.abs_pos:
            self._load_embed(state_dict[f"{prefix}pos_embed"], self.pos_embed)

    def _load_embed(self, src_embed: torch.Tensor, targ_embed: nn.Parameter):
        if src_embed.shape != targ_embed.shape:
            src_size = int(math.sqrt(src_embed.shape[1]))
            assert (
                src_size**2 == src_embed.shape[1]
            ), "Unable to interpolate non-square embedding"
            src_embed = rearrange(
                src_embed, "b (h w) c -> b c h w", h=src_size, w=src_size
            )
            src_embed = F.interpolate(
                src_embed,
                size=(self.num_rows, self.num_cols),
                mode="bicubic",
                align_corners=True,
                antialias=False,
            )
            src_embed = rearrange(src_embed, "b c h w -> b (h w) c")
        targ_embed.data.copy_(src_embed)

    def _load_projection(
        self, src_proj_weight: torch.Tensor, targ_proj_weight: torch.Tensor
    ):
        if src_proj_weight.shape != targ_proj_weight.shape:
            src_patch_size = int(math.sqrt(src_proj_weight.shape[1] // 3))
            assert (src_patch_size**2) * 3 == src_proj_weight.shape[
                1
            ], "Unable to interpolate non-square patch size"
            src_proj_weight = rearrange(
                src_proj_weight,
                "b (c h w) -> b c h w",
                c=3,
                h=src_patch_size,
                w=src_patch_size,
            )
            src_proj_weight = F.interpolate(
                src_proj_weight,
                size=(self.patch_size, self.patch_size),
                mode="bicubic",
                align_corners=True,
                antialias=False,
            )
            src_proj_weight = rearrange(src_proj_weight, "b c h w -> b (c h w)")
        targ_proj_weight.data.copy_(src_proj_weight)

    def embed_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Embed image patches"""
        patches = self.im_to_patches(x)
        return self.embedder(patches)

    def apply_pos_enc(
        self,
        patches: torch.Tensor,
        patch_idxs: Optional[torch.Tensor] = None,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        """Apply position embedding"""
        if not self.abs_pos:
            return patches

        pos_enc = self.get_pos_enc(patches.shape[0], patch_idxs, input_size)
        if self.training and self.pos_dropout > 0:
            keeps = (
                torch.rand(
                    patches.shape[0], 1, 1, dtype=pos_enc.dtype, device=pos_enc.device
                )
                > self.pos_dropout
            )
            pos_enc_drop = torch.where(keeps, pos_enc, 0)
        else:
            pos_enc_drop = pos_enc
        return patches + pos_enc_drop, pos_enc

    def get_pos_enc(
        self,
        batch_size: int,
        patch_idxs: Optional[torch.Tensor] = None,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Get position embedding"""
        if input_size is None:
            input_dims = self.input_dims
        else:
            input_dims = tuple(d // self.patch_size for d in input_size)
        pos_embed = self._get_pos_embeddings(batch_size, input_dims)
        if patch_idxs is None:
            return pos_embed
        exp_patch_idxs = patch_idxs.unsqueeze(-1).expand(-1, -1, pos_embed.shape[-1])
        return torch.gather(
            pos_embed.expand(patch_idxs.shape[0], -1, -1), dim=1, index=exp_patch_idxs
        )

    def _get_pos_embeddings(self, batch_size: int, input_dims: Tuple[int, int]):
        if (self.num_rows, self.num_cols) == input_dims:
            return self.pos_embed
        pos_embed = self.pos_embed.reshape(1, self.num_rows, self.num_cols, -1).permute(
            0, 3, 1, 2
        )

        def window_select(pos_embed):
            if input_dims[0] < pos_embed.shape[-2]:
                pos_embed = pos_embed[..., : input_dims[0], :]
            if input_dims[1] < pos_embed.shape[-1]:
                pos_embed = pos_embed[..., :, : input_dims[1]]
            return pos_embed

        if self.cpe_mode:
            if self.training:
                min_scale = math.sqrt(0.1)
                scale = (
                    torch.rand(batch_size, 1, 1, device=pos_embed.device)
                    * (1 - min_scale)
                    + min_scale
                )
                aspect_min = math.log(3 / 4)
                aspect_max = -aspect_min
                aspect = torch.exp(
                    torch.rand(batch_size, 1, 1, device=pos_embed.device)
                    * (aspect_max - aspect_min)
                    + aspect_min
                )
                scale_x = scale * aspect
                scale_y = scale * (1 / aspect)
                scale_xy = torch.stack([scale_x, scale_y], dim=-1).clamp_(0, 1)
                pos_xy = torch.rand(batch_size, 1, 1, 2, device=pos_embed.device) * (
                    1 - scale_xy
                )
                lin_x = torch.linspace(
                    0, 1, steps=input_dims[1], device=pos_embed.device
                )[None, None].expand(batch_size, input_dims[0], -1)
                lin_y = torch.linspace(
                    0, 1, steps=input_dims[0], device=pos_embed.device
                )[None, :, None].expand(batch_size, -1, input_dims[1])
                lin_xy = torch.stack([lin_x, lin_y], dim=-1)
                grid_xy = lin_xy * scale_xy + pos_xy
                grid_xy.mul_(2).sub_(1)  # Convert to [-1, 1] range.
                pos_embed = F.grid_sample(
                    pos_embed.float().expand(batch_size, -1, -1, -1),
                    grid=grid_xy,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                ).to(pos_embed.dtype)
            else:
                max_dim = max(input_dims)
                pos_embed = F.interpolate(
                    pos_embed.float(),
                    size=(max_dim, max_dim),
                    align_corners=True,
                    mode="bilinear",
                ).to(pos_embed.dtype)
                pos_embed = window_select(pos_embed)
        else:
            pos_embed = window_select(pos_embed)
        if pos_embed.shape[-2:] != input_dims:
            pos_embed = F.interpolate(
                pos_embed.float(), size=input_dims, align_corners=True, mode="bilinear"
            ).to(pos_embed.dtype)
        return pos_embed.flatten(2).permute(0, 2, 1)


class RADIOBase(nn.Module):
    """RADIO base class."""

    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 0,
        backbone: str = "vit_base_patch16_224",
        summary_idxs: Optional[List[int]] = None,
        window_size: Optional[int] = None,
        num_teacher: int = 4,
        cpe_max_size: int = 2048,
        register_multiple: int = 8,
    ):
        """Initialize the RADIO base model.

        Args:
            in_chans (int): Number of input image channels. Default: `3`.
            num_classes (int): Number of classes for classification head. Default: `0`.
            backbone (str): Name of the ViT backbone. Default: `"vit_base_patch16_224"`.
            summary_idxs (list): Indices of the summary tokens. Default: `None`.
            window_size (int): Window size for windowed attention. Default: `None`.
            num_teacher (int): Number of teachers. Default: `4`.
            cpe_max_size (int): Maximum size of the cropped positional embedding. Default: `2048`.
            register_multiple (int): Number of extra tokens. Default: `8`.
        """
        super().__init__()
        self.in_chans = int(in_chans)
        self.num_classes = int(num_classes)
        self.backbone = str(backbone)
        self.summary_idxs = summary_idxs
        self._window_size = window_size
        self.num_teacher = int(num_teacher)
        self.cpe_max_size = int(cpe_max_size)
        self.register_multiple = int(register_multiple)

        # Instantiate the RADIO backbone using TIMM.
        self.model_cfg = radio_model_cfg.get(self.backbone, None)
        if self.model_cfg is None:
            raise ValueError(
                f"Unsupported backbone: {self.backbone}. Supported backbones are: {list(radio_model_cfg.keys())}"
            )

        if "img_size" in self.model_cfg:
            img_size = self.model_cfg["img_size"]
            del self.model_cfg["img_size"]
        else:
            img_size = 224

        vit_backbone = VisionTransformer(
            img_size=img_size,
            in_chans=self.in_chans,
            num_classes=self.num_classes,
            drop_rate=0.0,
            drop_path_rate=0.0,
            global_pool="token",
            weight_init="skip",
            **self.model_cfg,
        )
        # CRADIO models replace `vit_backbone.norm` and `vit_backbone.head` with `nn.Identity()`.
        if hasattr(vit_backbone, "norm"):
            vit_backbone.norm = nn.Identity()
        vit_backbone.head = nn.Identity()

        # Enable cropped position embedding.
        self._enable_cpe(
            vit_backbone,
            max_img_size=self.cpe_max_size,
            num_cls_tokens=self.num_teacher,
            register_multiple=self.register_multiple,
        )
        self.model = vit_backbone

    @property
    def num_cls_tokens(self):
        """Number of class tokens."""
        return int(self.model.patch_generator.num_cls_tokens)

    @property
    def num_summary_tokens(self):
        """Number of summary tokens (include register tokens and class tokens)."""
        return int(self.model.patch_generator.num_skip)

    @property
    def patch_size(self):
        """Patch size."""
        return int(self.model.patch_generator.patch_size)

    @property
    def window_size(self):
        """Window size for windowed attetion."""
        return self._window_size

    def _enable_cpe(
        self,
        model: VisionTransformer,
        max_img_size: Union[int, Tuple[int, int]] = 2048,
        num_cls_tokens: int = 4,
        pos_dropout: float = 0.1,
        register_multiple: int = 8,
    ):
        """Enable cropped position embedding (CPE) for the ViT model.

        Args:
            model (VisionTransformer): ViT model.
            max_img_size (tuple): Maximum image size. Default: `2048`.
            num_cls_tokens (int): Number of class tokens. Default: `4`.
            pos_dropout (float): Dropout rate of the position embedding. Default: `0.1`.
            register_multiple (int): Number of extra tokens. Default: `8`.
        """
        if not isinstance(model, VisionTransformer):
            raise ValueError(
                f"CPE only supports for VisionTransformer model. Received: {type(model)}"
            )

        patch_size = model.patch_embed.patch_size[0]
        embed_dim = model.embed_dim
        input_dims = model.patch_embed.img_size
        normalize_patches = not isinstance(model.patch_embed.norm, nn.Identity)
        cls_token = model.cls_token is not None
        max_img_size = int(round(max_img_size / patch_size) * patch_size)

        patch_generator = ViTPatchGenerator(
            patch_size=patch_size,
            embed_dim=embed_dim,
            input_dims=input_dims,
            normalize_patches=normalize_patches,
            cls_token=cls_token,
            max_input_dims=max_img_size,
            pos_dropout=pos_dropout,
            num_cls_tokens=num_cls_tokens,
            register_multiple=register_multiple,
        )

        model.patch_generator = patch_generator
        model.patch_embed = None
        model.cls_token = None
        if hasattr(model, "reg_token"):
            model.reg_token = None  # type: ignore[attr-defined]
        model.pos_embed = None
        model.pos_drop = None
        model.patch_size = patch_size
        model.num_cls_tokens = num_cls_tokens
        model.num_registers = patch_generator.num_registers

        # Replace the forward function to use the new patch generator.
        def _forward_cpe(self: VisionTransformer, x: torch.Tensor):
            """Forward function for CPE."""
            x = self.patch_generator(x)
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint_seq(self.blocks, x)
            else:
                x = self.blocks(x)
            return self.norm(x)

        model.forward_features = types.MethodType(_forward_cpe, model)

    def forward_pre_logits(self, x):
        """Forward pass through the backbone, excluding the head."""
        return self.model.forward_features(x)

    def forward(self, x: torch.Tensor):
        """Forward."""
        y = self.model.forward_features(x)
        all_summary = y[:, : self.num_cls_tokens]
        if self.summary_idxs is not None:
            bb_summary = all_summary[:, self.summary_idxs]
        else:
            bb_summary = all_summary
        summary = bb_summary.flatten(1).to(torch.float32)
        features = y[:, self.num_summary_tokens :]
        features = features.to(torch.float32)
        return summary, features
