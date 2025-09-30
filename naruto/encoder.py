# ------------------------------------------------------------
# Encoder: ViT over VAE latents with spatial merger
#   Input : x [N, C, H, W]
#   Merge : factor×factor → tokens L = (H//factor)*(W//factor)
#   Output: z [N, D, L]   (and also z_seq [N, L, D] if you prefer)
#   Extras: 2D RoPE inside attention; optional padding to handle odd sizes.
# ------------------------------------------------------------
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import RMSNorm, _build_inv_freq, apply_rope_2d, build_hw_positions

# -----------------------------
# Spatial merger
# -----------------------------
@dataclass
class SpatialMergerCfg:
    factor: int = 2          # kernel_size = stride = factor
    embed_dim: int = 1024    # token dimension after merging (D)
    bias: bool = False
    pad_if_needed: bool = True  # pad so that H,W are divisible by factor

class SpatialMerger(nn.Module):
    """
    Merge non-overlapping factor×factor patches into tokens, then project to D.
      x: [N,C,H,W] → unfold(k=factor,s=factor) → [N, L, C*factor*factor] → Linear → [N,L,D]
    Returns (tokens, (Hf,Wf), pad_info)
    """
    def __init__(self, C: int, cfg: SpatialMergerCfg):
        super().__init__()
        self.C = C
        self.cfg = cfg
        self.proj = nn.Linear(C * cfg.factor * cfg.factor, cfg.embed_dim, bias=cfg.bias)

    def forward(self, x: torch.Tensor):
        N, C, H, W = x.shape
        assert C == self.C
        f = self.cfg.factor
        pad_info = {"pad_h": 0, "pad_w": 0}

        need_h = H % f != 0
        need_w = W % f != 0
        if (need_h or need_w) and not self.cfg.pad_if_needed:
            raise ValueError(f"H,W must be divisible by factor={f}, got {(H,W)}")

        if need_h or need_w:
            ph = (f - H % f) % f
            pw = (f - W % f) % f
            if ph or pw:
                x = F.pad(x, (0, pw, 0, ph))  # (left,right,top,bottom)
                H += ph; W += pw
                pad_info["pad_h"] = ph
                pad_info["pad_w"] = pw

        Hf, Wf = H // f, W // f
        patches = F.unfold(x, kernel_size=f, stride=f)  # [N, C*f*f, L]
        patches = patches.transpose(1, 2).contiguous()  # [N, L, C*f*f]
        tokens = self.proj(patches)                     # [N, L, D]
        return tokens, (Hf, Wf), pad_info


# -----------------------------
# Attention / MLP blocks
# -----------------------------
class RoPEMultiheadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = False):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, rope_pos_2d: Tuple[torch.Tensor, torch.Tensor]):
        """
        x: [N,T,D]; rope_pos_2d = (pos_h[T], pos_w[T])
        """
        N, T, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        qkv = self.qkv(x)                         # [N,T,3D]
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(N, T, H, Dh).transpose(1, 2)   # [N,H,T,Dh]
        k = k.view(N, T, H, Dh).transpose(1, 2)
        v = v.view(N, T, H, Dh).transpose(1, 2)

        # 2D RoPE
        pos_h, pos_w = rope_pos_2d
        q, k = apply_rope_2d(q, k, pos_h, pos_w)

        attn = torch.einsum("nhtd,nhsd->nhts", q, k) * (1.0 / math.sqrt(Dh))
        p = attn.softmax(dim=-1)
        o = torch.einsum("nhts,nhsd->nhtd", p, v)        # [N,H,T,Dh]
        o = o.transpose(1, 2).contiguous().view(N, T, D) # [N,T,D]
        return self.proj(o)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class ViTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, qkv_bias: bool, mlp_ratio: float):
        super().__init__()
        self.norm1 = RMSNorm(dim, elementwise_affine=False)
        self.attn  = RoPEMultiheadAttention(dim, heads, qkv_bias=qkv_bias)
        self.norm2 = RMSNorm(dim, elementwise_affine=False)
        self.mlp   = MLP(dim, mlp_ratio)

    def forward(self, x: torch.Tensor, rope_pos_2d: Tuple[torch.Tensor, torch.Tensor]):
        x = x + self.attn(self.norm1(x), rope_pos_2d)
        x = x + self.mlp(self.norm2(x))
        return x


# -----------------------------
# Model A (ViT with spatial merger)
# -----------------------------
@dataclass
class ModelACfg:
    factor: int = 2            # spatial downsample factor
    embed_dim: int = 1024      # token/channel dim D
    depth: int = 8
    heads: int = 16
    qkv_bias: bool = True
    mlp_ratio: float = 4.0
    pad_if_needed: bool = True # pad to multiple of factor for H/W

class ModelA_ViT(nn.Module):
    """
    Model A:
      - Merge factor×factor patches -> tokens [N,L,D]
      - ViT with 2D RoPE
      - Output as [N, D, L] (and also provide a seq-view if needed)
    """
    def __init__(self, C: int, cfg: ModelACfg = ModelACfg()):
        super().__init__()
        self.cfg = cfg
        self.merger = SpatialMerger(C, SpatialMergerCfg(
            factor=cfg.factor, embed_dim=cfg.embed_dim, pad_if_needed=cfg.pad_if_needed
        ))
        self.blocks = nn.ModuleList([
            ViTBlock(cfg.embed_dim, cfg.heads, cfg.qkv_bias, cfg.mlp_ratio)
            for _ in range(cfg.depth)
        ])
        self.out_norm = RMSNorm(cfg.embed_dim, elementwise_affine=False)
        # (optional) output projection; keep identity by default
        self.out_proj = nn.Identity()

        # Buffers for positions are created on-the-fly per input size,
        # so we don't pre-register fixed-size ones.

    def forward(self, x: torch.Tensor):
        """
        x: [N,C,H,W]
        Returns:
          z      : [N, D, L]  (channel-first for your downstream)
          z_seq  : [N, L, D]  (sequence-first if needed)
          (Hf,Wf): spatial size after merger
          pad_info: dict for unpadding if you need exact inversion later
        """
        N, C, H, W = x.shape
        # 1) merge to tokens
        tok, (Hf, Wf), pad_info = self.merger(x)     # tok: [N, L, D]
        L = Hf * Wf

        # 2) 2D RoPE positions on reduced grid
        pos_h, pos_w = build_hw_positions(Hf, Wf, device=tok.device, dtype=tok.dtype)

        # 3) ViT encoder
        for blk in self.blocks:
            tok = blk(tok, rope_pos_2d=(pos_h, pos_w))

        tok = self.out_proj(self.out_norm(tok))      # [N, L, D]

        # 4) arrange outputs
        z_seq = tok                                   # [N, L, D]
        z = tok.transpose(1, 2).contiguous()          # [N, D, L]
        return z, z_seq, (Hf, Wf), pad_info


# -----------------------------
# Quick sanity check
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    N, C, H, W = 2, 16, 34, 64
    cfg = ModelACfg(factor=2, embed_dim=1024, depth=6, heads=16, qkv_bias=True)
    modelA = ModelA_ViT(C, cfg)

    x = torch.randn(N, C, H, W)
    z, z_seq, (Hf, Wf), pad_info = modelA(x)
    print("z  :", tuple(z.shape))      # [N, D, L]
    print("z_seq:", tuple(z_seq.shape))# [N, L, D]
    print("(Hf,Wf):", (Hf, Wf), "L =", Hf*Wf, "pad:", pad_info)

