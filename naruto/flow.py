# mmdit_prefix_mask.py
# ------------------------------------------------------------
# Minimal Dual-Stream MMDiT with 2x2 Token Fusion, RoPE (2D/1D),
# and per-sample prefix masking on context tokens.
#
# - Borrowed design ideas:
#     * pre_attention() produces qkv, then joint attention mixes streams
#     * post_attention() applies projection + MLP with AdaLN-style modulation
# - Self-contained: no xformers. Uses torch SDPA-style math.
# - PyTorch 2.x
# ------------------------------------------------------------
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from util import _build_inv_freq, RMSNorm, apply_rope_2d, apply_rope_1d, build_hw_positions
from torch.utils.checkpoint import checkpoint


import torch, math

def _make_window_BL(L, t_B, w=0.05, kind='hann'):
    """
    Return a window of shape (B, L) with per-sample centers t_B in [0,1].
    w: scalar half-width (0.05 = ±5%) or a tensor of shape (B,).
    kind: 'hann' or 'bump'
    """
    device, dtype = t_B.device, t_B.dtype
    B = t_B.shape[0]
    pos = torch.linspace(0., 1., L, device=device, dtype=dtype)  # (L,)

    # Broadcast-friendly ratio r = (x - t)/w
    if torch.is_tensor(w):
        w_B1 = w.view(B, 1)
    else:
        w_B1 = torch.tensor(w, device=device, dtype=dtype).view(1, 1).expand(B, 1)

    r = (pos.view(1, L) - t_B.view(B, 1)) / w_B1  # (B, L)

    y = torch.zeros((B, L), device=device, dtype=dtype)
    if kind == 'hann':
        m = r.abs() <= 1
        y[m] = 0.5 * (1 + torch.cos(math.pi * r[m]))
    elif kind == 'bump':
        m = r.abs() < 1
        y[m] = torch.exp(1.0 - 1.0 / (1.0 - r[m]**2))
    else:
        raise ValueError("kind must be 'hann' or 'bump'")
    return y  # (B, L)

def apply_grad_window(x_BLD, t_B, w=0.05, kind='hann'):
    """
    x_BLD: (B, L, D) with requires_grad=True
    t_B:   (B,) centers in [0,1]
    w:     scalar or (B,) half-width(s)
    Forward  = x_BLD (unchanged)
    Backward = grad wrt x_BLD multiplied elementwise by window
    """
    B, L, D = x_BLD.shape
    win_BL = _make_window_BL(L, t_B, w=w, kind=kind)           # (B, L)
    W = win_BL.unsqueeze(-1)                                    # (B, L, 1) -> broadcasts to D
    W = W.detach()                                              # stop gradient through the window
    return x_BLD * W + x_BLD.detach() * (1.0 - W)
# -----------------------------
# Utilities
# -----------------------------
def modulate(x: torch.Tensor, shift: Optional[torch.Tensor], scale: Optional[torch.Tensor], dim: int = 1):
    """
    x: [..., D]
    shift, scale: shape broadcastable to x, expand along dim (sequence dim or batch dim depending on caller).
    - If shift/scale is None, treat as 0 / 0 respectively (i.e., identity).
    """
    if scale is None and shift is None:
        return x
    # We want: (1 + scale) * x + shift
    if scale is not None:
        x = x * (1 + scale.unsqueeze(dim))
    if shift is not None:
        x = x + shift.unsqueeze(dim)
    return x

# -----------------------------
# Attention Core (torch math)
# -----------------------------
def split_qkv(qkv: torch.Tensor, head_dim: int):
    """
    qkv: [B, L, 3*H*Dh]
    head_dim: Dh
    Returns q,k,v as [B, L, H*Dh] (flattened heads), matching the 3rd-party call style.
    """
    B, L, threeHDh = qkv.shape
    H = threeHDh // (3 * head_dim)
    qkv = qkv.reshape(B, L, 3, H, head_dim).movedim(2, 0)  # [3, B, L, H, Dh]
    q, k, v = qkv[0], qkv[1], qkv[2]                       # each [B, L, H, Dh]
    # return flattened (like their pre_attention does after norm)
    return (q.reshape(B, L, H * head_dim),
            k.reshape(B, L, H * head_dim),
            v.reshape(B, L, H * head_dim))

def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, mask: Optional[torch.Tensor] = None):
    """
    q,k,v: [B, L, H*Dh]  (flattened heads)
    num_heads: H
    mask: optional additive mask
      - For self-attn: shape [B, 1, 1, Lk] broadcast to [B,H,Lq,Lk]
      - For joint:     shape [B, 1, Lq, Lk] or [B,1,1,Lk]
    Returns: [B, L, H*Dh]
    """
    B, Lq, HDh = q.shape
    H = num_heads
    Dh = HDh // H
    assert HDh == H * Dh
    _, Lk, _ = k.shape

    q = q.view(B, Lq, H, Dh).transpose(1, 2)  # [B,H,Lq,Dh]
    k = k.view(B, Lk, H, Dh).transpose(1, 2)  # [B,H,Lk,Dh]
    v = v.view(B, Lk, H, Dh).transpose(1, 2)  # [B,H,Lk,Dh]

    attn = torch.einsum("bhqd,bhkd->bhqk", q, k) * (1.0 / math.sqrt(Dh))
    if mask is not None:
        attn = attn + mask  # broadcastable

    p = attn.softmax(dim=-1)
    o = torch.einsum("bhqk,bhkd->bhqd", p, v)  # [B,H,Lq,Dh]
    o = o.transpose(1, 2).contiguous().view(B, Lq, H * Dh)
    return o


# -----------------------------
# Self-Attention w/ RoPE
# -----------------------------
class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        rope_2d: bool = False,
        qk_norm: Optional[str] = None,
        dtype=None,
        device=None,
        init_method: Optional[int] = None,
        pre_only: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rope_2d = rope_2d

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        self.pre_only = pre_only
        if not pre_only:
            self.proj = nn.Linear(dim, dim, dtype=dtype, device=device)

        if init_method in (1, 2):
            nn.init.constant_(self.proj.weight, 0.0)
            if self.proj.bias is not None:
                nn.init.constant_(self.proj.bias, 0.0)

        # optional qk norm
        if qk_norm == "rms":
            self.ln_q = RMSNorm(self.head_dim, elementwise_affine=True)
            self.ln_k = RMSNorm(self.head_dim, elementwise_affine=True)
        elif qk_norm == "ln":
            self.ln_q = nn.LayerNorm(self.head_dim, elementwise_affine=True)
            self.ln_k = nn.LayerNorm(self.head_dim, elementwise_affine=True)
        else:
            self.ln_q = nn.Identity()
            self.ln_k = nn.Identity()

    def pre_attention(self, x: torch.Tensor, rope_pos=None):
        """
        x: [B,T,D]
        rope_pos:
           - if 2D: (pos_h[pos T], pos_w[pos T])
           - if 1D: pos_1d [T]
        Returns (q,k,v) flattened over heads: [B,T,H*Dh]
        """
        B, T, D = x.shape
        qkv = self.qkv(x)  # [B,T,3D]
        # split into heads for RoPE, then flatten again
        q, k, v = qkv.chunk(3, dim=-1)
        H = self.num_heads
        Dh = D // H
        q = q.view(B, T, H, Dh).transpose(1, 2)  # [B,H,T,Dh]
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)

        # qk norm per-head
        B_, H_, T_, Dh_ = q.shape
        q = self.ln_q(q.reshape(B_*H_, T_, Dh_)).reshape(B_, H_, T_, Dh_)
        k = self.ln_k(k.reshape(B_*H_, T_, Dh_)).reshape(B_, H_, T_, Dh_)

        # RoPE
        if rope_pos is not None:
            if self.rope_2d:
                pos_h, pos_w = rope_pos
                q, k = apply_rope_2d(q, k, pos_h, pos_w)
            else:
                pos_1d = rope_pos
                q, k = apply_rope_1d(q, k, pos_1d)

        # flatten heads like the reference pre_attention does
        q = q.transpose(1, 2).contiguous().view(B, T, H * Dh)
        k = k.transpose(1, 2).contiguous().view(B, T, H * Dh)
        v = v.transpose(1, 2).contiguous().view(B, T, H * Dh)
        return (q, k, v)

    def post_attention(self, x: torch.Tensor):
        return self.proj(x)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256):
        super().__init__()
        h = int(2 * hidden_dim / 3)
        h = multiple_of * ((h + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, h, bias=False)
        self.w2 = nn.Linear(h, dim, bias=False)
        self.w3 = nn.Linear(dim, h, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# -----------------------------
# Blocks and Joint Mixing
# -----------------------------
class DismantledBlock(nn.Module):
    """
    A DiT-like block with:
      - norm + SelfAttention (with optional AdaLN modulate)
      - norm + MLP (SwiGLU)
    Supports pre_attention/post_attention split to mirror the reference.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        rope_2d: bool = False,
        qkv_bias: bool = False,
        qk_norm: Optional[str] = None,
        init_method: Optional[int] = None,
        pre_only: bool = False,
    ):
        super().__init__()
        self.pre_only = pre_only
        self.norm1 = RMSNorm(hidden_size, elementwise_affine=False)
        self.attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            rope_2d=rope_2d,
            qk_norm=qk_norm,
            init_method=init_method,
            pre_only=pre_only,
        )
        if not pre_only:
            self.norm2 = RMSNorm(hidden_size, elementwise_affine=False)
            self.mlp = SwiGLUFeedForward(hidden_size, int(hidden_size * mlp_ratio))

        # AdaLN modulation chunk layout: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        self.adaln = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        nn.init.constant_(self.adaln[-1].weight, 0.0)
        nn.init.constant_(self.adaln[-1].bias, 0.0)

    def pre_attention(self, x: torch.Tensor, c: torch.Tensor, rope_pos=None, shift_gate_only=False):
        """
        x: [B,T,D], c: [B,D]
        """
        B, T, D = x.shape
        mods = self.adaln(c).chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods
        # norm+modulate then pre-attention (with RoPE)
        x_mod = modulate(self.norm1(x), shift_msa, scale_msa, dim=1)
        q, k, v = self.attn.pre_attention(x_mod, rope_pos=rope_pos)
        if self.pre_only:
            return (q, k, v), None
        return (q, k, v), (x, gate_msa, shift_mlp, scale_mlp, gate_mlp)

    def post_attention(self, attn_out: torch.Tensor, x: torch.Tensor, gate_msa, shift_mlp, scale_mlp, gate_mlp):
        assert not self.pre_only
        x = x + gate_msa.unsqueeze(1) * self.attn.post_attention(attn_out)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp, dim=1))
        return x


def block_mixing(
    # streams
    context: torch.Tensor, x: torch.Tensor,
    # blocks
    context_block: DismantledBlock,
    x_block: DismantledBlock,
    # rope
    rope_ctx, rope_img,
    # masks
    joint_mask: Optional[torch.Tensor],
    # condition
    c: torch.Tensor
):
    """
    - Precompute qkv for context and x
    - Concatenate along Lk dimension for joint attention
    - Split outputs and post-process
    """
    # pre-attn
    context_qkv, ctx_int = context_block.pre_attention(context, c, rope_pos=rope_ctx)
    x_qkv, x_int = x_block.pre_attention(x, c, rope_pos=rope_img)

    # concatenate (q from x; keys/vals are [context, x])
    # to mirror the "mixing" style, we concatenate q/k/v lists elementwise
    q = x_qkv[0]                               # [B, Tx, H*Dh]  (query from x)
    k = torch.cat([context_qkv[1], x_qkv[1]], dim=1)  # [B, Tc+Tx, H*Dh]
    v = torch.cat([context_qkv[2], x_qkv[2]], dim=1)  # [B, Tc+Tx, H*Dh]

    # attention over concatenated keys
    attn_x = attention(q, k, v, x_block.attn.num_heads, mask=joint_mask)  # [B, Tx, H*Dh]

    # post-attn for x (context_block could be pre_only=True, mirroring ref)
    x = x_block.post_attention(attn_x, *x_int)

    # Optionally also run context post-attn (here we skip to keep it concise / as in reference pre_only option)
    if (ctx_int is not None) and (not context_block.pre_only):
        # compute attention for context with its own queries (optional)
        q_ctx = context_qkv[0]
        attn_ctx = attention(q_ctx, k, v, context_block.attn.num_heads, mask=joint_mask)
        context = context_block.post_attention(attn_ctx, *ctx_int)
    else:
        context = context  # unchanged

    return context, x


# -----------------------------
# Position Helpers & Masks
# -----------------------------
def seq_prefix_mask(min_lengths: torch.Tensor, T: int, device) -> torch.Tensor:
    """
    max_lengths: [B] (visible prefix lengths end)
    min_lengths: [B] (visible prefix lengths start)
    Returns additive mask for keys: [B,1,1,T] (broadcast to [B,H,Q,T])
      masked positions get large negative.
    """
    B = min_lengths.numel()
    ar = torch.arange(T, device=device)[None, :]
    visible = ((ar > min_lengths[:, None])).float()
    mask = (1.0 - visible) * (-1e7)
    return mask[:, None, None, :]

def build_joint_mask(B: int, Tx: int, Tc: int, z_key_mask: Optional[torch.Tensor]):
    """
    Build additive mask for joint attention where K = [context, x].
    joint_mask: [B,1,Tx, Tc+Tx]
    - left part (context K): use z_key_mask (broadcasted)
    - right part (x K): unmasked
    """
    if z_key_mask is None:
        return None
    # z_key_mask is [B,1,1,Tc]
    left = z_key_mask.repeat(1, 1, Tx, 1)              # [B,1,Tx,Tc]
    right = torch.zeros((B, 1, Tx, Tx), device=left.device, dtype=left.dtype)
    return torch.cat([left, right], dim=-1)            # [B,1,Tx,Tc+Tx]


# -----------------------------
# Top-level Dual-Stream MMDiT
# -----------------------------
@dataclass
class MMDiTConfig:
    C: int = 16
    ctx_token_dim: int = 1024
    hidden_size_img: int = 1024
    hidden_size_ctx: int = 1024
    num_heads: int = 16
    depth: int = 8
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_norm: Optional[str] = None
    init_method: Optional[int] = None

class DualStreamMMDiT(nn.Module):
    """
    - Image stream from fused tokens (2x2) with 2D RoPE
    - Context stream from z tokens (sequence) with 1D RoPE
    - Per-sample prefix mask on z keys/values
    - Joint attention: x queries attend to [z, x] keys/vals
    - Final unpatchify head to predict image-shaped residual
    """
    def __init__(
        self,
        cfg: MMDiTConfig = MMDiTConfig()
    ):
        super().__init__()
        self.cfg = cfg

        # Token input projections
        self.img_in = nn.Linear(cfg.C, cfg.hidden_size_img, bias=False)

        # Blocks
        self.blocks = nn.ModuleList([
            nn.ModuleDict(dict(
                ctx_block=DismantledBlock(cfg.hidden_size_ctx, cfg.num_heads, cfg.mlp_ratio, rope_2d=False,
                                          qkv_bias=cfg.qkv_bias, qk_norm=cfg.qk_norm, init_method=cfg.init_method,
                                          pre_only=True),   # context as pre_only (like reference)
                x_block=DismantledBlock(cfg.hidden_size_img, cfg.num_heads, cfg.mlp_ratio, rope_2d=True,
                                        qkv_bias=cfg.qkv_bias, qk_norm=cfg.qk_norm, init_method=cfg.init_method,
                                        pre_only=False),
            )) for _ in range(cfg.depth)
        ])

        # Output heads
        # self.img_norm = RMSNorm(cfg.hidden_size_img, elementwise_affine=False)
        self.img_head = nn.Linear(cfg.hidden_size_img, cfg.C, bias=False)

        self.t_dim = 64
        self.t_proj = nn.Linear(self.t_dim, cfg.hidden_size_img, bias=False)
        # buffer of frequencies (shared across dtypes; cast at use time)
        half = 32
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half).float() / half)
        self.register_buffer("t_freqs", freqs, persistent=False)

        # Register dummy buffers for positions (just placeholders; built per forward)
        self.register_buffer("dummy", torch.empty(0), persistent=False)
        self.use_checkpoint = False

    def forward(
        self,
        x_t: torch.Tensor,        # [B,C,H,W]
        z_tok: torch.Tensor,      # [B,Lz,Dz] (already tokens)
        t: torch.Tensor,          # [B] in [0,1], controls time embedding
    ):
        B, C, H, W = x_t.shape
        x_tokens = self.img_in(x_t.reshape(B, C, H * W).transpose(1, 2))                  # [B,Tx,Dim]

        # --- Prepare context tokens
        Bz, Lz, Dz = z_tok.shape
        assert Bz == B
        ctx = apply_grad_window(z_tok, t, w=0.05, kind='hann')

        # min_lengths = torch.clamp((t * Lz).floor().long(), 0, Lz)       # [B]
        # z_tok_temp = z_tok.detach().clone()
        # for i in range(B):
        #     z_tok_temp[i, :min_lengths[i]] = z_tok[i, :min_lengths[i]]
        # ctx = z_tok_temp # [B,Lz,Dc]

        # --- RoPE positions
        pos_h, pos_w = build_hw_positions(H, W, device=x_t.device, dtype=x_t.dtype)
        pos_img = (pos_h, pos_w)                          # for 2D RoPE on image
        pos_ctx = torch.arange(Lz, device=z_tok.device, dtype=z_tok.dtype)  # 1D positions for z

        # --- z prefix mask
        joint_mask = None

        # z_key_mask = seq_prefix_mask(min_lengths, T=Lz, device=z_tok.device)    # [B,1,1,Lz]
        # joint_mask = build_joint_mask(B, H * W, Lz, z_key_mask)        # [B,1,Tx,Lz+Tx]
        # --- Iterate blocks
        img = x_tokens
        for md in self.blocks:
            if self.use_checkpoint:
                ctx, img = checkpoint(block_mixing, ctx, img, md["ctx_block"], md["x_block"], pos_ctx, pos_img, joint_mask, self.time_embed(t, dim=img.shape[-1], device=img.device, dtype=img.dtype), use_reentrant=False)
            else:
                ctx, img = block_mixing(ctx, img, md["ctx_block"], md["x_block"], pos_ctx, pos_img, joint_mask, self.time_embed(t, dim=img.shape[-1], device=img.device, dtype=img.dtype))

            # ctx, img = block_mixing(
            #     ctx, img,
            #     context_block=md["ctx_block"],
            #     x_block=md["x_block"],
            #     rope_ctx=pos_ctx,
            #     rope_img=pos_img,
            #     joint_mask=joint_mask,
            #     c=self.time_embed(t, dim=img.shape[-1], device=img.device, dtype=img.dtype)  # simple t-conditioning
            # )

        # --- Project back to tokens and unpatchify
        # img = self.img_head(self.img_norm(img))           # [B, L, D]
        img = self.img_head(img)           # [B, L, D]
        img = img.reshape(B, H, W, self.cfg.C).permute(0, 3, 1, 2)
        return img

    def time_embed(self, t: torch.Tensor, dim: int, device, dtype):
        half = self.t_freqs.shape[0]
        freqs = self.t_freqs.to(device=device, dtype=dtype)
        ang = t.unsqueeze(1) * freqs.unsqueeze(0)              # [B, half]
        emb = torch.cat([ang.sin(), ang.cos()], dim=-1)        # [B, 64]
        return self.t_proj(emb)                                # [B, hidden_size_img]
    # @staticmethod
    # def time_embed(t: torch.Tensor, dim: int, device, dtype):
    #     """
    #     Simple sinusoidal T-embedding → linear → returns [B,dim]
    #     """
    #     # make a fixed 64-d sincos then project
    #     B = t.shape[0]
    #     half = 32
    #     freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device, dtype=dtype) / half)
    #     ang = t.unsqueeze(1) * freqs.unsqueeze(0)  # [B,half]
    #     emb = torch.cat([ang.sin(), ang.cos()], dim=-1)  # [B,64]
    #     # project (cache per call for simplicity; in real use, register a layer)
    #     proj = torch.empty(64, dim, device=device, dtype=dtype)
    #     nn.init.normal_(proj, std=0.02)
    #     return emb @ proj  # [B,dim]


# -----------------------------
# Demo / Sanity Check
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # Build model
    cfg = MMDiTConfig(
        C=16,
        ctx_token_dim=1024,
        hidden_size_img=1024,
        hidden_size_ctx=1024,
        num_heads=16,
        depth=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=None,
        init_method=None,
    )

    B, C, H, W = 2, cfg.C, 34, 64
    x_t = torch.randn(B, C, H, W)
    # Condition tokens z: suppose already produced by your Model A (ViT/DiT encoder)
    Lz, Dz = 512, cfg.ctx_token_dim
    z = torch.randn(B, Lz, Dz)
    t = torch.rand(B)  # [0,1]


    model = DualStreamMMDiT(cfg=cfg)

    # Forward
    delta = model(x_t=x_t, z_tok=z, t=t)
    print("delta:", tuple(delta.shape))   # -> [B, C, H_even, W_even]
    print("x_t.shape:", tuple(x_t.shape))
    print("z.shape:", tuple(z.shape))

    # delta = model(x_t=x_t, z_tok=z_seq, t=t)
