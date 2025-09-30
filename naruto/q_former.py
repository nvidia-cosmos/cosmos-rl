import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Optional
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from torch.utils.checkpoint import checkpoint

def ckpt_wrapper(module):
    def ckpt_forward(*inputs):
        outputs = module(*inputs)
        return outputs
    return ckpt_forward

def modulate(x, shift, scale, dim=1):
    if shift is None or scale is None:
        return x
    return x * (1 + scale.unsqueeze(dim)) + shift.unsqueeze(dim)

def gate(x, gate):
    if gate is None:
        return x
    return gate.unsqueeze(0) * x

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, elementwise_affine: bool = True, eps: float = 1e-6, dtype=torch.float, device='cpu'):
        """
        Initialize the RMSNorm normalization layer.
        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.
        """
        super().__init__()
        self.eps = eps
        self.learnable_scale = elementwise_affine
        if self.learnable_scale:
            self.weight = nn.Parameter(torch.empty(dim).to(dtype=dtype, device=device))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self,):
        if self.learnable_scale:
            init.ones_(self.weight)

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.
        """
        x = self._norm(x)
        if self.learnable_scale:
            return x * self.weight.to(device=x.device, dtype=x.dtype)
        else:
            return x

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class ViTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn=Attention, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = attn(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DualAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            query_dim: int,
            num_heads: int = 8,
            query_heads: int=8,
            bidirectional: bool=True,
            zero_init: bool = False,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = RMSNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        if bidirectional:
            assert dim == query_dim
            assert num_heads == query_heads
        self.num_heads = num_heads
        self.query_heads = query_heads
        self.head_dim = dim // num_heads
        self.query_head_dim = query_dim // query_heads
        self.bidrectional = bidirectional

        if qk_norm:
            print("Encoder using qk norm...")
            
        # latent linear
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.to_query_kv = nn.Linear(dim, query_dim * 2, bias=qkv_bias) \
            if (not bidirectional) or zero_init else nn.Identity()
        # query linear
        self.query_linear = nn.Linear(query_dim, query_dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.query_qnorm = norm_layer(self.query_head_dim) if qk_norm else nn.Identity()
        self.query_knorm = norm_layer(self.query_head_dim) if qk_norm else nn.Identity()

        self.zero_init = zero_init
        if self.zero_init:
            self.gate = torch.nn.Parameter(torch.zeros(1, self.query_heads, 1,1))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.query_proj = nn.Linear(query_dim, query_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, query: torch.Tensor, mask: torch.Tensor=None, x_mask: torch.Tensor=None) -> torch.Tensor:
        B, N, C = x.shape
        _, query_N, query_C = query.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        query_qkv = self.query_linear(query).reshape(
            B, query_N, 3, self.query_heads, self.query_head_dim
        ).permute(2, 0, 3, 1, 4)

        if self.zero_init:
            kv = self.to_query_kv(x).reshape(B, N, 2, self.query_heads, self.query_head_dim).permute(2, 0, 3, 1, 4)
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=x_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
            k, v = kv.unbind(0)
            query_q, query_k, query_v = query_qkv.unbind(0)
            xk = torch.cat([k, query_k], dim=2)
            xv = torch.cat([v, query_v], dim=2)
            query_q, xk = self.query_qnorm(query_q), self.query_knorm(xk)
            scale_factor = 1 / math.sqrt(self.query_head_dim)
            scores = query_q @ xk.transpose(2,3) * scale_factor
            if mask is not None:
                attn_bias = torch.zeros([B, self.query_heads, query_q.shape[2], xk.shape[2]], dtype=query_q.dtype, device=query_q.device)
                if mask.dtype == torch.bool:
                    attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
                else:
                    attn_bias += mask
                scores = scores + attn_bias
            scores = torch.cat(
                [
                    self.gate.tanh() * F.softmax(scores[:, :, :, :N], dim=-1).type_as(query_q),
                    F.softmax(scores[:, :, :, N:], dim=-1).type_as(query_q),
                ],
                dim=-1,
            ).type_as(query_q)
            scores = torch.dropout(scores, self.attn_drop.p if self.training else 0, train=True)
            query = scores @ xv
        elif self.bidrectional:
            query_q, query_k, query_v = query_qkv.unbind(0)
            query_q, query_k = self.query_qnorm(query_q), self.query_knorm(query_k)
            q = torch.cat((q, query_q), dim=2)
            k = torch.cat((k, query_k), dim=2)
            v = torch.cat((v, query_v), dim=2)
            x_cat = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
            x, query = x_cat[:,:, :N, :], x_cat[:,:, N:, :]
        else:
            kv = self.to_query_kv(x).reshape(B, N, 2, self.query_heads, self.query_head_dim).permute(2, 0, 3, 1, 4)
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=x_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
            k, v = kv.unbind(0)
            query_q, query_k, query_v = query_qkv.unbind(0)

            k = torch.cat([k,query_k], dim=2)
            v = torch.cat([v,query_v], dim=2)
            q, k = self.query_qnorm(query_q), self.query_knorm(k)

            query = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )

        x = x.transpose(1, 2).reshape(B, N, C)
        query = query.transpose(1, 2).reshape(B, query_N, query_C)
        x = self.proj(x)
        query = self.query_proj(query)
        x = self.proj_drop(x)
        query = self.proj_drop(query)
        return x, query

class DualBlock(ViTBlock):
    """
    A dual block similar to SD3 setup.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, time_adaln=False, query_dim=256, diti=None, **block_kwargs):
        block_kwargs["query_dim"] = query_dim
        self.post_ln = block_kwargs.pop("post_ln", False)
        super().__init__(hidden_size, num_heads, mlp_ratio, attn=DualAttention, **block_kwargs)
        q_dim = query_dim
        self.q_norm1 = nn.LayerNorm(q_dim, elementwise_affine=False, eps=1e-6) if not self.post_ln else nn.Identity()
        self.q_norm2 = nn.LayerNorm(q_dim, elementwise_affine=False, eps=1e-6) if not self.post_ln else nn.Identity()
        self.post_norm1 = nn.LayerNorm(q_dim, elementwise_affine=False, eps=1e-6) if self.post_ln else nn.Identity()
        self.post_norm2 = nn.LayerNorm(q_dim, elementwise_affine=False, eps=1e-6) if self.post_ln else nn.Identity()
        self.ln_scale = 1.97 if self.post_ln else 1.0
        mlp_hidden_dim = int(q_dim * mlp_ratio)
        q_approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.q_mlp = Mlp(in_features=q_dim, hidden_features=mlp_hidden_dim, act_layer=q_approx_gelu, drop=0)
        self.time_adaln = time_adaln
        self.diti = diti

        if time_adaln:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(q_dim, 6 * q_dim, bias=True)
            )
            self.t_embedder = TimestepEmbedder(q_dim)
            self.init_block()

    def init_block(self):
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, q, mask=None):
        if self.time_adaln:
            K = q.shape[1]
            if self.diti is not None:
                pos_embed = self.diti.get_position(torch.arange(K).to(x.device))  # *7+1000
            else:
                pos_embed = torch.arange(K).to(x.device)
            t_emb = self.t_embedder(pos_embed)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb).chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = None,None,None,None,None,None
        x_attn, q_attn = self.attn(self.norm1(x), modulate(self.q_norm1(q), shift_msa, scale_msa, 0), mask=mask)
        x = x + x_attn
        x = x + self.mlp(self.norm2(x))

        q = self.post_norm1(self.ln_scale * q + gate(q_attn, gate_msa))
        q = self.post_norm2(self.ln_scale * q + gate(self.q_mlp(modulate(self.q_norm2(q), shift_mlp, scale_mlp, 0)), gate_mlp))
        return x, q

class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size: Optional[int] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        flatten: bool = True,
        bias: bool = True,
        strict_img_size: bool = True,
        dynamic_img_pad: bool = False,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        if img_size is not None:
            self.img_size = (img_size, img_size)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        # flatten spatial dim and transpose to channels last, kept for bwd compat
        self.flatten = flatten
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias, dtype=dtype, device=device
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        return x

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product
    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)
    return np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


class Encoder(nn.Module):
    def __init__(
        self, K, input_size=32, patch_size=8, in_channels=4,
        hidden_size=256, depth=None, num_heads=4, mlp_ratio=4.0,
        post_norm=True, query_dim=None, apply_losses_together=False,
        gradient_checkpointing=False, pos_embed_max_size=None, attn_mask=False, single_token=False, **kwargs
    ):
        super().__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        depth = depth or self.K
        self.depth = depth
        self.hidden_size = hidden_size
        self.post_norm = post_norm
        self.pos_embed_max_size = pos_embed_max_size
        query_dim = query_dim or hidden_size
        self.gradient_checkpointing = gradient_checkpointing
        self.n_tokens = K * (input_size // patch_size) ** 2
        self.apply_losses_together = apply_losses_together
        self.attn_mask = attn_mask
        self.single_token = single_token

        # models
        self.x_embedder = PatchEmbed(img_size=input_size,patch_size=patch_size, in_chans=in_channels, embed_dim=hidden_size, bias=True)
        if pos_embed_max_size is not None:
            num_patches = pos_embed_max_size * pos_embed_max_size
            self.x_embedder.num_patches = pos_embed_max_size * pos_embed_max_size
        else:
            num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        if num_patches is not None:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        else:
            self.pos_embed = None
        self.blocks = nn.ModuleList([
            ViTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        # self.final_layer_norm = nn.LayerNorm(query_dim, eps=1e-6)

        self.initialize_weights()
        
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        if self.pos_embed is not None:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                int(self.x_embedder.num_patches ** 0.5)
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
            
    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def get_encoder_outs(self, x, kwargs=None):
        outs = []
        for i, block in enumerate(self.blocks):
            if self.gradient_checkpointing:
                x = checkpoint(ckpt_wrapper(block), x, use_reentrant=False)
            else:
                x = block(x)
            if i >= self.depth - self.K:
                outs.append(x)
    
        assert len(outs) == self.K
        outs = torch.cat(outs, dim=1)
        return outs

    def forward(self, x=None, kwargs=None):
        """
        Forward pass of feature encoder.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        d: N, the depth for each sample
        """
        x = self.x_embedder(x) + self.pos_embed
        outs = self.get_encoder_outs(x, kwargs=kwargs) #torch.Size([4, 512, 512])
        # if self.post_norm:
        #     outs = self.final_layer_norm(outs)
        return outs

'''
    Encoder with a special input: mode, which can be either
        - 'qformer' with cross attention interaction between query and latent
        - 'concat' with self attention interaction between query and latent
        - 'dual-xx' with self attention interaction between query and latent, but query has its own transformer
            - xx='cross': query as q, latent as kv
            - xx='self': [query,latent] into self-attention
'''
class QformerEncoder(Encoder):
    def __init__(
        self, K, input_size=32, patch_size=8, in_channels=4,
        hidden_size=256, depth=None, num_heads=4, mlp_ratio=4.0,
        post_norm=True, qformer_mode='dual', time_adaln=False,
        gradient_checkpointing=False, pos_embed_max_size=None, apply_losses_together=False,
        xavier_init=False, diti=None, attn_mask=False, single_token=False, **kwargs
    ):
        super().__init__(
            K, input_size, patch_size, in_channels, hidden_size, depth, num_heads,
            mlp_ratio, post_norm, encoder_out_dim=kwargs['query_dim'],
            gradient_checkpointing=gradient_checkpointing, apply_losses_together=apply_losses_together,
            pos_embed_max_size=pos_embed_max_size, attn_mask=attn_mask, single_token=single_token, **kwargs
        )
        qformer_depth = depth
        self.num_query_token = K # num_query_token
        query_dim = kwargs['query_dim']
        self.query_tokens = nn.Parameter(torch.zeros(1, self.num_query_token, query_dim))
        self.query_tokens.data.normal_(mean=0.0, std=0.02) #initialization
        self.mode = qformer_mode
        self.diti = diti
        self.attn_mask = attn_mask
        self.single_token = single_token
        if diti:
            kwargs["diti"] = diti
        if self.mode == 'dual':
            self.blocks = nn.ModuleList([
                DualBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, time_adaln=time_adaln, **kwargs) for _ in range(depth)
            ])
        else:
            raise ValueError("Unknown mode to QFormerEncoder.")

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        if xavier_init:
            self.apply(_basic_init)

    def get_encoder_outs(self, x, kwargs=None):
        query_tokens = self.query_tokens.expand(x.shape[0], -1, -1)
        if self.mode == 'qformer':
            query_tokens = self.qformer(x, query_tokens) # [B, L, C]
        elif self.mode == 'concat':
            for i, block in enumerate(self.blocks):
                if self.gradient_checkpointing:
                    x, query_tokens = checkpoint(ckpt_wrapper(block), x, query_tokens, use_reentrant=False)
                else:
                    x, query_tokens = block(x, query_tokens)
        elif self.mode == 'dual':
            # attn mask
            
            if self.attn_mask: #False
                mask = mask = torch.ones(self.K, self.K).tril().bool().cuda()
                x_mask = torch.ones((self.K, x.shape[1])).cuda()
                mask = torch.cat((x_mask, mask), dim=1).bool()
                mask = mask.unsqueeze(0).unsqueeze(1).repeat(x.shape[0],1,1,1)
            else:
                mask = None

            for i, block in enumerate(self.blocks):
                if self.gradient_checkpointing:
                    x, query_tokens = checkpoint(ckpt_wrapper(block), x, query_tokens, mask, use_reentrant=False)
                else:
                    x, query_tokens = block(x, query_tokens, mask=mask)
        else:
            raise ValueError("Unknown mode to QFormerEncoder.")
        return query_tokens + (x * 0.0).sum()
    
if __name__ == "__main__":
    in_channels = 16
    K = 512
    input_size = 224
    query_dim = 512
    image_stream_hidden_size = 512
    mode = "dual"
    time_adaln = True
    test_input = torch.randn(1, in_channels, input_size, input_size)

    model = QformerEncoder(patch_size=2, hidden_size=image_stream_hidden_size, num_heads=4, depth=6, K=K,
        query_dim=query_dim, query_heads=8, bidirectional=False, in_channels=in_channels, input_size=input_size, qformer_mode=mode, time_adaln=time_adaln
    )
    out = model(test_input)
    print(f"out: {out.shape}")

    print(f"model: {model}")