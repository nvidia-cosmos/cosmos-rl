# pytorch >= 1.12 recommended
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def sinusoidal_time_embed(
    timesteps: torch.Tensor, dim: int, max_period: float = 10000.0
) -> torch.Tensor:
    """
    Continuous sinusoidal embedding for t in [0, 1].
    timesteps: [B] float32 in [0, 1]
    returns: [B, dim]
    """
    device = timesteps.device
    half = dim // 2
    # Frequencies from high to low, covering a wide band for [0,1]
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=device) / max(half - 1, 1)
    )
    # Use 2π for a true Fourier feature style embedding
    angles = (2.0 * math.pi) * timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class FiLM(nn.Module):
    """Simple FiLM (scale/shift) modulation."""

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * d_model))

    def forward(self, h: torch.Tensor, cond_embed: torch.Tensor):
        """
        h: [B, L, d_model]
        cond_embed: [B, hidden]
        returns: [B, L, d_model] modulated
        """
        gamma, beta = self.net(cond_embed).chunk(2, dim=-1)
        return h * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x_tokens: torch.Tensor, ctx_tokens: torch.Tensor):
        """
        x_tokens: [B, L, d_model] (queries)
        ctx_tokens: [B, C, d_model] (keys/values for cross attention)
        """
        # Self-attention
        h = self.ln1(x_tokens)
        s, _ = self.self_attn(h, h, h, need_weights=False)
        x = x_tokens + self.drop(s)

        # Cross-attention (x queries attend to context)
        h = self.ln2(x)
        c, _ = self.cross_attn(h, ctx_tokens, ctx_tokens, need_weights=False)
        x = x + self.drop(c)

        # Feed-forward
        h = self.ln3(x)
        f = self.ff(h)
        x = x + self.drop(f)
        return x


class DiffusionCondAttentionNet(nn.Module):
    """
    Maps x:[B,D1] -> y:[B,D1], conditioned on c:[B,D0], optionally timestep t:[B].
    Design:
      1) Tokenize x into L tokens via a linear projection.
      2) Encode c into C context tokens.
      3) Optional time embedding modulates blocks via FiLM.
      4) Transformer stack with self- and cross-attention.
      5) Project tokens back to D1.
    """

    def __init__(
        self,
        d_in: int,  # D1
        d_cond: int,  # D0
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        num_layers: int = 4,
        x_tokens: int = 8,  # L: number of tokens for x
        cond_tokens: int = 4,  # C: number of tokens for condition
        use_timestep: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_cond = d_cond
        self.d_model = d_model
        self.x_tokens = x_tokens
        self.cond_tokens = cond_tokens
        self.use_timestep = use_timestep

        # Tokenizers / detokenizers
        self.x_proj = nn.Linear(d_in, x_tokens * d_model)
        self.x_norm = nn.LayerNorm(d_model)

        # Encode condition into context tokens
        self.cond_encoder = nn.Sequential(
            nn.Linear(d_cond, d_model),
            nn.SiLU(),
            nn.Linear(d_model, cond_tokens * d_model),
        )
        self.cond_norm = nn.LayerNorm(d_model)

        # Optional timestep embedding -> combine with condition summary
        if use_timestep:
            self.t_embed_dim = d_model
            self.t_mlp = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.t_embed_dim = 0

        # A pooled conditioning vector for FiLM
        self.cond_pool = nn.Sequential(
            nn.Linear(d_cond + (self.t_embed_dim if use_timestep else 0), d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.film = FiLM(d_model, d_model)

        # Transformer stack
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        # Output head
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(x_tokens * d_model, d_in)

        # Initialize a small residual output (stable training)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor = None):
        """
        x:    [B, D1]        - noisy input at some diffusion step
        cond: [B, D0]        - conditioning vector
        t:    [B] or None    - diffusion timestep (optional)
        return: [B, D1]
        """
        B = x.shape[0]

        # Tokenize x -> [B, L, d_model]
        x_tokens = self.x_proj(x).view(B, self.x_tokens, self.d_model)
        x_tokens = self.x_norm(x_tokens)

        # Encode condition into C tokens
        ctx = self.cond_encoder(cond).view(B, self.cond_tokens, self.d_model)
        ctx = self.cond_norm(ctx)

        # Build pooled cond embedding (for FiLM)
        if self.use_timestep:
            assert (
                t is not None
            ), "timestep tensor t:[B] must be provided when use_timestep=True"
            t_emb = sinusoidal_time_embed(t, self.t_embed_dim)
            t_emb = self.t_mlp(t_emb)
            pooled = torch.cat([cond, t_emb], dim=-1)
        else:
            pooled = cond
        cond_embed = self.cond_pool(pooled)

        # Transformer + FiLM modulation
        h = x_tokens
        for blk in self.blocks:
            # pre-block FiLM (can also try post-block)
            h = self.film(h, cond_embed)
            # checkpoint
            h = checkpoint(blk, h, ctx, use_reentrant=False)
            # h = blk(h, ctx)

        h = self.out_norm(h)
        y = self.out_proj(h.reshape(B, self.x_tokens * self.d_model))  # [B, D1]
        return y


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    B, D0, D1 = 4, 64, 256
    x = torch.randn(B, D1)
    cond = torch.randn(B, D0)
    t = torch.rand(B)  # uniform in [0,1]

    net = DiffusionCondAttentionNet(
        d_in=D1,
        d_cond=D0,
        d_model=256,
        n_heads=8,
        d_ff=1024,
        num_layers=4,
        x_tokens=128,
        cond_tokens=128,
        use_timestep=True,
        dropout=0.1,
    )
    y = net(x, cond, t)  # y: [B, D1]
    print(y.shape)
