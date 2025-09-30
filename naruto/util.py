import torch
import torch.nn as nn
import random


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d)) if elementwise_affine else None

    def forward(self, x):
        n = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(n + self.eps)
        if self.weight is not None:
            x = x * self.weight
        return x


def _build_inv_freq(dim_half: int, base: float = 10000.0, dtype=torch.float32, device=None):
    return 1.0 / (base ** (torch.arange(0, dim_half, device=device, dtype=dtype) / dim_half))


def apply_rope_1d(q: torch.Tensor, k: torch.Tensor, pos: torch.Tensor):
    """
    q,k: [B,H,T,Dh]
    pos: [T] ints 0..T-1
    """
    B, H, T, Dh = q.shape
    assert Dh % 2 == 0
    dim_half = Dh // 2
    inv = _build_inv_freq(dim_half, device=q.device, dtype=q.dtype)
    ang = torch.einsum("t,d->td", pos.to(q.dtype), inv)  # [T,dim_half]
    sin = ang.sin()[None, None, :, :]
    cos = ang.cos()[None, None, :, :]

    def rot(x):
        x1, x2 = x[..., :dim_half], x[..., dim_half:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    return rot(q), rot(k)


def apply_rope_2d(q: torch.Tensor, k: torch.Tensor, pos_h: torch.Tensor, pos_w: torch.Tensor):
    """
    2D RoPE by splitting the head-dim in half: first half uses row positions, second uses col positions.
    q,k: [N, H, T, Dh]
    pos_h,pos_w: [T] integer row/col indices (0..Hf-1 / 0..Wf-1)
    """
    N, Hh, T, Dh = q.shape
    assert Dh % 2 == 0
    Dh2 = Dh // 2

    qh, qw = q[..., :Dh2], q[..., Dh2:]
    kh, kw = k[..., :Dh2], k[..., Dh2:]

    def rope_half(qx, kx, pos_axis):
        assert qx.size(-1) % 2 == 0
        d2 = qx.size(-1) // 2
        inv = _build_inv_freq(d2, device=qx.device, dtype=qx.dtype)
        ang = torch.einsum("t,d->td", pos_axis.to(qx.dtype), inv)  # [T, d2]
        sin, cos = ang.sin()[None, None, :, :], ang.cos()[None, None, :, :]

        def rot(x):
            x1, x2 = x[..., :d2], x[..., d2:]
            return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

        return rot(qx), rot(kx)

    qh, kh = rope_half(qh, kh, pos_h)
    qw, kw = rope_half(qw, kw, pos_w)
    return torch.cat([qh, qw], dim=-1), torch.cat([kh, kw], dim=-1)


def build_hw_positions(H: int, W: int, device, dtype):
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    return yy.reshape(-1), xx.reshape(-1)  # [T], [T]

# --- helpers that work directly with sigma instead of t ---

def freq_radius_grid(H, W, device):
    fy = torch.fft.fftfreq(H, d=1.0).to(device)   # [-0.5,0.5)
    fx = torch.fft.rfftfreq(W, d=1.0).to(device)  # [0,0.5]
    fy = fy[:, None]
    fx = fx[None, :]
    r = torch.sqrt((fy / 0.5) ** 2 + (fx / 0.5) ** 2)
    r = (r / (r.max() + 1e-8)).clamp(0, 1)
    return r

def gaussian_mask_sigma(r, sigma):
    # M(r; sigma) = exp(-(r/sigma)^2), broadcast over spatial freq
    return torch.exp(- (r / (sigma + 1e-8))**2)

def blur_with_sigma(x_01, sigma, r):
    B, C, H, W = x_01.shape
    X = torch.fft.rfft2(x_01, norm="ortho")
    M = gaussian_mask_sigma(r[None, None], sigma).to(X.dtype)
    Xf = X * M
    x_blur = torch.fft.irfft2(Xf, s=(H, W), norm="ortho").clamp(0, 1)
    return x_blur

def img_to_model_space(img_01: torch.Tensor) -> torch.Tensor:
    return img_01 * 2.0 - 1.0

def model_to_img_space(img_m1_1: torch.Tensor) -> torch.Tensor:
    return (img_m1_1.clamp(-1, 1) + 1.0) / 2.0

# Deterministic encoder (use mean/mode; no sampling)
@torch.no_grad()
def vae_encode_mode(vae, img_bchw_01: torch.Tensor) -> torch.Tensor:
    x = img_to_model_space(img_bchw_01)
    posterior = vae.encode(x).latent_dist
    z = posterior.mode() if hasattr(posterior, "mode") else posterior.mean
    SCALE = vae.config.scaling_factor
    return z * SCALE

@torch.no_grad()
def vae_decode(vae, z_bchw: torch.Tensor) -> torch.Tensor:
    SCALE = vae.config.scaling_factor
    z_bchw = z_bchw / SCALE
    x_hat = vae.decode(z_bchw).sample
    return model_to_img_space(x_hat)


def blur_with_sigma_schedule(img, ts, device=None, BETA = 0.7, SIGMA_MIN = 0.005, SIGMA_MAX = 1.50, BSEARCH_ITERS = 32):
    device = img.device if device is None else device
    # power spectrum of the input (sum over channels)
    H, W = img.shape[-2:]
    r = freq_radius_grid(H, W, device)
    X = torch.fft.rfft2(img, norm="ortho")                 # [1,3,H,W//2+1]
    P = (X.abs()**2).sum(dim=1, keepdim=True)              # [1,1,H,W//2+1]

    # cumulative passed energy E(sigma) = sum( M(sigma)^2 * P )
    def cumulative_energy(sigma):
        M = gaussian_mask_sigma(r[None, None], sigma).to(P.dtype)
        return ( (M*M) * P ).sum()

    E_min = cumulative_energy(SIGMA_MIN)
    E_max = cumulative_energy(SIGMA_MAX)
    # guard against degenerate cases
    if (E_max - E_min).abs() < 1e-12:
        raise RuntimeError("Energy range too small; check image or sigma bounds.")

    # target energy fractions from 0..1
    targets = ts.pow(BETA)

    # Replace your previous `targets = ...` with:
    # targets = bi_accel_schedule(N_STEPS, device=device, gamma=2.5)
    # print(f"Targets: {targets}")
    # binary search sigmas so that (E(sigma)-E_min)/(E_max-E_min) ~= target
    sigmas = []
    for t in targets:
        lo, hi = SIGMA_MIN, SIGMA_MAX
        for _ in range(BSEARCH_ITERS):
            mid = 0.5 * (lo + hi)
            Em = cumulative_energy(mid)
            frac = (Em - E_min) / (E_max - E_min)
            if frac < t:
                lo = mid
            else:
                hi = mid
        sigmas.append(0.5 * (lo + hi))
    sigmas = torch.tensor(sigmas, device=device)
    # --- generate frames using those sigmas
    frames = []
    for s in sigmas.tolist():
        x_t = blur_with_sigma(img, s, r)
        frames.append(x_t.squeeze(0))

    return torch.stack(frames, dim=0)


@torch.no_grad()
def vae_latent_jvp_via_infinite_diff(
    img, ts, vae, eps_list=[1e-2, 5e-3, 1e-3, 5e-4], return_zt_only=False
): 
    """
    Velocity for z_t = z(t).
    Returns vel_t.
    """
    img = img.to(vae.device)
    z = vae_encode_mode(vae, img)
    device, dtype = z.device, z.dtype
    zt_s = vae_encode_mode(vae, blur_with_sigma_schedule(img, ts, device=vae.device))

    if return_zt_only:
        return zt_s

    vel_t = []
    for t_f in ts:
        t_f = float(t_f)
        # Special cases
        # General case: try central differences, fall back to one-sided if needed
        dzdt_est, hs = [], []
        for eps in eps_list:
            if (t_f - eps >= 0.0) and (t_f + eps <= 1.0):
                left_right_imgs = blur_with_sigma_schedule(img, ts=torch.tensor([t_f - eps, t_f + eps], device=device))
                vae_left, vae_right = vae_encode_mode(vae, left_right_imgs)
                dz = (vae_right - vae_left) / (2.0 * eps)
                dzdt_est.append(dz); hs.append(eps)
            else:
                raise Exception(f"eps {eps} is out of range for t_f {t_f}")
            # elif t_f - eps >= 0.0:  # backward diff
            #     z_m = vae_encode_mode(vae, blur_with_sigma_schedule(img, ts=torch.tensor([t_f - eps], device=device)))
            #     dz = (z - z_m) / eps
            #     dzdt_est.append(dz[0]); hs.append(eps)
            # elif t_f + eps <= 1.0:  # forward diff
            #     z_p = vae_encode_mode(vae, blur_with_sigma_schedule(img, ts=torch.tensor([t_f + eps], device=device)))
            #     dz = (z_p - z) / eps
            #     dzdt_est.append(dz[0]); hs.append(eps)

        # Combine with Richardson extrapolation if we have >1 estimate
        dzdt = dzdt_est[0]
        for i in range(1, len(dzdt_est)):
            D_coarse, D_fine = dzdt, dzdt_est[i]
            r = hs[i-1] / hs[i]; p = 2.0
            coef = (r**p) / (r**p - 1.0 + 1e-12)
            dzdt = coef * (D_fine - D_coarse / (r**p))

        vel_t.append(dzdt)

    return zt_s, torch.stack(vel_t, dim=0)


def random_unique_timesteps(n, low=0.0101, high=0.989, n_levels=None):
    points = set()
    levels = set()
    while len(points) < n:
        sample = round(random.uniform(low, high), 6)
        if n_levels is not None:
            assert isinstance(n_levels, int) and n_levels > 0
            current_level = max(min(n_levels, round(sample * n_levels)), 1)
            if current_level not in levels:
                levels.add(current_level)
                points.add(sample)
        else:
            points.add(sample)
    return list(points)
