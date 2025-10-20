import os
import math
import torch
import argparse
import random
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from datasets import load_dataset
import numpy as np

from diffusers import AutoencoderKL
from util import vae_encode_mode, vae_decode, random_unique_timesteps, vae_latent_jvp_via_infinite_diff
import torchvision.transforms.functional as TF
# Your modules
from q_former import QformerEncoder
from flow import MMDiTConfig, DualStreamMMDiT
from functools import partial
# ---- Utilities (small local helpers) -----------------------------------------

# Classic fixed-step RK4 integrator for x' = f(x, t)
@torch.no_grad()
def rk4_integrate(x0, t0, t1, steps, f, condition=None):
    x = x0
    count = None
    ts = torch.linspace(t0.item(), t1.item(), steps + 1, device=x.device)
    for i in range(steps):
        tt = ts[i].reshape(1)
        cond = f(x, tt, condition=condition, count=count)#
        # uncond = f(x, ts[i].reshape(1), condition=torch.zeros_like(condition), count=count)
        # vel = (cond - uncond) * 2.0 + uncond
        vel = cond
        x = x + vel * (ts[i+1] - ts[i])

        # # count += 1
        # tA, tB = ts[i], ts[i+1]
        # h = (tB - tA)
        # k1 = f(x, torch.tensor([tA], device=x.device))
        # k2 = f(x + 0.5 * h * k1, torch.tensor([tA + 0.5 * h], device=x.device))
        # k3 = f(x + 0.5 * h * k2, torch.tensor([tA + 0.5 * h], device=x.device))
        # k4 = f(x + h * k3, torch.tensor([tA + h], device=x.device))
        # x = x + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x

# ---- Model setup --------------------------------------------------------------

def build_models(device, dtype, vae_latent_channels, IMG_SIZE=512, VAE_SCALE_IN_SPATIAL=8,
                 COND_LEN=2048, COND_DIM=1536,
                 ENCODER_PATCH_SIZE=2, ENCODER_TIME_ADALN=True, N_LAYERS=32, ENCODER_IMAGE_STREAM_HIDDEN_SIZE=1536):
    modelA = QformerEncoder(
        patch_size=ENCODER_PATCH_SIZE,
        hidden_size=ENCODER_IMAGE_STREAM_HIDDEN_SIZE,
        num_heads=4,
        depth=N_LAYERS,
        K=COND_LEN,
        query_dim=COND_DIM,
        query_heads=8,
        bidirectional=False,
        in_channels=vae_latent_channels,
        input_size=IMG_SIZE // VAE_SCALE_IN_SPATIAL,
        gradient_checkpointing=False,
        time_adaln=ENCODER_TIME_ADALN
    ).to(device).to(dtype)

    cfgB = MMDiTConfig(
        C=vae_latent_channels,
        ctx_token_dim=COND_DIM,
        hidden_size_img=COND_DIM,
        hidden_size_ctx=COND_DIM,
        num_heads=16,
        depth=N_LAYERS,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=None,
        init_method=None,
    )
    modelB = DualStreamMMDiT(cfg=cfgB).to(device).to(dtype)
    modelB.use_checkpoint = False
    return modelA, modelB

def load_latest_if_any(ckpt_dir, modelA, modelB, map_location, filename='latest_512.pt'):
    latest = os.path.join(ckpt_dir, filename)
    if not os.path.exists(latest):
        return 0
    ckpt = torch.load(latest, map_location=map_location)
    modelA.load_state_dict(ckpt["modelA"], strict=False)
    modelB.load_state_dict(ckpt["modelB"], strict=False)
    print(f"Loaded checkpoint from {latest}")
    return int(ckpt.get("step", 0))

# ---- Main pipeline ------------------------------------------------------------

def _random_crop(img: Image.Image, crop_size: int) -> Image.Image:
    """Randomly crop square patch, resize down to crop_size if needed."""
    w, h = img.size
    if min(w, h) < crop_size:
        # upscale minimally so crop fits
        scale = crop_size / min(w, h)
        img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.LANCZOS)
        w, h = img.size
    if w == crop_size and h == crop_size:
        return img
    x = random.randint(0, w - crop_size)
    y = random.randint(0, h - crop_size)
    return img.crop((x, y, x + crop_size, y + crop_size))

@torch.no_grad()
def reconstruct_from_most_blur(
    device,
    save_path,
    img: Image.Image = None,
    div2k_index: int = 0,
    steps: int = 40,
    ckpt_dir: str = "checkpoints",
):
    dtype = torch.bfloat16

    # VAE
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", subfolder="vae"
    ).to(device).eval().to(torch.float32)
    ENCODER_IMAGE_STREAM_HIDDEN_SIZE = 1536

    # Build models
    IMG_SIZE = 480
    COND_LEN = 1024
    COND_DIM = 2304#1920
    N_LAYERS = 40
    modelA, modelB = build_models(
        device=device, dtype=dtype, vae_latent_channels=vae.config.latent_channels, IMG_SIZE=IMG_SIZE, COND_LEN=COND_LEN, COND_DIM=COND_DIM, N_LAYERS=N_LAYERS, ENCODER_IMAGE_STREAM_HIDDEN_SIZE=ENCODER_IMAGE_STREAM_HIDDEN_SIZE
    )
    _ = load_latest_if_any(ckpt_dir, modelA, modelB, map_location='cpu')#, filename='latest_lite.pt')

    # Load a sample image if not provided
    if img is None:
        ds = load_dataset("shivamsark/div2k", split="train", keep_in_memory=False)
        img = ds[div2k_index]["image"].convert("RGB")
        crop = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        # Always return IMG_SIZExIMG_SIZE (exact) without distortion
        if crop.size != (IMG_SIZE, IMG_SIZE):
            crop = crop.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        crop.save("crop.png")
        img = TF.to_tensor(crop).unsqueeze(0).to(device)

    # Encode to latent at t=0 (sharp/source)
    # z1 = vae_encode_mode(vae_bfloat16, img.to(torch.bfloat16)).to(torch.float32)
    z1 = vae_encode_mode(vae, img)
    z1_recover = vae_decode(vae, z1)[0].permute(1, 2, 0) * 255.0
    z1_recover = Image.fromarray(z1_recover.cpu().numpy().astype(np.uint8))
    z1_recover.save("z1_recover.png")

    with torch.autocast(device_type=device.type, dtype=dtype):
        condition = modelA(z1)

    images = []
    BEGIN_T = 0.1
    END_T = 1.0
    t0 = torch.tensor([BEGIN_T], device=device, dtype=torch.float32)

    # zt_s, vel_s = vae_latent_jvp_via_infinite_diff(img, t0, vae)
    zt_s = t0.view(1, 1, 1, 1) * z1 + (1 - t0.view(1, 1, 1, 1)) * torch.randn_like(z1)

    # Velocity field callback (x' = f(x,t))
    def vel(x_t, t_scalar, condition, count=None):
        # if count is not None:
        #     return vel_s_gt[count]
        # else:
        # Prepare conditioning from modelA at (x, t_cond = t)
        # NOTE: modelA expects bfloat16, but "t" stays float32 (like your training)
        # print(f"t_scalar: {t_scalar}")
        with torch.autocast(device_type=device.type, dtype=dtype):
            v = modelB(x_t=x_t, z_tok=condition, t=t_scalar)  # float32 output in your training
        return v.to(torch.float32)

    for visible_prefix in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        condition_visible = condition.clone()
        condition_visible[:, int(visible_prefix * COND_LEN) :] = 0.0
        zT = rk4_integrate(zt_s, t0, torch.tensor([END_T], device=device, dtype=torch.float32), steps, partial(vel), condition=condition_visible)
        images.append(vae_decode(vae, zT)[0].permute(1, 2, 0) * 255.0)
    
    images = torch.cat(images, dim=1)
    out = Image.fromarray(images.clamp(0, 255).round().cpu().numpy().astype(np.uint8))
    out = out.convert("RGB")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    out.save(save_path)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--steps", type=int, default=100, help="RK4 steps from t=1 → t=0")
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints")
    ap.add_argument("--out", type=str, default="reconstructed.png")
    ap.add_argument("--div2k_index", type=int, default=0, help="Which DIV2K val image to fetch")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    _ = reconstruct_from_most_blur(
        device=device,
        save_path=args.out,
        img=None,                     # None -> fetch one DIV2K validation image
        div2k_index=args.div2k_index,
        steps=args.steps,
        ckpt_dir=args.ckpt_dir,
    )
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()