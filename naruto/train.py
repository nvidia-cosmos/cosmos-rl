import os
import argparse
from flow import MMDiTConfig, DualStreamMMDiT
# from encoder import ModelA_ViT, ModelACfg
from q_former import QformerEncoder
import random
import torch
import torch.distributed as dist
from diffusers import AutoencoderKL
from util import vae_encode_mode, vae_decode
from torch.nn.parallel import DistributedDataParallel as DDP
import itertools
from data import make_loader
from huggingface_hub import hf_hub_download

# --- NEW: wandb + checkpoint helpers ---
try:
    import wandb
except ImportError:
    wandb = None

def is_main_process() -> bool:
    return (not dist.is_initialized()) or (dist.get_rank() == 0)

def save_checkpoint(
    out_dir: str,
    step: int,
    modelA: DDP,
    modelB: DDP,
    optimizer: torch.optim.Optimizer,
    extra: dict,
    cond_len: int,
    cond_dim: int,
    layers: int,
):
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {
        "step": step,
        "modelA": modelA.module.state_dict(),
        "modelB": modelB.module.state_dict(),
        # "optimizer": optimizer.state_dict(),
        "extra": extra,
        "cond_len": cond_len,
        "cond_dim": cond_dim,
        "layers": layers,
    }
    path = os.path.join(out_dir, f"checkpoint_step{step}.pt")
    latest = os.path.join(out_dir, "latest.pt")
    torch.save(ckpt, path)
    # atomic "latest" update
    tmp = latest + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, latest)
    return path, latest


def get_hf_latest_ckpt(repo_id="Jiaxincc/naruto-beta", filename="latest.pt"):
    """
    Downloads latest.pt from HF once on rank 0, broadcasts the local path to all ranks.
    Returns the local filesystem path to the checkpoint file.
    """
    path = None
    if (not dist.is_initialized()) or dist.get_rank() == 0:
        # If you want faster transfer, you can set HF_HUB_ENABLE_HF_TRANSFER=1 and install hf_transfer.
        path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model", revision="ff67e8509d5e94b87d6c1e08dcacd5b9e5fe555f") #10489cf9848d87667455c6fcb46303a11a2c4e1d")
    return path

def load_checkpoint(
    path: str,
    modelA: DDP,
    modelB: DDP,
    optimizer: torch.optim.Optimizer,
    map_location: str | torch.device,
):
    ckpt = torch.load(path, map_location=map_location)
    modelA.module.load_state_dict(ckpt["modelA"], strict=False)
    modelB.module.load_state_dict(ckpt["modelB"], strict=False)
    # if "optimizer" in ckpt:
    #     optimizer.load_state_dict(ckpt["optimizer"])
    step = int(ckpt.get("step", 0))
    extra = ckpt.get("extra", {})
    return step, extra

def init_wandb(rank, args, cfg, param_counts):
    # non-main ranks: disable W&B chatter
    if not wandb:
        return None, None
    if rank != 0:
        os.environ.setdefault("WANDB_MODE", "disabled")
        return None, None

    run_id = None
    if args.resume and os.path.exists(os.path.join(args.ckpt_dir, "wandb_run_id.txt")):
        with open(os.path.join(args.ckpt_dir, "wandb_run_id.txt"), "r") as f:
            run_id = f.read().strip() or None

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or None,
        id=run_id,
        resume="allow" if args.resume else None,
        config={
            "IMG_SIZE": cfg["IMG_SIZE"],
            "COND_LEN": cfg["COND_LEN"],
            "COND_DIM": cfg["COND_DIM"],
            "ENCODER_IMAGE_STREAM_HIDDEN_SIZE": cfg["ENCODER_IMAGE_STREAM_HIDDEN_SIZE"],
            "ENCODER_PATCH_SIZE": cfg["ENCODER_PATCH_SIZE"],
            "ENCODER_TIME_ADALN": cfg["ENCODER_TIME_ADALN"],
            "N": cfg["N"],
            "VAE_SCALE_IN_SPATIAL": cfg["VAE_SCALE_IN_SPATIAL"],
            "lr": args.lr,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "dtype": "bfloat16",
            "ddp_world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "param_count_M_modelA": param_counts["A_M"],
            "param_count_M_modelB": param_counts["B_M"],
        },
    )
    # persist run id for resume
    os.makedirs(args.ckpt_dir, exist_ok=True)
    with open(os.path.join(args.ckpt_dir, "wandb_run_id.txt"), "w") as f:
        f.write(run.id)
    return run, run.id

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img_size", type=int, default=480)
    p.add_argument("--patch_size", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--encoder_image_stream_hidden_size", type=int, default=1536)
    p.add_argument("--freeze_decoder_steps", type=int, default=0)
    p.add_argument("--N", type=int, default=4)
    p.add_argument("--cond_len", type=int, default=1024)
    p.add_argument("--cond_dim", type=int, default=1024)
    p.add_argument("--layers", type=int, default=24)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--save_every", type=int, default=500, help="Save every this many optimizer steps (global).")
    p.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if available.")
    # W&B
    p.add_argument("--wandb_project", type=str, default="div2k-mmdt")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # DDP
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")
    device_id = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{device_id}")
    dtype = torch.bfloat16

    vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", subfolder="vae").to(device).eval().to(torch.float32)

    IMG_SIZE = args.img_size
    COND_LEN = args.cond_len
    COND_DIM = args.cond_dim
    ENCODER_IMAGE_STREAM_HIDDEN_SIZE = args.encoder_image_stream_hidden_size
    ENCODER_PATCH_SIZE = args.patch_size
    ENCODER_TIME_ADALN = True
    N = args.N
    VAE_SCALE_IN_SPATIAL = 8

    # cfgA = ModelACfg(factor=2, embed_dim=1024, depth=6, heads=16, qkv_bias=True)
    # modelA = ModelA_ViT(vae.config.latent_channels, cfgA).to(device).to(dtype)

    modelA = QformerEncoder(
        patch_size=ENCODER_PATCH_SIZE,
        hidden_size=ENCODER_IMAGE_STREAM_HIDDEN_SIZE,
        num_heads=4,
        depth=args.layers,
        K=COND_LEN,
        query_dim=COND_DIM,
        query_heads=8,
        bidirectional=False,
        in_channels=vae.config.latent_channels,
        input_size=IMG_SIZE // VAE_SCALE_IN_SPATIAL,
        gradient_checkpointing=True,
        time_adaln=ENCODER_TIME_ADALN
    ).to(device).to(dtype)
    modelA = DDP(modelA, device_ids=[device_id])

    cfgB = MMDiTConfig(
        C=vae.config.latent_channels,
        ctx_token_dim=COND_DIM,
        hidden_size_img=args.cond_dim,
        hidden_size_ctx=args.cond_dim,
        num_heads=16,
        depth=args.layers,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=None,
        init_method=None,
    )
    modelB = DualStreamMMDiT(cfg=cfgB).to(device).to(dtype)
    modelB.use_checkpoint = True
    modelB = DDP(modelB, device_ids=[device_id])

    # print param size in MB
    params_A_M = sum(p.numel() for p in modelA.parameters()) * 2 / 1024 / 1024
    params_B_M = sum(p.numel() for p in modelB.parameters()) * 2 / 1024 / 1024
    if is_main_process():
        print(f"Number of parameters in modelA: {params_A_M}")
        print(f"Number of parameters in modelB: {params_B_M}")
    param_counts = {"A_M": params_A_M, "B_M": params_B_M}

    # optimizer
    optimizer = torch.optim.AdamW(list(modelA.parameters()) + list(modelB.parameters()), lr=args.lr)
    optimizer.zero_grad()

    # W&B init
    cfg_for_wandb = dict(
        IMG_SIZE=IMG_SIZE, COND_LEN=COND_LEN, COND_DIM=COND_DIM,
        ENCODER_IMAGE_STREAM_HIDDEN_SIZE=ENCODER_IMAGE_STREAM_HIDDEN_SIZE,
        ENCODER_PATCH_SIZE=ENCODER_PATCH_SIZE, ENCODER_TIME_ADALN=ENCODER_TIME_ADALN,
        N=N, VAE_SCALE_IN_SPATIAL=VAE_SCALE_IN_SPATIAL
    )
    run, run_id = init_wandb(rank, args, cfg_for_wandb, param_counts)

    # --- Resume logic ---
    global_step = 0
    latest_path = os.path.join(args.ckpt_dir, "latest.pt")
    if args.resume:
        if is_main_process():
            if not os.path.exists(latest_path):
                latest_path = get_hf_latest_ckpt("Jiaxincc/fml", "latest.pt")
            # map checkpoints saved on cuda:0 to this rank's device
            map_loc = {"cuda:%d" % 0: "cuda:%d" % device_id}
            global_step, extra = load_checkpoint(latest_path, modelA, modelB, optimizer, map_location=map_loc)
        
        if dist.get_world_size() > 1:
            for param in modelA.parameters():
                dist.broadcast(param, src=0)
            for param in modelB.parameters():
                dist.broadcast(param, src=0)
            for param in optimizer.param_groups:
                for p in param["params"]:
                    dist.broadcast(p, src=0)
        print(f"[Resume] Loaded checkpoint from {latest_path} at step {global_step}")
    dist.barrier()

    # dataloader outside the loop to keep workers warm
    # train_loader = make_loader(split="train", batch_size=args.batch_size, IMG_SIZE=IMG_SIZE, num_workers=args.num_workers)
    train_loader = make_loader(
        bucket=os.getenv("R2_BUCKET"), 
        prefix=os.getenv("R2_PREFIX"), 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=IMG_SIZE
    )

    # --- Training ---
    modelA.train()
    modelB.train()

    global_step = 0
    if args.freeze_decoder_steps > 0:
        for param in modelB.parameters():
            param.requires_grad = False

    with torch.autocast(device_type=device.type, dtype=dtype):
        while True:
            if global_step > args.freeze_decoder_steps and args.freeze_decoder_steps > 0:
                for param in modelB.parameters():
                    param.requires_grad = True
            global_step += 1
            for img in train_loader:
                B, C, H, W = img.shape
                img = img.to(device, non_blocking=True)
                z1_s = vae_encode_mode(vae, img)
                # 1. sample gaussian noise
                noise = torch.randn_like(z1_s)
                # 2. sample timestep from 0 to 1
                t = torch.rand(B, device=device)
                input_xt = t.view(B, 1, 1, 1) * z1_s + (1 - t.view(B, 1, 1, 1)) * noise
                condition = modelA(z1_s)
                output = modelB(x_t=input_xt, z_tok=condition, t=t)
                loss = torch.nn.functional.mse_loss(output, (z1_s - noise))
                total_loss = loss
                total_loss.backward()
                loss = loss.detach()

                # clip gradient
                total_norm = torch.nn.utils.clip_grad_norm_(
                    itertools.chain(modelA.parameters(), modelB.parameters()),
                    max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # Reduce mean loss across ranks for logging/printing
                loss_for_log = loss.clone().detach()
                dist.all_reduce(loss_for_log, op=dist.ReduceOp.AVG)
                loss_item = loss_for_log.item()
                total_norm_for_log = total_norm.clone().detach()
                dist.all_reduce(total_norm_for_log, op=dist.ReduceOp.AVG)
                total_norm_item = total_norm_for_log.item()

                # Logging
                if is_main_process():
                    if wandb and run is not None:
                        wandb.log(
                            {"loss": loss_item, "lr": optimizer.param_groups[0]["lr"], "global_step": global_step, "total_norm": total_norm_item},
                            step=global_step,
                        )
                    # also keep stdout
                    if global_step % 10 == 0:
                        print(f"[step {global_step}] Loss: {loss_item:.6f}, Total Norm: {total_norm_item:.6f}")

                    # Checkpointing
                    if global_step % args.save_every == 0:
                        path, latest = save_checkpoint(
                            args.ckpt_dir,
                            global_step,
                            modelA,
                            modelB,
                            optimizer,
                            extra={"run_id": run_id},
                            cond_len=COND_LEN,
                            cond_dim=COND_DIM,
                            layers=args.layers,
                        )
                        print(f"[step {global_step}] Saved checkpoint: {path}")

            # (Optional) recreate the loader each "epoch" over the streaming dataset
            # to reshuffle worker RNG; DIV2K here is finite so we cycle.
            train_loader = make_loader(
                bucket=os.getenv("R2_BUCKET"), 
                prefix=os.getenv("R2_PREFIX"), 
                batch_size=args.batch_size, 
                num_workers=args.num_workers,
                image_size=IMG_SIZE
            )