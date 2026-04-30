import torch
import torch.nn as nn
import torch.multiprocessing as mp
import sys
import os
from timm.utils.model_ema import ModelEmaV3

from train_diffusion import UNET, DDPM_Scheduler, NUM_CLASSES, NULL_CLASS
from config import ROOT, NOVELTY_CNN_MIXUP, DIFF_CKPT_PATH, DIFF_OUT_DIR

# Compositional diffusion modes.
# Each step we predict eps for c1=2, c2=6, and null. With deltas
#   d_i = eps_c_i - eps_uncond
# the guided eps is:
#   "average": eps_uncond + guidance * 0.5 * (d_1 + d_2)   — samples ~ p(x|c1)^.5 * p(x|c2)^.5
#   "product": eps_uncond + guidance * (d_1 + d_2)         — samples ~ p(x|c1) * p(x|c2)


MODES = [
    'average', 
    # 'product'
]
C1, C2 = 2, 6

GPU_IDS    = [1, 2, 3]   # GPU 0 is in use by another process — leave it alone
N_SAMPLES  = 10000
BATCH_SIZE = 200         # ~5 GB peak per RTX 4000


def build_classifier(device):
    classifier = nn.Sequential(
        nn.Conv2d(1, 128, kernel_size=5, stride=1, padding=1),
        nn.Dropout(0.2),
        nn.ReLU(),
        nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
        nn.Dropout(0.2),
        nn.Flatten(),
        nn.Linear(64 * 676, 10),
        nn.Softmax(dim=1)
    ).to(device)
    classifier.load_state_dict(torch.load(NOVELTY_CNN_MIXUP, map_location=device))
    classifier.eval()
    return classifier


def load_diffusion_model(device):
    checkpoint = torch.load(DIFF_CKPT_PATH, map_location=device)
    model = UNET(num_classes=NUM_CLASSES).to(device)
    if 'weights' in checkpoint:
        sd = {k.replace('module.', ''): v for k, v in checkpoint['weights'].items()}
        model.load_state_dict(sd, strict=False)
    ema = ModelEmaV3(model, decay=0.9999)
    if 'ema' in checkpoint:
        es = {k.replace('module.module.', 'module.').replace('module.', ''): v
              for k, v in checkpoint['ema'].items()}
        ema.load_state_dict(es, strict=False)
    return ema.module.eval()


def compose_eps(model, z, t_tensor, c1_cls, c2_cls, null_cls, mode, guidance_scale):
    z3 = torch.cat([z, z, z])
    t3 = torch.cat([t_tensor, t_tensor, t_tensor])
    c3 = torch.cat([c1_cls, c2_cls, null_cls])

    eps_all = model(z3, t3, c3)
    eps_c1, eps_c2, eps_uncond = eps_all.chunk(3, dim=0)

    d1 = eps_c1 - eps_uncond
    d2 = eps_c2 - eps_uncond

    if mode == 'average':
        return eps_uncond + guidance_scale * 0.5 * (d1 + d2)
    elif mode == 'product':
        return eps_uncond + guidance_scale * (d1 + d2)
    else:
        raise ValueError(f"unknown mode: {mode}")


def generate(model, classifier, beta, alpha, mode, n_samples, batch_size,
             num_time_steps=1000, guidance_scale=1.5, device=None, log_prefix=""):
    all_images, all_scores = [], []
    n_batches = (n_samples + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        bs = min(batch_size, n_samples - batch_idx * batch_size)
        print(f"{log_prefix}batch {batch_idx + 1}/{n_batches} ({bs} samples)...", flush=True)

        with torch.no_grad():
            z = torch.randn(bs, 1, 32, 32, device=device)

            c1_cls   = torch.full((bs,), C1, device=device, dtype=torch.long)
            c2_cls   = torch.full((bs,), C2, device=device, dtype=torch.long)
            null_cls = torch.full((bs,), NULL_CLASS, device=device, dtype=torch.long)

            for t in reversed(range(1, num_time_steps)):
                t_tensor = torch.full((bs,), t, device=device)
                eps = compose_eps(model, z, t_tensor, c1_cls, c2_cls, null_cls,
                                  mode, guidance_scale)
                temp = beta[t] / (torch.sqrt(1 - alpha[t]) * torch.sqrt(1 - beta[t]))
                z = (1 / torch.sqrt(1 - beta[t])) * z - temp * eps
                z = z + torch.randn_like(z) * torch.sqrt(beta[t])

            t_tensor = torch.zeros(bs, dtype=torch.long, device=device)
            eps = compose_eps(model, z, t_tensor, c1_cls, c2_cls, null_cls,
                              mode, guidance_scale)
            temp = beta[0] / (torch.sqrt(1 - alpha[0]) * torch.sqrt(1 - beta[0]))
            x = (1 / torch.sqrt(1 - beta[0])) * z - temp * eps
            x = x[:, :, 2:30, 2:30]

            logits = classifier(x)
            scores = logits[:, C1] * logits[:, C2]

            all_images.append(x.cpu())
            all_scores.append(scores.cpu())

    return torch.cat(all_images, dim=0), torch.cat(all_scores, dim=0)


def split_samples(total, world_size):
    base, rem = divmod(total, world_size)
    return [base + (1 if r < rem else 0) for r in range(world_size)]


def worker(rank, world_size, mode, n_samples, batch_size, gpu_ids, shard_dir):
    gpu_id = gpu_ids[rank]
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)
    torch.manual_seed(1234 + rank * 7919 + hash(mode) % 10_000)

    counts = split_samples(n_samples, world_size)
    my_count = counts[rank]
    log_prefix = f"[mode={mode} rank={rank} gpu={gpu_id}] "
    print(f"{log_prefix}generating {my_count} samples", flush=True)

    model = load_diffusion_model(device)
    classifier = build_classifier(device)

    scheduler = DDPM_Scheduler(num_time_steps=1000)
    beta  = scheduler.beta.to(device)
    alpha = scheduler.alpha.to(device)

    images, scores = generate(model, classifier, beta, alpha, mode=mode,
                              n_samples=my_count, batch_size=batch_size,
                              device=device, log_prefix=log_prefix)

    shard_path = shard_dir / f'shard_{mode}_rank{rank}.pt'
    torch.save({'images': images, 'scores': scores}, shard_path)
    print(f"{log_prefix}saved shard {tuple(images.shape)} -> {shard_path}", flush=True)


def merge_shards(mode, world_size, shard_dir, out_dir):
    images_chunks, scores_chunks = [], []
    for rank in range(world_size):
        shard_path = shard_dir / f'shard_{mode}_rank{rank}.pt'
        shard = torch.load(shard_path, map_location='cpu')
        images_chunks.append(shard['images'])
        scores_chunks.append(shard['scores'])
        os.remove(shard_path)
    images = torch.cat(images_chunks, dim=0)
    scores = torch.cat(scores_chunks, dim=0)
    images_path = out_dir / f'generated_compose_{mode}.pt'
    scores_path = out_dir / f'scores_compose_{mode}.pt'
    torch.save(images, images_path)
    torch.save(scores, scores_path)
    print(f"merged {tuple(images.shape)} -> {images_path}")
    print(f"merged {tuple(scores.shape)} -> {scores_path}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    world_size = len(GPU_IDS)
    shard_dir = DIFF_OUT_DIR / 'shards'
    os.makedirs(shard_dir, exist_ok=True)

    for mode in MODES:
        print(f"\n=== mode={mode}  (world_size={world_size}, gpus={GPU_IDS}) ===")
        mp.spawn(worker,
                 args=(world_size, mode, N_SAMPLES, BATCH_SIZE, GPU_IDS, shard_dir),
                 nprocs=world_size, join=True)
        merge_shards(mode, world_size, shard_dir, DIFF_OUT_DIR)

    try:
        os.rmdir(shard_dir)
    except OSError:
        pass
