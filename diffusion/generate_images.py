import torch
import sys
import os
from timm.utils.model_ema import ModelEmaV3

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.diffusion.mnist_cls_diff import UNET, DDPM_Scheduler, NUM_CLASSES, NULL_CLASS


def generate(model, beta, alpha, swap_after_n, n_samples=10000, batch_size=100,
             num_time_steps=1000, guidance_scale=3.0, device=None):
    all_images = []
    n_batches = (n_samples + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        bs = min(batch_size, n_samples - batch_idx * batch_size)
        print(f"  batch {batch_idx + 1}/{n_batches} ({bs} samples)...")

        with torch.no_grad():
            z = torch.randn(bs, 1, 32, 32, device=device)

            # Each image independently starts on class 2 or 6 at random
            coin = torch.randint(0, 2, (bs,), device=device)  # 0 or 1
            start_cls = torch.where(coin == 0, torch.full((bs,), 2, device=device),
                                                torch.full((bs,), 6, device=device))
            end_cls   = torch.where(coin == 0, torch.full((bs,), 6, device=device),
                                                torch.full((bs,), 2, device=device))
            null_cls  = torch.full((bs,), NULL_CLASS, device=device, dtype=torch.long)

            for t in reversed(range(1, num_time_steps)):
                step = num_time_steps - t  # 1 on the first denoising step
                classes = start_cls if step <= swap_after_n else end_cls

                t_tensor  = torch.full((bs,), t, device=device)
                z_double  = torch.cat([z, z])
                t_double  = torch.cat([t_tensor, t_tensor])
                c_double  = torch.cat([classes, null_cls])

                eps_combined        = model(z_double, t_double, c_double)
                eps_cond, eps_uncond = eps_combined.chunk(2, dim=0)
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

                temp = beta[t] / (torch.sqrt(1 - alpha[t]) * torch.sqrt(1 - beta[t]))
                z = (1 / torch.sqrt(1 - beta[t])) * z - temp * eps
                z = z + torch.randn_like(z) * torch.sqrt(beta[t])

            # Final step t=0 — always using end_cls (step=1000 > any swap_after_n here)
            t_tensor  = torch.zeros(bs, dtype=torch.long, device=device)
            z_double  = torch.cat([z, z])
            t_double  = torch.cat([t_tensor, t_tensor])
            c_double  = torch.cat([end_cls, null_cls])

            eps_combined        = model(z_double, t_double, c_double)
            eps_cond, eps_uncond = eps_combined.chunk(2, dim=0)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            temp = beta[0] / (torch.sqrt(1 - alpha[0]) * torch.sqrt(1 - beta[0]))
            x = (1 / torch.sqrt(1 - beta[0])) * z - temp * eps

            # Crop 32x32 → 28x28 (undo the padding added during training)
            x = x[:, :, 2:30, 2:30]
            all_images.append(x.cpu())

    return torch.cat(all_images, dim=0)


if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    ckpt_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'ddpm_class')
    out_dir   = os.path.dirname(__file__)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model = UNET(num_classes=NUM_CLASSES).to(device)
    if 'weights' in checkpoint:
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['weights'].items()}
        model.load_state_dict(state_dict, strict=False)

    ema = ModelEmaV3(model, decay=0.9999)
    if 'ema' in checkpoint:
        ema_state = {k.replace('module.module.', 'module.').replace('module.', ''): v
                     for k, v in checkpoint['ema'].items()}
        ema.load_state_dict(ema_state, strict=False)

    model = ema.module.eval()

    scheduler = DDPM_Scheduler(num_time_steps=1000)
    beta  = scheduler.beta.to(device)
    alpha = scheduler.alpha.to(device)

    for n in [4, 2]:
        print(f"\n=== swap_after_n={n} ===")
        images = generate(model, beta, alpha, swap_after_n=n, device=device)
        out_path = os.path.join(out_dir, f'generated_swap_N{n}.pt')
        torch.save(images, out_path)
        print(f"Saved {tuple(images.shape)} to {out_path}")
