import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
from timm.utils.model_ema import ModelEmaV3

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.diffusion.mnist_cls_diff import UNET, DDPM_Scheduler, NUM_CLASSES, NULL_CLASS

CLASSIFIER_PATH = "/u/zup7mn/Classes/NN/digit4/state/diffusion/checkpoints/mnist_mixup_classifier.pth"


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
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
    classifier.eval()
    return classifier


def generate(model, classifier, beta, alpha, swap_every_n, n_samples=10000, batch_size=100,
             num_time_steps=1000, guidance_scale=3.0, device=None):
    all_images = []
    all_scores = []
    n_batches = (n_samples + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        bs = min(batch_size, n_samples - batch_idx * batch_size)
        print(f"  batch {batch_idx + 1}/{n_batches} ({bs} samples)...")

        with torch.no_grad():
            z = torch.randn(bs, 1, 32, 32, device=device)

            coin = torch.randint(0, 2, (bs,), device=device)
            start_cls = torch.where(coin == 0, torch.full((bs,), 2, device=device, dtype=torch.long),
                                                torch.full((bs,), 6, device=device, dtype=torch.long))
            end_cls   = torch.where(coin == 0, torch.full((bs,), 6, device=device, dtype=torch.long),
                                                torch.full((bs,), 2, device=device, dtype=torch.long))
            null_cls  = torch.full((bs,), NULL_CLASS, device=device, dtype=torch.long)

            for t in reversed(range(1, num_time_steps)):
                step_from_start = num_time_steps - t
                classes = start_cls if (step_from_start // swap_every_n) % 2 == 0 else end_cls

                t_tensor  = torch.full((bs,), t, device=device)
                z_double  = torch.cat([z, z])
                t_double  = torch.cat([t_tensor, t_tensor])
                c_double  = torch.cat([classes, null_cls])

                eps_combined         = model(z_double, t_double, c_double)
                eps_cond, eps_uncond = eps_combined.chunk(2, dim=0)
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

                temp = beta[t] / (torch.sqrt(1 - alpha[t]) * torch.sqrt(1 - beta[t]))
                z = (1 / torch.sqrt(1 - beta[t])) * z - temp * eps
                z = z + torch.randn_like(z) * torch.sqrt(beta[t])

            # Final step t=0
            step_from_start = num_time_steps
            final_cls = start_cls if (step_from_start // swap_every_n) % 2 == 0 else end_cls

            t_tensor  = torch.zeros(bs, dtype=torch.long, device=device)
            z_double  = torch.cat([z, z])
            t_double  = torch.cat([t_tensor, t_tensor])
            c_double  = torch.cat([final_cls, null_cls])

            eps_combined         = model(z_double, t_double, c_double)
            eps_cond, eps_uncond = eps_combined.chunk(2, dim=0)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            temp = beta[0] / (torch.sqrt(1 - alpha[0]) * torch.sqrt(1 - beta[0]))
            x = (1 / torch.sqrt(1 - beta[0])) * z - temp * eps

            x = x[:, :, 2:30, 2:30]

            logits = classifier(x)
            scores = logits[:, 2] * logits[:, 6]

            all_images.append(x.cpu())
            all_scores.append(scores.cpu())

    return torch.cat(all_images, dim=0), torch.cat(all_scores, dim=0)


if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    HERE = Path(__file__).resolve().parent
    # Find the index of the last /src/ chunk.
    src_ind = len(HERE.parts) - (HERE.parts[::-1].index('src') + 1)
    HERE_STATE = Path(*HERE.parts[:src_ind]).joinpath('state', *HERE.parts[src_ind + 1:])

    ckpt_path = os.path.join(HERE_STATE, '..', 'checkpoints', 'ddpm_class')
    out_dir = HERE_STATE

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

    classifier = build_classifier(device)

    scheduler = DDPM_Scheduler(num_time_steps=1000)
    beta  = scheduler.beta.to(device)
    alpha = scheduler.alpha.to(device)

    for n in [8, 4, 2]:
        print(f"\n=== swap_every_n={n} ===")
        images, scores = generate(model, classifier, beta, alpha, swap_every_n=n, device=device)
        images_path = out_dir / f'generated_swap_N{n}.pt'
        scores_path = out_dir / f'scores_swap_N{n}.pt'
        torch.save(images, images_path)
        torch.save(scores, scores_path)
        print(f"Saved {tuple(images.shape)} to {images_path}")
        print(f"Saved {tuple(scores.shape)} scores to {scores_path}")
