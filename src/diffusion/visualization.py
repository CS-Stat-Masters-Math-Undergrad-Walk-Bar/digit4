import torch
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from timm.utils import ModelEmaV3

# Add src to path so we can import from src.diffusion.mnist_cls_diff
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.diffusion.mnist_cls_diff import UNET, DDPM_Scheduler, NUM_CLASSES, NULL_CLASS

def visualize(checkpoint_path: str, num_time_steps: int=1000, guidance_scale: float=3.0, samples_per_class: int=2, ema_decay: float=0.9999):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = UNET(num_classes=NUM_CLASSES).to(device)
    
    if 'weights' in checkpoint:
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['weights'].items()}
        model.load_state_dict(state_dict, strict=False)
        
    ema = ModelEmaV3(model, decay=ema_decay)
    if 'ema' in checkpoint:
        ema_state = {k.replace('module.module.', 'module.').replace('module.', ''): v for k, v in checkpoint['ema'].items()}
        ema.load_state_dict(ema_state, strict=False)

    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    beta = scheduler.beta.to(device)
    alpha = scheduler.alpha.to(device)

    # Generate samples_per_class images per class (digits 0-9)
    classes = torch.arange(NUM_CLASSES, device=device).repeat(samples_per_class)
    null_classes = torch.full((NUM_CLASSES * samples_per_class,), NULL_CLASS, device=device, dtype=torch.long)
    batch_size = NUM_CLASSES * samples_per_class

    with torch.no_grad():
        model = ema.module.eval()
        z = torch.randn(batch_size, 1, 32, 32, device=device)

        for t in reversed(range(1, num_time_steps)):
            t_tensor = torch.full((batch_size,), t, device=device)

            z_double = torch.cat([z, z], dim=0)
            t_double = torch.cat([t_tensor, t_tensor], dim=0)
            c_double = torch.cat([classes, null_classes], dim=0)

            eps_combined = model(z_double, t_double, c_double)
            eps_cond, eps_uncond = eps_combined.chunk(2, dim=0)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            temp = beta[t] / (torch.sqrt(1 - alpha[t]) * torch.sqrt(1 - beta[t]))
            z = (1 / torch.sqrt(1 - beta[t])) * z - temp * eps
            e = torch.randn_like(z)
            z = z + e * torch.sqrt(beta[t])

        # Final step (t=0)
        t_tensor = torch.zeros(batch_size, dtype=torch.long, device=device)
        z_double = torch.cat([z, z], dim=0)
        t_double = torch.cat([t_tensor, t_tensor], dim=0)
        c_double = torch.cat([classes, null_classes], dim=0)
        eps_combined = model(z_double, t_double, c_double)
        eps_cond, eps_uncond = eps_combined.chunk(2, dim=0)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        temp = beta[0] / (torch.sqrt(1 - alpha[0]) * torch.sqrt(1 - beta[0]))
        x = (1 / torch.sqrt(1 - beta[0])) * z - temp * eps

        fig, axes = plt.subplots(samples_per_class, NUM_CLASSES, figsize=(2 * NUM_CLASSES, 2 * samples_per_class))
        for i in range(samples_per_class):
            for j in range(NUM_CLASSES):
                # x is ordered exactly as [0, 1, ..., 9, 0, 1, ..., 9] based on repeat
                idx = i * NUM_CLASSES + j
                img = x[idx].cpu().squeeze(0).numpy()
                ax = axes[i, j]
                ax.imshow(img, cmap='gray')
                if i == 0:
                    ax.set_title(f'Class {j}')
                ax.axis('off')
        
        plt.tight_layout()
        out_path = os.path.join(os.path.dirname(__file__), 'diffusion_samples.png')
        plt.savefig(out_path)
        print(f"Saved visualization to {out_path}")
        plt.show()

if __name__ == '__main__':
    checkpoint = os.path.join(os.path.dirname(__file__), 'checkpoints', 'ddpm_class')
    print(f"Loading checkpoint from: {checkpoint}")
    visualize(checkpoint, samples_per_class=2)
