import torch
import matplotlib.pyplot as plt
import sys
import os
import math
from timm.utils import ModelEmaV3

# Add src to path so we can import from src.diffusion.mnist_cls_diff
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.diffusion.mnist_cls_diff import UNET, DDPM_Scheduler, NUM_CLASSES, NULL_CLASS

def generate_alternating(checkpoint_path, out_path, num_time_steps=1000, guidance_scale=3.0, n_samples=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = UNET(num_classes=NUM_CLASSES).to(device)
    
    if 'weights' in checkpoint:
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['weights'].items()}
        model.load_state_dict(state_dict, strict=False)
        
    ema = ModelEmaV3(model, decay=0.9999)
    if 'ema' in checkpoint:
        ema_state = {k.replace('module.module.', 'module.').replace('module.', ''): v for k, v in checkpoint['ema'].items()}
        ema.load_state_dict(ema_state, strict=False)

    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    beta = scheduler.beta.to(device)
    alpha = scheduler.alpha.to(device)
    
    null_classes = torch.full((n_samples,), NULL_CLASS, device=device, dtype=torch.long)
    
    with torch.no_grad():
        model = ema.module.eval()
        z = torch.randn(n_samples, 1, 32, 32, device=device)

        for t in reversed(range(1, num_time_steps)):
            t_tensor = torch.full((n_samples,), t, device=device)
            
            # Switch between class 2 and 6 every 10 steps
            step_from_start = num_time_steps - t
            current_class = 2 if (step_from_start // 10) % 2 == 0 else 6
            classes = torch.full((n_samples,), current_class, device=device, dtype=torch.long)

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
        t_tensor = torch.zeros(n_samples, dtype=torch.long, device=device)
        current_class = 2 if (num_time_steps // 10) % 2 == 0 else 6
        classes = torch.full((n_samples,), current_class, device=device, dtype=torch.long)
        
        z_double = torch.cat([z, z], dim=0)
        t_double = torch.cat([t_tensor, t_tensor], dim=0)
        c_double = torch.cat([classes, null_classes], dim=0)
        
        eps_combined = model(z_double, t_double, c_double)
        eps_cond, eps_uncond = eps_combined.chunk(2, dim=0)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        temp = beta[0] / (torch.sqrt(1 - alpha[0]) * torch.sqrt(1 - beta[0]))
        x = (1 / torch.sqrt(1 - beta[0])) * z - temp * eps
        
        # Visualize samples
        cols = 10
        rows = math.ceil(n_samples / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols*1.5, rows*1.5))
        axes = axes.flatten()
        for i in range(n_samples):
            img = x[i].cpu().squeeze(0).numpy()
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.savefig(out_path)
        print(f"Saved alternating 2/6 visualization to {out_path}")

if __name__ == '__main__':
    ckpt = os.path.join(os.path.dirname(__file__), 'checkpoints', 'ddpm_class')
    out = os.path.join(os.path.dirname(__file__), 'alternating_2_6.png')
    generate_alternating(ckpt, out, n_samples=50)
