import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from timm.utils import ModelEmaV3

# Add src to path so we can import from src.diffusion.mnist_cls_diff
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.diffusion.mnist_cls_diff import UNET, DDPM_Scheduler, NUM_CLASSES, NULL_CLASS

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
    
    state_dict = torch.load("/u/zup7mn/Classes/NN/digit4/src/mnist_classifier/mnist_mixup_classifier.pth", map_location=device)
    classifier.load_state_dict(state_dict)
    classifier.eval()
    return classifier

def generate_and_score(
    checkpoint_path: str, 
    num_time_steps: int = 1000, 
    guidance_scale: float = 3.0, 
    ema_decay: float = 0.9999,
    total_samples: int = 1000,
    top_k: int = 20
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load classifier
    print("Loading classifier...")
    classifier = build_classifier(device)
    
    # Load diffusion model
    print("Loading diffusion model...")
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

    # We will generate in batches to not OOM
    batch_size = 100
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    all_images = []
    all_scores = []
    
    print(f"Generating {total_samples} samples...")
    model = ema.module.eval()
    
    with torch.no_grad():
        for b_idx in range(num_batches):
            current_batch_size = min(batch_size, total_samples - b_idx * batch_size)
            
            # Generate random classes
            classes = torch.randint(0, NUM_CLASSES, (current_batch_size,), device=device)
            null_classes = torch.full((current_batch_size,), NULL_CLASS, device=device, dtype=torch.long)
            
            z = torch.randn(current_batch_size, 1, 32, 32, device=device)

            for t in reversed(range(1, num_time_steps)):
                t_tensor = torch.full((current_batch_size,), t, device=device)

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
            t_tensor = torch.zeros(current_batch_size, dtype=torch.long, device=device)
            z_double = torch.cat([z, z], dim=0)
            t_double = torch.cat([t_tensor, t_tensor], dim=0)
            c_double = torch.cat([classes, null_classes], dim=0)
            
            eps_combined = model(z_double, t_double, c_double)
            eps_cond, eps_uncond = eps_combined.chunk(2, dim=0)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            temp = beta[0] / (torch.sqrt(1 - alpha[0]) * torch.sqrt(1 - beta[0]))
            x = (1 / torch.sqrt(1 - beta[0])) * z - temp * eps
            
            # x is shape [batch, 1, 32, 32]. Note: The classifier was trained on 28x28 (or 24x24?). 
            # In notebook for mixup classifier it uses 28x28? Wait, nn.Linear(64*676) = 64 * 26 * 26.
            # 28x28 -> conv5 (no padding padding=1) -> 28x28? wait, pool?
            # The classifier notebook actually does: model(torch.randn((1,1,28,28)))
            # If so, we need to center crop or resize x from 32x32 to 28x28
            import torchvision.transforms.functional as TF
            x_28 = torch.nn.functional.interpolate(x, size=(28, 28), mode='bilinear', align_corners=False)
            
            # Score images
            logits = classifier(x_28)
            # Metric: prob of class 2 * prob of class 6
            scores = logits[:, 2] * logits[:, 6]
            
            all_images.append(x.cpu())
            all_scores.append(scores.cpu())
            
            print(f"Batch {b_idx+1}/{num_batches} done.")
            
    all_images = torch.cat(all_images, dim=0)
    all_scores = torch.cat(all_scores, dim=0)
    
    # Get top K
    top_scores, top_indices = torch.topk(all_scores, top_k)
    
    print(f"Top {top_k} scores: {top_scores.tolist()}")
    
    # Visualize top K
    cols = 5
    rows = (top_k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    axes = axes.flatten()
    
    for i in range(top_k):
        idx = top_indices[i]
        img = all_images[idx].squeeze(0).numpy()
        score = top_scores[i].item()
        
        ax = axes[i]
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Score: {score:.4f}')
        ax.axis('off')
        
    for i in range(top_k, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'top_scored_samples.png')
    plt.savefig(out_path)
    print(f"Saved top {top_k} visualization to {out_path}")

if __name__ == '__main__':
    checkpoint = os.path.join(os.path.dirname(__file__), 'checkpoints', 'ddpm_class')
    generate_and_score(checkpoint, total_samples=1000, top_k=20)
