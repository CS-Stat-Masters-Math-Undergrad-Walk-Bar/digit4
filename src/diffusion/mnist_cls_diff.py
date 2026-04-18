# %%
# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange #pip install einops
from typing import List
import random
import math
from torchvision import datasets, transforms
from torch.utils.data import DataLoader 
from timm.utils import ModelEmaV3 #pip install timm 
from tqdm import tqdm #pip install tqdm
import matplotlib.pyplot as plt #pip install matplotlib
import torch.optim as optim
import numpy as np

class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps:int, embed_dim: int):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.embeddings = embeddings

    def forward(self, x, t):
        embeds = self.embeddings[t.cpu()].to(x.device)
        return embeds[:, :, None, None]

# %%
# Residual Blocks
class ResBlock(nn.Module):
    def __init__(self, C: int, num_groups: int, dropout_prob: float):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.gnorm1 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.gnorm2 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout_prob, inplace=True)

    def forward(self, x, embeddings):
        x = x + embeddings[:, :x.shape[1], :, :]
        r = self.conv1(self.relu(self.gnorm1(x)))
        r = self.dropout(r)
        r = self.conv2(self.relu(self.gnorm2(r)))
        return r + x

# %%
class Attention(nn.Module):
    def __init__(self, C: int, num_heads:int , dropout_prob: float):
        super().__init__()
        self.proj1 = nn.Linear(C, C*3)
        self.proj2 = nn.Linear(C, C)
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

    def forward(self, x):
        h, w = x.shape[2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.proj1(x)
        x = rearrange(x, 'b L (C H K) -> K b H L C', K=3, H=self.num_heads)
        q,k,v = x[0], x[1], x[2]
        x = F.scaled_dot_product_attention(q,k,v, is_causal=False, dropout_p=self.dropout_prob)
        x = rearrange(x, 'b H (h w) C -> b h w (C H)', h=h, w=w)
        x = self.proj2(x)
        return rearrange(x, 'b h w C -> b C h w')

# %%
class UnetLayer(nn.Module):
    def __init__(self, 
            upscale: bool, 
            attention: bool, 
            num_groups: int, 
            dropout_prob: float,
            num_heads: int,
            C: int):
        super().__init__()
        self.ResBlock1 = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        self.ResBlock2 = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        if upscale:
            self.conv = nn.ConvTranspose2d(C, C//2, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(C, C*2, kernel_size=3, stride=2, padding=1)
        if attention:
            self.attention_layer = Attention(C, num_heads=num_heads, dropout_prob=dropout_prob)

    def forward(self, x, embeddings):
        x = self.ResBlock1(x, embeddings)
        if hasattr(self, 'attention_layer'):
            x = self.attention_layer(x)
        x = self.ResBlock2(x, embeddings)
        return self.conv(x), x

# %%
class UNET(nn.Module):
    def __init__(self,
            Channels: List = [64, 128, 256, 512, 512, 384],
            Attentions: List = [False, True, False, False, False, True],
            Upscales: List = [False, False, False, True, True, True],
            num_groups: int = 32,
            dropout_prob: float = 0.1,
            num_heads: int = 8,
            input_channels: int = 1,
            output_channels: int = 1,
            time_steps: int = 1000,
            num_classes: int = 10):
        super().__init__()
        self.num_layers = len(Channels)
        self.shallow_conv = nn.Conv2d(input_channels, Channels[0], kernel_size=3, padding=1)
        out_channels = (Channels[-1]//2)+Channels[0]
        self.late_conv = nn.Conv2d(out_channels, out_channels//2, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(out_channels//2, output_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.embeddings = SinusoidalEmbeddings(time_steps=time_steps, embed_dim=max(Channels))
        # Class conditioning: num_classes + 1 null class for unconditional
        self.class_embedding = nn.Embedding(num_classes + 1, max(Channels))
        for i in range(self.num_layers):
            layer = UnetLayer(
                upscale=Upscales[i],
                attention=Attentions[i],
                num_groups=num_groups,
                dropout_prob=dropout_prob,
                C=Channels[i],
                num_heads=num_heads
            )
            setattr(self, f'Layer{i+1}', layer)

    def forward(self, x, t, class_labels):
        x = self.shallow_conv(x)
        # Combined time + class embedding computed once
        time_emb = self.embeddings(x, t)
        class_emb = self.class_embedding(class_labels)[:, :, None, None]
        embeddings = time_emb + class_emb

        residuals = []
        for i in range(self.num_layers//2):
            layer = getattr(self, f'Layer{i+1}')
            x, r = layer(x, embeddings)
            residuals.append(r)
        for i in range(self.num_layers//2, self.num_layers):
            layer = getattr(self, f'Layer{i+1}')
            x = torch.concat((layer(x, embeddings)[0], residuals[self.num_layers-i-1]), dim=1)
        return self.output_conv(self.relu(self.late_conv(x)))

# %%
class DDPM_Scheduler(nn.Module):
    def __init__(self, num_time_steps: int=1000):
        super().__init__()
        self.beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad=False)
        alpha = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim=0).requires_grad_(False)

    def forward(self, t):
        return self.beta[t], self.alpha[t]

# %%
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# %%
NUM_CLASSES = 10
NULL_CLASS = NUM_CLASSES  # index 10 = unconditional

def inference(checkpoint_path: str=None,
              num_time_steps: int=1000,
              ema_decay: float=0.9999,
              guidance_scale: float=3.0):
    device = torch.device('cuda:1')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = UNET(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(checkpoint['weights'])
    ema = ModelEmaV3(model, decay=ema_decay)
    # Strip extra 'module.' prefix caused by saving EMA of a DataParallel-wrapped model
    ema_state = checkpoint['ema']
    ema_state = {k.replace('module.module.', 'module.'): v for k, v in ema_state.items()}
    ema.load_state_dict(ema_state)
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    beta = scheduler.beta.to(device)
    alpha = scheduler.alpha.to(device)

    # Generate one image per class (digits 0-9)
    classes = torch.arange(NUM_CLASSES, device=device)
    null_classes = torch.full((NUM_CLASSES,), NULL_CLASS, device=device, dtype=torch.long)

    with torch.no_grad():
        model = ema.module.eval()
        z = torch.randn(NUM_CLASSES, 1, 32, 32, device=device)

        for t in reversed(range(1, num_time_steps)):
            t_tensor = torch.full((NUM_CLASSES,), t, device=device)

            # Classifier-free guidance: batch conditional + unconditional together
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

        # Final step (t=0, no noise added)
        t_tensor = torch.zeros(NUM_CLASSES, dtype=torch.long, device=device)
        z_double = torch.cat([z, z], dim=0)
        t_double = torch.cat([t_tensor, t_tensor], dim=0)
        c_double = torch.cat([classes, null_classes], dim=0)
        eps_combined = model(z_double, t_double, c_double)
        eps_cond, eps_uncond = eps_combined.chunk(2, dim=0)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        temp = beta[0] / (torch.sqrt(1 - alpha[0]) * torch.sqrt(1 - beta[0]))
        x = (1 / torch.sqrt(1 - beta[0])) * z - temp * eps

        # Display one image per class
        fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(20, 2))
        for i, ax in enumerate(axes.flat):
            img = x[i].cpu().squeeze(0).numpy()
            ax.imshow(img, cmap='gray')
            ax.set_title(str(i))
            ax.axis('off')
        plt.suptitle(f'CFG w={guidance_scale}')
        plt.show()

def train(batch_size: int=128,
          num_time_steps: int=1000,
          num_epochs: int=15,
          seed: int=-1,
          ema_decay: float=0.9999,
          lr=2e-5,
          checkpoint_path: str=None,
          p_uncond: float=0.1):
    set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)

    device = torch.device('cuda:1')
    gpu_ids = [1, 2, 3]

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    model = UNET(num_classes=NUM_CLASSES).to(device)
    model = nn.DataParallel(model, device_ids=gpu_ids, output_device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ema = ModelEmaV3(model, decay=ema_decay)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.module.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    criterion = nn.MSELoss(reduction='mean')

    for i in range(num_epochs):
        total_loss = 0
        for bidx, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {i+1}/{num_epochs}")):
            x = x.to(device)
            y = y.to(device)
            x = F.pad(x, (2,2,2,2))

            # Randomly drop class labels for classifier-free guidance
            drop_mask = torch.rand(batch_size, device=device) < p_uncond
            y = torch.where(drop_mask, torch.full_like(y, NULL_CLASS), y)

            t = torch.randint(0,num_time_steps,(batch_size,))
            e = torch.randn_like(x, requires_grad=False)
            a = scheduler.alpha[t].view(batch_size,1,1,1).to(device)
            x = (torch.sqrt(a)*x) + (torch.sqrt(1-a)*e)
            output = model(x, t, y)
            optimizer.zero_grad()
            loss = criterion(output, e)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            ema.update(model)

        checkpoint = {
        'weights': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ema': ema.state_dict()
        }

        torch.save(checkpoint, 'checkpoints/ddpm_class')

        # inference(checkpoint_path='checkpoints/ddpm_class')
        print(f'Epoch {i+1} | Loss {total_loss / (60000/batch_size):.5f}')

# %%
def main():
    train(batch_size = 256, lr=2e-5, num_epochs=75)
    inference('checkpoints/ddpm_class', guidance_scale=3.0)

if __name__ == '__main__':
    main()

# %%


# %%



