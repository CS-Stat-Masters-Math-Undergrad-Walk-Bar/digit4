"""Generate 10,000 latent interpolations between 2s and 6s using mnist_VAE.

For each sample:
  z2, z6 = VAE.Encoder(x_2).mean, VAE.Encoder(x_6).mean    # use mean, not sampled
  a ~ Uniform(-0.1, 0.1)
  z = ((1 + a) * z2 + (1 - a) * z6) / 2
  x = VAE.Decoder(z).reshape(1, 28, 28)
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision

ROOT = Path('/u/zup7mn/Classes/NN/digit4')
HERE = Path(__file__).resolve().parent
N_SAMPLES = 10000
BATCH_SIZE = 500
DATA_ROOT = str(ROOT / 'src/data')
VAE_PATH  = ROOT / 'state/full_VAE.pth'
OUT_PATH  = HERE / 'vae_interp_samples.pt'


# Architecture mirrored from /u/zup7mn/Classes/NN/digit4/mnist_VAE.ipynb
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.first_hidden = nn.Linear(input_dim, hidden_dim)
        self.second_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.var = nn.Linear(hidden_dim, latent_dim)
        self.activation = nn.LeakyReLU()
    def forward(self, x):
        y = self.activation(self.first_hidden(self.flatten(x)))
        y = self.activation(self.second_hidden(y))
        return self.mean(y), self.var(y)


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.first_hidden = nn.Linear(latent_dim, hidden_dim)
        self.second_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.activation(self.first_hidden(x))
        y = self.activation(self.second_hidden(y))
        return self.sigmoid(self.output(y))


class VAE(nn.Module):
    def __init__(self, Encoder, Decoder):
        super().__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder


def collect_class_images(dataset, target_class):
    idx = [i for i, y in enumerate(dataset.targets.tolist()) if y == target_class]
    imgs = dataset.data[idx].float() / 255.0    # (N, 28, 28)
    return imgs.unsqueeze(1)                    # (N, 1, 28, 28)


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    mnist = torchvision.datasets.MNIST(root=DATA_ROOT, train=True, download=True)
    twos  = collect_class_images(mnist, 2).to(device)
    sixes = collect_class_images(mnist, 6).to(device)
    print(f"MNIST 2s: {twos.shape[0]}  6s: {sixes.shape[0]}")

    vae = VAE(Encoder(28 * 28, 300, 10), Decoder(10, 300, 28 * 28)).to(device)
    vae.load_state_dict(torch.load(VAE_PATH, map_location=device))
    vae.eval()

    g = torch.Generator(device=device).manual_seed(2026)
    samples = torch.empty(N_SAMPLES, 1, 28, 28)

    with torch.no_grad():
        for start in range(0, N_SAMPLES, BATCH_SIZE):
            bs = min(BATCH_SIZE, N_SAMPLES - start)
            i2 = torch.randint(0, twos.shape[0],  (bs,), device=device, generator=g)
            i6 = torch.randint(0, sixes.shape[0], (bs,), device=device, generator=g)

            mu2, _ = vae.Encoder(twos[i2])
            mu6, _ = vae.Encoder(sixes[i6])

            a = (torch.rand(bs, 1, device=device, generator=g) * 0.2) - 0.1   # Uniform(-0.1, 0.1)
            z = ((1 + a) * mu2 + (1 - a) * mu6) / 2.0

            x_flat = vae.Decoder(z)                          # (bs, 784)
            x = x_flat.view(bs, 1, 28, 28).clamp(0, 1)
            samples[start:start + bs] = x.cpu()
            print(f"  batch {start // BATCH_SIZE + 1}/{(N_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE}", flush=True)

    torch.save(samples, OUT_PATH)
    print(f"saved {tuple(samples.shape)} -> {OUT_PATH}")


if __name__ == '__main__':
    main()
