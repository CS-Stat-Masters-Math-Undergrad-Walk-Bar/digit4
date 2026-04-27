"""Compare the binary-digit CNN and the GAN discriminator at detecting real images.

Four evaluation sets of 1000 images each:
  * pure Gaussian noise
  * GAN-generated samples
  * diffusion-generated samples
  * real MNIST digits

An image is "detected as real" when model probability > 0.5. Accuracy is the
fraction of correct predictions given the label (real MNIST = real, everything
else = fake).
"""

# %%
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from timm.utils import ModelEmaV3

HERE = Path(__file__).resolve().parent
# Find the index of the last /src/ chunk.
src_ind = len(HERE.parts) - (HERE.parts[::-1].index('src') + 1)
HERE_STATE = Path(*HERE.parts[:src_ind]).joinpath('state', *HERE.parts[src_ind + 1:])

SRC = HERE.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diffusion.mnist_cls_diff import UNET, DDPM_Scheduler, NUM_CLASSES, NULL_CLASS

# %%
CNN_CKPT = HERE_STATE / "../mnist_models/is_digit_binary_classifier/cnn/best_cnn.pth"
DISC_CKPT = HERE_STATE / "checkpoints/discriminators/discriminator_last.pth"
GEN_CKPT = HERE_STATE / "checkpoints/generators/generator_last.pth"
# This reference doesn't exist (and didn't before my rearrangement) so I'll just leave it dangling
# -John
DIFF_CKPT = HERE_STATE / "../diffusion/checkpoints/ddpm_class"

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
N_PER_SET = 1000

CNN_MEAN, CNN_STD = 0.1736, 0.3317

# %%
def build_cnn():
    model = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=5, padding=1, stride=1),
        nn.Dropout(0.25),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
        nn.Dropout(0.25),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(0.25),
        nn.Flatten(),
        nn.Linear(128 * 6 * 6, 256),
        nn.Dropout(0.25),
        nn.ReLU(),
        nn.Linear(256, 1),
    )
    model.load_state_dict(torch.load(CNN_CKPT, map_location=DEVICE))
    return model.to(DEVICE).eval()


def build_discriminator():
    model = nn.Sequential(
        nn.Conv2d(1, 64, (3, 3), (2, 2), padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.02),
        nn.Conv2d(64, 64, (3, 3), (2, 2), padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.02),
        nn.Flatten(),
        nn.Linear(3136, 1),
        nn.Sigmoid(),
    )
    model.load_state_dict(torch.load(DISC_CKPT, map_location=DEVICE))
    return model.to(DEVICE).eval()


def build_generator():
    model = nn.Sequential(
        nn.Linear(100, 64 * 7 * 7),
        nn.BatchNorm1d(64 * 7 * 7),
        nn.LeakyReLU(),
        nn.Unflatten(1, (64, 7, 7)),
        nn.ConvTranspose2d(64, 64, (4, 4), (2, 2), padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(64, 1, (4, 4), (2, 2), padding=1),
        nn.Sigmoid(),
    )
    model.load_state_dict(torch.load(GEN_CKPT, map_location=DEVICE))
    return model.to(DEVICE).eval()


# %%
# Image sources. All return tensors in [0, 1] shaped (N, 1, 28, 28).

@torch.no_grad()
def sample_noise(n=N_PER_SET):
    return torch.randn(n, 1, 28, 28).clamp(0, 1)


@torch.no_grad()
def sample_gan(generator, n=N_PER_SET, batch=200):
    out = []
    for i in range(0, n, batch):
        k = min(batch, n - i)
        z = torch.rand(k, 100, device=DEVICE)  # matches gan.py training-time latent
        out.append(generator(z).cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def sample_real(n=N_PER_SET):
    ds = torchvision.datasets.MNIST(
        root=str(SRC / "diffusion/data"),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=n, shuffle=True)
    images, _ = next(iter(loader))
    return images[:n]


@torch.no_grad()
def sample_diffusion(n=N_PER_SET, batch=50, num_time_steps=1000,
                     guidance_scale=3.0, ema_decay=0.9999):
    """Classifier-free-guided DDPM sampling. Returns (n,1,28,28) in ~[0,1]."""
    checkpoint = torch.load(DIFF_CKPT, map_location=DEVICE)
    model = UNET(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(checkpoint["weights"])
    ema = ModelEmaV3(model, decay=ema_decay)
    ema_state = {k.replace("module.module.", "module."): v for k, v in checkpoint["ema"].items()}
    ema.load_state_dict(ema_state)
    model = ema.module.eval()

    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    beta = scheduler.beta.to(DEVICE)
    alpha = scheduler.alpha.to(DEVICE)

    out = []
    remaining = n
    while remaining > 0:
        k = min(batch, remaining)
        # Uniform across 0..9 classes
        classes = torch.arange(k, device=DEVICE) % NUM_CLASSES
        null_classes = torch.full((k,), NULL_CLASS, device=DEVICE, dtype=torch.long)
        z = torch.randn(k, 1, 32, 32, device=DEVICE)

        for t in reversed(range(1, num_time_steps)):
            t_tensor = torch.full((k,), t, device=DEVICE)
            z_double = torch.cat([z, z], dim=0)
            t_double = torch.cat([t_tensor, t_tensor], dim=0)
            c_double = torch.cat([classes, null_classes], dim=0)
            eps_combined = model(z_double, t_double, c_double)
            eps_cond, eps_uncond = eps_combined.chunk(2, dim=0)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            temp = beta[t] / (torch.sqrt(1 - alpha[t]) * torch.sqrt(1 - beta[t]))
            z = (1 / torch.sqrt(1 - beta[t])) * z - temp * eps
            z = z + torch.randn_like(z) * torch.sqrt(beta[t])

        t_tensor = torch.zeros(k, dtype=torch.long, device=DEVICE)
        z_double = torch.cat([z, z], dim=0)
        t_double = torch.cat([t_tensor, t_tensor], dim=0)
        c_double = torch.cat([classes, null_classes], dim=0)
        eps_combined = model(z_double, t_double, c_double)
        eps_cond, eps_uncond = eps_combined.chunk(2, dim=0)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        temp = beta[0] / (torch.sqrt(1 - alpha[0]) * torch.sqrt(1 - beta[0]))
        x = (1 / torch.sqrt(1 - beta[0])) * z - temp * eps

        # 32x32 -> 28x28 (undo the F.pad(2,2,2,2) used at train time) and clamp
        x = x[:, :, 2:30, 2:30].clamp(0, 1).cpu()
        out.append(x)
        remaining -= k

    del model, ema, checkpoint
    torch.cuda.empty_cache()
    return torch.cat(out, dim=0)


# %%
@torch.no_grad()
def predict_real_prob(model, images, normalize_cnn: bool, batch=256):
    """Run `model` over `images` and return prob-of-real in [0,1]."""
    probs = []
    for i in range(0, images.shape[0], batch):
        x = images[i:i + batch].to(DEVICE)
        if normalize_cnn:
            x = (x - CNN_MEAN) / CNN_STD
        logits_or_prob = model(x).squeeze(1)
        p = torch.sigmoid(logits_or_prob) if normalize_cnn else logits_or_prob
        probs.append(p.cpu())
    return torch.cat(probs, dim=0)


def accuracy(probs: torch.Tensor, label_is_real: bool, threshold=0.5) -> float:
    preds_real = probs > threshold
    correct = preds_real if label_is_real else ~preds_real
    return correct.float().mean().item()


# %%
def main():
    print(f"device: {DEVICE}")

    print("loading classifiers...")
    cnn = build_cnn()
    disc = build_discriminator()

    print("building evaluation sets...")
    gen = build_generator()
    sets = {
        "gaussian_noise": sample_noise(),
        "gan": sample_gan(gen),
        "real_mnist": sample_real(),
    }
    del gen
    torch.cuda.empty_cache()

    print("running diffusion sampling (this takes a while)...")
    sets["diffusion"] = sample_diffusion()

    print("scoring...")
    label_real = {"gaussian_noise": False, "gan": False, "diffusion": False, "real_mnist": True}

    header = f"{'set':16s}  {'label':6s}  {'CNN acc':>9s}  {'Disc acc':>9s}  {'CNN p(real)':>12s}  {'Disc p(real)':>13s}"
    print(header)
    print("-" * len(header))
    for name, imgs in sets.items():
        cnn_p = predict_real_prob(cnn, imgs, normalize_cnn=True)
        disc_p = predict_real_prob(disc, imgs, normalize_cnn=False)
        lbl = label_real[name]
        print(
            f"{name:16s}  {'real' if lbl else 'fake':6s}  "
            f"{accuracy(cnn_p, lbl):9.3f}  {accuracy(disc_p, lbl):9.3f}  "
            f"{cnn_p.mean().item():12.3f}  {disc_p.mean().item():13.3f}"
        )


if __name__ == "__main__":
    main()
