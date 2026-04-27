"""
Generate digits that look like both 2s and 6s by mixing latent representations.
Strategies: linear interpolation, centroid mixing, stochastic blend, dimension swap, slerp.
"""

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Subset

import os
import sys

sys.path.insert(0, "/u/zup7mn/Classes/NN/digit4")
from src.models.vae import VariationalAutoEncoder

# ── Config ──────────────────────────────────────────────────────────────────

CHECKPOINT = "/u/zup7mn/Classes/NN/digit4/src/creative_vae/checkpoints/vae_epoch90.pth"
OUT_DIR = "/u/zup7mn/Classes/NN/digit4/src/creative_vae/generated"
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")


# ── Load model ───────────────────────────────────────────────────────────────

vae = VariationalAutoEncoder(real_dim=784, h_dims=[512, 256], bn_dim=16).to(device)
vae.load_state_dict(torch.load(CHECKPOINT, map_location=device))
vae.eval()
print("Loaded checkpoint")


# ── Load MNIST 2s and 6s ─────────────────────────────────────────────────────

transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

idx_2 = (test_dataset.targets == 2).nonzero(as_tuple=True)[0]
idx_6 = (test_dataset.targets == 6).nonzero(as_tuple=True)[0]

loader_2 = DataLoader(Subset(test_dataset, idx_2), batch_size=256, shuffle=False)
loader_6 = DataLoader(Subset(test_dataset, idx_6), batch_size=256, shuffle=False)

with torch.no_grad():
    x2 = next(iter(loader_2))[0].to(device)   # (N, 1, 28, 28)
    x6 = next(iter(loader_6))[0].to(device)

    x2_flat = x2.view(x2.size(0), -1)
    x6_flat = x6.view(x6.size(0), -1)

    mu2, lv2 = vae.encode(x2_flat)   # (N, 16)
    mu6, lv6 = vae.encode(x6_flat)

    # Class centroids in latent space
    c2 = mu2.mean(dim=0)   # (16,)
    c6 = mu6.mean(dim=0)

    # Per-sample std across the class (captures intra-class spread)
    std2 = mu2.std(dim=0)
    std6 = mu6.std(dim=0)


def decode_z(z: torch.Tensor, fname: str, nrow: int = 10):
    with torch.no_grad():
        imgs = vae.decode(z).view(-1, 1, 28, 28)
    save_image(imgs.cpu(), os.path.join(OUT_DIR, fname), nrow=nrow, normalize=False)
    print(f"Saved {fname}  ({z.shape[0]} images)")


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1: Linear interpolation between class centroids
# z = (1-alpha)*c2 + alpha*c6  for alpha in [0, 1]
# ─────────────────────────────────────────────────────────────────────────────

alphas = torch.linspace(0, 1, 20, device=device)
z_interp = torch.stack([(1 - a) * c2 + a * c6 for a in alphas])
decode_z(z_interp, "1_centroid_interpolation.png", nrow=20)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2: Midpoint centroid + noise exploration
# z = (c2+c6)/2 + noise  — random walk around the midpoint
# ─────────────────────────────────────────────────────────────────────────────

midpoint = (c2 + c6) / 2
spread = ((std2 + std6) / 2).clamp(min=0.1)
noise_scales = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
rows = []
for scale in noise_scales:
    noise = torch.randn(10, 16, device=device) * spread.unsqueeze(0) * scale
    rows.append(midpoint.unsqueeze(0) + noise)
z_midpoint = torch.cat(rows)
decode_z(z_midpoint, "2_midpoint_noise.png", nrow=10)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3: Pairwise interpolation — pair each 2 with a 6 and interpolate
# picks 8 pairs, 5 alphas each
# ─────────────────────────────────────────────────────────────────────────────

n_pairs = 8
alphas_pair = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=device)
rows = []
for i in range(n_pairs):
    m2 = mu2[i]
    m6 = mu6[i]
    for a in alphas_pair:
        rows.append((1 - a) * m2 + a * m6)
z_pairs = torch.stack(rows)
decode_z(z_pairs, "3_pairwise_interpolation.png", nrow=5)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 4: Dimension-wise mixing — swap k latent dims from 6 into 2's centroid
# sweeps k = 0..16 in order of how much the centroids differ
# ─────────────────────────────────────────────────────────────────────────────

diff = (c6 - c2).abs()
sorted_dims = diff.argsort(descending=True)  # dims most different first

rows = []
z_base = c2.clone()
rows.append(z_base.clone())  # k=0
for k in range(1, 17):
    z_base = z_base.clone()
    z_base[sorted_dims[k - 1]] = c6[sorted_dims[k - 1]]
    rows.append(z_base.clone())
z_dimswap = torch.stack(rows)
decode_z(z_dimswap, "4_dimension_swap.png", nrow=17)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 5: Spherical linear interpolation (slerp) between centroids
# ─────────────────────────────────────────────────────────────────────────────

def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    v0_n = F.normalize(v0, dim=0)
    v1_n = F.normalize(v1, dim=0)
    dot = (v0_n * v1_n).sum().clamp(-1, 1)
    omega = dot.acos()
    if omega.abs() < 1e-6:
        return (1 - t) * v0 + t * v1
    return (torch.sin((1 - t) * omega) / omega.sin()) * v0 + \
           (torch.sin(t * omega) / omega.sin()) * v1

alphas_slerp = torch.linspace(0, 1, 20).tolist()
z_slerp = torch.stack([slerp(c2, c6, a) for a in alphas_slerp])
decode_z(z_slerp, "5_slerp_interpolation.png", nrow=20)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 6: Stochastic blend — for each sample draw alpha ~ Beta(2,2)
# concentrates mass near 0.5, so most blends are ~50/50
# ─────────────────────────────────────────────────────────────────────────────

torch.manual_seed(42)
n_stoch = 50
alpha_dist = torch.distributions.Beta(torch.tensor(2.0), torch.tensor(2.0))
alphas_s = alpha_dist.sample((n_stoch,)).to(device)  # (50,)
z_stoch = alphas_s.unsqueeze(1) * c6 + (1 - alphas_s).unsqueeze(1) * c2
# add small per-sample noise
z_stoch = z_stoch + 0.3 * torch.randn_like(z_stoch) * spread.unsqueeze(0)
decode_z(z_stoch, "6_stochastic_beta_blend.png", nrow=10)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 7: Optimised search — gradient ascent in latent space to maximise
# P(class=2) * P(class=6) from the digit classifier (if available), else skip
# ─────────────────────────────────────────────────────────────────────────────

try:
    from src.mnist_classifier.model import DigitClassifier  # adapt if needed
    raise ImportError("skip")
except ImportError:
    print("Skipping strategy 7 (no importable digit classifier at this path)")


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 8: Gradient ascent with the built-in digit classifier from train.py
# ─────────────────────────────────────────────────────────────────────────────

from src.creative_vae.train import build_digit_classifier

digit_clf = build_digit_classifier().to(device)
try:
    digit_clf.load_state_dict(
        torch.load(
            "/u/zup7mn/Classes/NN/digit4/state/mnist_models/digit_classifier/mnist_mixup_classifier.pth",
            map_location=device,
        )
    )
    digit_clf.eval()
    for p in digit_clf.parameters():
        p.requires_grad_(False)

    # Start from the midpoint, optimise to maximise logit[2] + logit[6]
    z_opt = midpoint.clone().unsqueeze(0).requires_grad_(True)
    opt = torch.optim.Adam([z_opt], lr=0.05)

    results = []
    for step in range(300):
        opt.zero_grad()
        img = vae.decode(z_opt).view(1, 1, 28, 28)
        logits = digit_clf(img)
        loss = -(logits[0, 2] + logits[0, 6])  # maximise both
        loss.backward()
        opt.step()
        if step % 50 == 0:
            results.append(z_opt.detach().clone())

    z_grad = torch.cat(results)
    decode_z(z_grad, "8_gradient_ascent.png", nrow=len(results))
    print("Strategy 8 done")
except Exception as e:
    print(f"Strategy 8 skipped: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 9: Multi-start gradient ascent — diverse initialisations
# start from 20 random points near the midpoint and optimise
# ─────────────────────────────────────────────────────────────────────────────

try:
    n_starts = 20
    torch.manual_seed(0)
    z_init = midpoint.unsqueeze(0) + 0.5 * torch.randn(n_starts, 16, device=device) * spread
    z_ms = z_init.clone().requires_grad_(True)
    opt_ms = torch.optim.Adam([z_ms], lr=0.05)

    for step in range(400):
        opt_ms.zero_grad()
        imgs = vae.decode(z_ms).view(n_starts, 1, 28, 28)
        logits = digit_clf(imgs)
        loss = -(logits[:, 2] + logits[:, 6]).mean()
        loss.backward()
        opt_ms.step()

    decode_z(z_ms.detach(), "9_multistart_gradient_ascent.png", nrow=n_starts)
    print("Strategy 9 done")
except Exception as e:
    print(f"Strategy 9 skipped: {e}")


print(f"\nAll outputs saved to {OUT_DIR}/")
