# %% [markdown]
# # Method comparison — Deep Creativity (discriminator-based)
# Compares image-generation methods that produce 2-and-6-like digits. Each method contributes a tensor of 10,000 (1×28×28) images. We score with `DeepCreativity(use_discriminator=True)` and report mean/std per method, plus top-25 and random-25 visualizations.

# %%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from project_paths import ROOT, VALUE_GAN_DISC_PATH, BASE_VAE_PATH, VALUE_CNN_BEST_PATH, NOVELTY_CNN_MIXUP, VAE_INTERP_OUT, DIFF_OUT_DIR
from metrics import DeepCreativity

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# %%
# VAE architecture (mirrors mnist_VAE.ipynb)
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
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = mean + torch.exp(log_var / 2) * torch.randn_like(mean)
        return self.Decoder(z), mean, log_var

# %%
def build_is_digit():
    return nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=5, padding=1, stride=1),
        nn.Dropout(0.25), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
        nn.Dropout(0.25), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(0.25),
        nn.Flatten(),
        nn.Linear(128 * 6 * 6, 256),
        nn.Dropout(0.25), nn.ReLU(),
        nn.Linear(256, 1),
    )

def build_digit_classifier():
    return nn.Sequential(
        nn.Conv2d(1, 256, kernel_size=5, stride=1, padding=1),
        nn.Dropout(0.2), nn.ReLU(),
        nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
        nn.Dropout(0.2),
        nn.Flatten(),
        nn.Linear(128 * 676, 10),
    )

def build_discriminator():
    return nn.Sequential(
        nn.Conv2d(1, 64, (3, 3), (2, 2), padding=1),
        nn.BatchNorm2d(64), nn.LeakyReLU(0.02),
        nn.Conv2d(64, 64, (3, 3), (2, 2), padding=1),
        nn.BatchNorm2d(64), nn.LeakyReLU(0.02),
        nn.Flatten(),
        nn.Linear(3136, 1),
        nn.Sigmoid(),
    )

# %%
vae = VAE(Encoder(28 * 28, 300, 10), Decoder(10, 300, 28 * 28)).to(device)
vae.load_state_dict(torch.load(BASE_VAE_PATH, map_location=device))
vae.eval()

is_digit = build_is_digit().to(device)
is_digit.load_state_dict(torch.load(VALUE_CNN_BEST_PATH, map_location=device))
is_digit.eval()

digit_clf = build_digit_classifier().to(device)
digit_clf.load_state_dict(torch.load(NOVELTY_CNN_MIXUP, map_location=device))
digit_clf.eval()

disc = build_discriminator().to(device)
disc.load_state_dict(torch.load(
    VALUE_GAN_DISC_PATH, map_location=device))
disc.eval()

scorer = DeepCreativity(
    vae=vae,
    digit_classifier=digit_clf,
    is_digit_classifier=is_digit,
    discriminator=disc,
    use_discriminator=True,
).to(device)
print('models loaded; scorer uses GAN discriminator')

# %% [markdown]
# ## Methods to compare
# 
# Edit the dict below as new sample files arrive. Each path should point to a tensor of shape `(N, 1, 28, 28)` with values in `[0, 1]`.

# %%
samples_unfilt = {
    'vae_interp':       VAE_INTERP_OUT,
    'diffusion_avg':    DIFF_OUT_DIR / 'generated_compose_average.pt',
    'base_vae_latent_optim': ROOT / "output/VAE/base_VAE_latent_optim.pt",
    'creative_vae_latent_optim': ROOT / "output/VAE/creative_VAE_latent_optim.pt",
}

# Skip any method whose file is not yet on disk so the notebook stays runnable mid-pipeline
samples = {k: v for k, v in samples_unfilt.items() if v.exists()}
print('methods present:')
for k, v in samples.items():
    print(f'  {k:18s} -> {v}')

# %%
@torch.no_grad()
def score_all(images, batch_size=256):
    out = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size].to(device).clamp(0, 1)
        out.append(scorer.score(batch).cpu())
    return torch.cat(out)

def show_grid(images, title, scores=None, ncols=4):
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.7))
    axes = np.array(axes).reshape(nrows, ncols)
    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        if i < n:
            ax.imshow(images[i].reshape(28, 28).numpy(), cmap='gray', vmin=0, vmax=1)
            if scores is not None:
                ax.set_title(f'{scores[i]:.3f}', fontsize=7)
        ax.axis('off')
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.005)
    plt.tight_layout()
    return fig

n_show = 16
rng = np.random.default_rng(2026)

# %%
results = {}
for name, path in samples.items():
    imgs = torch.load(path, map_location='cpu')
    if imgs.dim() == 2:      # (N, 784) flat -> (N, 1, 28, 28)
        imgs = imgs.view(-1, 1, 28, 28)
    elif imgs.dim() == 3:  # (N, 28, 28) -> (N, 1, 28, 28)
        imgs = imgs.unsqueeze(1)
    imgs = imgs.clamp(0, 1)
    scores = score_all(imgs)

    top_idx = scores.argsort(descending=True)[:n_show]
    rand_idx = torch.from_numpy(rng.choice(len(imgs), size=n_show, replace=False))

    results[name] = {
        'imgs': imgs,
        'scores': scores,
        'top_idx': top_idx,
        'rand_idx': rand_idx,
    }
    print(f'{name:18s}  N={len(imgs):5d}  '
          f'mean={scores.mean().item():.4f}  std={scores.std().item():.4f}  '
          f'min={scores.min().item():.4f}  max={scores.max().item():.4f}')

# %% [markdown]
# ## Mean / std summary table

# %%
rows = []
for name, r in results.items():
    s = r['scores']
    rows.append([name, len(s), s.mean().item(), s.std().item(),
                 s.median().item(), s.min().item(), s.max().item()])

print(f"{'method':18s} {'N':>6s} {'mean':>9s} {'std':>9s} {'median':>9s} {'min':>9s} {'max':>9s}")
print('-' * 72)
for row in rows:
    print(f'{row[0]:18s} {row[1]:>6d} {row[2]:>9.4f} {row[3]:>9.4f} '
          f'{row[4]:>9.4f} {row[5]:>9.4f} {row[6]:>9.4f}')

# also save a compact summary tensor for slides
torch.save({name: results[name]['scores'] for name in results},
           ROOT / 'analysis/all_scores.pt')
print(f"\nsaved per-method score arrays -> {ROOT / 'analysis/all_scores.pt'}")

# %%
# Bar chart of mean DC with std error bars
names  = list(results.keys())
means  = [results[n]['scores'].mean().item() for n in names]
stds   = [results[n]['scores'].std().item()  for n in names]

fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(names)), 4))
ax.bar(names, means, yerr=stds, capsize=6, color='#4477aa', edgecolor='black')
ax.set_ylabel('Deep Creativity (discriminator)')
ax.set_title('Mean DC across generation methods (±1 std)')
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
fig.savefig(ROOT / 'analysis/dc_mean_std_bar.png', dpi=130, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Per-method visualizations
# 
# For each method we render the top 25 by DC and a random 25 from the population.

# %%
for i, (name, r) in enumerate(results.items()):
    top_imgs   = r['imgs'][r['top_idx']]
    top_scores = r['scores'][r['top_idx']]
    rand_imgs   = r['imgs'][r['rand_idx']]
    rand_scores = r['scores'][r['rand_idx']]

    fig_top = show_grid(top_imgs, f'{name} — top 16 by DC', top_scores, ncols=4)
    fig_top.savefig(ROOT / f'analysis/top16_{name}.png', dpi=130, bbox_inches='tight')
    plt.show(block=False)

    fig_rand = show_grid(rand_imgs, f'{name} — random 16', rand_scores, ncols=4)
    fig_rand.savefig(ROOT / f'analysis/rand16_{name}.png', dpi=130, bbox_inches='tight')
    plt.show(block = (i == len(results) - 1))


